"""Frozen native CPU/MLX parity and alternating timing experiment, research only.

Run --help. Each output is exclusively created; previous observations survive.
CPU oracle runs must put the release runtime first on PYTHONPATH and run outside
this checkout. No validation bypass or altered frozen artifact is supported.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint(path: Path, value: Any) -> None:
    """Atomically replace only this invocation's recoverable progress receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    temporary = path.with_name(path.name + ".partial.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(partial)


def finish(path: Path, value: Any) -> None:
    save(path, value)
    path.with_name(path.name + ".partial").unlink(missing_ok=True)


def pause(args: argparse.Namespace) -> None:
    # Outside every timed region; give foreground work opportunities to run.
    time.sleep(args.pause_ms / 1000)


def resource_policy(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "pause_ms_between_forwards": args.pause_ms,
        "cpu_intraop_threads": args.threads if args.arm == "cpu" else None,
        "cpu_interop_threads": 1 if args.arm == "cpu" else None,
        "mlx_memory_limit_bytes": args.memory_limit_mib * (1 << 20)
        if args.arm != "cpu"
        else None,
        "mlx_cache_limit_bytes": args.cache_limit_mib * (1 << 20)
        if args.arm != "cpu"
        else None,
        "memory_limit_interpretation": (
            "MLX process-local allocator guideline, not an operating-system hard cap"
        ),
        "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
    }


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def encoded_value(row: Any) -> dict[str, Any]:
    return {
        "question": {
            "id": row.question.id,
            "type": row.question.type,
            "instructions": row.question.instructions,
            "options": list(map(list, row.question.options)),
        },
        "ids": list(row.ids),
        "candidate_positions": list(row.candidate_positions),
        "query_position": row.query_position,
    }


def fixture() -> dict[str, Any]:
    return {
        "state": {"invoice": "overdue", "days": 12, "amount_eur": 1200},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "Choose invoice status.",
                "criteria": {"paid": "Paid in full.", "overdue": "Past due."},
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate urgency.",
                "criteria": ["Low urgency.", "Moderate urgency.", "High urgency."],
            },
            "large": {
                "type": "noul",
                "instructions": "The amount exceeds 1000 EUR.",
                "criteria": {"true": "True.", "false": "False."},
            },
        },
    }


def freeze(args: argparse.Namespace) -> None:
    from wyat import tasks
    from wyat.backends.kai.encoding import load_tokenizer

    tokenizer = load_tokenizer(args.artifact)
    manifest = read(args.artifact / "native_manifest.json")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    excluded = []
    for line in args.cohort.read_text().splitlines():
        source = json.loads(line)
        public = source[
            "public_record"
        ]  # Supervision and targets never enter inference.
        try:
            rows = tasks.encode_record(
                tokenizer, public, max_length=manifest["max_length"]
            )
        except ValueError as error:
            excluded.append({"id": source["id"], "error": str(error)})
            continue
        if len(rows) != 1:
            raise ValueError("DEV selection expects one question per source record")
        groups[(source["family"], rows[0].question.type)].append(
            {
                "id": source["id"],
                "family": source["family"],
                "public_record": public,
                "encoded": list(map(encoded_value, rows)),
            }
        )
    # Four predeclared within-stratum length quantiles; round robin all strata.
    ranked = {
        key: sorted(values, key=lambda v: (len(v["encoded"][0]["ids"]), v["id"]))
        for key, values in groups.items()
    }
    selected = []
    seen: set[str] = set()
    for quantile in (0.0, 1 / 3, 2 / 3, 1.0):
        for key in sorted(ranked):
            values = ranked[key]
            item = values[round(quantile * (len(values) - 1))]
            if item["id"] not in seen and len(selected) < args.records:
                selected.append(item)
                seen.add(item["id"])
    if len(selected) != args.records:
        raise ValueError(
            "not enough distinct stratified records for requested denominator"
        )
    # Candidate-width stress cases are declared before observing model outputs.
    for count in (77, 151, 255):
        public = {
            "state": "Select item zero.",
            "questions": {
                f"menu{count}": {
                    "type": "choice",
                    "instructions": "Select the item named in the state.",
                    "criteria": {f"k{i}": f"Item {i}." for i in range(count)},
                }
            },
        }
        rows = tasks.encode_record(tokenizer, public, max_length=manifest["max_length"])
        selected.append(
            {
                "id": f"synthetic-menu-{count}",
                "family": "synthetic-contract",
                "public_record": public,
                "encoded": list(map(encoded_value, rows)),
            }
        )
    reference_fixture = fixture()
    fixture_rows = tasks.encode_record(
        tokenizer, reference_fixture, max_length=manifest["max_length"]
    )
    records = selected + [
        {
            "id": "synthetic-three-independent-questions",
            "family": "synthetic-contract",
            "public_record": reference_fixture,
            "encoded": list(map(encoded_value, fixture_rows)),
        }
    ]
    cases = [
        {"id": f"single-{index:02d}", "records": [item["id"]]}
        for index, item in enumerate(selected)
    ]
    cases.append({"id": "three-independent-questions", "records": [records[-1]["id"]]})
    eligible = sorted(
        (
            item
            for item in selected
            if len(item["encoded"][0]["ids"]) * 8 <= manifest["max_batch_tokens"]
        ),
        key=lambda v: (len(v["encoded"][0]["ids"]), v["id"]),
    )
    if len(eligible) < 8:
        raise ValueError("fewer than eight rows fit the declared B8 padded budget")
    cases.extend(
        [
            {"id": "batch8-short", "records": [v["id"] for v in eligible[:8]]},
            {
                "id": "batch8-mixed",
                "records": [
                    eligible[round(i * (len(eligible) - 1) / 7)]["id"] for i in range(8)
                ],
            },
        ]
    )
    save(
        args.output,
        {
            "schema": "native-mlx-experiment-v1",
            "harness_sha256": digest(Path(__file__)),
            "artifact_manifest_sha256": digest(args.artifact / "native_manifest.json"),
            "model_id": manifest["model_id"],
            "cohort_sha256": digest(args.cohort),
            "tasks_sha256": digest(Path(tasks.__file__)),
            "records": records,
            "cases": cases,
            "excluded_before_selection": excluded,
            "quality_gates": {
                "logit_atol": 1e-3,
                "logit_rtol": 1e-3,
                "probability_atol": 1e-4,
                "noul_atol": 1e-4,
                "score_atol_per_interval": 1e-4,
                "exact_choice": True,
                "exact_full_rank": True,
            },
            "performance_gates": {
                "median_speedup_min": 1.05,
                "p05_speedup_min": 1.0,
                "minimum_pairs": 15,
            },
            "classification": (
                "reused DEV and synthetic implementation parity; "
                "no held-out quality claim"
            ),
        },
    )


def metadata() -> dict[str, Any]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "harness_sha256": digest(Path(__file__)),
        "packages": {},
    }
    for package in ("torch", "transformers", "mlx", "mlx-lm", "safetensors"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    result["sources"] = {}
    for name in (
        "wyat.tasks",
        "wyat.loading.native",
        "wyat.loading.native_artifact",
        "wyat.loading.mlx",
        "wyat.backends.kai.mlx",
        "wyat.backends.kai.metal",
        "mlx_lm.models.qwen3",
        "mlx_lm.models.base",
    ):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            result["sources"][name] = {"path": str(path), "sha256": digest(path)}
    result["MLX_ENABLE_TF32"] = os.environ.get("MLX_ENABLE_TF32", "unset")
    return result


def load_model(args: argparse.Namespace, metal: bool = False) -> Any:
    if args.arm == "cpu":
        import torch

        from wyat.loading import native

        runtime_root = args.runtime.resolve()
        if not Path(native.__file__).resolve().is_relative_to(runtime_root):
            raise RuntimeError("CPU oracle imported outside pinned release runtime")
        inventory = read(runtime_root / "SOURCE.json")["files_sha256"]
        for relative, expected in inventory.items():
            if digest(runtime_root / relative) != expected:
                raise RuntimeError("pinned runtime source hash differs: " + relative)
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        return native.load_native(args.artifact, device="cpu")
    import mlx.core as mx

    from wyat.loading.mlx import load_native_mlx

    mx.set_memory_limit(args.memory_limit_mib * (1 << 20))
    mx.set_cache_limit(args.cache_limit_mib * (1 << 20))
    if not mx.metal.is_available():
        raise RuntimeError("MLX GPU unavailable; experiment is unknown, not passed")
    return load_native_mlx(args.artifact, device="gpu", metal=metal)


def synchronize(arm: str) -> None:
    if arm != "cpu":
        import mlx.core as mx

        mx.synchronize()


def resident_bytes() -> int:
    if sys.platform == "darwin":

        class TimeValue(ctypes.Structure):
            _fields_ = [("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32)]

        class TaskInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time", TimeValue),
                ("system_time", TimeValue),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        task = ctypes.c_uint32.in_dll(system, "mach_task_self_").value
        info = TaskInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        system.task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        status = system.task_info(task, 20, ctypes.byref(info), ctypes.byref(count))
        if status != 0:
            raise RuntimeError(f"Mach task_info failed: {status}")
        return int(info.resident_size)
    if sys.platform.startswith("linux"):
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf(
            "SC_PAGE_SIZE"
        )
    raise RuntimeError("live RSS observation unsupported on this platform")


def memory(arm: str) -> dict[str, int]:
    result = {
        "rss_bytes": resident_bytes(),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if sys.platform != "darwin":
        result["peak_rss_bytes"] *= 1024
    if arm != "cpu":
        import mlx.core as mx

        result.update(
            active_mlx_bytes=mx.get_active_memory(),
            peak_mlx_bytes=mx.get_peak_memory(),
            cache_mlx_bytes=mx.get_cache_memory(),
        )
    return result


def prepare(model: Any, frozen: dict[str, Any]) -> dict[str, Any]:
    encoded = {}
    for item in frozen["records"]:
        rows = model.encode_record(item["public_record"])
        if list(map(encoded_value, rows)) != item["encoded"]:
            raise RuntimeError("encoded rows/endpoints differ: " + item["id"])
        encoded[item["id"]] = rows
    return {
        case["id"]: tuple(row for record in case["records"] for row in encoded[record])
        for case in frozen["cases"]
    }


def snapshot(rows: Any, logits: Any) -> list[dict[str, Any]]:
    from wyat import tasks

    if len(rows) != len(logits):
        raise RuntimeError("missing/extra output rows")
    result = []
    for row, values in zip(rows, logits, strict=True):
        answer = tasks.interpret(row.question, values)
        result.append(
            {
                "question_id": row.question.id,
                "type": row.question.type,
                "option_ids": list(answer.option_ids),
                "logits": list(answer.logits),
                "probabilities": list(answer.probabilities),
                "ranking": sorted(
                    range(len(values)), key=values.__getitem__, reverse=True
                ),
                "answer": answer.to_dict(),
            }
        )
    return result


def negative_checks(model: Any) -> list[dict[str, Any]]:
    records = [
        ("private_metadata", dict(fixture(), target="private")),
        ("invalid_task", {"state": "x", "questions": {"q": {"type": "invalid"}}}),
        (
            "one_option",
            {
                "state": "x",
                "questions": {"q": {"type": "choice", "criteria": {"a": "one"}}},
            },
        ),
        (
            "overlength",
            {
                "state": "token " * (model.max_length * 3),
                "questions": fixture()["questions"],
            },
        ),
    ]
    result = []
    for name, record in records:
        try:
            model.encode_record(record)
        except ValueError as error:
            result.append({"name": name, "pass": True, "error": str(error)})
        else:
            result.append({"name": name, "pass": False, "error": None})
    rows = model.encode_record(fixture())
    repetitions = model.max_batch_tokens // len(rows[0].ids) + 1
    try:
        model._logits((rows[0],) * repetitions)
    except ValueError as error:
        result.append(
            {"name": "padded_batch_budget", "pass": True, "error": str(error)}
        )
    else:
        result.append({"name": "padded_batch_budget", "pass": False, "error": None})
    return result


def check_frozen(frozen: dict[str, Any]) -> None:
    if frozen["harness_sha256"] != digest(Path(__file__)):
        raise RuntimeError("harness changed since cohort/gates were frozen")


def measure(args: argparse.Namespace) -> None:
    frozen = read(args.frozen)
    check_frozen(frozen)
    result: dict[str, Any] = {
        "arm": args.arm,
        "frozen_sha256": digest(args.frozen),
        "cases": [],
        "timings": [],
        "load_error": None,
        "cpu_threads": args.threads if args.arm == "cpu" else None,
        "resource_policy": resource_policy(args),
    }
    started = time.perf_counter()
    try:
        model = load_model(args, metal=args.arm == "metal")
        synchronize(args.arm)
        result.update(
            load_ms=(time.perf_counter() - started) * 1000,
            metadata=metadata(),
            memory_after_load=memory(args.arm),
            model_id=model.model_id,
            manifest_sha256=digest(args.artifact / "native_manifest.json"),
            metal_enabled=getattr(model, "metal_enabled", False),
        )
        batches = prepare(model, frozen)
        result["negative_checks"] = negative_checks(model)
    except Exception as error:
        result.update(load_error=repr(error), metadata=metadata())
        finish(args.output, result)
        return
    checkpoint(args.output, result)
    for name, rows in batches.items():
        case: dict[str, Any] = {
            "id": name,
            "rows": len(rows),
            "padded_width": max(len(row.ids) for row in rows),
            "outputs": None,
            "error": None,
        }
        try:
            synchronize(args.arm)
            started = time.perf_counter()
            values = model._logits(rows)
            synchronize(args.arm)
            case.update(
                elapsed_ms=(time.perf_counter() - started) * 1000,
                outputs=snapshot(rows, values),
            )
        except Exception as error:
            case["error"] = repr(error)
        finally:
            pause(args)
        result["cases"].append(case)
        checkpoint(args.output, result)
    fixture_case = next(
        case for case in result["cases"] if case["id"] == "three-independent-questions"
    )
    try:
        public = model.predict(fixture())
        result["predict_matches_logits_readout"] = {
            key: value.to_dict() for key, value in public.items()
        } == {row["question_id"]: row["answer"] for row in fixture_case["outputs"]}
        result["predict_error"] = None
    except Exception as error:
        result.update(predict_matches_logits_readout=False, predict_error=repr(error))
    finally:
        pause(args)
    checkpoint(args.output, result)
    records = {record["id"]: record for record in frozen["records"]}
    cases = {case["id"]: case for case in frozen["cases"]}
    for name in ("single-00", "batch8-short", "batch8-mixed"):
        rows = batches[name]
        for mode in ("encoded", "public"):
            observations = []
            timing = {
                "id": name,
                "mode": mode,
                "warmups": args.timing_warmups,
                "repeats_requested": args.timing_repeats,
                "observations": observations,
            }
            result["timings"].append(timing)
            for index in range(-args.timing_warmups, args.timing_repeats):
                observation: dict[str, Any] = {"error": None, "outputs": None}
                try:
                    synchronize(args.arm)
                    started = time.perf_counter()
                    if mode == "encoded":
                        values = model._logits(rows)
                    else:
                        # Full native preprocessing; identical independent row grouping.
                        if len(cases[name]["records"]) == 1:
                            values = model.logits(
                                records[cases[name]["records"][0]]["public_record"]
                            )
                        else:
                            current = tuple(
                                row
                                for record_id in cases[name]["records"]
                                for row in model.encode_record(
                                    records[record_id]["public_record"]
                                )
                            )
                            values = model._logits(current)
                    synchronize(args.arm)
                    observation.update(
                        elapsed_ms=(time.perf_counter() - started) * 1000,
                        outputs=snapshot(rows, values),
                    )
                except Exception as error:
                    observation["error"] = repr(error)
                finally:
                    pause(args)
                if index >= 0:
                    observations.append(observation)
                elif observation["error"] is not None:
                    result.setdefault("warmup_errors", []).append(
                        {"id": name, "mode": mode, "error": observation["error"]}
                    )
                checkpoint(args.output, result)
    result["memory_after_inference"] = memory(args.arm)
    finish(args.output, result)


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired(args: argparse.Namespace) -> None:
    frozen = read(args.frozen)
    check_frozen(frozen)
    try:
        ordinary = load_model(args, metal=False)
        optimized = load_model(args, metal=True)
    except Exception as error:
        finish(
            args.output,
            {
                "load_error": repr(error),
                "resource_policy": resource_policy(args),
                "metadata": metadata(),
                "frozen_sha256": digest(args.frozen),
                "cases": [],
                "metal_admission_pass": False,
            },
        )
        return
    models = {"ordinary": ordinary, "metal": optimized}
    batches = {arm: prepare(model, frozen) for arm, model in models.items()}
    names = ["single-00", "batch8-short", "batch8-mixed"]
    result = {
        "metadata": metadata(),
        "frozen_sha256": digest(args.frozen),
        "model_id": ordinary.model_id,
        "co_resident_models": True,
        "resource_policy": resource_policy(args),
        "metal_admission_pass": not ordinary.metal_enabled and optimized.metal_enabled,
        "metal_enabled": {arm: model.metal_enabled for arm, model in models.items()},
        "memory_interpretation": (
            "two models resident; use separate measure processes for startup/RSS"
        ),
        "warmups": args.warmups,
        "pairs_requested": args.pairs,
        "cases": [],
    }
    checkpoint(args.output, result)
    for name in names:
        pairs = []
        case = {
            "id": name,
            "pairs": pairs,
            "requested_pairs": args.pairs,
        }
        result["cases"].append(case)
        for _ in range(args.warmups):
            for arm, model in models.items():
                try:
                    model._logits(batches[arm][name])
                    synchronize(args.arm)
                except Exception as error:
                    result.setdefault("warmup_errors", []).append(
                        {"id": name, "arm": arm, "error": repr(error)}
                    )
                finally:
                    pause(args)
                checkpoint(args.output, result)
        for index in range(args.pairs):
            record: dict[str, Any] = {
                "pair": index,
                "order": ["ordinary", "metal"]
                if index % 2 == 0
                else ["metal", "ordinary"],
            }
            for arm in record["order"]:
                try:
                    synchronize(args.arm)
                    start = time.perf_counter()
                    values = models[arm]._logits(batches[arm][name])
                    synchronize(args.arm)
                    record[arm] = {
                        "elapsed_ms": (time.perf_counter() - start) * 1000,
                        "outputs": snapshot(batches[arm][name], values),
                        "error": None,
                    }
                except Exception as error:
                    record[arm] = {"error": repr(error), "outputs": None}
                finally:
                    pause(args)
            pairs.append(record)
            checkpoint(args.output, result)
        valid = [
            pair for pair in pairs if all(pair[arm]["error"] is None for arm in models)
        ]
        speeds = [
            pair["ordinary"]["elapsed_ms"] / pair["metal"]["elapsed_ms"]
            for pair in valid
        ]
        case.update(
            complete_pairs=len(valid),
            median_speedup=statistics.median(speeds) if speeds else None,
            p05_speedup=quantile(speeds, 0.05) if speeds else None,
        )
        checkpoint(args.output, result)
    result["memory_after_inference"] = memory(args.arm)
    finish(args.output, result)


def compare_rows(
    reference: Any, candidate: Any, gates: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    maximum = {
        "logit_absolute": 0.0,
        "probability_absolute": 0.0,
        "value_absolute": 0.0,
    }
    if reference is None or candidate is None or len(reference) != len(candidate):
        return {
            "pass": False,
            "failures": ["missing/extra output rows"],
            "max_error": maximum,
        }
    for index, (ref, cand) in enumerate(zip(reference, candidate, strict=True)):
        for key in ("question_id", "type", "option_ids", "ranking"):
            if ref[key] != cand[key]:
                failures.append(f"row {index}: {key} differs")
        for key, tolerance in [
            ("logits", gates["logit_atol"]),
            ("probabilities", gates["probability_atol"]),
        ]:
            if len(ref[key]) != len(cand[key]):
                failures.append(f"row {index}: {key} width differs")
                continue
            for original, observed in zip(ref[key], cand[key], strict=True):
                error = abs(original - observed)
                metric = "logit_absolute" if key == "logits" else "probability_absolute"
                maximum[metric] = max(maximum[metric], error)
                threshold = tolerance + (
                    gates["logit_rtol"] * abs(original) if key == "logits" else 0
                )
                if not math.isfinite(observed) or error > threshold:
                    failures.append(f"row {index}: {key} numeric gate failed")
                    break
        if ref["type"] == "choice":
            if ref["answer"]["choice"] != cand["answer"]["choice"]:
                failures.append(f"row {index}: selected choice differs")
        else:
            key = (
                "expected_ordinal_value"
                if ref["type"] == "score"
                else "probability_true"
            )
            error = abs(ref["answer"][key] - cand["answer"][key])
            maximum["value_absolute"] = max(maximum["value_absolute"], error)
            tolerance = (
                gates["score_atol_per_interval"] * max(1, len(ref["option_ids"]) - 1)
                if ref["type"] == "score"
                else gates["noul_atol"]
            )
            if error > tolerance:
                failures.append(f"row {index}: {key} gate failed")
    return {"pass": not failures, "failures": failures, "max_error": maximum}


def compare(args: argparse.Namespace) -> None:
    frozen, reference, candidate = map(
        read, (args.frozen, args.reference, args.candidate)
    )
    if reference.get("load_error") or candidate.get("load_error"):
        save(
            args.output,
            {
                "pass": False,
                "load_errors": {
                    "reference": reference.get("load_error"),
                    "candidate": candidate.get("load_error"),
                },
            },
        )
        return
    identity_ok = (
        reference["frozen_sha256"] == candidate["frozen_sha256"] == digest(args.frozen)
        and reference["model_id"] == candidate["model_id"] == frozen["model_id"]
    )
    cases = []
    candidate_cases = {case["id"]: case for case in candidate["cases"]}
    for ref in reference["cases"]:
        cand = candidate_cases.get(ref["id"], {"outputs": None})
        cases.append(
            {
                "id": ref["id"],
                **compare_rows(
                    ref["outputs"], cand["outputs"], frozen["quality_gates"]
                ),
            }
        )
    same_cases = (
        set(candidate_cases)
        == {case["id"] for case in reference["cases"]}
        == {case["id"] for case in frozen["cases"]}
    )
    negative_ok = all(
        len(value.get("negative_checks", [])) == 5
        and all(item["pass"] for item in value.get("negative_checks", []))
        for value in (reference, candidate)
    )
    repeated = []
    ref_cases = {case["id"]: case for case in reference["cases"]}
    for value in (reference, candidate):
        for timing in value.get("timings", []):
            for index, observation in enumerate(timing["observations"]):
                repeated.append(
                    {
                        "arm": value["arm"],
                        "id": timing["id"],
                        "mode": timing["mode"],
                        "repeat": index,
                        **compare_rows(
                            ref_cases[timing["id"]]["outputs"],
                            observation["outputs"],
                            frozen["quality_gates"],
                        ),
                    }
                )
    repeat_ok = (
        len(reference.get("timings", [])) == len(candidate.get("timings", [])) == 6
        and not reference.get("warmup_errors")
        and not candidate.get("warmup_errors")
        and all(case["pass"] for case in repeated)
        and all(
            len(timing["observations"]) == timing["repeats_requested"] > 0
            for value in (reference, candidate)
            for timing in value["timings"]
        )
    )
    save(
        args.output,
        {
            "reference_sha256": digest(args.reference),
            "candidate_sha256": digest(args.candidate),
            "frozen_sha256": digest(args.frozen),
            "identity_pass": identity_ok,
            "case_denominator_pass": same_cases,
            "negative_checks_pass": negative_ok,
            "repeated_quality_pass": repeat_ok,
            "repeated_quality": repeated,
            "predict_readout_pass": reference["predict_matches_logits_readout"]
            and candidate["predict_matches_logits_readout"],
            "pass": identity_ok
            and same_cases
            and negative_ok
            and repeat_ok
            and reference["predict_matches_logits_readout"]
            and candidate["predict_matches_logits_readout"]
            and all(case["pass"] for case in cases),
            "cases": cases,
        },
    )


def summarize_pairs(args: argparse.Namespace) -> None:
    frozen, reference, candidate = map(
        read, (args.frozen, args.reference, args.candidate)
    )
    if reference.get("load_error") or candidate.get("load_error"):
        save(
            args.output,
            {
                "pass": False,
                "load_errors": {
                    "reference": reference.get("load_error"),
                    "candidate": candidate.get("load_error"),
                },
            },
        )
        return
    gates = frozen["quality_gates"]
    refs = {case["id"]: case for case in reference["cases"]}
    cases = []
    for case in candidate["cases"]:
        checks = []
        for pair in case["pairs"]:
            checks.append(
                {
                    "pair": pair["pair"],
                    "cpu_to_ordinary": compare_rows(
                        refs[case["id"]]["outputs"], pair["ordinary"]["outputs"], gates
                    ),
                    "cpu_to_metal": compare_rows(
                        refs[case["id"]]["outputs"], pair["metal"]["outputs"], gates
                    ),
                    "ordinary_to_metal": compare_rows(
                        pair["ordinary"]["outputs"], pair["metal"]["outputs"], gates
                    ),
                }
            )
        quality = all(
            all(
                item[key]["pass"]
                for key in ("cpu_to_ordinary", "cpu_to_metal", "ordinary_to_metal")
            )
            for item in checks
        )
        performance = (
            case["complete_pairs"] == case["requested_pairs"] == len(case["pairs"])
            and case["requested_pairs"] >= frozen["performance_gates"]["minimum_pairs"]
            and case["median_speedup"] is not None
            and case["median_speedup"]
            >= frozen["performance_gates"]["median_speedup_min"]
            and case["p05_speedup"] >= frozen["performance_gates"]["p05_speedup_min"]
        )
        cases.append(
            {
                "id": case["id"],
                "quality_pass": quality,
                "performance_pass": performance,
                "median_speedup": case["median_speedup"],
                "p05_speedup": case["p05_speedup"],
                "checks": checks,
            }
        )
    identity = (
        reference["frozen_sha256"] == candidate["frozen_sha256"] == digest(args.frozen)
        and reference["model_id"] == candidate["model_id"] == frozen["model_id"]
    )
    passed = (
        identity
        and candidate["metal_admission_pass"]
        and not candidate.get("warmup_errors")
        and {case["id"] for case in cases}
        == {"single-00", "batch8-short", "batch8-mixed"}
        and all(case["quality_pass"] and case["performance_pass"] for case in cases)
    )
    save(
        args.output,
        {
            "pass": passed,
            "identity_pass": identity,
            "metal_admission_pass": candidate["metal_admission_pass"],
            "reference_sha256": digest(args.reference),
            "candidate_sha256": digest(args.candidate),
            "cases": cases,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--artifact", type=Path, required=True)
    freezing.add_argument("--cohort", type=Path, required=True)
    freezing.add_argument("--records", type=int, default=32)
    freezing.add_argument("--output", type=Path, required=True)
    for command in ("measure", "paired"):
        child = sub.add_parser(command)
        child.add_argument("--artifact", type=Path, required=True)
        child.add_argument("--frozen", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--arm", choices=("cpu", "ordinary", "metal"), required=True)
        child.add_argument("--runtime", type=Path)
        child.add_argument("--threads", type=int, default=4)
        child.add_argument("--pause-ms", type=float, default=250)
        child.add_argument("--memory-limit-mib", type=int, default=6144)
        child.add_argument("--cache-limit-mib", type=int, default=256)
        child.add_argument("--timing-repeats", type=int, default=3)
        child.add_argument("--timing-warmups", type=int, default=1)
        if command == "paired":
            child.add_argument("--pairs", type=int, default=15)
            child.add_argument("--warmups", type=int, default=3)
    for command in ("compare", "summarize-pairs"):
        comparing = sub.add_parser(command)
        for name in ("frozen", "reference", "candidate", "output"):
            comparing.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    for suffix in ("", ".partial", ".partial.tmp"):
        if args.output.with_name(args.output.name + suffix).exists():
            parser.error("output or progress receipt already exists; use a fresh name")
    if args.command in ("measure", "paired"):
        if args.pause_ms < 0 or not math.isfinite(args.pause_ms):
            parser.error("--pause-ms must be finite and nonnegative")
        if args.threads < 1 or args.memory_limit_mib < 1 or args.cache_limit_mib < 0:
            parser.error("threads/memory must be positive; cache must be nonnegative")
        if args.timing_repeats < 1 or args.timing_warmups < 0:
            parser.error("timing repeats must be positive and warmups nonnegative")
        if args.command == "paired" and (args.pairs < 1 or args.warmups < 0):
            parser.error("pairs must be positive and warmups nonnegative")
    if args.command == "measure" and args.arm == "cpu" and args.runtime is None:
        parser.error("CPU oracle requires --runtime and pinned PYTHONPATH")
    if args.command == "paired" and args.arm == "cpu":
        parser.error("paired compares the two MLX GPU arms only")
    {
        "freeze": freeze,
        "measure": measure,
        "paired": paired,
        "compare": compare,
        "summarize-pairs": summarize_pairs,
    }[args.command](args)


if __name__ == "__main__":
    main()
