"""Exploratory first8 endpoint screen; no speed or production promotion verdict."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from torch.autograd.profiler_util import FunctionEventAvg
from transformers.integrations import sdpa_attention

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.torch.prefix.production import study as original
from wyat import Model
from wyat.backends.kai.cpu.prefix import PrefixScorer
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, QualityGate, compare_cycles
from wyat.benchmark import _run as _run
from wyat.evaluation import EvaluationCase, _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

from .scorer import SelectedScorer

ROOT = original.ROOT
CORE = "cpu-readout-core-review.json"
CORE_SHA = "53a776840ce6d406a710b5beca586a38d4ae17e1a2e55e3f0c35e605fd7e1c2e"
HISTORY = {
    "torch-prefix-production77-b8.json": (
        "c089b47b6ac844b09c479ea1b77fd3f8b1c938ce3476827303b94f8d956873c3"
    ),
    "torch-prefix-production77-b1.json": (
        "0638e0ea866ddfde87ad53aaafe03d74db955af0aadd11ceb69db1a183a19b9a"
    ),
    "torch-prefix-production77-b8-review.json": (
        "d517fde53de0e60898b31d613524f46effcfebe8f4de18a04035cc8950dd2650"
    ),
    "torch-prefix-production77-b1-review.json": (
        "1e4e1f8ac0062444cd5e661609c9e5ba47c4f4ed82cd72e61a310ce00911dc12"
    ),
    "torch-prefix-production77-b8.wall.json": (
        "75ebbb25bc8526cceae808e66e6f752cb9bbf08a11980b511ea8909cdfeea46b"
    ),
    "torch-prefix-production77-b1.wall.json": (
        "e410dc4089481d18b794f07f2eb43a60bda2d12fd630182969d795c984797dc3"
    ),
    "torch-prefix-production77-b8.cycles.jsonl": (
        "c53065b1b60a7d6cfba49d94d754c8af1c1cb308015d7aa6d534937bc4962d48"
    ),
    "torch-prefix-production77-b1.cycles.jsonl": (
        "6c2935151da8d6c2906eda01e9ff846b7cc8ad226ba23ccc72e8f6dedee9164f"
    ),
    "torch-prefix-production77-b8.sources-22aa511fe5434b21.json": (
        "22aa511fe5434b21aa6b6d45c0d6cbe42ef71473d53dc0ceddec202bd0e9e409"
    ),
}
SCHEDULE = (
    ("warmup", "original"),
    ("warmup", "retained"),
    ("warmup", "selected"),
    ("ordinary", "original"),
    ("ordinary", "retained"),
    ("ordinary", "selected"),
    ("ordinary", "selected"),
    ("ordinary", "retained"),
    ("ordinary", "original"),
    ("profile", "retained"),
    ("profile", "selected"),
)


def admit_history() -> tuple[int, ...]:
    """Pinned executed archives identify prior runs independently of live source."""
    prefix = original.admit_history()
    original.verify_files({original.RESULTS / CORE: CORE_SHA})
    core = read_object(original.RESULTS / CORE)
    if (
        core["passed"] is not True
        or core["scope_approved"] != "experimental_core_mechanics_only"
    ):
        raise ValueError("Independent mechanics review is unsupported")
    original.verify_files(
        {
            ROOT / name: sha
            for name, sha in cast(dict[str, str], core["source_sha256"]).items()
        }
    )
    original.verify_files(
        {
            ROOT / name: sha
            for name, sha in cast(
                dict[str, str], core["production_source_sha256"]
            ).items()
        }
    )
    original.verify_files(
        {original.RESULTS / name: sha for name, sha in HISTORY.items()}
    )
    archive = read_object(
        original.RESULTS / next(name for name in HISTORY if ".sources-" in name)
    )
    files = cast(list[dict[str, object]], archive["files"])
    table: dict[str, str] = {}
    for item in files:
        sha = hashlib.sha256(cast(str, item["text"]).encode()).hexdigest()
        if item["sha256"] != sha or item["name"] in table:
            raise ValueError("Historical archive identity differs")
        table[cast(str, item["name"])] = sha
    for batch in (8, 1):
        saved = read_object(
            original.RESULTS / f"torch-prefix-production77-b{batch}.json"
        )
        if (
            saved["completed"] is not True
            or saved["quality_passed"] is not True
            or saved["accepted"] is not True
            or saved["source_sha256"] != table
            or saved["model_id"] != original.MODEL_ID
            or saved["dataset_sha256"] != original.DATA_SHA
            or saved["reference_sha256"] != original.REFERENCE_SHA
            or saved["full_input_sha256"] != original.INPUT_SHA
            or tuple(cast(list[int], saved["prefix_tokens"])) != prefix
        ):
            raise ValueError("Final production prefix confirmation differs")
    return prefix


def source_paths() -> dict[str, Path]:
    paths = original.source_paths()
    for path in Path(__file__).parent.glob("*.py"):
        paths[str(path.relative_to(ROOT))] = path
    for name in (
        "docs/analysis/cpu/readout.md",
        "tests/analysis/cpu/readout/test_block.py",
        "tests/analysis/cpu/readout/test_screen.py",
    ):
        paths[name] = ROOT / name
    return paths


def profile_cycle(
    engine: BenchmarkEngine,
    cases: tuple[EvaluationCase, ...],
    retain: Callable[[BenchmarkCycle], None],
) -> list[dict[str, object]]:
    """Retain produced outputs before profiler exit; reduce only after it exits."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as profiler:
        retain(_run(engine, cases))
    averages = cast(
        Sequence[FunctionEventAvg], profiler.key_averages(group_by_input_shape=True)
    )
    return [
        {
            "operator": item.key,
            "count": item.count,
            "input_shapes": item.input_shapes,
            "self_cpu_time_us": float(item.self_cpu_time_total),
        }
        for item in averages
    ]


def dispatch(events: list[dict[str, object]], *, selected: bool) -> bool:
    """Observed exact native shapes/counts, rather than HF eligibility flags."""
    if any(
        item["operator"]
        in ("aten::_scaled_dot_product_attention_math", "aten::repeat_interleave")
        and cast(int, item["count"]) > 0
        for item in events
    ):
        return False
    flash = [
        item
        for item in events
        if item["operator"] == "aten::_scaled_dot_product_flash_attention_for_cpu"
    ]
    full = [
        [8, 16, 167, 128],
        [8, 8, 192, 128],
        [8, 8, 192, 128],
        [],
        [],
        [8, 1, 167, 192],
        [],
    ]
    short = [
        [8, 16, 7, 128],
        [8, 8, 192, 128],
        [8, 8, 192, 128],
        [],
        [],
        [8, 1, 7, 192],
        [],
    ]
    expected = [(full, 27), (short, 1)] if selected else [(full, 28)]
    return len(flash) == len(expected) and all(
        sum(cast(int, item["count"]) for item in flash if item["input_shapes"] == shape)
        == count
        for shape, count in expected
    )


def study(model: Path, output: Path) -> bool:
    if (
        output.exists()
        or output.with_suffix(".cycles.jsonl").exists()
        or any(output.parent.glob(output.stem + ".sources-*.json"))
    ):
        raise FileExistsError("Preserve existing evidence; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    gate = QualityGate()
    payload: dict[str, object] = {
        "scope": (
            "Exploratory first8 CPU selected final readouts; "
            "development exposure unknown"
        ),
        "completed": False,
        "quality_passed": False,
        "dispatch_passed": False,
        "accepted": False,
        "performance_passed": False,
        "threads_restored": False,
        "phase": "admission",
        "cycles": rows,
        "pending_cohorts": list(range(11)),
        "quality_gate": asdict(gate),
        "protocol": {
            "schedule": [list(item) for item in SCHEDULE],
            "outcomes": 88,
            "batch_size": 8,
            "dtype": "float32",
            "device": "cpu",
            "threads": 8,
            "max_length": 512,
            "token_cache_size": 0,
            "prefix_length": 25,
            "timing": (
                "Ordinary whole-cohort latency; profiler calls separate/intrusive"
            ),
            "promotion": (
                "None; full77 retained-relative B1/B8 confirmation required separately"
            ),
            "ownership": (
                "One shared base; two independent eager prefix "
                "snapshots; fresh per-call stock caches"
            ),
        },
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    with output.open("x") as initial:
        initial.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    previous = torch.get_num_threads()
    restored = False

    def restore() -> None:
        nonlocal restored
        restored = True
        try:
            torch.set_num_threads(previous)
            if torch.get_num_threads() != previous:
                raise ValueError("Original process thread count was not restored")
        except BaseException as error:
            payload["thread_restoration_error"] = f"{type(error).__name__}: {error}"
            raise
        payload["threads_restored"] = True

    try:
        prefix = admit_history()
        inputs = {
            original.DATA: original.DATA_SHA,
            original.REFERENCE: original.REFERENCE_SHA,
        }
        inputs.update({model / name: sha for name, sha in original.MODEL_FILES.items()})
        original.verify_files(inputs)
        sources = freeze_sources(source_paths(), output)
        policy = original.process_policy()
        flags = {
            name: vars(sdpa_attention)[name]
            for name in (
                "_is_torch_mps_available",
                "_is_torch_greater_or_equal_than_2_13",
            )
        }
        payload.update(
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            historical_sha256=HISTORY,
            original_historical_sha256=original.HISTORY,
            asset_sha256={str(path): sha for path, sha in inputs.items()},
            process_policy=policy,
            sdpa_flags=flags,
            runtime={
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch_git_version": torch.version.git_version,
                "torch_build": torch.__config__.show(),
                "prior_threads": previous,
                "interop_threads": torch.get_num_interop_threads(),
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "tokenizers",
                        "safetensors",
                        "numpy",
                    )
                },
            },
            source_scope=(
                "On-disk Python/dependency source and runtime metadata; "
                "native binaries/imported objects not independently attested"
            ),
            phase="loading",
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError(
                "Native masked-GQA context differs; do not patch eligibility flags"
            )
        torch.set_num_threads(8)
        all_cases = tuple(
            case_from_dict(decode_json(line))
            for line in original.DATA.read_text().splitlines()
        )
        if len(all_cases) != 77 or _input_digest(all_cases) != original.INPUT_SHA:
            raise ValueError("All original ordered77 requests required")
        base = KaiScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != original.MODEL_ID:
            raise ValueError("Released model identity differs")
        reference = read_reference(
            original.REFERENCE, all_cases, base, dataset_sha256=original.DATA_SHA
        )
        encoded = tuple(base.encode_request(case.request) for case in all_cases)
        resources: list[PrefixScorer] = []
        build: list[float] = []
        for kind in (PrefixScorer, SelectedScorer):
            started = time.perf_counter_ns()
            resources.append(kind(base, prefix))
            build.append((time.perf_counter_ns() - started) / 1e6)
        if not all(resource._admits(row) for resource in resources for row in encoded):
            raise ValueError(
                "All77 original encodings must admit the exact frozen prefix"
            )
        cases = all_cases[:8]
        reference = BenchmarkCycle(0, reference.rows[:8], 0, 0)
        if max(len(row.ids) for row in encoded[:8]) != 192 or any(
            len(row.candidate_positions) != 6 for row in encoded[:8]
        ):
            raise ValueError("Fixed first8 geometry differs")
        payload.update(
            model_id=base.model_id,
            dataset_sha256=original.DATA_SHA,
            reference_sha256=original.REFERENCE_SHA,
            input_sha256=_input_digest(cases),
            full_input_sha256=original.INPUT_SHA,
            case_ids=[case.id for case in cases],
            prefix_tokens=list(prefix),
            token_lengths=[len(row.ids) for row in encoded],
            prefix_state_bytes=[resource.prefix_state_bytes for resource in resources],
            prefix_build_ms=build,
            phase="cycles",
        )
        write()
        engines = {
            "original": BenchmarkEngine("original", Model(base, base.model_id), 8),
            "retained": BenchmarkEngine(
                "retained", Model(resources[0], base.model_id), 8
            ),
            "selected": BenchmarkEngine(
                "selected", Model(resources[1], base.model_id), 8
            ),
        }
        observed: list[BenchmarkCycle] = []
        anchors: dict[str, BenchmarkCycle] = {}
        profiled: dict[str, bool] = {}

        def verify() -> None:
            original.verify_sources(sources)
            original.verify_files({original.RESULTS / CORE: CORE_SHA})
            original.verify_files(
                {
                    original.RESULTS / name: sha
                    for name, sha in (original.HISTORY | HISTORY).items()
                }
            )
            if original.process_policy() != policy or torch.get_num_threads() != 8:
                raise ValueError("Owned process policy or threads changed")

        with output.with_suffix(".cycles.jsonl").open("x") as journal:
            for index, (stage, arm) in enumerate(SCHEDULE):
                verify()
                before = process_diagnostic()
                cycle: BenchmarkCycle | None = None
                events: list[dict[str, object]] = []

                def retain(value: BenchmarkCycle) -> None:
                    nonlocal cycle
                    cycle = value

                error: BaseException | None = None
                started = time.perf_counter_ns()
                try:
                    if stage == "profile":
                        events = profile_cycle(engines[arm], cases, retain)
                    else:
                        retain(_run(engines[arm], cases))
                except BaseException as failure:
                    error = failure
                outer_ms = (time.perf_counter_ns() - started) / 1e6
                if cycle is not None:
                    row: dict[str, object] = {
                        "index": index,
                        "stage": stage,
                        "arm": arm,
                        "cycle": original.json_cycle(cycle),
                        "outer_wall_ms": outer_ms,
                        "before": asdict(before),
                        "after": asdict(process_diagnostic()),
                        "operator_self_times": events,
                        "unknown_case_ids": [
                            case.id
                            for case in cases
                            if not any(
                                item.id == case.id
                                and item.status == "ok"
                                and item.prediction is not None
                                for item in cycle.rows
                            )
                        ],
                    }
                    for event in events:
                        value = cast(float, event["self_cpu_time_us"])
                        if not math.isfinite(value):
                            event.update(
                                self_cpu_time_us=None, invalid_timing=repr(value)
                            )
                    if error is not None:
                        row["capture_error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    payload["pending_cohorts"] = list(range(index + 1, 11))
                    # One immutable raw journal entry, before any comparison.
                    # Main rows subsequently add quality reports; raw fields agree.
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    write()
                    original.require_timing(cycle)
                    quality = compare_cycles((reference,), (cycle,), gate)
                    comparisons = [
                        compare_cycles((anchor,), (cycle,), gate)
                        for anchor in anchors.values()
                    ]
                    row["reference_quality"] = asdict(quality)
                    row["anchor_quality"] = [asdict(report) for report in comparisons]
                    write()
                    if (
                        not quality.passed
                        or any(not report.passed for report in comparisons)
                        or cycle.failed
                        or cycle.correct != reference.correct
                    ):
                        raise ValueError(
                            "Cohort numerical/rank/failure/accuracy gate failed"
                        )
                    anchors.setdefault(arm, cycle)
                    if any(event["self_cpu_time_us"] is None for event in events):
                        raise ValueError("Profile clock evidence is nonfinite")
                    observed.append(cycle)
                    if stage == "profile":
                        profiled[arm] = dispatch(events, selected=arm == "selected")
                    verify()
                if error is not None:
                    raise error
                if cycle is None:
                    raise ValueError("Cohort did not return raw outcomes")
        quality = compare_cycles((reference,) * 11, tuple(observed), gate)
        pairs = [
            compare_cycles(
                (observed[3], observed[8]), (observed[4], observed[7]), gate
            ),
            compare_cycles(
                (observed[3], observed[8]), (observed[5], observed[6]), gate
            ),
            compare_cycles(
                (observed[4], observed[7]), (observed[5], observed[6]), gate
            ),
        ]
        verify()
        original.verify_files(inputs)
        restore()
        passed = quality.passed and all(pair.passed for pair in pairs)
        dispatch_passed = profiled == {"retained": True, "selected": True}
        payload.update(
            completed=True,
            quality_passed=passed,
            dispatch_passed=dispatch_passed,
            phase="complete",
            reference_quality=asdict(quality),
            paired_quality=[asdict(pair) for pair in pairs],
            ordinary_times_ms={
                arm: [
                    cycle.latency_ms
                    for (_, name), cycle in zip(
                        SCHEDULE[3:9], observed[3:9], strict=True
                    )
                    if name == arm
                ]
                for arm in engines
            },
            source_verified_after=True,
            asset_verified_after=True,
            limits=(
                "Two ordinary observations per arm; no speed gate, "
                "full77/held-out/peak-memory/online claim. Profiles intrusive; "
                "grouped self times are not per-layer ordinary fractions."
            ),
        )
        write()
        return passed and dispatch_passed
    except BaseException as error:
        payload.update(
            completed=False,
            quality_passed=False,
            dispatch_passed=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        raise
    finally:
        if not restored:
            try:
                restore()
                write()
            except BaseException:
                payload.update(
                    completed=False,
                    quality_passed=False,
                    dispatch_passed=False,
                    phase="failed",
                )
                write()
                raise
