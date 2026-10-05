"""Real first-layer activations, strict projection gates, and balanced timings."""

from __future__ import annotations

import gc
import hashlib
import inspect
import itertools
import json
import os
import platform
import sys
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
from mlx.nn.layers import base as module_base
from mlx.nn.layers import linear, normalization
from mlx_lm.models import activations, cache, qwen3
from mlx_lm.models import base as maintained_base
from safetensors.numpy import save_file

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.mlx.bfloat.run import (
    DATASET_SHA256,
    MODEL_ID,
    REFERENCE_SHA256,
)
from benchmarks.analysis.mlx.endpoints import bounded_cache, memory
from benchmarks.analysis.mlx.mixed.capture import (
    CaptureFailure,
    capture_input,
    detached_float32,
    exact_bfloat_bits,
    projection_quality,
)
from benchmarks.analysis.reference import read_reference
from wyat.backends.kai.artifact import load_artifact
from wyat.backends.kai.encoding import KaiEncoder
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.backends.kai.prefix import PrefixScorer
from wyat.benchmark import BenchmarkCycle, QualityGate, compare_cycles
from wyat.evaluation import EvaluationCase, _input_digest, case_from_dict, percentile
from wyat.wire import decode_json

type CallableProjection = Callable[[], mx.array]

ROOT = Path(__file__).resolve().parents[4]
ARMS = ("maintained", "custom-fp32", "custom-bf16")
ORDERS = tuple(itertools.permutations(ARMS))
SHAPES = (
    ("prefix-b1", 1, 167),
    ("prefix-b8", 8, 1336),
    ("full-b1", 1, 192),
    ("full-b8", 8, 1536),
)
PRODUCTION = {
    "wyat/backends/kai/mlx.py": "2d56fa59c2fc0f04b6d9ba0fa210af17"
    "55ad8b265fead8b87a63505f140c8829",
    "wyat/backends/kai/prefix.py": "3fa532729757cb69edc0e3f682c30feb"
    "ff64717a6989149f0f3795dc7c170bed",
    "wyat/backends/kai/readout.py": "736190858293cc89a4e08ee7e7670349"
    "86ec09872218a501ed03ca73dd5b6800",
}


def save(output: Path, payload: dict[str, object]) -> None:
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def admit(
    model: Path,
) -> tuple[
    tuple[EvaluationCase, ...], BenchmarkCycle, tuple[int, ...], dict[str, object]
]:
    """Model bytes and every reference token are checked before model allocation."""
    for name, expected in PRODUCTION.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Declared production capture source changed")
    dataset = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = ROOT / "benchmarks/real/torch-fp32-reference77.json"
    dataset_bytes = dataset.read_bytes()
    if (
        hashlib.sha256(dataset_bytes).hexdigest() != DATASET_SHA256
        or hashlib.sha256(reference_path.read_bytes()).hexdigest() != REFERENCE_SHA256
    ):
        raise ValueError("Frozen dataset/reference bytes changed")
    cases = tuple(
        case_from_dict(decode_json(line))
        for line in dataset_bytes.decode().splitlines()
    )
    if len(cases) != 77 or len({case.id for case in cases}) != 77:
        raise ValueError("Frozen cohort requires all77 unique rows")
    artifact = load_artifact(model, max_length=512)
    if artifact.model_id != MODEL_ID:
        raise ValueError("Frozen model identity changed")
    encoder = KaiEncoder.from_artifact(artifact, max_length=512, token_cache_size=0)

    class ReferenceInput:
        model_id = artifact.model_id
        encode_request = encoder.encode_request

    reference = read_reference(
        reference_path, cases, ReferenceInput(), dataset_sha256=DATASET_SHA256
    )
    selected = cases[:8]
    tokens = tuple(encoder.encode_request(case.request) for case in selected)
    prefix = tokens[0].ids[:25]
    if (
        any(item.ids[:25] != prefix for item in tokens)
        or len(tokens[7].ids) != 192
        or max(len(item.ids) for item in tokens) != 192
    ):
        raise ValueError("Declared scalar/batch/prefix token geometry changed")
    evidence: dict[str, object] = {
        "model_id": artifact.model_id,
        "dataset_sha256": DATASET_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "input_sha256": _input_digest(selected),
        "prefix_tokens": list(prefix),
        "scalar_case_index": 7,
        "cases": [
            {
                "id": case.id,
                "token_ids": list(item.ids),
                "candidate_positions": list(item.candidate_positions),
                "query_position": item.query_position,
                "token_sha256": hashlib.sha256(
                    json.dumps(list(item.ids), separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for case, item in zip(selected, tokens, strict=True)
        ],
    }
    return selected, BenchmarkCycle(0, reference.rows[:8], 0, 0), prefix, evidence


def capture_all(
    model: Path,
    cases: tuple[EvaluationCase, ...],
    reference: BenchmarkCycle,
    prefix: tuple[int, ...],
    *,
    progress: Callable[[dict[str, object]], None],
) -> tuple[
    dict[str, np.ndarray[tuple[int, int], np.dtype[np.float32]]], dict[str, object]
]:
    started = time.perf_counter_ns()
    base = MLXKaiScorer.from_directory(
        model, max_length=512, token_cache_size=0, compile=False, device="gpu"
    )
    mx.synchronize(mx.gpu)
    load_ms = (time.perf_counter_ns() - started) / 1e6
    prefix_started = time.perf_counter_ns()
    cached = PrefixScorer(base, prefix)
    mx.synchronize(mx.gpu)
    prefix_ms = (time.perf_counter_ns() - prefix_started) / 1e6
    owners = tuple(
        weakref.ref(owner) for owner in (base, cached, base.backbone, base.head)
    )
    arrays: dict[str, np.ndarray[tuple[int, int], np.dtype[np.float32]]] = {}
    receipts: dict[str, object] = {
        "model_load_ms": load_ms,
        "prefix_build_ms": prefix_ms,
    }
    try:
        arrays["weight"] = detached_float32(
            base.backbone.layers[0].mlp.gate_proj.weight
        )
        if arrays["weight"].shape != (3072, 1024) or not exact_bfloat_bits(
            arrays["weight"]
        ):
            raise ValueError("Gate weight cannot decode from BF16 bitwise exactly")
        for name, batch, rows in SHAPES:
            cohort = cases if batch == 8 else (cases[7],)
            control = (
                reference
                if batch == 8
                else BenchmarkCycle(0, (reference.rows[7],), 0, 0)
            )
            try:
                captured = capture_input(
                    base,
                    cached if name.startswith("prefix") else base,
                    cohort,
                    batch_size=batch,
                    expected_rows=rows,
                )
            except CaptureFailure as error:
                receipts[name] = {"cycle": error.cycle.to_dict(), "error": str(error)}
                progress(receipts)
                raise
            gate = compare_cycles((control,), (captured.cycle,), QualityGate())
            arrays[name] = captured.values
            receipts[name] = {
                **captured.metadata(),
                "case_ids": [case.id for case in cohort],
                "cycle": captured.cycle.to_dict(),
                "cpu_quality": asdict(gate),
            }
            progress(receipts)
        del captured
    finally:
        del cached, base
        gc.collect()
        mx.synchronize(mx.gpu)
        mx.clear_cache()
    if any(owner() is not None for owner in owners):
        raise ValueError("Capture retained a model/resource owner")
    receipts.update(
        model_owners_released=True,
        capture_elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
        memory_after_release=memory(),
    )
    return arrays, receipts


def performance(blocks: list[dict[str, object]]) -> dict[str, object]:
    ratios: list[float] = []
    for block in blocks:
        totals = cast(dict[str, list[int]], block["dispatch_ns"])
        if set(totals) != set(ARMS) or any(
            len(values) != 10
            or any(type(value) is not int or value <= 0 for value in values)
            for values in totals.values()
        ):
            raise ValueError("Performance requires all30 positive timings per block")
        ratios.append(sum(totals["maintained"]) / sum(totals["custom-bf16"]))
    if (
        len(blocks) != 6
        or tuple(tuple(cast(list[str], block["order"])) for block in blocks) != ORDERS
    ):
        raise ValueError("Performance requires allsix balanced permutations")
    median, p05 = percentile(ratios, 0.5), percentile(ratios, 0.05)
    return {
        "paired_ratios": ratios,
        "paired_median": median,
        "paired_p05": p05,
        "passed": median >= 1.05 and p05 > 1,
    }


def output_digest(value: mx.array) -> str:
    return hashlib.sha256(detached_float32(value).tobytes()).hexdigest()


@contextmanager
def restored_cache(payload: dict[str, object]) -> Iterator[int]:
    failure: BaseException | None = None
    payload["cache_cap_restoration_completed"] = False
    try:
        with bounded_cache(512 << 20) as previous:
            payload["original_cache_cap_bytes"] = previous
            try:
                yield previous
            except BaseException as error:
                failure = error
        payload["cache_cap_restoration_completed"] = True
        payload["cache_cap_restored_bytes"] = previous
    except BaseException as error:
        payload["cache_cap_restoration_error"] = f"{type(error).__name__}: {error}"
        raise
    if failure is not None:
        raise failure


def study(model: Path, output: Path) -> bool:
    if output.exists():
        raise FileExistsError("Preserve prior projection receipts")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Disable TF32 before importing MLX")
    payload: dict[str, object] = {
        "completed": False,
        "accepted": False,
        "python": sys.version,
        "platform": platform.platform(),
        "tf32": os.environ.get("MLX_ENABLE_TF32"),
        "phase": "admission",
        "scope": (
            "first gate projection;8 unique training-source cases, exposure "
            "unknown; no complete inference speed claim"
        ),
        "protocol": {
            "shapes": SHAPES,
            "arms": ARMS,
            "orders": ORDERS,
            "dispatches_per_arm_block": 10,
            "partial_receipt": (
                "blocks saved after each complete arm; timeout within "
                "ten dispatches leaves that arm unknown"
            ),
            "cold_dispatches": 1,
            "warm_dispatches": 1,
            "cache_cap_bytes": 512 << 20,
            "timeout_seconds": 180,
            "atol": 1e-4,
            "rtol": 1e-4,
            "bitwise_storage_controls": True,
            "median_minimum": 1.05,
            "p05_exclusive_minimum": 1.0,
            "observation": (
                "every warmed/timed output hashed against its cold output "
                "outside dispatch timer; observations separate dispatches; "
                "not unobserved steady throughput"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, payload)
    try:
        from .kernel import ProjectionKernel

        files = {
            str(path.relative_to(ROOT)): path
            for folder in (ROOT / "benchmarks/analysis/mlx/mixed", ROOT / "wyat")
            for path in folder.rglob("*.py")
        }
        files.update(
            {
                name: ROOT / name
                for name in (
                    "docs/analysis/mixed.md",
                    "benchmarks/analysis/evidence.py",
                    "benchmarks/analysis/reference.py",
                    "benchmarks/analysis/torch/process.py",
                    "benchmarks/analysis/mlx/endpoints.py",
                    "benchmarks/analysis/mlx/bfloat/run.py",
                    "tests/analysis/test_mlx_mixed_capture.py",
                    "tests/analysis/test_mlx_mixed_kernel.py",
                )
            }
        )
        for name, source in (
            ("qwen3", qwen3),
            ("cache", cache),
            ("attention", maintained_base),
            ("module", module_base),
            ("linear", linear),
            ("normalization", normalization),
            ("activation", activations),
        ):
            files["maintained_" + name] = Path(inspect.getfile(source))
        mlx_root = Path(mx.__file__).resolve().parent
        for name in (
            "core/fast.pyi",
            "include/mlx/backend/metal/kernels/bf16.h",
            "include/mlx/backend/common/metal_kernel.h",
        ):
            files["maintained_mlx_" + name] = mlx_root / name
        frozen = freeze_sources(files, output)
        payload.update(source_hashes=frozen.hashes(), source_archive=frozen.archive())
        cases, reference, prefix, inputs = admit(model)
        payload.update(inputs)
        payload["phase"] = "capture"
        save(output, payload)
        with restored_cache(payload) as original_cap, mx.stream(mx.gpu):
            payload["original_cache_cap_bytes"] = original_cap
            payload["versions"] = {
                name: version(name) for name in ("mlx", "mlx-lm", "numpy")
            }
            payload["device"] = mx.device_info(mx.gpu)

            def capture_progress(receipts: dict[str, object]) -> None:
                payload["captures"] = receipts
                save(output, payload)

            arrays, captures = capture_all(
                model, cases, reference, prefix, progress=capture_progress
            )
            payload["captures"] = captures
            payload["weight"] = {
                "shape": list(arrays["weight"].shape),
                "sha256": hashlib.sha256(arrays["weight"].tobytes()).hexdigest(),
                "fp32_bytes": arrays["weight"].nbytes,
                "bfloat_bytes": arrays["weight"].size * 2,
                "exact_bfloat_decode": True,
            }
            tensor_path = output.with_suffix(".tensors.safetensors")
            if tensor_path.exists():
                raise FileExistsError("Preserve frozen tensor artifact")
            save_file(arrays, str(tensor_path))
            payload["captured_bytes"] = {
                "path": str(tensor_path),
                "sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
            }
            quality = all(
                cast(
                    dict[str, object],
                    cast(dict[str, object], captures[name])["cpu_quality"],
                )["passed"]
                is True
                for name, _, _ in SHAPES
            )
            payload["capture_quality_passed"] = quality
            save(output, payload)
            if not quality:
                payload.update(completed=True, phase="capture-rejected")
            else:
                if any(
                    hashlib.sha256(Path(source.path).read_bytes()).hexdigest()
                    != source.sha256
                    for source in frozen.files
                ):
                    raise ValueError("Frozen assay source changed after capture")
                payload["phase"] = "projection-screen"
                save(output, payload)
                started = time.perf_counter_ns()
                kernel = ProjectionKernel()
                payload["kernel_creation_ms"] = (time.perf_counter_ns() - started) / 1e6
                started = time.perf_counter_ns()
                w32 = mx.array(arrays["weight"], dtype=mx.float32)
                mx.eval(w32)
                mx.synchronize(mx.gpu)
                payload["fp32_weight_reconstruction_ms"] = (
                    time.perf_counter_ns() - started
                ) / 1e6
                started = time.perf_counter_ns()
                w16 = w32.astype(mx.bfloat16)
                mx.eval(w16)
                mx.synchronize(mx.gpu)
                payload["bfloat_weight_conversion_ms"] = (
                    time.perf_counter_ns() - started
                ) / 1e6
                decoded = detached_float32(w16.astype(mx.float32))
                if not np.array_equal(
                    decoded.view(np.uint32), arrays["weight"].view(np.uint32)
                ):
                    raise ValueError("Device BF16 decode changed weight bits")
                del decoded
                results: dict[str, dict[str, object]] = {}
                operations: dict[str, dict[str, CallableProjection]] = {}
                cold_values: dict[
                    str, np.ndarray[tuple[int, int], np.dtype[np.float32]]
                ] = {}
                cold_hashes: dict[str, dict[str, str]] = {}
                for name, _, _ in SHAPES:
                    started = time.perf_counter_ns()
                    x = mx.array(arrays[name], dtype=mx.float32)
                    mx.eval(x)
                    mx.synchronize(mx.gpu)
                    reconstruction_ms = (time.perf_counter_ns() - started) / 1e6
                    started = time.perf_counter_ns()
                    fp32, bf16 = kernel.prepare(x, w32), kernel.prepare(x, w16)
                    preparation_ms = (time.perf_counter_ns() - started) / 1e6

                    def maintained(x: mx.array = x) -> mx.array:
                        return x @ w32.T

                    calls: dict[str, CallableProjection] = {
                        "maintained": maintained,
                        "custom-fp32": fp32.run,
                        "custom-bf16": bf16.run,
                    }
                    operations[name] = calls
                    cold: dict[str, float] = {}
                    values: dict[
                        str, np.ndarray[tuple[int, int], np.dtype[np.float32]]
                    ] = {}
                    for arm in ARMS:
                        mx.synchronize(mx.gpu)
                        started = time.perf_counter_ns()
                        y = calls[arm]()
                        mx.eval(y)
                        mx.synchronize(mx.gpu)
                        cold[arm] = (time.perf_counter_ns() - started) / 1e6
                        values[arm] = detached_float32(y)
                        del y
                    cold_hashes[name] = {
                        arm: hashlib.sha256(value.tobytes()).hexdigest()
                        for arm, value in values.items()
                    }
                    cold_values.update(
                        {name + "-" + arm: value for arm, value in values.items()}
                    )
                    controls_equal = np.array_equal(
                        values["custom-fp32"].view(np.uint32),
                        values["custom-bf16"].view(np.uint32),
                    )
                    gates = {
                        arm: projection_quality(values["maintained"], values[arm])
                        for arm in ARMS[1:]
                    }
                    results[name] = {
                        "input_reconstruction_ms": reconstruction_ms,
                        "preparation_ms": preparation_ms,
                        "cold_dispatch_ms": cold,
                        "cold_output_sha256": cold_hashes[name],
                        "quality": gates,
                        "custom_storage_bitwise_equal": controls_equal,
                        "passed": controls_equal
                        and all(gate["passed"] is True for gate in gates.values()),
                        "blocks": [],
                    }
                    payload["shapes"] = results
                    save(output, payload)
                cold_path = output.with_suffix(".outputs.tensors.safetensors")
                if cold_path.exists():
                    raise FileExistsError("Preserve cold projection outputs")
                save_file(cold_values, str(cold_path))
                payload["cold_outputs"] = {
                    "path": str(cold_path),
                    "sha256": hashlib.sha256(cold_path.read_bytes()).hexdigest(),
                }
                numerical = all(result["passed"] is True for result in results.values())
                payload["projection_quality_passed"] = numerical
                if numerical:
                    payload["phase"] = "timing"
                    save(output, payload)
                    for name, _, _ in SHAPES:
                        calls = operations[name]
                        warm_hashes: dict[str, str] = {}
                        for arm in ARMS:
                            y = calls[arm]()
                            mx.eval(y)
                            mx.synchronize(mx.gpu)
                            warm_hashes[arm] = output_digest(y)
                            del y
                        results[name]["warm_output_sha256"] = warm_hashes
                        if warm_hashes != cold_hashes[name]:
                            raise ValueError(
                                "Warm projection drifted from screened output"
                            )
                        blocks: list[dict[str, object]] = []
                        for order in ORDERS:
                            before = memory()
                            timings: dict[str, list[int]] = {}
                            output_hashes: dict[str, list[str]] = {}
                            block: dict[str, object] = {
                                "order": list(order),
                                "dispatch_ns": timings,
                                "output_sha256": output_hashes,
                                "completed": False,
                                "outputs_reproduce_cold": False,
                                "memory_before": before,
                            }
                            blocks.append(block)
                            results[name]["blocks"] = blocks
                            for arm in order:
                                timings[arm] = []
                                output_hashes[arm] = []
                                for _ in range(10):
                                    mx.synchronize(mx.gpu)
                                    started = time.perf_counter_ns()
                                    y = calls[arm]()
                                    mx.eval(y)
                                    mx.synchronize(mx.gpu)
                                    timings[arm].append(
                                        time.perf_counter_ns() - started
                                    )
                                    output_hashes[arm].append(output_digest(y))
                                    del y
                                save(output, payload)
                            reproducible = all(
                                value == cold_hashes[name][arm]
                                for arm, hashes in output_hashes.items()
                                for value in hashes
                            )
                            block.update(
                                completed=True,
                                outputs_reproduce_cold=reproducible,
                                memory_after=memory(),
                            )
                            results[name]["blocks"] = blocks
                            save(output, payload)
                            if not reproducible:
                                raise ValueError(
                                    "Timed projection drifted from screened output"
                                )
                        results[name]["performance"] = performance(blocks)
                payload.update(
                    completed=True,
                    accepted=numerical
                    and all(
                        cast(
                            dict[str, object], results[name].get("performance", {})
                        ).get("passed")
                        is True
                        for name in ("prefix-b1", "prefix-b8")
                    ),
                    phase="complete",
                )
            payload.update(
                versions={name: version(name) for name in ("mlx", "mlx-lm", "numpy")},
                memory_before_cap_restoration=memory(),
            )
        payload["cache_cap_restored_bytes"] = original_cap
        payload["cache_cap_restoration_completed"] = True
    except BaseException as error:
        payload.update(
            completed=False, accepted=False, error=f"{type(error).__name__}: {error}"
        )
    finally:
        try:
            save(output, payload)
        except Exception as error:
            payload.update(
                completed=False,
                accepted=False,
                final_write_error=f"{type(error).__name__}: {error}",
            )
            save(output, payload)
    return payload["completed"] is True and payload["accepted"] is True
