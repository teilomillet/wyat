"""Research-only exact-product FP32 rotary fusion with independent bit oracle.

The existing native RoPE owns all frequencies and tables. Only one loaded GPU
scorer's table application changes. Other RoPE objects, including CPU scorers,
retain their original method. No production file or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
from mlx_lm.models import activations, base, qwen3

from benchmarks.analysis.native_mlx_benchmark_20261012 import (
    check_frozen,
    checkpoint,
    compare_rows,
    digest,
    finish,
    memory,
    metadata,
    prepare,
    quantile,
    read,
    snapshot,
)
from wyat.loading.mlx import _RoPE, load_native_mlx

SOURCE = """
#pragma clang fp contract(off)
const uint i = thread_position_in_grid.x;
if (i >= Size) return;
const uint lane = i % Dims;
const uint partner = lane < Dims / 2 ? i + Dims / 2 : i - Dims / 2;
const float rotated = lane < Dims / 2 ? -X[partner] : X[partner];
const float original_product = X[i] * C[i % TableSize];
const float rotated_product = rotated * S[i % TableSize];
Y[i] = original_product + rotated_product;
"""


def operation() -> Callable[[mx.array, mx.array, mx.array], mx.array]:
    kernel = cast(
        Callable[..., list[mx.array]],
        mx.fast.metal_kernel(
            name="wyat_native_rope_exact_products_research",
            input_names=["X", "C", "S"],
            output_names=["Y"],
            source=SOURCE,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )

    def run(value: mx.array, cosine: mx.array, sine: mx.array) -> mx.array:
        if (
            value.ndim < 2
            or value.dtype != mx.float32
            or value.shape[-1] < 2
            or value.shape[-1] % 2
            or cosine.shape != value.shape[-2:]
            or sine.shape != cosine.shape
            or cosine.dtype != mx.float32
            or sine.dtype != mx.float32
        ):
            raise ValueError("requires full even FP32 rotary width and matching tables")
        return kernel(
            inputs=[value, cosine, sine],
            template=[
                ("Dims", value.shape[-1]),
                ("TableSize", cosine.size),
                ("Size", value.size),
            ],
            grid=(value.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[value.shape],
            output_dtypes=[mx.float32],
        )[0]

    return run


def primitive_cases() -> list[
    tuple[str, np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
]:
    rng = np.random.default_rng(20261012)
    shape = (2, 3, 17, 128)
    table_shape = shape[-2:]
    value = rng.standard_normal(shape).astype(np.float32)
    cosine = rng.uniform(-1, 1, table_shape).astype(np.float32)
    sine = rng.uniform(-1, 1, table_shape).astype(np.float32)
    cases = [("seeded-normal", value, cosine, sine)]
    exponents = rng.integers(-100, 121, shape)
    wide = np.ldexp(value / np.float32(8), exponents).astype(np.float32)
    cases.append(("seeded-wide-exponents", wide, cosine, sine))
    signs = np.resize(np.array([0, 0x80000000], dtype=np.uint32), value.size)
    zeros = signs.view(np.float32).reshape(shape)
    cases.append(("signed-zeros", zeros, cosine, sine))
    cancellation = value.copy()
    cancellation[..., 64:] = cancellation[..., :64]
    half = np.full(table_shape, 0.5, np.float32)
    cases.append(("exact-cancellation", cancellation, half, half))
    trap = np.ones(shape, np.float32)
    trap[..., :64] = np.float32(1 + 2**-23)
    trap[..., 64:] = np.float32(1 + 2**-22)
    trap_cosine = np.ones(table_shape, np.float32)
    trap_cosine[..., :64] = np.float32(1 + 2**-23)
    cases.append(("fma-rounding-trap", trap, trap_cosine, np.ones_like(half)))
    for exponent in (-100, 120):
        scaled = np.ldexp(np.sign(value), exponent).astype(np.float32)
        cases.append((f"finite-exponent-{exponent}", scaled, cosine, sine))
    # A different odd token width, head count and non-contiguous logical input
    # exercises table repetition, tail threads and required row-contiguous copy.
    odd = rng.standard_normal((1, 33, 3, 128)).astype(np.float32).transpose(0, 2, 1, 3)
    odd_cos = rng.uniform(-1, 1, (33, 128)).astype(np.float32)
    odd_sin = rng.uniform(-1, 1, (33, 128)).astype(np.float32)
    cases.append(("odd-width-strided", odd, odd_cos, odd_sin))
    return cases


def primitive(
    fused: Callable[[mx.array, mx.array, mx.array], mx.array],
    archive: Path,
    pause_ms: float,
) -> dict[str, Any]:
    records, arrays = [], {}
    for index, (name, value, cosine, sine) in enumerate(primitive_cases()):
        half = value.shape[-1] // 2
        rotated = np.concatenate([-value[..., half:], value[..., :half]], axis=-1)
        left = np.multiply(value, cosine, dtype=np.float32)
        right = np.multiply(rotated, sine, dtype=np.float32)
        oracle = np.add(left, right, dtype=np.float32)
        x, c, s = mx.array(value), mx.array(cosine), mx.array(sine)
        if name == "odd-width-strided":
            # mx.array copies a NumPy view; explicitly construct the equivalent
            # MLX transpose so the custom-kernel contiguity path is exercised.
            x = mx.array(value.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        rotated_mlx = mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        product_a, product_b = x * c, rotated_mlx * s
        mx.eval(product_a, product_b)
        baseline, candidate = product_a + product_b, fused(x, c, s)
        mx.eval(baseline, candidate)
        actual_base, actual_candidate = np.array(baseline), np.array(candidate)
        expected_bits = oracle.view(np.uint32)
        base_mismatch = np.count_nonzero(actual_base.view(np.uint32) != expected_bits)
        candidate_mismatch = np.count_nonzero(
            actual_candidate.view(np.uint32) != expected_bits
        )
        cross_mismatch = np.count_nonzero(
            actual_base.view(np.uint32) != actual_candidate.view(np.uint32)
        )
        records.append(
            {
                "id": name,
                "shape": value.shape,
                "elements": value.size,
                "baseline_vs_numpy_bit_mismatches": int(base_mismatch),
                "candidate_vs_numpy_bit_mismatches": int(candidate_mismatch),
                "candidate_vs_baseline_bit_mismatches": int(cross_mismatch),
                "passed": bool(
                    base_mismatch == candidate_mismatch == cross_mismatch == 0
                ),
            }
        )
        for label, data in [
            ("input", value),
            ("cosine", cosine),
            ("sine", sine),
            ("numpy", oracle),
            ("baseline", actual_base),
            ("candidate", actual_candidate),
        ]:
            arrays[f"case_{index}_{label}"] = data
        time.sleep(pause_ms / 1000)
    with archive.open("xb") as stream:
        np.savez(stream, **arrays)
    return {
        "passed": all(record["passed"] for record in records),
        "oracle": (
            "independent NumPy CPU float32 multiply, multiply, add; exact uint32 bits"
        ),
        "seed": 20261012,
        "numpy_version": np.__version__,
        "cases": records,
        "archive": str(archive),
        "archive_sha256": digest(archive),
    }


def forward(model: Any, rows: Any, pause_ms: float) -> dict[str, Any]:
    result: dict[str, Any] = {"error": None, "outputs": None}
    try:
        mx.synchronize(model.device)
        start = time.perf_counter_ns()
        logits = model._logits(rows)
        mx.synchronize(model.device)
        result.update(
            elapsed_ms=(time.perf_counter_ns() - start) / 1e6,
            outputs=snapshot(rows, logits),
            rotary_tables=table_state(model.backbone.layers[0].self_attn.rope),
        )
    except Exception as error:
        result["error"] = repr(error)
    finally:
        time.sleep(pause_ms / 1000)
    return result


def table_state(rope: _RoPE) -> dict[str, Any]:
    return {
        "entries": len(rope._tables),
        "keys_width_offset": [list(key) for key in rope._tables],
        "allocated_table_bytes": sum(
            cosine.nbytes + sine.nbytes for cosine, sine in rope._tables.values()
        ),
        "inverse_frequency_bytes": rope._inverse.nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "frozen", "reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--pause-ms", type=float, default=250)
    args = parser.parse_args()
    if (
        not 1 <= args.pairs <= 5
        or not math.isfinite(args.pause_ms)
        or args.pause_ms < 250
    ):
        parser.error("bounded experiment requires 1..5 pairs and at least 250 ms idle")
    archive = args.output.with_suffix(".primitive.npz")
    for path in [
        args.output,
        archive,
        args.output.with_name(args.output.name + ".partial"),
        args.output.with_name(args.output.name + ".partial.tmp"),
    ]:
        if path.exists():
            parser.error("output/progress/archive already exists; choose fresh output")
    frozen, reference = read(args.frozen), read(args.reference)
    check_frozen(frozen)
    if (
        reference["frozen_sha256"] != digest(args.frozen)
        or reference.get("arm") != "cpu"
    ):
        raise ValueError("requires CPU reference for the exact frozen inputs")
    refs = {case["id"]: case for case in reference["cases"]}
    if len(refs) != len(frozen["cases"]) or any(
        case.get("error") is not None for case in refs.values()
    ):
        raise ValueError("missing, duplicate or failed CPU reference cases")
    sources = [
        Path(__file__).resolve(),
        Path(inspect.getfile(prepare)),
        Path(inspect.getfile(_RoPE)),
        Path(inspect.getfile(activations)),
        Path(inspect.getfile(base)),
        Path(inspect.getfile(qwen3)),
        args.frozen.resolve(),
        args.reference.resolve(),
        (args.artifact / "native_manifest.json").resolve(),
    ]
    result: dict[str, Any] = {
        "schema": "native-mlx-rope-fusion-v1",
        "completed": False,
        "passed": False,
        "source_sha256": {str(path): digest(path) for path in sources},
        "protocol": {
            "one_model": True,
            "shared_tables": (
                "existing _RoPE owns frequencies, cosines, sines and eviction"
            ),
            "kernel_source": SOURCE,
            "kernel_math_mode": "safe",
            "fp_contract": "off",
            "tf32_environment": os.environ.get("MLX_ENABLE_TF32"),
            "cache_limit_bytes": 256 << 20,
            "mlx_memory_guideline_bytes": 6144 << 20,
            "pause_ms_between_forwards": args.pause_ms,
            "responsive": "unchanged native every-four-layer eval and 1 ms idle",
            "quality": "all frozen CPU cases and exact full rank before any timing",
            "quality_gates": frozen["quality_gates"],
            "pairs_per_timing_case": args.pairs,
            "qualification": "bounded exploratory screen only",
            "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
        },
        "quality_cases": [],
        "timing_cases": [],
    }
    checkpoint(args.output, result)
    previous_cache, previous_memory = (
        mx.set_cache_limit(256 << 20),
        mx.set_memory_limit(6144 << 20),
    )
    original = _RoPE.__call__
    try:
        fused = operation()
        result["primitive"] = primitive(fused, archive, args.pause_ms)
        checkpoint(args.output, result)
        if not result["primitive"]["passed"]:
            raise RuntimeError(
                "independent bitwise primitive oracle failed; model/timing skipped"
            )
        model = load_native_mlx(args.artifact, metal=True, responsive=True)
        if not model.metal_enabled:
            raise RuntimeError("qualified Metal base projections unavailable")
        if (
            model.model_id != reference["model_id"]
            or model.model_id != frozen["model_id"]
        ):
            raise ValueError("CPU, frozen cohort and candidate model identities differ")
        rope = model.backbone.layers[0].self_attn.rope
        if not isinstance(rope, _RoPE) or any(
            layer.self_attn.rope is not rope for layer in model.backbone.layers
        ):
            raise ValueError("expected one shared native rotary table owner")
        batches = prepare(model, frozen)
        if set(batches) != set(refs):
            raise ValueError("frozen/CPU case identities differ")
        result.update(
            metadata=metadata(),
            model_id=model.model_id,
            memory_after_load=memory("metal"),
            rotary_owner={
                "layers": len(model.backbone.layers),
                "distinct_owner_count": len(
                    {id(layer.self_attn.rope) for layer in model.backbone.layers}
                ),
                "all_layers_share_first_owner": True,
                "initial_tables": table_state(rope),
            },
        )

        def candidate(owner: _RoPE, value: mx.array, offset: int = 0) -> mx.array:
            if owner is not rope:
                return original(owner, value, offset)
            if (
                type(offset) is not int
                or offset < 0
                or value.ndim < 2
                or value.shape[-1] != owner.dims
                or value.shape[-2] < 1
                or value.dtype != mx.float32
            ):
                return original(owner, value, offset)
            # Preserve original validation and cache creation on a cold shape.
            # In this paired screen baseline runs first for every shape.
            key = (value.shape[-2], offset)
            if key not in owner._tables:
                return original(owner, value, offset)
            cosine, sine = owner._tables[key]
            return fused(value, cosine, sine)

        def select(label: str) -> None:
            cast(Any, _RoPE).__call__ = original if label == "baseline" else candidate

        for name, rows in batches.items():
            entry: dict[str, Any] = {"id": name, "arms": {}}
            result["quality_cases"].append(entry)
            for label in ("baseline", "candidate"):
                select(label)
                entry["arms"][label] = forward(model, rows, args.pause_ms)
            entry["quality"] = {
                label: compare_rows(
                    refs[name]["outputs"],
                    entry["arms"][label]["outputs"],
                    frozen["quality_gates"],
                )
                for label in ("baseline", "candidate")
            }
            entry["cross_quality"] = compare_rows(
                entry["arms"]["baseline"]["outputs"],
                entry["arms"]["candidate"]["outputs"],
                frozen["quality_gates"],
            )
            entry["passed"] = entry["cross_quality"]["pass"] and all(
                gate["pass"] for gate in entry["quality"].values()
            )
            checkpoint(args.output, result)
        result["quality_pass"] = all(case["passed"] for case in result["quality_cases"])
        if not result["quality_pass"]:
            raise RuntimeError(
                "complete CPU/cross-arm output gate failed; timing skipped"
            )
        for name in ("single-00", "batch8-short", "batch8-mixed"):
            rows = batches[name]
            entry = {"id": name, "warmup": {}, "pairs": []}
            result["timing_cases"].append(entry)
            for label in ("baseline", "candidate"):
                select(label)
                entry["warmup"][label] = forward(model, rows, args.pause_ms)
            entry["warmup_quality"] = {
                label: compare_rows(
                    refs[name]["outputs"],
                    entry["warmup"][label]["outputs"],
                    frozen["quality_gates"],
                )
                for label in ("baseline", "candidate")
            }
            for index in range(args.pairs):
                pair: dict[str, Any] = {
                    "index": index,
                    "order": ["baseline", "candidate"]
                    if index % 2 == 0
                    else ["candidate", "baseline"],
                }
                for label in pair["order"]:
                    select(label)
                    pair[label] = forward(model, rows, args.pause_ms)
                pair["quality"] = {
                    label: compare_rows(
                        refs[name]["outputs"],
                        pair[label]["outputs"],
                        frozen["quality_gates"],
                    )
                    for label in ("baseline", "candidate")
                }
                pair["passed"] = all(gate["pass"] for gate in pair["quality"].values())
                pair["cross_quality"] = compare_rows(
                    pair["baseline"]["outputs"],
                    pair["candidate"]["outputs"],
                    frozen["quality_gates"],
                )
                pair["passed"] = pair["passed"] and pair["cross_quality"]["pass"]
                entry["pairs"].append(pair)
                checkpoint(args.output, result)
            valid = [
                pair
                for pair in entry["pairs"]
                if pair["baseline"]["error"] is None
                and pair["candidate"]["error"] is None
            ]
            speeds = [
                pair["baseline"]["elapsed_ms"] / pair["candidate"]["elapsed_ms"]
                for pair in valid
            ]
            entry["summary"] = {
                "complete_pairs": len(valid),
                "requested_pairs": args.pairs,
                "median_speedup": statistics.median(speeds) if speeds else None,
                "p05_speedup": quantile(speeds, 0.05) if speeds else None,
                "quality_pass": (
                    all(pair["passed"] for pair in entry["pairs"])
                    and all(gate["pass"] for gate in entry["warmup_quality"].values())
                ),
            }
            entry["memory_after_case"] = memory("metal")
        result["completed"] = True
        result["passed"] = result["quality_pass"] and all(
            case["summary"]["quality_pass"] for case in result["timing_cases"]
        )
    except Exception as error:
        result["error"] = repr(error)
    finally:
        cast(Any, _RoPE).__call__ = original
        result["memory_at_end"] = memory("metal")
        mx.synchronize()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache)
        mx.set_memory_limit(previous_memory)
        result["source_end_sha256"] = {str(path): digest(path) for path in sources}
        result["source_fence_pass"] = (
            result["source_sha256"] == result["source_end_sha256"]
        )
        result["passed"] = result["passed"] and result["source_fence_pass"]
        finish(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
