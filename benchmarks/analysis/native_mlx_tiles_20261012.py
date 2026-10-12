"""Bounded native Metal direct-tile screen with unchanged split-K and FP32 math.

One model owns all weights. Only its shared projection owner's direct callable
changes; raw outputs and numerical/ranking failures remain in the receipt.
This exploratory screen does not qualify a production kernel or a speedup.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
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
from wyat.backends.kai.metal import MetalProjection, _direct, _MetalLinear, _partitions
from wyat.loading.mlx import load_native_mlx

BASELINE = (16, 32, 2)
CANDIDATES = ((32, 32, 4), (32, 64, 4))
CASES = ("single-00", "batch8-short", "batch8-mixed")


def owner_and_inventory(model: Any, batches: Any) -> tuple[MetalProjection, Any]:
    modules = [
        (name, module)
        for name, module in model.backbone.named_modules()
        if isinstance(module, _MetalLinear)
    ]
    owners = {id(module.operation): module.operation for _, module in modules}
    if len(owners) != 1 or len(modules) != 7 * len(model.backbone.layers):
        raise ValueError("expected seven base projections per layer with one owner")
    owner = next(iter(owners.values()))
    if not isinstance(owner, MetalProjection):
        raise ValueError("unexpected native projection owner")
    inventory = []
    for name, module in modules:
        columns, reduction = module.weight.shape
        routes = {}
        for case, encoded in batches.items():
            rows = len(encoded) * max(len(row.ids) for row in encoded)
            partitions = _partitions(rows, reduction, columns)
            routes[case] = {
                "flattened_rows": rows,
                "partitions": list(partitions),
                "uses_direct": partitions[0] == 1,
            }
        inventory.append(
            {
                "name": name,
                "weight_shape": list(module.weight.shape),
                "weight_dtype": str(module.weight.dtype),
                "routes": routes,
            }
        )
    return owner, inventory


def run(model: Any, rows: Any, pause_ms: float) -> dict[str, Any]:
    result: dict[str, Any] = {"error": None, "outputs": None}
    try:
        mx.synchronize(model.device)
        start = time.perf_counter_ns()
        values = model._logits(rows)
        mx.synchronize(model.device)
        result.update(
            elapsed_ms=(time.perf_counter_ns() - start) / 1e6,
            outputs=snapshot(rows, values),
        )
    except Exception as error:
        result["error"] = repr(error)
    finally:
        # Outside the measured interval, between every complete forward.
        time.sleep(pause_ms / 1000)
    return result


def summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        pair
        for pair in pairs
        if pair["baseline"]["error"] is None and pair["candidate"]["error"] is None
    ]
    speeds = [
        pair["baseline"]["elapsed_ms"] / pair["candidate"]["elapsed_ms"]
        for pair in valid
    ]
    return {
        "complete_pairs": len(valid),
        "requested_pairs": len(pairs),
        "all_pairs_retained": True,
        "median_speedup": statistics.median(speeds) if speeds else None,
        "p05_speedup": quantile(speeds, 0.05) if speeds else None,
        "aggregate_speedup": (
            sum(pair["baseline"]["elapsed_ms"] for pair in valid)
            / sum(pair["candidate"]["elapsed_ms"] for pair in valid)
            if valid
            else None
        ),
        "quality_pass": all(pair["passed"] for pair in pairs),
        "qualification": "exploratory screen only; five pairs cannot qualify promotion",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--case", action="append", choices=CASES, default=[])
    parser.add_argument(
        "--candidate", action="append", choices=("32-32-4", "32-64-4"), default=[]
    )
    parser.add_argument("--pause-ms", type=float, default=250)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 5:
        parser.error("bounded screen permits 1..5 pairs only")
    if not math.isfinite(args.pause_ms) or args.pause_ms < 250:
        parser.error("at least 250 ms idle is required between complete forwards")
    for suffix in ("", ".partial", ".partial.tmp"):
        if args.output.with_name(args.output.name + suffix).exists():
            parser.error("output/progress receipt exists; choose a fresh path")
    names = args.case or list(CASES)
    candidates = (
        [tuple(map(int, name.split("-"))) for name in args.candidate]
        if args.candidate
        else list(CANDIDATES)
    )
    if len(set(names)) != len(names) or len(set(candidates)) != len(candidates):
        parser.error("duplicate cases/candidates are not allowed")
    frozen = read(args.frozen)
    check_frozen(frozen)
    result: dict[str, Any] = {
        "schema": "native-mlx-direct-tiles-v1",
        "completed": False,
        "passed": False,
        "frozen_sha256": digest(args.frozen),
        "artifact_manifest_sha256": digest(args.artifact / "native_manifest.json"),
        "protocol": {
            "baseline_tile_m_n_simdgroups": BASELINE,
            "candidate_tiles_m_n_simdgroups": candidates,
            "pairs_per_case_per_candidate": args.pairs,
            "one_model": True,
            "change": (
                "shared MetalProjection.direct only; split-K and weights unchanged"
            ),
            "paired_order": "baseline/candidate then candidate/baseline, alternating",
            "warmup": "one baseline reference plus one candidate forward per case",
            "pause_ms_between_forwards": args.pause_ms,
            "responsive": True,
            "layer_pacing": (
                "unchanged native _hidden default: eval plus 1 ms idle every "
                "four layers before the final block group"
            ),
            "timing": (
                "exact native _logits including packing, native responsive pacing, "
                "GPU completion and host logits; native interpretation outside timer"
            ),
            "cache_limit_bytes": 256 << 20,
            "mlx_memory_limit_bytes": 6144 << 20,
            "memory_scope": (
                "process-local MLX allocator guideline, not OS hard RSS cap"
            ),
            "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
            "quality_gates": frozen["quality_gates"],
        },
        "cases": [],
    }
    source_paths = [
        Path(__file__).resolve(),
        Path(inspect.getfile(prepare)).resolve(),
        Path(inspect.getfile(load_native_mlx)).resolve(),
        Path(inspect.getfile(_direct)).resolve(),
        Path(inspect.getfile(activations)).resolve(),
        Path(inspect.getfile(base)).resolve(),
        Path(inspect.getfile(qwen3)).resolve(),
        args.frozen.resolve(),
        (args.artifact / "native_manifest.json").resolve(),
    ]
    result["source_sha256"] = {str(path): digest(path) for path in source_paths}
    checkpoint(args.output, result)
    previous_cache = mx.set_cache_limit(256 << 20)
    previous_memory = mx.set_memory_limit(6144 << 20)
    owner, original_direct = None, None
    try:
        model = load_native_mlx(args.artifact, metal=True, responsive=True)
        if not model.metal_enabled:
            raise RuntimeError("Metal not admitted; candidate cannot be evaluated")
        all_batches = prepare(model, frozen)
        batches = {name: all_batches[name] for name in names}
        owner, inventory = owner_and_inventory(model, batches)
        original_direct = owner.direct
        baseline = _direct(*BASELINE)
        result.update(
            metadata=metadata(),
            model_id=model.model_id,
            metal_enabled=model.metal_enabled,
            projection_inventory=inventory,
            memory_after_load=memory("metal"),
        )
        for name, rows in batches.items():
            entry: dict[str, Any] = {
                "id": name,
                "lengths": [len(row.ids) for row in rows],
                "candidate_counts": [len(row.candidate_positions) for row in rows],
                "candidates": [],
                "passed": False,
            }
            result["cases"].append(entry)
            owner.direct = baseline
            reference = run(model, rows, args.pause_ms)
            entry["baseline_warmup_reference"] = reference
            if reference["error"] is not None:
                entry["error"] = (
                    "baseline reference failed; candidates retained as unknown"
                )
                checkpoint(args.output, result)
                continue
            for geometry in candidates:
                candidate = _direct(*geometry)
                arm: dict[str, Any] = {"tile": geometry, "pairs": []}
                entry["candidates"].append(arm)
                owner.direct = candidate
                arm["warmup"] = run(model, rows, args.pause_ms)
                arm["warmup_quality"] = compare_rows(
                    reference["outputs"],
                    arm["warmup"]["outputs"],
                    frozen["quality_gates"],
                )
                for index in range(args.pairs):
                    pair: dict[str, Any] = {
                        "index": index,
                        "order": ["baseline", "candidate"]
                        if index % 2 == 0
                        else ["candidate", "baseline"],
                    }
                    for label in pair["order"]:
                        owner.direct = baseline if label == "baseline" else candidate
                        pair[label] = run(model, rows, args.pause_ms)
                    pair["quality"] = {
                        label: compare_rows(
                            reference["outputs"],
                            pair[label]["outputs"],
                            frozen["quality_gates"],
                        )
                        for label in ("baseline", "candidate")
                    }
                    pair["cross_quality"] = compare_rows(
                        pair["baseline"]["outputs"],
                        pair["candidate"]["outputs"],
                        frozen["quality_gates"],
                    )
                    pair["passed"] = (
                        pair["cross_quality"]["pass"]
                        and all(gate["pass"] for gate in pair["quality"].values())
                        and all(
                            pair[label]["error"] is None
                            for label in ("baseline", "candidate")
                        )
                    )
                    arm["pairs"].append(pair)
                    checkpoint(args.output, result)
                arm["summary"] = summary(arm["pairs"])
                arm["passed"] = (
                    arm["warmup_quality"]["pass"] and arm["summary"]["quality_pass"]
                )
                arm["memory_after_candidate"] = memory("metal")
                checkpoint(args.output, result)
            entry["passed"] = bool(entry["candidates"]) and all(
                arm["passed"] for arm in entry["candidates"]
            )
        result["completed"] = True
        result["passed"] = all(case["passed"] for case in result["cases"])
    except Exception as error:
        result["error"] = repr(error)
    finally:
        if owner is not None and original_direct is not None:
            owner.direct = original_direct
        result["memory_at_end"] = memory("metal")
        mx.synchronize()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache)
        mx.set_memory_limit(previous_memory)
        result["source_end_sha256"] = {str(path): digest(path) for path in source_paths}
        result["source_fence_pass"] = (
            result["source_sha256"] == result["source_end_sha256"]
        )
        result["passed"] = result["passed"] and result["source_fence_pass"]
        finish(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
