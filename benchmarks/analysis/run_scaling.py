"""Bounded six-shape FP32 MLX experiment on the fixed real 77-case dev cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

from wyat import Model
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import (
    _input_digest,
    case_from_dict,
    percentile,
)
from wyat.wire import decode_json, read_object

from .reference import balanced_orders, read_reference
from .scaling import ScalingPoint, TransformerShape, estimate_batch_work, fit_usl

BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DATASET_SHA256 = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape(model_path: Path) -> TransformerShape:
    config = read_object(model_path / "backbone/config.json")
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    )
    values = tuple(config[key] for key in keys)
    if any(type(value) is not int for value in values):
        raise ValueError("Transformer dimensions must be declared integers")
    return TransformerShape(*(cast(int, value) for value in values))


def _work(
    lengths: Sequence[int], shape: TransformerShape, batch_size: int
) -> dict[str, object]:
    batches = tuple(
        estimate_batch_work(lengths[start : start + batch_size], shape)
        for start in range(0, len(lengths), batch_size)
    )
    useful = sum(batch.useful_tokens for batch in batches)
    padded = sum(batch.padded_tokens for batch in batches)
    return {
        "batches": [asdict(batch) for batch in batches],
        "useful_tokens": useful,
        "padded_tokens": padded,
        "padding_fraction": 1 - useful / padded,
        "projection_flops": sum(batch.projection_flops for batch in batches),
        "dense_attention_flops": sum(batch.dense_attention_flops for batch in batches),
        "ideal_causal_attention_flops": sum(
            batch.ideal_causal_attention_flops for batch in batches
        ),
    }


def sweep(model_path: Path, output: Path, *, repeats: int = 6) -> dict[str, object]:
    if type(repeats) is not int or repeats < 6 or repeats % 6:
        raise ValueError(
            "Use a positive multiple of six repetitions for balanced order"
        )
    root = Path(__file__).resolve().parents[2]
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    if digest(dataset) != DATASET_SHA256:
        raise ValueError("Declared real development cohort hash changed")
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    if len(cases) != 77:
        raise ValueError("Scaling experiment requires the complete 77-case cohort")
    output.parent.mkdir(parents=True, exist_ok=True)
    scorer = MLXKaiScorer.from_directory(
        model_path, max_length=512, token_cache_size=0, device="gpu", compile=False
    )
    reference = read_reference(
        reference_path, cases, scorer, dataset_sha256=DATASET_SHA256
    )
    model = Model(scorer, scorer.model_id)
    lengths = tuple(len(scorer.encode_request(case.request).ids) for case in cases)
    shape = _shape(model_path)
    gate = QualityGate()
    engines = {
        size: BenchmarkEngine(f"mlx-fp32-batch{size}", model, size)
        for size in BATCH_SIZES
    }
    warmup = {size: _run(engine, cases) for size, engine in engines.items()}
    measured: dict[int, list[BenchmarkCycle]] = {size: [] for size in BATCH_SIZES}
    journal = output.with_suffix(".cycles.jsonl")
    orders = balanced_orders(BATCH_SIZES, repeats)
    with journal.open("w", encoding="utf-8") as stream:
        for repeat, order in enumerate(orders):
            for size in order:
                cycle = _run(engines[size], cases)
                measured[size].append(cycle)
                stream.write(
                    json.dumps(
                        {
                            "repeat": repeat,
                            "batch_size": size,
                            "cycle": cycle.to_dict(),
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
            print(
                json.dumps(
                    {"completed_round": repeat + 1, "rounds": repeats, "order": order}
                ),
                flush=True,
            )
    sizes: dict[str, object] = {}
    points: list[ScalingPoint] = []
    for size in BATCH_SIZES:
        cycles = tuple(measured[size])
        quality = compare_cycles((reference,) * repeats, cycles, gate)
        seconds = math.fsum(cycle.latency_ms for cycle in cycles) / 1000
        attempted = len(cases) * repeats
        failed = sum(cycle.failed for cycle in cycles)
        successful = attempted - failed
        timings = [cycle.latency_ms for cycle in cycles]
        residence = [row.latency_ms for cycle in cycles for row in cycle.rows]
        successful_rps = successful / seconds
        warmup_quality = compare_cycles((reference,), (warmup[size],), gate)
        passed = quality.passed and warmup_quality.passed
        if passed:
            points.append(ScalingPoint(size, successful_rps))
        sizes[str(size)] = {
            "quality": asdict(quality),
            "warmup_quality": asdict(warmup_quality),
            "passed": passed,
            "attempted": attempted,
            "failed": failed,
            "successful_requests_per_second": successful_rps,
            "equivalent_requests_per_second": successful_rps if passed else 0,
            "labeled_correct_requests_per_second": sum(
                cycle.correct for cycle in cycles
            )
            / seconds,
            "cycle_p50_ms": percentile(timings, 0.5),
            "cycle_p95_ms": percentile(timings, 0.95),
            "batch_residence_p50_ms": percentile(residence, 0.5),
            "batch_residence_p95_ms": percentile(residence, 0.95),
            "measured_cycles": [cycle.to_dict() for cycle in cycles],
            "warmup_cycle": warmup[size].to_dict(),
            "work": _work(lengths, shape, size),
        }
    fit = asdict(fit_usl(points)) if len(points) == 6 else None
    result: dict[str, object] = {
        "schema_version": 1,
        "scope": "real fixed-cohort FP32 engine scaling experiment",
        "model_id": scorer.model_id,
        "dataset_sha256": DATASET_SHA256,
        "reference_sha256": digest(reference_path),
        "input_sha256": _input_digest(cases),
        "case_ids": [case.id for case in cases],
        "token_lengths": list(lengths),
        "token_length_summary": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": math.fsum(lengths) / len(lengths),
        },
        "shape": asdict(shape),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": version("mlx"),
            "mlx_lm": version("mlx-lm"),
            "device": "gpu",
            "dtype": "float32",
            "compile": False,
            "token_cache_size": 0,
        },
        "protocol": {
            "batch_sizes": BATCH_SIZES,
            "repeats": repeats,
            "warmup_per_size": 1,
            "orders": orders,
            "quality_gate": asdict(gate),
            "cohort_order": "unchanged canonical caller order",
            "timing": "complete cycles; native host-float synchronization; "
            "model/packing/tokenization/decision bookkeeping",
            "excludes": ["loading", "warmup", "JSON serialization", "HTTP"],
        },
        "sizes": sizes,
        "usl_descriptive_batch_fit": fit,
        "roofline": {
            "status": "unknown",
            "reason": "No independent sustained compute/DRAM ceiling and actual "
            "traffic measurements supplied.",
        },
        "source_sha256": {
            "runner": digest(Path(__file__)),
            "scaling": digest(root / "benchmarks/analysis/scaling.py"),
            "reference": digest(root / "benchmarks/analysis/reference.py"),
            "mlx_backend": digest(root / "wyat/backends/kai/mlx.py"),
        },
        "limits": [
            "Batch size is not processor count; coefficients do not identify "
            "cache coherence.",
            "Same 77 labeled development cases repeated; no held-out quality claim.",
            "Arithmetic excludes norms/RoPE/softmax/activation/head; causal "
            "flash schedule is unknown.",
            "Process peak RSS is shared cumulative high-water, not isolated "
            "per-shape device memory.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


class Arguments(argparse.Namespace):
    model: Path
    output: Path
    repeats: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/mlx-scaling77.json"),
    )
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args(namespace=Arguments())
    start = time.perf_counter()
    result = sweep(args.model, args.output, repeats=args.repeats)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": time.perf_counter() - start,
                "input_sha256": result["input_sha256"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
