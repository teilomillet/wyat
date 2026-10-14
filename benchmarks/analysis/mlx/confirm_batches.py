"""Fresh B2/B8 baseline confirmation with continuous default-cache history."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.mlx.cache import DATASET_SHA256, memory, paired_orders, swap
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import EvaluationCase, case_from_dict, percentile
from wyat.wire import decode_json


def observe(
    engine: BenchmarkEngine,
    cases: tuple[EvaluationCase, ...],
    reference: BenchmarkCycle,
) -> tuple[BenchmarkCycle, dict[str, object]]:
    before_swap = swap()
    before = process_diagnostic(device_memory_bytes=memory())
    cycle = _run(engine, cases)
    after = process_diagnostic(device_memory_bytes=memory())
    after_swap = swap()
    gate = compare_cycles((reference,), (cycle,), QualityGate())
    return cycle, {
        "batch_size": engine.batch_size,
        "cycle": cycle.to_dict(),
        "quality": asdict(gate),
        "passed": gate.passed,
        "process_before": asdict(before),
        "process_after": asdict(after),
        "swap_before": before_swap,
        "swap_after": after_swap,
    }


def confirm(model_path: Path, output: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "cache_telemetry": root / "benchmarks/analysis/mlx/cache.py",
            "evidence": root / "benchmarks/analysis/evidence.py",
            "reference": root / "benchmarks/analysis/reference.py",
            "backend": root / "wyat/backends/wyat/mlx.py",
            "benchmark": root / "wyat/benchmark.py",
        },
        output,
    )
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if dataset_hash != DATASET_SHA256:
        raise ValueError("Frozen cohort hash changed")
    cases = tuple(
        case_from_dict(decode_json(row)) for row in dataset.read_text().splitlines()
    )
    if len(cases) != 77:
        raise ValueError("Confirmation requires every frozen development case")
    scorer = MLXWyatScorer.from_directory(
        model_path, max_length=512, token_cache_size=0, device="gpu", compile=False
    )
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    reference = read_reference(
        reference_path, cases, scorer, dataset_sha256=dataset_hash
    )
    engines = {
        size: BenchmarkEngine(
            f"mlx-fp32-batch{size}", Model(scorer, scorer.model_id), size
        )
        for size in (2, 8)
    }
    # No cap mutation or clearing: the fresh process retains normal load and
    # variable-shape cache history through both warmups and all paired samples.
    warmups = [observe(engine, cases, reference)[1] for engine in engines.values()]
    measured: dict[int, list[BenchmarkCycle]] = {2: [], 8: []}
    samples: list[dict[str, object]] = []
    orders = paired_orders(5, reverse=False)
    with output.with_suffix(".cycles.jsonl").open("w", encoding="utf-8") as stream:
        for repeat, order in enumerate(orders):
            for arm in order:
                size = 2 if arm == "default" else 8
                cycle, measurement = observe(engines[size], cases, reference)
                measured[size].append(cycle)
                measurement["repeat"] = repeat
                samples.append(measurement)
                stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                stream.flush()
            print(json.dumps({"completed_pair": repeat + 1}), flush=True)
    ratios = [
        first.latency_ms / second.latency_ms
        for first, second in zip(measured[2], measured[8], strict=True)
    ]
    result: dict[str, object] = {
        "scope": "fresh B2/B8 full77 default-cache FP32 baseline confirmation",
        "completed": True,
        "passed": all(row["passed"] is True for row in warmups + samples),
        "model_id": scorer.model_id,
        "dataset_sha256": dataset_hash,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_sha256": sources.hashes(),
        "source_archive": sources.archive(),
        "device": "gpu",
        "dtype": "float32",
        "compile": False,
        "token_cache_size": 0,
        "protocol": {
            "warmup_per_shape": 1,
            "pair_orders_batch_sizes": [
                [2 if arm == "default" else 8 for arm in order] for order in orders
            ],
            "allocator": "fresh process default; no mutation/clearing between arms",
            "quality_gate": asdict(QualityGate()),
            "timing": "complete cohort; telemetry, warmup, loading and JSON excluded",
        },
        "warmups": warmups,
        "samples": samples,
        "paired_b8_speedups_over_b2": ratios,
        "median_speedup": percentile(ratios, 0.5),
        "p05_speedup": percentile(ratios, 0.05),
        "p95_speedup": percentile(ratios, 0.95),
        "limits": [
            "Two shapes do not reproduce the original six-shape cache history.",
            "No cycle removed; no historical-outlier cause can be inferred.",
            "Counters include untimed snapshots; global swap is noncausal context.",
            "Source archive attests disk bytes, not already-imported execution.",
        ],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


class Arguments(Namespace):
    model: Path
    output: Path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/mlx-batch2-8-confirm77.json"),
    )
    options = parser.parse_args(namespace=Arguments())
    if confirm(options.model, options.output)["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
