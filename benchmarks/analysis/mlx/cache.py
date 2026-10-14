"""Controlled warm-per-sample comparison of inactive MLX allocator cache caps."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
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

DATASET_SHA256 = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"
CAPS = (512 * 1024**2, 1024**3)


def paired_orders(repeats: int, *, reverse: bool) -> tuple[tuple[str, str], ...]:
    """Alternate arm order; reverse the next comparison to balance odd repeats."""
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    return tuple(
        ("capped", "default") if (index + int(reverse)) % 2 else ("default", "capped")
        for index in range(repeats)
    )


def memory() -> dict[str, int]:
    """A synchronized snapshot; peak measures active bytes only."""
    mx.synchronize()
    active = int(mx.get_active_memory())
    cached = int(mx.get_cache_memory())
    return {
        "active": active,
        "cache": cached,
        "active_plus_cache": active + cached,
        "peak_active": int(mx.get_peak_memory()),
    }


def swap() -> dict[str, object]:
    """Global untimed observation, retained verbatim without causal attribution."""
    wall = time.time_ns()
    start = time.perf_counter_ns()
    result: dict[str, object] = {
        "wall_time_ns": wall,
        "command": ["/usr/sbin/sysctl", "vm.swapusage"],
    }
    try:
        response = subprocess.run(
            ["/usr/sbin/sysctl", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        result.update(
            returncode=response.returncode,
            stdout=response.stdout,
            stderr=response.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["collection_elapsed_ns"] = time.perf_counter_ns() - start
    return result


def sample(
    engine: BenchmarkEngine,
    cases: tuple[EvaluationCase, ...],
    reference: BenchmarkCycle,
    cap: int,
) -> tuple[BenchmarkCycle, dict[str, object]]:
    """Identical clear/warm treatment precedes every arm's measured cohort."""
    mx.synchronize()
    previous = int(mx.set_cache_limit(cap))
    mx.clear_cache()
    mx.reset_peak_memory()
    cleared = memory()
    warm_before = process_diagnostic(device_memory_bytes=cleared)
    warmup = _run(engine, cases)
    warm_memory = memory()
    warm_after = process_diagnostic(device_memory_bytes=warm_memory)
    mx.reset_peak_memory()
    before_memory = memory()
    before_swap = swap()
    before = process_diagnostic(device_memory_bytes=before_memory)
    cycle = _run(engine, cases)
    after_memory = memory()
    after = process_diagnostic(device_memory_bytes=after_memory)
    after_swap = swap()
    gate = QualityGate()
    warmup_quality = compare_cycles((reference,), (warmup,), gate)
    measured_quality = compare_cycles((reference,), (cycle,), gate)
    return cycle, {
        "cap_bytes": cap,
        "previous_cap_bytes": previous,
        "cleared_memory_bytes": cleared,
        "warmup_memory_bytes": warm_memory,
        "warmup_cycle": warmup.to_dict(),
        "warmup_quality": asdict(warmup_quality),
        "warmup_process_before": asdict(warm_before),
        "warmup_process_after": asdict(warm_after),
        "cycle": cycle.to_dict(),
        "quality": asdict(measured_quality),
        "passed": warmup_quality.passed and measured_quality.passed,
        "process_before": asdict(before),
        "process_after": asdict(after),
        "swap_before": before_swap,
        "swap_after": after_swap,
    }


def study(model_path: Path, output: Path, *, batch_size: int = 8) -> dict[str, object]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = Path(__file__).resolve().parents[3]
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "backend": root / "wyat/backends/wyat/mlx.py",
            "benchmark": root / "wyat/benchmark.py",
            "reference": root / "benchmarks/analysis/reference.py",
            "evidence": root / "benchmarks/analysis/evidence.py",
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
        raise ValueError("Study requires every frozen development case")
    # The setter returns the previous cap; no cache-cap getter exists in this pin.
    original = int(mx.set_cache_limit(CAPS[0]))
    mx.set_cache_limit(original)
    result: dict[str, object] = {
        "scope": "full77 controlled warm-per-sample inactive allocator cap study",
        "dataset_sha256": dataset_hash,
        "source_sha256": sources.hashes(),
        "source_archive": sources.archive(),
        "original_cache_cap_bytes": original,
        "batch_size": batch_size,
        "completed": False,
        "passed": False,
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
            "repeats_per_cap": 5,
            "quality_gate": asdict(QualityGate()),
            "treatment": "synchronize,set cap,clear,one full-cohort warm,reset "
            "active peak,untimed diagnostics,one timed full-cohort sample",
            "order": "alternate each paired repeat; reverse initial order for "
            "second cap so default/capped arm positions balance across10pairs",
            "excludes": [
                "loading",
                "cache clearing",
                "warmup",
                "telemetry",
                "JSON",
                "HTTP",
            ],
        },
        "limits": [
            "One fixed batch shape with per-sample cache clearing does not establish "
            "long-lived variable-shape allocator behavior.",
            "The original cap is restored; no production setting changes.",
            "Peak is active allocation only; active+cache is separately retained.",
            "Cumulative process counters/global swap do not establish a cause.",
            "Same77 training-source examples; released-model exposure unknown.",
            "Source snapshots attest setup-time disk bytes, not already-imported "
            "code or external dependency execution.",
        ],
    }
    reports: dict[str, object] = {}
    passed_reports: list[bool] = []
    try:
        scorer = MLXWyatScorer.from_directory(
            model_path, max_length=512, token_cache_size=0, device="gpu", compile=False
        )
        reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
        reference = read_reference(
            reference_path, cases, scorer, dataset_sha256=dataset_hash
        )
        result["model_id"] = scorer.model_id
        result["reference_sha256"] = hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest()
        engine = BenchmarkEngine(
            "mlx-fp32-cache-study", Model(scorer, scorer.model_id), batch_size
        )
        journal = output.with_suffix(".cycles.jsonl")
        with journal.open("w", encoding="utf-8") as stream:
            for comparison, cap in enumerate(CAPS):
                arms: dict[str, list[BenchmarkCycle]] = {"default": [], "capped": []}
                samples: list[dict[str, object]] = []
                orders = paired_orders(5, reverse=bool(comparison))
                for repeat, order in enumerate(orders):
                    for arm in order:
                        cycle, measurement = sample(
                            engine,
                            cases,
                            reference,
                            original if arm == "default" else cap,
                        )
                        arms[arm].append(cycle)
                        measurement.update(
                            comparison_cap_bytes=cap, repeat=repeat, arm=arm
                        )
                        samples.append(measurement)
                        stream.write(json.dumps(measurement, allow_nan=False) + "\n")
                        stream.flush()
                    print(
                        json.dumps({"cap_bytes": cap, "completed_pair": repeat + 1}),
                        flush=True,
                    )
                ratios = [
                    baseline.latency_ms / candidate.latency_ms
                    for baseline, candidate in zip(
                        arms["default"], arms["capped"], strict=True
                    )
                ]
                reports[str(cap)] = {
                    "orders": orders,
                    "samples": samples,
                    "passed": all(row["passed"] is True for row in samples),
                    "paired_speedups": ratios,
                    "median_speedup": percentile(ratios, 0.5),
                    "p05_speedup": percentile(ratios, 0.05),
                    "p95_speedup": percentile(ratios, 0.95),
                    "timing": {
                        arm: {
                            "cohort_p50_ms": percentile(
                                [cycle.latency_ms for cycle in cycles], 0.5
                            ),
                            "cohort_p95_ms": percentile(
                                [cycle.latency_ms for cycle in cycles], 0.95
                            ),
                            "successful_requests_per_second": sum(
                                77 - cycle.failed for cycle in cycles
                            )
                            / (math.fsum(cycle.latency_ms for cycle in cycles) / 1000),
                        }
                        for arm, cycles in arms.items()
                    },
                }
                passed_reports.append(all(row["passed"] is True for row in samples))
        result["completed"] = len(reports) == len(CAPS)
        result["passed"] = len(passed_reports) == len(CAPS) and all(passed_reports)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        previous = int(mx.set_cache_limit(original))
        result["restored_from_cache_cap_bytes"] = previous
        result["restored_cache_cap_bytes"] = original
        try:
            result["final_memory_bytes"] = memory()
        except Exception as error:
            result["final_memory_error"] = f"{type(error).__name__}: {error}"
        result["reports"] = reports
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


class Arguments(Namespace):
    model: Path
    output: Path
    batch_size: int


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/mlx-cache77-b8.json"),
    )
    options = parser.parse_args(namespace=Arguments())
    result = study(options.model, options.output, batch_size=options.batch_size)
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
