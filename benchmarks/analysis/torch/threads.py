"""Bounded six-count CPU FP32 thread sweep; run after the Metal handoff.

python -m benchmarks.analysis.torch.threads MODEL --output RECEIPT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch

from wyat import Model
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict, percentile
from wyat.wire import decode_json

from ..evidence import freeze_sources, process_diagnostic
from ..reference import balanced_orders, read_reference
from ..scaling import ScalingPoint, fit_usl

THREAD_COUNTS = (1, 2, 4, 6, 8, 12)
DATASET_SHA256 = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sweep(model_path: Path, output: Path, *, repeats: int = 6) -> dict[str, object]:
    orders = balanced_orders(THREAD_COUNTS, repeats)
    root = Path(__file__).resolve().parents[3]
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "reference": root / "benchmarks/analysis/reference.py",
            "scaling": root / "benchmarks/analysis/scaling.py",
            "evidence": root / "benchmarks/analysis/evidence.py",
            "backend": root / "wyat/backends/kai/torch.py",
            "encoding": root / "wyat/backends/kai/encoding.py",
            "artifact": root / "wyat/backends/kai/artifact.py",
            "core": root / "wyat/core.py",
            "benchmark": root / "wyat/benchmark.py",
            "evaluation": root / "wyat/evaluation.py",
            "wire": root / "wyat/wire.py",
        },
        output,
    )
    reference_sha256 = _digest(reference_path)
    if _digest(dataset) != DATASET_SHA256:
        raise ValueError("Declared development cohort hash changed")
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    if len(cases) != 77:
        raise ValueError("Thread experiment requires all 77 declared cases")
    original_threads = torch.get_num_threads()
    interop_threads = torch.get_num_interop_threads()
    torch.set_num_threads(8)
    try:
        scorer = KaiScorer.from_directory(
            model_path,
            device="cpu",
            max_length=512,
            token_cache_size=0,
            attention="sdpa",
        )
        reference = read_reference(
            reference_path, cases, scorer, dataset_sha256=DATASET_SHA256
        )
        model = Model(scorer, scorer.model_id)
        token_lengths = [len(scorer.encode_request(case.request).ids) for case in cases]
        engines = {
            count: BenchmarkEngine(f"torch-fp32-intraop{count}", model, 8)
            for count in THREAD_COUNTS
        }
        gate = QualityGate()
        warmup: dict[int, BenchmarkCycle] = {}
        measured: dict[int, list[BenchmarkCycle]] = {
            count: [] for count in THREAD_COUNTS
        }
        parallel: dict[int, str] = {}
        warmup_diagnostics: dict[int, dict[str, object]] = {}
        measured_diagnostics: dict[int, list[dict[str, object]]] = {
            count: [] for count in THREAD_COUNTS
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        journal = output.with_suffix(".cycles.jsonl")
        with journal.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage": "setup",
                        "source_sha256": sources.hashes(),
                        "source_archive": sources.archive(),
                        "reference_sha256": reference_sha256,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            for count in THREAD_COUNTS:
                torch.set_num_threads(count)
                parallel[count] = torch.__config__.parallel_info()
                before = process_diagnostic()
                warmup[count] = _run(engines[count], cases)
                after = process_diagnostic()
                warmup_diagnostics[count] = {
                    "before": asdict(before),
                    "after": asdict(after),
                }
                stream.write(
                    json.dumps(
                        {
                            "stage": "warmup",
                            "intraop_threads": count,
                            "cycle": warmup[count].to_dict(),
                            "diagnostics": warmup_diagnostics[count],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
            for repeat, order in enumerate(orders):
                for count in order:
                    torch.set_num_threads(count)
                    if (
                        torch.get_num_threads() != count
                        or torch.get_num_interop_threads() != interop_threads
                    ):
                        raise RuntimeError(
                            "Process thread settings changed unexpectedly"
                        )
                    before = process_diagnostic()
                    cycle = _run(engines[count], cases)
                    after = process_diagnostic()
                    measured[count].append(cycle)
                    diagnostics: dict[str, object] = {
                        "before": asdict(before),
                        "after": asdict(after),
                    }
                    measured_diagnostics[count].append(diagnostics)
                    stream.write(
                        json.dumps(
                            {
                                "stage": "measured",
                                "repeat": repeat,
                                "intraop_threads": count,
                                "cycle": cycle.to_dict(),
                                "diagnostics": diagnostics,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    stream.flush()
                print(
                    json.dumps(
                        {
                            "completed_round": repeat + 1,
                            "rounds": repeats,
                            "order": order,
                        }
                    ),
                    flush=True,
                )
        arms: dict[str, object] = {}
        points: list[ScalingPoint] = []
        all_passed = True
        for count in THREAD_COUNTS:
            cycles = tuple(measured[count])
            quality = compare_cycles((reference,) * repeats, cycles, gate)
            warmup_quality = compare_cycles((reference,), (warmup[count],), gate)
            passed = quality.passed and warmup_quality.passed
            all_passed &= passed
            seconds = math.fsum(cycle.latency_ms for cycle in cycles) / 1000
            attempted = len(cases) * repeats
            failed = sum(cycle.failed for cycle in cycles)
            rate = (attempted - failed) / seconds
            if passed:
                points.append(ScalingPoint(count, rate))
            baseline_cycles = measured[8]
            paired_speedups = [
                base.latency_ms / actual.latency_ms
                for base, actual in zip(baseline_cycles, cycles, strict=True)
            ]
            arms[str(count)] = {
                "passed": passed,
                "quality": asdict(quality),
                "warmup_quality": asdict(warmup_quality),
                "attempted": attempted,
                "failed": failed,
                "successful_requests_per_second": rate,
                "equivalent_requests_per_second": rate if passed else 0,
                "cycle_p50_ms": percentile([cycle.latency_ms for cycle in cycles], 0.5),
                "cycle_p95_ms": percentile(
                    [cycle.latency_ms for cycle in cycles], 0.95
                ),
                "batch_residence_p50_ms": percentile(
                    [row.latency_ms for cycle in cycles for row in cycle.rows], 0.5
                ),
                "batch_residence_p95_ms": percentile(
                    [row.latency_ms for cycle in cycles for row in cycle.rows], 0.95
                ),
                "same_round_speedups_vs8": paired_speedups,
                "same_round_speedup_p50_vs8": percentile(paired_speedups, 0.5),
                "measured_cycles": [cycle.to_dict() for cycle in cycles],
                "warmup_cycle": warmup[count].to_dict(),
                "parallel_info": parallel[count],
                "measured_diagnostics": measured_diagnostics[count],
                "warmup_diagnostics": warmup_diagnostics[count],
            }
        result: dict[str, object] = {
            "schema_version": 1,
            "scope": "configured ATen intraop-thread sweep; "
            "fixed full77 FP32 development cohort",
            "passed": all_passed,
            "model_id": scorer.model_id,
            "dataset_sha256": DATASET_SHA256,
            "reference_sha256": reference_sha256,
            "input_sha256": _input_digest(cases),
            "case_ids": [case.id for case in cases],
            "token_lengths": token_lengths,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": version("torch"),
                "transformers": version("transformers"),
                "torch_build": torch.__config__.show(),
                "mkldnn_available": cast(
                    Callable[[], bool], torch.backends.mkldnn.is_available
                )(),
                "mkldnn_enabled": torch.backends.mkldnn.enabled,
                "interop_threads": interop_threads,
                "original_intraop_threads": original_threads,
                "thread_environment": {
                    key: os.environ.get(key)
                    for key in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                        "OMP_PROC_BIND",
                    )
                },
                "device": "cpu",
                "dtype": "float32",
                "autocast": False,
                "compile": False,
                "token_cache_size": 0,
            },
            "protocol": {
                "intraop_counts": THREAD_COUNTS,
                "batch_size": 8,
                "repeats": repeats,
                "warmup_per_count": 1,
                "orders": orders,
                "quality_gate": asdict(gate),
                "cohort_order": "canonical unchanged caller order",
                "setting_owner": "isolated benchmark process; "
                "intraop mutation per arm, interop fixed",
                "timing": "complete synchronous cohort; tokens, packing, forward, "
                "host results and decisions",
                "excludes": [
                    "loading",
                    "warmup",
                    "thread-setting mutation",
                    "receipt serialization",
                    "HTTP",
                    "queue",
                    "before/after diagnostic collection",
                ],
                "diagnostics": "untimed own-process cumulative rusage, OS load average "
                "and wall/monotonic timestamps; counters do not establish cause",
            },
            "counts": arms,
            "usl_descriptive_intraop_fit": asdict(fit_usl(points))
            if len(points) == 6
            else None,
            "source_sha256": sources.hashes(),
            "source_archive": sources.archive(),
            "limits": [
                "Configured ATen counts are not guaranteed Accelerate worker counts "
                "or uniform physical processors.",
                "Same-round ratios are balanced sweep diagnostics; "
                "any winner needs independent paired confirmation.",
                "No held-out model-quality claim; "
                "all original rows and failures retained.",
                "No measured DRAM bandwidth or independent sustained-compute ceiling.",
            ],
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        torch.set_num_threads(original_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/torch-threads77.json"),
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be finite and positive")
    if args.worker:
        result = sweep(args.model, args.output, repeats=args.repeats)
        print(
            json.dumps({"output": str(args.output), "model_id": result["model_id"]}),
            flush=True,
        )
        if not result["passed"]:
            sys.exit(1)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    exit_code: int | None = None
    command = [
        sys.executable,
        "-m",
        "benchmarks.analysis.torch.threads",
        str(args.model),
        "--output",
        str(args.output),
        "--repeats",
        str(args.repeats),
        "--worker",
    ]
    try:
        exit_code = subprocess.run(
            command, timeout=args.timeout_seconds, check=False
        ).returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    receipt = {
        "elapsed_seconds": time.perf_counter() - start,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "completed": not timed_out and exit_code == 0,
    }
    args.output.with_suffix(".wall.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
    if not receipt["completed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
