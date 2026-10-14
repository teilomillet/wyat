"""Observe real FP32 CPU SDPA dispatch; two controls and two shape profiles.

python -m benchmarks.analysis.torch.profile MODEL --output RECEIPT.json
Only CPU profiler activity is enabled; no model or attention policy is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from torch.autograd.profiler_util import FunctionEventAvg
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from wyat import Model
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict

from ..evidence import freeze_sources, process_diagnostic
from ..reference import read_reference

ROOT = Path(__file__).resolve().parents[3]
DATASET_SHA256 = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"


class Arguments(argparse.Namespace):
    model: Path
    output: Path
    worker: bool


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(model_path: Path, output: Path) -> dict[str, object]:
    from wyat.wire import decode_json

    dataset = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = ROOT / "benchmarks/real/torch-fp32-reference77.json"
    gate = QualityGate(logit_atol=1e-3, logit_rtol=1e-3, probability_atol=1e-4)
    sources = freeze_sources(
        {
            "runner": Path(__file__),
            "reference": ROOT / "benchmarks/analysis/reference.py",
            "evidence": ROOT / "benchmarks/analysis/evidence.py",
            "backend": ROOT / "wyat/backends/wyat/torch.py",
            "encoding": ROOT / "wyat/backends/wyat/encoding.py",
            "artifact": ROOT / "wyat/backends/wyat/artifact.py",
            "core": ROOT / "wyat/core.py",
            "benchmark": ROOT / "wyat/benchmark.py",
            "sdpa_integration": Path(sdpa_attention.__file__),
            "qwen3": Path(modeling_qwen3.__file__),
            "cpu_sdpa_guards": Path(torch.__file__).parent
            / "include/ATen/native/transformers/sdp_utils_cpp.h",
        },
        output,
    )
    reference_hash = digest(reference_path)
    if digest(dataset) != DATASET_SHA256:
        raise ValueError("Frozen development dataset changed")
    all_cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    if len(all_cases) != 77:
        raise ValueError("Expected the fixed 77-case development cohort")
    cases = all_cases[:8]
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        scorer = WyatScorer.from_directory(
            model_path, device="cpu", max_length=512, token_cache_size=0
        )
        saved = read_reference(
            reference_path, all_cases, scorer, dataset_sha256=DATASET_SHA256
        )
        reference = BenchmarkCycle(0.0, saved.rows[:8], 0, 0)
        engine = BenchmarkEngine(
            "CPU FP32 SDPA profile", Model(scorer, scorer.model_id), 8
        )
        cycles: list[dict[str, object]] = []
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.with_suffix(".cycles.jsonl").open("w") as stream:
            for stage in ("warmup", "ordinary", "ordinary", "profile", "profile"):
                before = process_diagnostic()
                events: list[dict[str, object]] = []
                outer_start = time.perf_counter_ns()
                if stage == "profile":
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=True,
                        with_stack=False,
                        profile_memory=False,
                    ) as profiler:
                        cycle = _run(engine, cases)
                    outer_ms = (time.perf_counter_ns() - outer_start) / 1e6
                    averages = cast(
                        Sequence[FunctionEventAvg],
                        profiler.key_averages(group_by_input_shape=True),
                    )
                    events = [
                        {
                            "operator": event.key,
                            "count": event.count,
                            "input_shapes": event.input_shapes,
                            "self_cpu_time_us": float(event.self_cpu_time_total),
                        }
                        for event in averages
                    ]
                else:
                    cycle = _run(engine, cases)
                    outer_ms = (time.perf_counter_ns() - outer_start) / 1e6
                after = process_diagnostic()
                quality = compare_cycles((reference,), (cycle,), gate)
                row: dict[str, object] = {
                    "stage": stage,
                    "cycle": cycle.to_dict(),
                    "quality": asdict(quality),
                    "passed": quality.passed,
                    "outer_wall_ms": outer_ms,
                    "before": asdict(before),
                    "after": asdict(after),
                    "operator_self_times": events,
                    "self_cpu_sum_us": math.fsum(
                        cast(float, event["self_cpu_time_us"]) for event in events
                    ),
                }
                cycles.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        ordinary_ms = [
            cast(float, cast(dict[str, object], row["cycle"])["latency_ms"])
            for row in cycles
            if row["stage"] == "ordinary"
        ]
        profile_ms = [
            cast(float, cast(dict[str, object], row["cycle"])["latency_ms"])
            for row in cycles
            if row["stage"] == "profile"
        ]
        result: dict[str, object] = {
            "schema_version": 1,
            "scope": "first eight fixed BANKING77 cases; CPU dispatch diagnostic",
            "passed": all(row["passed"] is True for row in cycles),
            "model_id": scorer.model_id,
            "dataset_sha256": DATASET_SHA256,
            "reference_sha256": reference_hash,
            "input_sha256": _input_digest(cases),
            "case_ids": [case.id for case in cases],
            "token_lengths": [
                len(scorer.encode_request(case.request).ids) for case in cases
            ],
            "versions": {
                "torch": version("torch"),
                "transformers": version("transformers"),
            },
            "torch_git_version": torch.version.git_version,
            "source_sha256": sources.hashes(),
            "source_archive": sources.archive(),
            "protocol": {
                "dtype": "float32",
                "batch_size": 8,
                "intraop_threads": 8,
                "interop_threads": torch.get_num_interop_threads(),
                "max_length": 512,
                "token_cache_size": 0,
                "warmup": 1,
                "ordinary": 2,
                "profile": 2,
                "profiler": "CPU activity, shapes; no stacks or memory",
                "quality_gate": asdict(gate),
                "times": "operator SELF time only; no inclusive-time summation",
            },
            "sdpa_flags": {
                key: vars(sdpa_attention)[key]
                for key in (
                    "_is_torch_mps_available",
                    "_is_torch_greater_or_equal_than_2_13",
                )
            },
            "ordinary_median_ms": statistics.median(ordinary_ms),
            "profiled_call_median_ms": statistics.median(profile_ms),
            "instrumentation_ratio": statistics.median(profile_ms)
            / statistics.median(ordinary_ms),
            "cycles": cycles,
            "limits": [
                "Profiles alter execution and retain shapes; fractions are diagnostic.",
                "ATen self time does not expose Accelerate topology or DRAM traffic.",
                "Source eligibility is not observed dispatch; inspect events.",
                "No production change, alternate attention or unseen quality claim.",
            ],
        }
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result
    finally:
        torch.set_num_threads(prior_threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmarks/analysis/results/torch-profile8.json",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(namespace=Arguments())
    args.model, args.output = ROOT / args.model, ROOT / args.output
    if args.worker:
        result = inspect(args.model, args.output)
        if not result["passed"]:
            sys.exit(1)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    with args.output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.analysis.torch.profile",
                str(args.model),
                "--output",
                str(args.output),
                "--worker",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    args.output.with_suffix(".wall.json").write_text(
        json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "budget_seconds": 180,
                "timed_out": timed_out,
                "exit_code": process.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    if timed_out or process.returncode != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
