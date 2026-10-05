"""CPU Inductor experiment with a 120-second subprocess wall budget.

Run: python -m benchmarks.real.torch.compile MODEL
Only the supervisor's newly created process group is terminated on timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[3]


class Arguments(argparse.Namespace):
    model: Path
    output: Path
    worker: bool


def write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def worker(args: Arguments) -> None:
    import torch
    from transformers import Qwen3Model

    from wyat.backends.kai.torch import KaiScorer
    from wyat.benchmark import (
        BenchmarkEngine,
        BenchmarkProtocol,
        QualityGate,
        benchmark_pair,
    )
    from wyat.core import Model
    from wyat.evaluation import case_from_dict
    from wyat.wire import decode_json

    protocol = BenchmarkProtocol(
        repeats=5,
        warmup=1,
        quality_gate=QualityGate(
            logit_atol=1e-3, logit_rtol=1e-3, probability_atol=1e-4
        ),
    )
    progress = args.output / "torch-compile-progress.json"
    write(progress, {"phase": "loading", "protocol": "frozen five paired repeats"})
    torch.set_num_threads(8)
    path = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = tuple(
        case_from_dict(decode_json(line)) for line in path.read_text().splitlines()
    )[:16]
    scorer = KaiScorer.from_directory(args.model, max_length=2048, token_cache_size=256)
    compiled = cast(Qwen3Model, torch.compile(scorer.backbone, backend="inductor"))
    candidate = KaiScorer(
        compiled,
        scorer.head,
        scorer.tokenizer,
        device=torch.device("cpu"),
        max_length=scorer.max_length,
        max_options=scorer.max_options,
        token_cache_size=256,
        model_id=scorer.model_id,
    )
    write(progress, {"phase": "JIT and first cohort", "model_id": scorer.model_id})
    cold_start = time.perf_counter()
    candidate_model = Model(candidate, model_id=candidate.model_id)
    for start in range(0, len(cases), 8):
        candidate_model.decide_many(
            tuple(case.request for case in cases[start : start + 8])
        )
    cold_seconds = time.perf_counter() - cold_start
    write(progress, {"phase": "paired measurement", "cold_seconds": cold_seconds})
    conditions = {"device": "cpu", "threads": "8", "dtype": "float32"}
    report = benchmark_pair(
        BenchmarkEngine(
            "CPU FP32 SDPA", Model(scorer, model_id=scorer.model_id), 8, conditions
        ),
        BenchmarkEngine("CPU FP32 Inductor backbone", candidate_model, 8, conditions),
        cases,
        protocol=protocol,
    )
    payload = report.to_dict()
    payload["cold_seconds"] = cold_seconds
    payload["scope"] = (
        "BANKING77 training-source development prefix16; Kai exposure unknown"
    )
    write(args.output / "torch-compile-16.json", payload)
    write(progress, {"phase": "complete", "passed": report.passed})


def supervise(args: Arguments) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix="wyat-inductor-", dir="/private/tmp"))
    environment = os.environ.copy()
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    start = time.perf_counter()
    timed_out = False
    with (args.output / "torch-compile.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.real.torch.compile",
                str(args.model),
                "--output",
                str(args.output),
                "--worker",
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        finally:
            # Compiler workers may outlive their leader even after normal exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    write(
        args.output / "torch-compile-wall.json",
        {
            "wall_seconds": time.perf_counter() - start,
            "budget_seconds": 120,
            "timed_out": timed_out,
            "exit_code": process.returncode,
            "cache_directory": str(cache),
            "interpretation": (
                "Compilation/timing unknown within practical wall budget"
                if timed_out or process.returncode != 0
                else "Inspect retained quality gate and paired measurements"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/real")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(namespace=Arguments())
    args.model = ROOT / args.model
    args.output = ROOT / args.output
    if args.worker:
        worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
