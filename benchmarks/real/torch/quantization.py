"""Bounded real-data QNNPACK experiment; quantization is never a runtime default.

Run: python -m benchmarks.real.torch.quantization PATH_TO_KAI
The strict gate and development prefix are frozen before model predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/real"))
    args = parser.parse_args()
    protocol = BenchmarkProtocol(
        repeats=3,
        warmup=1,
        quality_gate=QualityGate(
            logit_atol=1e-3, logit_rtol=1e-3, probability_atol=1e-4
        ),
    )
    path = Path("benchmarks/real/banking77-train-dev-menu6.jsonl")
    cases = tuple(
        case_from_dict(decode_json(line)) for line in path.read_text().splitlines()
    )
    # This executable owns the process-wide resource and kernel settings.
    torch.set_num_threads(8)
    torch.backends.quantized.engine = "qnnpack"
    baseline = KaiScorer.from_directory(
        args.model, max_length=2048, token_cache_size=256
    )
    quantized = cast(
        Qwen3Model,
        # PyTorch's deprecated conversion API has no type annotations.
        quantize_dynamic(  # type: ignore[no-untyped-call]
            baseline.backbone, {nn.Linear}, dtype=torch.qint8, inplace=False
        ),
    )
    candidate = KaiScorer(
        quantized,
        baseline.head,
        baseline.tokenizer,
        device=torch.device("cpu"),
        max_length=baseline.max_length,
        max_options=baseline.max_options,
        token_cache_size=256,
        model_id=baseline.model_id,
    )
    engines = (
        BenchmarkEngine(
            "FP32 CPU SDPA",
            Model(baseline, model_id=baseline.model_id),
            batch_size=8,
            conditions={"device": "cpu", "torch_threads": "8", "dtype": "FP32"},
        ),
        BenchmarkEngine(
            "QNNPACK dynamic INT8 backbone Linear; FP32 embeddings/norms/head",
            Model(candidate, model_id=candidate.model_id),
            batch_size=8,
            conditions={
                "device": "cpu",
                "torch_threads": "8",
                "quantization": "torch.ao dynamic Linear qint8 qnnpack",
            },
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for size in (16, 77):
        report = benchmark_pair(*engines, cases[:size], protocol=protocol)
        payload = report.to_dict()
        payload["scope"] = (
            "BANKING77 training-source development; Kai exposure unknown; "
            "strict engine parity experiment, not generalization evidence"
        )
        payload["torch_version"] = torch.__version__
        (args.output / f"torch-qnnpack-{size}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {"cases": size, "passed": report.passed, "quality": payload["quality"]}
            ),
            flush=True,
        )
        if not report.passed:
            break


if __name__ == "__main__":
    main()
