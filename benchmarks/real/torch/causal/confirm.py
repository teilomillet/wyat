"""Confirm CPU causal parity against the preserved legacy masked reference.

Run: python -m benchmarks.real.torch.causal.confirm MODEL
Reports default to benchmarks/real/reproduced to preserve historical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.real.torch.causal.screen import ROOT, Arguments, score
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import (
    BenchmarkEngine,
    BenchmarkProtocol,
    QualityGate,
    benchmark_pair,
)
from wyat.core import Model, Request
from wyat.evaluation import case_from_dict
from wyat.wire import decode_json


@dataclass(frozen=True, slots=True)
class Arm:
    scorer: KaiScorer
    legacy: bool

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        return score(self.scorer, requests, legacy=self.legacy)


def main() -> None:
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/real/reproduced"
    )
    args = parser.parse_args(namespace=Arguments())
    args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = tuple(
        case_from_dict(decode_json(line)) for line in path.read_text().splitlines()
    )
    scorer = KaiScorer.from_directory(
        ROOT / args.model, max_length=2048, token_cache_size=256
    )
    baseline = Model(Arm(scorer, True), model_id=scorer.model_id)
    candidate = Model(Arm(scorer, False), model_id=scorer.model_id)
    protocol = BenchmarkProtocol(
        repeats=7,
        warmup=1,
        quality_gate=QualityGate(
            logit_atol=2e-5, logit_rtol=2e-5, probability_atol=1e-5
        ),
    )
    for batch in [1, 8]:
        report = benchmark_pair(
            BenchmarkEngine(
                "legacy rounded explicit pad mask",
                baseline,
                batch,
                {
                    "device": "cpu",
                    "threads": "8",
                    "dtype": "float32",
                    "attention": "sdpa",
                },
            ),
            BenchmarkEngine(
                "exact width causal-only",
                candidate,
                batch,
                {
                    "device": "cpu",
                    "threads": "8",
                    "dtype": "float32",
                    "attention": "sdpa",
                },
            ),
            cases[:16],
            protocol=protocol,
        )
        payload = report.to_dict()
        (args.output / f"torch-causal-confirm-batch{batch}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "batch": batch,
                    "passed": report.passed,
                    "quality": payload["quality"],
                    "speed": payload["speed"],
                },
                default=str,
            ),
            flush=True,
        )
        if not report.passed:
            raise AssertionError("Quality gate failed")
    rows: list[dict[str, object]] = []
    for start in range(0, len(cases), 8):
        batchcases = cases[start : start + 8]
        values = [case.request for case in batchcases]
        old = score(scorer, values, legacy=True)
        new = score(scorer, values, legacy=False)
        for case, legacy, causal in zip(batchcases, old, new, strict=True):
            encoded = scorer.encode_request(case.request)
            rows.append(
                {
                    "id": case.id,
                    "candidate_ids": [value.id for value in case.request.candidates],
                    "token_ids": list(encoded.ids),
                    "candidate_positions": list(encoded.candidate_positions),
                    "query_position": encoded.query_position,
                    "legacy_logits": legacy,
                    "causal_logits": causal,
                    "expected": list(case.expected),
                }
            )
    reference: dict[str, object] = {
        "model_id": scorer.model_id,
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": (
            "real BANKING77 training-source development 77cases; "
            "Kai exposure unknown; fullFP32CPUreference"
        ),
        "device": "cpu",
        "torch_threads": 8,
        "backbone_dtype": "float32",
        "head_dtype": "float32",
        "attention": "sdpa",
        "batch_size": 8,
        "rows": rows,
    }
    (args.output / "torch-fp32-reference77.json").write_text(
        json.dumps(reference, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps({"full_reference_cases": len(rows), "model_id": scorer.model_id}),
        flush=True,
    )


if __name__ == "__main__":
    main()
