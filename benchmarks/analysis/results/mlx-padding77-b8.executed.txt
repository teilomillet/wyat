"""Test exact widths and length grouping without changing production batching."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx

from benchmarks.analysis.mlx.profile import quality
from wyat import Model, Request
from wyat.backends.kai.encoding import EncodedRequest
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.benchmark import BenchmarkEngine, BenchmarkProtocol, benchmark_pair
from wyat.evaluation import case_from_dict
from wyat.wire import decode_json


class ExactWidth(MLXKaiScorer):
    """Experimental packing change; causal positions and forward stay identical."""

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        encoded = [self.encode_request(request) for request in requests]
        length = max(len(row.ids) for row in encoded)
        width = max(len(row.candidate_positions) for row in encoded)
        with mx.stream(self.device):
            ids = mx.array(
                [
                    list(row.ids) + [self.encoder.pad_id] * (length - len(row.ids))
                    for row in encoded
                ],
                dtype=mx.int32,
            )
            endpoints = mx.array(
                [
                    list(row.candidate_positions)
                    + [0] * (width - len(row.candidate_positions))
                    for row in encoded
                ],
                dtype=mx.int32,
            )
            positions = mx.array(
                [row.query_position for row in encoded], dtype=mx.int32
            )
            logits = self._forward(ids, endpoints, positions)
            mx.eval(logits)
            values = cast(list[list[float]], logits.tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]


class EncodedScorer(Protocol):
    def encode_request(self, request: Request) -> EncodedRequest: ...

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]: ...


class Microbatches:
    """Ready-cohort scheduler; sorting cost is included and output order restored.

    Sorting encodes requests to obtain actual lengths, then the unchanged scorer
    encodes each microbatch. This intentionally charges the candidate for that
    extra work. There is no online admission queue or latency claim.
    """

    def __init__(self, scorer: EncodedScorer, batch_size: int, *, sort: bool) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch size must be a positive integer")
        self.scorer = scorer
        self.batch_size = batch_size
        self.sort = sort

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        order = list(range(len(requests)))
        if self.sort:
            lengths = [len(self.scorer.encode_request(row).ids) for row in requests]
            order.sort(key=lambda index: lengths[index])
        result: list[list[float]] = [[] for _ in requests]
        for offset in range(0, len(order), self.batch_size):
            indices = order[offset : offset + self.batch_size]
            values = self.scorer.score_many([requests[index] for index in indices])
            for index, row in zip(indices, values, strict=True):
                result[index] = row
        return result


class Arguments(Namespace):
    batch_size: int
    output: Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks/analysis/results/mlx-padding77-b8.json",
    )
    options = parser.parse_args(namespace=Arguments())
    options.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = tuple(
        case_from_dict(decode_json(row)) for row in dataset.read_text().splitlines()
    )
    reference = decode_json(
        (root / "benchmarks/real/torch-fp32-reference77.json").read_text()
    )
    assert isinstance(reference, dict)
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert dataset_hash == reference["dataset_sha256"]
    reference_rows = reference["rows"]
    assert isinstance(reference_rows, list)
    scorer = MLXKaiScorer.from_directory(
        root / ".models/Decision-2.0-Kai-0.6B",
        max_length=512,
        token_cache_size=0,
        device="gpu",
    )
    assert scorer.model_id == reference["model_id"]
    saved: list[list[float]] = []
    for case, row in zip(cases, reference_rows, strict=True):
        assert isinstance(row, dict)
        assert case.id == row["id"]
        encoded = scorer.encode_request(case.request)
        assert list(encoded.ids) == row["token_ids"]
        assert list(encoded.candidate_positions) == row["candidate_positions"]
        assert encoded.query_position == row["query_position"]
        logits = row["legacy_logits"]
        assert isinstance(logits, list)
        saved.append([float(cast(float, value)) for value in logits])
    exact = ExactWidth(
        scorer.backbone,
        scorer.head,
        scorer.encoder,
        model_id=scorer.model_id,
        device=scorer.device,
    )
    baseline = Microbatches(scorer, options.batch_size, sort=False)
    requests = tuple(case.request for case in cases)
    baseline_gate = quality(
        cases, saved, baseline.score_many(requests), scorer.model_id
    )
    assert baseline_gate.passed
    reports: dict[str, object] = {}
    for name, candidate in (
        ("exact", Microbatches(exact, options.batch_size, sort=False)),
        ("sorted", Microbatches(scorer, options.batch_size, sort=True)),
        ("sorted-exact", Microbatches(exact, options.batch_size, sort=True)),
    ):
        try:
            gate = quality(
                cases, saved, candidate.score_many(requests), scorer.model_id
            )
        except Exception as error:
            reports[name] = {
                "screen_error": f"{type(error).__name__}: {error}",
                "timed": False,
            }
            continue
        if not gate.passed:
            reports[name] = {"screen": asdict(gate), "timed": False}
            continue
        report = benchmark_pair(
            BenchmarkEngine(
                "caller-rounded8", Model(baseline, scorer.model_id), len(cases)
            ),
            BenchmarkEngine(name, Model(candidate, scorer.model_id), len(cases)),
            cases,
            protocol=BenchmarkProtocol(repeats=5, warmup=1),
        )
        reports[name] = report.to_dict()
        print(
            json.dumps(
                {
                    "arm": name,
                    "passed": report.passed,
                    "speed": report.to_dict()["speed"],
                }
            ),
            flush=True,
        )
    payload = {
        "scope": "77-case training-source runtime assay; ready-cohort scheduling only",
        "dataset_sha256": dataset_hash,
        "model_id": scorer.model_id,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "backend_sha256": hashlib.sha256(
            (root / "wyat/backends/kai/mlx.py").read_bytes()
        ).hexdigest(),
        "reference_sha256": hashlib.sha256(
            (root / "benchmarks/real/torch-fp32-reference77.json").read_bytes()
        ).hexdigest(),
        "microbatch_size": options.batch_size,
        "backbone_dtype": "float32",
        "token_cache_size": 0,
        "residence_scope": (
            "Each outer call contains the full77 ready cohort; row residence is "
            "cohort-call residence, not internal microbatch or online queue latency"
        ),
        "reports": reports,
    }
    options.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
