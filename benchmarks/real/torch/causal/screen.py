"""Frozen real-data CPU screen preserving the pre-optimization masked baseline.

Run from the checkout: python -m benchmarks.real.torch.causal.screen MODEL
Reports default to benchmarks/real/reproduced to preserve historical receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from wyat.backends.kai.torch import KaiScorer
from wyat.core import Request
from wyat.evaluation import case_from_dict
from wyat.wire import decode_json

ROOT = Path(__file__).resolve().parents[4]


class Arguments(argparse.Namespace):
    model: Path
    output: Path


def score(
    scorer: KaiScorer, requests: Sequence[Request], *, legacy: bool
) -> list[list[float]]:
    if scorer.device.type != "cpu":
        raise ValueError("This frozen reference supports CPU only")
    if not requests:
        return []
    encoded = [scorer.encode_request(value) for value in requests]
    length = max(len(row.ids) for row in encoded)
    if legacy:
        length = -(-length // 8) * 8
    ids = torch.full((len(encoded), length), scorer.pad_id, dtype=torch.long)
    width = max(len(row.candidate_positions) for row in encoded)
    endpoints = torch.zeros((len(encoded), width), dtype=torch.long)
    mask = torch.zeros_like(ids) if legacy else None
    for index, row in enumerate(encoded):
        ids[index, : len(row.ids)] = torch.tensor(row.ids)
        endpoints[index, : len(row.candidate_positions)] = torch.tensor(
            row.candidate_positions
        )
        if mask is not None:
            mask[index, : len(row.ids)] = 1
    rows = torch.arange(len(encoded))
    query_positions = torch.tensor([row.query_position for row in encoded])
    with torch.inference_mode():
        output = cast(
            BaseModelOutputWithPast,
            scorer.backbone(input_ids=ids, attention_mask=mask, use_cache=False),
        )
        hidden = output.last_hidden_state
        if hidden is None:
            raise ValueError("Kai backbone did not return hidden states")
        logits = cast(
            torch.Tensor,
            scorer.head(
                hidden[rows[:, None], endpoints], hidden[rows, query_positions]
            ),
        )
    values = cast(list[list[float]], logits.float().cpu().tolist())
    return [
        row[: len(item.candidate_positions)]
        for row, item in zip(values, encoded, strict=True)
    ]


def run(
    scorer: KaiScorer, requests: Sequence[Request], batch_size: int, *, legacy: bool
) -> list[list[float]]:
    return [
        row
        for start in range(0, len(requests), batch_size)
        for row in score(scorer, requests[start : start + batch_size], legacy=legacy)
    ]


def main() -> None:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/real/reproduced"
    )
    args = parser.parse_args(namespace=Arguments())
    args.output = ROOT / args.output
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / args.model
    data_path = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = [
        case_from_dict(decode_json(line)) for line in data_path.read_text().splitlines()
    ][:16]
    requests = [case.request for case in cases]
    scorer = KaiScorer.from_directory(model_path, max_length=2048, token_cache_size=256)
    report: dict[str, object] = {
        "schema_version": 1,
        "model_id": scorer.model_id,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "case_ids": [case.id for case in cases],
        "scope": (
            "real BANKING77 training-source development prefix16; "
            "trained Kai exposure unknown; fixed six-candidate menus; screening only"
        ),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "backbone_dtype": "float32",
        "head_dtype": "float32",
        "warmup": 1,
        "repeats": 3,
        "order": "alternating baseline/candidate",
        "includes": [
            "tokenization",
            "tensor packing",
            "backbone",
            "head",
            "CPU score materialization",
        ],
        "excludes": ["model load", "warmup", "decision interpretation", "HTTP"],
        "baseline": (
            "original PyTorch SDPA; right-pad to multiple of8; explicit pad mask"
        ),
        "candidate": (
            "same weights PyTorch SDPA; exact max width; causal-only attention"
        ),
        "results": [],
    }
    results: list[dict[str, object]] = []
    for threads in [4, 1, 8]:
        torch.set_num_threads(threads)
        for batch_size in [1, 4, 8]:
            baseline = run(scorer, requests, batch_size, legacy=True)
            candidate = run(scorer, requests, batch_size, legacy=False)
            delta = max(
                abs(a - b)
                for x, y in zip(baseline, candidate, strict=True)
                for a, b in zip(x, y, strict=True)
            )
            agree = all(
                max(range(len(x)), key=x.__getitem__)
                == max(range(len(y)), key=y.__getitem__)
                for x, y in zip(baseline, candidate, strict=True)
            )
            torch.testing.assert_close(
                torch.tensor(baseline), torch.tensor(candidate), atol=2e-5, rtol=2e-5
            )
            rank_equal = all(
                sorted(range(len(x)), key=x.__getitem__)
                == sorted(range(len(y)), key=y.__getitem__)
                for x, y in zip(baseline, candidate, strict=True)
            )
            if not rank_equal:
                raise AssertionError("Candidate score ranking changed")
            if not agree:
                raise AssertionError("Selected candidate changed")
            timings: dict[str, list[float]] = {"baseline": [], "candidate": []}
            for iteration in range(3):
                order = [True, False] if iteration % 2 == 0 else [False, True]
                for legacy in order:
                    start = time.perf_counter_ns()
                    run(scorer, requests, batch_size, legacy=legacy)
                    timings["baseline" if legacy else "candidate"].append(
                        (time.perf_counter_ns() - start) / 1e6
                    )
            row: dict[str, object] = {
                "threads": threads,
                "batch_size": batch_size,
                "baseline_ms": timings["baseline"],
                "candidate_ms": timings["candidate"],
                "median_speedup": statistics.median(timings["baseline"])
                / statistics.median(timings["candidate"]),
                "max_abs_logit_difference": delta,
                "selected_equal": agree,
                "rank_equal": rank_equal,
                "candidate_logits": candidate,
                "baseline_logits": baseline,
                "baseline_correct": sum(
                    case.request.candidates[
                        max(range(len(logits)), key=logits.__getitem__)
                    ].id
                    in case.expected
                    for case, logits in zip(cases, baseline, strict=True)
                ),
                "candidate_correct": sum(
                    case.request.candidates[
                        max(range(len(logits)), key=logits.__getitem__)
                    ].id
                    in case.expected
                    for case, logits in zip(cases, candidate, strict=True)
                ),
                "smallest_baseline_top_two_margin": min(
                    sorted(logits, reverse=True)[0] - sorted(logits, reverse=True)[1]
                    for logits in baseline
                ),
            }
            results.append(row)
            report["results"] = results
            (args.output / "torch-causal-screen.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"candidate_logits", "baseline_logits"}
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
