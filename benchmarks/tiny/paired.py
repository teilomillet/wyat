"""Offline paired benchmarks; synthetic fixture results never establish quality.

Run from the checkout: uv run --all-extras python -m benchmarks.tiny.paired
The uncached baseline uses the same native backend with option caching disabled.
Loading and warmup are excluded; suffix pretokenization is shared by both paths.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import NotRequired, TypedDict

import torch
import transformers

from benchmarks.fixtures.kai import create_kai_bundle
from wyat.backends.kai.torch import KaiScorer
from wyat.core import Candidate, Model, Request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class Timing(TypedDict):
    baseline_cycle_p50_ms: float
    baseline_cycle_p95_ms: float
    candidate_cycle_p50_ms: float
    candidate_cycle_p95_ms: float
    median_speedup: float
    baseline_samples_ms: list[float]
    candidate_samples_ms: list[float]
    baseline_requests_per_second: NotRequired[float]
    candidate_requests_per_second: NotRequired[float]


def measure_pair(
    left: Callable[[], object],
    right: Callable[[], object],
    *,
    repeats: int,
    warmup: int,
) -> Timing:
    for _ in range(warmup):
        left()
        right()
    left_ms: list[float] = []
    right_ms: list[float] = []
    for index in range(repeats):
        # Alternate order to reduce a systematic warm-state/order advantage.
        operations = [(left, left_ms), (right, right_ms)]
        if index % 2:
            operations.reverse()
        for operation, timings in operations:
            start = time.perf_counter_ns()
            operation()
            timings.append((time.perf_counter_ns() - start) / 1_000_000)
    left_median, right_median = statistics.median(left_ms), statistics.median(right_ms)
    return {
        "baseline_cycle_p50_ms": left_median,
        "baseline_cycle_p95_ms": percentile(left_ms, 0.95),
        "candidate_cycle_p50_ms": right_median,
        "candidate_cycle_p95_ms": percentile(right_ms, 0.95),
        "median_speedup": left_median / right_median,
        "baseline_samples_ms": left_ms,
        "candidate_samples_ms": right_ms,
    }


def benchmark(
    *, repeats: int, warmup: int, requests_per_cycle: int
) -> dict[str, object]:
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="wyat-benchmark-") as temporary:
        bundle = Path(temporary)
        create_kai_bundle(bundle)
        uncached = KaiScorer.from_directory(bundle, max_length=256, token_cache_size=0)
        cached = KaiScorer.from_directory(bundle, max_length=256, token_cache_size=32)
        uncached_model = Model(uncached, model_id=uncached.model_id)
        cached_model = Model(cached, model_id=cached.model_id)
        requests = tuple(
            Request(
                query=f"Choose action {index}",
                candidates=(
                    Candidate("a", "first action"),
                    Candidate("b", "second action"),
                ),
                context="Benchmark context",
            )
            for index in range(requests_per_cycle)
        )
        uncached_encoding = [uncached.encode_request(request) for request in requests]
        cached_encoding = [cached.encode_request(request) for request in requests]
        if uncached_encoding != cached_encoding:
            raise AssertionError("Cache changed token IDs or endpoint positions")
        singleton = torch.tensor(
            [uncached.score_many((request,))[0] for request in requests]
        )
        cache_singleton = torch.tensor(
            [cached.score_many((request,))[0] for request in requests]
        )
        batch = torch.tensor(cached.score_many(requests))
        torch.testing.assert_close(cache_singleton, singleton, rtol=0, atol=0)
        torch.testing.assert_close(batch, singleton, rtol=1e-5, atol=1e-5)
        if not torch.equal(batch.argmax(-1), singleton.argmax(-1)):
            raise AssertionError("Batching changed selected candidate")
        tokenization = measure_pair(
            lambda: [uncached.encode_request(request) for request in requests],
            lambda: [cached.encode_request(request) for request in requests],
            repeats=repeats,
            warmup=warmup,
        )
        end_to_end_cache = measure_pair(
            lambda: [uncached_model.decide(request) for request in requests],
            lambda: [cached_model.decide(request) for request in requests],
            repeats=repeats,
            warmup=warmup,
        )
        end_to_end_batch = measure_pair(
            lambda: [cached_model.decide(request) for request in requests],
            lambda: cached_model.decide_many(requests),
            repeats=repeats,
            warmup=warmup,
        )
        for timing in (tokenization, end_to_end_cache, end_to_end_batch):
            timing["baseline_requests_per_second"] = (
                requests_per_cycle * 1000 / timing["baseline_cycle_p50_ms"]
            )
            timing["candidate_requests_per_second"] = (
                requests_per_cycle * 1000 / timing["candidate_cycle_p50_ms"]
            )
        return {
            "schema_version": 1,
            "scope": "synthetic offline Qwen3 fixture; no trained-model quality claim",
            "baseline": "native Kai; FP32 CPU, SDPA; option cache disabled",
            "candidate": "warm option-token cache; batching measured separately",
            "model_id": uncached.model_id,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": "cpu",
                "dtype": "float32",
                "torch_threads": torch.get_num_threads(),
            },
            "protocol": {
                "seed": 7,
                "repeats": repeats,
                "warmup": warmup,
                "requests_per_cycle": requests_per_cycle,
                "timing_unit": "milliseconds for complete cycle",
                "paired_order": "alternating",
                "includes": [
                    "tokenization",
                    "backbone",
                    "head",
                    "decision interpretation",
                ],
                "excludes": ["load", "warmup", "HTTP", "JSON serialization"],
            },
            "equivalence": {
                "cache_tokens_and_positions_exact": True,
                "cache_logits_exact": True,
                "batch_max_abs_logit_difference": float(
                    (batch - singleton).abs().max()
                ),
                "batch_tolerance": {"rtol": 1e-5, "atol": 1e-5},
                "batch_selected_equal": True,
            },
            "tokenization_cache": tokenization,
            "end_to_end_cache": end_to_end_cache,
            "end_to_end_batch": end_to_end_batch,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--requests-per-cycle", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1 or args.requests_per_cycle < 2:
        parser.error("repeats >=3, warmup >=1 and requests-per-cycle >=2 required")
    report = benchmark(
        repeats=args.repeats,
        warmup=args.warmup,
        requests_per_cycle=args.requests_per_cycle,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
