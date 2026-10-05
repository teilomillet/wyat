"""Bounded full-checkpoint MLX measurement; setup stays outside timed samples."""

from __future__ import annotations

import hashlib
import json
import math
import time
from importlib.metadata import version
from pathlib import Path
from typing import cast

from wyat import Model
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.benchmark import BenchmarkEngine, BenchmarkProtocol, benchmark_pair
from wyat.core import decision_from_logits
from wyat.evaluation import case_from_dict
from wyat.wire import decode_json

root = Path(__file__).resolve().parents[3]
dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
cases = tuple(
    case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
)
reference = decode_json(
    (root / "benchmarks/real/torch-fp32-reference77.json").read_text()
)
assert isinstance(reference, dict)
reference_rows = reference["rows"]
assert isinstance(reference_rows, list)
assert hashlib.sha256(dataset.read_bytes()).hexdigest() == reference["dataset_sha256"]
start = time.perf_counter()
scorer = MLXKaiScorer.from_directory(
    root / ".models/Decision-2.0-Kai-0.6B",
    max_length=512,
    device="gpu",
    token_cache_size=256,
)
load_seconds = time.perf_counter() - start
assert scorer.model_id == reference["model_id"]
requests = tuple(case.request for case in cases)
logits: list[list[float]] = []
start = time.perf_counter()
for offset in range(0, len(requests), 8):
    logits.extend(scorer.score_many(requests[offset : offset + 8]))
first_scoring_seconds = time.perf_counter() - start
rows: list[dict[str, object]] = []
max_logit_error = 0.0
max_probability_error = 0.0
top1_equal = 0
ranking_equal = 0
correct = 0
all_logits_within_gate = True
for case, ref, actual in zip(cases, reference_rows, logits, strict=True):
    assert isinstance(ref, dict)
    encoded = scorer.encode_request(case.request)
    assert list(encoded.ids) == ref["token_ids"]
    assert list(encoded.candidate_positions) == ref["candidate_positions"]
    assert encoded.query_position == ref["query_position"]
    assert case.id == ref["id"]
    saved = ref["legacy_logits"]
    assert isinstance(saved, list)
    expected = tuple(float(cast(float, value)) for value in saved)
    baseline = decision_from_logits(case.request, expected, scorer.model_id)
    native = decision_from_logits(case.request, actual, scorer.model_id)
    max_logit_error = max(
        max_logit_error,
        *(abs(left - right) for left, right in zip(actual, expected, strict=True)),
    )
    max_probability_error = max(
        max_probability_error,
        *(
            abs(left - right)
            for left, right in zip(
                native.probabilities, baseline.probabilities, strict=True
            )
        ),
    )
    all_logits_within_gate &= all(
        math.isclose(left, right, abs_tol=1e-3, rel_tol=1e-3)
        for left, right in zip(actual, expected, strict=True)
    )
    top1_equal += native.selected == baseline.selected
    ranking_equal += native.ranking == baseline.ranking
    correct += native.selected in case.expected
    rows.append(
        {
            "id": case.id,
            "logits": actual,
            "ranking": native.ranking,
            "selected": native.selected,
            "correct": native.selected in case.expected,
        }
    )
parity: dict[str, object] = {
    "schema_version": 1,
    "model_id": scorer.model_id,
    "dataset_sha256": reference["dataset_sha256"],
    "scope": (
        "real BANKING77 training-source development77; "
        "Kai exposure unknown; runtime parity only"
    ),
    "mlx": version("mlx"),
    "mlx_lm": version("mlx-lm"),
    "backbone_dtype": "float32",
    "head_dtype": "float32",
    "device": "Apple M4 Pro Metal GPU",
    "batch_size": 8,
    "load_seconds": load_seconds,
    "first_scoring_seconds": first_scoring_seconds,
    "token_ids_and_endpoints_match": True,
    "max_absolute_logit_error": max_logit_error,
    "max_absolute_probability_error": max_probability_error,
    "top1_equal": top1_equal,
    "all_ranking_equal": ranking_equal,
    "correct": correct,
    "cases": len(cases),
    "logit_gate": {"atol": 1e-3, "rtol": 1e-3},
    "probability_atol": 1e-4,
    "passed": all_logits_within_gate
    and max_probability_error <= 1e-4
    and ranking_equal == len(cases),
    "rows": rows,
}
(root / "benchmarks/real/mlx-fp32-parity77.json").write_text(
    json.dumps(parity, indent=2) + "\n"
)
print(
    json.dumps({key: value for key, value in parity.items() if key != "rows"}),
    flush=True,
)
compiled = MLXKaiScorer(
    scorer.backbone,
    scorer.head,
    scorer.encoder,
    model_id=scorer.model_id,
    device=scorer.device,
    compile=True,
)
report = benchmark_pair(
    BenchmarkEngine("mlx-fp32-eager-batch8", Model(scorer, scorer.model_id), 8),
    BenchmarkEngine("mlx-fp32-compiled-batch8", Model(compiled, compiled.model_id), 8),
    cases,
    protocol=BenchmarkProtocol(repeats=5, warmup=1),
)
payload = report.to_dict()
(root / "benchmarks/real/mlx-fp32-compile77.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(
    json.dumps({"compile_pair_passed": report.passed, "speed": payload["speed"]}),
    flush=True,
)
