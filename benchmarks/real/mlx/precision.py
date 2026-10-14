"""Test strict quality gates before measuring reduced-precision speed."""

from __future__ import annotations

import hashlib
import json
import time
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx_lm.models.qwen3 import Qwen3Model

from wyat import Model
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    QualityGate,
    benchmark_pair,
    compare_cycles,
)
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationRow, case_from_dict
from wyat.wire import decode_json

parser = ArgumentParser(description=__doc__)
parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
dtype_name = str(parser.parse_args().dtype)
storage_dtype = mx.float16 if dtype_name == "float16" else mx.bfloat16
precision = "fp16" if dtype_name == "float16" else "bf16"

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
scorer = MLXWyatScorer.from_directory(
    root / ".models/Decision-2.0-Kai-0.6B",
    max_length=512,
    device="gpu",
    token_cache_size=256,
)
assert scorer.model_id == reference["model_id"]
with mx.stream(scorer.device):
    # Experiment only: all backbone storage/arithmetic uses the selected dtype,
    # embeddings and normalization. The independently validated FP32 head stays
    # unchanged. This is not the released CUDA autocast recipe.
    backbone = Qwen3Model(scorer.backbone.args)
    weights = cast(
        dict[str, mx.array],
        mx.load(str(root / ".models/Decision-2.0-Kai-0.6B/backbone/model.safetensors")),
    )
    converted = {name: value.astype(storage_dtype) for name, value in weights.items()}
    for name, value in converted.items():
        flat = value.reshape(-1)
        for chunk_start in range(0, flat.size, 1 << 20):
            if not bool(
                mx.all(mx.isfinite(flat[chunk_start : chunk_start + (1 << 20)])).item()
            ):
                raise ValueError(f"Nonfinite converted checkpoint tensor: {name}")
    backbone.load_weights(list(converted.items()), strict=True)
    backbone.eval()
    mx.eval(converted)
candidate = MLXWyatScorer(
    backbone,
    scorer.head,
    scorer.encoder,
    model_id=scorer.model_id,
    device=scorer.device,
)
left: list[EvaluationRow] = []
right: list[EvaluationRow] = []
scores: list[list[float]] = []
requests = tuple(case.request for case in cases)
start = time.perf_counter()
for offset in range(0, len(requests), 8):
    scores.extend(candidate.score_many(requests[offset : offset + 8]))
first_scoring_seconds = time.perf_counter() - start
for case, ref, logits in zip(cases, reference_rows, scores, strict=True):
    assert isinstance(ref, dict)
    assert case.id == ref["id"]
    saved = ref["legacy_logits"]
    assert isinstance(saved, list)
    expected = tuple(float(cast(float, value)) for value in saved)
    a = decision_from_logits(case.request, expected, scorer.model_id)
    b = decision_from_logits(case.request, logits, scorer.model_id)
    # These cycles are quality comparisons against stored reference scores,
    # with zero timing fields; no reference runtime speed is inferred from them.
    left.append(
        EvaluationRow(
            case.id, case.expected, "ok", a, a.selected in case.expected, 0.0, None
        )
    )
    right.append(
        EvaluationRow(
            case.id, case.expected, "ok", b, b.selected in case.expected, 0.0, None
        )
    )
gate = QualityGate()
drift = compare_cycles(
    [BenchmarkCycle(0.0, tuple(left), 0, 0)],
    [BenchmarkCycle(0.0, tuple(right), 0, 0)],
    gate,
)
receipt: dict[str, object] = {
    "schema_version": 1,
    "hypothesis": (
        f"{precision.upper()} all backbone tensors with FP32 decision head "
        "preserves strict FP32 reference gates"
    ),
    "model_id": scorer.model_id,
    "dataset_sha256": reference["dataset_sha256"],
    "cases": len(cases),
    "backbone_dtype": dtype_name,
    "head_dtype": "float32",
    "finite_parameters_and_outputs": True,
    "device": "Apple M4 Pro Metal GPU",
    "scope": "frozen BANKING77 development77; checkpoint data exposure unknown",
    "first_scoring_seconds": first_scoring_seconds,
    "quality_gate": asdict(gate),
    "quality": asdict(drift),
    "passed": drift.passed,
    "candidate_rows": [row.to_dict() for row in right],
    "production_dtype_flag_added": False,
}
(root / f"benchmarks/real/mlx-{precision}-screen77.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(
    json.dumps(
        {key: value for key, value in receipt.items() if key != "candidate_rows"}
    ),
    flush=True,
)
if drift.passed:
    report = benchmark_pair(
        BenchmarkEngine("mlx-fp32-eager", Model(scorer, scorer.model_id), 8),
        BenchmarkEngine(
            f"mlx-{precision}-backbone-fp32-head",
            Model(candidate, candidate.model_id),
            8,
        ),
        cases,
        protocol=BenchmarkProtocol(repeats=5, warmup=1, quality_gate=gate),
    )
    payload = report.to_dict()
    (root / f"benchmarks/real/mlx-{precision}-performance77.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(
        json.dumps({"performance_passed": report.passed, "speed": payload["speed"]}),
        flush=True,
    )
