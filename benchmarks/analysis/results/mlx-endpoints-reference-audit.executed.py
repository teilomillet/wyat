"""Pure post-run audit of every endpoint cycle against frozen CPU scores."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from wyat.benchmark import BenchmarkCycle, QualityGate, compare_cycles
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict
from wyat.wire import decode_json, read_object


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected object")
    return cast(dict[str, object], value)


def sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected list")
    return cast(list[object], value)


def cycle(
    raw: object, cases: tuple[EvaluationCase, ...], model_id: str
) -> BenchmarkCycle:
    rows = sequence(object_value(raw)["rows"])
    if len(rows) != len(cases):
        raise ValueError("Cycle must contain every declared request")
    result: list[EvaluationRow] = []
    for case, item in zip(cases, rows, strict=True):
        row = object_value(item)
        if (
            row["id"] != case.id
            or row["expected"] != list(case.expected)
            or row.get("status", "ok") != "ok"
        ):
            raise ValueError("Cycle IDs, labels and successful coverage must match")
        prediction = row.get("prediction")
        if prediction is None:
            scores = sequence(row["legacy_logits"])
            if row["candidate_ids"] != [
                option.id for option in case.request.candidates
            ]:
                raise ValueError("Frozen candidate ordering changed")
        else:
            scores = sequence(object_value(prediction)["scores"])
        if any(type(score) not in (int, float) for score in scores):
            raise ValueError("Every score must be numeric")
        decision = decision_from_logits(
            case.request, tuple(float(cast(float, score)) for score in scores), model_id
        )
        if prediction is not None:
            recorded = object_value(prediction)
            if (
                recorded["model_id"] != model_id
                or recorded["selected"] != decision.selected
                or recorded["ranking"] != list(decision.ranking)
                or recorded["probabilities"] != list(decision.probabilities)
                or row["correct"] != (decision.selected in case.expected)
            ):
                raise ValueError("Recorded prediction differs from canonical scores")
        result.append(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                decision,
                decision.selected in case.expected,
                0,
                None,
            )
        )
    return BenchmarkCycle(0, tuple(result), 0, 0)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
    reference = object_value(read_object(reference_path))
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    model_id = cast(str, reference["model_id"])
    frozen = cycle(reference, cases, model_id)
    audits: dict[str, object] = {}
    for batch in (8, 1):
        path = root / f"benchmarks/analysis/results/mlx-endpoints77-b{batch}.json"
        receipt = object_value(read_object(path))
        if (
            receipt["passed"] is not True
            or receipt["completed"] is not True
            or receipt["reference_sha256"]
            != hashlib.sha256(reference_path.read_bytes()).hexdigest()
            or receipt["dataset_sha256"]
            != hashlib.sha256(dataset.read_bytes()).hexdigest()
            or receipt["model_id"] != model_id
        ):
            raise ValueError("Receipt identity or completion mismatch")
        paired = object_value(receipt["paired"])
        warmup = object_value(paired["warmup"])
        arms: dict[str, object] = {}
        for arm in ("baseline", "candidate"):
            warm = sequence(warmup[f"{arm}_cycles"])
            measured = sequence(object_value(paired[arm])["cycles"])
            if len(warm) != 1 or len(measured) != 5:
                raise ValueError("Expected one warmup and five complete cycles")
            cycles = tuple(cycle(raw, cases, model_id) for raw in warm + measured)
            gate = compare_cycles((frozen,) * len(cycles), cycles, QualityGate())
            arms[arm] = asdict(gate)
            if not gate.passed:
                raise ValueError(f"CPU reference gate failed: batch{batch}/{arm}")
        audits[f"batch{batch}"] = {
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cycles_per_arm": 6,
            "arms": arms,
        }
    output = Path(__file__).with_name("mlx-endpoints-reference-audit.json")
    payload = {
        "passed": True,
        "audits": audits,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Pure post-run audit; no additional model execution or timing",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
