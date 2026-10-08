"""Frozen synthetic Noul inputs -> caller policy -> full-denominator accounting.

Cutoffs are required application parameters, not Wyat defaults. These hand-picked
fixtures demonstrate accounting and boundary comparisons, not model quality,
calibration, policy selection, or completed human review.
"""

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from wyat import ModelOutputError
from wyat.tasks import NoulAnswer, interpret, public_questions


@dataclass(frozen=True)
class FixtureCase:
    id: str
    label: bool
    logits: tuple[float, ...] | None
    status: Literal["ok", "missing", "failed", "unsupported"] = "ok"


# Fixed before execution: six positives, six negatives, eight interpretable
# readouts and four operational failures. Labels never enter the question.
FROZEN_CASES = (
    FixtureCase("correct-accept", True, (0.0, math.log(9.0))),
    FixtureCase("false-accept-extreme", False, (0.0, 100.0)),
    FixtureCase("correct-reject", False, (0.0, -math.log(9.0))),
    FixtureCase("false-reject-extreme", True, (0.0, -100.0)),
    FixtureCase("uncertain-positive", True, (0.0, math.log(3.0))),
    FixtureCase("uncertain-negative", False, (0.0, 0.0)),
    FixtureCase("boundary-positive", True, (0.0, math.log(4.0) - 1e-6)),
    FixtureCase("false-accept-high", False, (0.0, math.log(9.0))),
    FixtureCase("missing-positive", True, None, "missing"),
    FixtureCase("failed-negative", False, None, "failed"),
    FixtureCase("invalid-positive", True, (0.0,)),
    FixtureCase("unsupported-negative", False, None, "unsupported"),
)


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def evaluate_policy(
    cases: Sequence[FixtureCase],
    *,
    policy_id: str,
    accept_min: float,
    reject_max: float,
) -> dict[str, object]:
    if not policy_id.strip() or not 0 <= reject_max < accept_min <= 1:
        raise ValueError("provide a named policy with ordered cutoffs in [0, 1]")
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("cases must be nonempty and have unique IDs")
    _, questions = public_questions(
        {
            "state": "Synthetic binary-event fixtures",
            "questions": {"event": {"type": "noul"}},
        }
    )
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    for case in cases:
        row: dict[str, object] = {"id": case.id, "label": case.label}
        answer: NoulAnswer | None = None
        status: str = case.status
        if case.status == "ok":
            if case.logits is None:
                raise ValueError("an ok fixture requires logits")
            try:
                result = interpret(questions[0], case.logits)
                if not isinstance(result, NoulAnswer):
                    raise TypeError("expected a Noul readout")
                answer = result
            except ModelOutputError as error:
                status = "invalid"
                row["error"] = str(error)
        else:
            row["error"] = f"synthetic {case.status} prediction; no readout"
        row["status"] = status
        statuses[status] += 1
        if answer is None:
            # Operational referral is not a semantic review decision and earns
            # no correctness credit. Retain it in every population denominator.
            outcome = "operator"
            counts["operational_failures"] += 1
        else:
            probability = answer.probability_true
            row["logits"] = answer.logits
            row["answer"] = answer.to_dict()
            if probability >= accept_min:  # Compare canonical, unrounded values.
                outcome = "accept"
                counts["true_accepts" if case.label else "false_accepts"] += 1
            elif probability <= reject_max:
                outcome = "reject"
                counts["false_rejects" if case.label else "true_rejects"] += 1
            else:
                outcome = "review"
                counts["uncertainty_reviews"] += 1
        if outcome in {"review", "operator"}:
            counts[
                "unresolved_positives" if case.label else "unresolved_negatives"
            ] += 1
        row["outcome"] = outcome
        row["reason"] = (
            "operational"
            if outcome == "operator"
            else "uncertainty"
            if outcome == "review"
            else "valid"
        )
        counts[outcome] += 1
        rows.append(row)

    total = len(cases)
    positives = sum(case.label for case in cases)
    negatives = total - positives
    autonomous = counts["accept"] + counts["reject"]
    errors = counts["false_accepts"] + counts["false_rejects"]
    return {
        "policy": {
            "id": policy_id,
            "accept_min": accept_min,
            "reject_max": reject_max,
            "comparisons": (
                "accept >= accept_min; reject <= reject_max; otherwise review"
            ),
        },
        "population": {"total": total, "positive": positives, "negative": negatives},
        "statuses": dict(statuses),
        "counts": {
            name: counts[name]
            for name in (
                "accept",
                "reject",
                "review",
                "operator",
                "true_accepts",
                "false_accepts",
                "true_rejects",
                "false_rejects",
                "uncertainty_reviews",
                "operational_failures",
                "unresolved_positives",
                "unresolved_negatives",
            )
        },
        "metrics": {
            "valid_prediction_coverage": ratio(statuses["ok"], total),
            "autonomous_coverage": ratio(autonomous, total),
            "uncertainty_review_load": ratio(counts["review"], total),
            "operational_failure_load": ratio(counts["operator"], total),
            "false_accept_rate": ratio(counts["false_accepts"], negatives),
            "false_reject_rate": ratio(counts["false_rejects"], positives),
            "accept_precision": ratio(counts["true_accepts"], counts["accept"]),
            "autonomous_error_rate": ratio(errors, autonomous),
        },
        "rows": rows,
    }


def run_demo() -> dict[str, object]:
    report = evaluate_policy(
        FROZEN_CASES,
        policy_id="illustrative-selective-v1",
        accept_min=0.8,
        reject_max=0.2,
    )
    report["inference"] = "fixed logits and declared synthetic failures; no weights"
    report["limitation"] = (
        "synthetic accounting only; reviews unresolved; no quality, calibration, "
        "or optimal-threshold claim"
    )
    return report


def main() -> None:
    print(json.dumps(run_demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
