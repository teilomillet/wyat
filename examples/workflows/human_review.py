"""Pure Noul readout -> illustrative policy -> simulated review -> toy effect.

No native Noul inference, threshold recommendation, or real reviewer evidence.
"""

import json
import math
from typing import Literal

from wyat import ModelOutputError
from wyat.tasks import NoulAnswer, interpret, public_questions


def apply_policy(
    probability_true: float, *, accept_min: float, reject_max: float
) -> Literal["accept", "reject", "review"]:
    if not 0 <= reject_max < accept_min <= 1:
        raise ValueError("policy requires 0 <= reject_max < accept_min <= 1")
    if not math.isfinite(probability_true) or not 0 <= probability_true <= 1:
        raise ValueError("policy requires a valid probability")
    if probability_true >= accept_min:
        return "accept"
    if probability_true <= reject_max:
        return "reject"
    return "review"


def resolve_review(
    review: dict[str, object], resolution: dict[str, str], effects: list[dict[str, str]]
) -> None:
    """Apply only this toy pending record's separate reviewer decision."""
    if review["event_id"] != resolution["event_id"]:
        raise ValueError("resolution must match the review event")
    if review["status"] != "pending":
        raise ValueError("review is already resolved")
    if resolution["decision"] not in {"approve_billing_route", "reject_billing_route"}:
        raise ValueError("unknown reviewer decision")
    if resolution["decision"] == "approve_billing_route":
        effects.append({"event_id": resolution["event_id"], "queue": "billing"})
    review["resolution"] = dict(resolution)
    review["status"] = "resolved"


def run_demo() -> dict[str, object]:
    state = "Does this synthetic message concern billing?"
    _, questions = public_questions(
        {"state": state, "questions": {"billing": {"type": "noul"}}}
    )
    answer = interpret(questions[0], (0.0, math.log(3.0)))
    if not isinstance(answer, NoulAnswer):
        raise TypeError("expected a Noul readout")
    accept_min, reject_max = 0.9, 0.1
    policy = {
        "version": "illustrative-review-v1",
        "accept_min": accept_min,
        "reject_max": reject_max,
    }
    outcome = apply_policy(
        answer.probability_true, accept_min=accept_min, reject_max=reject_max
    )
    reviews: list[dict[str, object]] = []
    effects: list[dict[str, str]] = []
    if outcome == "review":
        reviews.append(
            {
                "event_id": "event-1",
                "state": state,
                "result": answer.to_dict(),
                "logits": answer.logits,
                "policy": policy,
                "status": "pending",
                "reason": "uncertainty",
            }
        )
    pending_reviews = [dict(review) for review in reviews]
    effects_before_resolution = list(effects)
    resolution = {
        "event_id": "event-1",
        "reviewer": "simulated-reviewer",
        "decision": "approve_billing_route",
    }
    resolve_review(reviews[0], resolution, effects)
    failures = []
    try:
        interpret(questions[0], (0.0, float("nan")))
    except ModelOutputError as error:
        failures.append(
            {
                "event_id": "event-2",
                "stage": "readout",
                "error": str(error),
                "recovery": "retain input; repair producer before resubmission",
            }
        )
    return {
        "inference": "pure typed interpreter; no native Noul loader",
        "result": answer.to_dict(),
        "policy": policy,
        "policy_decision": outcome,
        "pending_reviews": pending_reviews,
        "effects_before_resolution": effects_before_resolution,
        "resolved_reviews": reviews,
        "effects": effects,
        "operational_failures": failures,
        "limitation": "illustrative policy and simulated review; no calibration claim",
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
