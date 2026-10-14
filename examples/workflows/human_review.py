# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Fixed Noul scorer -> caller policy -> pending review -> effect after approval.

No weights or external effects. The cutoffs are illustrative caller choices, not
Wyat defaults or calibrated thresholds, and the reviewer is simulated.
"""

import json
import math
from collections.abc import Sequence
from typing import Literal

import wyat


class FixtureScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows: list[tuple[float, ...]] = []
        for request in requests:
            if request.context == "invalid output":
                rows.append((0.0, float("nan")))
            else:
                # Logits for (false, true): probability_true is exactly 0.75.
                rows.append((0.0, math.log(3.0)))
        return tuple(rows)


def prepare_request(message: str) -> wyat.Request:
    return wyat.Request.noul("Does this message concern billing?", context=message)


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
    """Apply a reviewer decision to its own pending review, at most once."""
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
    model = wyat.Model(FixtureScorer(), "review-fixed-logits")
    accept_min, reject_max = 0.9, 0.1
    policy = {
        "version": "illustrative-review-v1",
        "accept_min": accept_min,
        "reject_max": reject_max,
    }
    reviews: list[dict[str, object]] = []
    effects: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    request = prepare_request("Please fix the duplicate invoice.")
    decision = model.decide(request)
    outcome = apply_policy(
        decision.probability_true, accept_min=accept_min, reject_max=reject_max
    )
    if outcome == "accept":
        effects.append({"event_id": "event-1", "queue": "billing"})
    elif outcome == "review":
        # The pending record keeps everything a reviewer or a later audit needs.
        reviews.append(
            {
                "event_id": "event-1",
                "context": request.context,
                "result": decision.to_dict(),
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

    try:
        model.decide(prepare_request("invalid output"))
    except wyat.ModelOutputError as error:
        failures.append(
            {
                "event_id": "event-2",
                "stage": "output",
                "error": str(error),
                "recovery": "keep the input; fix the producer before resubmitting",
            }
        )
    return {
        "inference": "fixed scorer; no weights",
        "result": decision.to_dict(),
        "probability_true": decision.probability_true,
        "policy": policy,
        "policy_decision": outcome,
        "pending_reviews": pending_reviews,
        "effects_before_resolution": effects_before_resolution,
        "resolved_reviews": reviews,
        "effects": effects,
        "operational_failures": failures,
        "limitation": "illustrative cutoffs and simulated reviewer; no calibration",
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
