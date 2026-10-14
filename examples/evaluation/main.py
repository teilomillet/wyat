# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Evaluate fixed logits with a correct case, a wrong case, and a failed call.

This demonstrates failure-inclusive reporting, not trained model quality.
"""

import json
from collections.abc import Sequence

import wyat
from wyat.evaluation import EvaluationCase, evaluate


class DemonstrationScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        if any(request.context == "Backend unavailable" for request in requests):
            raise RuntimeError("demonstration backend failure")
        return tuple((0.0, 2.0) for _ in requests)


def request(context: str) -> wyat.Request:
    return wyat.Request(
        query="Choose the requested action",
        context=context,
        candidates=[
            wyat.Candidate("create", "Create a booking"),
            wyat.Candidate("cancel", "Cancel the booking"),
        ],
    )


def main() -> None:
    model = wyat.Model(DemonstrationScorer(), model_id="demo-fixed-logits")
    cases = [
        EvaluationCase("correct", request("Cancel my booking"), ("cancel",)),
        EvaluationCase("wrong", request("Create a booking"), ("create",)),
        EvaluationCase("backend-failure", request("Backend unavailable"), ("cancel",)),
    ]
    report = evaluate(model, cases)
    # All three attempts count: accuracy is 1/3 and coverage is 2/3.
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
