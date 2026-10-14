# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Fixed-logit scoring and keyed results; no trained model or quality claim.

Softmax probabilities depend on the supplied candidates; they are not calibrated
confidence.
"""

from collections.abc import Sequence

import wyat


class FixedScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.0, 2.0) for _ in requests)


def main() -> None:
    model = wyat.Model(FixedScorer(), model_id="demo-fixed-logits")
    request = wyat.Request(
        query="Choose the requested action",
        context="Cancel my booking",
        candidates=[
            wyat.Candidate("create", "Create a booking"),
            wyat.Candidate("cancel", "Cancel the booking"),
        ],
    )
    decision = model.decide(request)
    result = decision.to_dict()
    result["selected_score"] = decision.scores_by_id[decision.selected]
    result["selected_probability"] = decision.probabilities_by_id[decision.selected]
    print(result)


if __name__ == "__main__":
    main()
