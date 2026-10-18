# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Turn a customer review into aspect/sentiment columns for an analytics table."""

import json

import wyat
from examples.use_cases.common import load_model

PAIRS = {
    "food_positive": "food with a positive sentiment",
    "service_negative": "service with a negative sentiment",
    "service_positive": "service with a positive sentiment",
    "ambience_negative": "ambience with a negative sentiment",
}


def tag_review(model: wyat.Model, review_id: str, text: str) -> dict[str, object]:
    requests = [
        wyat.Request.noul(
            f"Does the review explicitly express {pair}?",
            context=text,
        )
        for pair in PAIRS.values()
    ]
    decisions = model.decide_many(requests)
    # Each pair is an independent question. Missing aspects are not negative.
    probabilities = {
        name: decision.probability_true
        for name, decision in zip(PAIRS, decisions, strict=True)
    }
    return {
        "review_id": review_id,
        "tags": {name: p >= 0.5 for name, p in probabilities.items()},
        "probabilities": probabilities,
        "model_id": model.model_id,
    }


def run(model: wyat.Model) -> dict[str, object]:
    return tag_review(
        model,
        "review-001",
        "La soupe était délicieuse. En revanche, le serveur nous a ignorés "
        "pendant quarante minutes.",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
