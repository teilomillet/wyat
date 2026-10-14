# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choice per aspect: turn free-text reviews into a table.

One question per aspect you care about, all on the same review.
``predict_many`` scores every review in one call.
"""

from common import load_model

import wyat

SENTIMENT = {
    "positive": "Positive.",
    "negative": "Negative.",
    "not_mentioned": "Not mentioned.",
}
ASPECTS = ["food", "service", "price"]

model = load_model(__doc__)

reviews = [
    "The pasta was great but the waiter ignored us for twenty minutes.",
    "Cheap and fast. Nothing special.",
]
records = [
    {
        "state": review,
        "questions": {
            aspect: {
                "type": "choice",
                "instructions": f"What does the review say about the {aspect}?",
                "criteria": SENTIMENT,
            }
            for aspect in ASPECTS
        },
    }
    for review in reviews
]
print("review | " + " | ".join(ASPECTS))
for review, answers in zip(reviews, model.predict_many(records), strict=True):
    cells: list[str] = []
    for aspect in ASPECTS:
        answer = answers[aspect]
        assert isinstance(answer, wyat.ChoiceAnswer)
        cells.append(answer.choice)
    print(f"{review[:30]}... | " + " | ".join(cells))
