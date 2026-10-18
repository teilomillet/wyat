# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choice per aspect: turn free-text reviews into a table.

One question per aspect you care about, all on the same review.
``predict_many`` scores every review in one call.

wyat-1 800M reads clear statements well ("the waiter ignored us" is negative
service) but misses one-word cues: here it reads "Cheap" as not mentioning
the price. Check a sample of rows before trusting the table.
"""

from common import load_model

import wyat

SENTIMENT = {
    "positive": "Positive: the review praises it.",
    "negative": "Negative: the review complains about it.",
    "not_mentioned": "Not mentioned: the review says nothing about it.",
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
