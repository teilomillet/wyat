# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""pandas: label a whole table in one batched call and analyse the result.

Analysts often start from a CSV export. Build one request per row, call
``decide_many`` once, and add the answer and its margin as new columns. Low
margins become a review sample instead of silent errors.

    uv run --with 'pandas>=3' python 07_pandas_labels.py --demo
"""

from common import load_model, missing

import wyat

try:
    import pandas as pd
except ImportError as error:
    missing("pandas>=3", error)

TOPICS = {
    "delivery": "Delivery: late, lost or damaged parcels.",
    "quality": "Product quality: broken, defective, poor material.",
    "service": "Customer service: staff, answers, waiting time.",
    "price": "Price: too expensive, discounts, value for money.",
}
REVIEW_BELOW = 0.2

model = load_model(__doc__)
df = pd.DataFrame(
    {
        "rating": [1, 2, 4, 1, 3],
        "comment": [
            "Parcel arrived late and damaged",
            "The zipper broke after two days, poor quality",
            "Good value for the price",
            "Waited an hour on the phone, rude staff",
            "It is fine I guess",
        ],
    }
)

decisions = model.decide_many(
    [
        wyat.Request.choice("What is this review mainly about?", TOPICS, context=c)
        for c in df["comment"]
    ]
)
df["topic"] = [d.selected for d in decisions]
df["margin"] = [round(d.probability_margin, 2) for d in decisions]
df["review"] = df["margin"] < REVIEW_BELOW

print(df.to_string(index=False))
print("\nAverage rating by topic (confident rows only):")
print(df[~df["review"]].groupby("topic")["rating"].mean())
