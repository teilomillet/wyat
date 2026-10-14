# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Score: rerank search results by graded relevance.

A fast search returns candidates; the model grades each one against the
query. Works for product search, documents before RAG, or help articles.
"""

from common import load_model

import wyat

RELEVANCE = [
    "Unrelated to the query.",
    "Related, but not what was asked.",
    "A good substitute.",
    "Exactly what was asked.",
]

model = load_model(__doc__)

query = "usb-c charger 65w"
results = [
    "Phone case with card holder",
    "65W USB-C GaN wall charger",
    "45W USB-C charger",
    "USB-C to USB-C cable, 2m",
]
decisions = model.decide_many(
    [
        wyat.Request.score(
            "How well does this product match the search query?",
            RELEVANCE,
            context=f"Query: {query}\nProduct: {product}",
        )
        for product in results
    ]
)
for product, decision in sorted(
    zip(results, decisions, strict=True),
    key=lambda pair: pair[1].expected_ordinal_value,
    reverse=True,
):
    print(f"{decision.expected_ordinal_value:.2f}  {product}")
