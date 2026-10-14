# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Plain-Python RAG: BM25 search, Wyat filter, prompt for any LLM.

No framework needed. BM25 (or your vector store) is fast but shallow: it
returns passages that share words, relevant or not. Wyat grades each passage
against the question in one batched call, and only passages that at least
partly answer it go into the prompt. If none survive, the system answers
"I don't know" without calling the LLM at all.

    uv run --with 'rank-bm25>=0.2' python 05_bm25_rag.py --demo
"""

from common import load_model, missing

import wyat

try:
    from rank_bm25 import BM25Okapi
except ImportError as error:
    missing("rank-bm25>=0.2", error)

PASSAGES = [
    "Refunds are accepted within 30 days with the receipt.",
    "Refund requests without a receipt get store credit only.",
    "The refund desk is next to the main entrance.",
    "Gift cards cannot be refunded or exchanged for cash.",
    "Shipping takes 3 to 5 working days within the EU.",
]
TOP_K = 4
GRADES = [
    "Unrelated to the question.",
    "Related background only.",
    "Partly answers the question.",
    "Directly answers the question.",
]
KEEP_AT_LEAST = 2.0  # "partly answers" or better; when unsure, abstain

bm25 = BM25Okapi([passage.lower().split() for passage in PASSAGES])
model = load_model(__doc__)


def answer_prompt(question: str) -> str | None:
    scores = bm25.get_scores(question.lower().split())
    candidates = sorted(range(len(PASSAGES)), key=lambda i: -scores[i])[:TOP_K]
    grades = model.decide_many(
        [
            wyat.Request.score(
                "How well does this passage answer the question?",
                GRADES,
                context=f"Question: {question}\nPassage: {PASSAGES[i]}",
            )
            for i in candidates
        ]
    )
    kept = [
        PASSAGES[i]
        for i, grade in zip(candidates, grades, strict=True)
        if grade.expected_ordinal_value >= KEEP_AT_LEAST
    ]
    print(f"  bm25 top {TOP_K} -> wyat kept {len(kept)}")
    if not kept:
        return None
    return "Answer only from these facts:\n- " + "\n- ".join(kept) + f"\n\n{question}"


for question in ["Can I get a refund without a receipt?", "Do you repair bikes?"]:
    print(f"Q: {question}")
    prompt = answer_prompt(question)
    print(prompt if prompt else "  -> I don't know (no LLM call)")
    print()
