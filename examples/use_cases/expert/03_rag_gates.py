# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""RAG with two decision gates: what goes into the prompt, and what comes out.

1. Retrieval returns many passages. One batched call grades each passage
   against the question (Score). Only passages that at least partly answer
   it enter the LLM prompt, within a character budget. No such passage means
   the system says it does not know instead of letting the LLM improvise.
2. After generation, a second Noul checks the answer against the kept
   passages. Unsupported answers are not shown.
"""

from common import load_model

import wyat

GRADES = [
    "Unrelated to the question.",
    "Related background only.",
    "Partly answers the question.",
    "Directly answers the question.",
]
KEEP_AT_LEAST = 2.0  # "partly answers" or better; when unsure, abstain
SHOW_ABOVE = 0.6
BUDGET_CHARS = 400

DOCUMENTS = {
    "returns": "Items can be returned within 30 days with the receipt for a refund.",
    "shipping": "Standard shipping takes 3 to 5 working days within the EU.",
    "warranty": "Electronics have a two year warranty covering manufacturing defects.",
    "stores": "Our Lyon store opens at 9am and closes at 7pm, Monday to Saturday.",
}


def select_context(model: wyat.Model, question: str) -> list[str]:
    # Question and passage both go in the context: the model reads evidence
    # there. A grade ranks better than a yes/no "does it help?" here.
    grades = model.decide_many(
        [
            wyat.Request.score(
                "How well does this passage answer the question?",
                GRADES,
                context=f"Question: {question}\nPassage: {text}",
            )
            for text in DOCUMENTS.values()
        ]
    )
    helpful = sorted(
        (
            (grade.expected_ordinal_value, doc_id)
            for doc_id, grade in zip(DOCUMENTS, grades, strict=True)
            if grade.expected_ordinal_value >= KEEP_AT_LEAST
        ),
        reverse=True,
    )
    kept, used = [], 0
    for _, doc_id in helpful:
        if used + len(DOCUMENTS[doc_id]) > BUDGET_CHARS:
            break
        kept.append(doc_id)
        used += len(DOCUMENTS[doc_id])
    return kept


def grounded(model: wyat.Model, kept: list[str], answer: str) -> float:
    request = wyat.Request.noul(
        "Is every claim in the answer stated in the passages?",
        context={"passages": [DOCUMENTS[d] for d in kept], "answer": answer},
    )
    return model.decide(request).probability_true


def fake_llm(question: str, kept: list[str]) -> str:
    """Stands in for your LLM call. It invents a claim when asked about costs."""
    answer = " ".join(DOCUMENTS[d] for d in kept)
    if "cost" in question:
        answer += " Shipping is always free."
    return answer


model = load_model(__doc__)
for question in [
    "How long does shipping take within the EU?",
    "How long does shipping take and what does it cost?",
    "Can I return items without a receipt?",
    "Do you sell gift cards?",
]:
    kept = select_context(model, question)
    print(f"Q: {question}\n   context: {kept or 'none'}")
    if not kept:
        print("   -> 'I don't know, let me connect you with an agent.'")
        continue
    answer = fake_llm(question, kept)
    p = grounded(model, kept, answer)
    verdict = "show" if p >= SHOW_ABOVE else "hold for review"
    print(f"   answer: {answer}\n   grounded p={p:.2f} -> {verdict}")
