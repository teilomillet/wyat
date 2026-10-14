# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an LLM's answers: rubric scores per criterion, then A/B with swap.

Use a decision model as a cheap, fast judge for prompts or model versions:
1. Score each answer on several criteria in one ``predict`` call.
2. Compare two answers head to head. Ask twice with A and B swapped, so a
   preference for "the first one" cancels out. Only a consistent preference
   counts as a win; otherwise it is a tie.
"""

from common import load_model

import wyat

CRITERIA = {
    "correct": ["Wrong.", "Partly correct.", "Fully correct."],
    "grounded": ["Invents facts.", "Mostly supported.", "Fully supported by policy."],
    "tone": ["Rude or cold.", "Neutral.", "Warm and professional."],
}

model = load_model(__doc__)
policy = "Refunds within 30 days with receipt. Shipping 3 to 5 working days."
question = "I bought shoes 20 days ago, can I get my money back?"
answers = {
    "prompt_v1": "Yes, refunds are accepted within 30 days with your receipt.",
    "prompt_v2": "Refunds are possible anytime, just send the shoes back.",
}


def rubric(answer: str) -> dict[str, float]:
    result = model.predict(
        {
            "state": {"policy": policy, "question": question, "answer": answer},
            "questions": {
                name: {
                    "type": "score",
                    "instructions": f"Rate the answer: {name}.",
                    "criteria": levels,
                }
                for name, levels in CRITERIA.items()
            },
        }
    )
    levels: dict[str, float] = {}
    for name in CRITERIA:
        answer = result[name]
        assert isinstance(answer, wyat.ScoreAnswer)
        levels[name] = answer.expected_ordinal_value
    return levels


for name, answer in answers.items():
    scores = rubric(answer)
    print(f"{name}: " + ", ".join(f"{k}={v:.2f}/2" for k, v in scores.items()))


def prefer(first: str, second: str) -> wyat.Request:
    return wyat.Request(
        "Which answer is better for the customer, given the policy?",
        [wyat.Candidate("first", first), wyat.Candidate("second", second)],
        context=f"Policy: {policy}\nQuestion: {question}",
    )


a, b = answers["prompt_v1"], answers["prompt_v2"]
forward, backward = model.decide_many([prefer(a, b), prefer(b, a)])
a_wins = forward.selected == "first" and backward.selected == "second"
b_wins = forward.selected == "second" and backward.selected == "first"
print("A/B:", "prompt_v1 wins" if a_wins else "prompt_v2 wins" if b_wins else "tie")
