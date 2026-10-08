"""Explain typed readouts and caller policy using synthetic logits, no weights."""

import json
import math

from wyat.tasks import (
    RECIPE,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    interpret,
    public_questions,
)


def main() -> None:
    record = {
        "state": {"message": "Please fix the duplicate invoice — 請求書"},
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the allowed action.",
                "criteria": {"wait": "Wait.", "refund": "Refund."},
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate urgency against the ordered rubric.",
                "criteria": ["Can wait.", "This week.", "Today."],
            },
            "billing": {
                "type": "noul",
                "instructions": "Does the message concern billing?",
            },
            "urgency_split": {
                "type": "score",
                "instructions": "Rate urgency against the ordered rubric.",
                "criteria": ["Can wait.", "This week.", "Today."],
            },
        },
    }
    fixture_logits = {
        "action": (0.0, 2.0),
        "urgency": (0.0, math.log(3.0), math.log(2.0)),
        "billing": (0.0, math.log(3.0)),
        "urgency_split": (2.0, -4.0, 2.0),
    }
    _, questions = public_questions(record)
    notes: list[dict[str, object]] = []
    for question in questions:
        answer = interpret(question, fixture_logits[question.id])
        note: dict[str, object] = {
            "question_id": question.id,
            "question": question.instructions,
            "ordered_options": [
                {"id": key, "description": description}
                for key, description in question.options
            ],
            "readout_logits": answer.logits_by_id,
            "result": answer.to_dict(),
            "action_executed": False,
        }
        if isinstance(answer, ChoiceAnswer):
            note["calculation"] = (
                f"{answer.choice} has the largest logit; exact ties use input order."
            )
            note["policy"] = {"id": "demo-display-v1", "decision": "display"}
        elif isinstance(answer, ScoreAnswer):
            terms = [
                f"{index} * {probability:.9f}"
                for index, probability in enumerate(answer.probabilities)
            ]
            note["calculation"] = (
                " + ".join(terms) + f" ≈ {answer.expected_ordinal_value:.9f}"
            )
            note["unit"] = "zero-based rubric index"
            note["policy"] = {"id": "demo-display-v1", "decision": "display"}
        elif isinstance(answer, NoulAnswer):
            note["calculation"] = "Read the probability attached to the ID true."
            # Illustrative caller policy, not a library default or validated cutoff.
            true_cutoff, false_cutoff = 0.90, 0.10
            if answer.probability_true >= true_cutoff:
                decision = "accept_true"
            elif answer.probability_true <= false_cutoff:
                decision = "accept_false"
            else:
                decision = "review"
            note["policy"] = {
                "id": "demo-billing-review-v1",
                "accept_true_if": f"p_true >= {true_cutoff}",
                "accept_false_if": f"p_true <= {false_cutoff}",
                "otherwise": "review",
                "decision": decision,
            }
        notes.append(note)
    print(
        json.dumps(
            {
                "input": record["state"],
                "model_identity": "synthetic logits; no model",
                "recipe": RECIPE,
                "readout": (
                    "wyat.tasks.interpret; record deployed version in production"
                ),
                "probability_meaning": "relative model weights; calibration untested",
                "notes": notes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
