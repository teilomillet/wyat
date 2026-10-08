"""Fixed-logit typed readout: no model loading or learned-quality claim."""

import json
import math

from wyat.tasks import (
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
        },
    }
    fixture_logits = {
        "action": (0.0, 2.0),
        "urgency": (0.0, math.log(3.0), math.log(2.0)),
        "billing": (0.0, math.log(3.0)),
    }
    _, questions = public_questions(record)
    answers = {
        question.id: interpret(question, fixture_logits[question.id])
        for question in questions
    }
    action, urgency, billing = (
        answers["action"],
        answers["urgency"],
        answers["billing"],
    )
    assert isinstance(action, ChoiceAnswer) and action.choice == "refund"
    assert isinstance(urgency, ScoreAnswer)
    assert math.isclose(urgency.expected_ordinal_value, 7 / 6, abs_tol=1e-6)
    assert isinstance(billing, NoulAnswer)
    assert math.isclose(billing.probability_true, 0.75, abs_tol=1e-6)
    print(json.dumps({key: value.to_dict() for key, value in answers.items()}))


if __name__ == "__main__":
    main()
