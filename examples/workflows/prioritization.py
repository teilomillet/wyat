"""Pure Score interpretation -> explicit ordinal policy -> toy queue update.

Fixed logits only; no native Score inference loader or priority-quality claim.
"""

import json
from collections.abc import Mapping, Sequence

from wyat import ModelOutputError
from wyat.tasks import ScoreAnswer, interpret, public_questions

RUBRIC = ("Can wait", "This week", "Today")


def rank_queue(queue: Sequence[str], answers: Mapping[str, ScoreAnswer]) -> list[str]:
    """Descending expectation; exact ties retain the previous queue order."""
    if any(ticket_id not in answers for ticket_id in queue):
        raise ValueError("incomplete readouts; preserve the previous queue")
    return sorted(
        queue,
        key=lambda ticket_id: answers[ticket_id].expected_ordinal_value,
        reverse=True,
    )


def run_demo() -> dict[str, object]:
    _, questions = public_questions(
        {
            "state": "Synthetic support tickets",
            "questions": {
                "urgency": {
                    "type": "score",
                    "instructions": "Rate urgency against the ordered rubric.",
                    "criteria": list(RUBRIC),
                }
            },
        }
    )
    question = questions[0]
    queue = ["split", "urgent", "invalid"]
    answers: dict[str, ScoreAnswer] = {}
    failures = []
    for ticket_id, logits in (
        ("split", (2.0, -4.0, 2.0)),
        ("urgent", (0.0, 0.0, 4.0)),
        ("invalid", (0.0, float("nan"), 2.0)),
    ):
        try:
            answer = interpret(question, logits)
            if not isinstance(answer, ScoreAnswer):
                raise TypeError("expected a Score readout")
            answers[ticket_id] = answer
        except ModelOutputError as error:
            failures.append(
                {"ticket_id": ticket_id, "stage": "readout", "error": str(error)}
            )
    initial_results = {key: answer.to_dict() for key, answer in answers.items()}
    try:
        proposed = rank_queue(queue, answers)
    except ValueError:
        proposed = list(queue)
    queue_after_failure = proposed

    # A retained replacement fixture represents operator recovery, not an
    # erased first failure or a repaired model-quality observation.
    replacement = interpret(question, (4.0, 0.0, 0.0))
    if not isinstance(replacement, ScoreAnswer):
        raise TypeError("expected a Score readout")
    answers["invalid"] = replacement
    queue = rank_queue(queue, answers)
    return {
        "inference": "pure typed interpreter; no native Score loader",
        "rubric": RUBRIC,
        "policy": "descending expected ordinal index; stable ties; atomic queue update",
        "initial_attempts": 3,
        "initial_results": initial_results,
        "operational_failures": failures,
        "queue_after_failure": queue_after_failure,
        "recovery": {"ticket_id": "invalid", "replacement_logits": [4.0, 0.0, 0.0]},
        "queue_after_recovery": queue,
        "results": {key: answer.to_dict() for key, answer in answers.items()},
        "limitation": "expectation is the caller's ranking key, not a winning category",
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
