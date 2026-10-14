# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Support triage: three decisions per ticket, one policy, one audit record.

Three typed questions about each ticket: which queue (Choice), how urgent
(Score) and whether it needs a human (Noul). ``predict_many`` answers them
for every ticket in one call.
A small policy then turns the answers into an action, and abstains to a human
when the queue decision is not clear-cut. Every decision is logged with the
numbers behind it, so it can be reviewed and the policy re-tuned later.
"""

import json
from dataclasses import asdict, dataclass

from common import load_model

import wyat

POLICY_VERSION = "triage-2026-10-a"
MIN_MARGIN = 0.15  # top queue must beat the runner-up by this much
HUMAN_IF_ABOVE = 0.6  # p(needs human) above this always escalates
SLA_HOURS = {0: 72, 1: 24, 2: 4, 3: 1}  # rounded urgency level -> response time

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "Billing: charges, refunds, invoices, payment methods.",
            "technical": "Technical: login, errors, crashes, app bugs.",
            "account": "Account: profile, security, password, data access.",
            "delivery": "Delivery: shipping, tracking, damaged or missing parcels.",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": [
            "Can wait: a question or feedback.",
            "This week: inconvenient but working.",
            "Today: the customer is blocked.",
            "Now: money lost, security risk, or many users affected.",
        ],
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Is the customer angry, threatening to leave, or a legal risk?",
    },
}


@dataclass(frozen=True)
class Triage:
    ticket_id: str
    action: str  # "route" or "human"
    queue: str
    sla_hours: int
    reason: str
    queue_probabilities: dict[str, float]
    urgency: float
    p_needs_human: float
    model_id: str
    policy: str = POLICY_VERSION


def triage(
    ticket_id: str,
    answers: dict[str, wyat.ChoiceAnswer | wyat.ScoreAnswer | wyat.NoulAnswer],
    model_id: str,
) -> Triage:
    """The policy: plain code over the model's answers, easy to test and change."""
    queue, urgency, human = answers["queue"], answers["urgency"], answers["needs_human"]
    assert isinstance(queue, wyat.ChoiceAnswer)
    assert isinstance(urgency, wyat.ScoreAnswer)
    assert isinstance(human, wyat.NoulAnswer)

    top, second = sorted(queue.probabilities, reverse=True)[:2]
    level = urgency.expected_ordinal_value
    if human.probability_true > HUMAN_IF_ABOVE:
        action, reason = "human", "sensitive customer situation"
    elif top - second < MIN_MARGIN:
        action, reason = "human", f"unclear queue (margin {top - second:.2f})"
    else:
        action, reason = "route", "clear queue"
    return Triage(
        ticket_id=ticket_id,
        action=action,
        queue=queue.choice,
        sla_hours=SLA_HOURS[round(level)],
        reason=reason,
        queue_probabilities={
            key: round(p, 3) for key, p in queue.probabilities_by_id.items()
        },
        urgency=round(level, 3),
        p_needs_human=round(human.probability_true, 3),
        model_id=model_id,
    )


model = load_model(__doc__)
tickets = {
    "T-1": {
        "subject": "Charged twice",
        "body": "My card was charged twice for the same invoice, please refund.",
    },
    "T-2": {
        "subject": "Cannot login",
        "body": "Password reset email never arrives, I am locked out of my account.",
    },
    "T-3": {
        "subject": "Fed up",
        "body": "Third time I write. Fix this or I cancel and contact my lawyer.",
    },
}
records = [{"state": ticket, "questions": QUESTIONS} for ticket in tickets.values()]
for ticket_id, answers in zip(tickets, model.predict_many(records), strict=True):
    result = triage(ticket_id, answers, model.model_id)
    print(
        f"{ticket_id}: {result.action:<5} queue={result.queue:<9} "
        f"sla={result.sla_hours}h  ({result.reason})"
    )
    # One JSON line per decision: replayable, diffable, easy to load in a notebook.
    print("   audit:", json.dumps(asdict(result)))
