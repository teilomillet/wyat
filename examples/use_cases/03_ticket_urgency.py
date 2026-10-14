# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Score: rate urgency against an ordered rubric, then sort the queue.

Score returns an expected level between 0 and the last rubric level,
so tickets can be sorted even when two share the same top level.
"""

from common import load_model

import wyat

RUBRIC = [
    "Can wait: a question or feedback.",
    "This week: something is inconvenient.",
    "Today: the customer is blocked.",
    "Now: money, security or many users affected.",
]

model = load_model(__doc__)

tickets = [
    "How do I change my profile picture?",
    "Someone logged into my account from another country.",
    "I can't download my invoice for this month.",
]
requests = [
    wyat.Request.score("How urgent is this ticket?", RUBRIC, context=ticket)
    for ticket in tickets
]
decisions = model.decide_many(requests)  # one batched call

ranked = sorted(
    zip(tickets, decisions, strict=True),
    key=lambda pair: pair[1].expected_ordinal_value,
    reverse=True,
)
for ticket, decision in ranked:
    print(f"{decision.expected_ordinal_value:.2f}  {ticket}")
