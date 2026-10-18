# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Route over a large intent taxonomy in two short steps instead of one long menu.

Real support catalogs have 50 to 200 intents. Two stages keep each menu short
and the descriptions distinct: first the domain, then the intent inside it.
Each stage has its own fallback, so the system says "I'm not sure which
billing intent" instead of guessing a wrong one. The second stage for every
message runs as one batch. Give the catch-all a description too: with a bare
"None of these domains.", wyat-1 800M sent "a statement for my landlord" there.
"""

from common import load_model

import wyat

TAXONOMY = {
    "cards": {
        "card_lost": "Card lost or stolen; block it.",
        "card_declined": "A card payment was declined.",
        "card_delivery": "When will my new card arrive?",
        "card_pin": "Forgot or want to change the PIN.",
    },
    "transfers": {
        "transfer_pending": "A transfer is pending or not received.",
        "transfer_failed": "A transfer failed or was returned.",
        "transfer_fee": "Question about transfer fees.",
    },
    "account": {
        "account_close": "Close my account.",
        "account_verify": "Identity verification or documents.",
        "account_statement": "Get a statement or proof of balance.",
    },
}
DOMAINS = {
    "cards": "Debit or credit cards: loss, payments, delivery, PIN.",
    "transfers": "Transfers: sending or receiving money between accounts.",
    "account": (
        "Account: opening, closing, identity checks, statements, proof of balance."
    ),
    "other": "Other: not about the three domains above.",
}
MIN_MARGIN = 0.10


def clear(decision: wyat.Decision) -> bool:
    return decision.probability_margin >= MIN_MARGIN


model = load_model(__doc__)
messages = [
    "I lost my card on the train, please block it",
    "My transfer to my sister is still pending since Monday",
    "How do I get a statement for my landlord?",
    "Where can I buy concert tickets?",
]

# Stage 1: one batched call picks the domain for every message.
domains = model.decide_many(
    [
        wyat.Request.choice("Which banking domain is this about?", DOMAINS, context=m)
        for m in messages
    ]
)

# Stage 2: only messages with a clear, known domain go to their intent menu.
routable = [
    (message, domain.selected)
    for message, domain in zip(messages, domains, strict=True)
    if domain.selected in TAXONOMY and clear(domain)
]
intents = model.decide_many(
    [
        wyat.Request.choice(
            f"Which {domain} request is this?",
            {**TAXONOMY[domain], "unsure": f"A {domain} request not listed here."},
            context=message,
        )
        for message, domain in routable
    ]
)
by_message = {
    message: (domain, intent)
    for (message, domain), intent in zip(routable, intents, strict=True)
}

for message, domain in zip(messages, domains, strict=True):
    if message not in by_message:
        print(f"{message!r}\n  -> human (domain {domain.selected!r} not clear)")
        continue
    name, intent = by_message[message]
    if intent.selected == "unsure" or not clear(intent):
        print(f"{message!r}\n  -> {name} specialist (intent not clear)")
    else:
        print(f"{message!r}\n  -> {intent.selected}")
