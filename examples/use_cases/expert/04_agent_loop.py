# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""A tool-using agent where Wyat picks each step and code enforces the rules.

Each turn, the model sees the goal and everything observed so far, and picks
one step: a tool (described by its JSON schema), ask the user, or finish.
The loop, not the model, owns safety: only listed tools run, irreversible
tools wait for a person, a tool that already ran is no longer offered, and
there is a hard step limit. Arguments come from your own extraction code; the
model only decides what happens next.

Offering only the steps still open also helps the model stop: with every tool
still on the menu, wyat-1 800M picked lookup_order again instead of finish.
"""

from collections.abc import Callable

from common import load_model

import wyat

TOOLS: dict[str, tuple[dict[str, object], Callable[[], str]]] = {
    "lookup_order": (
        {
            "name": "lookup_order",
            "description": "Find an order and its delivery status by order number.",
            "parameters": {"order_id": "string"},
        },
        lambda: "Order 1042: shipped Monday, carrier says delayed in transit.",
    ),
    "issue_refund": (
        {
            "name": "issue_refund",
            "description": "Refund a delivered or lost order. Irreversible.",
            "parameters": {"order_id": "string", "amount": "number"},
        },
        lambda: "Refund of 49.90 EUR issued.",
    ),
    "open_carrier_claim": (
        {
            "name": "open_carrier_claim",
            "description": "Open a carrier investigation for a delayed parcel.",
            "parameters": {"order_id": "string"},
        },
        lambda: "Claim C-77 opened, answer within 48h.",
    ),
}
CONTROL = {
    "ask_user": wyat.Candidate("ask_user", "Ask for missing information."),
    "finish": wyat.Candidate(
        "finish", "Finish: everything needed is known; send the final reply."
    ),
}
NEEDS_APPROVAL = {"issue_refund"}  # irreversible: a person confirms first
MAX_STEPS = 5


def next_step(model: wyat.Model, goal: str, history: list[dict[str, str]]) -> str:
    done = {item["step"] for item in history}
    candidates = [
        wyat.Candidate.from_schema(name, schema)
        for name, (schema, _) in TOOLS.items()
        if name not in done
    ]
    request = wyat.Request(
        query="Choose the next step that moves the goal forward.",
        context={"goal": goal, "history": history},
        candidates=[*candidates, *CONTROL.values()],
    )
    return model.decide(request).selected


model = load_model(__doc__)
goal = "Customer: my order 1042 has not arrived after 10 days, what is happening?"
history: list[dict[str, str]] = []
for step in range(1, MAX_STEPS + 1):
    choice = next_step(model, goal, history)
    if choice in CONTROL:
        print(f"step {step}: {choice} -> stop")
        break
    if choice in NEEDS_APPROVAL:
        print(f"step {step}: {choice} proposed -> waiting for human approval")
        break
    _, run = TOOLS[choice]  # only listed tools can be reached here
    observation = run()
    history.append({"step": choice, "observation": observation})
    print(f"step {step}: {choice} -> {observation}")
else:
    print(f"stopped after {MAX_STEPS} steps -> hand to a human")
