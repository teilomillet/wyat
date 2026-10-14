# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""LangGraph: Wyat decides the edges of a support workflow graph.

Conditional edges in LangGraph are plain functions that return the next node.
Here they call Wyat: a Choice picks the branch after intake, and a Noul check
decides whether the drafted reply can be sent or needs a human. Nodes stay
ordinary Python (or LLM calls); every edge is a scored, logged decision.

    uv run --with 'langgraph>=1.2' python 03_langgraph_workflow.py --demo
"""

from typing import TypedDict

from common import load_model, missing

import wyat

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as error:
    missing("langgraph>=1.2", error)


class Ticket(TypedDict, total=False):
    message: str
    reply: str
    path: list[str]


model = load_model(__doc__)


def intake(state: Ticket) -> Ticket:
    return {"path": ["intake"]}


def choose_branch(state: Ticket) -> str:
    decision = model.decide(
        wyat.Request.choice(
            "Which workflow handles this message?",
            {
                "refund": "Refund or money back request.",
                "faq": "General question about shipping, hours or products.",
                "human": "Complaint, threat, or anything unclear.",
            },
            context=state.get("message", ""),
        )
    )
    return decision.selected if decision.probability_margin >= 0.1 else "human"


def refund(state: Ticket) -> Ticket:
    return {
        "reply": "Your refund was started.",
        "path": state.get("path", []) + ["refund"],
    }


def faq(state: Ticket) -> Ticket:
    reply = "Shipping takes 3 to 5 working days."
    return {"reply": reply, "path": state.get("path", []) + ["faq"]}


def human(state: Ticket) -> Ticket:
    return {
        "reply": "A colleague will answer you.",
        "path": state.get("path", []) + ["human"],
    }


def reply_is_safe(state: Ticket) -> str:
    check = model.decide(
        wyat.Request.noul(
            "Does the reply address the customer message?",
            context={"message": state.get("message"), "reply": state.get("reply")},
        )
    )
    return "send" if check.probability_true >= 0.5 else "human"


graph = StateGraph(Ticket)
for name, node in {
    "intake": intake,
    "refund": refund,
    "faq": faq,
    "human": human,
}.items():
    graph.add_node(name, node)
graph.add_edge(START, "intake")
graph.add_conditional_edges("intake", choose_branch, ["refund", "faq", "human"])
for automated in ("refund", "faq"):
    graph.add_conditional_edges(
        automated, reply_is_safe, {"send": END, "human": "human"}
    )
graph.add_edge("human", END)
app = graph.compile()

for message in [
    "I want my money back, refund please",
    "How long does shipping take?",
    "This is the worst service ever",
]:
    result = app.invoke({"message": message})
    print(
        f"{message!r}\n  path={' -> '.join(result['path'])}  reply={result['reply']!r}"
    )
