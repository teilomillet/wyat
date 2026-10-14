# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""LangChain: Wyat as the router of an LCEL chain.

Wyat becomes a ``RunnableLambda`` that adds a ``route`` key; a
``RunnableBranch`` sends the input to the matching sub-chain. Routing is a
fast, cheap, logged decision instead of an extra LLM call that returns text
you then have to parse. The sub-chains here use a fake LLM; swap in any
LangChain chat model.

    uv run --with 'langchain-core>=1.6' python 01_langchain_router.py --demo
"""

from typing import Any

from common import load_model, missing

import wyat

try:
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda
except ImportError as error:
    missing("langchain-core>=1.6", error)

ROUTES = {
    "billing": "Billing: charges, refunds, invoices.",
    "technical": "Technical: errors, crashes, login problems.",
    "smalltalk": "Small talk or anything else.",
}

model = load_model(__doc__)


def route(inputs: dict[str, Any]) -> dict[str, Any]:
    decision = model.decide(
        wyat.Request.choice(
            "Which team handles this?", ROUTES, context=inputs["question"]
        )
    )
    clear = decision.probability_margin >= 0.1  # unclear routes go to the default
    return {**inputs, "route": decision.selected if clear else "general"}


def team(name: str) -> Runnable[dict[str, Any], str]:
    llm = FakeListLLM(responses=[f"[{name} assistant] I can help with that."])
    return PromptTemplate.from_template("You are the {route} team.\n{question}") | llm


chain = RunnableLambda(route) | RunnableBranch(
    (lambda x: x["route"] == "billing", team("billing")),
    (lambda x: x["route"] == "technical", team("technical")),
    team("general"),
)

for question in [
    "I was charged twice on my invoice",
    "The app shows an error at login",
    "What a nice day!",
]:
    print(f"{question!r} -> {chain.invoke({'question': question})}")
