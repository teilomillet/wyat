# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""MCP: give any agent (chat clients, IDE assistants, custom agents) Wyat as a tool.

The Model Context Protocol lets an agent call tools on a server. These tools
expose typed decisions: the agent hands Wyat a situation and its own options,
and gets back a ranked, scored answer instead of guessing in free text.

    uv run --with 'mcp>=2.3' python 06_mcp_server.py --demo              # try the tools
    uv run --with 'mcp>=2.3' python 06_mcp_server.py --model DIR --serve # stdio server

To connect a client, register the second command as a stdio MCP server.
"""

import sys
from typing import Any

from common import load_model, missing

import wyat

try:
    from mcp.server.mcpserver import MCPServer
except ImportError as error:
    missing("mcp>=2.3", error)

model = load_model(__doc__)
server = MCPServer("wyat-decisions")


@server.tool()
def choose(question: str, options: dict[str, str], context: str = "") -> dict[str, Any]:
    """Pick one option. `options` maps option IDs to descriptions."""
    decision = model.decide(wyat.Request.choice(question, options, context=context))
    return {
        "selected": decision.selected,
        "probabilities": decision.probabilities_by_id,
        "margin": decision.probability_margin,
    }


@server.tool()
def check(question: str, context: str) -> dict[str, Any]:
    """Answer a yes/no question about the context with a probability."""
    decision = model.decide(wyat.Request.noul(question, context=context))
    return {"probability_true": decision.probability_true}


@server.tool()
def rate(question: str, levels: list[str], context: str) -> dict[str, Any]:
    """Place the context on an ordered scale; levels go from lowest to highest."""
    decision = model.decide(wyat.Request.score(question, levels, context=context))
    return {"expected_level": decision.expected_ordinal_value, "of": len(levels) - 1}


if "--serve" in sys.argv:
    server.run("stdio")
else:
    # The decorated tools stay plain functions: what an agent's calls return.
    print(choose("Which team?", {"billing": "Refunds.", "tech": "Bugs."}, "refund"))
    print(check("Does the message mention a refund?", "I want a refund."))
    print(rate("How urgent?", ["Can wait.", "Today.", "Now."], "App down for all."))
