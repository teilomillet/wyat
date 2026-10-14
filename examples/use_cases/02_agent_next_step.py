# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choice: decide what an agent does next before it calls a tool.

The model only chooses. Your code checks the choice and runs the action.
"""

from common import load_model

import wyat

STEPS = {
    "call_tool": "Call book_table now: every required field is known.",
    "ask_user": "Ask the user for the missing information first.",
    "decline": "Say that no available tool can do this.",
}

model = load_model(__doc__)

tool = {"name": "book_table", "required": ["restaurant", "time", "party_size"]}
for message in [
    "Book a table for 4 at Chez Lou at 8pm.",
    "Book me a table at Chez Lou tonight.",
    "Order me a taxi to the airport.",
]:
    request = wyat.Request.choice(
        "What should the assistant do next?",
        STEPS,
        context={"tool": tool, "user": message},
    )
    print(f"{message!r} -> {model.decide(request).selected}")
