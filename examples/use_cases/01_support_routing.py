# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choice: route a customer message to a support queue.

You own the menu. Add an explicit "none of these" option so the model is
never forced to guess a queue that does not fit.
"""

from common import load_model

import wyat

QUEUES = {
    "billing": "Payments, invoices, refunds.",
    "technical": "Login problems, bugs, errors in the app.",
    "delivery": "Shipping, tracking, missing parcels.",
    "other": "None of the above: send to a human.",
}

model = load_model(__doc__)

for message in [
    "I was charged twice for my order.",
    "The app crashes when I open my profile.",
    "Can you recommend a good restaurant nearby?",
]:
    request = wyat.Request.choice(
        "Which support queue should handle this message?", QUEUES, context=message
    )
    decision = model.decide(request)
    print(f"{message!r} -> {decision.selected}")
