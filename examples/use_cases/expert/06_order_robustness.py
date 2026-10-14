# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Check that a decision does not depend on the order of the options.

A decision model can lean toward options listed first or last. For decisions
that matter, ask again with the options rotated so each option appears once in
every position (one batched call), average the probabilities by option ID, and
treat a changing winner as "unsure".
IDs make this safe: the answer is the same option whatever its position.
"""

from collections import Counter

from common import load_model

import wyat

model = load_model(__doc__)
options = [
    wyat.Candidate("approve", "Approve the refund: the item arrived broken."),
    wyat.Candidate("partial", "Offer a partial refund and keep the item."),
    wyat.Candidate("reject", "Reject: the return window has passed."),
    wyat.Candidate("escalate", "Escalate to a supervisor."),
]
context = "Customer sends a photo of a broken lamp, delivered 12 days ago."

orderings = [options[i:] + options[:i] for i in range(len(options))]
decisions = model.decide_many(
    [
        wyat.Request("Which resolution fits the case?", order, context=context)
        for order in orderings
    ]
)

winners = Counter(decision.selected for decision in decisions)
average = {
    option.id: sum(d.probabilities_by_id[option.id] for d in decisions) / len(decisions)
    for option in options
}
for order, decision in zip(orderings, decisions, strict=True):
    print(f"order {[o.id for o in order]} -> {decision.selected}")
print("averaged:", {key: round(value, 2) for key, value in average.items()})

best = max(average, key=average.__getitem__)
if winners[best] == len(decisions):
    print(f"stable -> {best}")
else:
    print(f"winner changed with order {dict(winners)} -> treat as unsure")
