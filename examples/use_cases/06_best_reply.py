# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choice: pick the best reply among drafts written elsewhere.

Generate a few drafts with any LLM or template, then let the model choose.
The ranking and probabilities show how close the alternatives were.
"""

from common import load_model

import wyat

model = load_model(__doc__)

customer = "My parcel is a week late and nobody answers my emails."
drafts = [
    wyat.Candidate("a", "Sorry for the delay. I checked: it arrives Friday."),
    wyat.Candidate("b", "Please read our shipping FAQ."),
    wyat.Candidate("c", "Delays happen. Please be patient."),
]
decision = model.decide(
    wyat.Request(
        query="Which reply best resolves the customer's problem?",
        context=customer,
        candidates=drafts,
    )
)
print("Best draft:", decision.selected)
for draft_id in decision.ranking:
    print(f"  {draft_id}: p={decision.probabilities_by_id[draft_id]:.2f}")
