# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Noul: pick a reply draft by checking each one against what it must do.

Generate a few drafts with any LLM or template, then check every draft
against one concrete requirement and send the draft that passes most clearly.
If none passes, a person writes the reply.

Name the requirement. A vague question such as "Which reply best resolves the
problem?" made wyat-1 800M pick "Delays happen. Please be patient." here;
asking whether each draft gives a concrete answer picks the right one.
"""

from common import load_model

import wyat

model = load_model(__doc__)

customer = "My parcel is a week late and nobody answers my emails."
requirement = "Does this reply give the customer a concrete answer about their parcel?"
drafts = {
    "a": "Sorry for the delay. I checked: it arrives Friday.",
    "b": "Please read our shipping FAQ.",
    "c": "Delays happen. Please be patient.",
}
checks = model.decide_many(
    [
        wyat.Request.noul(requirement, context=f"Customer: {customer}\nReply: {draft}")
        for draft in drafts.values()
    ]
)
pass_probability = {
    draft_id: check.probability_true
    for draft_id, check in zip(drafts, checks, strict=True)
}
for draft_id, probability in pass_probability.items():
    print(f"  {draft_id}: p(passes)={probability:.2f}  {drafts[draft_id]}")

best = max(pass_probability, key=pass_probability.__getitem__)
if pass_probability[best] >= 0.5:
    print("Send draft:", best)
else:
    print("No draft passes: a person writes the reply.")
