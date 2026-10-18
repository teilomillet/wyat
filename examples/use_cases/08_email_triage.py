# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Noul signals plus your own rule: explainable email triage.

Ask small yes/no questions, then combine them in code you can read and
audit. The model measures; your rule decides.
"""

from common import load_model

import wyat

model = load_model(__doc__)

email = {
    "from": "security@paypa1-support.com",
    "subject": "Urgent: verify your account within 24h",
    "body": "Click here to avoid suspension.",
}
signals = {
    "urgency": "Does the email pressure the reader to act fast?",
    "lookalike_sender": "Does the sender imitate a known brand's domain?",
    "asks_for_login": "Does the email ask the reader to log in or click a link?",
}
answers: wyat.Answers = model.predict(
    {
        "state": email,
        "questions": {
            key: {"type": "noul", "instructions": text} for key, text in signals.items()
        },
    }
)
p = {key: answers.noul(key).probability_true for key in signals}
for key, value in p.items():
    print(f"{key}: {value:.2f}")

# Your policy, in plain code:
risky = sum(value >= 0.5 for value in p.values())
print("Action:", "quarantine" if risky >= 2 else "deliver")
