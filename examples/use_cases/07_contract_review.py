# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Several questions about one document in a single call.

``model.predict`` takes one state and many questions, of any task type, and
returns one typed answer per question. wyat-1 800M still reads the document
once per question, so this keeps the code short; it is not a speedup.
"""

from common import load_model

import wyat

FOUND = {
    "yes": "The contract says so.",
    "no": "The contract says the opposite.",
    "not_mentioned": "The contract does not say.",
}

model = load_model(__doc__)

contract = (
    "4. The supplier may share data with its employees who need it. "
    "7. Either party may end this agreement with 30 days' notice."
)
answers: wyat.Answers = model.predict(
    {
        "state": contract,
        "questions": {
            "share_with_employees": {
                "type": "choice",
                "instructions": "May the supplier share data with employees?",
                "criteria": FOUND,
            },
            "share_with_partners": {
                "type": "choice",
                "instructions": "May the supplier share data with partners?",
                "criteria": FOUND,
            },
            "easy_exit": {
                "type": "noul",
                "instructions": "Can we leave with less than 60 days' notice?",
            },
            "risk": {
                "type": "score",
                "instructions": "How risky is this contract for us?",
                "criteria": ["Low.", "Medium.", "High."],
            },
        },
    }
)
# Each question type has a typed accessor on Answers: .choice(), .noul(), .score().
for key in ("share_with_employees", "share_with_partners"):
    print(f"{key}: {answers.choice(key).choice}")
print(f"easy_exit: p={answers.noul('easy_exit').probability_true:.2f}")
print(f"risk: {answers.score('risk').expected_ordinal_value:.2f} of 2")
