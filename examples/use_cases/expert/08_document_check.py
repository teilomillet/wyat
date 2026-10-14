# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Onboarding document check: many typed questions feed a readable rule engine.

OCR or a form gives you structured text. One ``predict_many`` call asks
everything about every upload at once: the document type (Choice), whether
each required field is present (Noul, one per field), and how readable it is
(Score). Plain rules then decide: accept, or ask for a new upload naming
exactly what is missing.
The questions come from the requirements table, so adding a document type or
a field is a data change, not new code.
"""

from common import load_model

import wyat

REQUIRED = {
    "id_card": ["full name", "date of birth", "expiry date", "document number"],
    "proof_of_address": ["full name", "postal address", "issue date"],
}
TYPES = {
    "id_card": "A national identity card or passport.",
    "proof_of_address": "A utility bill, bank statement or tax notice.",
    "other": "Something else.",
}
PRESENT_ABOVE = 0.7
READABLE_AT_LEAST = 1.0  # at least "partly readable" on the 0..2 scale


def questions(expected: str) -> dict[str, dict[str, object]]:
    asked: dict[str, dict[str, object]] = {
        "type": {
            "type": "choice",
            "instructions": "What document is this?",
            "criteria": TYPES,
        },
        "legibility": {
            "type": "score",
            "instructions": "How readable is the extracted text?",
            "criteria": ["Unreadable.", "Partly readable.", "Clear."],
        },
    }
    for field in REQUIRED[expected]:
        asked[field] = {"type": "noul", "instructions": f"Does it show the {field}?"}
    return asked


def check(
    expected: str,
    answers: dict[str, wyat.ChoiceAnswer | wyat.ScoreAnswer | wyat.NoulAnswer],
) -> str:
    kind, legibility = answers["type"], answers["legibility"]
    assert isinstance(kind, wyat.ChoiceAnswer)
    assert isinstance(legibility, wyat.ScoreAnswer)
    missing = []
    for field in REQUIRED[expected]:
        present = answers[field]
        assert isinstance(present, wyat.NoulAnswer)
        if present.probability_true < PRESENT_ABOVE:
            missing.append(field)
    if kind.choice != expected:
        return f"ask again: this looks like {kind.choice!r}, expected {expected!r}"
    if legibility.expected_ordinal_value < READABLE_AT_LEAST:
        return "ask again: photo not readable enough"
    if missing:
        return "ask again: missing " + ", ".join(missing)
    return "accept"


model = load_model(__doc__)
uploads = [
    (
        "id_card",
        {
            "text": "IDENTITY CARD. Full name: Lea Martin. Date of birth: 1990-04-02. "
            "Document number: X12345. Expiry date: 2031-01-01."
        },
    ),
    (
        "proof_of_address",
        {"text": "Electricity bill. Full name: Lea Martin. Amount due: 54 EUR."},
    ),
]
records = [
    {"state": extracted, "questions": questions(expected)}
    for expected, extracted in uploads
]
for (expected, _), answers in zip(uploads, model.predict_many(records), strict=True):
    print(f"{expected}: {check(expected, answers)}")
