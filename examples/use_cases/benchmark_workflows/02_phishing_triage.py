# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Put a suspicious message in a review queue before anyone follows its link."""

import json

import wyat
from examples.use_cases.common import load_model


def triage_email(
    model: wyat.Model, message: str, usual_domain: str
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.noul(
            "Does this message show signs of phishing?",
            context={"usual_domain": usual_domain, "message": message},
        )
    )
    probability = decision.probability_true
    # Illustrative application policy; choose thresholds on your labeled emails.
    queue = "suspected_phishing" if probability >= 0.8 else "standard_review"
    return {
        "queue": queue,
        "probability_phishing": probability,
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return triage_email(
        model,
        "Votre compte sera bloqué dans 30 minutes. Confirmez immédiatement "
        "votre mot de passe et votre code SMS sur banque-azur-verification.example.",
        "banque-azur.example",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
