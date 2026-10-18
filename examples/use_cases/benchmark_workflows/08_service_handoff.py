# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Check a destination team's remit before handing it a refund request."""

import json

import wyat
from examples.use_cases.common import load_model


def prepare_handoff(
    model: wyat.Model, ticket: str, payment_record: str, routing_rules: str
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.noul(
            "Le service Facturation du réseau est compétent pour traiter "
            "maintenant le remboursement demandé.",
            context={
                "ticket": ticket,
                "validated_payment_record": payment_record,
                "routing_rules": routing_rules,
            },
        )
    )
    probability = decision.probability_true
    # A rejected destination does not establish a different team's ownership.
    # Illustrative threshold; validate it on your labeled routing tickets.
    destination = "network_billing" if probability >= 0.8 else "routing_review"
    return {
        "destination": destination,
        "probability_network_billing": probability,
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return prepare_handoff(
        model,
        "Émile demande le remboursement d'un second débit pour un seul titre "
        "de transport. Les deux débits sont confirmés.",
        "Le titre a été vendu et encaissé directement par le réseau municipal.",
        "Facturation traite les remboursements des paiements encaissés "
        "directement par le réseau. Si une agence partenaire a vendu et encaissé "
        "le titre, seul le service Clients de cette agence traite le remboursement.",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
