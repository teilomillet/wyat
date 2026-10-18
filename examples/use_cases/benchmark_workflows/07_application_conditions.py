# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Check supplied prerequisites and send incomplete evidence to the right next step."""

import json

import wyat
from examples.use_cases.common import load_model

RUBRIC = [
    "0 — Au moins une condition obligatoire est établie comme non remplie.",
    "1 — Aucune condition obligatoire n'est établie comme non remplie, "
    "mais au moins une reste à vérifier.",
    "2 — Toutes les conditions obligatoires sont établies comme remplies.",
]
NEXT_STEPS = {
    "0": "review_unmet_condition",
    "1": "request_missing_evidence",
    "2": "ready_for_final_review",
}


def check_application(
    model: wyat.Model, application: str, rules: str
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.score(
            "Quel est le niveau d'adéquation établie du dossier aux conditions ?",
            RUBRIC,
            context={"application": application, "rules": rules},
        )
    )
    return {
        "level": int(decision.selected),
        "next_step": NEXT_STEPS[decision.selected],
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return check_application(
        model,
        "Lina demande une place à l'atelier de céramique avancée. Le registre "
        "validé confirme qu'elle est majeure et sa cotisation annuelle à jour. "
        "L'attestation de son initiation à la sécurité n'a pas été retrouvée "
        "et le registre de cette initiation n'a pas encore été consulté.",
        "L'accès exige d'être majeur, d'avoir une cotisation à jour et d'avoir "
        "terminé l'initiation à la sécurité. Cette initiation doit être établie "
        "par une attestation ou par le registre validé.",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
