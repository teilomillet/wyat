# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Use an impact rubric to prepare an incident for the appropriate work queue."""

import json

import wyat
from examples.use_cases.common import load_model

RUBRIC = [
    "0 — Demande d'information ; aucune opération de travail n'est perturbée.",
    "1 — Une opération est perturbée, mais une solution de remplacement "
    "permet de l'accomplir.",
    "2 — Une personne est bloquée sans solution de remplacement.",
    "3 — Une équipe ou un site entier est bloqué sans solution de remplacement ; "
    "les autres équipes ou sites fonctionnent normalement.",
    "4 — Plusieurs équipes ou sites indépendants sont bloqués "
    "sans solution de remplacement.",
]
QUEUES = {
    "0": "information",
    "1": "routine_support",
    "2": "individual_blocker",
    "3": "team_incident",
    "4": "multi_site_incident",
}


def prioritize_incident(model: wyat.Model, incident: str) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.score(
            "Quel niveau de priorité d'intervention humaine ce dossier appelle-t-il ?",
            RUBRIC,
            context=incident,
        )
    )
    return {
        "level": int(decision.selected),
        "queue": QUEUES[decision.selected],
        "expected_impact": decision.expected_ordinal_value,
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return prioritize_incident(
        model,
        "L'imprimante d'accueil ne produit plus aucun badge. Les badges imprimés "
        "sont le seul moyen autorisé d'admettre les visiteurs. Toute l'équipe "
        "d'accueil est empêchée d'accomplir cette opération. Aucun autre appareil "
        "ni autre procédure ne peut la remplacer aujourd'hui. Les autres équipes "
        "et les autres sites fonctionnent normalement.",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
