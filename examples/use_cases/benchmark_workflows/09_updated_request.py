# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Rank fulfillment proposals against the latest request, preserving other fields."""

import json
from collections.abc import Mapping

import wyat
from examples.use_cases.common import load_model

RUBRIC = [
    "0 — Aucune des trois exigences actuelles n'est respectée.",
    "1 — Une seule des trois exigences actuelles est respectée.",
    "2 — Deux des trois exigences actuelles sont respectées.",
    "3 — Les trois exigences actuelles sont respectées, "
    "mais la préférence facultative n'est pas satisfaite.",
    "4 — Les trois exigences actuelles et la préférence facultative sont respectées.",
]


def rank_proposals(
    model: wyat.Model, request_history: str, proposals: Mapping[str, str]
) -> dict[str, object]:
    if len(proposals) < 2:
        raise ValueError("supply at least two proposals to compare")
    requests = [
        wyat.Request.score(
            "Quel est le degré de conformité de la proposition à la demande actuelle ?",
            RUBRIC,
            context={"request_history": request_history, "proposal": proposal},
        )
        for proposal in proposals.values()
    ]
    decisions = model.decide_many(requests)
    scores = {
        key: decision.expected_ordinal_value
        for key, decision in zip(proposals, decisions, strict=True)
    }
    ranking = sorted(scores, key=lambda key: scores[key], reverse=True)
    tied = len(ranking) > 1 and scores[ranking[0]] == scores[ranking[1]]
    return {
        "proposal_id": None if tied else ranking[0],
        "next_step": "review_tie" if tied else "review_fulfillment_proposal",
        "ranking": ranking,
        "expected_scores": scores,
        "decisions": [decision.to_dict() for decision in decisions],
    }


def run(model: wyat.Model) -> dict[str, object]:
    return rank_proposals(
        model,
        "La commande prévoyait dix repas végétariens à retirer à 12 h 30. "
        "La cliente corrige uniquement la quantité à quatorze. Elle confirme "
        "que végétarien et retrait à 12 h 30 restent obligatoires, et préfère "
        "un plateau réutilisable si possible. La dernière correction explicite "
        "remplace seulement le champ qu'elle vise.",
        {
            "old_quantity": (
                "Dix repas végétariens, retrait à 12 h 30, plateau jetable."
            ),
            "updated_quantity": (
                "Quatorze repas végétariens, retrait à 12 h 30, plateau jetable."
            ),
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
