# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choose a discussion draft from stated preferences, then put it to a vote."""

import json
from collections.abc import Mapping, Sequence

import wyat
from examples.use_cases.common import load_model


def propose_statement(
    model: wyat.Model, opinions: Sequence[str], drafts: Mapping[str, str]
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.choice(
            "Which draft best accommodates the group's stated preferences?",
            drafts,
            context={"opinions": list(opinions)},
        )
    )
    return {
        "draft_id": decision.selected,
        "draft": drafts[decision.selected],
        "next_step": "collect_group_votes",
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return propose_statement(
        model,
        [
            "Léa souhaite réduire le bruit des voitures après 23 h.",
            "Malik veut pouvoir rentrer chez lui à vélo la nuit.",
            "Chloé demande que les secours puissent toujours traverser le quartier.",
        ],
        {
            "ban_all": "Interdire toute circulation dans le quartier, jour et nuit.",
            "night_access": (
                "Fermer le quartier aux voitures de 23 h à 6 h, en maintenant "
                "le passage des vélos et des secours."
            ),
            "unchanged": "Maintenir la circulation automobile sans restriction.",
        },
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
