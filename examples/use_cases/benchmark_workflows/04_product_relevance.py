# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Separate exact matches, alternatives and accessories in a shopping interface."""

import json

import wyat
from examples.use_cases.common import load_model

RELATIONS = {
    "exact": "The product itself matches the item requested in the search.",
    "substitute": "An alternative product that can replace the requested item.",
    "complement": "An accessory used with the requested item; cannot replace it.",
    "irrelevant": "Unrelated to the requested item.",
}
SECTIONS = {
    "exact": "results",
    "substitute": "alternatives",
    "complement": "accessories",
    "irrelevant": "excluded",
}


def place_product(
    model: wyat.Model, query: str, product_id: str, description: str
) -> dict[str, object]:
    decision = model.decide(
        wyat.Request.choice(
            "How does this product relate to the shopping query?",
            RELATIONS,
            context={"search": query, "product": description},
        )
    )
    return {
        "product_id": product_id,
        "relation": decision.selected,
        "section": SECTIONS[decision.selected],
        "decision": decision.to_dict(),
    }


def run(model: wyat.Model) -> dict[str, object]:
    return place_product(
        model,
        "appareil photo Lumina X2",
        "battery-002",
        "Batterie de remplacement compatible avec l'appareil photo Lumina X2. "
        "Appareil photo vendu séparément.",
    )


if __name__ == "__main__":
    print(json.dumps(run(load_model(__doc__)), ensure_ascii=False, indent=2))
