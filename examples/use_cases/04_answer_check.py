# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Noul: check that a generated answer is supported by its source.

Noul gives one probability for a yes/no question. The threshold is your
policy, not the model's: pick it from your own labelled examples.
"""

from common import load_model

import wyat

THRESHOLD = 0.5  # tune on your own data

model = load_model(__doc__)

source = "The store opens at 9am and closes at 7pm. Parking is paid."
for answer in [
    "The store opens at 9am.",
    "The store opens at 9am and parking is free.",
]:
    request = wyat.Request.noul(
        "Is every claim in the answer supported by the source?",
        context={"source": source, "answer": answer},
    )
    p_supported = model.decide(request).probability_true
    action = "show" if p_supported >= THRESHOLD else "send for review"
    print(f"p={p_supported:.2f} -> {action}: {answer!r}")
