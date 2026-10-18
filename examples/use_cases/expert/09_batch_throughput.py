# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Process a large backlog: dedupe, chunk, isolate failures.

Three habits for bulk jobs (tagging a ticket archive, labelling a catalog):
1. Load the model once and reuse it.
2. Identical inputs are scored once (requests are hashable values).
3. Score in chunks with ``decide_many(..., errors="return")``. An input the
   model refuses (too long, for example) gets its ``ModelInputError`` in its
   slot, so one bad input cannot sink the job, and it is recorded rather than
   silently dropped. Backend failures still raise: retry those at job level.

wyat-1 800M checks every input of a chunk before scoring any, so a refusal
costs no model work. It then scores the requests one after another: time
grows with the number of unique inputs, and deduping is where the saving is.
"""

import time
from collections.abc import Sequence

from common import load_model

import wyat

CHUNK = 32
TAGS = {
    "bug": "Bug: something is broken or stopped working.",
    "feature": "Feature request: please add something new, like a mode.",
    "question": "Question: how to invite, configure or use something.",
    "praise": "Praise: they love or like the product.",
}


def score_all(
    model: wyat.Model, requests: Sequence[wyat.Request]
) -> tuple[dict[wyat.Request, wyat.Decision], dict[wyat.Request, str]]:
    unique = list(dict.fromkeys(requests))  # dedupe, keep first-seen order
    done: dict[wyat.Request, wyat.Decision] = {}
    refused: dict[wyat.Request, str] = {}
    for start in range(0, len(unique), CHUNK):
        chunk = unique[start : start + CHUNK]
        for request, result in zip(
            chunk, model.decide_many(chunk, errors="return"), strict=True
        ):
            if isinstance(result, wyat.ModelInputError):
                refused[request] = result.reason
            else:
                done[request] = result
    return done, refused


model = load_model(__doc__)
feedback = [
    "The export button is broken since the update",
    "Please add dark mode",
    "How do I invite a teammate?",
    "Love the new dashboard!",
] * 50  # 200 rows, only 4 distinct
feedback.append("Crash log: " + "ERROR worker restarted. " * 20_000)  # pasted, huge
requests = [
    wyat.Request.choice("Which tag fits this feedback?", TAGS, context=text)
    for text in feedback
]

start = time.perf_counter()
done, refused = score_all(model, requests)
elapsed = time.perf_counter() - start

tags = [done[r].selected for r in requests if r in done]
print(f"{len(requests)} rows, {len(done)} scored once each, {len(refused)} refused")
print("refused:", sorted(set(refused.values())))
print(f"{elapsed * 1000:.1f} ms total")
print("tag counts:", {tag: tags.count(tag) for tag in TAGS})
