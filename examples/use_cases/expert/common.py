# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Shared by the expert examples: a local model, or a transparent stand-in."""

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import wyat


def words(text: str) -> set[str]:
    """Lowercase stems (first four letters) of words longer than three letters."""
    return {w[:4] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}


MAX_CONTEXT_CHARS = 100_000  # like a real model's input limit


class OverlapScorer:
    """Word overlap between each option and the input. NOT a model.

    It only makes the demo pipelines produce varied numbers, so thresholds,
    rankings and fallbacks visibly do something. Its decisions mean nothing.
    """

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        rows = []
        for request in requests:
            if len(request.context) > MAX_CONTEXT_CHARS:
                raise wyat.ModelInputError("context too long", reason="input-length")
            seen = words(request.context)
            if request.task == "noul":
                asked = words(request.query)
                share = len(asked & words(request.context)) / max(len(asked), 1)
                true = 4.0 * share - 1.0
                rows.append(
                    tuple(true if c.id == "true" else 0.0 for c in request.candidates)
                )
            elif request.task == "score" and "\n" in request.context:
                # Overlap of the first context line (e.g. the question) with the
                # rest (e.g. the passage), mapped onto the rubric levels.
                first, _, rest = request.context.partition("\n")
                asked = words(first)
                share = len(asked & words(rest)) / max(len(asked), 1)
                top = share * (len(request.candidates) - 1)
                rows.append(
                    tuple(-((i - top) ** 2) for i in range(len(request.candidates)))
                )
            else:
                rows.append(
                    tuple(float(len(words(c.text) & seen)) for c in request.candidates)
                )
        return tuple(rows)


def load_model(description: str | None) -> wyat.Model:
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=Path,
        help="Model name (teilomillet/wyat-1-800m) or local model directory",
    )
    source.add_argument(
        "--demo", action="store_true", help="Word-overlap stand-in, no model"
    )
    parser.add_argument(
        "--device",
        help="Torch device, for example mps on Apple Silicon (default: cpu)",
    )
    arguments = parser.parse_args()
    if arguments.demo:
        print("Demo mode: word-overlap stand-in, not a model.\n")
        return wyat.Model(OverlapScorer(), model_id="demo-overlap")
    return wyat.load(arguments.model, device=arguments.device)
