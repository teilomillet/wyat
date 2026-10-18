# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Choose a booking action with a local model or an explicit no-weights demo."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import wyat


class DemoScorer:
    """Give cancellation a fixed score to demonstrate the API without weights."""

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                2.0 if candidate.id == "cancel" else 0.0
                for candidate in request.candidates
            )
            for request in requests
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model", type=Path, help="teilomillet/wyat-1-800m, or a local model directory"
    )
    source.add_argument(
        "--demo", action="store_true", help="Try fixed scores without weights"
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "mlx"),
        default="torch",
        help="Runtime for --model",
    )
    arguments = parser.parse_args()

    # Only model construction changes between demo mode and trained inference.
    if arguments.demo:
        model = wyat.Model(DemoScorer(), model_id="demo-fixed-scores")
        print("Demo mode: fixed scores, no trained model.")
    else:
        model = wyat.load(
            cast(Path, arguments.model),
            backend=cast(Literal["torch", "mlx"], arguments.backend),
        )

    request = wyat.Request(
        query="Choose the requested booking action.",
        context="Please cancel my booking.",
        candidates=[
            wyat.Candidate(id="create", text="Create a booking"),
            wyat.Candidate(id="cancel", text="Cancel the booking"),
        ],
    )
    decision = model.decide(request)

    print(f"Message: {request.context}")
    print("Choices:")
    for candidate in request.candidates:
        print(f"  {candidate.id}: {candidate.text}")
    print(f"Selected: {decision.selected}")


if __name__ == "__main__":
    main()
