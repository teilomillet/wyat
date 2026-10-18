# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Shared by every use case: load a local model once, or run without weights."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import wyat


class DemoScorer:
    """Gives every option the same score. Shows the API; decides nothing."""

    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(0.0 for _ in request.candidates) for request in requests)


def load_model(description: str | None) -> wyat.Model:
    parser = argparse.ArgumentParser(description=description)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model",
        type=Path,
        help="Model name (teilomillet/wyat-1-800m) or local model directory",
    )
    source.add_argument("--demo", action="store_true", help="Run without weights")
    parser.add_argument(
        "--backend",
        choices=("torch", "mlx"),
        default="torch",
        help="Runtime for --model (default: torch)",
    )
    parser.add_argument(
        "--device",
        help="Device: cpu, mps or cuda for torch; gpu or cpu for mlx",
    )
    arguments = parser.parse_args()
    if arguments.demo:
        print("Demo mode: equal scores, no trained model.\n")
        return wyat.Model(DemoScorer(), model_id="demo")
    return wyat.load(
        arguments.model,
        backend=cast(Literal["torch", "mlx"], arguments.backend),
        device=arguments.device,
    )
