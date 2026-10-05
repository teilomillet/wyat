"""Compose immutable prefix reuse with final-block endpoint selection."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import cast

from benchmarks.analysis.mlx.prefix import Arguments, study
from wyat.backends.kai.prefix import PrefixScorer as PrefixEndpointScorer


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/mlx-combined77-b8.json"),
    )
    options = parser.parse_args(namespace=Arguments())
    receipt = study(
        options.output,
        batch_size=options.batch_size,
        candidate_type=PrefixEndpointScorer,
        strategy="prefix25-endpoints",
    )
    print(
        json.dumps(
            {
                "screen": receipt.get("screen_quality"),
                "performance_gate_passed": receipt.get("performance_gate_passed"),
                "paired_speed": cast(dict[str, object], receipt.get("paired", {})).get(
                    "speed"
                ),
            }
        ),
        flush=True,
    )
    if receipt["completed"] is not True or receipt["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
