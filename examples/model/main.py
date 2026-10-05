"""Load a supported local model once, then print a structured decision."""

import argparse
import json
from pathlib import Path
from typing import Literal, cast

import wyat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local model directory")
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    arguments = parser.parse_args()
    model = wyat.load(
        cast(Path, arguments.model),
        backend=cast(Literal["torch", "mlx"], arguments.backend),
    )
    request = wyat.Request(
        query="Choose the requested action",
        context="Cancel my booking",
        candidates=(
            wyat.Candidate(id="cancel", text="Cancel the booking"),
            wyat.Candidate(id="create", text="Create a booking"),
        ),
    )
    print(json.dumps(model.decide(request).to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
