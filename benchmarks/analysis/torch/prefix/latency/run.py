"""Launch one reviewed full77 comparator/shape in an owned 180-second child."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, cast

from benchmarks.analysis.torch.process import launch

ROOT = Path(__file__).resolve().parents[5]


class Arguments(Namespace):
    model: Path
    comparator: str
    batch_size: int
    output: Path | None
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--comparator", choices=("retained", "full"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    model = ROOT / args.model
    output = ROOT / (
        args.output
        or Path(
            f"benchmarks/analysis/results/torch-one-use-{args.comparator}-b{args.batch_size}.json"
        )
    )
    if args.worker:
        from .study import study

        passed = study(
            model,
            output,
            comparator=cast(Literal["retained", "full"], args.comparator),
            batch_size=args.batch_size,
        )
        raise SystemExit(0 if passed else 1)
    raise SystemExit(
        launch(
            __package__ + ".run",
            model,
            output,
            timeout=180,
            worker_args=(
                "--comparator",
                args.comparator,
                "--batch-size",
                str(args.batch_size),
            ),
        )
    )


if __name__ == "__main__":
    main()
