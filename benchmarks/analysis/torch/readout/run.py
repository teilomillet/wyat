"""Run one exploratory CPU endpoint screen in an owned 180-second process."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from benchmarks.analysis.torch.process import launch


class Arguments(Namespace):
    model: Path
    output: Path
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help="internal owned worker")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        from .study import study

        raise SystemExit(0 if study(args.model.resolve(), args.output.resolve()) else 1)
    raise SystemExit(
        launch(
            __package__ + ".run",
            args.model.resolve(),
            args.output.resolve(),
            timeout=180,
        )
    )


if __name__ == "__main__":
    main()
