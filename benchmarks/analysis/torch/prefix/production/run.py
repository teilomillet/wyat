"""Confirm one CPU prefix load in an owned 180-second worker process."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from benchmarks.analysis.torch.process import launch


class Arguments(Namespace):
    model: Path
    output: Path
    batch_size: int
    worker: bool
    screen: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument(
        "--screen", action="store_true", help="first8 numerical screen; B8 only"
    )
    parser.add_argument("--worker", action="store_true", help="internal owned worker")
    args = parser.parse_args(namespace=Arguments())
    if args.screen and args.batch_size != 8:
        parser.error("First8 screen requires batch size8")
    if args.worker:
        from .study import study

        passed = study(
            args.model.resolve(),
            args.output.resolve(),
            args.batch_size,
            screen=args.screen,
        )
        raise SystemExit(0 if passed else 1)
    raise SystemExit(
        launch(
            __package__ + ".run",
            args.model.resolve(),
            args.output.resolve(),
            timeout=180,
            worker_args=(
                "--batch-size",
                str(args.batch_size),
                *(("--screen",) if args.screen else ()),
            ),
        )
    )


if __name__ == "__main__":
    main()
