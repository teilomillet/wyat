"""Owned 120-second primitive worker; parent imports only standard library."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path


class Arguments(Namespace):
    model: Path
    output: Path
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        from .study import study

        code = 0 if study(args.model.resolve(), args.output.resolve()) else 1
    else:
        from benchmarks.analysis.torch.process import launch

        code = launch(
            __package__ + ".run",
            args.model.resolve(),
            args.output.resolve(),
            timeout=120,
        )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
