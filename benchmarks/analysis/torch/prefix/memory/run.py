"""Launch one fresh CPU cache-resource arm; run three alternating pairs serially."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal, cast

from benchmarks.analysis.torch.process import launch

ROOT = Path(__file__).resolve().parents[5]


class Arguments(Namespace):
    model: Path
    output: Path | None
    arm: str
    repeat: int
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--arm", choices=("retained", "one-use"), required=True)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    model = ROOT / args.model
    output = ROOT / (
        args.output
        or Path(
            f"benchmarks/analysis/results/torch-prefix-memory-{args.arm}-{args.repeat}.json"
        )
    )
    if args.worker:
        from .study import study

        passed = study(
            model,
            output,
            arm=cast(Literal["retained", "one-use"], args.arm),
            repeat=args.repeat,
        )
        raise SystemExit(0 if passed else 1)
    raise SystemExit(
        launch(
            __package__ + ".run",
            model,
            output,
            timeout=180,
            worker_args=("--arm", args.arm, "--repeat", str(args.repeat)),
        )
    )


if __name__ == "__main__":
    main()
