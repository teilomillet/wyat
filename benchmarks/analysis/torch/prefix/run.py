"""Launch the native CPU prefix dispatch screen under an owned 180-second cap.

python -m benchmarks.analysis.torch.prefix.run MODEL --output RECEIPT.json
Run in the released native Mac context; no flag patching or compilation.
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path

from benchmarks.analysis.torch.process import launch

from .study import ROOT, study


class Arguments(Namespace):
    model: Path
    output: Path | None
    worker: bool
    batch_size: int
    pairs: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--pairs", action="store_true", help="full77 confirmation")
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--worker", action="store_true", help="internal owned process")
    args = parser.parse_args(namespace=Arguments())
    if not args.pairs and args.batch_size != 8:
        parser.error("first-eight dispatch screen requires batch size8")
    output = args.output or ROOT / (
        f"benchmarks/analysis/results/torch-prefix77-b{args.batch_size}.json"
        if args.pairs
        else "benchmarks/analysis/results/torch-prefix8.json"
    )
    model, output = ROOT / args.model, ROOT / output
    if args.worker:
        if args.pairs:
            from .pairs import study as confirm

            passed = confirm(model, output, batch_size=args.batch_size)
        else:
            passed = study(model, output)
        raise SystemExit(0 if passed else 1)
    worker_args = (
        ("--pairs", "--batch-size", str(args.batch_size)) if args.pairs else ()
    )
    raise SystemExit(
        launch(
            __package__ + ".run", model, output, timeout=180, worker_args=worker_args
        )
    )


if __name__ == "__main__":
    main()
