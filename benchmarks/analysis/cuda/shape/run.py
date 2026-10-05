"""Stdlib-only owner of one fresh, bounded real-shape worker."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.analysis.torch.process import launch


class Arguments(argparse.Namespace):
    model: Path
    output: Path
    sources: Path
    sources_sha256: str
    worker: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        from .study import study

        return (
            0
            if study(
                args.model,
                args.output,
                sources=args.sources,
                sources_sha256=args.sources_sha256,
            )
            else 1
        )
    return launch(
        "benchmarks.analysis.cuda.shape.run",
        args.model,
        args.output,
        timeout=180,
        worker_args=(
            "--sources",
            str(args.sources),
            "--sources-sha256",
            args.sources_sha256,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
