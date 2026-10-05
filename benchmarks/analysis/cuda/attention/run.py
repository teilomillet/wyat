"""Stdlib owned parent; the worker alone imports the inference framework."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.analysis.torch.process import launch


class Arguments(argparse.Namespace):
    model: Path
    manifest: Path
    manifest_sha256: str
    sources: Path
    sources_sha256: str
    control: Path
    control_sha256: str
    batch_size: int
    output: Path
    worker: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    for field in ("manifest", "sources", "control"):
        parser.add_argument(f"--{field}", type=Path, required=True)
        parser.add_argument(f"--{field}-sha256", required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        from .study import study

        return (
            0
            if study(
                args.model,
                args.output,
                manifest=args.manifest,
                manifest_sha256=args.manifest_sha256,
                sources=args.sources,
                sources_sha256=args.sources_sha256,
                control=args.control,
                control_sha256=args.control_sha256,
                batch_size=args.batch_size,
            )
            else 1
        )
    return launch(
        "benchmarks.analysis.cuda.attention.run",
        args.model,
        args.output,
        timeout=180,
        worker_args=(
            "--manifest",
            str(args.manifest),
            "--manifest-sha256",
            args.manifest_sha256,
            "--sources",
            str(args.sources),
            "--sources-sha256",
            args.sources_sha256,
            "--control",
            str(args.control),
            "--control-sha256",
            args.control_sha256,
            "--batch-size",
            str(args.batch_size),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
