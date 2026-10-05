"""Stdlib parent owns the cap; Torch is imported only by the worker."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from benchmarks.analysis.torch.process import launch


class Arguments(argparse.Namespace):
    model: Path
    manifest: Path
    manifest_sha256: str
    arm: Literal["fp32", "released"]
    batch_size: int
    control: Path | None
    control_sha256: str | None
    output: Path
    worker: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--arm", choices=("fp32", "released"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--control-sha256")
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
                arm=args.arm,
                batch_size=args.batch_size,
                control=args.control,
                control_sha256=args.control_sha256,
            )
            else 1
        )
    extra: tuple[str, ...] = (
        "--manifest",
        str(args.manifest),
        "--manifest-sha256",
        args.manifest_sha256,
        "--arm",
        args.arm,
        "--batch-size",
        str(args.batch_size),
    )
    if args.control is not None:
        extra += ("--control", str(args.control))
    if args.control_sha256 is not None:
        extra += ("--control-sha256", args.control_sha256)
    return launch(
        "benchmarks.analysis.cuda.run",
        args.model,
        args.output,
        timeout=180,
        worker_args=extra,
    )


if __name__ == "__main__":
    raise SystemExit(main())
