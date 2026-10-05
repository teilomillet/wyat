"""Stdlib process owner; inference imports occur only in the bounded worker."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from benchmarks.analysis.torch.process import launch


class Arguments(argparse.Namespace):
    model: Path
    output: Path
    manifest: Path
    manifest_sha256: str
    sources: Path
    sources_sha256: str
    control: Path
    control_sha256: str
    mode: Literal["confirm", "timing"]
    batch_size: int
    confirmation: Path | None
    confirmation_sha256: str | None
    confirmation_wall_sha256: str | None
    worker: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for field in ("manifest", "sources", "control"):
        parser.add_argument(f"--{field}", type=Path, required=True)
        parser.add_argument(f"--{field}-sha256", required=True)
    parser.add_argument("--mode", choices=("confirm", "timing"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--confirmation-sha256")
    parser.add_argument("--confirmation-wall-sha256")
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
                mode=args.mode,
                batch_size=args.batch_size,
                confirmation=args.confirmation,
                confirmation_sha256=args.confirmation_sha256,
                confirmation_wall_sha256=args.confirmation_wall_sha256,
            )
            else 1
        )
    if any(
        path.exists()
        for path in (
            args.output,
            args.output.with_suffix(".wall.json"),
            args.output.with_suffix(".log"),
        )
    ):
        raise ValueError("A new trial must not overwrite an existing receipt/wall/log")
    options = [
        "--mode",
        args.mode,
        "--batch-size",
        str(args.batch_size),
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
    ]
    if args.confirmation is not None:
        options.extend(("--confirmation", str(args.confirmation)))
    if args.confirmation_sha256 is not None:
        options.extend(("--confirmation-sha256", args.confirmation_sha256))
    if args.confirmation_wall_sha256 is not None:
        options.extend(("--confirmation-wall-sha256", args.confirmation_wall_sha256))
    return launch(
        "benchmarks.analysis.cuda.latency.run",
        args.model,
        args.output,
        timeout=180,
        worker_args=tuple(options),
    )


if __name__ == "__main__":
    raise SystemExit(main())
