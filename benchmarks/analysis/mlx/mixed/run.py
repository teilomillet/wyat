"""Framework-free parent for one owned, bounded mixed projection assay."""

from __future__ import annotations

import os
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
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        if os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError("Set MLX_ENABLE_TF32=0 before importing MLX")
        from .study import study

        raise SystemExit(0 if study(args.model, args.output) else 1)
    previous = os.environ.get("MLX_ENABLE_TF32")
    try:
        os.environ["MLX_ENABLE_TF32"] = "0"
        code = launch(__package__ + ".run", args.model, args.output, timeout=180)
    finally:
        if previous is None:
            os.environ.pop("MLX_ENABLE_TF32", None)
        else:
            os.environ["MLX_ENABLE_TF32"] = previous
    raise SystemExit(code)


if __name__ == "__main__":
    main()
