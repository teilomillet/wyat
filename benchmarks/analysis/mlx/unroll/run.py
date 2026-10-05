"""Framework-free admission and owner of one 180-second fixed64 unroll trial."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from benchmarks.analysis.mlx.tile.admit import (
    ROOT,
    Admission,
    admit,
    archive,
    checked_file,
    digest,
    object_value,
)
from benchmarks.analysis.torch.process import launch
from wyat.wire import JsonObject, read_object

MAIN = ROOT / "benchmarks/analysis/results/mlx-tile64-gate.json"
AUDIT = MAIN.with_name("mlx-tile64-gate-review.json")
CHECKS = ROOT / "benchmarks/real/mlx-tile-tooling-verification.json"
PINNED = {
    MAIN: "22dcca467b451e6fd0129671762bb74f9bbdb08d32c32aecdb9803263284e16a",
    AUDIT: "9099425df4676a02b514ef821181c7b45d40e530c54eb80f44b4109454c4ab26",
    CHECKS: "966ba6a37aa30faa933a73b51a79d6da763a27cbb499ee1b5a5b9716bc578844",
}


@dataclass(frozen=True, slots=True)
class Context:
    original: Admission
    prior: JsonObject
    cold: Path
    sources: dict[str, Path]


def context(model: Path) -> Context:
    for path, expected in PINNED.items():
        if digest(path) != expected:
            raise ValueError("Frozen64 evidence changed")
    prior, audit, checks = map(read_object, PINNED)
    if (
        prior["completed"] is not True
        or prior["projection_quality_passed"] is not True
        or audit["audit_passed"] is not True
        or checks["completed"] is not True
    ):
        raise ValueError("Frozen64 evidence is incomplete")
    sources = archive(
        checks["source_archive"], object_value(checks["source_sha256"]), 111
    )
    hashes = object_value(prior["source_hashes"])
    exact_sources = archive(prior["source_archive"], hashes, 123)
    if any(digest(path) != hashes[name] for name, path in exact_sources.items()):
        raise ValueError("Frozen64 source bytes changed")
    sources.update(exact_sources)
    for record in object_value(audit["artifacts"]).values():
        checked_file(record)
    original = admit(model)
    if checked_file(prior["reused_inputs"]) != original.inputs:
        raise ValueError("Frozen64 trial used different captured inputs")
    return Context(original, prior, checked_file(prior["cold_outputs"]), sources)


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
        admitted = context(args.model)
        from .study import study

        raise SystemExit(0 if study(admitted, args.output) else 1)
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
