"""Fresh TF32-disabled child, bounded BF16 screen or one survivor confirmation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

from benchmarks.analysis.evidence import freeze_sources
from wyat.wire import read_object

MODEL_ID = "kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee"
REFERENCE_SHA256 = "b8a90b31941b60e793357a8cb360767520da70ec33d2161c1242606962b22664"
DATASET_SHA256 = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"
type Arm = Literal["all-down", "all-gate-up", "all-mlp"]
ARMS: tuple[Arm, ...] = ("all-down", "all-gate-up", "all-mlp")


def validate_prior(
    path: Path, arm: str, source_hashes: dict[str, str], case_ids: tuple[str, ...]
) -> str:
    """Require full declared history; current worker independently rescreens."""
    prior = read_object(path)
    screens = prior.get("arms")
    if (
        arm not in ARMS
        or len(case_ids) != 77
        or len(set(case_ids)) != 77
        or prior.get("completed") is not True
        or prior.get("phase") != "screen"
        or prior.get("source_hashes") != source_hashes
        or prior.get("model_id") != MODEL_ID
        or prior.get("dataset_sha256") != DATASET_SHA256
        or prior.get("reference_sha256") != REFERENCE_SHA256
        or not isinstance(screens, dict)
        or set(screens) != set(ARMS)
    ):
        raise ValueError("Requires completed unchanged-source pinned family screen")
    for name in ARMS:
        outcome = screens[name]
        if not isinstance(outcome, dict):
            raise ValueError("Missing declared arm outcome")
        gate, shapes = outcome.get("cpu_quality"), outcome.get("screens")
        if (
            not isinstance(gate, dict)
            or type(gate.get("passed")) is not bool
            or type(gate.get("total")) is not int
            or gate.get("total") != 154
            or outcome.get("passed") != gate["passed"]
            or not isinstance(shapes, dict)
            or set(shapes) != {"8", "1"}
        ):
            raise ValueError("Missing full dual-shape arm quality history")
        for cycle in shapes.values():
            if not isinstance(cycle, dict) or not isinstance(cycle.get("rows"), list):
                raise ValueError("Missing retained cycle rows")
            rows = cycle["rows"]
            if (
                not isinstance(rows, list)
                or len(rows) != 77
                or any(not isinstance(row, dict) for row in rows)
            ):
                raise ValueError("Cycle must retain all77 outcomes")
            if (
                tuple(row.get("id") for row in rows if isinstance(row, dict))
                != case_ids
            ):
                raise ValueError("Prior cycle case identity/order changed")
    selected = screens[arm]
    if not isinstance(selected, dict) or selected.get("passed") is not True:
        raise ValueError("Only a dual-shape numerical survivor can be confirmed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(output: Path, *, arm: str | None = None, screen: Path | None = None) -> int:
    if arm not in (None, *ARMS) or ((arm is None) != (screen is None)):
        raise ValueError(
            "Confirmation requires one fixed arm and its full-family screen"
        )
    wall = output.with_suffix(".wall.json")
    if output.exists() or wall.exists():
        raise FileExistsError("Preserve existing BF16 receipts")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = freeze_sources({"launcher": Path(__file__)}, wall)
    command = [
        sys.executable,
        "-m",
        "benchmarks.analysis.mlx.bfloat.run",
        "--worker",
        "--output",
        str(output.resolve()),
    ]
    if arm is not None and screen is not None:
        command.extend(["--time-arm", arm, "--screen", str(screen.resolve())])
    receipt: dict[str, object] = {
        "command": command,
        "timeout_seconds": 180,
        "completed": False,
        "timed_out": False,
        "tf32": "0",
        "sources": source.archive(),
    }
    wall.write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            env={**os.environ, "MLX_ENABLE_TF32": "0"},
            check=False,
            timeout=180,
        )
        code = result.returncode
        receipt["completed"] = True
    except subprocess.TimeoutExpired:
        receipt["timed_out"] = True
        code = 124
    except OSError as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        code = 1
    receipt.update(exit_code=code, elapsed_seconds=time.perf_counter() - start)
    wall.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return code


class Arguments(Namespace):
    output: Path
    time_arm: str | None
    screen: Path | None
    worker: bool


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-arm", choices=ARMS)
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args(namespace=Arguments())
    if not arguments.worker:
        return launch(arguments.output, arm=arguments.time_arm, screen=arguments.screen)
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Disable TF32 before importing the MLX worker")
    from benchmarks.analysis.mlx.bfloat.study import study

    return study(
        arguments.output, arm=arguments.time_arm, prior_screen=arguments.screen
    )


if __name__ == "__main__":
    raise SystemExit(main())
