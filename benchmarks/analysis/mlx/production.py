"""Bound the existing combined assay against final production source."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

from benchmarks.analysis.evidence import freeze_sources


def launch(python: Path, output: Path, *, batch_size: int) -> int:
    """Retain parent status; the shared study owns timing and quality policy."""
    if type(batch_size) is not int or batch_size not in (1, 8):
        raise ValueError("Production confirmation requires batch size 1 or 8")
    root = Path(__file__).resolve().parents[3]
    output = output.resolve()
    wall = output.with_suffix(".wall.json")
    if output.exists() or wall.exists():
        raise FileExistsError("Preserve existing assay receipts; choose a new output")
    source = freeze_sources({"launcher": Path(__file__)}, wall)
    command = [
        str(python.absolute()),
        "-m",
        "benchmarks.analysis.mlx.combined",
        "--batch-size",
        str(batch_size),
        "--output",
        str(output),
    ]
    started = time.perf_counter()
    receipt: dict[str, object] = {
        "command": command,
        "started_wall_time_ns": time.time_ns(),
        "timeout_seconds": 180,
        "timed_out": False,
        "completed": False,
        "sources_sha256": source.hashes(),
        "sources_archive": source.archive(),
        "scope": "Parent process status; child study owns numerical/performance gates",
    }
    wall.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        result = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
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
    receipt.update(
        process_exit_code=code,
        elapsed_seconds=time.perf_counter() - started,
        finished_wall_time_ns=time.time_ns(),
    )
    wall.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt), flush=True)
    return code


class Arguments(Namespace):
    batch_size: int
    python: Path
    output: Path | None


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=8)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    options = parser.parse_args(namespace=Arguments())
    output = options.output or Path(
        f"benchmarks/analysis/results/mlx-prefix-production77-b{options.batch_size}.json"
    )
    raise SystemExit(launch(options.python, output, batch_size=options.batch_size))


if __name__ == "__main__":
    main()
