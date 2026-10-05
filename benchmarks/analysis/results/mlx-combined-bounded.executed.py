from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path("/Users/teilomillet/Code/wyat")
batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 8
output = root / f"benchmarks/analysis/results/mlx-combined77-b{batch_size}.json"
command = [
    "/tmp/wyat-mlx-only/bin/python",
    str(root / "benchmarks/analysis/mlx/combined.py"),
    "--batch-size",
    str(batch_size),
    "--output",
    str(output),
]
started_wall = time.time_ns()
started = time.perf_counter()
receipt: dict[str, object] = {
    "command": command,
    "started_wall_time_ns": started_wall,
    "timeout_seconds": 180,
    "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "timed_out": False,
}
try:
    result = subprocess.run(
        command,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=False,
        timeout=180,
    )
    code = result.returncode
except subprocess.TimeoutExpired:
    receipt["timed_out"] = True
    code = 124
receipt["process_exit_code"] = code
receipt["elapsed_seconds"] = time.perf_counter() - started
receipt["finished_wall_time_ns"] = time.time_ns()
output.with_suffix(".wall.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
