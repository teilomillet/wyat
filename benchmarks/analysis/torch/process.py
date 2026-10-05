"""A hard wall cap for an experiment and only its owned process group."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def launch(
    module: str,
    model: Path,
    output: Path,
    *,
    timeout: float,
    worker_args: tuple[str, ...] = (),
) -> int:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive")
    output = output.resolve()
    if any(output.parent.glob(output.stem + ".*")):
        raise FileExistsError("Preserve existing receipts; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        module,
        str(model),
        "--output",
        str(output),
        *worker_args,
        "--worker",
    ]
    environment = {**os.environ}
    if module.endswith("autotune"):
        environment.update(
            TORCHINDUCTOR_FREEZING="1",
            TORCH_LOGS="output_code",
            TORCHINDUCTOR_COMPILE_THREADS="1",
        )
        environment["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
            prefix="wyat-cpu-autotune-", dir="/private/tmp"
        )
    started = time.perf_counter()
    status: dict[str, object] = {
        "command": command,
        "timeout_seconds": timeout,
        "completed": False,
        "timed_out": False,
        "inductor_cache": environment.get("TORCHINDUCTOR_CACHE_DIR"),
    }
    wall = output.with_suffix(".wall.json")
    wall.write_text(json.dumps(status, indent=2) + "\n")
    code = 1
    with output.with_suffix(".log").open("w") as log:
        child = subprocess.Popen(
            command, env=environment, stdout=log, stderr=log, start_new_session=True
        )
        try:
            code = child.wait(timeout=timeout)
            status["completed"] = code == 0
        except subprocess.TimeoutExpired:
            status["timed_out"] = True
            code = 124
        finally:
            # Workers inherit this new session; never signal another experiment.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
    status.update(exit_code=code, elapsed_seconds=time.perf_counter() - started)
    wall.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    return code
