"""Owned-group timeout checks without spawning compilers or loading tensors."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

from benchmarks.analysis.torch import process
from wyat.wire import read_object


def test_timeout_kills_only_owned_group_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waited: list[float | None] = []
    signals: list[tuple[int, int]] = []
    creation: list[dict[str, object]] = []

    class Child:
        pid = 12345

        def wait(self, timeout: float | None = None) -> int:
            waited.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("owned worker", timeout)
            return -9

    def popen(*args: object, **kwargs: object) -> Child:
        creation.append({"command": args[0], **kwargs})
        return Child()

    def killpg(group: int, sig: int) -> None:
        signals.append((group, sig))

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "run.json"
    assert (
        process.launch(
            "benchmarks.analysis.torch.autotune",
            tmp_path / "model",
            output,
            timeout=0.01,
            worker_args=("--batch-size", "1"),
        )
        == 124
    )
    assert signals == [(Child.pid, signal.SIGKILL)]
    assert waited == [0.01, None]
    assert creation[0]["start_new_session"] is True
    command = creation[0]["command"]
    assert isinstance(command, list)
    assert command[-3:] == ["--batch-size", "1", "--worker"]
    environment = creation[0]["env"]
    assert isinstance(environment, dict)
    assert environment["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert environment["TORCHINDUCTOR_FREEZING"] == "1"
    cache = environment["TORCHINDUCTOR_CACHE_DIR"]
    assert isinstance(cache, str)
    assert Path(cache).parent == tmp_path and Path(cache).is_dir()
    receipt = read_object(output.with_suffix(".wall.json"))
    assert receipt["inductor_cache"] == cache
    assert receipt["timed_out"] is True
    assert receipt["completed"] is False
    assert not output.exists()


def test_launcher_preserves_old_receipts(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    output.with_suffix(".wall.json").write_text("old")
    with pytest.raises(FileExistsError, match="Preserve"):
        process.launch("unused", tmp_path / "model", output, timeout=1)
