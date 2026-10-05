"""CPU experiment state and timeout boundaries without executing a model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from benchmarks.analysis.torch import threads
from wyat.backends.kai.torch import KaiScorer


def test_load_failure_restores_intraop_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"threads": 3}

    def set_threads(value: int) -> None:
        state["threads"] = value

    def fail_load(*args: object, **kwargs: object) -> None:
        # Provenance must exist even when loading never reaches inference.
        archives = list(tmp_path.glob("result.sources-*.json"))
        assert len(archives) == 1
        assert "runner" in {
            item["name"] for item in json.loads(archives[0].read_text())["files"]
        }
        raise ValueError("failed artifact")

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(KaiScorer, "from_directory", fail_load)
    with pytest.raises(ValueError, match="failed artifact"):
        threads.sweep(tmp_path / "model", tmp_path / "result.json")
    assert state["threads"] == 3


def test_timeout_records_unknown_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    journal = output.with_suffix(".cycles.jsonl")
    journal.write_text('{"stage":"warmup"}\n')

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("test command", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    monkeypatch.setattr(sys, "argv", ["threads", "model", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        threads.main()
    assert error.value.code == 1
    receipt = json.loads(output.with_suffix(".wall.json").read_text())
    assert receipt["timed_out"] is True
    assert receipt["completed"] is False
    assert not output.exists()
    assert journal.read_text() == '{"stage":"warmup"}\n'
