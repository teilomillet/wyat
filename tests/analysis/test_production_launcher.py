"""No model imports or child execution: bounded-parent failure boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

from benchmarks.analysis.mlx import production


@pytest.mark.parametrize("code", [0, 7])
def test_exit_status_and_archived_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    observed: list[list[str]] = []

    def child(
        command: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        assert options["timeout"] == 180
        assert options["check"] is False
        observed.append(command)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(subprocess, "run", child)
    output = tmp_path / "production.json"
    assert production.launch(Path("python"), output, batch_size=8) == code
    receipt: dict[str, object] = json.loads(
        output.with_suffix(".wall.json").read_text()
    )
    assert receipt["completed"] is True
    assert receipt["process_exit_code"] == code
    assert receipt["timed_out"] is False
    assert observed[0][-4:] == ["--batch-size", "8", "--output", str(output)]
    archive = receipt["sources_archive"]
    assert isinstance(archive, dict)
    data = Path(archive["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == archive["sha256"]


def test_timeout_retains_incomplete_parent_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def child(
        command: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        assert options["timeout"] == 180
        raise subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(subprocess, "run", child)
    output = tmp_path / "timeout.json"
    assert production.launch(Path("python"), output, batch_size=1) == 124
    receipt: dict[str, object] = json.loads(
        output.with_suffix(".wall.json").read_text()
    )
    assert receipt["completed"] is False
    assert receipt["timed_out"] is True
    assert receipt["process_exit_code"] == 124


def test_existing_receipt_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "immutable.json"
    output.write_text("existing evidence\n")
    with pytest.raises(FileExistsError, match="Preserve"):
        production.launch(Path("python"), output, batch_size=8)
    assert output.read_text() == "existing evidence\n"


def test_virtualenv_symlink_path_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    def child(
        command: list[str], **options: object
    ) -> subprocess.CompletedProcess[str]:
        assert command[0] == str(venv_python.absolute())
        assert command[0] != str(venv_python.resolve())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", child)
    assert production.launch(venv_python, tmp_path / "venv.json", batch_size=8) == 0


@pytest.mark.parametrize("invalid", [True, 8.0])
def test_batch_size_rejects_boolean_and_noninteger(
    tmp_path: Path, invalid: bool | float
) -> None:
    with pytest.raises(ValueError, match="batch size"):
        production.launch(
            Path("python"), tmp_path / "invalid.json", batch_size=cast(int, invalid)
        )
