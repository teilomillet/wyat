"""Pure lifecycle/preselection boundaries; imports neither MLX nor Torch."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.analysis.mlx.bfloat.run import (
    ARMS,
    DATASET_SHA256,
    MODEL_ID,
    REFERENCE_SHA256,
    launch,
    validate_prior,
)


def prior_document() -> dict[str, object]:
    cycle = {"rows": [{"id": str(index)} for index in range(77)]}
    return {
        "completed": True,
        "phase": "screen",
        "source_hashes": {"runner": "fixed"},
        "model_id": MODEL_ID,
        "dataset_sha256": DATASET_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "arms": {
            name: {
                "passed": name == "all-down",
                "cpu_quality": {"passed": name == "all-down", "total": 154},
                "screens": {"8": cycle, "1": cycle},
            }
            for name in ARMS
        },
    }


def test_full_family_history_admits_only_declared_survivor(tmp_path: Path) -> None:
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior_document()))
    ids = tuple(str(index) for index in range(77))
    assert len(validate_prior(path, "all-down", {"runner": "fixed"}, ids)) == 64
    with pytest.raises(ValueError, match="survivor"):
        validate_prior(path, "all-mlp", {"runner": "fixed"}, ids)


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "dataset_sha256",
        "reference_sha256",
        "source_hashes",
        "phase",
        "completed",
    ],
)
def test_changed_family_identity_never_admits_timing(
    tmp_path: Path, field: str
) -> None:
    document = prior_document()
    document[field] = "changed"
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        validate_prior(
            path,
            "all-down",
            {"runner": "fixed"},
            tuple(str(index) for index in range(77)),
        )


@pytest.mark.parametrize(
    "fault",
    ["missing-arm", "missing-shape", "missing-gate", "missing-row", "wrong-order"],
)
def test_incomplete_nonselected_arm_history_never_admits_timing(
    tmp_path: Path, fault: str
) -> None:
    document = prior_document()
    arms = document["arms"]
    assert isinstance(arms, dict)
    if fault == "missing-arm":
        del arms["all-mlp"]
    else:
        arm = arms["all-mlp"]
        assert isinstance(arm, dict)
        if fault == "missing-gate":
            del arm["cpu_quality"]
        else:
            screens = arm["screens"]
            assert isinstance(screens, dict)
            if fault == "missing-shape":
                del screens["1"]
            else:
                cycle = screens["1"]
                assert isinstance(cycle, dict)
                rows = cycle["rows"]
                assert isinstance(rows, list)
                if fault == "missing-row":
                    rows.pop()
                else:
                    rows.reverse()
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        validate_prior(
            path,
            "all-down",
            {"runner": "fixed"},
            tuple(str(index) for index in range(77)),
        )


def test_fresh_child_sets_tf32_before_worker_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(
        command: list[str], *, env: dict[str, str], check: bool, timeout: int
    ) -> subprocess.CompletedProcess[str]:
        assert env["MLX_ENABLE_TF32"] == "0" and timeout == 180
        assert "--worker" in command
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "negative.json"
    assert launch(output) == 1
    assert json.loads(output.with_suffix(".wall.json").read_text())["completed"] is True
    with pytest.raises(FileExistsError):
        launch(output)


def test_parent_timeout_and_invalid_timing_are_nonpasses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout(
        command: list[str], *, env: dict[str, str], check: bool, timeout: int
    ) -> None:
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(subprocess, "run", timeout)
    output = tmp_path / "timeout.json"
    assert launch(output) == 124
    wall = json.loads(output.with_suffix(".wall.json").read_text())
    assert wall["timed_out"] is True and wall["completed"] is False
    with pytest.raises(ValueError):
        launch(tmp_path / "invalid.json", arm="all-mlp")
