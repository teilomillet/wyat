"""Artifact verification rejects changes before any inference is possible."""

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

VERIFIER = Path(__file__).resolve().parents[2] / "examples" / "model" / "verify.py"


@pytest.fixture
def artifact(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    values = runpy.run_path(str(VERIFIER))
    hashes = {}
    for name in values["FILES"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    payload = json.dumps({"files_sha256": hashes}).encode()
    (tmp_path / "native_manifest.json").write_bytes(payload)
    # A tiny synthetic artifact exercises the same integrity boundaries without
    # downloading weights. The production pin is checked separately against Hub.
    values["verify"].__globals__["MANIFEST_SHA256"] = hashlib.sha256(
        payload
    ).hexdigest()
    return tmp_path, values


def test_complete_pinned_payloads_are_reported(
    artifact: tuple[Path, dict[str, Any]],
) -> None:
    root, values = artifact
    report = values["verify"](root)
    assert set(report["files_sha256"]) == set(values["FILES"])
    assert report["revision"] == values["REVISION"]
    assert "no inference" in report["scope"]


def test_manifest_is_checked_before_trusting_its_payload_hashes(
    artifact: tuple[Path, dict[str, Any]],
) -> None:
    root, values = artifact
    (root / "native_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest differs"):
        values["verify"](root)


@pytest.mark.parametrize("name", ["tokenizer.json", "backbone/model.safetensors"])
def test_payload_changes_are_rejected(
    artifact: tuple[Path, dict[str, Any]], name: str
) -> None:
    root, values = artifact
    (root / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file differs"):
        values["verify"](root)


def test_missing_payload_is_not_success(
    artifact: tuple[Path, dict[str, Any]],
) -> None:
    root, values = artifact
    (root / "decision_head.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        values["verify"](root)


@pytest.mark.parametrize(
    "name", ["special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"]
)
def test_mixed_tokenizer_files_are_rejected(
    artifact: tuple[Path, dict[str, Any]], name: str
) -> None:
    root, values = artifact
    (root / name).write_text("{}")
    with pytest.raises(ValueError, match="unexpected tokenizer"):
        values["verify"](root)
