"""The shared artifact owner preserves strict identities without loading Torch."""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wyat.input_rendering import input_rendering_policy
from wyat.loading import native_artifact as artifact


def test_head_identity_matches_historical_inspect_contract() -> None:
    torch_runtime = pytest.importorskip("wyat.backends.wyat.torch")
    expected = hashlib.sha256(
        inspect.getsource(torch_runtime.CandidateHead).encode("utf-8")
    ).hexdigest()
    assert artifact.artifact_contract()["predictor_sha256"] == expected


def test_native_loader_reexports_same_contract() -> None:
    native = pytest.importorskip("wyat.loading.native")
    assert native.artifact_contract is artifact.artifact_contract
    assert native.validate_artifact is artifact.validate_artifact
    assert native.file_sha256 is artifact.file_sha256
    assert native.artifact_contract() == artifact.artifact_contract()


def test_contract_import_and_execution_do_not_import_torch() -> None:
    code = (
        "import sys; from wyat.loading.native_artifact import artifact_contract; "
        "assert artifact_contract()['recipe']; "
        "assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    (root / "backbone").mkdir(parents=True)
    names = (
        "backbone/config.json",
        "backbone/model.safetensors",
        "decision_head.safetensors",
        "native-adapters.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for name in names:
        (root / name).write_bytes(name.encode())
    hashes = {name: artifact.file_sha256(root / name) for name in names}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = {
        "format": artifact.FORMAT,
        "model_id": "native-" + identity,
        "provenance": {},
        "production_wyat_compatible": False,
        "resume_supported": False,
        "contract": artifact.artifact_contract(),
        "adapter": {
            "rank": 16,
            "alpha": 32,
            "targets": ["q_proj", "v_proj"],
            "merged": False,
        },
        "max_length": 8,
        "max_batch_tokens": 8,
        "head_dim": 4,
        "files_sha256": hashes,
    }
    (root / artifact.MANIFEST).write_text(json.dumps(manifest))
    return root


def test_valid_inventory_is_accepted(directory: Path) -> None:
    assert artifact.validate_artifact(directory)["format"] == artifact.FORMAT


def test_rendering_is_authenticated_and_changes_model_identity(directory: Path) -> None:
    path = directory / artifact.MANIFEST
    manifest = json.loads(path.read_text())
    legacy_id = manifest["model_id"]
    assert artifact.model_identity(manifest["files_sha256"]) == legacy_id
    selected = input_rendering_policy("api-bank-schema-next-v1")
    manifest["input_rendering"] = selected
    with pytest.raises(ValueError, match="model identity"):
        path.write_text(json.dumps(manifest))
        artifact.validate_artifact(directory)
    manifest["model_id"] = artifact.model_identity(manifest["files_sha256"], selected)
    assert manifest["model_id"] != legacy_id
    path.write_text(json.dumps(manifest))
    assert artifact.validate_artifact(directory)["input_rendering"] == selected


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {"mode": "unknown", "source_sha256": "0" * 64},
        {"mode": "api-bank-schema-next-v1", "source_sha256": "0" * 64},
    ],
)
def test_explicit_invalid_rendering_policy_is_rejected(directory: Path, policy) -> None:
    path = directory / artifact.MANIFEST
    manifest = json.loads(path.read_text())
    manifest["input_rendering"] = policy
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="input rendering"):
        artifact.validate_artifact(directory)


@pytest.mark.parametrize("change", ["extra", "hash", "contract", "symlink"])
def test_strict_inventory_and_contract_rejection(directory: Path, change: str) -> None:
    manifest_path = directory / artifact.MANIFEST
    if change == "extra":
        (directory / "extra").write_text("undeclared")
    elif change == "hash":
        (directory / "decision_head.safetensors").write_bytes(b"changed")
    elif change == "contract":
        manifest = json.loads(manifest_path.read_text())
        manifest["contract"]["tasks_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    else:
        path = directory / "tokenizer.json"
        contents = path.read_bytes()
        path.unlink()
        other = directory.parent / "tokenizer.json"
        other.write_bytes(contents)
        path.symlink_to(other)
    with pytest.raises(ValueError):
        artifact.validate_artifact(directory)
