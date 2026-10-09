"""Native artifact inventory: published documentation is tolerated, nothing else."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

native = pytest.importorskip("wyat.loading.native")

REQUIRED = (
    "backbone/config.json",
    "backbone/model.safetensors",
    "decision_head.safetensors",
    "native-adapters.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """A manifest-valid artifact; validation never decodes tensor payloads."""
    root = tmp_path / "artifact"
    (root / "backbone").mkdir(parents=True)
    for index, name in enumerate(REQUIRED):
        (root / name).write_bytes(f"file-{index}".encode())
    files = {name: native.file_sha256(root / name) for name in REQUIRED}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    manifest = {
        "format": native.FORMAT,
        "model_id": "native-" + identity,
        "contract": native.artifact_contract(),
        "adapter": {
            "rank": 16,
            "alpha": 32,
            "targets": ["q_proj", "v_proj"],
            "merged": False,
        },
        "max_length": 8,
        "max_batch_tokens": 8,
        "head_dim": 4,
        "files_sha256": files,
        "provenance": {},
        "production_kai_compatible": False,
        "resume_supported": False,
    }
    (root / native.MANIFEST).write_text(json.dumps(manifest))
    return root


def test_published_documentation_and_download_metadata_are_ignored(artifact):
    for name in sorted(native.DOCUMENTATION_FILES):
        (artifact / name).write_text("documentation\n")
    metadata = artifact / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    (metadata / "native-adapters.safetensors.metadata").write_text("etag\n")
    (artifact / ".cache/huggingface/.gitignore").write_text("*\n")
    assert native.validate_artifact(artifact)["format"] == native.FORMAT


@pytest.mark.parametrize(
    "name", ["notes.txt", "backbone/README.md", "docs/LICENSE", ".cache/other/x"]
)
def test_other_undeclared_files_are_still_rejected(artifact, name):
    path = artifact / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("extra\n")
    with pytest.raises(ValueError, match="inventory differs"):
        native.validate_artifact(artifact)


def test_declared_files_are_still_hash_verified_beside_documentation(artifact):
    (artifact / "README.md").write_text("card\n")
    (artifact / "decision_head.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        native.validate_artifact(artifact)
