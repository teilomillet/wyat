# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Fork release admissions retain the publication content and inventory guards."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from scripts.package import privacy


def _selected(tmp_path: Path, name: str, raw: bytes) -> dict[str, Path]:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {name: path}


def _tensor_bytes(*, metadata: bool = False) -> bytes:
    header: dict[str, object] = {
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
    }
    if metadata:
        header["__metadata__"] = {"note": "unreviewed payload"}
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + struct.pack("<f", 0)


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("model/base/config.json", b'{"model_type":"qwen3_5"}'),
        ("model/base/model.safetensors", _tensor_bytes()),
        ("model/adapters.safetensors", _tensor_bytes()),
        ("model/head.safetensors", _tensor_bytes()),
        ("model/fork_manifest.json", b'{"format":"wyat-option-fork-v1"}'),
        ("model/merges.txt", b"#version: 0.2\na b\n"),
        ("runtime/wyat/loading/fork.py", b'"""Frozen fork runtime."""\n'),
        (
            "runtime/wyat/loading/fork_artifact.py",
            b'"""Frozen fork artifact contract."""\n',
        ),
    ],
)
def test_fork_release_names_are_explicitly_admitted(
    tmp_path: Path, name: str, raw: bytes
) -> None:
    privacy.check_files(_selected(tmp_path, name, raw), profile="model")


def test_production_vocabulary_pin_matches_reviewed_qwen_export() -> None:
    assert privacy._FORK_VOCAB_SHA256 == (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    )


@pytest.fixture
def pinned_vocabulary(monkeypatch: pytest.MonkeyPatch) -> bytes:
    # A small stand-in exercises the hash boundary without copying the public
    # 6.7 MB vocabulary into tests. The release check uses the production pin.
    raw = b'{"password":1,"records":2,"predictions":3}'
    monkeypatch.setattr(privacy, "_FORK_VOCAB_SHA256", hashlib.sha256(raw).hexdigest())
    return raw


def test_exact_pinned_vocabulary_can_contain_ordinary_record_words(
    tmp_path: Path, pinned_vocabulary: bytes
) -> None:
    privacy.check_files(
        _selected(tmp_path, "model/vocab.json", pinned_vocabulary), profile="model"
    )


@pytest.mark.parametrize("change", ["whitespace", "token-id"])
def test_modified_vocabulary_is_rejected(
    tmp_path: Path, pinned_vocabulary: bytes, change: str
) -> None:
    modified = (
        pinned_vocabulary + b" "
        if change == "whitespace"
        else pinned_vocabulary.replace(b":1", b":9")
    )
    with pytest.raises(ValueError, match="unrecognized public vocabulary"):
        privacy.check_files(
            _selected(tmp_path, "model/vocab.json", modified),
            profile="model",
        )


def test_valid_unrecognized_vocabulary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognized public vocabulary"):
        privacy.check_files(
            _selected(tmp_path, "model/vocab.json", b'{"ordinary":1}'),
            profile="model",
        )


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({"note": "h" + "f_" + "A" * 35}).encode(),
        json.dumps({"note": "/" + "home/alice/private.txt"}).encode(),
    ],
    ids=["credential", "private-path"],
)
def test_vocabulary_hash_admission_never_skips_content_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    monkeypatch.setattr(privacy, "_FORK_VOCAB_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="credential|private filesystem path"):
        privacy.check_files(
            _selected(tmp_path, "model/vocab.json", raw), profile="model"
        )


@pytest.mark.parametrize("name", ["model/fork_manifest.json", "model/base/config.json"])
def test_fork_metadata_does_not_admit_raw_records(tmp_path: Path, name: str) -> None:
    raw = b'{"nested":{"records":[{"note":"private example"}]}}'
    with pytest.raises(ValueError, match="record field") as failure:
        privacy.check_files(_selected(tmp_path, name, raw), profile="model")
    assert "private example" not in str(failure.value)


@pytest.mark.parametrize(
    "name",
    [
        "model/base/training.pt",
        "model/raw_records.json",
        "runtime/wyat/loading/fork_private.py",
    ],
)
def test_adjacent_unadmitted_files_still_fail(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="unadmitted model artifact"):
        privacy.check_files(_selected(tmp_path, name, b"{}"), profile="model")


@pytest.mark.parametrize(
    "name",
    [
        "model/base/model.safetensors",
        "model/adapters.safetensors",
        "model/head.safetensors",
    ],
)
def test_fork_tensor_admission_does_not_allow_arbitrary_metadata(
    tmp_path: Path, name: str
) -> None:
    with pytest.raises(ValueError, match="arbitrary tensor metadata"):
        privacy.check_files(
            _selected(tmp_path, name, _tensor_bytes(metadata=True)), profile="model"
        )
