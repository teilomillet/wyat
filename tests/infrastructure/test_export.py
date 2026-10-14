# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Public exports are exact, fresh, history-free snapshots with external hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.package import check, export


@pytest.fixture
def source(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "wyat").mkdir(parents=True)
    (root / "wyat/__init__.py").write_bytes(b"public = True\n")
    (root / "README.md").write_bytes(b"Public readme\n")
    (root / "pyproject.toml").write_text(
        "[tool.hatch.build.targets.sdist]\n"
        'only-include=["wyat/__init__.py", "README.md", "pyproject.toml"]\n',
        encoding="utf-8",
    )
    return root


def test_export_contains_only_selected_bytes_and_external_hashes(
    source: Path, tmp_path: Path
) -> None:
    for name in (".git/config", "training/secret.json", "wyat/private_training.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Private\n")
    destination = tmp_path / "export"
    manifest = export.export_source(destination, source)
    expected = check.required_sources(source)
    assert {
        p.relative_to(destination).as_posix()
        for p in destination.rglob("*")
        if p.is_file()
    } == expected
    assert manifest == tmp_path / "export.SHA256SUMS"
    assert manifest.read_text() == "".join(
        f"{hashlib.sha256((source / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(expected)
    )
    for name in expected:
        assert (destination / name).read_bytes() == (source / name).read_bytes()


@pytest.mark.parametrize("collision", ["directory", "file", "manifest"])
def test_existing_output_is_preserved(
    source: Path, tmp_path: Path, collision: str
) -> None:
    destination = tmp_path / "export"
    path = (
        destination.with_name("export.SHA256SUMS")
        if collision == "manifest"
        else destination
    )
    if collision == "directory":
        path.mkdir()
        path = path / "existing"
    path.write_bytes(b"Keep me\n")
    with pytest.raises(FileExistsError):
        export.export_source(destination, source)
    assert path.read_bytes() == b"Keep me\n"


@pytest.mark.parametrize("link", ["destination", "parent", "manifest", "source"])
def test_symbolic_paths_are_rejected_without_affecting_target(
    source: Path, tmp_path: Path, link: str
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    marker = private / "keep"
    marker.write_bytes(b"Keep me\n")
    destination = tmp_path / "export"
    if link == "destination":
        destination.symlink_to(private, target_is_directory=True)
    elif link == "parent":
        parent = tmp_path / "linked-parent"
        parent.symlink_to(private, target_is_directory=True)
        destination = parent / "export"
    elif link == "manifest":
        destination.with_name("export.SHA256SUMS").symlink_to(marker)
    else:
        (source / "README.md").unlink()
        (source / "README.md").symlink_to(marker)
    with pytest.raises(ValueError, match="[Ss]ymbolic"):
        export.export_source(destination, source)
    assert marker.read_bytes() == b"Keep me\n"
    assert set(private.iterdir()) == {marker}


@pytest.mark.parametrize("relative", [".", "wyat/export", ".git/export"])
def test_export_cannot_nest_in_public_sources_or_git(
    source: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="outside"):
        export.export_source(source / relative, source)
    assert not (source / "wyat/export").exists()


def test_failed_verification_removes_only_owned_outputs(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "export"
    sibling = tmp_path / "other"
    sibling.write_bytes(b"Keep me\n")

    def fail(*_: object) -> None:
        raise ValueError("Verification failed")

    monkeypatch.setattr(export, "verify_export", fail)
    with pytest.raises(ValueError, match="Verification failed"):
        export.export_source(destination, source)
    assert not destination.exists()
    assert not destination.with_name("export.SHA256SUMS").exists()
    assert sibling.read_bytes() == b"Keep me\n"
    assert (source / "README.md").read_bytes() == b"Public readme\n"


def test_manifest_created_during_copy_is_preserved(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "export"
    manifest = tmp_path / "export.SHA256SUMS"

    def collide(*_: object) -> None:
        manifest.write_bytes(b"Created by another writer\n")

    monkeypatch.setattr(export, "verify_export", collide)
    with pytest.raises(FileExistsError):
        export.export_source(destination, source)
    assert not destination.exists()
    assert manifest.read_bytes() == b"Created by another writer\n"


def test_source_mutation_rejects_export(source: Path, tmp_path: Path) -> None:
    destination = tmp_path / "export"
    export.export_source(destination, source)
    contents = {
        name: (source / name).read_bytes() for name in check.required_sources(source)
    }
    (source / "README.md").write_bytes(b"Changed\n")
    with pytest.raises(ValueError, match="changed while copying"):
        export.verify_export(destination, contents, source)


def test_export_content_failure_creates_no_output(source: Path, tmp_path: Path) -> None:
    private_path = "/" + "Users/" + "private/research/train.jsonl"
    (source / "README.md").write_text(private_path)
    destination = tmp_path / "export"
    with pytest.raises(ValueError):
        export.export_source(destination, source)
    assert not destination.exists()
    assert not destination.with_name("export.SHA256SUMS").exists()
