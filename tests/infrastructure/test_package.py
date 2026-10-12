"""Independent malformed archives and metadata challenge release admission."""

from __future__ import annotations

import base64
import hashlib
import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.package import check

PROJECT = check.Project("wyat", "0.0.1", ">=3.12", "Apache-2.0", (), ())
METADATA = b"""Metadata-Version: 2.4
Name: wyat
Version: 0.0.1
Requires-Python: >=3.12
License-Expression: Apache-2.0
License-File: LICENSE
License-File: NOTICE

Example package.
"""


def archive(path: Path, members: tuple[tuple[str, bytes, bytes], ...]) -> None:
    with tarfile.open(path, "w:gz") as output:
        for name, raw, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(raw) if kind == tarfile.REGTYPE else 0
            member.linkname = "../outside"
            output.addfile(member, io.BytesIO(raw) if member.isfile() else None)


def make_wheel(path: Path, root: Path) -> None:
    info = "wyat-0.0.1.dist-info"
    files = {
        "wyat/__init__.py": (root / "wyat/__init__.py").read_bytes(),
        "wyat/py.typed": b"",
        f"{info}/METADATA": METADATA,
        f"{info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{info}/entry_points.txt": b"[console_scripts]\nwyat = wyat.cli:main\n",
        f"{info}/licenses/LICENSE": b"License text",
        f"{info}/licenses/NOTICE": b"Attribution text",
    }
    rows = []
    for name, raw in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=")
        rows.append(f"{name},sha256={digest.decode('ascii')},{len(raw)}\n")
    files[f"{info}/RECORD"] = ("".join(rows) + f"{info}/RECORD,,\n").encode("utf-8")
    with zipfile.ZipFile(path, "w") as output:
        for name, raw in files.items():
            output.writestr(name, raw)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "wyat").mkdir(parents=True)
    (source / "wyat/__init__.py").write_text("value = 7\n", encoding="utf-8")
    (source / "wyat/py.typed").write_bytes(b"")
    (source / "LICENSE").write_bytes(b"License text")
    (source / "NOTICE").write_bytes(b"Attribution text")
    (source / "pyproject.toml").write_text(
        '[project]\nname="wyat"\nversion="0.0.1"\nrequires-python=">=3.12"\n'
        'license="Apache-2.0"\ndependencies=[]\n'
        "[tool.hatch.build.targets.sdist]\n"
        'only-include=["wyat/__init__.py", "wyat/py.typed", "README.md", '
        '"LICENSE", "NOTICE", "pyproject.toml"]\n',
        encoding="utf-8",
    )
    for name in ("uv.lock", "README.md", "CONTRIBUTING.md"):
        (source / name).write_bytes(b"Example\n")
    return source


@given(
    prefix=st.sampled_from(("../", "/", "./", "a/../", "a//", "a\\", "C:/")),
    tail=st.text(alphabet="abcdefgh", min_size=1, max_size=20),
)
def test_unsafe_paths_are_rejected(prefix: str, tail: str) -> None:
    with pytest.raises(ValueError, match="Unsafe archive path"):
        check.safe_name(prefix + tail)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE])
def test_source_links_and_special_files_are_rejected(
    tmp_path: Path, kind: bytes
) -> None:
    path = tmp_path / "source.tar.gz"
    archive(path, (("wyat-0.0.1/link", b"", kind),))
    with pytest.raises(ValueError, match="regular files"):
        check.source_files(path, PROJECT)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("kind", ["duplicate", "traversal", "foreign_root", "weights"])
def test_source_archive_failures(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "source.tar.gz"
    name = {
        "duplicate": "wyat-0.0.1/a.py",
        "traversal": "wyat-0.0.1/../a.py",
        "foreign_root": "other-1.0/a.py",
        "weights": "wyat-0.0.1/weights.safetensors",
    }[kind]
    members = ((name, b"data", tarfile.REGTYPE),)
    archive(path, members * 2 if kind == "duplicate" else members)
    with pytest.raises(ValueError):
        check.source_files(path, PROJECT)


def test_wheel_symlink_and_duplicate_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wheel.whl"
    member = zipfile.ZipInfo("wyat/link.py")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as output:
        output.writestr(member, b"../../outside")
    with pytest.raises(ValueError, match="symbolic link"):
        check.wheel_files(path)
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("a.py", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr("a.py", b"second")
    with pytest.raises(ValueError, match="Duplicate wheel"):
        check.wheel_files(path)


def test_wheel_matches_source_and_independent_record(
    root: Path, tmp_path: Path
) -> None:
    path = tmp_path / "wyat-0.0.1-py3-none-any.whl"
    make_wheel(path, root)
    files = check.check_wheel(path, root, PROJECT)
    files["wyat/__init__.py"] = b"value = 8\n"
    with pytest.raises(ValueError, match="RECORD digest"):
        check.check_record(files, "wyat-0.0.1.dist-info/RECORD")
    (root / "wyat/__init__.py").write_bytes(b"value = 9\n")
    with pytest.raises(ValueError, match="runtime resources differ"):
        check.check_wheel(path, root, PROJECT)


@pytest.mark.parametrize(
    "before,after",
    [
        (b"Name: wyat", b"Name: other"),
        (b"Version: 0.0.1", b"Version: 9.0.0"),
        (b"Requires-Python: >=3.12", b"Requires-Python: >=3.10"),
        (b"License-Expression: Apache-2.0", b"License-Expression: MIT"),
        (b"License-File: NOTICE", b"Provides-Extra: surprise"),
        (b"License-File: NOTICE", b"License-File: NOTICE\nRequires-Dist: torch"),
    ],
)
def test_metadata_drift_is_rejected(before: bytes, after: bytes) -> None:
    check.check_metadata(METADATA, PROJECT)
    with pytest.raises(ValueError):
        check.check_metadata(METADATA.replace(before, after), PROJECT)


@pytest.mark.parametrize("count", [0, 2])
def test_distribution_cardinality(tmp_path: Path, count: int) -> None:
    (tmp_path / "wyat-0.0.1.tar.gz").write_bytes(b"source")
    for index in range(count):
        (tmp_path / f"wheel-{index}.whl").write_bytes(b"wheel")
    with pytest.raises(ValueError, match="exactly one wheel"):
        check.distributions(tmp_path)


def test_source_bytes_and_required_files(root: Path, tmp_path: Path) -> None:
    path = tmp_path / "source.tar.gz"
    members = tuple(
        ("wyat-0.0.1/" + name, (root / name).read_bytes(), tarfile.REGTYPE)
        for name in sorted(check.required_sources(root))
    ) + (("wyat-0.0.1/PKG-INFO", METADATA, tarfile.REGTYPE),)
    archive(path, members)
    assert check.check_source(path, root, PROJECT)["wyat/__init__.py"] == b"value = 7\n"
    (root / "README.md").write_bytes(b"Changed\n")
    with pytest.raises(ValueError, match="differs from the checkout"):
        check.check_source(path, root, PROJECT)
    archive(path, members[1:])
    with pytest.raises(ValueError, match="omits required"):
        check.check_source(path, root, PROJECT)


def test_source_selection_keeps_private_training_siblings_out(root: Path) -> None:
    private = (
        "training/run.toml",
        "scripts/french_data/decision_training_run.py",
        "docs/french-first/selected-training-recipe.md",
        "tests/test_decision_training_run.py",
        "data/train.jsonl",
        "artifacts/training-review.json",
    )
    for name in private:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Private training information\n")
    assert check.required_sources(root) == {
        "wyat/__init__.py",
        "wyat/py.typed",
        "README.md",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
    }


@pytest.mark.parametrize(
    "name",
    [
        "training/run.toml",
        "scripts/french_data/train.py",
        "docs/french-first/review.md",
    ],
)
def test_injected_private_source_is_rejected(
    root: Path, tmp_path: Path, name: str
) -> None:
    private = root / name
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_bytes(b"Private information")
    path = tmp_path / "source.tar.gz"
    members = tuple(
        ("wyat-0.0.1/" + item, (root / item).read_bytes(), tarfile.REGTYPE)
        for item in sorted(check.required_sources(root))
    ) + (
        ("wyat-0.0.1/PKG-INFO", METADATA, tarfile.REGTYPE),
        ("wyat-0.0.1/" + name, private.read_bytes(), tarfile.REGTYPE),
    )
    archive(path, members)
    with pytest.raises(ValueError, match="outside the public selection"):
        check.check_source(path, root, PROJECT)


@pytest.mark.parametrize(
    "selection", ["", "../private", "/private", "wyat/**", "wyat", "absent"]
)
def test_invalid_source_selection_is_rejected(root: Path, selection: str) -> None:
    project = root / "pyproject.toml"
    project.write_text(
        '[tool.hatch.build.targets.sdist]\nonly-include=["' + selection + '"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        check.required_sources(root)


def test_source_selection_cannot_follow_symbolic_parent(
    root: Path, tmp_path: Path
) -> None:
    external = tmp_path / "private"
    external.mkdir()
    (external / "run.toml").write_bytes(b"Private information")
    (root / "link").symlink_to(external, target_is_directory=True)
    (root / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\nonly-include=["link/run.toml"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="symbolic source selection"):
        check.required_sources(root)


def test_unlisted_private_runtime_is_not_required(root: Path) -> None:
    (root / "wyat/private_training.py").write_bytes(b"private = True\n")
    assert set(check.runtime_files(root)) == {"wyat/__init__.py", "wyat/py.typed"}


def test_unlisted_runtime_cannot_enter_wheel(root: Path, tmp_path: Path) -> None:
    path = tmp_path / "wyat-0.0.1-py3-none-any.whl"
    make_wheel(path, root)
    (root / "wyat/private_training.py").write_bytes(b"private = True\n")
    check.check_wheel(path, root, PROJECT)
    with zipfile.ZipFile(path, "a") as output:
        output.writestr("wyat/private_training.py", b"private = True\n")
    with pytest.raises(ValueError, match="runtime resources differ"):
        check.check_wheel(path, root, PROJECT)


@pytest.mark.parametrize("selection", ["artifacts/private.py", "wyat/private.pyc"])
def test_forbidden_selection_is_rejected(root: Path, selection: str) -> None:
    path = root / selection
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"private")
    (root / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\nonly-include=["' + selection + '"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Forbidden source selection"):
        check.required_sources(root)


def test_duplicate_source_selection_is_rejected(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\nonly-include=["README.md", "README.md"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate source selection"):
        check.required_sources(root)


def test_hash_manifest_is_exact_and_never_rewritten(tmp_path: Path) -> None:
    wheel, source = tmp_path / "example.whl", tmp_path / "example.tar.gz"
    wheel.write_bytes(b"wheel")
    source.write_bytes(b"source")
    check.seal_hashes(tmp_path, (wheel, source))
    manifest = tmp_path / "SHA256SUMS"
    original = manifest.read_bytes()
    expected = (
        hashlib.sha256(b"source").hexdigest()
        + "  example.tar.gz\n"
        + hashlib.sha256(b"wheel").hexdigest()
        + "  example.whl\n"
    ).encode("ascii")
    assert original == expected
    check.seal_hashes(tmp_path, (wheel, source))
    wheel.write_bytes(b"modified")
    with pytest.raises(ValueError, match="does not match"):
        check.seal_hashes(tmp_path, (wheel, source))
    assert manifest.read_bytes() == original


def test_release_tag_fails_before_artifact_execution(
    root: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="match the project version exactly"):
        check.check(tmp_path / "absent", root, tag="v9.0.0")
