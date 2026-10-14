"""The build backend admits only explicitly reviewed regular source files."""

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.package.hatch import CustomBuildHook


def hook(root: Path, target: str) -> CustomBuildHook:
    return CustomBuildHook(str(root), {}, None, None, None, target)  # type: ignore[arg-type]


def project(root: Path, names: list[str]) -> None:
    (root / "pyproject.toml").write_text(
        "[tool.hatch.build.targets.sdist]\nonly-include=" + repr(names) + "\n"
    )


def test_wheel_uses_only_admitted_runtime_files(tmp_path: Path) -> None:
    (tmp_path / "wyat").mkdir()
    (tmp_path / "wyat/__init__.py").write_text("public = True\n")
    (tmp_path / "wyat/private.py").write_text("private = True\n")
    (tmp_path / "README.md").write_text("Public documentation\n")
    project(tmp_path, ["wyat/__init__.py", "README.md"])
    data: dict[str, object] = {}
    hook(tmp_path, "wheel").initialize("standard", data)
    assert data["force_include"] == {
        str(tmp_path / "wyat/__init__.py"): "wyat/__init__.py"
    }
    data = {}
    hook(tmp_path, "sdist").initialize("standard", data)
    assert data == {}

    data = {}
    hook(tmp_path, "wheel").initialize("editable", data)
    assert data == {}


def test_editable_wheel_links_source_without_copying_runtime(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from hatchling.build import build_editable; import sys; "
            "print(build_editable(sys.argv[1]))",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(tmp_path / result.stdout.strip()) as archive:
        assert archive.read("_wyat.pth").decode() == str(root)
        assert not any(name.startswith("wyat/") for name in archive.namelist())


@pytest.mark.parametrize("name", ["wyat", "../private", "/private", "wyat/*.py"])
def test_build_rejects_nonliteral_or_missing_files(tmp_path: Path, name: str) -> None:
    (tmp_path / "wyat").mkdir()
    project(tmp_path, [name])
    with pytest.raises(ValueError):
        hook(tmp_path, "wheel").initialize("standard", {})


def test_build_rejects_symbolic_parent(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "__init__.py").write_text("private = True\n")
    (tmp_path / "wyat").symlink_to(private, target_is_directory=True)
    project(tmp_path, ["wyat/__init__.py"])
    with pytest.raises(ValueError, match="symbolic"):
        hook(tmp_path, "wheel").initialize("standard", {})
