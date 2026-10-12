"""The build backend admits only explicitly reviewed regular source files."""

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
