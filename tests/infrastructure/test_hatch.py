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


def test_hatchling_loads_hook_outside_source_import_path(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    # Reproduce the backend loader in a fresh process with neither cwd nor
    # PYTHONPATH making the checkout importable.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from hatchling.builders.hooks.custom import CustomBuildHook; "
            "import sys; "
            "hook = CustomBuildHook(sys.argv[1], "
            "{'path': 'scripts/package/hatch.py'}, None, None, None, 'wheel'); "
            "assert hook.target_name == 'wheel'",
            str(root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_build_rejects_sensitive_contents_in_admitted_files(
    tmp_path: Path, target: str
) -> None:
    (tmp_path / "wyat").mkdir()
    (tmp_path / "wyat/__init__.py").write_text("public = True\n")
    (tmp_path / "README.md").write_text("/" + "Users/" + "private/research/train.jsonl")
    project(tmp_path, ["wyat/__init__.py", "README.md"])
    with pytest.raises(ValueError):
        hook(tmp_path, target).initialize("standard", {})


@pytest.mark.parametrize("escaped", [False, True])
def test_build_diagnostics_do_not_echo_secret_file_names(
    tmp_path: Path, escaped: bool
) -> None:
    secret = "hf_" + "A" * 32
    selected = (r"\u0068f_" + "A" * 32 if escaped else secret) + ".py"
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.sdist]\nonly-include=["' + selected + '"]\n'
    )
    with pytest.raises(ValueError) as rejected:
        hook(tmp_path, "wheel").initialize("standard", {})
    assert secret not in str(rejected.value)
    assert "privacy check failed" in str(rejected.value)
