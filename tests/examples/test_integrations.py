# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Integration examples run without weights when their library is installed."""

import re
import runpy
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import wyat

ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS = ROOT / "examples/use_cases/integrations"
LIBRARIES = {
    "01_langchain_router.py": "langchain_core",
    "02_langchain_rerank.py": "langchain_core",
    "03_langgraph_workflow.py": "langgraph",
    "04_llamaindex_rerank.py": "llama_index.core",
    "05_bm25_rag.py": "rank_bm25",
    "06_mcp_server.py": "mcp.server.mcpserver",
    "07_pandas_labels.py": "pandas",
    "08_pydantic_schema.py": "pydantic",
}


def test_every_integration_is_listed() -> None:
    names = {path.name for path in INTEGRATIONS.glob("[0-9][0-9]_*.py")}
    assert names == set(LIBRARIES)
    readme = (INTEGRATIONS / "README.md").read_text(encoding="utf-8")
    assert all(f"({name})" in readme for name in names)


@pytest.mark.parametrize("name", sorted(LIBRARIES))
def test_integration_runs_in_demo_mode(
    name: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip(
        LIBRARIES[name],
        reason="example library not installed; `uv sync --group examples`",
    )

    def unexpected_load(*args: object, **kwargs: object) -> wyat.Model:
        raise AssertionError("demo mode must not load weights")

    monkeypatch.setattr(wyat, "load", unexpected_load)
    monkeypatch.delitem(sys.modules, "common", raising=False)
    monkeypatch.syspath_prepend(str(INTEGRATIONS))
    monkeypatch.setattr(sys, "argv", [name, "--demo"])
    runpy.run_path(str(INTEGRATIONS / name), run_name="__main__")
    captured = capsys.readouterr()
    # The banner goes to stderr so stdout stays usable as a protocol stream.
    assert captured.err.startswith("Demo mode")
    assert "Demo mode" not in captured.out and captured.out.strip()


def test_requirements_match_the_project() -> None:
    """Each example asks for the same floor the project's examples group pins."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    declared = [
        *project["dependency-groups"]["examples"],
        *project["project"]["optional-dependencies"]["server"],
    ]
    for name in LIBRARIES:
        source = (INTEGRATIONS / name).read_text("utf-8")
        (asked,) = re.findall(r'missing\("([^"]+)", error\)', source)
        assert any(entry.startswith(asked + ",") for entry in declared), name
        assert f"--with '{asked}' python {name}" in source, name


def run_without(
    name: str, monkeypatch: pytest.MonkeyPatch, installed: str | None
) -> str:
    """Run ``name`` with its library unimportable; return the exit message."""
    # None entries make imports of the package, and of any submodule an
    # earlier test already loaded, raise ImportError.
    package = LIBRARIES[name].split(".")[0]
    for module in [m for m in sys.modules if m.split(".")[0] == package]:
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setitem(sys.modules, package, None)

    def version(distribution: str) -> str:
        if installed is None:
            raise metadata.PackageNotFoundError(distribution)
        return installed

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.delitem(sys.modules, "common", raising=False)
    monkeypatch.syspath_prepend(str(INTEGRATIONS))
    monkeypatch.setattr(sys, "argv", [name, "--demo"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(INTEGRATIONS / name), run_name="__main__")
    return str(stopped.value.code)


@pytest.mark.parametrize("name", sorted(LIBRARIES))
def test_missing_library_says_what_to_install(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = run_without(name, monkeypatch, installed=None)
    assert "which Wyat does not install" in message
    assert f"python {name} --demo" in message
    assert "uv sync --group examples" in message


@pytest.mark.parametrize("name", sorted(LIBRARIES))
def test_outdated_library_says_to_upgrade(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = run_without(name, monkeypatch, installed="0.0.1")
    assert "0.0.1 fails to import" in message
    assert "pip install --upgrade" in message
