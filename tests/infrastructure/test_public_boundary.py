# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Release admission is independent of private files in the development checkout."""

import ast
import re
import tomllib
from pathlib import Path

from scripts.checks import main as checks
from scripts.package.check import required_sources, runtime_files

ROOT = Path(__file__).resolve().parents[2]


def test_public_inventory_excludes_research_and_execution_state() -> None:
    selected = required_sources(ROOT)
    # macOS can hide case mismatches that fail on Linux checkouts.
    for name in selected:
        parent = ROOT
        for part in Path(name).parts:
            assert part in {entry.name for entry in parent.iterdir()}, name
            parent = parent / part
    private_roots = {"benchmarks", "artifacts", "data", "training", ".git", ".models"}
    assert not any(Path(name).parts[0] in private_roots for name in selected)
    runtime = runtime_files(ROOT)
    assert "wyat/cli.py" in runtime
    assert "wyat/evaluation.py" in runtime
    assert "wyat/tasks.py" in runtime
    assert "wyat/loading/native.py" in runtime
    assert not any(
        name.startswith(("wyat/bench/", "wyat/data/")) or name in {"wyat/benchmark.py"}
        for name in runtime
    )


def test_public_python_imports_resolve_to_admitted_project_modules() -> None:
    selected = required_sources(ROOT)
    for name in selected:
        if not name.endswith(".py"):
            continue
        tree = ast.parse((ROOT / name).read_text("utf-8"), filename=name)
        for node in ast.walk(tree):
            imports: list[str] = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package = Path(name).parent.parts
                    prefix = package[: len(package) - node.level + 1]
                    imports = [".".join((*prefix, node.module or "")).rstrip(".")]
                elif node.module:
                    imports = [node.module]
            for module in imports:
                assert not module.startswith(("benchmarks", "training")), (name, module)
                if module.startswith("wyat"):
                    path = module.replace(".", "/")
                    assert (
                        path + ".py" in selected or path + "/__init__.py" in selected
                    ), (name, module)


def test_public_documentation_local_links_have_public_targets() -> None:
    selected = required_sources(ROOT)
    for name in selected:
        if not name.endswith(".md"):
            continue
        content = (ROOT / name).read_text("utf-8")
        assert "/Users/" not in content and "ssh gpu" not in content, name
        for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", content):
            target = target.split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = (ROOT / name).parent / target
            relative = path.resolve().relative_to(ROOT).as_posix()
            assert relative in selected, (name, target)


def test_quality_commands_select_only_public_python_and_tests() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    selected = set(
        configuration["tool"]["hatch"]["build"]["targets"]["sdist"]["only-include"]
    )
    commands = checks.commands("python", ROOT / "artifacts")
    for command in commands:
        paths = {part for part in command if part.endswith(".py")}
        assert paths <= selected
        assert paths
    pytest = commands[-1]
    assert "tests/test_core.py" in pytest
    assert not any("benchmarks" in part for part in pytest)
