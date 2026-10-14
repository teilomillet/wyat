# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Run the same bounded quality checks locally and in CI."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Arguments(argparse.Namespace):
    require_mlx: bool
    require_metal: bool


def commands(python: str, reports: Path) -> tuple[tuple[str, ...], ...]:
    """Check exactly the maintained public source selection, including its tests."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    selected = configuration["tool"]["hatch"]["build"]["targets"]["sdist"][
        "only-include"
    ]
    source = tuple(name for name in selected if name.endswith(".py"))
    tests = tuple(name for name in source if name.startswith("tests/"))
    return (
        (python, "-m", "ruff", "check", *source),
        (python, "-m", "ruff", "format", "--check", *source),
        (
            python,
            "-m",
            "mypy",
            "--explicit-package-bases",
            "--ignore-missing-imports",
            "--follow-imports=silent",
            *source,
        ),
        (
            python,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-x",
            "--tb=short",
            *tests,
            f"--junitxml={reports / 'tests.xml'}",
            "--cov=wyat",
            "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            f"--cov-report=xml:{reports / 'coverage.xml'}",
            f"--cov-report=json:{reports / 'coverage.json'}",
        ),
    )


def stop(child: subprocess.Popen[bytes]) -> None:
    """Stop only this check's owned process group, including its descendants."""
    try:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGKILL)
        elif child.poll() is None:
            child.kill()
    except ProcessLookupError:
        pass


def run(command: tuple[str, ...], timeout: int = 180) -> int:
    print("+ " + " ".join(command), flush=True)
    child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Check exceeded {timeout} seconds", file=sys.stderr)
            return 124
    finally:
        stop(child)
        child.wait()


def relativize_junit(path: Path, root: Path = ROOT) -> None:
    """Replace checkout root prefixes in JUnit XML with relative paths."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    for candidate in {root, root.resolve()}:
        for prefix in {
            f"{candidate.as_posix()}/",
            f"{candidate}{os.sep}",
            str(candidate).replace("\\", "\\\\") + "\\\\",
        }:
            text = text.replace(prefix, "")
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-mlx", action="store_true", help="require native MLX")
    parser.add_argument(
        "--require-metal", action="store_true", help="require Metal GPU"
    )
    arguments = parser.parse_args(namespace=Arguments())
    if arguments.require_mlx or arguments.require_metal:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            print("Native MLX checks require Apple Silicon macOS", file=sys.stderr)
            return 1
        import mlx.core as mx
        import mlx_lm  # noqa: F401 -- admit both parts of the optional runtime.

        metal = mx.metal.is_available()
        print(f"MLX available; Metal available: {metal}", flush=True)
        if arguments.require_metal and not metal:
            print("A usable Metal GPU is required for this check", file=sys.stderr)
            return 1
    reports = ROOT / "artifacts"
    reports.mkdir(exist_ok=True)
    for command in commands(sys.executable, reports):
        # Hosted Apple Silicon needs more time than the local native runner.
        timeout = 600 if command[1:3] == ("-m", "pytest") else 180
        status = run(command, timeout=timeout)
        relativize_junit(reports / "tests.xml", ROOT)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
