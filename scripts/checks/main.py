"""Run the same bounded quality checks locally and in CI."""

from __future__ import annotations

import argparse
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class Arguments(argparse.Namespace):
    require_mlx: bool
    require_metal: bool


def commands(python: str, reports: Path) -> tuple[tuple[str, ...], ...]:
    """Keep maintained code scopes explicit; historical snapshots stay sealed."""
    lint_scope = ("wyat", "tests", "benchmarks", "examples", "scripts", "main.py")
    type_scope = (
        "wyat",
        "tests",
        "benchmarks/analysis",
        "benchmarks/real/dmbench",
        "main.py",
    )
    exclude = ("--exclude", "benchmarks/analysis/results")
    return (
        (python, "-m", "ruff", "check", *lint_scope, *exclude),
        (python, "-m", "ruff", "format", "--check", *lint_scope, *exclude),
        (
            python,
            "-m",
            "mypy",
            "--ignore-missing-imports",
            "--follow-imports=silent",
            *type_scope,
            *exclude,
        ),
        (
            python,
            "-m",
            "mypy",
            "--explicit-package-bases",
            "--ignore-missing-imports",
            "--follow-imports=silent",
            "examples",
            "scripts",
        ),
        (
            python,
            "-m",
            "pytest",
            "-q",
            "-rs",
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
        status = run(command)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
