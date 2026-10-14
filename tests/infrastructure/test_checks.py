# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Check failure propagation and ownership, independently of inference behavior."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from scripts.checks import main as checks


def test_child_failure_status_is_preserved() -> None:
    assert checks.run((sys.executable, "-c", "raise SystemExit(7)")) == 7


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group lifetime")
def test_timeout_reaps_owned_process_group(tmp_path: Path) -> None:
    identity = tmp_path / "child.pid"
    source = (
        "import os,pathlib,time;"
        f"pathlib.Path({str(identity)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    assert checks.run((sys.executable, "-c", source), timeout=1) == 124
    pid = int(identity.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)


def test_failed_check_stops_later_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(checks, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["checks"])
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        checks,
        "commands",
        lambda python, reports: ((python, "first"), (python, "later")),
    )

    def fail(command: tuple[str, ...], timeout: int = 180) -> int:
        calls.append(command)
        return 3

    monkeypatch.setattr(checks, "run", fail)
    assert checks.main() == 3
    assert len(calls) == 1


def test_mlx_requirement_rejects_unsupported_platform_before_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["checks", "--require-mlx"])
    monkeypatch.setattr(sys, "platform", "linux")
    assert checks.main() == 1


def test_relativize_junit_strips_host_checkout_paths(tmp_path: Path) -> None:
    from scripts.package.privacy import check_text

    root = Path("/" + "home/runner/work/wyat/wyat")
    report = tmp_path / "tests.xml"
    report.write_text(
        f"<skipped>('{root.as_posix()}/tests/test_mlx.py', 10, 'Skipped')</skipped>",
        encoding="utf-8",
    )
    checks.relativize_junit(report, root)
    assert report.read_text(encoding="utf-8") == (
        "<skipped>('tests/test_mlx.py', 10, 'Skipped')</skipped>"
    )
    check_text("tests.xml", report.read_bytes())
