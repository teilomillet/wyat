# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Exercise release guards with fake Git/registry effects, never publication."""

from __future__ import annotations

import ast
import hashlib
import io
import subprocess
import textwrap
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from email.message import Message
from pathlib import Path
from types import CodeType
from typing import TextIO, cast

import pytest
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
WHEEL = "wyat-0.0.2-py3-none-any.whl"
SDIST = "wyat-0.0.2.tar.gz"


def mapping(value: object) -> dict[object, object]:
    assert isinstance(value, dict)
    return cast(dict[object, object], value)


def workflow() -> dict[object, object]:
    # PyYAML's untyped external boundary is narrowed before any traversal.
    value: object = yaml.safe_load(
        (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    return mapping(value)


def job(name: str) -> dict[object, object]:
    return mapping(mapping(workflow()["jobs"])[name])


def steps(name: str) -> tuple[dict[object, object], ...]:
    value = job(name)["steps"]
    assert isinstance(value, list)
    return tuple(mapping(item) for item in cast(list[object], value))


def script(name: str) -> CodeType:
    runs = [step["run"] for step in steps(name) if "run" in step]
    assert len(runs) == 1
    run = runs[0]
    assert isinstance(run, str)
    marker = "python3 - <<'PY'\n"
    assert run.startswith(marker) and run.rstrip().endswith("\nPY")
    source = textwrap.dedent(run[len(marker) :].rsplit("\nPY", 1)[0])
    return compile(ast.parse(source), f"release.yml:{name}", "exec")


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> dict[str, io.StringIO]:
    values = {"output": io.StringIO(), "summary": io.StringIO()}
    for key, value in {
        "WORKFLOW_REPOSITORY": "teilomillet/wyat",
        "WORKFLOW_REF": "refs/heads/main",
        "RELEASE_TAG": "v0.0.2",
        "GITHUB_OUTPUT": "output",
        "GITHUB_STEP_SUMMARY": "summary",
    }.items():
        monkeypatch.setenv(key, value)
    return values


def prepare(reports: dict[str, io.StringIO]) -> None:
    @contextmanager
    def capture_open(path: str, mode: str, *, encoding: str) -> Iterator[TextIO]:
        assert mode == "a" and encoding == "utf-8"
        yield reports[path]

    exec(script("prepare"), {"open": capture_open})


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def output(command: list[str], *, text: bool) -> str:
        assert text is True
        calls.append(tuple(command))
        if command == ["git", "rev-parse", "--verify", "refs/tags/v0.0.2^{commit}"]:
            return COMMIT + "\n"
        assert command == ["git", "show", f"{COMMIT}:pyproject.toml"]
        return '[project]\nname="wyat"\nversion="0.0.2"\n'

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert command == ["git", "merge-base", "--is-ancestor", COMMIT, "origin/main"]
        calls.append(tuple(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_existing_main_tag_emits_exact_commit_and_version(
    reports: dict[str, io.StringIO], git: list[tuple[str, ...]]
) -> None:
    prepare(reports)
    assert reports["output"].getvalue() == (
        f"commit={COMMIT}\ntag=v0.0.2\nversion=0.0.2\n"
    )
    assert len(git) == 3


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("WORKFLOW_REPOSITORY", "fork/wyat"),
        ("WORKFLOW_REF", "refs/heads/feature"),
        ("RELEASE_TAG", "../../main"),
        ("RELEASE_TAG", "v0.0.2\ninjected=yes"),
        ("RELEASE_TAG", "v0.0.2$(touch injected)"),
    ],
)
def test_untrusted_dispatch_stops_before_git_or_outputs(
    key: str,
    value: str,
    reports: dict[str, io.StringIO],
    git: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit):
        prepare(reports)
    assert git == [] and reports["output"].getvalue() == ""


@pytest.mark.parametrize("failure", ["missing-tag", "unmerged", "version", "name"])
def test_git_and_metadata_rejections_preserve_no_release_output(
    failure: str,
    reports: dict[str, io.StringIO],
    git: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def output(command: list[str], *, text: bool) -> str:
        assert text is True
        if command[1] == "rev-parse":
            if failure == "missing-tag":
                raise subprocess.CalledProcessError(128, command)
            return COMMIT
        version = "0.0.3" if failure == "version" else "0.0.2"
        name = "other" if failure == "name" else "wyat"
        return f'[project]\nname="{name}"\nversion="{version}"\n'

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        if failure == "unmerged":
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((SystemExit, subprocess.CalledProcessError)):
        prepare(reports)
    assert reports["output"].getvalue() == ""


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_VERSION", "0.0.2")
    directory = tmp_path / "checked"
    directory.mkdir()
    for name in (WHEEL, SDIST):
        (directory / name).write_bytes(f"fixture:{name}".encode())
    manifest = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(directory.iterdir())
    )
    (directory / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    return directory


class Response(AbstractContextManager["Response"]):
    def __exit__(self, *_: object) -> None:
        return None


def registry(monkeypatch: pytest.MonkeyPatch, status: int) -> list[str]:
    calls: list[str] = []

    def open_url(url: str, *, timeout: int) -> Response:
        assert timeout == 15
        assert url == "https://pypi.org/pypi/wyat/0.0.2/json"
        calls.append(url)
        if status != 200:
            raise urllib.error.HTTPError(url, status, "fixture", Message(), None)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    return calls


def test_only_registry_404_admits_exact_checked_artifacts(
    artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = {name: (artifacts / name).read_bytes() for name in (WHEEL, SDIST)}
    calls = registry(monkeypatch, 404)
    exec(script("publish"), {})
    destination = artifacts.parent / "dist"
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original
    assert (artifacts / "SHA256SUMS").is_file()
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 401, 429, 500, 503])
def test_existing_release_or_registry_error_never_stages_uploads(
    status: int, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry(monkeypatch, status)
    with pytest.raises((SystemExit, urllib.error.HTTPError)):
        exec(script("publish"), {})
    assert not (artifacts.parent / "dist").exists()
    assert all((artifacts / name).is_file() for name in (WHEEL, SDIST))


@pytest.mark.parametrize(
    "failure",
    ["modified", "extra", "duplicate", "traversal", "inventory", "wrong-kind"],
)
def test_invalid_artifacts_stop_before_registry_and_upload_staging(
    failure: str, artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = artifacts / "SHA256SUMS"
    if failure == "modified":
        (artifacts / WHEEL).write_bytes(b"tampered")
    elif failure == "extra":
        (artifacts / "unexpected").write_bytes(b"unexpected")
    elif failure == "duplicate":
        manifest.write_text(manifest.read_text() * 2)
    elif failure == "traversal":
        manifest.write_text(manifest.read_text().replace(WHEEL, "../" + WHEEL))
    elif failure == "inventory":
        manifest.write_text(manifest.read_text().replace(WHEEL, "other.whl"))
    else:
        (artifacts / SDIST).rename(artifacts / "other.whl")
    calls = registry(monkeypatch, 404)
    with pytest.raises(SystemExit):
        exec(script("publish"), {})
    assert calls == [] and not (artifacts.parent / "dist").exists()


def test_publication_policy_is_disabled_and_reuses_gated_ci_artifact() -> None:
    value = workflow()
    # YAML 1.1 resolves the unquoted GitHub key "on" to True.
    events = mapping(value.get("on", value.get(True)))
    assert set(events) == {"workflow_dispatch"}
    inputs = mapping(mapping(events["workflow_dispatch"])["inputs"])
    assert mapping(inputs["publish"])["default"] is False
    assert mapping(value["permissions"]) == {"contents": "read"}
    assert mapping(value["concurrency"])["cancel-in-progress"] is False
    checked = job("checks")
    assert checked["needs"] == "prepare"
    assert checked["uses"] == "./.github/workflows/ci.yml"
    assert mapping(checked["with"]) == {
        "ref": "${{ needs.prepare.outputs.commit }}",
        "release_tag": "${{ needs.prepare.outputs.tag }}",
    }
    published = job("publish")
    assert published["needs"] == ["prepare", "checks"]
    assert published["if"] == "inputs.publish && vars.PYPI_PUBLISH_ENABLED == 'true'"
    assert mapping(published["environment"])["name"] == "pypi"
    assert mapping(published["permissions"]) == {"id-token": "write"}
    assert "permissions" not in job("prepare") and "permissions" not in checked
    used = [step for step in steps("publish") if "uses" in step]
    assert len(used) == 2
    assert str(used[0]["uses"]).startswith("actions/download-artifact@")
    assert mapping(used[0]["with"]) == {
        "name": "ci-distributions",
        "path": "checked",
    }
    assert str(used[1]["uses"]).startswith("pypa/gh-action-pypi-publish@")
    assert mapping(used[1]["with"]) == {
        "packages-dir": "dist/",
        "skip-existing": False,
        "attestations": True,
    }


def test_ci_checks_contents_before_tests_and_retaining_reports() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    assert configuration["tool"]["coverage"]["run"]["relative_files"] is True
    value = mapping(
        yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
    )
    checks = mapping(mapping(value["jobs"])["checks"])
    entries = checks["steps"]
    assert isinstance(entries, list)
    by_name = {
        str(mapping(entry)["name"]): mapping(entry)
        for entry in cast(list[object], entries)
    }
    names = list(by_name)
    scan = "Check public source contents before execution"
    tests = "Check types, style, and regressions"
    assert names.index(scan) < names.index(tests)
    assert names.index(scan) < names.index("Install locked dependencies")
    assert "scripts/package/privacy.py --source" in str(by_name[scan]["run"])
    assert "--no-project --no-sync --python" in str(by_name[scan]["run"])
    reports = by_name["Check test report contents before retaining them"]
    assert reports["id"] == "report-privacy" and reports["if"] == "always()"
    assert "check_text(name, path.read_bytes())" in str(reports["run"])
    retained = by_name["Retain test results, including skips"]
    assert retained["if"] == "always() && steps.report-privacy.outcome == 'success'"
