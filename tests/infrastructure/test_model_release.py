# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Model publication requires attribution before effects and frozen source bytes."""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.package import model_release as release
from scripts.package import privacy


def inventory(root: Path) -> dict[str, Any]:
    pins = {
        "repo_id": "teilomillet/wyat-test",
        "parent_commit": "reviewed-parent",
        "files": {
            p.relative_to(root).as_posix(): {
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in root.rglob("*")
            if p.is_file()
        },
    }
    pins["privacy_review"] = {
        "status": "PASS",
        "policy_sha256": release.sha256(Path(privacy.__file__)),
        "files_sha256": hashlib.sha256(
            json.dumps(pins["files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    return pins


@pytest.mark.parametrize(
    "raw",
    [
        b'"""Retain the module docstring."""\n'
        b"from __future__ import annotations\nx = 1\n",
        b"#!/usr/bin/env python3\n# coding: utf-8\nx = 1\n",
        b"\xef\xbb\xbf# coding: utf-8\nx = 1\n",
        b"#!/usr/bin/python3",
        b"# Existing third-party notice remains here.\nx = 1\n",
        b"",
    ],
)
def test_headers_preserve_python_and_preamble(raw: bytes) -> None:
    signed = release.signed_python(raw)
    assert signed.count(release.HEADER) == 1
    assert ast.dump(ast.parse(signed)) == ast.dump(ast.parse(raw))
    assert release.signed_python(signed) == signed
    if raw.startswith(b"#!"):
        assert signed.splitlines()[0] == raw.splitlines()[0]
    if raw.startswith(b"\xef\xbb\xbf"):
        assert signed.startswith(b"\xef\xbb\xbf")
    if b"coding:" in raw:
        assert b"coding:" in signed.splitlines()[0] + signed.splitlines()[1]
    if b"third-party" in raw:
        assert signed.endswith(raw)


def test_prepare_all_python_is_idempotent_and_preserves_modes(tmp_path: Path) -> None:
    for name in ("example.py", "types.pyi", "runtime.PY"):
        (tmp_path / name).write_bytes(b"x = 1\n")
    (tmp_path / "example.py").chmod(0o755)
    assert set(release.prepare(tmp_path)) == {"example.py", "types.pyi", "runtime.PY"}
    assert len(release.check_attribution(tmp_path)) == 3
    assert (tmp_path / "example.py").stat().st_mode & 0o777 == 0o755
    (tmp_path / "SOURCE.json").write_text("{}")
    assert release.prepare(tmp_path) == ()


@pytest.mark.parametrize("name", sorted(release.SEALED))
def test_prepare_refuses_to_invalidate_a_sealed_release(
    tmp_path: Path, name: str
) -> None:
    (tmp_path / "runtime.py").write_bytes(b"x = 1\n")
    (tmp_path / name).write_text("sealed")
    with pytest.raises(ValueError, match="before creating"):
        release.prepare(tmp_path)
    assert (tmp_path / "runtime.py").read_bytes() == b"x = 1\n"


@pytest.mark.parametrize(
    "bad",
    [
        b"# SPDX-License-Identifier: MIT\nx = 1\n",
        b"# coding: latin-1\nx = 1\n",
        b"broken (\n",
    ],
)
def test_prepare_validates_all_files_before_writing(tmp_path: Path, bad: bytes) -> None:
    (tmp_path / "a.py").write_bytes(b"x = 1\n")
    (tmp_path / "z.py").write_bytes(bad)
    with pytest.raises((ValueError, SyntaxError)):
        release.prepare(tmp_path)
    assert (tmp_path / "a.py").read_bytes() == b"x = 1\n"
    assert (tmp_path / "z.py").read_bytes() == bad


@pytest.mark.parametrize(
    "name",
    [
        "run.sh",
        "runtime.js",
        "module.pyc",
        "module.pyd",
        "run",
        "page.html",
        "code.xyz",
        "script.txt",
    ],
)
def test_unsupported_code_cannot_escape_the_gate(tmp_path: Path, name: str) -> None:
    raw = (
        b"code without a shebang\n"
        if name in {"page.html", "code.xyz"}
        else b"#!/bin/sh\necho hello\n"
    )
    (tmp_path / name).write_bytes(raw)
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        release.check_attribution(tmp_path)


def test_links_are_rejected_before_signing(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"x = 1\n")
    root = tmp_path / "release"
    root.mkdir()
    (root / "linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="link"):
        release.prepare(root)
    assert outside.read_bytes() == b"x = 1\n"


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    (tmp_path / "example.py").write_bytes(release.HEADER + b"x = 1\n")
    weights = tmp_path / "model/native-adapters.safetensors"
    weights.parent.mkdir()
    header = json.dumps(
        {"probe": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    weights.write_bytes(
        struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    )
    return tmp_path


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    state = SimpleNamespace(
        initialized=0,
        commits=0,
        message=None,
        private=True,
        parent="reviewed-parent",
        extra_files=[],
        uploaded={},
        mutate=None,
        corrupt=False,
    )

    class Operation:
        def __init__(self, path_in_repo: str, path_or_fileobj: Path) -> None:
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj
            raw = path_or_fileobj.read_bytes()
            self.upload_info = SimpleNamespace(
                size=len(raw), sha256=hashlib.sha256(raw).digest()
            )

    class API:
        def __init__(self) -> None:
            state.initialized += 1

        def model_info(self, repo: str, **kwargs: Any) -> SimpleNamespace:
            if not kwargs:
                if state.mutate:
                    state.mutate()
                return SimpleNamespace(private=state.private, sha=state.parent)
            siblings = []
            for name, raw in state.uploaded.items():
                digest = hashlib.sha256(raw).hexdigest()
                if state.corrupt:
                    digest = "0" * 64
                siblings.append(
                    SimpleNamespace(
                        rfilename=name,
                        size=len(raw),
                        lfs=SimpleNamespace(sha256=digest)
                        if name.endswith(".safetensors")
                        else None,
                    )
                )
            return SimpleNamespace(private=state.private, siblings=siblings)

        def list_repo_files(self, repo: str, **kwargs: Any) -> list[str]:
            return list(state.extra_files)

        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            state.commits += 1
            state.message = kwargs["commit_message"]
            assert kwargs["parent_commit"] == "reviewed-parent"
            state.uploaded = {
                op.path_in_repo: op.path_or_fileobj.read_bytes()
                for op in kwargs["operations"]
            }
            return SimpleNamespace(oid="new-commit")

    def download(repo: str, name: str, **kwargs: Any) -> str:
        path = tmp_path.with_name(tmp_path.name + "-downloads") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(state.uploaded[name])
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=API,
            CommitOperationAdd=Operation,
            hf_hub_download=download,
        ),
    )
    return state


def publish(root: Path, pins: dict[str, Any]) -> str:
    return release.publish_release(
        root, pins, "teilomillet/wyat-test", "reviewed-parent"
    )


@pytest.mark.parametrize(
    "change", ["unsigned", "added", "modified", "destination", "parent"]
)
def test_bad_local_candidate_blocks_before_network(
    bundle: Path, hub: SimpleNamespace, change: str
) -> None:
    pins = inventory(bundle)
    if change == "unsigned":
        (bundle / "example.py").write_bytes(b"x = 1\n")
    elif change == "added":
        (bundle / "unreviewed.txt").write_text("extra")
    elif change == "modified":
        (bundle / "example.py").write_bytes(release.HEADER + b"x = 2\n")
    else:
        pins["repo_id" if change == "destination" else "parent_commit"] = "different"
    with pytest.raises(ValueError):
        publish(bundle, pins)
    assert hub.initialized == hub.commits == 0


@pytest.mark.parametrize("change", ["public", "parent", "extra", "local-edit"])
def test_remote_or_local_changes_block_commit(
    bundle: Path, hub: SimpleNamespace, change: str
) -> None:
    pins = inventory(bundle)
    if change == "public":
        hub.private = False
    elif change == "parent":
        hub.parent = "different"
    elif change == "extra":
        hub.extra_files = ["unreviewed.py"]
    else:
        hub.mutate = lambda: (bundle / "example.py").write_bytes(
            release.HEADER + b"x = 2\n"
        )
    with pytest.raises(ValueError):
        publish(bundle, pins)
    assert hub.commits == 0


def test_publish_uploads_only_reviewed_files_and_verifies(
    bundle: Path, hub: SimpleNamespace
) -> None:
    assert publish(bundle, inventory(bundle)) == "new-commit"
    assert set(hub.uploaded) == {"example.py", "model/native-adapters.safetensors"}
    assert hub.uploaded["example.py"].startswith(release.HEADER)
    assert hub.commits == 1


def test_committed_hash_mismatch_is_not_success(
    bundle: Path, hub: SimpleNamespace
) -> None:
    hub.corrupt = True
    with pytest.raises(ValueError, match="Published file hash differs"):
        publish(bundle, inventory(bundle))
    assert hub.commits == 1


@pytest.mark.parametrize("change", ["missing", "status", "policy", "files"])
def test_privacy_review_binds_policy_and_inventory_before_network(
    bundle: Path, hub: SimpleNamespace, change: str
) -> None:
    pins = inventory(bundle)
    if change == "missing":
        pins.pop("privacy_review")
    else:
        key = {"status": "status", "policy": "policy_sha256", "files": "files_sha256"}[
            change
        ]
        pins["privacy_review"][key] = "different"
    with pytest.raises(ValueError, match="privacy review"):
        publish(bundle, pins)
    assert hub.initialized == hub.commits == 0


def test_sensitive_reviewed_file_still_blocks_before_network(
    bundle: Path, hub: SimpleNamespace
) -> None:
    (bundle / "data.csv").write_text("query,answer\nprivate,record\n")
    with pytest.raises(ValueError):
        publish(bundle, inventory(bundle))
    assert hub.initialized == hub.commits == 0


def test_cli_check_without_inventory_still_applies_privacy(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (bundle / "data.csv").write_text("query,answer\nprivate,record\n")
    monkeypatch.setattr(sys, "argv", ["model-release", "check", str(bundle)])
    with pytest.raises(SystemExit) as stopped:
        release.main()
    assert stopped.value.code == 1


@pytest.mark.parametrize(
    "message",
    [
        None,
        1,
        "",
        "   ",
        "a" * 73,
        "first\nsecond",
        "tab\there",
        "nul\x00here",
        "del\x7fhere",
        "nel\x85here",
        "line\u2028here",
        "hf_" + "a" * 32,
    ],
)
def test_bad_commit_message_blocks_before_network(
    bundle: Path, hub: SimpleNamespace, message: object
) -> None:
    pins = inventory(bundle)
    pins["commit_message"] = message
    with pytest.raises(ValueError):
        publish(bundle, pins)
    assert hub.initialized == hub.commits == 0


def test_publish_forwards_the_exact_reviewed_commit_message(
    bundle: Path, hub: SimpleNamespace
) -> None:
    pins = inventory(bundle)
    pins["commit_message"] = "Remove private diagnostics from the Wyat model release"
    assert publish(bundle, pins) == "new-commit"
    assert hub.message == pins["commit_message"]


def test_publish_uses_the_default_message_when_omitted(
    bundle: Path, hub: SimpleNamespace
) -> None:
    assert publish(bundle, inventory(bundle)) == "new-commit"
    assert hub.message == release.DEFAULT_COMMIT_MESSAGE


@pytest.mark.parametrize("action", ["prepare", "check", "inventory", "publish"])
@pytest.mark.parametrize("content", ["credential", "private-prose"])
def test_cli_syntax_failures_do_not_echo_staged_contents(
    tmp_path: Path, action: str, content: str
) -> None:
    sensitive = (
        "hf_" + "A" * 32 if content == "credential" else "confidential-customer-detail"
    )
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "example.py").write_bytes(
        release.HEADER + f'token = "{sensitive}\n'.encode()
    )
    args = ["check" if action == "inventory" else action, str(root)]
    if action in {"inventory", "publish"}:
        pins = tmp_path / "inventory.json"
        pins.write_text(json.dumps(inventory(root)))
        args.extend(["--inventory", str(pins)])
    if action == "publish":
        args.extend(
            ["--repo-id", "teilomillet/wyat-test", "--parent-commit", "reviewed-parent"]
        )
    blocked_hub = (
        "import sys,types,runpy;"
        "sys.modules['huggingface_hub']=types.ModuleType('huggingface_hub');"
        f"sys.argv={[str(Path(release.__file__)), *args]!r};"
        f"runpy.run_path({str(Path(release.__file__))!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", blocked_hub],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert sensitive not in result.stdout + result.stderr
    assert "SyntaxError" not in result.stderr
    assert "Traceback" not in result.stderr
    assert "ImportError" not in result.stderr


def test_prepare_preserves_native_artifact_tasks_contract() -> None:
    from wyat import tasks
    from wyat.loading import native_artifact

    tasks_path = Path(tasks.__file__)
    raw = tasks_path.read_bytes()
    assert release.signed_python(raw) == raw
    assert (
        native_artifact.artifact_contract()["tasks_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
