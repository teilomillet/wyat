# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""The pinned download keeps only verified files and explains every failure."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import runpy
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from email.message import Message
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

import pytest

from wyat import cli
from wyat import download as module

VERIFIER = Path(__file__).resolve().parents[1] / "examples" / "model" / "verify.py"
Hub = dict[str, bytes]
MODEL_600M = module.RELEASE_600M.repository


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Hub:
    """A synthetic release whose pins the module trusts for this test."""
    release = module.RELEASES[MODEL_600M]
    assert release is not None
    files = {f"model/{name}": name.encode() for name in release.model_files}
    files |= {name: name.encode() for name, _ in release.license_files}
    digests = {
        name: hashlib.sha256(files[f"model/{name}"]).hexdigest()
        for name in release.model_files
    }
    files[release.manifest] = json.dumps({"files_sha256": digests}).encode()
    release = replace(
        release,
        revision="0" * 40,
        manifest_sha256=hashlib.sha256(files[release.manifest]).hexdigest(),
        license_files=tuple(
            (name, hashlib.sha256(name.encode()).hexdigest())
            for name, _ in release.license_files
        ),
    )
    monkeypatch.setattr(
        module, "RELEASES", MappingProxyType({**module.RELEASES, MODEL_600M: release})
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent")
    return files


def serve(
    files: Hub, requests: list[urllib.request.Request] | None = None
) -> Callable[[urllib.request.Request], BinaryIO]:
    release = module.RELEASES[MODEL_600M]
    assert release is not None
    prefix = f"https://huggingface.co/{MODEL_600M}/resolve/{release.revision}/"

    def opener(request: urllib.request.Request) -> BinaryIO:
        if requests is not None:
            requests.append(request)
        assert request.full_url.startswith(prefix)
        return io.BytesIO(files[request.full_url.removeprefix(prefix)])

    return opener


def quiet(line: str) -> None:
    pass


def test_download_places_verified_release_and_returns_the_model_path(
    hub: Hub, tmp_path: Path
) -> None:
    model = module.download(
        tmp_path, repository=MODEL_600M, opener=serve(hub), log=quiet
    )
    assert model == tmp_path / "model"
    for name, payload in hub.items():
        assert (tmp_path / name).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))


def test_rerun_keeps_verified_files_and_refetches_only_changed_ones(
    hub: Hub, tmp_path: Path
) -> None:
    module.download(tmp_path, repository=MODEL_600M, opener=serve(hub), log=quiet)
    (tmp_path / "model" / "tokenizer.json").write_bytes(b"changed")
    requests: list[urllib.request.Request] = []
    module.download(
        tmp_path, repository=MODEL_600M, opener=serve(hub, requests), log=quiet
    )
    assert [request.full_url.rsplit("/", 1)[1] for request in requests] == [
        "tokenizer.json"
    ]
    assert (tmp_path / "model" / "tokenizer.json").read_bytes() == b"tokenizer.json"


def test_manifest_is_checked_before_any_model_payload_is_requested(
    hub: Hub, tmp_path: Path
) -> None:
    hub[module.RELEASE_600M.manifest] = b"{}"
    requests: list[urllib.request.Request] = []
    with pytest.raises(module.DownloadError, match="does not match"):
        module.download(
            tmp_path, repository=MODEL_600M, opener=serve(hub, requests), log=quiet
        )
    assert requests[-1].full_url.endswith(module.RELEASE_600M.manifest)
    assert not (tmp_path / module.RELEASE_600M.manifest).exists()


def test_altered_payload_leaves_no_file_behind(hub: Hub, tmp_path: Path) -> None:
    hub["model/backbone/model.safetensors"] = b"altered"
    with pytest.raises(module.DownloadError, match="model.safetensors"):
        module.download(tmp_path, repository=MODEL_600M, opener=serve(hub), log=quiet)
    backbone = tmp_path / "model" / "backbone"
    assert not (backbone / "model.safetensors").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_files_from_another_release_are_rejected(hub: Hub, tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "vocab.json").write_text("{}")
    with pytest.raises(module.DownloadError, match="vocab.json"):
        module.download(tmp_path, repository=MODEL_600M, opener=serve(hub), log=quiet)


@pytest.mark.parametrize(
    "failure,message",
    [
        (urllib.error.HTTPError("u", 401, "x", Message(), None), "HF_TOKEN"),
        (urllib.error.URLError("offline"), "internet connection"),
    ],
)
def test_failures_explain_the_next_step(
    hub: Hub, tmp_path: Path, failure: Exception, message: str
) -> None:
    def opener(request: urllib.request.Request) -> BinaryIO:
        raise failure

    with pytest.raises(module.DownloadError, match=message):
        module.download(tmp_path, repository=MODEL_600M, opener=opener, log=quiet)


@pytest.mark.parametrize(
    "failure",
    [
        http.client.IncompleteRead(b"partial", 100),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
    ],
)
def test_interrupted_transfer_keeps_verified_files_and_no_partial(
    hub: Hub, tmp_path: Path, failure: Exception
) -> None:
    complete = serve(hub)

    class Dropped(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise failure

    def opener(request: urllib.request.Request) -> BinaryIO:
        if request.full_url.endswith("/model.safetensors"):
            return Dropped()
        return complete(request)

    with pytest.raises(module.DownloadError, match="internet connection"):
        module.download(tmp_path, repository=MODEL_600M, opener=opener, log=quiet)
    assert (tmp_path / module.RELEASE_600M.manifest).read_bytes() == hub[
        module.RELEASE_600M.manifest
    ]
    assert not list(tmp_path.rglob("*.part"))


def test_token_reaches_the_hub_but_not_its_redirect_target(
    hub: Hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    requests: list[urllib.request.Request] = []
    module.download(
        tmp_path, repository=MODEL_600M, opener=serve(hub, requests), log=quiet
    )
    for request in requests:
        assert request.unredirected_hdrs == {"Authorization": "Bearer secret"}
        assert "Authorization" not in request.headers


def test_cli_prints_only_the_model_path_on_stdout(
    hub: Hub,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(module, "_open", serve(hub))
    assert cli.main(["download", str(tmp_path), "--model", MODEL_600M]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"{tmp_path / 'model'}\n"
    assert "wyat.load" in captured.err


def test_cli_reports_failure_without_a_traceback(
    hub: Hub,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def offline(request: urllib.request.Request) -> BinaryIO:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(module, "_open", offline)
    assert cli.main(["download", str(tmp_path), "--model", MODEL_600M]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines()[-1].startswith("wyat: Could not reach")
    assert "Traceback" not in captured.err


def test_unpinned_default_refuses_before_any_request_or_cache_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def opener(request: urllib.request.Request) -> BinaryIO:
        raise AssertionError("no request without a release pin")

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        module,
        "RELEASES",
        MappingProxyType({**module.RELEASES, module.REPOSITORY: None}),
    )
    with pytest.raises(
        module.DownloadError, match="wyat-1-800m.*no release checkpoint"
    ):
        module.download(opener=opener, log=quiet)
    assert not any(tmp_path.iterdir())


def test_unknown_release_refuses_before_any_request(tmp_path: Path) -> None:
    def opener(request: urllib.request.Request) -> BinaryIO:
        raise AssertionError("unknown releases cannot download")

    with pytest.raises(module.DownloadError, match="does not recognize"):
        module.download(tmp_path, repository="someone/other-model", opener=opener)
    assert not any(tmp_path.iterdir())


def test_incomplete_pins_refuse_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = replace(module.RELEASE_600M, revision="")
    monkeypatch.setattr(
        module, "RELEASES", MappingProxyType({**module.RELEASES, MODEL_600M: release})
    )

    def opener(request: urllib.request.Request) -> BinaryIO:
        raise AssertionError("incomplete release pins cannot download")

    with pytest.raises(module.DownloadError, match="complete release pins"):
        module.download(tmp_path, repository=MODEL_600M, opener=opener)
    assert not any(tmp_path.iterdir())


def test_cli_default_reports_unreleased_800m_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def opener(request: urllib.request.Request) -> BinaryIO:
        raise AssertionError("the unavailable default cannot request another model")

    monkeypatch.setattr(module, "_open", opener)
    monkeypatch.setattr(
        module,
        "RELEASES",
        MappingProxyType({**module.RELEASES, module.REPOSITORY: None}),
    )
    assert cli.main(["download", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "wyat-1-800m" in captured.err
    assert "explicitly choose teilomillet/wyat-1-600m" in captured.err
    assert "Traceback" not in captured.err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("foreign", [None, "chat_template.jinja"])
def test_default_download_verifies_the_option_fork_layout(
    foreign: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = module.RELEASE_800M
    files = {f"model/{name}": name.encode() for name in release.model_files}
    files |= {name: name.encode() for name, _ in release.license_files}
    files[release.manifest] = json.dumps(
        {
            "files_sha256": {
                name: hashlib.sha256(files[f"model/{name}"]).hexdigest()
                for name in release.model_files
            }
        }
    ).encode()
    release = replace(
        release,
        revision="1" * 40,
        manifest_sha256=hashlib.sha256(files[release.manifest]).hexdigest(),
        license_files=tuple(
            (name, hashlib.sha256(name.encode()).hexdigest())
            for name, _ in release.license_files
        ),
    )
    monkeypatch.setattr(
        module,
        "RELEASES",
        MappingProxyType({**module.RELEASES, module.REPOSITORY: release}),
    )
    monkeypatch.setattr(module, "_token", lambda: None)
    prefix = f"https://huggingface.co/{module.REPOSITORY}/resolve/{release.revision}/"
    requested: list[str] = []

    def opener(request: urllib.request.Request) -> BinaryIO:
        assert request.full_url.startswith(prefix)
        name = request.full_url.removeprefix(prefix)
        requested.append(name)
        return io.BytesIO(files[name])

    if foreign:
        (tmp_path / "model").mkdir()
        (tmp_path / "model" / foreign).write_text("unexpected")
        with pytest.raises(module.DownloadError, match=foreign):
            module.download(tmp_path, opener=opener, log=quiet)
    else:
        assert module.download(tmp_path, opener=opener, log=quiet) == tmp_path / "model"
        assert requested == [
            *(name for name, _ in release.license_files),
            release.manifest,
            *(f"model/{name}" for name in release.model_files),
        ]
        assert not list(tmp_path.rglob("*.part"))


def test_copyable_verifier_pins_the_same_release() -> None:
    values = runpy.run_path(str(VERIFIER))
    release = module.RELEASES[values["REPOSITORY"]]
    assert release is not None
    assert values["REVISION"] == release.revision
    assert values["MANIFEST_SHA256"] == release.manifest_sha256
    assert set(values["FILES"]) == set(release.model_files)


def test_complete_copy_is_verified_quietly(hub: Hub, tmp_path: Path) -> None:
    module.download(tmp_path, repository=MODEL_600M, opener=serve(hub), log=quiet)
    lines: list[str] = []
    module.download(
        tmp_path, repository=MODEL_600M, opener=serve(hub), log=lines.append
    )
    assert lines and all(line.startswith("  ok ") for line in lines)


def test_default_directory_is_the_wyat_cache(
    hub: Hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    model = module.download(repository=MODEL_600M, opener=serve(hub), log=quiet)
    assert model == tmp_path / "wyat" / "wyat-1-600m" / "model"
    assert module.cache_directory(MODEL_600M) / "model" == model
    assert module.cache_directory() == tmp_path / "wyat" / "wyat-1-800m"


def bundle(target: Path) -> Path:
    """A small seeded model that wyat.load accepts."""
    from fixtures.wyat import create_wyat_bundle

    from wyat.wire import read_object

    path = create_wyat_bundle(target)
    # This rotary-position fixture needs a cap that admits the public load default.
    config_path = path / "backbone" / "config.json"
    config = read_object(config_path)
    config["max_position_embeddings"] = 2048
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_load_by_repository_name_uses_the_verified_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import wyat

    local = bundle(tmp_path / "cache")
    calls: list[tuple[object, object]] = []

    def fake_download(directory: object = None, **kwargs: object) -> Path:
        calls.append((directory, kwargs["repository"]))
        return local

    monkeypatch.setattr(module, "download", fake_download)
    monkeypatch.chdir(tmp_path)
    model = wyat.load(MODEL_600M)
    assert wyat.load(Path(MODEL_600M)).model_id == model.model_id
    assert calls == [(None, MODEL_600M), (None, MODEL_600M)]
    assert model.decide(
        wyat.Request("Pick", [wyat.Candidate("a", "A"), wyat.Candidate("b", "B")])
    ).selected in {"a", "b"}


@pytest.mark.parametrize("repository", [module.REPOSITORY, MODEL_600M])
def test_local_directory_takes_precedence_over_the_name(
    repository: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import wyat

    bundle(tmp_path / repository)

    def no_download(*args: object, **kwargs: object) -> Path:
        raise AssertionError("a local directory must not trigger a download")

    monkeypatch.setattr(module, "download", no_download)
    monkeypatch.chdir(tmp_path)
    assert wyat.load(repository).model_id


def test_other_names_are_local_paths_and_never_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import wyat

    def no_download(*args: object, **kwargs: object) -> Path:
        raise AssertionError("only the pinned release may be downloaded")

    monkeypatch.setattr(module, "download", no_download)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError):
        wyat.load("someone/other-model")
