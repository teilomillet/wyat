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
from email.message import Message
from pathlib import Path
from typing import BinaryIO

import pytest

from wyat import cli
from wyat import download as module

VERIFIER = Path(__file__).resolve().parents[1] / "examples" / "model" / "verify.py"
Hub = dict[str, bytes]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Hub:
    """A synthetic release whose pins the module trusts for this test."""
    files = {f"model/{name}": name.encode() for name in module.MODEL_FILES}
    files |= {name: name.encode() for name in module.LICENSE_FILES}
    digests = {
        name: hashlib.sha256(files[f"model/{name}"]).hexdigest()
        for name in module.MODEL_FILES
    }
    files[module.MANIFEST] = json.dumps({"files_sha256": digests}).encode()
    monkeypatch.setattr(module, "REVISION", "0" * 40)
    monkeypatch.setattr(
        module, "MANIFEST_SHA256", hashlib.sha256(files[module.MANIFEST]).hexdigest()
    )
    monkeypatch.setattr(
        module,
        "LICENSE_FILES",
        {
            name: hashlib.sha256(name.encode()).hexdigest()
            for name in module.LICENSE_FILES
        },
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", "/nonexistent")
    return files


def serve(
    files: Hub, requests: list[urllib.request.Request] | None = None
) -> Callable[[urllib.request.Request], BinaryIO]:
    prefix = f"https://huggingface.co/{module.REPOSITORY}/resolve/{module.REVISION}/"

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
    model = module.download(tmp_path, opener=serve(hub), log=quiet)
    assert model == tmp_path / "model"
    for name, payload in hub.items():
        assert (tmp_path / name).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))


def test_rerun_keeps_verified_files_and_refetches_only_changed_ones(
    hub: Hub, tmp_path: Path
) -> None:
    module.download(tmp_path, opener=serve(hub), log=quiet)
    (tmp_path / "model" / "tokenizer.json").write_bytes(b"changed")
    requests: list[urllib.request.Request] = []
    module.download(tmp_path, opener=serve(hub, requests), log=quiet)
    assert [request.full_url.rsplit("/", 1)[1] for request in requests] == [
        "tokenizer.json"
    ]
    assert (tmp_path / "model" / "tokenizer.json").read_bytes() == b"tokenizer.json"


def test_manifest_is_checked_before_any_model_payload_is_requested(
    hub: Hub, tmp_path: Path
) -> None:
    hub[module.MANIFEST] = b"{}"
    requests: list[urllib.request.Request] = []
    with pytest.raises(module.DownloadError, match="does not match"):
        module.download(tmp_path, opener=serve(hub, requests), log=quiet)
    assert requests[-1].full_url.endswith(module.MANIFEST)
    assert not (tmp_path / module.MANIFEST).exists()


def test_altered_payload_leaves_no_file_behind(hub: Hub, tmp_path: Path) -> None:
    hub["model/backbone/model.safetensors"] = b"altered"
    with pytest.raises(module.DownloadError, match="model.safetensors"):
        module.download(tmp_path, opener=serve(hub), log=quiet)
    backbone = tmp_path / "model" / "backbone"
    assert not (backbone / "model.safetensors").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_files_from_another_release_are_rejected(hub: Hub, tmp_path: Path) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "vocab.json").write_text("{}")
    with pytest.raises(module.DownloadError, match="vocab.json"):
        module.download(tmp_path, opener=serve(hub), log=quiet)


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
        module.download(tmp_path, opener=opener, log=quiet)


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
        module.download(tmp_path, opener=opener, log=quiet)
    assert (tmp_path / module.MANIFEST).read_bytes() == hub[module.MANIFEST]
    assert not list(tmp_path.rglob("*.part"))


def test_token_reaches_the_hub_but_not_its_redirect_target(
    hub: Hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    requests: list[urllib.request.Request] = []
    module.download(tmp_path, opener=serve(hub, requests), log=quiet)
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
    assert cli.main(["download", str(tmp_path)]) == 0
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
    assert cli.main(["download", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines()[-1].startswith("wyat: Could not reach")
    assert "Traceback" not in captured.err


def test_unpinned_release_refuses_before_any_request(tmp_path: Path) -> None:
    def opener(request: urllib.request.Request) -> BinaryIO:
        raise AssertionError("no request without a release pin")

    if module.REVISION and module.MANIFEST_SHA256:
        pytest.skip("this checkout pins a release")
    with pytest.raises(module.DownloadError, match="does not pin a model release"):
        module.download(tmp_path, opener=opener, log=quiet)
    assert not any(tmp_path.iterdir())


def test_copyable_verifier_pins_the_same_release() -> None:
    values = runpy.run_path(str(VERIFIER))
    assert values["REPOSITORY"] == module.REPOSITORY
    assert values["REVISION"] == module.REVISION
    assert values["MANIFEST_SHA256"] == module.MANIFEST_SHA256
    assert set(values["FILES"]) == set(module.MODEL_FILES)


def test_complete_copy_is_verified_quietly(hub: Hub, tmp_path: Path) -> None:
    module.download(tmp_path, opener=serve(hub), log=quiet)
    lines: list[str] = []
    module.download(tmp_path, opener=serve(hub), log=lines.append)
    assert lines and all(line.startswith("  ok ") for line in lines)


def test_default_directory_is_the_wyat_cache(
    hub: Hub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    model = module.download(opener=serve(hub), log=quiet)
    assert model == tmp_path / "wyat" / "wyat-1-600m" / "model"
    assert module.cache_directory() / "model" == model


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
    calls: list[object] = []

    def fake_download(directory: object = None, **kwargs: object) -> Path:
        calls.append(directory)
        return local

    monkeypatch.setattr(module, "download", fake_download)
    monkeypatch.chdir(tmp_path)
    model = wyat.load(module.REPOSITORY)
    assert wyat.load(Path(module.REPOSITORY)).model_id == model.model_id
    assert calls == [None, None]
    assert model.decide(
        wyat.Request("Pick", [wyat.Candidate("a", "A"), wyat.Candidate("b", "B")])
    ).selected in {"a", "b"}


def test_local_directory_takes_precedence_over_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    import wyat

    bundle(tmp_path / module.REPOSITORY)

    def no_download(*args: object, **kwargs: object) -> Path:
        raise AssertionError("a local directory must not trigger a download")

    monkeypatch.setattr(module, "download", no_download)
    monkeypatch.chdir(tmp_path)
    assert wyat.load(module.REPOSITORY).model_id


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
