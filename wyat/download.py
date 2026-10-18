# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Download and verify an explicitly pinned Wyat model release.

This is Wyat's only network path. It runs when ``wyat download`` is called, or
when ``wyat.load`` receives a recognized release name and no local path of that
name exists; any other path loads locally. Every file is checked against a
pinned SHA-256 digest before it is moved into place, so an interrupted or
altered download never looks complete.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

REPOSITORY = "teilomillet/wyat-1-800m"


@dataclass(frozen=True)
class ModelRelease:
    """The reviewed identity, payload layout and license pins of one release."""

    repository: str
    revision: str
    manifest_sha256: str
    license_files: tuple[tuple[str, str], ...]
    manifest: str
    model_files: tuple[str, ...]
    foreign_tokenizer_files: tuple[str, ...]
    download_size: str


RELEASE_600M = ModelRelease(
    repository="teilomillet/wyat-1-600m",
    revision="d651830acc7e37aa96ac466876895a3d4dac6c92",
    manifest_sha256="0c7e78ea5c0a888206c3bc0f5a7e15479ac6832e5cfac7ec4943f5c821019deb",
    license_files=(
        ("LICENSE", "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
        ("NOTICE", "b612666d7e55bb93e93443dca1d783cdf8719893c11ca5ecf6a50462b4b7dbd2"),
    ),
    manifest="model/native_manifest.json",
    model_files=(
        "backbone/config.json",
        "backbone/model.safetensors",
        "decision_head.safetensors",
        "native-adapters.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    foreign_tokenizer_files=(
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ),
    download_size="about 2.44 GB",
)
RELEASE_800M = ModelRelease(
    repository=REPOSITORY,
    revision="552166bdaa102bab14d9c0e4cea893a5a5aa234f",
    manifest_sha256="b363a65eaed1693e9d6139d08542f46e2f1b5c2e12b6c85a08a3afbcdf159901",
    license_files=(
        ("LICENSE", "50cbab8a892c5f2993b8c7351a99182507472def3b1374558308605d99b86b32"),
        ("NOTICE", "9a74a05b5ae902ac68b47806a912d96bca4bc9f38e3013322faaa8148140c8b0"),
    ),
    manifest="model/fork_manifest.json",
    model_files=(
        "adapters.safetensors",
        "base/config.json",
        "base/model.safetensors",
        "head.safetensors",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
    foreign_tokenizer_files=(
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ),
    download_size="about 1.82 GB",
)
RELEASES: Mapping[str, ModelRelease | None] = MappingProxyType(
    {REPOSITORY: RELEASE_800M, RELEASE_600M.repository: RELEASE_600M}
)
PROGRESS_BYTES = 100_000_000

Opener = Callable[[urllib.request.Request], BinaryIO]


class DownloadError(RuntimeError):
    """The pinned release could not be downloaded or did not verify."""


def _token() -> str | None:
    """Hugging Face's usual token sources; needed only while access is gated."""
    if token := os.environ.get("HF_TOKEN", "").strip():
        return token
    home = os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface"
    try:
        return (Path(home) / "token").read_text("utf-8").strip() or None
    except OSError:
        return None


def cache_directory(repository: str = REPOSITORY) -> Path:
    """Where the named release is cached; each release has its own directory."""
    if repository not in RELEASES:
        raise DownloadError(f"Wyat does not recognize model release {repository!r}.")
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "wyat" / repository.rsplit("/", 1)[1]


def _release(repository: str) -> ModelRelease:
    if repository not in RELEASES:
        raise DownloadError(f"Wyat does not recognize model release {repository!r}.")
    release = RELEASES[repository]
    if release is None:
        raise DownloadError(
            f"{repository} is the default model, but no release checkpoint is "
            "selected and pinned in this version of Wyat. Upgrade after its "
            "release, or explicitly choose teilomillet/wyat-1-600m."
        )
    if not (
        release.revision
        and release.manifest_sha256
        and release.license_files
        and all(digest for _, digest in release.license_files)
    ):
        raise DownloadError(
            f"{repository} does not have complete release pins in this version "
            "of Wyat. Upgrade after the release is pinned."
        )
    return release


def _request(
    release: ModelRelease, name: str, token: str | None
) -> urllib.request.Request:
    url = (
        f"https://huggingface.co/{release.repository}/resolve/{release.revision}/{name}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "wyat"})
    if token:
        # Unredirected: the token reaches huggingface.co, never the storage
        # host it redirects to.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return request


def _open(request: urllib.request.Request) -> BinaryIO:
    response: BinaryIO = urllib.request.urlopen(request, timeout=60)
    return response


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _size(count: float) -> str:
    for unit in ("B", "KB", "MB"):
        if count < 1000:
            return f"{count:.0f} {unit}"
        count /= 1000
    return f"{count:.2f} GB"


def _fetch(
    release: ModelRelease,
    name: str,
    target: Path,
    expected: str,
    *,
    token: str | None,
    opener: Opener,
    log: Callable[[str], None],
    announce: Callable[[], None],
) -> None:
    """Place ``name`` at ``target`` only once its digest equals ``expected``."""
    if target.is_file() and _sha256(target) == expected:
        log(f"  ok        {name}")
        return
    announce()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        with (
            opener(_request(release, name, token)) as response,
            partial.open("wb") as out,
        ):
            total: int | None = getattr(response, "length", None)
            done, reported = 0, 0
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
                done += len(chunk)
                # Large files report every 10% so a long wait never looks stuck.
                if total and total >= PROGRESS_BYTES and done * 10 // total > reported:
                    reported = done * 10 // total
                    log(f"            {name}: {reported * 10}% of {_size(total)}")
    except urllib.error.HTTPError as error:
        partial.unlink(missing_ok=True)
        if error.code in (401, 403):
            raise DownloadError(
                f"Hugging Face refused access to {release.repository} "
                f"(HTTP {error.code}). "
                "The model may not be public yet. If you have been given access, "
                "create a read token at https://huggingface.co/settings/tokens "
                "and set it as HF_TOKEN, then run this again."
            ) from error
        raise DownloadError(f"{name}: HTTP {error.code} from Hugging Face") from error
    # A connection dropped mid-transfer surfaces from read(), outside urlopen's
    # URLError wrapping.
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.HTTPException,
    ) as error:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            "Could not reach huggingface.co. Check your internet connection, then "
            "run this again; files already downloaded are kept."
        ) from error
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != expected:
        partial.unlink()
        raise DownloadError(
            f"{name} does not match the pinned release; nothing was kept. "
            "Run this again; if it keeps failing, report it as an issue."
        )
    os.replace(partial, target)
    log(f"  verified  {name}")


def download(
    directory: str | Path | None = None,
    *,
    repository: str = REPOSITORY,
    token: str | None = None,
    opener: Opener | None = None,
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> Path:
    """Download the named pinned release; return its local model path.

    The default directory is :func:`cache_directory`, which ``wyat.load``
    also uses. Files already present and matching their digests are kept, so
    running it again resumes an interrupted download or re-verifies a complete
    one. The returned path is what ``wyat.load`` takes.
    """
    import json

    release = _release(repository)
    root = cache_directory(repository) if directory is None else Path(directory)
    model = root / "model"
    token = token or _token()
    opener = opener or _open
    fetched = False

    def announce() -> None:
        # Only a download that actually fetches says so; a verified copy is quiet.
        nonlocal fetched
        if not fetched:
            log(f"Wyat model {repository} @ {release.revision[:12]} -> {root}")
            log(
                f"Downloading {release.download_size}; files already verified are kept."
            )
            fetched = True

    def fetch(name: str, expected: str) -> None:
        _fetch(
            release,
            name,
            root / name,
            expected,
            token=token,
            opener=opener,
            log=log,
            announce=announce,
        )

    for name, expected in release.license_files:
        fetch(name, expected)
    # The manifest is trusted only after its own pinned digest matches.
    fetch(release.manifest, release.manifest_sha256)
    manifest = json.loads((root / release.manifest).read_text("utf-8"))
    digests: Mapping[str, str] = manifest["files_sha256"]
    for name in release.model_files:
        fetch(f"model/{name}", digests[name])
    foreign = [
        name for name in release.foreign_tokenizer_files if (model / name).exists()
    ]
    if foreign:
        raise DownloadError(
            f"{model} also contains {', '.join(foreign)} from another release. "
            "Download into a new, empty directory."
        )
    if fetched:
        log("Done. Review the model's LICENSE and NOTICE before redistributing it.")
    return model
