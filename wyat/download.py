# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Download and verify the pinned Wyat model release.

This is Wyat's only network path, and it runs only when explicitly called.
``wyat.load`` stays local. Every file is checked against a pinned SHA-256
digest before it is moved into place, so an interrupted or altered download
never looks complete.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO

REPOSITORY = "teilomillet/wyat-1-600m"
MANIFEST = "model/native_manifest.json"
# The release pin: the Hub commit and the SHA-256 digests that make a download
# trustworthy. Fill all three from the published release (docs/releasing.md);
# until then download() refuses before any network access.
REVISION = ""
MANIFEST_SHA256 = ""
LICENSE_FILES = {"LICENSE": "", "NOTICE": ""}
# Payloads whose digests the pinned manifest records, relative to model/.
MODEL_FILES = (
    "backbone/config.json",
    "backbone/model.safetensors",
    "decision_head.safetensors",
    "native-adapters.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
# Optional tokenizer files change loading behavior and are absent from this
# release; their presence means another release's files were mixed in.
FOREIGN_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
DOWNLOAD_SIZE = "about 1.5 GB"  # a 600M-parameter backbone plus small files
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


def cache_directory() -> Path:
    """Where ``wyat download`` and ``wyat.load(REPOSITORY)`` keep the release."""
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "wyat" / REPOSITORY.rsplit("/", 1)[1]


def _url(name: str) -> str:
    return f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"


def _request(name: str, token: str | None) -> urllib.request.Request:
    request = urllib.request.Request(_url(name), headers={"User-Agent": "wyat"})
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
        with opener(_request(name, token)) as response, partial.open("wb") as out:
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
                f"Hugging Face refused access to {REPOSITORY} (HTTP {error.code}). "
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
    token: str | None = None,
    opener: Opener | None = None,
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr),
) -> Path:
    """Download the pinned release into ``directory``; return its model path.

    The default directory is :func:`cache_directory`, which ``wyat.load``
    also uses. Files already present and matching their digests are kept, so
    running it again resumes an interrupted download or re-verifies a complete
    one. The returned path is what ``wyat.load`` takes.
    """
    import json

    if not (REVISION and MANIFEST_SHA256 and all(LICENSE_FILES.values())):
        raise DownloadError(
            "This version of Wyat does not pin a model release yet. Upgrade Wyat, "
            "or obtain the model files and check them with examples/model/verify.py."
        )
    root = cache_directory() if directory is None else Path(directory)
    model = root / "model"
    token = token or _token()
    opener = opener or _open
    fetched = False

    def announce() -> None:
        # Only a download that actually fetches says so; a verified copy is quiet.
        nonlocal fetched
        if not fetched:
            log(f"Wyat model {REPOSITORY} @ {REVISION[:12]} -> {root}")
            log(f"Downloading {DOWNLOAD_SIZE}; files already verified are kept.")
            fetched = True

    def fetch(name: str, expected: str) -> None:
        _fetch(
            name,
            root / name,
            expected,
            token=token,
            opener=opener,
            log=log,
            announce=announce,
        )

    for name, expected in LICENSE_FILES.items():
        fetch(name, expected)
    # The manifest is trusted only after its own pinned digest matches.
    fetch(MANIFEST, MANIFEST_SHA256)
    manifest = json.loads((root / MANIFEST).read_text("utf-8"))
    digests: Mapping[str, str] = manifest["files_sha256"]
    for name in MODEL_FILES:
        fetch(f"model/{name}", digests[name])
    foreign = [name for name in FOREIGN_TOKENIZER_FILES if (model / name).exists()]
    if foreign:
        raise DownloadError(
            f"{model} also contains {', '.join(foreign)} from another release. "
            "Download into a new, empty directory."
        )
    if fetched:
        log("Done. Review the model's LICENSE and NOTICE before redistributing it.")
    return model
