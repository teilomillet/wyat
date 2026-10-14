# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Verify the pinned full Wyat download without loading weights or model code."""

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = "teilomillet/wyat-1-600m"
# Copy these from the release; they must equal the pins in wyat/download.py.
REVISION = ""
MANIFEST_SHA256 = ""
FILES = (
    "backbone/config.json",
    "backbone/model.safetensors",
    "decision_head.safetensors",
    "native-adapters.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory: Path) -> dict[str, object]:
    if not MANIFEST_SHA256:
        raise ValueError("no model release is pinned in this verifier yet")
    root = directory.resolve(strict=True)
    manifest_path = root / "native_manifest.json"
    if file_sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("manifest differs from the pinned Wyat release")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified: dict[str, str] = {}
    for name in FILES:
        digest = file_sha256(root / name)
        if digest != manifest["files_sha256"][name]:
            raise ValueError(f"file differs from the pinned Wyat release: {name}")
        verified[name] = digest
    # These optional files affect tokenizer/loading behavior but are absent from
    # this pinned download. Do not silently accept an older mixed artifact.
    for name in (
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ):
        if (root / name).exists():
            raise ValueError(
                f"unexpected tokenizer file in the pinned download: {name}"
            )
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256,
        "files_sha256": verified,
        "scope": "pinned download integrity; no inference or quality validation",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))


if __name__ == "__main__":
    main()
