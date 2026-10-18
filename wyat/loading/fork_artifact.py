# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Framework-free contract and validation for option-fork Wyat artifacts.

An option-fork model reads each question as one canonical prefix (state,
instructions and every option) followed by one isolated branch per option, and
scores each branch's last token with a shared FP32 scalar head. It is a
different model family from the native candidate-head format
(``wyat.loading.native_artifact``), with its own manifest and contract.
Importing this module never imports an inference framework.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wyat.input_rendering import validate_input_rendering_policy
from wyat.loading.native_artifact import ADAPTER, ADAPTER_TARGETS, file_sha256

FORMAT = "wyat-option-fork-v1"
MANIFEST = "fork_manifest.json"
# Encoding and readout versions. Bump on any change to the prefix/branch text,
# their tokenization boundaries, the readout position or the head math.
PROMPT = "canonical-id-description-prefix-isolated-option-v1"
READOUT = "last-nonpadding-post-final-norm"
HEAD = "layernorm-linear-gelu-linear-fp32"
ADAPTER_MATH = "base(x) + 2 * linear(linear(float32(x), down), up), fp32, no autocast"

# Qwen3.5 hybrid blocks adapt every linear projection of each block type.
# Attention blocks use the Qwen3
# names; gated-delta-net blocks add their own input/output projections.
QWEN3_5_ADAPTER_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "full_attention": tuple(ADAPTER_TARGETS),
    "linear_attention": (
        ("linear_attn", "in_proj_qkv"),
        ("linear_attn", "in_proj_z"),
        ("linear_attn", "in_proj_b"),
        ("linear_attn", "in_proj_a"),
        ("linear_attn", "out_proj"),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
    ),
}
QWEN3_5_ADAPTER = {
    "rank": ADAPTER["rank"],
    "alpha": ADAPTER["alpha"],
    "targets": {
        kind: [f"{owner}.{name}" for owner, name in targets]
        for kind, targets in QWEN3_5_ADAPTER_TARGETS.items()
    },
    "merged": False,
}
BACKBONES = ("qwen3_5_text",)
BASE_FILES = ("base/config.json",)
REQUIRED = frozenset(
    {
        "base/config.json",
        "base/model.safetensors",
        "adapters.safetensors",
        "head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
OPTIONAL = frozenset(
    {
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    }
)
# Directories an artifact may contain; anything else (or any symlink) is
# refused. A shard index is not admitted: its weight map could point outside.
DIRECTORIES = frozenset({"base"})
DOCUMENTATION_FILES = frozenset(
    {"README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", ".gitattributes"}
)


def contract() -> dict[str, object]:
    return {
        "prompt": PROMPT,
        "readout": READOUT,
        "head": HEAD,
        "adapter_math": ADAPTER_MATH,
        "independent_questions": True,
    }


def model_identity(files: dict[str, str], policy: object = None) -> str:
    selected = validate_input_rendering_policy(policy)
    payload = {"files_sha256": files, "input_rendering": selected}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return "fork-" + digest


def _positive(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_artifact(directory: str | Path) -> dict[str, Any]:
    """Reject missing, changed and undeclared files before any tensor loading."""
    root = Path(directory).resolve()
    if (root / MANIFEST).is_symlink():
        raise ValueError("fork manifest must be a local regular file")
    manifest = json.loads((root / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("unsupported fork artifact format")
    if manifest.get("contract") != contract():
        raise ValueError("fork artifact prompt/readout/head contract differs")
    if manifest.get("backbone") not in BACKBONES:
        raise ValueError("unsupported fork backbone")
    if manifest.get("adapter") != QWEN3_5_ADAPTER:
        raise ValueError("unsupported fork adapter contract")
    maximum = _positive(manifest.get("max_length"), "max_length")
    budget = _positive(manifest.get("max_batch_tokens"), "max_batch_tokens")
    if budget < maximum:
        raise ValueError("batch token budget must admit one complete branch")
    _positive(manifest.get("head_width"), "head_width")
    if not isinstance(manifest.get("provenance"), dict):
        raise ValueError("fork artifact provenance must be an object")
    files = manifest.get("files_sha256")
    if not isinstance(files, dict) or not REQUIRED <= files.keys():
        raise ValueError("fork artifact lacks required file identities")
    if not set(files) <= REQUIRED | OPTIONAL:
        raise ValueError("fork artifact contains unsupported files")
    if manifest.get("model_id") != model_identity(
        files, manifest.get("input_rendering")
    ):
        raise ValueError("fork model identity differs from artifact files")
    actual = _inventory(root)
    if actual != set(files) | {MANIFEST}:
        raise ValueError("fork artifact file inventory differs")
    for name, expected in files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent not in {root, root / "base"}
        ):
            raise ValueError("fork artifact files must be local regular files")
        if file_sha256(path) != expected:
            raise ValueError("fork artifact file identity differs: " + name)
    return manifest


def _inventory(root: Path) -> set[str]:
    """Every file under root; refuses symlinks and undeclared directories.

    ``rglob`` does not descend into symlinked directories, so a link would hide
    files that tokenizer loading still reads; walk explicitly instead.
    """
    names = set()
    for entry in root.iterdir():
        name = entry.name
        if entry.is_symlink():
            raise ValueError("fork artifact must not contain symlinks: " + name)
        if entry.is_dir():
            if name == ".cache":
                continue  # `hf download --local-dir` metadata, never read
            if name not in DIRECTORIES:
                raise ValueError(
                    "fork artifact contains an undeclared directory: " + name
                )
            for child in entry.iterdir():
                if child.is_symlink() or not child.is_file():
                    raise ValueError("fork artifact base must hold regular files only")
                names.add(f"{name}/{child.name}")
        elif entry.is_file():
            if name not in DOCUMENTATION_FILES:
                names.add(name)
        else:
            raise ValueError("fork artifact contains a non-regular entry: " + name)
    return names


def manifest_for(
    root: str | Path,
    *,
    max_length: int,
    max_batch_tokens: int,
    head_width: int,
    input_rendering: object,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build the manifest for files already written under ``root``."""
    root = Path(root)
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != MANIFEST
    }
    policy = validate_input_rendering_policy(input_rendering)
    return {
        "format": FORMAT,
        "contract": contract(),
        "backbone": "qwen3_5_text",
        "adapter": QWEN3_5_ADAPTER,
        "max_length": max_length,
        "max_batch_tokens": max_batch_tokens,
        "head_width": head_width,
        "input_rendering": policy,
        "provenance": provenance,
        "files_sha256": files,
        "model_id": model_identity(files, policy),
    }
