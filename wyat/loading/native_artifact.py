# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Framework-free identity and validation for full native Wyat artifacts.

An artifact binds to versioned model-facing behavior, not to source bytes, so
edits that keep encoding and readout unchanged keep artifacts loadable.
Importing this module never imports either inference framework.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wyat import tasks
from wyat.input_rendering import validate_input_rendering_policy

# The shared CandidateHead readout (Torch and MLX). Bump it when the head's
# parameters or math change, as tasks.RECIPE is bumped when the encoded text
# changes. Golden tests in tests/test_tasks.py and
# tests/backends/wyat/test_torch.py fail until the matching version moves.
PREDICTOR = "wyat-bilinear-gelu-candidate-head-v1"

# Earlier exports recorded source hashes in place of the two versions. These
# are the exact v1 sources the golden tests were recorded from (tasks.py
# dd4847e3, CandidateHead dca28885; the v7 training pins), so an artifact
# carrying them has the v1 encoding and readout. Remove once none must load.
_V1_VERSIONS = {
    "recipe": "wyat-native-typed-segmented-independent-questions-v1",
    "predictor": "wyat-bilinear-gelu-candidate-head-v1",
}
_V1_SOURCE_HASHES = {
    "tasks_sha256": "dd4847e3b51e4f6502011874aa1a8eef8453920483b9a781da1fa87cefb59fe2",
    "predictor_sha256": (
        "dca288850bf6a8de3ffc258e98bf282e9998b4b8289f4da7f99a30694d1ab502"
    ),
}

# Adapter policy for new native models: rank-16 LoRA on every linear projection
# of every block. Attention-only LoRA underperforms full fine-tuning; adapting
# every weight matrix, the MLP especially, closes the gap (Thinking Machines,
# "LoRA Without Regret", 2025). Training and export must write ADAPTER; a test
# checks that ADAPTER_TARGETS names every linear module of a Qwen3 block.
ADAPTER_TARGETS: tuple[tuple[str, str], ...] = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)
ADAPTER = {
    "rank": 16,
    "alpha": 32,
    "targets": [f"{owner}.{name}" for owner, name in ADAPTER_TARGETS],
    "merged": False,
}
# Released models (v7, wyat-1-600m) adapt q/v only. They still load; no new
# model may be trained or exported with this form.
_LEGACY_QV_ADAPTER = {
    "rank": 16,
    "alpha": 32,
    "targets": ["q_proj", "v_proj"],
    "merged": False,
}

FORMAT = "wyat-native-full-v1"
MANIFEST = "native_manifest.json"
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
DOCUMENTATION_FILES = frozenset(
    {"README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", ".gitattributes"}
)
DOWNLOAD_METADATA = ".cache/huggingface/"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_contract() -> dict[str, object]:
    """Bind the encoding recipe and shared-head readout versions."""
    return {
        "recipe": tasks.RECIPE,
        "predictor": PREDICTOR,
        "adapter_math": "base(x) + 2 * linear(linear(float32(x), down), up)",
        "independent_question_rows": True,
        "padding_attention_mask": False,
        "backbone_storage": ["torch.float32", "torch.bfloat16"],
        "cpu_bf16_policy": "lossless-fp32-promotion",
    }


def _accepts_contract(recorded: object) -> bool:
    current = artifact_contract()
    if recorded == current:
        return True
    versions = {"recipe": current["recipe"], "predictor": current.pop("predictor")}
    return versions == _V1_VERSIONS and recorded == current | _V1_SOURCE_HASHES


def model_identity(files: dict[str, str], policy: object = None) -> str:
    """Bind opt-in rendering to model identity, retaining legacy byte identity."""
    selected = validate_input_rendering_policy(policy)
    payload = (
        files
        if selected is None
        else {"files_sha256": files, "input_rendering": selected}
    )
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return "native-" + identity


def adapter_targets(manifest: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return the (block module, projection) pairs a manifest's adapters wrap."""
    adapter = manifest.get("adapter")
    if adapter == ADAPTER:
        return ADAPTER_TARGETS
    if adapter == _LEGACY_QV_ADAPTER:
        return (("self_attn", "q_proj"), ("self_attn", "v_proj"))
    raise ValueError("unsupported native adapter contract")


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def budget_groups(
    encoded: tuple[tasks.EncodedQuestion, ...], max_batch_tokens: int
) -> list[tuple[tasks.EncodedQuestion, ...]]:
    """Split rows, in order, into contiguous padded batches within the budget.

    Rows are independent and right-padded, so grouping changes batch shape but
    not which tokens each row attends to. A record that fits stays one group.
    A single row is never split; the executor still rejects one over budget.
    """
    groups = []
    start = 0
    width = 0
    for index, row in enumerate(encoded):
        widened = max(width, len(row.ids))
        if index > start and widened * (index - start + 1) > max_batch_tokens:
            groups.append(encoded[start:index])
            start = index
            widened = len(row.ids)
        width = widened
    if encoded:
        groups.append(encoded[start:])
    return groups


def validate_artifact(directory: str | Path) -> dict[str, Any]:
    """Reject missing, changed and undeclared artifact files before tensor loading."""
    root = Path(directory).resolve()
    if (root / MANIFEST).is_symlink():
        raise ValueError("native manifest must be a local regular file")
    manifest = json.loads((root / MANIFEST).read_text())
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise ValueError("unsupported native artifact format")
    if (
        manifest.get("production_wyat_compatible", False) is not False
        or manifest.get("resume_supported") is not False
    ):
        raise ValueError(
            "native artifact must declare distinct inference and no resume"
        )
    if not isinstance(manifest.get("model_id"), str) or not manifest[
        "model_id"
    ].startswith("native-"):
        raise ValueError("native artifact requires a model identity")
    if not isinstance(manifest.get("provenance"), dict):
        raise ValueError("native artifact provenance must be an object")
    if not _accepts_contract(manifest.get("contract")):
        raise ValueError("native artifact task/predictor contract differs")
    adapter_targets(manifest)
    maximum = _positive_integer(manifest.get("max_length"), "max_length")
    budget = _positive_integer(manifest.get("max_batch_tokens"), "max_batch_tokens")
    if budget < maximum:
        raise ValueError("batch token budget must admit one complete row")
    _positive_integer(manifest.get("head_dim"), "head_dim")
    files = manifest.get("files_sha256")
    required = {
        "backbone/config.json",
        "backbone/model.safetensors",
        "decision_head.safetensors",
        "native-adapters.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if not isinstance(files, dict) or not required <= files.keys():
        raise ValueError("native artifact lacks required file identities")
    if "input_rendering" in manifest and manifest["input_rendering"] is None:
        raise ValueError("input rendering policy must be omitted for legacy artifacts")
    if manifest["model_id"] != model_identity(files, manifest.get("input_rendering")):
        raise ValueError("native model identity differs from artifact files")
    permitted = required | set(TOKENIZER_FILES)
    if not set(files) <= permitted:
        raise ValueError("native artifact contains unsupported files")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    # Published repositories carry documentation beside the weights, and
    # `hf download --local-dir` writes its own metadata. Neither is ever read.
    actual = {
        name
        for name in actual
        if name not in DOCUMENTATION_FILES and not name.startswith(DOWNLOAD_METADATA)
    }
    if actual != set(files) | {MANIFEST}:
        raise ValueError("native artifact file inventory differs")
    for name, expected in files.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent not in {root, root / "backbone"}
        ):
            raise ValueError("native artifact files must be local regular files")
        if file_sha256(path) != expected:
            raise ValueError("native artifact file identity differs: " + name)
    return manifest
