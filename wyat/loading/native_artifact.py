"""Framework-free identity and validation for full native Wyat artifacts.

The byte-level task and shared-head identities preserve the existing format.
Importing this module never imports either inference framework.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from wyat import tasks
from wyat.input_rendering import validate_input_rendering_policy


def _head_source_sha256() -> str:
    """Hash the exact class source without importing the optional Torch runtime."""
    path = Path(__file__).parents[1] / "backends/wyat/torch.py"
    source = path.read_text(encoding="utf-8")
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "CandidateHead"
    ]
    if len(classes) != 1 or classes[0].decorator_list:
        raise ValueError("shared head source requires one undecorated CandidateHead")
    node = classes[0]
    # Keep the final newline: ast.get_source_segment omits it, while the
    # historical inspect.getsource contract includes it.
    lines = source.splitlines(keepends=True)
    text = "".join(lines[node.lineno - 1 : node.end_lineno])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    """Bind encoding/readout and shared-head code, without recording local paths."""
    return {
        "recipe": tasks.RECIPE,
        "tasks_sha256": file_sha256(Path(tasks.__file__)),
        "predictor_sha256": _head_source_sha256(),
        "adapter_math": "base(x) + 2 * linear(linear(float32(x), down), up)",
        "independent_question_rows": True,
        "padding_attention_mask": False,
        "backbone_storage": ["torch.float32", "torch.bfloat16"],
        "cpu_bf16_policy": "lossless-fp32-promotion",
    }


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


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
    if manifest.get("contract") != artifact_contract():
        raise ValueError("native artifact task/predictor contract differs")
    if manifest.get("adapter") != {
        "rank": 16,
        "alpha": 32,
        "targets": ["q_proj", "v_proj"],
        "merged": False,
    }:
        raise ValueError("unsupported native adapter contract")
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
