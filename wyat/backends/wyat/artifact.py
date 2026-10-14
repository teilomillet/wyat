"""Validate local Wyat configuration, storage and content identity before loading."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from wyat.wire import JsonObject, JsonValue, decode_json, read_object

ARCHITECTURE = "qwen3-text-endpoints-global-query-shared-bilinear-mlp"


PROMPT_VERSION = "decision2-segmented-options-global-query-v1"


def positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def normalize_config(raw: JsonObject) -> JsonObject:
    if raw.get("model_type") != "qwen3":
        raise ValueError("Wyat backend requires a Qwen3 text backbone")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("Wyat requires the SiLU backbone activation")
    config = dict(raw)
    layers = config.get("layer_types")
    if layers is not None and (
        not isinstance(layers, list)
        or any(layer != "full_attention" for layer in layers)
    ):
        raise ValueError("Wyat requires full causal attention in every layer")
    if config.get("use_sliding_window", False):
        raise ValueError("Wyat does not support sliding-window attention")
    rotary = config.pop("rope_parameters", None)
    if rotary is not None:
        if (
            not isinstance(rotary, dict)
            or rotary.get("rope_type") != "default"
            or set(rotary) - {"rope_type", "rope_theta"}
            or "rope_theta" not in rotary
            or config.get("rope_scaling") is not None
            or ("rope_theta" in config and config["rope_theta"] != rotary["rope_theta"])
        ):
            raise ValueError("Unsupported or contradictory rotary configuration")
        config["rope_theta"] = rotary["rope_theta"]
    elif config.get("rope_scaling") is not None:
        raise ValueError("Wyat requires the unscaled default rotary recipe")
    theta = config.get("rope_theta", 10000.0)
    if (
        isinstance(theta, bool)
        or not isinstance(theta, (float, int))
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("Rotary theta must be finite and positive")
    config["rope_theta"] = theta
    return config


@dataclass(frozen=True, slots=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]


def tensor_specs(path: Path) -> dict[str, TensorSpec]:
    """Inspect original storage before a framework can silently cast weights."""
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError(f"Invalid Safetensors file: {path.name}")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 8 << 20:
            raise ValueError("Safetensors header exceeds 8 MiB limit")
        header = decode_json(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    result: dict[str, TensorSpec] = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Invalid tensor specification: {name}")
        dtype, shape = value.get("dtype"), value.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"Invalid tensor dtype or shape: {name}")
        dimensions: list[int] = []
        for dimension in shape:
            if type(dimension) is not int or dimension < 0:
                raise ValueError(f"Invalid tensor dimension: {name}")
            dimensions.append(dimension)
        result[name] = TensorSpec(dtype, tuple(dimensions))
    return result


def _declared_hashes(value: JsonValue) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Manifest files_sha256 must map names to 64-hex digests")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in digest)
        ):
            raise ValueError("Manifest files_sha256 must map names to 64-hex digests")
        result[name] = digest.lower()
    return result


def _identity(root: Path, declared: dict[str, str] | None) -> str:
    names = [
        "decision_config.json",
        "decision_head.safetensors",
        "backbone/config.json",
        "backbone/model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    names.extend(
        name
        for name in (
            "config.json",
            "MODEL_MANIFEST.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if (root / name).is_file()
    )
    digest = hashlib.sha256()
    for name in sorted(names):
        file_digest = hashlib.sha256()
        with (root / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                file_digest.update(block)
        if declared is not None and name in declared:
            if file_digest.hexdigest() != declared[name]:
                raise ValueError(
                    f"Checkpoint file differs from declared digest: {name}"
                )
        digest.update(name.encode("utf-8") + b"\0" + file_digest.digest())
    return "wyat:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WyatArtifact:
    root: Path
    config: JsonObject
    head_dim: int
    max_options: int
    model_id: str


def load_artifact(path: str | Path, max_length: int) -> WyatArtifact:
    root = Path(path).resolve(strict=True)
    max_length = positive_integer(max_length, "max_length")
    metadata = read_object(root / "decision_config.json")
    if (
        metadata.get("architecture") != ARCHITECTURE
        or metadata.get("prompt_version") != PROMPT_VERSION
        or metadata.get("head_variant", "shared") != "shared"
        or metadata.get("checkpoint_format") not in (None, "full")
        or metadata.get("readout") not in (None, "candidate")
        or metadata.get("dec_residual") is not None
        or metadata.get("head_compute_dtype", "float32") != "float32"
        or metadata.get("backbone_model_type", "qwen3") != "qwen3"
    ):
        raise ValueError(
            "Unsupported Wyat checkpoint architecture, prompt, head or format"
        )
    head_dim = positive_integer(metadata.get("head_dim"), "head_dim")
    max_options = positive_integer(metadata.get("max_options"), "max_options")
    if not 2 <= max_options <= 255:
        raise ValueError("Wyat max_options must be within 2..255")
    config = normalize_config(read_object(root / "backbone/config.json"))
    cap = positive_integer(
        config.get("max_position_embeddings"), "max_position_embeddings"
    )
    declared = None
    manifest_path = root / "MODEL_MANIFEST.json"
    if manifest_path.exists():
        manifest = read_object(manifest_path)
        if (
            manifest.get("profile") != "qwen-full"
            or manifest.get("storage") is not None
            or manifest.get("calibration") is not None
        ):
            raise ValueError("Unsupported Wyat package profile, storage or calibration")
        cap = min(
            cap, positive_integer(manifest.get("max_input_tokens"), "max_input_tokens")
        )
        if "files_sha256" in manifest:
            declared = _declared_hashes(manifest["files_sha256"])
    if max_length > cap:
        raise ValueError(f"max_length exceeds checkpoint limit {cap}")
    read_object(root / "tokenizer_config.json")
    if (root / "config.json").exists():
        read_object(root / "config.json")
    for name, spec in tensor_specs(root / "backbone/model.safetensors").items():
        if spec.dtype not in {"F32", "BF16"}:
            raise ValueError(f"Unsupported backbone tensor dtype: {name}")
    for name, spec in tensor_specs(root / "decision_head.safetensors").items():
        if spec.dtype != "F32":
            raise ValueError(f"Wyat head tensor must use float32: {name}")
    return WyatArtifact(root, config, head_dim, max_options, _identity(root, declared))
