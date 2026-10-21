# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Framework-free contract, manifest validation, and encoding for Ouro artifacts.

An Ouro decision model reads each question as one canonical shared context
(state, instructions, and UTF-8-sorted option menu) followed by one isolated
candidate continuation per option in caller order, and scores each candidate's
last token after the recurrent candidate passes with a shared FP32 scalar head.

Importing this module never imports PyTorch, Transformers, or Safetensors;
manifest validation, directory inventory checks, and segmented prompt encoding
live here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from wyat import tasks
from wyat.core import ModelInputError
from wyat.loading.native_artifact import file_sha256

FORMAT = "wyat-ouro-decision-v1"
MANIFEST = "ouro_manifest.json"

BASE_MODEL_ID = "ByteDance/Ouro-1.4B"
BASE_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
BASE_WEIGHT_FILE = "model.safetensors"
SOURCE_FILES = ("config.json", "configuration_ouro.py", "modeling_ouro.py")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
ADAPTERS_FILE = "adapters.safetensors"
HEAD_FILE = "head.safetensors"

REQUIRED_FILES = frozenset(
    SOURCE_FILES + TOKENIZER_FILES + (BASE_WEIGHT_FILE, ADAPTERS_FILE, HEAD_FILE)
)
OPTIONAL_FILES = frozenset(
    {"README.md", "LICENSE", "NOTICE", "ATTRIBUTION.md", ".gitattributes"}
)

# Pinned ByteDance/Ouro-1.4B release file identities (revision 574fa66cb8bf...).
PINNED_BASE_FILES: dict[str, dict[str, int | str]] = {
    "config.json": {
        "sha256": "ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3",
        "size": 1472,
    },
    "configuration_ouro.py": {
        "sha256": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
        "size": 11550,
    },
    "modeling_ouro.py": {
        "sha256": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
        "size": 33350,
    },
    "tokenizer.json": {
        "sha256": "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d",
        "size": 3522644,
    },
    "tokenizer_config.json": {
        "sha256": "7e010da95d71b0fa1aa809552922c74cc1dd98d628f21ba39ca6c35aa18d5d91",
        "size": 4211,
    },
    "special_tokens_map.json": {
        "sha256": "55087bb8409060d9cb0f80495e34f4e0d8a84f68a799a6b0cab3132be0aae319",
        "size": 825,
    },
    "model.safetensors": {
        "sha256": "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af",
        "size": 2869336434,
    },
}

# Clean domain nomenclature for Universal Transformer recurrent steps and
# two-stage context + candidate scoring.
TOTAL_STEPS = 4
CONTEXT_STEPS = 4
CONTEXT_KV_STEP = 4
CANDIDATE_STEPS = 2
CANDIDATE_KV_POLICY = "last"

ADAPTER_RANK = 16
ADAPTER_ALPHA = 32
ADAPTER_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
NATIVE_LAYERS = 24
NATIVE_MAX_POSITION_EMBEDDINGS = 65536
HEAD_WIDTH = 512

PROMPT = "canonical-id-description-prefix-isolated-option-v1"
INPUT_RENDERING_MODES = frozenset(
    {
        "api-bank-schema-next-v1",
        "tool-decision-schema-inline-v3",
        "tool-decision-schema-inline-v4.1",
        "tool-decision-schema-inline-v5",
    }
)
READOUT = "last-nonpadding-post-final-norm"
HEAD = "layernorm-linear-gelu-linear-fp32"
ADAPTER_MATH = (
    "base(x) + (alpha/rank) * linear(linear(float32(x), A), B), fp32, no autocast"
)

EXECUTION_BASE_WEIGHT_DTYPE = "bfloat16"
EXECUTION_COMPUTE_DTYPE = "bfloat16"
EXECUTION_ADAPTER_DTYPE = "float32"
EXECUTION_HEAD_DTYPE = "float32"
ATTENTION_IMPLEMENTATIONS = frozenset({"sdpa-math", "eager"})
DEVICE_POLICIES = frozenset({"cuda-bf16-autocast", "cpu-fp32"})

TENSOR_DTYPES = frozenset({"float32"})

CACHE_POLICY = "retained-prefix-v1"
CACHE_MATCH = "exact-token-prefix"
CACHE_DEFAULT_ENTRIES = 0
CACHE_MAX_ENTRIES = 16
CACHE_DEFAULT_STATE_BLOCK_TOKENS = 0
CACHE_MAX_STATE_BLOCK_TOKENS = 512
CACHE_TOLERANCE_ATOL = 2e-5
CACHE_TOLERANCE_RTOL = 1e-4
CACHE_DISABLED_IS_BITWISE = True

HEAD_TENSORS = (
    "head.norm.weight",
    "head.norm.bias",
    "head.projection.weight",
    "head.projection.bias",
    "head.output.weight",
    "head.output.bias",
)

_TOP_LEVEL = frozenset(
    {
        "format",
        "model_id",
        "contract",
        "base",
        "tokenizer",
        "prompt",
        "tensors",
        "execution",
        "cache",
        "files_sha256",
    }
)
_BASE_FIELDS = frozenset({"model_id", "revision", "checkpoint_sha256", "source_sha256"})
_PROMPT_FIELDS = frozenset({"encoder_sha256", "input_rendering"})
_RENDERING_FIELDS = frozenset({"mode", "source_sha256"})
_TENSOR_GROUPS = frozenset({"adapters", "head"})
_EXECUTION_FIELDS = frozenset(
    {
        "max_length",
        "base_weight_dtype",
        "compute_dtype",
        "adapter_dtype",
        "head_dtype",
        "attention",
        "device_policy",
    }
)
_CACHE_FIELDS = frozenset(
    {
        "policy",
        "match",
        "entries",
        "state_block_tokens",
        "tolerance",
        "bitwise_when_disabled",
    }
)
_TOLERANCE_FIELDS = frozenset({"atol", "rtol"})
_HEX = frozenset("0123456789abcdef")


def contract() -> dict[str, object]:
    """Return the frozen execution contract recorded in every Ouro manifest."""
    return {
        "prompt": PROMPT,
        "readout": READOUT,
        "head": HEAD,
        "adapter_math": ADAPTER_MATH,
        "loops": TOTAL_STEPS,
        "prefix_loops": CONTEXT_STEPS,
        "prefix_bank": CONTEXT_KV_STEP,
        "branch_loops": CANDIDATE_STEPS,
        "branch_policy": CANDIDATE_KV_POLICY,
        "adapter_targets": list(ADAPTER_TARGETS),
        "adapter_rank": ADAPTER_RANK,
        "adapter_alpha": ADAPTER_ALPHA,
        "head_width": HEAD_WIDTH,
        "independent_questions": True,
    }


def adapter_tensor_names(
    num_layers: int = NATIVE_LAYERS, candidate_steps: int = CANDIDATE_STEPS
) -> tuple[str, ...]:
    """Canonical module-qualified step-LoRA adapter keys."""
    names = []
    for layer in range(num_layers):
        for target in ADAPTER_TARGETS:
            owner, projection = target.split(".")
            for step in range(candidate_steps):
                for factor in ("A", "B"):
                    names.append(
                        f"backbone.model.layers.{layer}.{owner}.{projection}"
                        f".lora_{factor}.{step}"
                    )
    return tuple(names)


def model_identity(
    files_sha256: Mapping[str, str], prompt: Mapping[str, object]
) -> str:
    """Derive the artifact identity from its files, contract, and prompt."""
    payload = {
        "contract": contract(),
        "prompt": dict(prompt),
        "files_sha256": dict(files_sha256),
    }
    digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return "ouro-" + digest


def build_manifest(
    *,
    files_sha256: Mapping[str, str],
    encoder_sha256: str,
    adapter_dtypes: Mapping[str, str],
    head_dtypes: Mapping[str, str],
    max_length: int,
    attention: str,
    device_policy: str,
    cache_entries: int = CACHE_DEFAULT_ENTRIES,
    cache_state_block_tokens: int = CACHE_DEFAULT_STATE_BLOCK_TOKENS,
    input_rendering: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Assemble, identify, and validate one JSON-serializable Ouro manifest."""
    files = _mapping(files_sha256, "files_sha256")
    _validate_file_hashes(files)
    _hash(encoder_sha256, "prompt.encoder_sha256")
    prompt: dict[str, object] = {
        "encoder_sha256": encoder_sha256,
        "input_rendering": (
            None if input_rendering is None else _rendering(input_rendering)
        ),
    }
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "model_id": "",
        "contract": contract(),
        "base": {
            "model_id": BASE_MODEL_ID,
            "revision": BASE_REVISION,
            "checkpoint_sha256": files.get(BASE_WEIGHT_FILE),
            "source_sha256": {name: files.get(name) for name in SOURCE_FILES},
        },
        "tokenizer": {name: files.get(name) for name in TOKENIZER_FILES},
        "prompt": prompt,
        "tensors": {
            "adapters": _mapping(adapter_dtypes, "adapter_dtypes"),
            "head": _mapping(head_dtypes, "head_dtypes"),
        },
        "execution": {
            "max_length": max_length,
            "base_weight_dtype": EXECUTION_BASE_WEIGHT_DTYPE,
            "compute_dtype": EXECUTION_COMPUTE_DTYPE,
            "adapter_dtype": EXECUTION_ADAPTER_DTYPE,
            "head_dtype": EXECUTION_HEAD_DTYPE,
            "attention": attention,
            "device_policy": device_policy,
        },
        "cache": {
            "policy": CACHE_POLICY,
            "match": CACHE_MATCH,
            "entries": cache_entries,
            "state_block_tokens": cache_state_block_tokens,
            "tolerance": {
                "atol": CACHE_TOLERANCE_ATOL,
                "rtol": CACHE_TOLERANCE_RTOL,
            },
            "bitwise_when_disabled": CACHE_DISABLED_IS_BITWISE,
        },
        "files_sha256": files,
    }
    manifest["model_id"] = model_identity(files, prompt)
    validate_manifest(manifest)
    return manifest


def validate_manifest(value: object) -> None:
    """Refuse any manifest that does not match the frozen release contract."""
    manifest = _object(value, "Ouro manifest")
    _exact_keys(manifest, _TOP_LEVEL, "manifest")
    if manifest["format"] != FORMAT:
        raise ValueError(f"unsupported Ouro artifact format: {manifest['format']!r}")
    _validate_contract(manifest["contract"])

    base = _object(manifest["base"], "base")
    _exact_keys(base, _BASE_FIELDS, "base")
    if base["model_id"] != BASE_MODEL_ID:
        raise ValueError("Ouro base model identity differs from the pinned release")
    if base["revision"] != BASE_REVISION:
        raise ValueError("Ouro base revision differs from the pinned release")
    _hash(base["checkpoint_sha256"], "base.checkpoint_sha256")
    source = _object(base["source_sha256"], "base.source_sha256")
    _exact_keys(source, SOURCE_FILES, "base.source_sha256")
    for name in SOURCE_FILES:
        _hash(source[name], f"base.source_sha256.{name}")

    tokenizer = _object(manifest["tokenizer"], "tokenizer")
    _exact_keys(tokenizer, TOKENIZER_FILES, "tokenizer")
    for name in TOKENIZER_FILES:
        _hash(tokenizer[name], f"tokenizer.{name}")

    prompt = _object(manifest["prompt"], "prompt")
    _exact_keys(prompt, _PROMPT_FIELDS, "prompt")
    _hash(prompt["encoder_sha256"], "prompt.encoder_sha256")
    rendering = prompt["input_rendering"]
    if rendering is not None:
        _rendering(rendering)

    _validate_tensors(manifest["tensors"])
    _validate_execution(manifest["execution"])
    _validate_cache(manifest["cache"])

    files = _object(manifest["files_sha256"], "files_sha256")
    _validate_file_hashes(files)
    if base["checkpoint_sha256"] != files[BASE_WEIGHT_FILE]:
        raise ValueError("base checkpoint identity differs from files_sha256")
    for name in SOURCE_FILES:
        if source[name] != files[name]:
            raise ValueError(
                f"native source identity differs from files_sha256: {name}"
            )
    for name in TOKENIZER_FILES:
        if tokenizer[name] != files[name]:
            raise ValueError(f"tokenizer identity differs from files_sha256: {name}")

    if manifest["model_id"] != model_identity(files, prompt):
        raise ValueError("Ouro model identity differs from artifact files")


def manifest_digest(manifest: object) -> str:
    """Canonical SHA-256 over a validated manifest, independent of key order."""
    validate_manifest(manifest)
    return hashlib.sha256(_canonical(manifest).encode()).hexdigest()


def decode_manifest(text: str) -> dict[str, Any]:
    """Decode manifest JSON, refusing duplicate keys and nonfinite numbers."""
    if type(text) is not str:
        raise ValueError("manifest JSON must be text")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = item
        return result

    def constant(value: str) -> Any:
        raise ValueError("nonfinite manifest number")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite manifest number")
        return number

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=finite_float,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid manifest JSON") from error
    validate_manifest(decoded)
    return cast(dict[str, Any], decoded)


def validate_artifact(
    directory: str | Path, *, verify_release_base: bool = False
) -> dict[str, Any]:
    """Validate an Ouro artifact directory before any tensor loading.

    Refuses symlinks, subdirectories, undeclared files, duplicate JSON keys,
    manifest schema mismatches, and file SHA-256 mismatches. When
    ``verify_release_base=True``, also verifies that the base weights, native
    runtime sources, and tokenizer files match the pinned ByteDance/Ouro-1.4B
    release bytes and sizes in ``PINNED_BASE_FILES``.
    """
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Ouro manifest must be a local regular file")
    manifest = decode_manifest(manifest_path.read_text(encoding="utf-8"))
    actual = _directory_inventory(root)
    files = manifest["files_sha256"]
    if actual != set(files) | {MANIFEST}:
        raise ValueError("Ouro artifact file inventory differs")
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise ValueError("Ouro artifact files must be local regular files")
        if file_sha256(path) != expected:
            raise ValueError("Ouro artifact file identity differs: " + name)
    if verify_release_base:
        verify_release_base_files(root, files)
    return manifest


def verify_release_base_files(
    directory: str | Path, files_sha256: Mapping[str, str] | None = None
) -> None:
    """Verify that base weights, native source, and tokenizer match the release pin."""
    root = Path(directory).resolve()
    for name, expected in PINNED_BASE_FILES.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular Ouro base file: {name}")
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"Ouro base file size differs from release pin: {name}")
        digest = (
            files_sha256[name]
            if files_sha256 is not None and name in files_sha256
            else file_sha256(path)
        )
        if digest != expected["sha256"]:
            raise ValueError(f"Ouro base file SHA-256 differs from release pin: {name}")


def _directory_inventory(root: Path) -> set[str]:
    """Every non-documentation file in ``root``; refuses symlinks and subdirectories."""
    names: set[str] = set()
    for entry in root.iterdir():
        name = entry.name
        if entry.is_symlink():
            raise ValueError("Ouro artifact must not contain symlinks: " + name)
        if entry.is_dir():
            raise ValueError(
                "Ouro artifact must be flat and contain no subdirectories: " + name
            )
        if entry.is_file():
            if name not in OPTIONAL_FILES:
                names.add(name)
        else:
            raise ValueError("Ouro artifact contains a non-regular entry: " + name)
    return names


def _payload(option: tuple[str, str]) -> str:
    return json.dumps(
        {"key": option[0], "description": option[1]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class CandidateBatch:
    """Segmented question prompt: shared context tokens + K candidate continuations.

    Provides ``prefix_ids`` and ``option_ids`` properties as aliases for
    ``context_ids`` and ``candidate_ids`` so callers written against
    ``ForkQuestion`` work without modification.
    """

    question: tasks.Question
    context_ids: tuple[int, ...]
    candidate_ids: tuple[tuple[int, ...], ...]
    state: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.question, tasks.Question):
            raise ValueError("Ouro candidate batch requires a validated question")
        if not isinstance(self.candidate_ids, tuple):
            raise ValueError("Ouro candidate continuations must be an immutable tuple")
        if len(self.candidate_ids) != len(self.question.options):
            raise ValueError("one candidate continuation is required per option")
        for ids in (self.context_ids, *self.candidate_ids):
            if (
                not isinstance(ids, tuple)
                or not ids
                or any(type(token) is not int or token < 0 for token in ids)
            ):
                raise ValueError(
                    "Ouro tokens must be nonempty nonnegative integer tuples"
                )


def encode_record(
    tokenizer: tasks.Tokenizer, record: object, *, max_length: int
) -> tuple[CandidateBatch, ...]:
    """Encode a record into shared context tokens and caller-order candidate tokens.

    Shared context and candidate continuations are tokenized separately with
    ``add_special_tokens=False``. Over-long inputs are refused, never truncated.
    """
    if type(max_length) is not int or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    state, questions = tasks.public_questions(record)

    def tokens(text: str) -> tuple[int, ...]:
        if len(text.encode("utf-8")) > max_length * 32:
            raise ValueError("segment exceeds tokenization byte limit")
        values = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError("tokenizer must return a sequence of token IDs")
        ids = tuple(values)
        if not ids or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("tokenizer must return nonempty nonnegative token IDs")
        return ids

    result = []
    for question in questions:
        options = [_payload(option) for option in question.options]
        canonical = sorted(options, key=lambda text: text.encode("utf-8"))
        context = (
            f"State:\n{state}\n\nTask type: {question.type}\n"
            f"Question ID: {question.id}\nInstructions:\n{question.instructions}\n"
            "All options:\n" + "\n".join(canonical) + "\n\nEvaluate option:\n"
        )
        context_ids = tokens(context)
        candidates = tuple(tokens(option) for option in options)
        if len(context_ids) + max(map(len, candidates)) > max_length:
            raise ValueError(
                "complete candidate sequence exceeds max_length; no truncation"
            )
        result.append(CandidateBatch(question, context_ids, candidates, state))
    return tuple(result)


def reuse_context_prefix(
    tokenizer: tasks.Tokenizer, row: CandidateBatch
) -> tuple[int, ...]:
    """Longest exact token prefix shared by the rendered state block and context."""
    if type(row) is not CandidateBatch or type(row.state) is not str or not row.state:
        return ()
    header = f"State:\n{row.state}\n"
    if len(header.encode("utf-8")) > 32 * max(len(row.context_ids), 1):
        return ()
    try:
        ids = tuple(tokenizer.encode(header, add_special_tokens=False))
    except (TypeError, ValueError, OverflowError):
        return ()
    if not ids or any(type(token) is not int or token < 0 for token in ids):
        return ()
    matched = 0
    for expected, actual in zip(ids, row.context_ids, strict=False):
        if expected != actual:
            break
        matched += 1
    return row.context_ids[:matched]


def input_error(error: ValueError) -> ModelInputError:
    """Classify an encoder refusal of a valid record with the native reasons."""
    message = str(error)
    reason: Literal["input-length", "input-segment"] = (
        "input-length"
        if "max_length" in message or "byte limit" in message
        else "input-segment"
    )
    return ModelInputError(message, reason=reason)


def _validate_contract(value: object) -> None:
    frozen = contract()
    selected = _object(value, "contract")
    _exact_keys(selected, frozen, "contract")
    for name, expected in frozen.items():
        if selected[name] != expected:
            raise ValueError(
                f"Ouro contract {name} differs from the frozen release recipe"
            )


def _validate_tensors(value: object) -> None:
    tensors = _object(value, "tensors")
    _exact_keys(tensors, _TENSOR_GROUPS, "tensors")
    adapters = _object(tensors["adapters"], "tensors.adapters")
    expected_adapters = set(adapter_tensor_names())
    _inventory(adapters, expected_adapters, "adapter")
    for name in adapter_tensor_names():
        _dtype(adapters[name], f"tensors.adapters.{name}")
    head = _object(tensors["head"], "tensors.head")
    _inventory(head, set(HEAD_TENSORS), "head")
    for name in HEAD_TENSORS:
        _dtype(head[name], f"tensors.head.{name}")


def _rendering(value: object) -> dict[str, str]:
    rendering = _object(value, "prompt.input_rendering")
    _exact_keys(rendering, _RENDERING_FIELDS, "prompt.input_rendering")
    mode = rendering["mode"]
    if type(mode) is not str or mode not in INPUT_RENDERING_MODES:
        raise ValueError(
            f"unsupported prompt.input_rendering mode: {rendering['mode']!r}"
        )
    source = _hash(rendering["source_sha256"], "prompt.input_rendering.source_sha256")
    return {"mode": mode, "source_sha256": source}


def _validate_execution(value: object) -> None:
    execution = _object(value, "execution")
    _exact_keys(execution, _EXECUTION_FIELDS, "execution")
    max_length = execution["max_length"]
    if (
        type(max_length) is not int
        or not 0 < max_length <= NATIVE_MAX_POSITION_EMBEDDINGS
    ):
        raise ValueError(
            "execution.max_length must be an integer in "
            f"[1, {NATIVE_MAX_POSITION_EMBEDDINGS}]"
        )
    pinned = (
        ("base_weight_dtype", EXECUTION_BASE_WEIGHT_DTYPE),
        ("compute_dtype", EXECUTION_COMPUTE_DTYPE),
        ("adapter_dtype", EXECUTION_ADAPTER_DTYPE),
        ("head_dtype", EXECUTION_HEAD_DTYPE),
    )
    for name, expected in pinned:
        if execution[name] != expected:
            raise ValueError(f"unsupported execution {name}: {execution[name]!r}")
    attention = execution["attention"]
    if type(attention) is not str or attention not in ATTENTION_IMPLEMENTATIONS:
        raise ValueError(f"unsupported attention implementation: {attention!r}")
    device = execution["device_policy"]
    if type(device) is not str or device not in DEVICE_POLICIES:
        raise ValueError(f"unsupported device policy: {device!r}")


def _validate_cache(value: object) -> None:
    cache = _object(value, "cache")
    _exact_keys(cache, _CACHE_FIELDS, "cache")
    if cache["policy"] != CACHE_POLICY:
        raise ValueError(f"unsupported cache policy: {cache['policy']!r}")
    if cache["match"] != CACHE_MATCH:
        raise ValueError(f"unsupported cache match: {cache['match']!r}")
    entries = cache["entries"]
    if type(entries) is not int or not 0 <= entries <= CACHE_MAX_ENTRIES:
        raise ValueError(
            f"cache entries must be an integer in [0, {CACHE_MAX_ENTRIES}]"
        )
    tokens = cache["state_block_tokens"]
    if type(tokens) is not int or not 0 <= tokens <= CACHE_MAX_STATE_BLOCK_TOKENS:
        raise ValueError(
            "cache state_block_tokens must be an integer in "
            f"[0, {CACHE_MAX_STATE_BLOCK_TOKENS}]"
        )
    if (entries == 0) != (tokens == 0):
        raise ValueError(
            "cache entries and state_block_tokens must both be zero (disabled) "
            "or both positive (enabled)"
        )
    tolerance = _object(cache["tolerance"], "cache.tolerance")
    _exact_keys(tolerance, _TOLERANCE_FIELDS, "cache.tolerance")
    for name, expected in (
        ("atol", CACHE_TOLERANCE_ATOL),
        ("rtol", CACHE_TOLERANCE_RTOL),
    ):
        number = tolerance[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError(f"cache tolerance {name} must be finite and positive")
        if number != expected:
            raise ValueError(
                f"cache tolerance {name} differs from the declared reuse gate"
            )
    if cache["bitwise_when_disabled"] is not CACHE_DISABLED_IS_BITWISE:
        raise ValueError("cache-disabled execution must be declared bitwise")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    )


def _object(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def _mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return {key: item for key, item in value.items()}


def _validate_file_hashes(files: Mapping[str, object]) -> None:
    _exact_keys(files, REQUIRED_FILES, "files_sha256")
    for name in sorted(REQUIRED_FILES):
        _hash(files[name], f"files_sha256.{name}")


def _exact_keys(
    value: Mapping[str, object],
    expected: Mapping[str, object] | Collection[str],
    name: str,
) -> None:
    expected_names = set(expected)
    unknown = set(value) - expected_names
    if unknown:
        raise ValueError(f"unsupported {name} fields: {_labels(unknown)}")
    missing = expected_names - set(value)
    if missing:
        raise ValueError(f"missing {name} fields: {_labels(missing)}")


def _inventory(value: Mapping[str, object], expected: set[str], name: str) -> None:
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"unsupported {name} tensors: {_labels(unknown)}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"missing {name} tensors: {_labels(missing)}")


def _labels(values: Collection[object]) -> str:
    return ", ".join(sorted(str(value) for value in values))


def _hash(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _dtype(value: object, name: str) -> str:
    if type(value) is not str or value not in TENSOR_DTYPES:
        raise ValueError(f"unsupported tensor dtype for {name}: {value!r}")
    return value
