# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Fail closed on known publication disclosures before freezing release bytes.

These local checks complement an independent review of the exact inventory.
They do not identify arbitrary confidential prose or measure privacy in weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._~+/-]{16,}=*)"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:([^\s/@]+)@"),
)
_ASSIGNMENT = re.compile(
    r"(?i)[\"']?\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|"
    r"secret[_-]?key|client[_-]?secret|password|passwd|token)\b[\"']?"
    r"\s*[:=]\s*[\"']([^\"'\r\n]{8,})[\"']"
)
_PRIVATE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:/(?:Users|home|root)/|[A-Z]:[\\/]Users[\\/])"
    r"([^\s/\\\"'<>]+)"
)
_INTERNAL_PATH = re.compile(
    r"\bartifacts/(?:[a-z0-9_.-]*\d[a-z0-9_.-]*|huggingface-releases/[a-z0-9_.-]+)"
)
_PLACEHOLDERS = {
    "example",
    "example_password",
    "your_password",
    "your_token",
    "your_api_key",
    "your_secret",
    "placeholder",
    "redacted",
    "changeme",
    "password",
    "token",
    "username",
    "user",
    "yourname",
    "name",
    "test",
    "test_password",
}
_MODEL_NAMES = {
    ".gitattributes",
    "FILES.sha256",
    "LICENSE",
    "NOTICE",
    "README.md",
    "RELEASE.json",
    "example.py",
    "runtime/SOURCE.json",
    "runtime/requirements.txt",
    "model/adapters.safetensors",
    "model/backbone/config.json",
    "model/backbone/model.safetensors",
    "model/base/config.json",
    "model/base/model.safetensors",
    "model/decision_head.safetensors",
    "model/fork_manifest.json",
    "model/ouro_manifest.json",
    "model/config.json",
    "model/model.safetensors",
    "model/configuration_ouro.py",
    "model/modeling_ouro.py",
    "model/special_tokens_map.json",
    "model/head.safetensors",
    "model/merges.txt",
    "model/native-adapters.safetensors",
    "model/native_manifest.json",
    "model/tokenizer.json",
    "model/tokenizer_config.json",
    "model/vocab.json",
    *(
        f"benchmarks/{name}"
        for name in (
            "BENCHMARKS.md",
            "PROVENANCE.json",
            "comparison.json",
            "coverage.json",
            "decisionbench-toxicity-excluded.json",
            "decisionbench.json",
            "dmbench-french.json",
            "execution.json",
            "quality.json",
            "screens.json",
        )
    ),
    *(
        f"verification/{name}.json"
        for name in (
            "ATTRIBUTION",
            "BUILD",
            "EXAMPLE-VERIFICATION",
            "FAILURE-BOUNDARY",
            "VERIFICATION",
            "PRIVACY",
        )
    ),
    *(
        f"runtime/wyat/{name}.py"
        for name in (
            "__init__",
            "backends/__init__",
            "backends/wyat/__init__",
            "backends/wyat/artifact",
            "backends/wyat/encoding",
            "backends/wyat/torch",
            "core",
            "download",
            "input_rendering",
            "loading/__init__",
            "loading/fork",
            "loading/fork_artifact",
            "loading/local",
            "loading/native",
            "loading/native_artifact",
            "loading/ouro",
            "loading/ouro_artifact",
            "tasks",
            "wire",
        )
    ),
}
# Exact public Qwen vocabulary carried by the reviewed option-fork export.
# Vocabulary tokens can resemble private record keys; only these bytes qualify.
_FORK_VOCAB_SHA256 = "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
_RAW_KEYS = {
    "raw_request",
    "raw_requests",
    "request_text",
    "raw_predictions",
    "predictions",
    "training_data",
    "training_examples",
    "training_records",
    "train_records",
    "row_ids",
    "question_ids",
    "rows_data",
    "records",
    "raw_records_data",
    "per_record",
    "by_record",
    "customer_email",
    "customer_name",
    "customer_address",
    "email_address",
    "phone_number",
    "access_token",
    "api_key",
    "client_secret",
    "password",
    "credential",
    "credentials",
}
_BENCHMARK_KEYS = _RAW_KEYS | {
    "request",
    "requests",
    "input",
    "inputs",
    "target",
    "targets",
    "candidate",
    "candidates",
    "labels",
    "state",
    "answers",
    "per_class_f1",
    "gold_label_share",
    "pred_label_share",
    "over_picked",
    "under_picked",
    "by_gold_category",
    "gold_category_share",
    "pred_category_share",
    "never_gold_categories_picked",
    "gold_slot_share",
    "pred_slot_share",
    "accuracy_by_gold_slot",
    "by_length_bucket",
    "by_category",
    "row_ids_sha256",
    "question_keys_sha256",
    "prediction_sha256",
    "prediction_pin",
    "cohort_sha256",
    "screen_sha256",
    "source_artifacts",
    "original_source_artifacts",
    "evidence_pins",
    "panels",
    "panel_presentations",
    "suite_sha256",
    "plan_sha256",
}
_DENOMINATORS = {"n", "rows", "questions", "n_questions", "n_requests"}
MINIMUM_AGGREGATE_SIZE = 20
_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _placeholder(value: str) -> bool:
    value = value.strip().lower()
    return (
        value in _PLACEHOLDERS
        or value.startswith(("${", "$env:", "<", "os.environ", "example_", "your_"))
        or value == "*" * len(value)
    )


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or len(name) > 180
        or not re.fullmatch(r"[A-Za-z0-9_./+-]+", name)
        or any(pattern.search(name) for pattern in _SECRET_PATTERNS)
    ):
        return "[unrecognized file]"
    return name


def _fail(name: str, category: str, line: int | None = None) -> None:
    location = _safe_name(name)
    if line is not None:
        location += f":{line}"
    raise ValueError(f"Publication privacy check failed ({category}): {location}")


def _scan_text(name: str, text: str) -> None:
    for pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if match.lastindex and _placeholder(match.group(1)):
                continue
            _fail(name, "credential material", text.count("\n", 0, match.start()) + 1)
    for match in _ASSIGNMENT.finditer(text):
        value = match.group(1)
        if not _placeholder(value):
            _fail(name, "credential assignment", text.count("\n", 0, match.start()) + 1)
    for match in _PRIVATE_PATH.finditer(text):
        if not _placeholder(match.group(1)):
            _fail(
                name, "private filesystem path", text.count("\n", 0, match.start()) + 1
            )
    internal_match = _INTERNAL_PATH.search(text)
    if internal_match:
        _fail(
            name,
            "private operational path",
            text.count("\n", 0, internal_match.start()) + 1,
        )


def check_text(name: str, raw: bytes) -> None:
    """Scan UTF-8 source/metadata without printing matched private content."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        _fail(name, "unsupported non-text payload")
    _scan_text(name, text)
    # Escaped JSON strings carry the same disclosure as their literal spelling.
    if name.endswith((".json", ".jsonl")):
        documents = text.splitlines() if name.endswith(".jsonl") else [text]
        for document in documents:
            value = _json(name, document.encode("utf-8"))
            stack = [value]
            while stack:
                current = stack.pop()
                if isinstance(current, str):
                    _scan_text(name, current)
                elif isinstance(current, dict):
                    stack.extend(current.keys())
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys")
        result[key] = value
    return result


def _json(name: str, raw: bytes) -> Any:
    try:

        def reject_constant(value: str) -> Any:
            raise ValueError("Non-finite JSON number")

        return json.loads(
            raw, object_pairs_hook=_object, parse_constant=reject_constant
        )
    except (ValueError, UnicodeError, RecursionError):
        _fail(name, "invalid or ambiguous JSON")


def _model_json(name: str, value: Any, *, raw: bytes) -> None:
    """Reject raw collections and class-level diagnostics, including object keys."""
    if not isinstance(value, dict):
        _fail(name, "unexpected JSON root schema")
    forbidden = _BENCHMARK_KEYS if name.startswith("benchmarks/") else _RAW_KEYS
    if name == "model/tokenizer.json":
        # Public vocabulary equality is reviewed separately and bound to inventory.
        return
    if name == "model/vocab.json":
        if hashlib.sha256(raw).hexdigest() != _FORK_VOCAB_SHA256:
            _fail(name, "unrecognized public vocabulary")
        return
    if name == "verification/VERIFICATION.json":
        fixture = value.get("synthetic_fixture")
        if fixture is not True and not (
            isinstance(fixture, dict)
            and set(fixture) == {"state", "questions"}
            and value.get("synthetic_fixture_documented") is True
        ):
            _fail(name, "undocumented verification fixture")
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if name.startswith("benchmarks/") and any(
                key in current and (type(current[key]) is not int or current[key] < 0)
                for key in _DENOMINATORS
            ):
                _fail(name, "invalid aggregate count")
            if name.startswith("benchmarks/") and any(
                type(current.get(key)) is int
                and 0 < current[key] < MINIMUM_AGGREGATE_SIZE
                for key in _DENOMINATORS
            ):
                if current.get("suppressed") is not True or not set(
                    current
                ) <= _DENOMINATORS | {"suppressed"}:
                    _fail(name, "unsuppressed small aggregate cell")
            for key, item in current.items():
                normalized = key.lower().replace("-", "_").replace(" ", "_")
                if normalized in forbidden:
                    _fail(name, "raw or fine-grained record field")
                if key.lstrip().startswith(("{", "[")) or any(
                    ord(char) < 32 for char in key
                ):
                    _fail(name, "input-shaped object key")
                stack.append(item)
        elif isinstance(current, list):
            stack.extend(current)


def _safetensors(name: str, path: Path) -> None:
    """Validate the container without loading its tensor data into memory."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            _fail(name, "invalid tensor header")
        length = struct.unpack("<Q", prefix)[0]
        if length < 2 or length > min(100_000_000, size - 8):
            _fail(name, "invalid tensor header length")
        header_raw = stream.read(length)
    check_text(name, header_raw)
    header = _json(name, header_raw)
    if not isinstance(header, dict) or "__metadata__" in header:
        _fail(name, "arbitrary tensor metadata")
    offsets = []
    for tensor in header.values():
        if not isinstance(tensor, dict) or set(tensor) != {
            "dtype",
            "shape",
            "data_offsets",
        }:
            _fail(name, "invalid tensor declaration")
        dtype, shape, span = tensor["dtype"], tensor["shape"], tensor["data_offsets"]
        if (
            not isinstance(dtype, str)
            or dtype not in _DTYPE_BYTES
            or not isinstance(shape, list)
            or len(shape) > 32
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(span, list)
            or len(span) != 2
            or any(type(offset) is not int or offset < 0 for offset in span)
            or span[1] - span[0] != math.prod(shape) * _DTYPE_BYTES[dtype]
        ):
            _fail(name, "invalid tensor dimensions or offsets")
        offsets.append(tuple(span))
    end = 0
    for start, stop in sorted(offsets):
        if start != end or stop < start:
            _fail(name, "tensor data gaps or overlaps")
        end = stop
    if end != size - 8 - length:
        _fail(name, "unreferenced tensor bytes")


def verbatim_ouro_source(name: str, raw: bytes) -> bool:
    """Admit only the unchanged, release-pinned upstream native Python files.

    This is an attribution exemption, never a content-scan exemption. Keep the
    upstream copyright and license notices instead of adding project authorship.
    """
    if name not in {"model/configuration_ouro.py", "model/modeling_ouro.py"}:
        return False
    from wyat.loading.ouro_artifact import PINNED_BASE_FILES

    expected = PINNED_BASE_FILES[PurePosixPath(name).name]
    if (
        len(raw) != expected["size"]
        or hashlib.sha256(raw).hexdigest() != expected["sha256"]
    ):
        _fail(name, "modified or unrecognized Ouro native source")
    return True


def check_files(
    selected: Mapping[str, Path], *, profile: Literal["source", "model"] = "source"
) -> None:
    """Check selected regular files before export, archive, or model publication."""
    if profile not in {"source", "model"}:
        raise ValueError("Unknown publication privacy profile")
    for name, path in sorted(selected.items()):
        logical = PurePosixPath(name)
        if (
            logical.is_absolute()
            or ".." in logical.parts
            or "\\" in name
            or name != logical.as_posix()
            or not logical.parts
            or any(part in {".git", "__pycache__"} for part in logical.parts)
            or path.is_symlink()
            or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents)
        ):
            _fail(name, "unsafe publication path")
        if profile == "model" and name not in _MODEL_NAMES:
            _fail("[unadmitted file]", "unadmitted model artifact")
        if profile == "model" and path.suffix == ".safetensors":
            _safetensors(name, path)
        else:
            raw = path.read_bytes()
            check_text(name, raw)
            if profile == "model":
                verbatim_ouro_source(name, raw)
            if profile == "model" and name.endswith(".json"):
                _model_json(name, _json(name, raw), raw=raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, nargs="?", const=Path.cwd(), required=True
    )
    args = parser.parse_args()
    from scripts.package.check import required_sources

    root = args.source.resolve()
    check_files({name: root / name for name in required_sources(root)})
    print("Public source content checks passed; independent review remains required.")


if __name__ == "__main__":
    main()
