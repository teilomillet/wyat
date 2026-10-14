"""Read-only admission of the audited initial projection and exact captured bytes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from wyat.backends.wyat.artifact import load_artifact
from wyat.wire import JsonObject, JsonValue, read_object

ROOT = Path(__file__).resolve().parents[4]
INITIAL = ROOT / "benchmarks/analysis/results/mlx-mixed-gate.json"
REVIEW = INITIAL.with_name("mlx-mixed-gate-review.json")
VERIFICATION = ROOT / "benchmarks/real/mlx-mixed-tooling-verification.json"
IDENTITIES = {
    INITIAL: "fb8bde4a9a7d0c5b24c40a67183f582dc46c56c8d3ccbb79461d87065c023b8c",
    REVIEW: "08d8b0bd6f60b8f24a2374a1ead6a43cc32d0228d10c77b3f714379314c9e985",
    VERIFICATION: "e332f2fafda3712eca17f5637cc0490165349021a7f7b826432e84e0c7cd681b",
}
DOC = "docs/analysis/mixed.md"
DOC_SHA256 = "ab0ce4e460dd7614e6ee63a5bf839671902b446ccb9bb26044c33218d29d0f59"


def object_value(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected object")
    return value


def string(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected string")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(record: JsonValue) -> Path:
    entry = object_value(record)
    path = Path(string(entry["path"]))
    if digest(path) != string(entry["sha256"]):
        raise ValueError(f"Frozen artifact changed: {path.name}")
    return path


def archive(record: JsonValue, hashes: JsonObject, count: int) -> dict[str, Path]:
    """Archive text is immutable; only the named documentation may now differ."""
    value = read_object(checked_file(record))
    entries = value["files"]
    if not isinstance(entries, list) or len(entries) != count or len(hashes) != count:
        raise ValueError("Frozen source inventory changed")
    paths: dict[str, Path] = {}
    for part in entries:
        entry = object_value(part)
        name, text = string(entry["name"]), string(entry["text"])
        expected = string(entry["sha256"])
        if name in paths or hashlib.sha256(text.encode()).hexdigest() != expected:
            raise ValueError("Invalid archived source text or repeated name")
        if hashes.get(name) != expected:
            raise ValueError("Source archive differs from recorded inventory")
        path = Path(string(entry["path"]))
        if name == DOC and (
            digest(path) != DOC_SHA256 or not path.read_text().startswith(text)
        ):
            raise ValueError("Only the audited documentation append is admitted")
        if name != DOC and digest(path) != expected:
            raise ValueError(f"Frozen executable source changed: {name}")
        paths[name] = path
    if set(paths) != set(hashes):
        raise ValueError("Frozen source names changed")
    return paths


@dataclass(frozen=True, slots=True)
class Admission:
    initial: JsonObject
    inputs: Path
    cold: Path
    sources: dict[str, Path]


def admit(model: Path) -> Admission:
    for path, expected in IDENTITIES.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen admission receipt changed: {path.name}")
    initial, review, verification = map(read_object, IDENTITIES)
    if (
        initial["completed"] is not True
        or initial["capture_quality_passed"] is not True
        or initial["projection_quality_passed"] is not True
        or review["audit_passed"] is not True
        or verification["completed"] is not True
    ):
        raise ValueError("Initial projection evidence is incomplete")
    sources = archive(
        initial["source_archive"], object_value(initial["source_hashes"]), 43
    )
    sources.update(
        archive(
            verification["source_archive"],
            object_value(verification["source_sha256"]),
            104,
        )
    )
    artifacts = object_value(review["artifacts"])
    for record in artifacts.values():
        checked_file(record)
    inputs, cold = (
        checked_file(initial["captured_bytes"]),
        checked_file(initial["cold_outputs"]),
    )
    if inputs != checked_file(artifacts["inputs"]) or cold != checked_file(
        artifacts["cold_outputs"]
    ):
        raise ValueError("Independent audit names different tensor artifacts")
    artifact = load_artifact(model, max_length=512)
    if artifact.model_id != initial["model_id"]:
        raise ValueError("Frozen checkpoint identity changed")
    for path, field in (
        (ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl", "dataset_sha256"),
        (ROOT / "benchmarks/real/torch-fp32-reference77.json", "reference_sha256"),
    ):
        if digest(path) != initial[field]:
            raise ValueError("Frozen data/reference identity changed")
    return Admission(initial, inputs, cold, sources)
