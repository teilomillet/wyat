"""Pinned histories relocate; new executions freeze actual current source owners."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from transformers import cache_utils, masking_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.torch.prefix.latency import study as latency
from benchmarks.analysis.torch.prefix.memory import study as memory
from benchmarks.analysis.torch.prefix.pairs import admit_history


@pytest.fixture
def relocated(tmp_path: Path) -> Path:
    results = tmp_path / memory.RESULTS_DIRECTORY
    results.mkdir(parents=True)
    for path in (memory.ROOT / memory.RESULTS_DIRECTORY).glob("torch-prefix*"):
        if path.suffix in {".json", ".jsonl"}:
            shutil.copyfile(path, results / path.name)
    for name in (
        "banking77-train-dev-menu6.jsonl",
        "torch-fp32-reference77.json",
    ):
        source = memory.ROOT / "benchmarks/real" / name
        target = tmp_path / "benchmarks/real" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for path in memory.source_paths().values():
        if path.is_relative_to(memory.ROOT) and ".venv" not in path.parts:
            target = tmp_path / path.relative_to(memory.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return tmp_path


def deny_old_results(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    original = Path.read_bytes
    reads: list[Path] = []

    def read(path: Path) -> bytes:
        reads.append(path)
        if path.is_relative_to(memory.LEGACY_RESULTS_DIRECTORY):
            raise FileNotFoundError("Original developer checkout is unavailable")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    return reads


def test_pinned_histories_admit_without_accessing_original_checkout(
    relocated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = deny_old_results(monkeypatch)
    # This is the formerly failing join: an absolute operand discards the new root.
    legacy_archive = memory.LEGACY_RESULTS_DIRECTORY / (
        "torch-prefix77-b8.sources-d86eb841ae184bbb.json"
    )
    with pytest.raises(FileNotFoundError, match="developer checkout"):
        (relocated / legacy_archive).read_bytes()
    reads.clear()
    history = admit_history(relocated)
    assert memory.admit_confirmation(history, relocated)["raw_outputs"] == 2156
    admitted, paths = latency.admit_resource(relocated)
    assert admitted["model_id"] == history["model_id"]
    assert paths == memory.source_paths(relocated)
    assert len(paths) == 21 and all(path.is_file() for path in paths.values())
    assert reads and all(
        not path.is_relative_to(legacy_archive.parent) for path in reads
    )


def test_current_source_registry_uses_current_imported_dependencies(
    relocated: Path,
) -> None:
    paths = memory.source_paths(relocated)
    dependencies = {
        "maintained_cache": Path(cache_utils.__file__),
        "maintained_mask": Path(masking_utils.__file__),
        "qwen3": Path(modeling_qwen3.__file__),
        "sdpa": Path(sdpa_attention.__file__),
    }
    assert {name: paths[name] for name in dependencies} == dependencies
    assert all(
        path.is_relative_to(relocated)
        for name, path in paths.items()
        if name not in dependencies
    )


@pytest.mark.parametrize(
    "value",
    [
        "/unrelated/benchmarks/analysis/results/history.json",
        "/Users/teilomillet/Code/wyat/private/history.json",
        "benchmarks/analysis/results/../history.json",
        "benchmarks/analysis/results/nested/history.json",
        "benchmarks/analysis/results",
        "history.json",
        "benchmarks/analysis/results\\history.json",
    ],
)
def test_evidence_location_cannot_escape_pinned_results_folder(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError, match="Historical evidence"):
        memory.evidence_path(value, tmp_path)


def test_legacy_and_relative_locations_bind_same_current_artifact(
    tmp_path: Path,
) -> None:
    name = "torch-prefix77-b8.sources-d86eb841ae184bbb.json"
    expected = tmp_path / memory.RESULTS_DIRECTORY / name
    assert memory.evidence_path(
        str(memory.LEGACY_RESULTS_DIRECTORY / name), tmp_path
    ) == (expected)
    assert (
        memory.evidence_path(str(memory.RESULTS_DIRECTORY / name), tmp_path) == expected
    )


def test_evidence_symlink_outside_results_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / memory.RESULTS_DIRECTORY
    results.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (results / "history.json").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        memory.evidence_path(str(memory.RESULTS_DIRECTORY / "history.json"), tmp_path)


@pytest.mark.parametrize("kind", ["full77", "memory"])
def test_relocated_tampered_archive_still_fails_its_original_hash(
    relocated: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    deny_old_results(monkeypatch)
    name = (
        "torch-prefix77-b8.sources-d86eb841ae184bbb.json"
        if kind == "full77"
        else "torch-prefix-memory-retained-0.sources-2ea7fcb085c78c3d.json"
    )
    archive = relocated / memory.RESULTS_DIRECTORY / name
    archive.write_bytes(archive.read_bytes() + b" ")
    with pytest.raises(
        ValueError, match="history changed|receipt/journal/wall/archive"
    ):
        latency.admit_resource(relocated)
