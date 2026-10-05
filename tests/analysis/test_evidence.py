"""Source snapshot integrity and diagnostics without inference execution."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic


def test_frozen_sources_keep_exact_bytes_after_edit(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    original = '# café\r\nVALUE = "旧"\r\n'.encode()
    source.write_bytes(original)
    snapshot = freeze_sources({"runner": source}, tmp_path / "receipt.json")
    source.write_text("VALUE = 99\n")
    assert snapshot.hashes() == {"runner": hashlib.sha256(original).hexdigest()}
    archive = Path(snapshot.archive_path)
    payload = archive.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == snapshot.archive_sha256
    stored = json.loads(payload)["files"][0]
    assert stored["text"].encode() == original
    assert stored["sha256"] == hashlib.sha256(original).hexdigest()


def test_existing_archive_is_reused_only_when_identical(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    output = tmp_path / "receipt.json"
    first = freeze_sources({"runner": source}, output)
    assert freeze_sources({"runner": source}, output) == first
    Path(first.archive_path).write_text("altered archive")
    with pytest.raises(ValueError, match="archive differs"):
        freeze_sources({"runner": source}, output)


def test_diagnostic_uses_own_process_counters_and_records_missing_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic = iter((100, 150))
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(monotonic))
    monkeypatch.setattr(time, "time_ns", lambda: 1234)

    def rusage(who: int) -> SimpleNamespace:
        assert who == resource.RUSAGE_SELF
        return SimpleNamespace(
            ru_utime=1.5,
            ru_stime=0.25,
            ru_minflt=10,
            ru_majflt=2,
            ru_nvcsw=3,
            ru_nivcsw=4,
        )

    def unavailable() -> tuple[float, float, float]:
        raise OSError("load unavailable")

    monkeypatch.setattr(resource, "getrusage", rusage)
    monkeypatch.setattr(os, "getloadavg", unavailable)
    observation = process_diagnostic()
    assert observation.wall_time_ns == 1234
    assert observation.perf_counter_ns == 100
    assert observation.collection_elapsed_ns == 50
    assert observation.user_seconds == 1.5
    assert observation.system_seconds == 0.25
    assert (observation.minor_faults, observation.major_faults) == (10, 2)
    assert observation.voluntary_context_switches == 3
    assert observation.involuntary_context_switches == 4
    assert observation.load_average is None
    assert observation.load_average_error == "OSError: load unavailable"


def test_device_counter_snapshot_does_not_retain_mutable_input() -> None:
    counters = {"active": 10, "cache": 20, "peak": 40}
    observation = process_diagnostic(device_memory_bytes=counters)
    counters["cache"] = 999
    assert observation.device_memory_bytes == {"active": 10, "cache": 20, "peak": 40}
