"""Own-RSS observer parsing, failure and shutdown; no models or large allocations."""

from __future__ import annotations

import subprocess
import time
from threading import Event, Thread
from typing import cast

import pytest

from benchmarks.analysis.torch.prefix.memory import measure


def fake_snapshot() -> measure.MemorySnapshot:
    now = time.perf_counter_ns()
    return measure.MemorySnapshot(123, time.time_ns(), now, 4096, 8192, 0, None)


def test_current_rss_units_hwm_separation_and_query_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, " 1234\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(measure, "process_peak_rss_bytes", lambda: 7654321)
    actual = measure.memory_snapshot()
    assert actual.rss_bytes == 1234 * 1024
    assert actual.peak_rss_bytes == 7654321 and actual.error is None
    assert calls[0][0] == ["/bin/ps", "-o", "rss=", "-p", str(actual.pid)]
    assert calls[0][1]["timeout"] == 1 and calls[0][1]["check"] is True
    assert actual.collection_elapsed_ns >= 0


@pytest.mark.parametrize("text", ["", "12\n34", "-1", "NaN", "１２"])
def test_unavailable_rss_is_not_filled_with_hwm_or_zero(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, text, ""),
    )
    monkeypatch.setattr(measure, "process_peak_rss_bytes", lambda: 999)
    actual = measure.memory_snapshot()
    assert actual.rss_bytes is None and actual.peak_rss_bytes == 999
    assert actual.error is not None and "current_rss" in actual.error


def test_timeout_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["/bin/ps"], 1)

    monkeypatch.setattr(subprocess, "run", timed_out)
    actual = measure.memory_snapshot()
    assert actual.rss_bytes is None
    assert actual.error is not None and "TimeoutExpired" in actual.error


def test_observer_coverage_reuse_and_stopped_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = measure.MemorySampler(interval_seconds=0.01)
    observations_ready = Event()
    interior_calls = 0

    def observed() -> measure.MemorySnapshot:
        nonlocal interior_calls
        snapshot = fake_snapshot()
        if (
            observer._body_start_ns
            and snapshot.perf_counter_ns >= observer._body_start_ns
        ):
            interior_calls += 1
            if interior_calls >= 3:
                observations_ready.set()
        return snapshot

    monkeypatch.setattr(measure, "memory_snapshot", observed)
    with pytest.raises(RuntimeError, match="stopped"):
        observer.report()
    with observer:
        # Wait for observations, without promising an OS scheduling frequency.
        assert observations_ready.wait(timeout=3)
    result = observer.report()
    assert result.completed and result.interior_valid_samples >= 3
    assert result.body_start_ns < result.body_end_ns
    assert result.sampled_peak_rss_bytes == 4096
    assert not cast(Thread, observer._thread).is_alive()
    with pytest.raises(RuntimeError, match="reused"):
        with observer:
            pass


def test_zero_interior_observations_remain_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measure, "memory_snapshot", fake_snapshot)
    with measure.MemorySampler(interval_seconds=1) as observer:
        pass
    result = observer.report()
    assert result.completed and result.interior_valid_samples == 0
    assert len(result.body_gaps_ns) == 1
    assert result.sampled_peak_rss_bytes is None
    assert result.samples[0].rss_bytes == 4096


def test_interior_peak_and_conservative_unknown_read_gaps() -> None:
    observer = measure.MemorySampler()
    observer._finished = True
    observer._body_start_ns = 0
    observer._body_end_ns = 300_000_000
    observer._samples = [
        measure.MemorySnapshot(1, 0, -30_000_000, 9000, 9000, 10_000_000, None),
        measure.MemorySnapshot(1, 0, 20_000_000, 10, 9000, 80_000_000, None),
        measure.MemorySnapshot(1, 0, 100_000_000, 20, 9000, 80_000_000, None),
        measure.MemorySnapshot(1, 0, 180_000_000, 15, 9000, 80_000_000, None),
        measure.MemorySnapshot(1, 0, 310_000_000, 9999, 9999, 10_000_000, None),
    ]
    result = observer.report()
    assert result.sampled_peak_rss_bytes == 20 and result.interior_valid_samples == 3
    assert result.body_gaps_ns == (100_000_000, 160_000_000, 160_000_000, 120_000_000)
    # Unknown read times inside long queries make 80ms start gaps insufficient.
    assert max(result.body_gaps_ns) > 100_000_000
    assert result.samples[0].rss_bytes == 9000 and result.samples[-1].rss_bytes == 9999


def test_scripted_snapshot_windows_cover_the_body_with_bounded_gaps() -> None:
    observer = measure.MemorySampler()
    observer._finished = True
    observer._body_start_ns = 0
    observer._body_end_ns = 100_000_000
    observer._samples = [
        measure.MemorySnapshot(1, 0, -10_000_000, 9000, 9000, 0, None),
        measure.MemorySnapshot(1, 0, 20_000_000, 10, 9000, 10_000_000, None),
        measure.MemorySnapshot(1, 0, 50_000_000, 20, 9000, 10_000_000, None),
        measure.MemorySnapshot(1, 0, 80_000_000, 15, 9000, 10_000_000, None),
        measure.MemorySnapshot(1, 0, 110_000_000, 9999, 9999, 0, None),
    ]
    result = observer.report()
    assert result.completed and result.interior_valid_samples == 3
    assert result.sampled_peak_rss_bytes == 20
    assert result.body_gaps_ns == (30_000_000, 40_000_000, 40_000_000, 20_000_000)
    assert max(result.body_gaps_ns) <= 100_000_000


def test_thread_failure_cannot_report_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    failed = Event()

    def observed() -> measure.MemorySnapshot:
        nonlocal calls
        calls += 1
        if calls == 2:
            failed.set()
            raise RuntimeError("injected observer failure")
        return fake_snapshot()

    monkeypatch.setattr(measure, "memory_snapshot", observed)
    with measure.MemorySampler(interval_seconds=0.01) as observer:
        assert failed.wait(timeout=3)
    result = observer.report()
    assert not result.completed and result.observer_errors
    assert "injected observer failure" in result.observer_errors[0]
    assert not cast(Thread, observer._thread).is_alive()


def test_body_exception_cleanup_and_partial_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measure, "memory_snapshot", fake_snapshot)
    observer = measure.MemorySampler()
    with pytest.raises(ValueError, match="body failed"), observer:
        raise ValueError("body failed")
    result = observer.report()
    assert not result.completed and result.body_error == "ValueError: body failed"
    assert not cast(Thread, observer._thread).is_alive()


def test_sample_and_time_limits_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measure, "memory_snapshot", fake_snapshot)
    with measure.MemorySampler(interval_seconds=0.01, max_samples=2) as observer:
        thread = cast(Thread, observer._thread)
        thread.join(timeout=2)
        assert not thread.is_alive()
    result = observer.report()
    assert not result.completed and result.sample_limit_reached
    assert len(result.samples) == 2
    with measure.MemorySampler(interval_seconds=0.01, max_seconds=0.01) as observer:
        thread = cast(Thread, observer._thread)
        thread.join(timeout=2)
        assert not thread.is_alive()
    result = observer.report()
    assert not result.completed and result.time_limit_reached


@pytest.mark.parametrize(
    "options",
    [
        {"interval_seconds": True},
        {"interval_seconds": float("nan")},
        {"max_samples": True},
        {"max_samples": 1},
        {"max_seconds": float("inf")},
    ],
)
def test_invalid_bounds(options: dict[str, float | int]) -> None:
    with pytest.raises(ValueError, match="bounded"):
        measure.MemorySampler(**options)  # type: ignore[arg-type]
