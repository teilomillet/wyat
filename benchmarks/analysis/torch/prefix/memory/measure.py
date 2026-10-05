"""Bounded own-process RSS observations, exclusively for a memory-only cycle.

macOS ps(1) defines RSS in 1024-byte units. SELF ru_maxrss is cumulative and
uses the existing benchmark conversion; it is never substituted for current RSS.
ps snapshots may miss brief peaks. A sampler thread and ps children perturb this
cycle, so its duration is not an ordinary inference-performance observation.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from dataclasses import dataclass
from threading import Event, Thread
from types import TracebackType

from wyat.benchmark import process_peak_rss_bytes


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    pid: int
    wall_time_ns: int
    perf_counter_ns: int
    rss_bytes: int | None
    peak_rss_bytes: int | None
    collection_elapsed_ns: int
    error: str | None


def memory_snapshot() -> MemorySnapshot:
    """Read own PID only; record unavailable observations without zero filling."""
    start = time.perf_counter_ns()
    wall = time.time_ns()
    pid = os.getpid()
    rss: int | None = None
    peak: int | None = None
    errors: list[str] = []
    try:
        result = subprocess.run(
            ["/bin/ps", "-o", "rss=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
        text = result.stdout.strip()
        if not text.isascii() or not text.isdecimal():
            raise ValueError("ps did not return one nonnegative integer RSS")
        rss = int(text) * 1024
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append(f"current_rss: {type(error).__name__}: {str(error)[:256]}")
    try:
        peak = process_peak_rss_bytes()
    except OSError as error:
        errors.append(f"cumulative_hwm: {type(error).__name__}: {str(error)[:256]}")
    return MemorySnapshot(
        pid,
        wall,
        start,
        rss,
        peak,
        time.perf_counter_ns() - start,
        "; ".join(errors) or None,
    )


@dataclass(frozen=True, slots=True)
class MemoryReport:
    completed: bool
    requested_interval_ns: int
    max_samples: int
    max_seconds: float
    sample_limit_reached: bool
    time_limit_reached: bool
    collection_elapsed_ns: int
    observed_gaps_ns: tuple[int, ...]
    body_start_ns: int
    body_end_ns: int
    interior_valid_samples: int
    body_gaps_ns: tuple[int, ...]
    observer_errors: tuple[str, ...]
    body_error: str | None
    sampled_peak_rss_bytes: int | None
    samples: tuple[MemorySnapshot, ...]


class MemorySampler:
    """One bounded observer lifetime; no tensors or model references are held."""

    def __init__(
        self,
        *,
        interval_seconds: float = 0.02,
        max_samples: int = 512,
        max_seconds: float = 30,
    ) -> None:
        if (
            isinstance(interval_seconds, bool)
            or not math.isfinite(interval_seconds)
            or not 0.01 <= interval_seconds <= 1
            or type(max_samples) is not int
            or not 2 <= max_samples <= 1024
            or isinstance(max_seconds, bool)
            or not math.isfinite(max_seconds)
            or not 0.01 <= max_seconds <= 30
        ):
            raise ValueError("Memory sampler requires finite bounded settings")
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.max_seconds = max_seconds
        self._stop = Event()
        self._thread: Thread | None = None
        self._samples: list[MemorySnapshot] = []
        self._sample_limit = False
        self._time_limit = False
        self._started_ns = 0
        self._finished = False
        self._body_start_ns = 0
        self._body_end_ns = 0
        self._observer_errors: list[str] = []
        self._body_error: str | None = None

    def __enter__(self) -> MemorySampler:
        if self._started_ns:
            raise RuntimeError("Memory sampler cannot be reused")
        self._started_ns = time.perf_counter_ns()
        self._samples.append(memory_snapshot())
        self._thread = Thread(target=self._sample, name="own-rss-memory-cycle")
        self._thread.start()
        self._body_start_ns = time.perf_counter_ns()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if len(self._samples) >= self.max_samples - 1:
                self._sample_limit = True
                return
            elapsed = (time.perf_counter_ns() - self._started_ns) / 1e9
            if elapsed >= self.max_seconds:
                self._time_limit = True
                return
            try:
                self._samples.append(memory_snapshot())
            except Exception as error:
                self._observer_errors.append(f"{type(error).__name__}: {error}")
                return

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._body_end_ns = time.perf_counter_ns()
        if exc_value is not None:
            self._body_error = f"{type(exc_value).__name__}: {exc_value}"
        self._stop.set()
        if self._thread is None:
            raise RuntimeError("Memory sampler was not started")
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            raise RuntimeError("Memory observer failed to stop within its bound")
        self._samples.append(memory_snapshot())
        if (self._body_end_ns - self._started_ns) / 1e9 >= self.max_seconds:
            self._time_limit = True
        self._finished = True

    def report(self) -> MemoryReport:
        if not self._finished:
            raise RuntimeError("Memory report requires a stopped observer")
        samples = tuple(self._samples)
        interior = tuple(
            sample
            for sample in samples
            if sample.error is None
            and sample.rss_bytes is not None
            and sample.perf_counter_ns >= self._body_start_ns
            and sample.perf_counter_ns + sample.collection_elapsed_ns
            <= self._body_end_ns
        )
        rss = tuple(
            sample.rss_bytes for sample in interior if sample.rss_bytes is not None
        )
        # ps reads RSS somewhere within each recorded collection interval. Bound
        # the largest possible unknown read gap, including both body edges.
        gaps = (
            (
                interior[0].perf_counter_ns
                + interior[0].collection_elapsed_ns
                - self._body_start_ns,
            )
            + tuple(
                right.perf_counter_ns
                + right.collection_elapsed_ns
                - left.perf_counter_ns
                for left, right in zip(interior, interior[1:], strict=False)
            )
            + (self._body_end_ns - interior[-1].perf_counter_ns,)
            if interior
            else (self._body_end_ns - self._body_start_ns,)
        )
        return MemoryReport(
            not self._sample_limit
            and not self._time_limit
            and not self._observer_errors
            and self._body_error is None
            and all(sample.error is None for sample in samples),
            int(self.interval_seconds * 1e9),
            self.max_samples,
            self.max_seconds,
            self._sample_limit,
            self._time_limit,
            sum(sample.collection_elapsed_ns for sample in samples),
            tuple(
                right.perf_counter_ns - left.perf_counter_ns
                for left, right in zip(samples, samples[1:], strict=False)
            ),
            self._body_start_ns,
            self._body_end_ns,
            len(interior),
            gaps,
            tuple(self._observer_errors),
            self._body_error,
            max(rss) if rss else None,
            samples,
        )
