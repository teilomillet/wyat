# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Cooperative macOS admission checks that leave room for other applications.

The memory-pressure free percentage is a conservative capacity signal, not an
exact process RSS measurement. Checks can refuse new work and stop between
operations; they cannot preempt an already submitted kernel or control another
application. PyTorch thread and MPS allocator settings apply to the whole process.
No runtime package is imported here, and non-macOS guards do no system probing.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from typing import Any


class ResourceLimitError(MemoryError):
    """Local work was refused to preserve the configured resource headroom."""


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Physical capacity and the available amount reported by memory pressure."""

    total_bytes: int
    available_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes


_WORK_LOCK = RLock()


_CURRENT_GUARD: ContextVar[ResourceGuard | None] = ContextVar(
    "wyat_resource_guard", default=None
)


class ResourceGuard:
    """Refuse projected memory use above ``max_fraction`` on macOS.

    The default leaves 15% of the Mac's capacity available. Estimates supplied
    by callers must include additional peak allocations, rather than just the
    final retained tensors. Missing telemetry refuses work rather than silently
    disabling the guard. Other operating systems keep their existing behavior.
    """

    def __init__(self, max_fraction: float = 0.85) -> None:
        if (
            isinstance(max_fraction, bool)
            or not isinstance(max_fraction, Real)
            or not math.isfinite(max_fraction)
            or not 0 < max_fraction <= 1
        ):
            raise ValueError("max_resource_fraction must be a finite number in (0, 1]")
        self.max_fraction: float = float(max_fraction)
        self.enabled: bool = sys.platform == "darwin"
        self._total_bytes: int | None = None
        self._last_checkpoint: float | None = None

    def _snapshot(self) -> ResourceSnapshot:
        try:
            if self._total_bytes is None:
                total = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=1.0,
                )
                self._total_bytes = int(total.stdout.strip())
                if self._total_bytes <= 0:
                    self._total_bytes = None
                    raise ValueError("nonpositive physical memory capacity")
            pressure = subprocess.run(
                ["/usr/bin/memory_pressure", "-Q"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            match = re.search(
                r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%",
                pressure.stdout,
            )
            if match is None:
                raise ValueError("memory pressure did not report available capacity")
            free_percent = float(match.group(1))
            if not 0 <= free_percent <= 100:
                raise ValueError("invalid available capacity percentage")
            return ResourceSnapshot(
                self._total_bytes, int(self._total_bytes * free_percent / 100)
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ResourceLimitError(
                "Cannot check macOS memory headroom; local model work was refused. "
                "Check that sysctl and memory_pressure are available."
            ) from error

    def check(
        self, additional_bytes: int = 0, operation: str = "inference"
    ) -> ResourceSnapshot | None:
        """Check a fresh system snapshot before an additional peak allocation."""
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("additional_bytes must be a nonnegative integer")
        if not self.enabled:
            return None
        snapshot = self._snapshot()
        projected = snapshot.used_bytes + additional_bytes
        ceiling = int(snapshot.total_bytes * self.max_fraction)
        if projected > ceiling:
            raise ResourceLimitError(
                f"Refusing {operation}: projected memory use exceeds the "
                f"{self.max_fraction:.0%} macOS resource limit "
                f"({projected / 2**30:.2f} GiB projected, "
                f"{ceiling / 2**30:.2f} GiB allowed). "
                "Close other applications, shorten the input, or explicitly change "
                "max_resource_fraction."
            )
        return snapshot

    def configure_torch(self, torch: Any, device: Any) -> None:
        """Bound CPU threads and, on MPS, the process allocator's memory limit."""
        if not self.enabled:
            return
        device_type = getattr(device, "type", str(device).split(":", 1)[0])
        if device_type not in {"cpu", "mps"}:
            return
        snapshot = self.check(operation="runtime configuration")
        assert snapshot is not None
        threads = max(1, math.floor((os.cpu_count() or 1) * self.max_fraction))
        current_threads = torch.get_num_threads()
        if current_threads > threads:
            torch.set_num_threads(threads)
        if device_type == "mps":
            try:
                recommended = torch.mps.recommended_max_memory()
                if (
                    isinstance(recommended, bool)
                    or not isinstance(recommended, Real)
                    or not math.isfinite(recommended)
                    or recommended <= 0
                ):
                    raise ValueError("invalid recommended MPS memory capacity")
                fraction = min(
                    self.max_fraction,
                    self.max_fraction * snapshot.total_bytes / recommended,
                )
                torch.mps.set_per_process_memory_fraction(fraction)
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise ResourceLimitError(
                    "Cannot configure the Apple GPU memory limit; model work was "
                    "refused. Use a PyTorch version with MPS memory-limit support."
                ) from error

    def configure_mlx(self, mx: Any) -> None:
        """Reduce MLX's memory guideline and disable retained free allocations.

        MLX documents its memory limit as a guideline, so admission and
        cooperative checkpoints remain necessary. Existing tighter limits are
        preserved. The cache setting controls unused allocations, not model KV.
        """
        if not self.enabled:
            return
        snapshot = self.check(operation="runtime configuration")
        assert snapshot is not None
        try:
            current = mx.get_memory_limit()
            if type(current) is not int or current <= 0:
                raise ValueError("invalid MLX memory limit")
            mx.set_memory_limit(
                min(current, int(snapshot.total_bytes * self.max_fraction))
            )
            mx.set_cache_limit(0)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ResourceLimitError(
                "Cannot configure the MLX memory limit; model work was refused. "
                "Use an MLX version with memory-limit support."
            ) from error

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this guard available to cooperative inference checkpoints."""
        with _WORK_LOCK if self.enabled else nullcontext():
            token = _CURRENT_GUARD.set(self)
            try:
                yield
            finally:
                _CURRENT_GUARD.reset(token)


def current_guard() -> ResourceGuard | None:
    """Return the guard belonging to the current load or inference operation."""
    return _CURRENT_GUARD.get()


def checkpoint() -> None:
    """Check the active guard at most once per second between operations."""
    guard = _CURRENT_GUARD.get()
    if guard is None or not guard.enabled:
        return
    now = time.monotonic()
    if guard._last_checkpoint is None or now - guard._last_checkpoint >= 1.0:
        guard.check()
        guard._last_checkpoint = now
