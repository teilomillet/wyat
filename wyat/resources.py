# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Cooperative macOS admission checks that leave room for other applications.

Available RAM is estimated from free plus inactive VM pages. Compressed and
wired memory are not added to that estimate. This is not process RSS.
Checks refuse new work and stop between operations; they cannot preempt a
submitted kernel or control another application. PyTorch thread and MPS
allocator settings apply to the whole process. No runtime package is imported
here, and non-macOS guards do no system probing.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from typing import Any

DEFAULT_MAX_RESOURCE_FRACTION = 0.90
"""Default share of macOS capacity that admitted work may project to use."""


class ResourceLimitError(MemoryError):
    """Local work was refused to preserve the configured resource headroom."""


class MemoryHeadroomWarning(UserWarning):
    """Local work may exceed the default memory headroom and was not refused."""


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Physical capacity and available RAM estimated from Mach VM page counts."""

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
    """Check projected memory use against ``max_fraction`` on macOS.

    Without an explicit fraction the guard is advisory: work that would pass
    the default 90% line still runs, and a ``MemoryHeadroomWarning`` says how
    much memory it needs and how much is available, once per kind of
    operation. An explicit fraction makes the guard refuse with
    ``ResourceLimitError`` instead, and also bounds the Apple GPU allocators.
    Estimates supplied by callers must include additional peak allocations,
    rather than just the final retained tensors. Other operating systems keep
    their existing behavior.
    """

    def __init__(self, max_fraction: float | None = None) -> None:
        self.enforcing: bool = max_fraction is not None
        if max_fraction is None:
            max_fraction = DEFAULT_MAX_RESOURCE_FRACTION
        if (
            isinstance(max_fraction, bool)
            or not isinstance(max_fraction, Real)
            or not math.isfinite(max_fraction)
            or not 0 < max_fraction <= 1
        ):
            raise ValueError("max_resource_fraction must be a finite number in (0, 1]")
        self.max_fraction: float = float(max_fraction)
        self._warned: set[str] = set()
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
            statistics = subprocess.run(
                ["/usr/bin/vm_stat"],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            page_size = re.search(r"page size of (\d+) bytes", statistics.stdout)
            pages = dict(
                re.findall(
                    r"^Pages (free|inactive):\s+(\d+)\.\s*$",
                    statistics.stdout,
                    re.MULTILINE,
                )
            )
            if page_size is None or set(pages) != {"free", "inactive"}:
                raise ValueError("VM statistics did not report available page counts")
            size = int(page_size.group(1))
            if size <= 0:
                raise ValueError("invalid VM page size")
            # Darwin's conventional available-RAM estimate is free + inactive.
            # Do not add speculative/purgeable pages again or compressed pages:
            # those overlap resident categories or are already occupied RAM.
            available = (int(pages["free"]) + int(pages["inactive"])) * size
            return ResourceSnapshot(
                self._total_bytes, min(self._total_bytes, available)
            )
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ResourceLimitError(
                "Cannot check macOS memory headroom; local model work was refused. "
                "Check that sysctl and vm_stat are available."
            ) from error

    def check(
        self, additional_bytes: int = 0, operation: str = "inference"
    ) -> ResourceSnapshot | None:
        """Check a fresh system snapshot before an additional peak allocation."""
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("additional_bytes must be a nonnegative integer")
        if not self.enabled:
            return None
        try:
            snapshot = self._snapshot()
        except ResourceLimitError:
            if self.enforcing:
                raise
            return None
        projected = snapshot.used_bytes + additional_bytes
        ceiling = int(snapshot.total_bytes * self.max_fraction)
        if projected > ceiling and not self.enforcing:
            if operation not in self._warned:
                self._warned.add(operation)
                warnings.warn(
                    f"{operation[:1].upper()}{operation[1:]} needs about "
                    f"{additional_bytes / 2**30:.1f} GiB of memory with about "
                    f"{snapshot.available_bytes / 2**30:.1f} GiB available, taking "
                    f"this Mac to about {projected / 2**30:.1f} of "
                    f"{snapshot.total_bytes / 2**30:.0f} GiB, past the "
                    f"{self.max_fraction:.0%} line. It will continue, but may be "
                    "slow; close other applications to free memory.",
                    MemoryHeadroomWarning,
                    stacklevel=2,
                )
        elif projected > ceiling:
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
        threads = max(1, math.floor((os.cpu_count() or 1) * self.max_fraction))
        current_threads = torch.get_num_threads()
        if current_threads > threads:
            torch.set_num_threads(threads)
        if device_type == "mps" and self.enforcing:
            assert snapshot is not None
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
        if not self.enabled or not self.enforcing:
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
