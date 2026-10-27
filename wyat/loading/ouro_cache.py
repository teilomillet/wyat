# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Question-owned, exact FP32 context storage with bounded staging buffers.

A layer read borrows the bank's staging tensors. Their contents are replaced by
its next read, so consumers must finish using a layer before requesting another.
The anonymous file is reclaimed when closed, including by the OS on process exit.
"""

from __future__ import annotations

import io
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import torch

from wyat.resources import ResourceGuard, ResourceLimitError


@dataclass(frozen=True, slots=True)
class _LayerRecord:
    offset: int
    shape: tuple[int, ...]
    tensor_bytes: int


class DiskKVBank:
    """Store exact FP32 K/V bytes; retain at most one CPU and one GPU pair.

    The supplied guard controls RAM admission and the disk-space reserve. Disk
    admission includes the remaining planned bytes rather than charging already
    written data twice. Disk space is not reserved against other processes;
    checks and I/O failures therefore remain necessary on every append.
    """

    def __init__(
        self,
        guard: ResourceGuard,
        expected_bytes: int,
        temp_dir: str | Path | None = None,
    ) -> None:
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError("expected_bytes must be a nonnegative integer")
        self.guard = guard
        self.expected_bytes = expected_bytes
        self._directory = Path(tempfile.gettempdir() if temp_dir is None else temp_dir)
        self._file: io.FileIO | None = None
        self._entries: list[_LayerRecord] = []
        self._written = 0
        self._cpu: tuple[torch.Tensor, torch.Tensor] | None = None
        self._gpu: tuple[torch.Tensor, torch.Tensor] | None = None
        self._check_disk(expected_bytes)
        try:
            self._file = tempfile.TemporaryFile(mode="w+b", buffering=0, dir=temp_dir)
            if sys.platform == "darwin":
                import fcntl

                # Darwin sys/fcntl.h: disable file-data caching for this fd.
                fcntl.fcntl(self._file.fileno(), 48, 1)
        except (OSError, ImportError, AttributeError) as error:
            self.close()
            raise ResourceLimitError(
                "Cannot open uncached temporary context storage"
            ) from error

    def __enter__(self) -> DiskKVBank:
        self._open_file()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._file is None or self._file.closed

    @property
    def resident_bytes(self) -> int:
        """Bytes owned by this bank's allocated CPU and GPU staging tensors."""
        return sum(
            tensor.numel() * tensor.element_size()
            for pair in (self._cpu, self._gpu)
            if pair is not None
            for tensor in pair
        )

    def __len__(self) -> int:
        return len(self._entries)

    def _open_file(self) -> io.FileIO:
        if self._file is None or self._file.closed:
            raise ValueError("context storage is closed")
        return self._file

    def _check_disk(self, remaining: int) -> None:
        try:
            usage = shutil.disk_usage(self._directory)
            if (
                usage.total <= 0
                or not 0 <= usage.free <= usage.total
                or not 0 <= usage.used <= usage.total
            ):
                raise ValueError("invalid filesystem capacity")
            # total-free also counts filesystem reservations and shared usage.
            occupied = max(usage.used, usage.total - usage.free)
            ceiling = math.floor(usage.total * self.guard.max_fraction)
            if remaining > usage.free or occupied + remaining > ceiling:
                raise ResourceLimitError(
                    "Temporary context storage exceeds the configured disk-space "
                    f"budget ({self.guard.max_fraction:.0%}); no context was spilled"
                )
        except (OSError, ValueError) as error:
            raise ResourceLimitError(
                "Cannot check temporary context disk capacity"
            ) from error

    @staticmethod
    def _validate(key: torch.Tensor, value: torch.Tensor) -> None:
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("context K/V must be FP32 tensors")
        if key.dtype != torch.float32 or value.dtype != torch.float32:
            raise ValueError("context K/V must be FP32 tensors")
        if key.shape != value.shape or key.ndim != 4 or any(n <= 0 for n in key.shape):
            raise ValueError(
                "context K/V must have identical nonempty four-dimensional shapes"
            )
        if key.device != value.device or key.device.type == "meta":
            raise ValueError("context K/V must share an allocated device")

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append one layer without retaining either source tensor."""
        self._validate(key, value)
        file = self._open_file()
        tensor_bytes = key.numel() * key.element_size()
        pair_bytes = 2 * tensor_bytes
        try:
            if self._written + pair_bytes > self.expected_bytes:
                raise ResourceLimitError(
                    "Context storage exceeded its admitted byte count"
                )
            self._check_disk(self.expected_bytes - self._written)
            self.guard.check(operation="spill context layer")
            file.seek(self._written)
            for tensor in (key, value):
                needs_copy = tensor.device.type != "cpu" or not tensor.is_contiguous()
                if needs_copy:
                    self.guard.check(tensor_bytes, operation="stage context spill")
                    host = tensor.detach().to(
                        device="cpu", memory_format=torch.contiguous_format, copy=True
                    )
                else:
                    host = tensor.detach()
                data = memoryview(host.numpy()).cast("B")
                try:
                    if file.write(data) != tensor_bytes:
                        raise ResourceLimitError("Incomplete temporary context write")
                finally:
                    data.release()
                    del host
            self._entries.append(
                _LayerRecord(self._written, tuple(key.shape), tensor_bytes)
            )
            self._written += pair_bytes
        except Exception as error:
            self.close()
            if not isinstance(error, OSError):
                raise
            raise ResourceLimitError(
                "Cannot write temporary context storage"
            ) from error

    def read_layer(
        self, index: int, device: str | torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow exact layer values; the next read may overwrite the buffers."""
        file = self._open_file()
        if type(index) is not int or not 0 <= index < len(self._entries):
            raise IndexError("context layer index is out of range")
        target = torch.device(device)
        if target.type not in {"cpu", "mps"} or target.index not in {None, 0}:
            raise ValueError("disk context reads support CPU or the Apple GPU")
        entry = self._entries[index]
        try:
            self.guard.check(operation="read context layer")
            if self._gpu is not None:
                torch.mps.synchronize()
            if self._cpu is None or tuple(self._cpu[0].shape) != entry.shape:
                self._cpu = None
                self._gpu = None
                self.guard.check(2 * entry.tensor_bytes, operation="read context layer")
                self._cpu = (
                    torch.empty(entry.shape, dtype=torch.float32, device="cpu"),
                    torch.empty(entry.shape, dtype=torch.float32, device="cpu"),
                )
            file.seek(entry.offset)
            for tensor in self._cpu:
                data = memoryview(tensor.numpy()).cast("B")
                try:
                    if file.readinto(data) != entry.tensor_bytes:
                        raise ResourceLimitError("Incomplete temporary context read")
                finally:
                    data.release()
            if target.type == "cpu":
                return self._cpu
            if self._gpu is None:
                self.guard.check(
                    2 * entry.tensor_bytes, operation="stage Apple GPU context"
                )
                self._gpu = (
                    torch.empty(entry.shape, dtype=torch.float32, device=target),
                    torch.empty(entry.shape, dtype=torch.float32, device=target),
                )
            for destination, source in zip(self._gpu, self._cpu, strict=True):
                destination.copy_(source, non_blocking=False)
            return self._gpu
        except Exception as error:
            self.close()
            if not isinstance(error, OSError):
                raise
            raise ResourceLimitError("Cannot read temporary context storage") from error

    def close(self) -> None:
        """Release owned tensors and close the anonymous file idempotently."""
        try:
            if self._gpu is not None:
                # A submitted Apple GPU copy may still read these tensors.
                synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
                if synchronize is not None:
                    try:
                        synchronize()
                    except RuntimeError:
                        pass
            if self._file is not None:
                self._file.close()
        finally:
            self._file = None
            self._cpu = None
            self._gpu = None
