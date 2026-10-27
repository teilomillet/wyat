# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Exact context spilling, bounded staging, disk admission and cleanup."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
from wyat.loading import ouro_cache as cache  # noqa: E402
from wyat.resources import ResourceGuard, ResourceLimitError  # noqa: E402


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> ResourceGuard:
    result = ResourceGuard()
    monkeypatch.setattr(result, "check", Mock())
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=2**30, used=0, free=2**30),
    )
    return result


def pair(offset: int = 0) -> tuple[Any, Any]:
    key = torch.arange(32, dtype=torch.float32).reshape(1, 1, 8, 4) + offset
    return key, key * -0.5


def test_bit_exact_roundtrip_and_source_lifetime(
    guard: ResourceGuard, tmp_path: Path
) -> None:
    key, value = pair()
    # Include non-finite payloads and negative zero: storage must not normalize.
    key.view(torch.int32).reshape(-1)[:3] = torch.tensor(
        [-2147483648, 2139095040, 2143289345], dtype=torch.int32
    )
    expected_key, expected_value = key.clone(), value.clone()
    with cache.DiskKVBank(guard, 256, tmp_path) as bank:
        bank.append(key, value)
        key.zero_()
        value.zero_()
        actual_key, actual_value = bank.read_layer(0, "cpu")
        assert torch.equal(actual_key.view(torch.int32), expected_key.view(torch.int32))
        assert torch.equal(
            actual_value.view(torch.int32), expected_value.view(torch.int32)
        )
        assert len(bank) == 1
        assert bank.resident_bytes == 256
        assert os.fstat(bank._open_file().fileno()).st_nlink == 0
        assert list(tmp_path.iterdir()) == []
    assert bank.closed and bank.resident_bytes == 0
    bank.close()


def test_reuses_one_cpu_pair_and_admits_before_allocation(
    guard: ResourceGuard, tmp_path: Path
) -> None:
    with cache.DiskKVBank(guard, 512, tmp_path) as bank:
        bank.append(*pair())
        bank.append(*pair(123))
        first = bank.read_layer(0, "cpu")
        pointers = tuple(t.data_ptr() for t in first)
        second = bank.read_layer(1, "cpu")
        assert tuple(t.data_ptr() for t in second) == pointers
        assert torch.equal(second[0], pair(123)[0])
        assert bank.resident_bytes == 256
        positive = [call.args[0] for call in guard.check.call_args_list if call.args]
        assert positive == [256]


def test_noncontiguous_source_uses_exact_logical_order(
    guard: ResourceGuard, tmp_path: Path
) -> None:
    key, value = (t.transpose(2, 3) for t in pair())
    assert not key.is_contiguous()
    with cache.DiskKVBank(guard, 256, tmp_path) as bank:
        bank.append(key, value)
        actual = bank.read_layer(0, "cpu")
        assert torch.equal(actual[0], key)
        assert torch.equal(actual[1], value)


def test_disk_admission_includes_reserved_space_before_opening(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1000, used=1, free=300),
    )
    create = Mock(side_effect=AssertionError("file opened before admission"))
    monkeypatch.setattr(cache.tempfile, "TemporaryFile", create)
    with pytest.raises(ResourceLimitError, match="disk-space budget"):
        cache.DiskKVBank(guard, 151, tmp_path)
    create.assert_not_called()


def test_disk_budget_does_not_charge_written_bytes_twice(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        Mock(
            side_effect=[
                SimpleNamespace(total=1000, used=200, free=800),
                SimpleNamespace(total=1000, used=200, free=800),
                SimpleNamespace(total=1000, used=456, free=544),
            ]
        ),
    )
    with cache.DiskKVBank(guard, 512, tmp_path) as bank:
        bank.append(*pair())
        bank.append(*pair(1))
        assert len(bank) == 2


def test_disk_pressure_during_append_closes_storage(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = cache.DiskKVBank(guard, 256, tmp_path)
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1000, used=900, free=100),
    )
    with pytest.raises(ResourceLimitError):
        bank.append(*pair())
    assert bank.closed


def test_ram_refusal_precedes_staging_allocation(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = cache.DiskKVBank(guard, 256, tmp_path)
    bank.append(*pair())

    def admit(additional_bytes: int = 0, **kwargs: Any) -> None:
        if additional_bytes:
            raise ResourceLimitError("no staging headroom")

    monkeypatch.setattr(guard, "check", admit)
    monkeypatch.setattr(torch, "empty", Mock(side_effect=AssertionError("allocated")))
    with pytest.raises(ResourceLimitError, match="no staging headroom"):
        bank.read_layer(0, "cpu")
    assert bank.closed


@pytest.mark.parametrize("operation", ["write", "readinto"])
def test_short_io_fails_closed(
    guard: ResourceGuard, tmp_path: Path, operation: str
) -> None:
    bank = cache.DiskKVBank(guard, 256, tmp_path)
    if operation == "readinto":
        bank.append(*pair())
    original = bank._open_file()

    class ShortIO:
        def __getattr__(self, name: str) -> Any:
            return getattr(original, name)

        def write(self, data: Any) -> int:
            original.write(data)
            return len(data) - 1

        def readinto(self, data: Any) -> int:
            original.readinto(data)
            return len(data) - 1

    bank._file = ShortIO()  # type: ignore[assignment]
    with pytest.raises(ResourceLimitError, match="Incomplete"):
        if operation == "write":
            bank.append(*pair())
        else:
            bank.read_layer(0, "cpu")
    assert bank.closed and original.closed


def test_context_failure_closes_file(guard: ResourceGuard, tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="caller failed"):
        with cache.DiskKVBank(guard, 256, tmp_path) as bank:
            bank.append(*pair())
            raise RuntimeError("caller failed")
    assert bank.closed
    with pytest.raises(ValueError, match="closed"):
        bank.read_layer(0, "cpu")


def test_over_budget_append_and_invalid_inputs(
    guard: ResourceGuard, tmp_path: Path
) -> None:
    with cache.DiskKVBank(guard, 255, tmp_path) as bank:
        with pytest.raises(ValueError, match="FP32"):
            bank.append(*(t.half() for t in pair()))
        with pytest.raises(ValueError, match="shapes"):
            bank.append(pair()[0], pair()[1][:, :, :1])
        with pytest.raises(ResourceLimitError, match="byte count"):
            bank.append(*pair())
        assert bank.closed


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple GPU unavailable"
)
def test_mps_exact_spill_read_and_buffer_reuse(
    guard: ResourceGuard, tmp_path: Path
) -> None:
    original = pair()
    with cache.DiskKVBank(guard, 512, tmp_path) as bank:
        bank.append(*(t.to("mps") for t in original))
        bank.append(*(t.to("mps") for t in pair(99)))
        first = bank.read_layer(0, "mps")
        assert torch.equal(first[0].cpu(), original[0])
        pointers = tuple(t.data_ptr() for t in first)
        second = bank.read_layer(1, "mps")
        assert tuple(t.data_ptr() for t in second) == pointers
        assert torch.equal(second[1].cpu(), pair(99)[1])
        assert bank.resident_bytes == 512


@pytest.mark.skipif(
    cache.sys.platform != "darwin", reason="Darwin uncached-file policy"
)
def test_uncached_file_failure_closes_the_descriptor(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    opened = []
    create = cache.tempfile.TemporaryFile

    def tracked_file(*args: Any, **kwargs: Any) -> Any:
        file = create(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(cache.tempfile, "TemporaryFile", tracked_file)
    monkeypatch.setattr(
        fcntl, "fcntl", Mock(side_effect=OSError("no uncached support"))
    )
    with pytest.raises(ResourceLimitError, match="uncached"):
        cache.DiskKVBank(guard, 256, tmp_path)
    assert len(opened) == 1 and opened[0].closed
    assert list(tmp_path.iterdir()) == []


def test_custom_disk_budget_is_not_silently_raised(
    guard: ResourceGuard, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cache.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1000, used=400, free=600),
    )
    guard.max_fraction = 0.5
    with pytest.raises(ResourceLimitError, match="50%"):
        cache.DiskKVBank(guard, 101, tmp_path)
