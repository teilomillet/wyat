# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from wyat import resources
from wyat.resources import (
    MemoryHeadroomWarning,
    ResourceGuard,
    ResourceLimitError,
    ResourceSnapshot,
)


@pytest.fixture
def mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")


@pytest.mark.parametrize(
    "fraction", [True, 0, -0.1, 1.01, float("nan"), float("inf"), "85%"]
)
def test_invalid_fraction_refused(fraction: object) -> None:
    with pytest.raises(ValueError, match="max_resource_fraction"):
        ResourceGuard(fraction)  # type: ignore[arg-type]


def test_non_macos_does_not_probe_or_configure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    probe = Mock(side_effect=AssertionError("must not probe"))
    monkeypatch.setattr(subprocess, "run", probe)
    guard = ResourceGuard(0.90)
    assert not guard.enabled
    assert guard.check(10**30) is None
    guard.configure_torch(object(), "cpu")
    with guard.activate():
        resources.checkpoint()
    probe.assert_not_called()


def test_default_and_custom_memory_admission(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 300)
    )
    explicit = ResourceGuard(0.90)
    assert explicit.check(200) == ResourceSnapshot(1000, 300)
    with pytest.raises(ResourceLimitError, match="90%"):
        explicit.check(201, "context prefill")
    assert ResourceGuard(0.85).check(150) is not None
    with pytest.raises(ResourceLimitError, match="85%"):
        ResourceGuard(0.85).check(151)
    with pytest.raises(ResourceLimitError):
        ResourceGuard(0.6).check()


def test_default_warns_once_per_operation_and_continues(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000 << 30, 300 << 30)
    )
    default = ResourceGuard()
    assert default.max_fraction == 0.90
    assert not default.enforcing
    with pytest.warns(
        MemoryHeadroomWarning,
        match="needs about 201.0 GiB of memory with about 300.0 GiB",
    ) as caught:
        assert default.check(201 << 30, "model loading") is not None
        default.check(300 << 30, "model loading")
    assert len(caught) == 1
    assert "past the 90% line" in str(caught[0].message)
    with pytest.warns(MemoryHeadroomWarning, match="Inference needs"):
        default.check(201 << 30)


def test_default_tolerates_unavailable_telemetry(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=OSError("missing")))
    assert ResourceGuard().check(1 << 40) is None


def test_default_leaves_apple_gpu_allocators_unbounded(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 900)
    )
    torch = SimpleNamespace(
        get_num_threads=Mock(return_value=1),
        set_num_threads=Mock(),
        mps=SimpleNamespace(set_per_process_memory_fraction=Mock()),
    )
    ResourceGuard().configure_torch(torch, "mps")
    torch.mps.set_per_process_memory_fraction.assert_not_called()
    mx = SimpleNamespace(set_memory_limit=Mock(), set_cache_limit=Mock())
    ResourceGuard().configure_mlx(mx)
    mx.set_memory_limit.assert_not_called()


@pytest.mark.parametrize("additional", [-1, 1.5, True])
def test_invalid_estimates_refused(additional: object) -> None:
    with pytest.raises(ValueError, match="additional_bytes"):
        ResourceGuard(0.90).check(additional)  # type: ignore[arg-type]


def test_snapshot_uses_bounded_commands_and_caches_only_total(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Mock(
        side_effect=[
            SimpleNamespace(stdout="10000\n"),
            SimpleNamespace(
                stdout=(
                    "Mach Virtual Memory Statistics: (page size of 100 bytes)\n"
                    "Pages free: 10.\nPages inactive: 20.\n"
                    "Pages stored in compressor: 999.\nPages purgeable: 999.\n"
                    "Pages speculative: 999.\n"
                )
            ),
            SimpleNamespace(
                stdout=(
                    "Mach Virtual Memory Statistics: (page size of 100 bytes)\n"
                    "Pages free: 5.\nPages inactive: 15.\n"
                )
            ),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    guard = ResourceGuard(0.90)
    snapshot = guard.check()
    assert snapshot == ResourceSnapshot(10000, 3000)
    assert snapshot.used_bytes == 7000
    assert guard.check() == ResourceSnapshot(10000, 2000)
    assert run.call_count == 3
    for call in run.call_args_list:
        assert call.kwargs == dict(
            check=True, capture_output=True, text=True, timeout=1.0
        )
    assert run.call_args_list[0].args[0] == ["/usr/sbin/sysctl", "-n", "hw.memsize"]
    assert run.call_args_list[1].args[0] == ["/usr/bin/vm_stat"]


@pytest.mark.parametrize(
    "output",
    [
        "missing",
        "System-wide memory free percentage: 99%",
        "bad total",
        "page size of 0 bytes\nPages free: 10.\nPages inactive: 20.",
        "page size of 100 bytes\nPages free: 10.",
        "page size of 100 bytes\nPages free: -10.\nPages inactive: 20.",
    ],
)
def test_invalid_telemetry_fails_closed(
    mac: None, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    run = Mock(
        side_effect=[
            SimpleNamespace(stdout="bad" if output == "bad total" else "1000"),
            SimpleNamespace(stdout=output),
        ]
    )
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ResourceLimitError, match="Cannot check"):
        ResourceGuard(0.90).check()


@pytest.mark.parametrize(
    "error", [OSError("missing"), subprocess.TimeoutExpired("vm_stat", 1.0)]
)
def test_unavailable_telemetry_fails_closed(
    mac: None, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
    with pytest.raises(ResourceLimitError, match="Cannot check"):
        ResourceGuard(0.90).check()


def test_torch_threads_and_mps_limit(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 900)
    )
    torch = SimpleNamespace(
        get_num_threads=Mock(return_value=10),
        set_num_threads=Mock(),
        mps=SimpleNamespace(
            recommended_max_memory=Mock(return_value=800),
            set_per_process_memory_fraction=Mock(),
        ),
    )
    guard = ResourceGuard(0.90)
    guard.configure_torch(torch, "mps")
    torch.set_num_threads.assert_called_once_with(9)
    torch.mps.set_per_process_memory_fraction.assert_called_once_with(0.90)
    torch.get_num_threads.return_value = 4
    torch.set_num_threads.reset_mock()
    guard.configure_torch(torch, "cpu")
    torch.set_num_threads.assert_not_called()
    torch.mps.set_per_process_memory_fraction.reset_mock()
    torch.mps.recommended_max_memory.return_value = 2000
    guard.configure_torch(torch, "mps")
    torch.mps.set_per_process_memory_fraction.assert_called_once_with(0.45)


def test_missing_mps_limit_support_refuses(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 900)
    )
    torch = SimpleNamespace(get_num_threads=lambda: 1, mps=object())
    with pytest.raises(ResourceLimitError, match="Cannot configure"):
        ResourceGuard(0.90).configure_torch(torch, "mps")


def test_checkpoint_is_throttled_and_nested_context_restores_guard(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = ResourceGuard(0.90), ResourceGuard(0.7)
    check_first, check_second = Mock(), Mock()
    monkeypatch.setattr(first, "check", check_first)
    monkeypatch.setattr(second, "check", check_second)
    ticks = iter([10.0, 10.1, 10.2, 11.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    with first.activate():
        resources.checkpoint()
        resources.checkpoint()
        with second.activate():
            resources.checkpoint()
        resources.checkpoint()
    resources.checkpoint()
    assert check_first.call_count == 2
    check_second.assert_called_once_with()


def test_mlx_limits_do_not_raise_existing_lower_limit(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 900)
    )
    mx = SimpleNamespace(
        get_memory_limit=Mock(return_value=2000),
        set_memory_limit=Mock(),
        set_cache_limit=Mock(),
    )
    guard = ResourceGuard(0.90)
    guard.configure_mlx(mx)
    mx.set_memory_limit.assert_called_once_with(900)
    mx.set_cache_limit.assert_called_once_with(0)
    mx.get_memory_limit.return_value = 500
    mx.set_memory_limit.reset_mock()
    guard.configure_mlx(mx)
    mx.set_memory_limit.assert_called_once_with(500)


def test_missing_mlx_limit_support_refuses(
    mac: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ResourceGuard, "_snapshot", lambda self: ResourceSnapshot(1000, 900)
    )
    with pytest.raises(ResourceLimitError, match="Cannot configure the MLX"):
        ResourceGuard(0.90).configure_mlx(object())


def test_current_guard_restored_after_exception() -> None:
    guard = ResourceGuard(0.90)
    assert resources.current_guard() is None
    with pytest.raises(RuntimeError), guard.activate():
        assert resources.current_guard() is guard
        raise RuntimeError("operation failed")
    assert resources.current_guard() is None


def test_mac_activations_serialize_distinct_guards(mac: None) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    started, entered, release = Event(), Event(), Event()
    first, second = ResourceGuard(0.90), ResourceGuard(0.90)

    def work() -> None:
        started.set()
        with second.activate():
            entered.set()
            assert release.wait(2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with first.activate():
            pending = pool.submit(work)
            assert started.wait(2)
            assert not entered.wait(0.02)
        release.set()
        pending.result(timeout=2)
    assert entered.is_set()
