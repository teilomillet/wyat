"""Pure projection gates and fake failure boundaries; no real model or timing."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.evidence import FrozenSources
from benchmarks.analysis.mlx.mixed import run
from benchmarks.analysis.mlx.mixed import study as runner
from benchmarks.analysis.mlx.mixed.capture import (
    ProjectionOwner,
    exact_bfloat_bits,
    observe_projection,
    projection_quality,
)
from benchmarks.analysis.mlx.mixed.study import ARMS, ORDERS, performance
from wyat.benchmark import BenchmarkCycle


@given(st.lists(st.integers(0, 0x7F7F), min_size=1, max_size=30))
def test_exact_storage_preserves_arbitrary_finite_bfloat_bits(bits: list[int]) -> None:
    values = (np.array(bits, dtype=np.uint32) << 16).view(np.float32).reshape(1, -1)
    assert exact_bfloat_bits(values)
    changed = values.view(np.uint32).copy()
    changed[0, 0] |= 1
    assert not exact_bfloat_bits(changed.view(np.float32))


def test_signed_zero_storage_and_nonfinite_rejection() -> None:
    assert exact_bfloat_bits(np.array([[0.0, -0.0]], dtype=np.float32))
    assert not exact_bfloat_bits(np.array([[np.inf]], dtype=np.float32))
    assert not exact_bfloat_bits(np.array([[np.nan]], dtype=np.float32))


@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
def test_elementwise_projection_gate_matches_declared_formula(value: float) -> None:
    reference = np.array([[value]], dtype=np.float32)
    assert projection_quality(reference, reference.copy())["passed"] is True
    candidate = reference + np.float32(0.1)
    report = projection_quality(reference, candidate)
    assert report["passed"] is False
    assert report["failing_elements"] == 1


def test_nonfinite_and_geometry_are_not_passes() -> None:
    reference = np.ones((1, 2), dtype=np.float32)
    candidate = np.array([[np.nan, np.inf]], dtype=np.float32)
    report = projection_quality(reference, candidate)
    assert report["passed"] is False and report["max_abs_error"] is None
    with pytest.raises(ValueError):
        projection_quality(reference, np.ones((2, 1), dtype=np.float32))


def blocks(scale: float = 1.1) -> list[dict[str, object]]:
    return [
        {
            "order": list(order),
            "dispatch_ns": {
                arm: [int(10000 * (scale if arm == "maintained" else 1))] * 10
                for arm in ARMS
            },
        }
        for order in ORDERS
    ]


def test_performance_requires_original_both_gates_and_every_permutation() -> None:
    assert performance(blocks())["passed"] is True
    assert performance(blocks(1.04))["passed"] is False
    with pytest.raises(ValueError):
        performance(blocks()[:-1])
    wrong = blocks()
    wrong[1]["order"] = wrong[0]["order"]
    with pytest.raises(ValueError):
        performance(wrong)
    wrong = blocks()
    cast(dict[str, list[int]], wrong[0]["dispatch_ns"])["maintained"][0] = 0
    with pytest.raises(ValueError):
        performance(wrong)


def test_observer_restores_exact_callable_on_error() -> None:
    class Owner:
        gate_proj: Callable[[object], object] = staticmethod(lambda value: value)

    owner = Owner()
    original = owner.gate_proj
    with (
        pytest.raises(RuntimeError),
        observe_projection(cast(ProjectionOwner, owner)),
    ):
        raise RuntimeError("capture failed")
    assert owner.gate_proj is original


@pytest.mark.parametrize("error", (False, True))
def test_real_module_restores_registered_projection_identity(error: bool) -> None:
    from mlx.nn.layers.base import Module
    from mlx.nn.layers.linear import Linear

    # MLX's base type is unavailable on CPU-only installations.
    class Owner(Module):  # type: ignore[misc, unused-ignore]
        def __init__(self) -> None:
            super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
            self.gate_proj = Linear(2, 2, bias=False)

    owner = Owner()
    original = owner.gate_proj
    weight = original.weight
    try:
        with observe_projection(cast(ProjectionOwner, owner)):
            assert owner.gate_proj is not original
            if error:
                raise RuntimeError("captured error")
    except RuntimeError:
        assert error
    assert owner.gate_proj is original
    assert owner.gate_proj.weight is weight
    assert owner["gate_proj"] is original


@pytest.mark.parametrize("initial", (None, "1"))
def test_parent_environment_restored_on_launcher_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, initial: str | None
) -> None:
    if initial is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", initial)
    monkeypatch.setattr(
        sys, "argv", ["run", "model", "--output", str(tmp_path / "result.json")]
    )

    def fail(*args: object, **kwargs: object) -> int:
        assert os.environ["MLX_ENABLE_TF32"] == "0"
        assert kwargs["timeout"] == 180
        raise RuntimeError("launch failed")

    monkeypatch.setattr(run, "launch", fail)
    with pytest.raises(RuntimeError):
        run.main()
    assert os.environ.get("MLX_ENABLE_TF32") == initial


@pytest.mark.parametrize(
    "failure",
    (
        "capture-error",
        "capture-quality",
        "kernel-error",
        "final-write",
        "cache-restore",
    ),
)
def test_worker_failure_retains_receipt_and_restores_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    from benchmarks.analysis.mlx.mixed import kernel

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    output = tmp_path / "assay.json"
    restored: list[bool] = []

    @contextmanager
    def bounded(limit: int) -> Iterator[int]:
        assert limit == 512 << 20
        try:
            yield 123
        finally:
            restored.append(True)
            if failure == "cache-restore":
                raise RuntimeError("restore failed")

    monkeypatch.setattr(runner, "bounded_cache", bounded)
    monkeypatch.setattr(runner, "memory", lambda: {"active": 0})
    monkeypatch.setattr(mx, "device_info", lambda device: {"device_name": "fake"})
    monkeypatch.setattr(mx, "stream", lambda device: nullcontext())
    monkeypatch.setattr(
        runner, "freeze_sources", lambda *args: FrozenSources((), "fake", "f" * 64)
    )
    monkeypatch.setattr(
        runner, "admit", lambda model: ((), BenchmarkCycle(0, (), 0, 0), (), {})
    )

    def capture(
        *args: object,
        **kwargs: object,
    ) -> tuple[
        dict[str, np.ndarray[tuple[int, int], np.dtype[np.float32]]], dict[str, object]
    ]:
        if failure == "capture-error":
            raise ValueError("declared capture failed")
        arrays = {"weight": np.ones((2, 2), dtype=np.float32)}
        arrays.update(
            {name: np.ones((1, 2), dtype=np.float32) for name, _, _ in runner.SHAPES}
        )
        receipts = {
            name: {"cpu_quality": {"passed": failure == "kernel-error"}}
            for name, _, _ in runner.SHAPES
        }
        return arrays, cast(dict[str, object], receipts)

    def fail_kernel() -> None:
        raise RuntimeError("declared kernel failed")

    monkeypatch.setattr(runner, "capture_all", capture)
    monkeypatch.setattr(kernel, "ProjectionKernel", fail_kernel)
    if failure == "final-write":
        original_save = runner.save
        failed = False

        def save_once(path: Path, payload: dict[str, object]) -> None:
            nonlocal failed
            if payload.get("completed") is True and not failed:
                failed = True
                raise OSError("late write failed")
            original_save(path, payload)

        monkeypatch.setattr(runner, "save", save_once)
    assert not runner.study(tmp_path / "model", output)
    receipt = json.loads(output.read_text())
    assert receipt["accepted"] is False
    assert receipt["completed"] is (failure == "capture-quality")
    assert restored == [True]
    assert receipt["cache_cap_restoration_completed"] is (failure != "cache-restore")
    if failure == "final-write":
        assert "late write failed" in receipt["final_write_error"]
    if failure == "cache-restore":
        assert "restore failed" in receipt["cache_cap_restoration_error"]
    if failure != "capture-error":
        assert output.with_suffix(".tensors.safetensors").exists()


@pytest.mark.parametrize("drift", (False, True))
def test_full_fake_protocol_visits_all_shapes_orders_and_checks_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: bool
) -> None:
    from benchmarks.analysis.mlx.mixed import kernel

    class Array:
        def __init__(
            self, values: np.ndarray[tuple[int, int], np.dtype[np.float32]]
        ) -> None:
            self.values = values

        @property
        def T(self) -> Array:
            return Array(self.values.T)

        def astype(self, dtype: object) -> Array:
            return self

        def __matmul__(self, other: Array) -> Array:
            return Array(self.values @ other.values)

    class Prepared:
        def __init__(self, x: Array, weight: Array) -> None:
            self.x, self.weight = x, weight
            self.calls = 0

        def run(self) -> Array:
            self.calls += 1
            result = self.x @ self.weight.T
            if drift and self.calls == 3:
                result.values[0, 0] += 1
            return result

    class Kernel:
        def prepare(self, x: Array, weight: Array) -> Prepared:
            return Prepared(x, weight)

    @contextmanager
    def bounded(limit: int) -> Iterator[int]:
        yield 123

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(runner, "bounded_cache", bounded)
    monkeypatch.setattr(runner, "memory", lambda: {"active": 0})
    monkeypatch.setattr(mx, "device_info", lambda device: {"device_name": "fake"})
    monkeypatch.setattr(mx, "stream", lambda device: nullcontext())
    monkeypatch.setattr(mx, "array", lambda value, **kwargs: Array(value.copy()))
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "detached_float32", lambda value: value.values.copy())
    monkeypatch.setattr(
        runner,
        "output_digest",
        lambda value: __import__("hashlib").sha256(value.values.tobytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner, "freeze_sources", lambda *args: FrozenSources((), "fake", "f" * 64)
    )
    monkeypatch.setattr(
        runner, "admit", lambda model: ((), BenchmarkCycle(0, (), 0, 0), (), {})
    )
    monkeypatch.setattr(kernel, "ProjectionKernel", Kernel)
    arrays = {"weight": np.ones((2, 2), dtype=np.float32)}
    arrays.update(
        {name: np.ones((1, 2), dtype=np.float32) for name, _, _ in runner.SHAPES}
    )
    captures = {name: {"cpu_quality": {"passed": True}} for name, _, _ in runner.SHAPES}
    monkeypatch.setattr(
        runner, "capture_all", lambda *args, **kwargs: (arrays, captures)
    )
    output = tmp_path / "fake.json"
    runner.study(tmp_path / "model", output)
    receipt = json.loads(output.read_text())
    assert receipt["cache_cap_restoration_completed"] is True
    assert receipt["projection_quality_passed"] is True
    assert output.with_suffix(".outputs.tensors.safetensors").exists()
    assert receipt["completed"] is not drift
    if drift:
        assert receipt["accepted"] is False
        assert (
            receipt["shapes"]["prefix-b1"]["blocks"][0]["outputs_reproduce_cold"]
            is False
        )
    else:
        assert set(receipt["shapes"]) == {name for name, _, _ in runner.SHAPES}
        for result in receipt["shapes"].values():
            assert len(result["blocks"]) == 6
            assert tuple(tuple(block["order"]) for block in result["blocks"]) == ORDERS
            assert all(block["outputs_reproduce_cold"] for block in result["blocks"])
            assert (
                sum(
                    len(times)
                    for block in result["blocks"]
                    for times in block["dispatch_ns"].values()
                )
                == 180
            )
