"""Frozen-history admission and fake unroll gates; no models or device work."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarks.analysis.mlx.unroll import run
from wyat.wire import JsonObject

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.evidence import FrozenSources
from benchmarks.analysis.mlx.mixed import study as original_runner
from benchmarks.analysis.mlx.mixed.study import SHAPES
from benchmarks.analysis.mlx.tile.admit import Admission
from benchmarks.analysis.mlx.tile.study import Array, Operation
from benchmarks.analysis.mlx.unroll import study as runner


def blocks(ratios: list[int]) -> list[dict[str, object]]:
    return [
        {
            "order": list(order),
            "dispatch_ns": {
                runner.ARMS[0]: [ratio] * 10,
                runner.ARMS[1]: [20000] * 10,
                runner.ARMS[2]: [10000] * 10,
            },
        }
        for order, ratio in zip(runner.ORDERS, ratios, strict=True)
    ]


@given(st.lists(st.integers(9000, 20000), min_size=6, max_size=6))
def test_performance_keeps_original_gates_and_true_comparator(
    values: list[int],
) -> None:
    actual = runner.comparison(blocks(values), runner.ARMS[0])
    assert actual["passed"] is (
        cast(float, actual["paired_median"]) >= 1.05
        and cast(float, actual["paired_p05"]) > 1
    )
    separate = runner.comparison(blocks(values), runner.ARMS[1])
    assert separate["paired_median"] == 2 and "passed" not in separate


@pytest.mark.parametrize("fault", ["missing", "order", "zero", "bool", "count", "arm"])
def test_bad_raw_timings_are_not_a_pass(fault: str) -> None:
    data = blocks([11000] * 6)
    if fault == "missing":
        data.pop()
    elif fault == "order":
        data[1]["order"] = data[0]["order"]
    else:
        times = cast(dict[str, list[int]], data[0]["dispatch_ns"])
        if fault == "count":
            times[runner.ARMS[0]].pop()
        elif fault == "arm":
            del times[runner.ARMS[1]]
        else:
            times[runner.ARMS[0]][0] = 0 if fault == "zero" else True
    with pytest.raises(ValueError):
        runner.comparison(data, runner.ARMS[0])


def test_parent_import_never_loads_inference_frameworks() -> None:
    script = (
        "import sys; import benchmarks.analysis.mlx.unroll.run; "
        "assert not any(x in sys.modules for x in "
        "('torch','mlx','mlx.core','mlx_lm'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("fail", [False, True])
def test_parent_restores_tf32_even_when_launch_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    monkeypatch.setenv("MLX_ENABLE_TF32", "prior")
    monkeypatch.setattr(
        sys, "argv", ["run", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )

    def launcher(*args: object, **kwargs: object) -> int:
        assert os.environ["MLX_ENABLE_TF32"] == "0" and kwargs["timeout"] == 180
        if fail:
            raise OSError("launch failed")
        return 1

    monkeypatch.setattr(run, "launch", launcher)
    with pytest.raises(OSError if fail else SystemExit):
        run.main()
    assert os.environ["MLX_ENABLE_TF32"] == "prior"


def test_changed64_receipt_prevents_checkpoint_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "main.json"
    path.write_text("changed")
    monkeypatch.setattr(run, "PINNED", {path: "0" * 64})

    def forbidden(*args: object) -> None:
        raise AssertionError("Checkpoint admission reached before64 history gate")

    monkeypatch.setattr(run, "admit", forbidden)
    with pytest.raises(ValueError, match="evidence changed"):
        run.context(tmp_path)


def test_reopened64_tensors_are_hashed_before_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "cold"
    path.write_bytes(b"changed")
    admitted = run.Context(
        Admission({}, path, path, {}), {"cold_outputs": {"sha256": "0" * 64}}, path, {}
    )

    def forbidden(raw: bytes) -> None:
        raise AssertionError("Parser consumed changed bytes")

    monkeypatch.setattr(runner, "load", forbidden)
    with pytest.raises(ValueError, match="before consumption"):
        runner.previous_outputs(admitted)


@pytest.mark.parametrize("drift", ["none", "warm", "timed", "write"])
def test_measure_retains_all_observations_or_explicit_partial_failure(
    monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "memory", lambda: {"active": 0})
    monkeypatch.setattr(runner, "output_digest", lambda value: str(value))
    clock = itertools.count(1, 10)
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(clock))

    def operation(arm: str) -> Operation:
        calls = 0

        def dispatch() -> mx.array:
            nonlocal calls
            calls += 1
            changed = arm == runner.ARMS[2] and (
                (drift == "warm" and calls == 1) or (drift == "timed" and calls == 3)
            )
            return cast(mx.array, "changed" if changed else arm)

        return dispatch

    calls = {arm: operation(arm) for arm in runner.ARMS}
    result: dict[str, object] = {}
    writes = 0

    def progress() -> None:
        nonlocal writes
        writes += 1
        if drift == "write" and writes == 2:
            raise OSError("arm write failed")

    if drift != "none":
        with pytest.raises(OSError if drift == "write" else ValueError):
            runner.measure(calls, {arm: arm for arm in runner.ARMS}, result, progress)
        assert "performance" not in result
        if drift == "warm":
            assert "blocks" not in result
        elif drift == "write":
            assert (
                cast(list[dict[str, object]], result["blocks"])[0]["completed"] is False
            )
        return
    runner.measure(calls, {arm: arm for arm in runner.ARMS}, result, progress)
    assert writes == 26
    rows = cast(list[dict[str, object]], result["blocks"])
    assert len(rows) == 6 and all(row["outputs_reproduce_cold"] is True for row in rows)
    assert (
        sum(
            len(hashes)
            for row in rows
            for hashes in cast(dict[str, list[str]], row["output_sha256"]).values()
        )
        == 180
    )


@pytest.mark.parametrize("failure", ["numerical", "timing", "cache", "final-write"])
def test_outer_trial_preserves_negatives_and_resource_restoration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    from benchmarks.analysis.mlx.unroll import kernel

    class FakeArray:
        def __init__(self, value: Array) -> None:
            self.value = value

        @property
        def T(self) -> FakeArray:
            return FakeArray(self.value.T)

        def astype(self, dtype: object) -> FakeArray:
            return self

        def __matmul__(self, other: FakeArray) -> FakeArray:
            return FakeArray(self.value @ other.value)

    class Prepared:
        def __init__(self, x: FakeArray, w: FakeArray) -> None:
            self.x, self.w = x, w

        def run(self) -> FakeArray:
            return self.x @ self.w.T

    class Kernel:
        def __init__(self, *args: object) -> None:
            pass

        def prepare(self, x: FakeArray, w: FakeArray) -> Prepared:
            return Prepared(x, w)

    restored: list[bool] = []

    @contextmanager
    def bounded(limit: int) -> Iterator[int]:
        assert limit == 512 << 20
        try:
            yield 123
        finally:
            restored.append(True)
            if failure == "cache":
                raise OSError("restoration failed")

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(original_runner, "bounded_cache", bounded)
    monkeypatch.setattr(mx, "device_info", lambda *args: {"name": "fake"})
    monkeypatch.setattr(mx, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(mx, "array", lambda value, **kwargs: FakeArray(value.copy()))
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "detached_float32", lambda value: value.value.copy())
    for owner, name in (
        (runner, "OriginalKernel"),
        (runner, "FrozenKernel"),
        (kernel, "ProjectionKernel"),
    ):
        monkeypatch.setattr(owner, name, Kernel)
    monkeypatch.setattr(
        runner, "freeze_sources", lambda *args: FrozenSources((), "fake", "f" * 64)
    )
    inputs = {"weight": np.ones((2, 2), dtype=np.float32)}
    inputs.update({name: np.ones((1, 2), dtype=np.float32) for name, _, _ in SHAPES})
    old = {
        name + "-" + arm: np.full((1, 2), 2, dtype=np.float32)
        for name, _, _ in SHAPES
        for arm in runner.ARMS[:2]
    }
    monkeypatch.setattr(runner, "read_arrays", lambda *args: (inputs, {}))
    monkeypatch.setattr(runner, "previous_outputs", lambda *args: old)
    path = tmp_path / "inputs"
    path.write_bytes(b"fake")
    prior: JsonObject = {
        "cold_outputs": {},
        "model_id": "fake",
        "dataset_sha256": "fake",
        "reference_sha256": "fake",
        "input_sha256": "fake",
    }
    admitted = run.Context(Admission({}, path, path, {}), prior, path, {})
    original_screen = runner.screen

    def screen(
        calls: dict[str, Operation], original: dict[str, Array], name: str
    ) -> tuple[dict[str, object], dict[str, Array]]:
        result, values = original_screen(calls, original, name)
        if failure == "numerical":
            result["passed"] = False
        return result, values

    monkeypatch.setattr(runner, "screen", screen)
    count = 0

    def measure(
        calls: dict[str, Operation],
        cold: dict[str, str],
        result: dict[str, object],
        progress: object,
    ) -> None:
        nonlocal count
        count += 1
        if failure == "timing":
            raise ValueError("timing failed")
        result["performance"] = {"passed": True}

    monkeypatch.setattr(runner, "measure", measure)
    failed_write = False

    def save(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed_write
        if (
            failure == "final-write"
            and payload.get("accepted") is True
            and not failed_write
        ):
            failed_write = True
            raise OSError("late write failed")
        original_runner.save(path, payload)

    monkeypatch.setattr(runner, "save", save)
    output = tmp_path / "trial.json"
    assert not runner.study(admitted, output)
    receipt = json.loads(output.read_text())
    assert receipt["accepted"] is False and restored == [True]
    assert receipt["completed"] is (failure == "numerical")
    assert receipt["cache_cap_restoration_completed"] is (failure != "cache")
    assert count == (0 if failure == "numerical" else 1 if failure == "timing" else 4)
    assert output.with_suffix(".outputs.tensors.safetensors").exists()
    if failure == "final-write":
        assert "late write failed" in receipt["final_write_error"]


@pytest.mark.parametrize("fault", ["history", "storage", "nonfinite"])
def test_cold_controls_reject_bits_or_nonfinite_even_before_timing(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "detached_float32", lambda value: value)
    base = np.ones((1, 1), dtype=np.float32)
    values = {arm: base.copy() for arm in (*runner.ARMS, runner.CONTROL)}
    old = {"shape-" + arm: base for arm in runner.ARMS[:2]}
    changed = runner.ARMS[1] if fault == "history" else runner.CONTROL
    values[changed][0, 0] = (
        np.nan if fault == "nonfinite" else np.nextafter(np.float32(1), np.float32(2))
    )

    def operation(arm: str) -> Operation:
        def dispatch() -> mx.array:
            return cast(mx.array, values[arm])

        return dispatch

    result, _ = runner.screen({arm: operation(arm) for arm in values}, old, "shape")
    assert result["passed"] is False
    if fault == "history":
        assert result["original_cold_outputs_reproduced"] is False
    elif fault == "storage":
        assert result["custom_storage_bitwise_equal"] is False


def test_even_documentation_in123_source_inventory_requires_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    source = tmp_path / "tile.md"
    source.write_text("original")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("original with append")
    objects: list[JsonObject] = [
        {
            "completed": True,
            "projection_quality_passed": True,
            "source_hashes": {"docs/analysis/tile.md": expected},
            "source_archive": {},
        },
        {"audit_passed": True, "artifacts": {}},
        {"completed": True, "source_archive": {}, "source_sha256": {}},
    ]
    pinned: dict[Path, str] = {}
    for index, value in enumerate(objects):
        path = tmp_path / f"receipt{index}.json"
        path.write_text(json.dumps(value))
        pinned[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(run, "PINNED", pinned)

    def archive(*args: object) -> dict[str, Path]:
        return {"docs/analysis/tile.md": source} if args[2] == 123 else {}

    def forbidden(*args: object) -> None:
        raise AssertionError("Checkpoint reached despite123 source mutation")

    monkeypatch.setattr(run, "archive", archive)
    monkeypatch.setattr(run, "admit", forbidden)
    with pytest.raises(ValueError, match="source bytes changed"):
        run.context(tmp_path)
