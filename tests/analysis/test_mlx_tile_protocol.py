"""Pure history gates and fake dispatch/failure observations; no device work."""

from __future__ import annotations

import hashlib
import itertools
import json
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

from benchmarks.analysis.mlx.tile import admit as admission
from wyat.wire import JsonObject

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.mlx.mixed.study import SHAPES
from benchmarks.analysis.mlx.tile import study as runner


def blocks(ratios: list[float]) -> list[dict[str, object]]:
    return [
        {
            "order": list(order),
            "dispatch_ns": {
                "maintained": [int(ratio * 10000)] * 10,
                "tile32-bf16": [20000] * 10,
                "tile64-bf16": [10000] * 10,
            },
        }
        for order, ratio in zip(runner.ORDERS, ratios, strict=True)
    ]


@given(st.lists(st.integers(10100, 20000), min_size=6, max_size=6))
def test_keeps_original_threshold_and_separate32_comparison(values: list[int]) -> None:
    ratios = [value / 10000 for value in values]
    actual = runner.comparison(blocks(ratios), "maintained")
    assert actual["passed"] is (
        cast(float, actual["paired_median"]) >= 1.05
        and cast(float, actual["paired_p05"]) > 1
    )
    separate = runner.comparison(blocks(ratios), "tile32-bf16")
    assert separate["paired_median"] == 2 and "passed" not in separate


@pytest.mark.parametrize("fault", ["missing", "order", "zero", "bool", "count", "arm"])
def test_incomplete_or_unbalanced_timings_fail(fault: str) -> None:
    data = blocks([1.1] * 6)
    if fault == "missing":
        data.pop()
    elif fault == "order":
        data[1]["order"] = data[0]["order"]
    else:
        times = cast(dict[str, list[int]], data[0]["dispatch_ns"])
        if fault == "count":
            times["maintained"].pop()
        elif fault == "arm":
            del times["tile32-bf16"]
        else:
            times["maintained"][0] = 0 if fault == "zero" else True
    with pytest.raises(ValueError):
        runner.comparison(data, "maintained")


def test_parent_import_is_framework_free() -> None:
    script = (
        "import sys; import benchmarks.analysis.mlx.tile.run; "
        "assert not any(x in sys.modules for x in "
        "('torch','mlx','mlx.core','mlx_lm'))"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def make_archive(
    tmp_path: Path, *, documentation: bool = False
) -> tuple[JsonObject, JsonObject, Path]:
    source = tmp_path / "source.py"
    text = "original\n"
    source.write_text(text)
    name = admission.DOC if documentation else "source.py"
    sha = hashlib.sha256(text.encode()).hexdigest()
    archive = tmp_path / "archive.json"
    archive.write_text(
        json.dumps(
            {
                "files": [
                    {"name": name, "path": str(source), "text": text, "sha256": sha}
                ]
            }
        )
    )
    return (
        {"path": str(archive), "sha256": admission.digest(archive)},
        {name: sha},
        source,
    )


def test_archive_and_current_executable_bytes_both_required(tmp_path: Path) -> None:
    record, hashes, source = make_archive(tmp_path)
    assert admission.archive(record, hashes, 1) == {"source.py": source}
    source.write_text("changed")
    with pytest.raises(ValueError, match="executable"):
        admission.archive(record, hashes, 1)


def test_only_pinned_append_can_change_documentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, hashes, source = make_archive(tmp_path, documentation=True)
    source.write_text("original\nresult append\n")
    monkeypatch.setattr(admission, "DOC_SHA256", admission.digest(source))
    assert admission.archive(record, hashes, 1)
    source.write_text("arbitrary rewrite\n")
    monkeypatch.setattr(admission, "DOC_SHA256", admission.digest(source))
    with pytest.raises(ValueError, match="append"):
        admission.archive(record, hashes, 1)


def test_changed_history_fails_before_checkpoint_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "initial.json"
    path.write_text("changed")
    monkeypatch.setattr(admission, "IDENTITIES", {path: "0" * 64})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Checkpoint loader reached before history gate")

    monkeypatch.setattr(admission, "load_artifact", forbidden)
    with pytest.raises(ValueError, match="receipt"):
        admission.admit(tmp_path)


def fake_dispatch(monkeypatch: pytest.MonkeyPatch) -> dict[str, runner.Operation]:
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "memory", lambda: {"active": 0})
    monkeypatch.setattr(runner, "output_digest", lambda value: str(value))
    clock = itertools.count(1, 10)
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(clock))

    def operation(arm: str) -> runner.Operation:
        def dispatch() -> mx.array:
            return cast(mx.array, arm)

        return dispatch

    return {arm: operation(arm) for arm in runner.ARMS}


def test_every_warm_and_timed_hash_checked_with_complete_arm_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_dispatch(monkeypatch)
    result: dict[str, object] = {}
    writes: list[int] = []
    runner.timed_blocks(
        calls, {arm: arm for arm in runner.ARMS}, result, lambda: writes.append(1)
    )
    assert len(writes) == 26  # warm +18 arms +6 complete blocks +summary
    data = cast(list[dict[str, object]], result["blocks"])
    assert len(data) == 6
    assert (
        sum(
            len(row)
            for block in data
            for row in cast(dict[str, list[str]], block["output_sha256"]).values()
        )
        == 180
    )
    assert all(
        block["completed"] is True and block["outputs_reproduce_cold"] is True
        for block in data
    )


def test_warm_drift_blocks_timing_and_keeps_raw_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_dispatch(monkeypatch)
    result: dict[str, object] = {}
    with pytest.raises(ValueError, match="Warm"):
        runner.timed_blocks(
            calls, {arm: "wrong" for arm in runner.ARMS}, result, lambda: None
        )
    assert "warm_output_sha256" in result and "blocks" not in result


def test_incomplete_arm_write_does_not_claim_complete_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_dispatch(monkeypatch)
    result: dict[str, object] = {}
    count = 0

    def fail() -> None:
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("arm receipt failed")

    with pytest.raises(OSError):
        runner.timed_blocks(calls, {arm: arm for arm in runner.ARMS}, result, fail)
    block = cast(list[dict[str, object]], result["blocks"])[0]
    assert block["completed"] is False
    assert len(cast(dict[str, list[int]], block["dispatch_ns"])) == 1


def test_cold64_storage_and_live32_history_are_independent_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "detached_float32", lambda value: value)
    base = np.array([[1.0]], dtype=np.float32)
    arrays = {arm: base.copy() for arm in (*runner.ARMS, runner.CONTROL)}
    original = {"shape-maintained": base, "shape-custom-bf16": base}

    def observe() -> dict[str, object]:
        def operation(arm: str) -> runner.Operation:
            def dispatch() -> mx.array:
                return cast(mx.array, arrays[arm])

            return dispatch

        calls = {arm: operation(arm) for arm in arrays}
        return runner.screen(calls, original, "shape")[0]

    assert observe()["passed"] is True
    arrays[runner.CONTROL][0, 0] = np.nextafter(np.float32(1), np.float32(2))
    assert observe()["passed"] is False  # inside numerical tolerance, different bits
    arrays[runner.CONTROL] = base.copy()
    arrays["tile32-bf16"][0, 0] = np.nextafter(np.float32(1), np.float32(2))
    assert observe()["original_cold_outputs_reproduced"] is False


@pytest.mark.parametrize("fail", [False, True])
def test_parent_restores_tf32_on_launch_success_or_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: bool
) -> None:
    from benchmarks.analysis.mlx.tile import run

    monkeypatch.setenv("MLX_ENABLE_TF32", "prior-value")
    monkeypatch.setattr(
        sys, "argv", ["run", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )

    def launch(*args: object, **kwargs: object) -> int:
        import os

        assert os.environ["MLX_ENABLE_TF32"] == "0"
        assert kwargs["timeout"] == 180
        if fail:
            raise OSError("launcher failure")
        return 1

    monkeypatch.setattr(run, "launch", launch)
    with pytest.raises(OSError if fail else SystemExit):
        run.main()
    import os

    assert os.environ["MLX_ENABLE_TF32"] == "prior-value"


def test_late_receipt_write_failure_never_leaves_accepted_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from benchmarks.analysis.evidence import FrozenSources

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(
        runner, "freeze_sources", lambda *args: FrozenSources((), "unused", "unused")
    )

    def no_inputs(*args: object) -> None:
        raise ValueError("fake input admission failure")

    monkeypatch.setattr(runner, "read_arrays", no_inputs)
    writes = 0
    from benchmarks.analysis.mlx.mixed.study import save as original

    def save(path: Path, value: dict[str, object]) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("late disk failure")
        original(path, value)

    monkeypatch.setattr(runner, "save", save)
    output = tmp_path / "trial.json"
    admitted = admission.Admission({}, tmp_path, tmp_path, {})
    assert runner.study(admitted, output) is False
    receipt = json.loads(output.read_text())
    assert receipt["completed"] is False and receipt["accepted"] is False
    assert receipt["final_write_error"] == "OSError: late disk failure"
    assert receipt["error"] == "ValueError: fake input admission failure"


@pytest.mark.parametrize("failure", ["numerical", "timing", "cache", "final-write"])
def test_outer_trial_retains_negative_and_restores_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    from benchmarks.analysis.evidence import FrozenSources
    from benchmarks.analysis.mlx.mixed import study as original_runner
    from benchmarks.analysis.mlx.tile import kernel

    class Array:
        def __init__(self, value: runner.Array) -> None:
            self.value = value

        @property
        def T(self) -> Array:
            return Array(self.value.T)

        def astype(self, dtype: object) -> Array:
            return self

        def __matmul__(self, other: Array) -> Array:
            return Array(self.value @ other.value)

    class Prepared:
        def __init__(self, x: Array, w: Array) -> None:
            self.x, self.w = x, w

        def run(self) -> Array:
            return self.x @ self.w.T

    class Kernel:
        def __init__(self, *args: object) -> None:
            pass

        def prepare(self, x: Array, w: Array) -> Prepared:
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
                raise OSError("cache restoration failed")

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(original_runner, "bounded_cache", bounded)
    monkeypatch.setattr(mx, "device_info", lambda *args: {"name": "fake"})
    monkeypatch.setattr(mx, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(mx, "array", lambda value, **kwargs: Array(value.copy()))
    monkeypatch.setattr(mx, "eval", lambda *args: None)
    monkeypatch.setattr(mx, "synchronize", lambda *args: None)
    monkeypatch.setattr(runner, "detached_float32", lambda value: value.value.copy())
    monkeypatch.setattr(runner, "FrozenKernel", Kernel)
    monkeypatch.setattr(kernel, "ProjectionKernel", Kernel)
    monkeypatch.setattr(
        runner, "freeze_sources", lambda *args: FrozenSources((), "fake", "f" * 64)
    )
    inputs = {"weight": np.ones((2, 2), dtype=np.float32)}
    inputs.update({name: np.ones((1, 2), dtype=np.float32) for name, _, _ in SHAPES})
    old = {
        name + "-" + arm: np.full((1, 2), 2, dtype=np.float32)
        for name, _, _ in SHAPES
        for arm in ("maintained", "custom-bf16")
    }
    monkeypatch.setattr(runner, "read_arrays", lambda *args: (inputs, old))
    tensor = tmp_path / "inputs"
    tensor.write_bytes(b"fake")
    initial: JsonObject = {
        "cold_outputs": {},
        "model_id": "fake",
        "dataset_sha256": "fake",
        "reference_sha256": "fake",
        "input_sha256": "fake",
    }
    admitted = admission.Admission(initial, tensor, tensor, {})
    original_screen = runner.screen

    def screen(*args: object) -> tuple[dict[str, object], dict[str, runner.Array]]:
        result, values = original_screen(
            *cast(
                tuple[dict[str, runner.Operation], dict[str, runner.Array], str], args
            )
        )
        if failure == "numerical":
            result["passed"] = False
        return result, values

    monkeypatch.setattr(runner, "screen", screen)
    timing_calls = 0

    def timing(
        calls: dict[str, runner.Operation],
        cold: dict[str, str],
        result: dict[str, object],
        progress: object,
    ) -> None:
        nonlocal timing_calls
        timing_calls += 1
        if failure == "timing":
            raise ValueError("fake timing failed")
        result["performance"] = {"passed": True}

    monkeypatch.setattr(runner, "timed_blocks", timing)
    failed_write = False

    def save(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed_write
        if (
            failure == "final-write"
            and payload.get("accepted") is True
            and not failed_write
        ):
            failed_write = True
            raise OSError("late disk failure")
        original_runner.save(path, payload)

    monkeypatch.setattr(runner, "save", save)
    output = tmp_path / "trial.json"
    assert runner.study(admitted, output) is False
    receipt = json.loads(output.read_text())
    assert restored == [True] and receipt["accepted"] is False
    assert receipt["completed"] is (failure == "numerical")
    assert receipt["cache_cap_restoration_completed"] is (failure != "cache")
    assert timing_calls == (
        0 if failure == "numerical" else 1 if failure == "timing" else 4
    )
    assert output.with_suffix(".outputs.tensors.safetensors").exists()
    if failure == "final-write":
        assert "late disk failure" in receipt["final_write_error"]


def test_tensor_reopen_must_still_match_before_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "inputs"
    path.write_bytes(b"changed")
    initial: JsonObject = {"captured_bytes": {"sha256": "0" * 64}}
    admitted = admission.Admission(initial, path, path, {})

    def forbidden(raw: bytes) -> None:
        raise AssertionError("Parser consumed unadmitted bytes")

    monkeypatch.setattr(runner, "load", forbidden)
    with pytest.raises(ValueError, match="before consumption"):
        runner.read_arrays(admitted)
