"""Projection algebra, exact shared storage and effect/failure boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.analysis.torch.prefix.production import study as original
from benchmarks.analysis.torch.projection import qkv, run, study


def weights(width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.arange(4 * width * width, dtype=torch.float32).reshape(
        4 * width, width
    )
    return (
        values[: 2 * width] / 100,
        values[2 * width : 3 * width] / 100,
        values[3 * width :] / 100,
    )


@settings(max_examples=12, deadline=None, derandomize=True)
@given(width=st.integers(2, 12), batch=st.integers(1, 3), length=st.integers(1, 11))
def test_packed_projection_matches_independent_linear(
    width: int, batch: int, length: int
) -> None:
    source = weights(width)
    packed = qkv.PackedQKV.from_weights(*source)
    value = torch.sin(
        torch.arange(batch * length * width, dtype=torch.float32)
    ).reshape(batch, length, width)
    baseline = cast(
        qkv.Outputs, tuple(torch.nn.functional.linear(value, w) for w in source)
    )
    pointer = packed.weight.untyped_storage().data_ptr()
    assert all(
        w.untyped_storage().data_ptr() == pointer and w.is_contiguous()
        for w in packed.views
    )
    assert all(torch.equal(a, b) for a, b in zip(source, packed.views, strict=True))
    for arm in ("separate", "kv", "qkv"):
        actual = packed.project(value, arm)
        assert qkv.quality(baseline, actual)["passed"] is True
        assert tuple(t.shape for t in actual) == (
            (batch, length, 2 * width),
            (batch, length, width),
            (batch, length, width),
        )
    assert packed.weight.numel() * 4 == sum(w.numel() * 4 for w in source)


def test_output_views_preserve_observed_stride_difference() -> None:
    packed = qkv.PackedQKV.from_weights(*weights(4))
    value = torch.ones(2, 3, 4)
    separate = packed.project(value, "separate")
    combined = packed.project(value, "qkv")
    assert separate[0].stride() == (24, 8, 1)
    assert combined[0].stride() == (48, 16, 1)
    assert (
        combined[0].untyped_storage().data_ptr()
        == combined[1].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("bad", ["dtype", "gradient", "width", "finite", "autocast"])
def test_unsupported_weights_fail_before_cat(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    source = list(weights(4))
    if bad == "dtype":
        source[0] = source[0].double()
    if bad == "gradient":
        source[0].requires_grad_(True)
    if bad == "width":
        source[0] = source[0][:-1]
    if bad == "finite":
        source[0][0, 0] = float("nan")

    def unexpected(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("No packing allocation before admission")

    monkeypatch.setattr(torch, "cat", unexpected)
    if bad == "autocast":
        with torch.autocast("cpu", dtype=torch.bfloat16), pytest.raises(ValueError):
            qkv.PackedQKV.from_weights(*source)
        assert not torch.is_autocast_enabled("cpu")
    else:
        with pytest.raises(ValueError):
            qkv.PackedQKV.from_weights(*source)


def test_symmetric_max_gate_accepts_near_zero_and_rejects_drift() -> None:
    expected = (torch.tensor([0.0]),) * 3
    close = (torch.tensor([0.0009]),) * 3
    far = (torch.tensor([0.0011]),) * 3
    assert qkv.quality(expected, close)["passed"] is True
    assert qkv.quality(expected, far)["passed"] is False
    assert qkv.quality(expected, (torch.tensor([float("nan")]),) * 3)["finite"] is False


def test_launcher_uses_importable_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    from benchmarks.analysis.torch import process

    captured: list[str] = []

    def launch(module: str, model: Path, output: Path, *, timeout: float) -> int:
        captured.append(module)
        assert timeout == 120
        return 0

    monkeypatch.setattr(process, "launch", launch)
    monkeypatch.setattr(run, "__name__", "__main__")
    monkeypatch.setattr(
        sys,
        "argv",
        ["projection.run", str(tmp_path), "--output", str(tmp_path / "run.json")],
    )
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 0
    assert captured == ["benchmarks.analysis.torch.projection.run"]


class Frozen:
    def hashes(self) -> dict[str, str]:
        return {}

    def archive(self) -> dict[str, object]:
        return {}


def fake_setup(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    threads = {"value": 3}

    def set_threads(value: int) -> None:
        threads["value"] = value

    def get_threads() -> int:
        return threads["value"]

    def no_verify(files: object) -> None:
        pass

    def fake_operands(
        model: Path, rows: object, pad: int
    ) -> tuple[
        qkv.PackedQKV,
        list[tuple[str, torch.Tensor, dict[str, object]]],
        dict[str, object],
    ]:
        packed = qkv.PackedQKV.from_weights(*weights(4))
        return (
            packed,
            [(name, torch.ones(1, 2, 4), {}) for name, _ in study.PROFILES],
            {},
        )

    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(torch, "get_num_threads", get_threads)
    monkeypatch.setattr(study, "input_vectors", lambda: [])
    monkeypatch.setattr(study, "padding_id", lambda model: 151643)
    monkeypatch.setattr(study, "operands", fake_operands)
    monkeypatch.setattr(study, "freeze_sources", lambda paths, output: Frozen())
    monkeypatch.setattr(original, "verify_files", no_verify)
    monkeypatch.setattr(original, "verify_sources", no_verify)
    return threads


def test_fake_worker_retains_all_calls_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    threads = fake_setup(monkeypatch)
    output = tmp_path / "primitive.json"
    assert study.study(tmp_path, output)
    result = json.loads(output.read_text())
    assert (
        result["completed"] and result["quality_passed"] and result["accepted"] is False
    )
    assert sum(len(entry["calls"]) for entry in result["operands"]) == 96
    assert all(entry["pending_calls"] == 0 for entry in result["operands"])
    assert result["threads_restored"] and threads["value"] == 3
    for entry in result["operands"]:
        assert Path(entry["raw_tensors"]).exists()


def test_quality_exception_retains_produced_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    threads = fake_setup(monkeypatch)

    def fail(reference: qkv.Outputs, actual: qkv.Outputs) -> dict[str, object]:
        raise RuntimeError("gate failed after output")

    monkeypatch.setattr(study, "quality", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        study.study(tmp_path, output)
    result = json.loads(output.read_text())
    first = result["operands"][0]
    assert len(first["calls"]) == 1 and first["calls"][0]["quality"] is None
    assert Path(first["raw_tensors"]).exists()
    assert not result["completed"] and not result["quality_passed"]
    assert threads["value"] == 3 and first["pending_calls"] == 23


@pytest.mark.parametrize("boundary", ["restore", "final_save"])
def test_completion_failure_clears_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, boundary: str
) -> None:
    threads = fake_setup(monkeypatch)
    output = tmp_path / "completion-failed.json"
    if boundary == "restore":

        def set_threads(value: int) -> None:
            if value == 3:
                raise RuntimeError("restoration failed")
            threads["value"] = value

        monkeypatch.setattr(torch, "set_num_threads", set_threads)
    else:
        original_write = Path.write_text
        failed = False

        def write_text(
            path: Path,
            data: str,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            nonlocal failed
            if path == output and json.loads(data)["completed"] and not failed:
                failed = True
                raise OSError("final save failed")
            return original_write(
                path, data, encoding=encoding, errors=errors, newline=newline
            )

        monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises((RuntimeError, OSError)):
        study.study(tmp_path, output)
    result = json.loads(output.read_text())
    assert not result["completed"] and not result["quality_passed"]
    assert result["primitive_eligible"] == {"kv": False, "qkv": False}
    assert sum(len(entry["calls"]) for entry in result["operands"]) == 96


def test_parent_sidecars_survive_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_setup(monkeypatch)
    output = tmp_path / "parent-owned.json"
    sidecars = {
        output.with_suffix(".wall.json"): b"owned wall",
        output.with_suffix(".log"): b"owned log",
    }
    for path, data in sidecars.items():
        path.write_bytes(data)
    assert study.study(tmp_path, output)
    assert all(path.read_bytes() == data for path, data in sidecars.items())


@pytest.mark.parametrize(
    "suffix", [".json", ".unknown", ".tensors.safetensors", ".sources-executed.json"]
)
def test_existing_experiment_family_fails_before_admission(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str
) -> None:
    output = tmp_path / "preserved.json"
    existing = output.with_suffix(suffix)
    existing.write_bytes(b"preserve exact prior bytes")

    def unexpected() -> list[dict[str, object]]:
        raise AssertionError("No admission or weights before collision guard")

    monkeypatch.setattr(study, "input_vectors", unexpected)
    with pytest.raises(FileExistsError):
        study.study(tmp_path, output)
    assert existing.read_bytes() == b"preserve exact prior bytes"
