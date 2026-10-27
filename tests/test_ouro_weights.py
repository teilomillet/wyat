# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Exact, bounded weight staging with tiny modules only."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from wyat.loading.ouro_weights import StreamedWeights  # noqa: E402
from wyat.resources import ResourceGuard, ResourceLimitError  # noqa: E402


def _guard() -> ResourceGuard:
    guard = ResourceGuard()
    guard.enabled = False
    return guard


def _source(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    values = {
        "model.layers.0.weight": torch.arange(32).reshape(4, 8).to(torch.bfloat16) / 16,
        "model.layers.0.bias": torch.arange(4).to(torch.bfloat16) / 8,
    }
    path = tmp_path / "base.safetensors"
    save_file(values, str(path))
    return path, values


def _expected(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: torch.empty(value.shape, device="meta") for name, value in values.items()
    }


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_staging_is_bitwise_equal_to_resident_fp32_and_restores_objects(
    tmp_path: Path, device: str
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS unavailable")
    path, values = _source(tmp_path)
    target = torch.device(device)
    storage = StreamedWeights(path, _expected(values), target, _guard())
    with torch.device("meta"):
        layer = nn.Linear(8, 4)
    layer.register_buffer("scratch", torch.tensor([9.0]), persistent=False)
    original = dict(layer.named_parameters())
    scratch = layer.scratch
    hidden = torch.arange(24, device=target, dtype=torch.float32).reshape(3, 8) / 10
    expected = F.linear(
        hidden,
        values["model.layers.0.weight"].float().to(target),
        values["model.layers.0.bias"].float().to(target),
    )
    for _ in range(2):
        with storage.stage(layer, "model.layers.0"):
            assert layer.weight.device == hidden.device
            assert layer.weight.dtype == torch.float32
            assert layer.scratch.device == hidden.device
            assert torch.equal(layer(hidden), expected)
        assert all(
            dict(layer.named_parameters())[key] is value
            for key, value in original.items()
        )
        assert layer.scratch is scratch
    assert storage.max_layer_bytes == (32 + 4) * 4
    assert storage.peak_bytes >= storage.max_layer_bytes


def test_adapter_values_remain_fp32_and_base_name_is_normalized(tmp_path: Path) -> None:
    path, values = _source(tmp_path)
    with torch.device("meta"):
        layer = nn.Module()
        layer.base = nn.Linear(8, 4)
    layer.adapter = nn.Parameter(torch.tensor([0.123456789], dtype=torch.float32))
    adapter = layer.adapter
    base = layer.base.weight
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    with storage.stage(layer, "model.layers.0"):
        assert torch.equal(layer.base.weight, values["model.layers.0.weight"].float())
        assert torch.equal(layer.adapter, adapter)
        assert layer.adapter.dtype == torch.float32
    assert layer.adapter is adapter and layer.base.weight is base


def test_execution_failure_restores_parameters_and_buffers(tmp_path: Path) -> None:
    path, values = _source(tmp_path)
    layer = nn.Linear(8, 4, device="meta")
    layer.register_buffer("scratch", torch.tensor([7.0]), persistent=False)
    before = layer.weight, layer.bias, layer.scratch
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    with pytest.raises(RuntimeError, match="execution failed"):
        with storage.stage(layer, "model.layers.0"):
            raise RuntimeError("execution failed")
    assert all(
        actual is wanted
        for actual, wanted in zip(
            (layer.weight, layer.bias, layer.scratch), before, strict=True
        )
    )


def test_mid_stage_failure_restores_already_replaced_slots(tmp_path: Path) -> None:
    path, values = _source(tmp_path)
    layer = nn.Linear(8, 4, device="meta")
    layer.missing_adapter = nn.Parameter(torch.empty(1, device="meta"))
    original = dict(layer.named_parameters())
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    with pytest.raises(ValueError, match="no materialized value"):
        with storage.stage(layer, "model.layers.0"):
            pytest.fail("body must not be reached")
    assert all(
        dict(layer.named_parameters())[key] is value for key, value in original.items()
    )


@pytest.mark.parametrize("kind", ["keys", "dtype", "shape", "nonfinite"])
def test_source_validation_refuses_invalid_checkpoint(
    tmp_path: Path, kind: str
) -> None:
    path, values = _source(tmp_path)
    expected = _expected(values)
    bad = dict(values)
    if kind == "keys":
        bad["unexpected"] = torch.ones(1, dtype=torch.bfloat16)
    elif kind == "dtype":
        bad["model.layers.0.weight"] = bad["model.layers.0.weight"].float()
    elif kind == "shape":
        bad["model.layers.0.weight"] = torch.ones((5, 8), dtype=torch.bfloat16)
    else:
        bad["model.layers.0.weight"] = bad["model.layers.0.weight"].clone()
        bad["model.layers.0.weight"][0, 0] = float("nan")
    save_file(bad, str(path))
    with pytest.raises(ValueError):
        StreamedWeights(path, expected, torch.device("cpu"), _guard())


def test_only_missing_lm_head_is_permitted(tmp_path: Path) -> None:
    path, values = _source(tmp_path)
    expected = _expected(values)
    expected["lm_head.weight"] = torch.empty((2, 2), device="meta")
    StreamedWeights(path, expected, torch.device("cpu"), _guard())
    expected["model.norm.weight"] = torch.empty(2, device="meta")
    with pytest.raises(ValueError, match="native architecture"):
        StreamedWeights(path, expected, torch.device("cpu"), _guard())


def test_source_change_after_validation_is_refused_without_mutation(
    tmp_path: Path,
) -> None:
    path, values = _source(tmp_path)
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    layer = nn.Linear(8, 4, device="meta")
    original = layer.weight
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed after validation"):
        with storage.stage(layer, "model.layers.0"):
            pytest.fail("changed source must not execute")
    assert layer.weight is original


def test_guard_refusal_happens_before_staging_mutates_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, values = _source(tmp_path)
    guard = _guard()
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), guard)
    layer = nn.Linear(8, 4, device="meta")
    original = layer.weight

    def refuse(*args: object, **kwargs: object) -> None:
        raise ResourceLimitError("budget exhausted")

    monkeypatch.setattr(guard, "check", refuse)
    with pytest.raises(ResourceLimitError, match="budget exhausted"):
        with storage.stage(layer, "model.layers.0"):
            pytest.fail("refused stage must not execute")
    assert layer.weight is original


def test_constructor_refuses_validation_when_guard_cannot_admit_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, values = _source(tmp_path)
    guard = _guard()

    def refuse(*args: object, **kwargs: object) -> None:
        raise ResourceLimitError("validation budget exhausted")

    monkeypatch.setattr(guard, "check", refuse)
    with pytest.raises(ResourceLimitError, match="validation budget exhausted"):
        StreamedWeights(path, _expected(values), torch.device("cpu"), guard)


def test_module_shape_mismatch_is_refused_before_installing_weights(
    tmp_path: Path,
) -> None:
    path, values = _source(tmp_path)
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    layer = nn.Linear(7, 4, device="meta")
    original = layer.weight
    with pytest.raises(ValueError, match="module shape differs"):
        with storage.stage(layer, "model.layers.0"):
            pytest.fail("incorrect module must not execute")
    assert layer.weight is original


def test_temporary_adapter_mutation_does_not_change_retained_fp32_values(
    tmp_path: Path,
) -> None:
    path, values = _source(tmp_path)
    layer = nn.Module()
    layer.base = nn.Linear(8, 4, device="meta")
    layer.adapter = nn.Parameter(torch.tensor([0.123456789]))
    original = layer.adapter
    before = original.detach().clone()
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), _guard())
    with storage.stage(layer, "model.layers.0"), torch.no_grad():
        layer.adapter.add_(1)
    assert layer.adapter is original
    assert torch.equal(layer.adapter, before)


def test_existing_buffer_dtype_is_preserved_and_counted_in_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, values = _source(tmp_path)
    guard = _guard()
    storage = StreamedWeights(path, _expected(values), torch.device("cpu"), guard)
    layer = nn.Linear(8, 4, device="meta")
    layer.register_buffer(
        "scratch", torch.ones(3, dtype=torch.float64), persistent=False
    )
    original = layer.scratch
    calls: list[int] = []

    def check(additional_bytes: int = 0, operation: str = "inference") -> None:
        calls.append(additional_bytes)

    monkeypatch.setattr(guard, "check", check)
    with storage.stage(layer, "model.layers.0"):
        assert layer.scratch.dtype == torch.float64
    assert layer.scratch is original
    assert calls == [(32 + 4) * 4 + 3 * 8 + 32 * 2]
