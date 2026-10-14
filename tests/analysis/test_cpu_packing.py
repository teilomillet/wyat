"""Tiny independent algebra and ownership checks; no real weights or JIT."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from benchmarks.analysis.torch import packing
from benchmarks.analysis.torch.autotune import output_gate, retain_tensors
from wyat.backends.wyat.torch import WyatScorer


def mlp() -> Qwen3MLP:
    constructor = cast(Callable[[Qwen3Config], Qwen3MLP], Qwen3MLP)
    result = constructor(Qwen3Config(hidden_size=8, intermediate_size=12))
    with torch.no_grad():
        for offset, parameter in enumerate(result.parameters()):
            parameter.copy_(
                torch.linspace(-0.15 + offset * 0.01, 0.1, parameter.numel()).reshape(
                    parameter.shape
                )
            )
    result.eval()
    return result


@settings(max_examples=12, deadline=None)
@given(batch=st.integers(1, 3), tokens=st.integers(1, 9))
def test_packing_matches_independent_projection_algebra(
    batch: int, tokens: int
) -> None:
    original = mlp()
    candidate = packing.PackedMLP(original)
    value = torch.linspace(-2, 2, batch * tokens * 8).reshape(batch, tokens, 8)
    with torch.inference_mode():
        gate = torch.nn.functional.linear(value, original.gate_proj.weight)
        up = torch.nn.functional.linear(value, original.up_proj.weight)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, original.down_proj.weight
        )
        actual = candidate(value)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert candidate.gate_up.weight.dtype == torch.float32
    assert candidate.down_proj is original.down_proj
    assert not candidate.training
    assert torch.equal(candidate.gate_up.weight[:12], original.gate_proj.weight)
    assert torch.equal(candidate.gate_up.weight[12:], original.up_proj.weight)


def test_packed_storage_cannot_mutate_original_projections() -> None:
    original = mlp()
    before = original.gate_proj.weight.detach().clone()
    candidate = packing.PackedMLP(original)
    with torch.no_grad():
        candidate.gate_up.weight.zero_()
    assert torch.equal(original.gate_proj.weight, before)


@pytest.mark.parametrize("invalid", ["bias", "dtype", "shape"])
def test_packing_rejects_outside_declared_contract(invalid: str) -> None:
    original = mlp()
    if invalid == "bias":
        original.gate_proj = nn.Linear(8, 12, bias=True)
    elif invalid == "dtype":
        original.up_proj.half()
    else:
        original.up_proj = nn.Linear(8, 11, bias=False)
    with pytest.raises(ValueError, match="matching biasless CPU FP32"):
        packing.PackedMLP(original)


def test_backbone_ownership_and_input_perturbation() -> None:
    config = Qwen3Config(
        vocab_size=40,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
    )
    original = Qwen3Model(config)
    cast(nn.Module, original).eval()
    before = dict(original.named_modules())
    candidate = packing.packed_backbone(original)
    assert candidate._modules is not original._modules
    assert candidate.layers is not original.layers
    for old, new in zip(original.layers, candidate.layers, strict=True):
        assert old is not new
        assert old._modules is not new._modules
        assert old.self_attn is new.self_attn
        assert old.mlp is not new.mlp
        old_mlp = cast(Qwen3MLP, old.mlp)
        new_mlp = cast(packing.PackedMLP, new.mlp)
        assert old_mlp.down_proj is new_mlp.down_proj
    assert dict(original.named_modules()) == before
    with torch.inference_mode():
        for ids in (torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]), torch.tensor([[1, 3]])):
            expected = original(ids, use_cache=False).last_hidden_state
            actual = candidate(ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # Replacing a candidate-owned child cannot overwrite the baseline dictionary.
    candidate.layers[0].mlp = nn.Identity()
    assert original.layers[0].mlp is before["layers.0.mlp"]


def test_projection_gate_rejects_drift_and_nonfinite() -> None:
    reference = torch.tensor([[1.0, -2.0, 0.0]])
    assert output_gate(reference, reference.clone())["passed"] is True
    assert output_gate(reference, reference + 0.01)["passed"] is False
    assert (
        output_gate(reference, torch.full_like(reference, float("nan")))["passed"]
        is False
    )
    with pytest.raises(ValueError, match="shape or dtype"):
        output_gate(reference, reference.half())


def test_projection_gate_is_explicitly_distinct_from_engine_tolerance() -> None:
    # This additive local probe does not certify the engine's stricter max gate.
    reference = torch.tensor([1.0])
    assert output_gate(reference, torch.tensor([1.0015]))["passed"] is True
    assert output_gate(reference, torch.tensor([1.003]))["passed"] is False


def test_archive_preserves_aliased_and_strided_values(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    path = tmp_path / "raw.safetensors"
    digest = retain_tensors({"reference": value, "same": value, "view": value.T}, path)
    restored = load_file(path)
    assert len(digest) == 64
    assert torch.equal(restored["reference"], value)
    assert torch.equal(restored["same"], value)
    assert torch.equal(restored["view"], value.T)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_load_failure_restores_threads_and_retains_nonacceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    state = {"threads": 3}

    def set_threads(value: int) -> None:
        state["threads"] = value

    def fail_load(*args: object, **kwargs: object) -> None:
        raise ValueError("failed artifact")

    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    monkeypatch.setattr(WyatScorer, "from_directory", fail_load)
    with pytest.raises(ValueError, match="failed artifact"):
        packing.study(
            tmp_path / "model", tmp_path / "result.json", batch_size=batch_size
        )
    assert state["threads"] == 3
    from wyat.wire import read_object

    receipt = read_object(tmp_path / "result.json")
    assert receipt["completed"] is False
    assert receipt["accepted"] is False
    assert receipt["phase"] == "failed"
    assert receipt["batch_size"] == batch_size


@pytest.mark.parametrize("batch_size", [True, False, 0, 2, 16])
def test_study_rejects_invalid_batch_size_before_effects(
    tmp_path: Path,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="batch size 1 or 8"):
        packing.study(
            tmp_path / "model", tmp_path / "result.json", batch_size=batch_size
        )
    assert list(tmp_path.iterdir()) == []
