"""CPU-only projection recipe/oracle and fresh candidate ownership checks."""

from __future__ import annotations

import json
import platform
from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)
pytest.importorskip("mlx_lm", exc_type=ImportError)
pytest.importorskip("torch", exc_type=ImportError)

import mlx.core as mx
import torch
from mlx.nn.layers.linear import Linear
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3 import MLP, ModelArgs, Qwen3Model

from benchmarks.analysis.mlx.bfloat.products import (
    ARMS,
    Arm,
    BFloatMLP,
    candidate_backbone,
    exact_bfloat_linear,
    product,
)
from wyat.backends.wyat.encoding import WyatEncoder
from wyat.backends.wyat.mlx import MLXCandidateHead, MLXWyatScorer
from wyat.core import Candidate, Request


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) % 128 for character in text]


@pytest.fixture
def backbone() -> Qwen3Model:
    with mx.stream(mx.cpu):
        mx.random.seed(91)
        model = Qwen3Model(
            ModelArgs("qwen3", 16, 2, 32, 2, 1e-6, 128, 1, 512, 1e6, 16, True)
        )
        for block in model.layers:
            for projection in (
                block.mlp.gate_proj,
                block.mlp.up_proj,
                block.mlp.down_proj,
            ):
                projection.weight = projection.weight.astype(mx.bfloat16).astype(
                    mx.float32
                )
        model.eval()
        return model


def tensor(value: mx.array) -> torch.Tensor:
    return torch.tensor(cast(list[list[float]], value.tolist()), dtype=torch.float32)


def oracle(original: MLP, value: torch.Tensor, arm: Arm) -> torch.Tensor:
    """Independent Torch CPU Linear/SwiGLU with explicit declared rounding."""
    gate_up = arm != "all-down"
    down = arm != "all-gate-up"
    operand = value.to(torch.bfloat16) if gate_up else value
    dtype = torch.bfloat16 if gate_up else torch.float32
    gate = torch.nn.functional.linear(
        operand, tensor(original.gate_proj.weight).to(dtype)
    ).float()
    up = torch.nn.functional.linear(
        operand, tensor(original.up_proj.weight).to(dtype)
    ).float()
    activation = torch.nn.functional.silu(gate) * up
    operand = activation.to(torch.bfloat16) if down else activation
    weight = (
        tensor(original.down_proj.weight).to(torch.bfloat16)
        if down
        else tensor(original.down_proj.weight)
    )
    return torch.nn.functional.linear(operand, weight).float()


def failure_stages(
    message: str,
    original: MLP,
    projection: BFloatMLP,
    value: mx.array,
    arm: Arm,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> str:
    """Replay this tiny recipe only on failure; preserve both original outputs."""
    gate_input = value.astype(mx.bfloat16) if projection.gate_up_reduced else value
    gate = product(projection.gate_proj, gate_input)
    up = product(projection.up_proj, gate_input)
    activation = cast(Callable[[mx.array, mx.array], mx.array], swiglu)(gate, up)
    operand = tensor(value).bfloat16() if arm != "all-down" else tensor(value)
    dtype = torch.bfloat16 if arm != "all-down" else torch.float32
    torch_gate = torch.nn.functional.linear(
        operand, tensor(original.gate_proj.weight).to(dtype)
    ).float()
    torch_up = torch.nn.functional.linear(
        operand, tensor(original.up_proj.weight).to(dtype)
    ).float()
    torch_activation = torch.nn.functional.silu(torch_gate) * torch_up

    def primitive(array: torch.Tensor) -> list[list[float]]:
        # The fixture is H16 / intermediate32, never real model parameters.
        if array.ndim != 2 or array.numel() > 512:
            raise ValueError("BF16 diagnostic exceeds the tiny fixture boundary")
        return cast(list[list[float]], array.tolist())

    payload = {
        "arm": arm,
        "input": primitive(tensor(value)),
        "weights": {
            "gate": primitive(tensor(original.gate_proj.weight)),
            "up": primitive(tensor(original.up_proj.weight)),
            "down": primitive(tensor(original.down_proj.weight)),
        },
        "mlx": {
            "gate": primitive(tensor(gate)),
            "up": primitive(tensor(up)),
            "activation": primitive(tensor(activation)),
            "down": primitive(actual),
        },
        "torch": {
            "gate": primitive(torch_gate),
            "up": primitive(torch_up),
            "activation": primitive(torch_activation),
            "down": primitive(expected),
        },
        "runtime": {
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "mlx": version("mlx"),
            "threads": torch.get_num_threads(),
            "stream": "cpu",
        },
    }
    return message + "\nBFLOAT_STAGES=" + json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("arm", ARMS)
@given(values=st.lists(st.integers(-16, 16), min_size=16, max_size=16))
@example(values=[0] * 15 + [3])
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_projection_roles_match_independent_cpu_recipe(
    backbone: Qwen3Model, arm: Arm, values: list[int]
) -> None:
    with mx.stream(mx.cpu):
        original = backbone.layers[0].mlp
        projection = BFloatMLP(original, arm)
        value = mx.array([values], dtype=mx.float32) / 8
        actual = tensor(projection(value))
        expected = oracle(original, tensor(value), arm)
        torch.testing.assert_close(
            actual,
            expected,
            atol=2e-6,
            rtol=2e-5,
            msg=lambda message: failure_stages(
                message, original, projection, value, arm, actual, expected
            ),
        )
        assert projection(value).dtype == mx.float32
        assert projection.gate_proj.weight.dtype == (
            mx.bfloat16 if arm != "all-down" else mx.float32
        )
        assert projection.down_proj.weight.dtype == (
            mx.bfloat16 if arm != "all-gate-up" else mx.float32
        )


@pytest.mark.parametrize("arm", ARMS)
def test_fresh_modules_preserve_baseline_attention_norm_embedding_and_weights(
    backbone: Qwen3Model, arm: Arm
) -> None:
    with mx.stream(mx.cpu):
        ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        before = tensor(backbone(ids)[0])
        candidate = candidate_backbone(backbone, arm)
        for left, right in zip(backbone.layers, candidate.layers, strict=True):
            assert left is not right and left.mlp is not right.mlp
            for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
                assert (
                    getattr(left.self_attn, role).weight
                    is getattr(right.self_attn, role).weight
                )
            assert left.input_layernorm.weight is right.input_layernorm.weight
            assert left.mlp.gate_proj.weight.dtype == mx.float32
        assert backbone.embed_tokens.weight is candidate.embed_tokens.weight
        assert backbone.norm.weight is candidate.norm.weight
        result = candidate(ids)
        assert result.dtype == mx.float32 and bool(mx.all(mx.isfinite(result)).item())
        torch.testing.assert_close(tensor(backbone(ids)[0]), before, rtol=0, atol=0)
        duplicate = candidate_backbone(backbone, arm)
        assert duplicate.layers[0].mlp is not candidate.layers[0].mlp


@pytest.mark.parametrize("value", [1.0001, float("nan"), float("inf")])
def test_inexact_or_nonfinite_source_parameters_rejected(value: float) -> None:
    with mx.stream(mx.cpu):
        projection = Linear(1, 1, bias=False)
        projection.weight = mx.array([[value]], dtype=mx.float32)
        with pytest.raises(ValueError, match="finite exact"):
            exact_bfloat_linear(projection)


@pytest.mark.parametrize(
    ("weight", "expected"),
    [(0.1240234375, 0.04638671875), (-0.1240234375, -0.04638671875)],
)
def test_cpu_bfloat_product_rounds_halfway_to_even(
    weight: float, expected: float
) -> None:
    with mx.stream(mx.cpu):
        projection = Linear(1, 1, bias=False)
        projection.weight = mx.array([[weight]], dtype=mx.bfloat16)
        result = product(projection, mx.array([[0.375]], dtype=mx.bfloat16))
        assert result.dtype == mx.float32
        assert cast(float, result.item()) == expected


def test_dtype_bias_and_unknown_arm_boundaries(backbone: Qwen3Model) -> None:
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="bias-free"):
            exact_bfloat_linear(Linear(1, 1, bias=True))
        projection = exact_bfloat_linear(backbone.layers[0].mlp.down_proj)
        with pytest.raises(ValueError, match="share their dtype"):
            product(projection, mx.ones((1, 32), dtype=mx.float32))
        with pytest.raises(ValueError, match="Unknown"):
            candidate_backbone(backbone, cast(Arm, "last-one"))
        with pytest.raises(ValueError, match="remain FP32"):
            BFloatMLP(backbone.layers[0].mlp, "all-down")(
                mx.ones((1, 16), dtype=mx.bfloat16)
            )


@pytest.mark.parametrize("arm", ARMS)
@given(
    menus=st.lists(
        st.lists(
            st.text(alphabet="ab", min_size=1, max_size=8), min_size=2, max_size=5
        ),
        min_size=1,
        max_size=3,
    )
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_full_readout_mixed_lengths_counts_and_scalar_batch(
    backbone: Qwen3Model, arm: Arm, menus: list[list[str]]
) -> None:
    values = tuple(
        Request(
            query="pick",
            candidates=tuple(
                Candidate(str(index), text) for index, text in enumerate(menu)
            ),
        )
        for menu in menus
    )
    with mx.stream(mx.cpu):
        head = MLXCandidateHead(16, 8)
        head.eval()
        scorer = MLXWyatScorer(
            candidate_backbone(backbone, arm),
            head,
            WyatEncoder(CharacterTokenizer(), max_length=512, max_options=8),
            model_id="tiny-bfloat-asymmetric",
            device=mx.Device(mx.cpu),
        )
        scalar = [scorer.score_many((value,))[0] for value in values]
        for batch in (values, values[::-1], values):
            actual = scorer.score_many(batch)
            expected = scalar if batch == values else scalar[::-1]
            for left, right, request in zip(actual, expected, batch, strict=True):
                assert len(left) == len(request.candidates)
                assert left == pytest.approx(right, abs=3e-6, rel=1e-5)
        assert head.key.weight.dtype == head.scalar.weight.dtype == mx.float32
