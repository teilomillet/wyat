"""Experiment-only BF16 MLP products; every other maintained stage stays FP32."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3 import MLP, Qwen3Model

from benchmarks.analysis.mlx.bfloat.run import ARMS as ARMS
from benchmarks.analysis.mlx.bfloat.run import Arm as Arm
from benchmarks.analysis.mlx.profile import ArrayMap


def exact_bfloat_linear(original: Linear) -> Linear:
    """Finite, value-exact BF16 parameters; approximation belongs to products."""
    weight = original.weight
    if weight.dtype != mx.float32 or "bias" in original:
        raise ValueError("Requires bias-free FP32 source projection")
    converted = weight.astype(mx.bfloat16)
    source, restored = weight.reshape(-1), converted.astype(mx.float32).reshape(-1)
    for start in range(0, source.size, 1 << 18):
        before, after = (
            source[start : start + (1 << 18)],
            restored[start : start + (1 << 18)],
        )
        if not bool(mx.all(mx.isfinite(before) & (before == after)).item()):
            raise ValueError("Projection parameters are not finite exact BF16 values")
    projection = Linear(weight.shape[1], weight.shape[0], bias=False)
    projection.weight = converted
    mx.eval(projection.weight)
    return projection


def product(projection: Linear, value: mx.array) -> mx.array:
    expected = projection.weight.dtype
    if expected not in (mx.bfloat16, mx.float32) or value.dtype != expected:
        raise ValueError("Declared projection operands must share their dtype")
    result = projection(value)
    if result.dtype != expected:
        raise ValueError("Unexpected projection promotion changes the declared recipe")
    return result.astype(mx.float32)


# MLX's base type is unavailable on CPU-only installations.
class BFloatMLP(Module):  # type: ignore[misc, unused-ignore]
    def __init__(self, original: MLP, arm: Arm) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        if arm not in ARMS:
            raise ValueError("Unknown fixed BF16 role arm")
        self.gate_up_reduced = arm in ("all-gate-up", "all-mlp")
        self.down_reduced = arm in ("all-down", "all-mlp")
        self.gate_proj = (
            exact_bfloat_linear(original.gate_proj)
            if self.gate_up_reduced
            else original.gate_proj
        )
        self.up_proj = (
            exact_bfloat_linear(original.up_proj)
            if self.gate_up_reduced
            else original.up_proj
        )
        self.down_proj = (
            exact_bfloat_linear(original.down_proj)
            if self.down_reduced
            else original.down_proj
        )

    def __call__(self, x: mx.array) -> mx.array:
        if x.dtype != mx.float32:
            raise ValueError("MLP input/SwiGLU must remain FP32")
        gate_input = x.astype(mx.bfloat16) if self.gate_up_reduced else x
        gate = product(self.gate_proj, gate_input)
        up = product(self.up_proj, gate_input)
        activation = cast(Callable[[mx.array, mx.array], mx.array], swiglu)(gate, up)
        down_input = activation.astype(mx.bfloat16) if self.down_reduced else activation
        return product(self.down_proj, down_input)


class MutableBlock(Protocol):
    mlp: ArrayMap


def candidate_backbone(original: Qwen3Model, arm: Arm) -> Qwen3Model:
    """Fresh module owners; unchanged FP32 arrays are shared, never mutated."""
    if arm not in ARMS:
        raise ValueError("Unknown fixed BF16 role arm")
    candidate = Qwen3Model(original.args)
    parameters = cast(Callable[[], object], original.parameters)()
    flattened = cast(list[tuple[str, mx.array]], tree_flatten(parameters))
    candidate.load_weights(flattened, strict=True)
    for block in cast(Sequence[MutableBlock], candidate.layers):
        block.mlp = BFloatMLP(cast(MLP, block.mlp), arm)
    candidate.eval()
    return candidate
