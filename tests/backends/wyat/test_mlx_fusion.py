"""Packing mechanics against unchanged upstream operations; no quality claims."""

from __future__ import annotations

from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model  # noqa: E402

from benchmarks.analysis.mlx.fusion import (  # noqa: E402
    PackedAttention,
    packed_backbone,
)
from wyat.backends.wyat.mlx import MLXCandidateHead  # noqa: E402


def backbone() -> Qwen3Model:
    # The real checkpoint also has Q=2*hidden and K/V=hidden. Existing Wyat
    # fixtures exercise different head geometry, so that fixture is insufficient.
    return Qwen3Model(ModelArgs("qwen3", 16, 2, 32, 2, 1e-6, 64, 1, 256, 1e6, 16, True))


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("qkv,gate_up", [(True, False), (False, True), (True, True)])
@given(choices=st.integers(1, 6), extra=st.integers(1, 12), padding=st.integers(1, 9))
@settings(max_examples=8, deadline=None)
def test_asymmetric_packing_endpoint_and_causal_parity(
    device: str, qkv: bool, gate_up: bool, choices: int, extra: int, padding: int
) -> None:
    if device == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU unavailable")
    with mx.stream(mx.Device(mx.gpu if device == "gpu" else mx.cpu)):
        mx.random.seed(17)
        original = backbone()
        packed = packed_backbone(original, qkv=qkv, gate_up=gate_up)
        head = MLXCandidateHead(16, 8)
        length = choices + extra + 1
        single_ids = mx.arange(length, dtype=mx.int32)[None, :]
        batched_ids = mx.stack(
            [
                mx.concatenate([single_ids[0], mx.zeros((padding,), dtype=mx.int32)]),
                mx.arange(length + padding, dtype=mx.int32),
            ]
        )
        original_hidden = cast(mx.array, original(batched_ids))
        packed_hidden = cast(mx.array, packed(batched_ids))
        scalar_hidden = cast(mx.array, packed(single_ids))
        positions = mx.arange(choices)
        reference = head(original_hidden[:, positions], original_hidden[:, length - 1])
        actual = head(packed_hidden[:, positions], packed_hidden[:, length - 1])
        scalar = head(scalar_hidden[:, positions], scalar_hidden[:, length - 1])
        mx.eval(reference, actual, scalar)
        assert bool(mx.all(mx.isfinite(actual)).item())
        assert cast(list[list[float]], actual.tolist())[0] == pytest.approx(
            cast(list[list[float]], scalar.tolist())[0], abs=2e-5, rel=1e-5
        )
        for left, right in zip(
            cast(list[list[float]], actual.tolist()),
            cast(list[list[float]], reference.tolist()),
            strict=True,
        ):
            assert left == pytest.approx(right, abs=2e-5, rel=1e-5)


def test_packed_attention_rejects_cache() -> None:
    with mx.stream(mx.cpu):
        attention = PackedAttention(backbone().layers[0].self_attn)
        with pytest.raises(ValueError, match="prefill"):
            attention(mx.zeros((1, 2, 16)), "causal", object())
