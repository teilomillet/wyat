"""Independent default-Qwen rotary oracles and bounded table ownership."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("torch")
import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from transformers import Qwen3Config  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import (  # noqa: E402
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

from wyat.loading.mlx import _RoPE  # noqa: E402


@pytest.fixture(params=["cpu", "gpu"])
def device(request: pytest.FixtureRequest) -> mx.Device:
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("Metal GPU is unavailable")
    return mx.Device(mx.gpu if request.param == "gpu" else mx.cpu)


@pytest.mark.parametrize(
    ("offset", "width"), [(0, 17), (2040, 17), (5428, 9), (8183, 9)]
)
def test_native_rope_matches_independent_torch_qwen_at_long_positions(
    device: mx.Device, offset: int, width: int
) -> None:
    generator = torch.Generator(device="cpu").manual_seed(81)
    # Unit-bounded inputs keep the absolute rotation error interpretable.
    source = torch.rand((2, 3, width, 128), generator=generator) * 2 - 1
    config = Qwen3Config(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=8192,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    reference = Qwen3RotaryEmbedding(config)  # type: ignore[no-untyped-call, unused-ignore]
    positions = torch.arange(offset, offset + width).unsqueeze(0).expand(2, -1)
    with torch.inference_mode():
        cosine, sine = reference(source, positions)
        expected, _ = apply_rotary_pos_emb(  # type: ignore[no-untyped-call, unused-ignore]
            source, source, cosine, sine
        )
    with mx.stream(device):
        rope = _RoPE(128, 1000000.0)
        actual = rope(mx.array(source.numpy()), offset=offset)
        mx.eval(actual)
        assert actual.dtype == mx.float32
        assert actual.shape == source.shape
        converted = torch.tensor(actual.tolist(), dtype=torch.float32)
    # FP32 frequency rounding is amplified by position. This component budget
    # allows <= 0.001 absolute error through position 8191 on unit inputs;
    # trained-model logits, probabilities and complete rankings have their own
    # stricter end-to-end qualification and are never inferred from this test.
    torch.testing.assert_close(converted, expected, atol=1e-3, rtol=2e-6)
    if offset == 0:
        assert torch.equal(converted[..., 0, :], source[..., 0, :])


def test_native_rope_caches_one_width_and_replaces_offset_or_width(
    device: mx.Device,
) -> None:
    with mx.stream(device):
        rope = _RoPE(128, 1000000.0)
        first = mx.ones((1, 2, 5, 128), dtype=mx.float32)
        mx.eval(rope(first))
        assert set(rope._tables) == {(5, 0)}
        saved = rope._tables[(5, 0)]
        mx.eval(rope(first))
        assert rope._tables[(5, 0)][0] is saved[0]
        assert rope._tables[(5, 0)][1] is saved[1]
        mx.eval(rope(first, offset=17))
        assert set(rope._tables) == {(5, 17)}
        assert rope._tables[(5, 17)][0] is not saved[0]
        second = mx.ones((2, 3, 7, 128), dtype=mx.float32)
        mx.eval(rope(second))
        assert set(rope._tables) == {(7, 0)}
        cosine, sine = rope._tables[(7, 0)]
        assert cosine.shape == sine.shape == (7, 128)
        assert cosine.dtype == sine.dtype == mx.float32


@pytest.mark.parametrize("dims", [0, -2, 1, 3, True, 128.0])
def test_native_rope_rejects_invalid_rotary_width(dims: object) -> None:
    with pytest.raises(ValueError, match="rotary width"):
        _RoPE(cast(int, dims), 1000000.0)


@pytest.mark.parametrize("base", [0.0, -1.0, float("inf"), float("nan"), True])
def test_native_rope_rejects_invalid_base(base: float) -> None:
    with pytest.raises(ValueError, match="rotary base"):
        _RoPE(128, base)


@pytest.mark.parametrize("offset", [-1, 1.0, True, None])
def test_native_rope_rejects_noninteger_or_negative_offsets(
    device: mx.Device, offset: object
) -> None:
    with mx.stream(device):
        rope = _RoPE(128, 1000000.0)
        value = mx.ones((1, 2, 128), dtype=mx.float32)
        with pytest.raises(ValueError, match="rotary input"):
            rope(value, offset=cast(int, offset))
        assert not rope._tables


@pytest.mark.parametrize("shape", [(128,), (1, 2, 127), (1, 0, 128)])
def test_native_rope_rejects_partial_or_empty_rotary_shapes(
    device: mx.Device, shape: tuple[int, ...]
) -> None:
    with mx.stream(device):
        rope = _RoPE(128, 1000000.0)
        value = mx.zeros(shape, dtype=mx.float32)
        with pytest.raises(ValueError, match="rotary input"):
            rope(value)
        assert not rope._tables


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.int32])
def test_native_rope_rejects_input_precision_changes(
    device: mx.Device, dtype: mx.Dtype
) -> None:
    with mx.stream(device):
        rope = _RoPE(128, 1000000.0)
        value = mx.ones((1, 2, 128), dtype=dtype)
        with pytest.raises(ValueError, match="rotary input"):
            rope(value)
        assert not rope._tables
