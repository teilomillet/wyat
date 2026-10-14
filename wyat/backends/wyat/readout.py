# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Selected final Qwen3 readouts with original causal and rotary positions."""

# Portions adapted from MLX-LM Qwen3, Copyright © 2023–2024 Apple Inc.
# MIT License; the complete permission notice is included in repository NOTICE.

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import mlx.core as mx
from mlx.nn.layers.linear import Linear
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3 import Qwen3Model, TransformerBlock

type ArrayMap = Callable[[mx.array], mx.array]


class RotaryMap(Protocol):
    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array: ...


class AttentionWeights(Protocol):
    n_heads: int
    n_kv_heads: int
    scale: float
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: ArrayMap
    k_norm: ArrayMap
    rope: RotaryMap


class EndpointCache(Protocol):
    offset: int

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]: ...


def endpoint_last_block(
    block: TransformerBlock,
    hidden: mx.array,
    positions: mx.array,
    cache: EndpointCache | None = None,
) -> mx.array:
    """Select suffix rows, retaining cached K/V and absolute causal positions."""
    batch, length, _ = hidden.shape
    offset = 0 if cache is None else cache.offset
    absolute_positions = positions + offset
    rows = mx.arange(batch)[:, None]
    residual = hidden[rows, positions]
    normalize = cast(ArrayMap, block.input_layernorm)
    normalized = normalize(hidden)
    selected = normalized[rows, positions]
    attention = cast(AttentionWeights, block.self_attn)
    count = positions.shape[1]
    queries = attention.q_norm(
        attention.q_proj(selected).reshape(batch, count, attention.n_heads, -1)
    )
    dimension = queries.shape[-1]
    # Treat each original endpoint as a separate one-token sequence. Native
    # RoPE accepts one offset per batch row, preserving nonconsecutive positions.
    queries = (
        attention.rope(
            queries.reshape(batch * count, attention.n_heads, 1, dimension),
            offset=absolute_positions.reshape(-1),
        )
        .reshape(batch, count, attention.n_heads, dimension)
        .transpose(0, 2, 1, 3)
    )
    keys = attention.k_norm(
        attention.k_proj(normalized).reshape(batch, length, attention.n_kv_heads, -1)
    ).transpose(0, 2, 1, 3)
    keys = attention.rope(keys, offset=offset)
    values = (
        attention.v_proj(normalized)
        .reshape(batch, length, attention.n_kv_heads, dimension)
        .transpose(0, 2, 1, 3)
    )
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    mask = (
        mx.arange(keys.shape[2])[None, None, None, :]
        <= absolute_positions[:, None, :, None]
    )
    attended = (
        mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=attention.scale, mask=mask
        )
        .transpose(0, 2, 1, 3)
        .reshape(batch, count, -1)
    )
    residual = residual + attention.o_proj(attended)
    post_normalize = cast(ArrayMap, block.post_attention_layernorm)
    mlp = cast(ArrayMap, block.mlp)
    return residual + mlp(post_normalize(residual))


class CachedBlock(Protocol):
    def __call__(
        self, hidden: mx.array, mask: str | mx.array | None, cache: EndpointCache
    ) -> mx.array: ...


attention_mask = cast(
    Callable[[mx.array, EndpointCache], str | mx.array | None], create_attention_mask
)


def cached_endpoint_hidden(
    backbone: Qwen3Model,
    ids: mx.array,
    positions: mx.array,
    caches: Sequence[EndpointCache],
) -> mx.array:
    """Use upstream cached early blocks, then selected final suffix readouts."""
    if backbone.args.rope_scaling is not None:
        raise ValueError("Endpoint readout requires unscaled native RoPE")
    if len(caches) != len(backbone.layers) or not caches:
        raise ValueError("One fresh prefix cache is required per maintained layer")
    if any(cache.offset != caches[0].offset for cache in caches):
        raise ValueError("All caches must start at the same original prefix offset")
    hidden = backbone.embed_tokens(ids)
    mask = attention_mask(hidden, caches[0])
    blocks = cast(Sequence[CachedBlock], backbone.layers[:-1])
    for block, cache in zip(blocks, caches[:-1], strict=True):
        hidden = block(hidden, mask, cache)
    hidden = endpoint_last_block(backbone.layers[-1], hidden, positions, caches[-1])
    normalize = cast(ArrayMap, backbone.norm)
    return normalize(hidden)
