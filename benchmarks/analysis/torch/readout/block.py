"""Selected final queries with full K/V and original absolute causal positions.

Callers supply penultimate suffix hidden states and maintained RoPE embeddings
for P..P+S-1, prepared before cache updates. The fresh stock cache is disposable:
an exception after its final-layer append consumes that owner, not its snapshots.
This experiment changes GEMM/SDPA shapes; real-arithmetic equivalence is not a
claim of bit-identical floating-point outputs, dispatch or a measured speed gain.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward as sdpa_attention_forward,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    rotate_half,
)

# Narrow the maintained untyped rotation boundary, not its implementation.
rotate = cast(Callable[[torch.Tensor], torch.Tensor], rotate_half)


def _validate(
    block: Qwen3DecoderLayer,
    norm: Qwen3RMSNorm,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    embeddings: tuple[torch.Tensor, torch.Tensor],
    cache: DynamicCache,
    prefix_length: int,
) -> None:
    if (
        type(embeddings) is not tuple
        or len(embeddings) != 2
        or any(not isinstance(value, torch.Tensor) for value in embeddings)
    ):
        raise ValueError("RoPE requires exactly two maintained tensors")
    config = block.self_attn.config
    rotary, layer_types = config.rope_parameters, config.layer_types
    if (
        type(block) is not Qwen3DecoderLayer
        or type(norm) is not Qwen3RMSNorm
        or tuple(norm.weight.shape) != (config.hidden_size,)
        or torch.is_autocast_enabled("cpu")
        or config._attn_implementation != "sdpa"
        or not isinstance(rotary, dict)
        or rotary.get("rope_type", "default") != "default"
        or layer_types is None
        or any(kind != "full_attention" for kind in layer_types)
        or block.self_attn.layer_idx != config.num_hidden_layers - 1
    ):
        raise ValueError("Readouts require stock final CPU FP32 SDPA/default RoPE")
    modules = (*block.modules(), *norm.modules())
    if any(
        module.training or module._forward_hooks or module._forward_pre_hooks
        for module in modules
    ) or any(
        value.device.type != "cpu" or value.dtype != torch.float32
        for module in (block, norm)
        for value in (*module.parameters(), *module.buffers())
    ):
        raise ValueError("Readouts require unhooked eval FP32 CPU modules")
    if (
        hidden.device.type != "cpu"
        or hidden.dtype != torch.float32
        or hidden.requires_grad
        or hidden.ndim != 3
        or hidden.shape[0] < 1
        or hidden.shape[1] < 1
        or hidden.shape[2] != config.hidden_size
        or positions.device.type != "cpu"
        or positions.dtype != torch.long
        or positions.ndim != 2
        or positions.shape[0] != hidden.shape[0]
        or positions.shape[1] < 1
        or torch.any(positions < 0).item()
        or torch.any(positions >= hidden.shape[1]).item()
        or type(prefix_length) is not int
        or prefix_length < 0
    ):
        raise ValueError("Readouts require valid CPU suffix rows and ordered indices")
    if (
        type(cache) is not DynamicCache
        or len(cache.layers) != config.num_hidden_layers
        or any(type(layer) is not DynamicLayer for layer in cache.layers)
        or cache.get_seq_length(block.self_attn.layer_idx) != prefix_length
    ):
        raise ValueError("Readouts require a fresh stock final-layer prefix cache")
    layer = cast(DynamicLayer, cache.layers[block.self_attn.layer_idx])
    for state in (layer.keys, layer.values):
        if prefix_length and (
            state is None
            or tuple(state.shape)
            != (
                hidden.shape[0],
                config.num_key_value_heads,
                prefix_length,
                config.head_dim,
            )
            or state.device.type != "cpu"
            or state.dtype != torch.float32
            or state.stride(-1) != 1
        ):
            raise ValueError("Final prefix K/V must retain every FP32 token/head")
    for value in embeddings:
        if (
            value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.ndim != 3
            or value.shape[0] not in (1, hidden.shape[0])
            or tuple(value.shape[1:]) != (hidden.shape[1], config.head_dim)
        ):
            raise ValueError(
                "RoPE must cover the full suffix at its absolute positions"
            )


def _rotate(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    # Maintained Qwen3 rotation, separately shaped for selected Q and full K.
    return value * cosine[:, None] + rotate(value) * sine[:, None]


def selected_last_block(
    block: Qwen3DecoderLayer,
    norm: Qwen3RMSNorm,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache: DynamicCache,
    *,
    prefix_length: int,
) -> torch.Tensor:
    """Return normalized ordered readouts; duplicates are preserved, never deduped."""
    _validate(block, norm, hidden, positions, position_embeddings, cache, prefix_length)
    with torch.inference_mode():
        batch, length, _ = hidden.shape
        rows = torch.arange(batch)[:, None]
        residual = hidden[rows, positions]
        normalized = block.input_layernorm(hidden)
        attention = block.self_attn
        head_dim = attention.head_dim
        query = attention.q_norm(
            attention.q_proj(normalized[rows, positions]).view(
                batch,
                positions.shape[1],
                attention.config.num_attention_heads,
                head_dim,
            )
        ).transpose(1, 2)
        key = attention.k_norm(
            attention.k_proj(normalized).view(batch, length, -1, head_dim)
        ).transpose(1, 2)
        value = (
            attention.v_proj(normalized)
            .view(batch, length, -1, head_dim)
            .transpose(1, 2)
        )
        cosine, sine = (value.expand(batch, -1, -1) for value in position_embeddings)
        query = _rotate(query, cosine[rows, positions], sine[rows, positions])
        key = _rotate(key, cosine, sine)
        key, value = cache.update(key, value, attention.layer_idx)
        mask = torch.arange(prefix_length + length)[None, None, None, :] <= (
            prefix_length + positions[:, None, :, None]
        )
        attended, _ = sdpa_attention_forward(
            attention,
            query,
            key,
            value,
            mask,
            dropout=0.0,
            scaling=attention.scaling,
            is_causal=False,
        )
        attended = attended.reshape(batch, positions.shape[1], -1).contiguous()
        selected = residual + attention.o_proj(attended)
        selected = selected + block.mlp(block.post_attention_layernorm(selected))
        return cast(torch.Tensor, norm(selected))
