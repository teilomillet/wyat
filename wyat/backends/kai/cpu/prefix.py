"""Opt-in exact retained-prefix reuse with maintained CPU FP32 caches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache, DynamicLayer

from wyat.core import Request

from ..encoding import EncodedRequest
from ..torch import KaiScorer

type KVState = tuple[torch.Tensor, torch.Tensor]


class CacheConstructor(Protocol):
    def __call__(
        self,
        *,
        config: Qwen3Config,
        ddp_cache_data: tuple[KVState, ...] | None = None,
    ) -> DynamicCache: ...


# The maintained constructor is untyped; narrow only this external boundary.
new_cache = cast(CacheConstructor, DynamicCache)


def _cache(
    config: Qwen3Config, states: tuple[KVState, ...] | None = None
) -> DynamicCache:
    try:
        owner = new_cache(config=config, ddp_cache_data=states)
        if (
            type(owner) is not DynamicCache
            or len(owner.layers) != config.num_hidden_layers
            or any(type(layer) is not DynamicLayer for layer in owner.layers)
        ):
            raise ValueError("CPU prefix requires stock full-attention DynamicCache")
        return owner
    except (TypeError, AttributeError) as error:
        raise ValueError("CPU prefix requires maintained DynamicCache APIs") from error


def _prepared_cache(config: Qwen3Config) -> DynamicCache:
    """Check seed/copy/append semantics before any model prefix forward."""
    try:
        empty = _cache(config)
        if empty.get_seq_length() != 0 or empty.get_mask_sizes(1, 0) != (1, 0):
            raise ValueError("CPU prefix requires an empty full-attention cache")
        kv_heads = config.num_key_value_heads
        if kv_heads is None or kv_heads <= 0:
            raise ValueError("CPU prefix requires configured key/value heads")
        seed = torch.zeros(
            (1, kv_heads, 1, config.head_dim), dtype=torch.float32, device="cpu"
        )
        owner = _cache(config, tuple((seed, seed) for _ in empty.layers))
        for raw_layer in owner.layers:
            layer = cast(DynamicLayer, raw_layer)  # Exact stock type checked by _cache.
            if (
                layer.keys is None
                or layer.values is None
                or layer.keys.data_ptr() == seed.data_ptr()
                or layer.values.data_ptr() == seed.data_ptr()
                or not torch.equal(layer.keys, seed)
                or not torch.equal(layer.values, seed)
            ):
                raise ValueError("CPU prefix requires independently owned cache seeds")
        if owner.get_seq_length() != 1 or owner.get_mask_sizes(1, 0) != (2, 0):
            raise ValueError(
                "CPU prefix requires absolute full-attention cache lengths"
            )
        seeded = cast(DynamicLayer, owner.layers[0]).keys
        suffix = torch.ones_like(seed)
        keys, values = owner.update(suffix, suffix, 0)
        if (
            seeded is None
            or owner.get_seq_length() != 2
            or keys.data_ptr() == seeded.data_ptr()
            or not torch.equal(seeded, seed)
            or not torch.equal(keys[..., :1, :], seed)
            or not torch.equal(values[..., :1, :], seed)
            or not torch.equal(keys[..., 1:, :], suffix)
            or not torch.equal(values[..., 1:, :], suffix)
        ):
            raise ValueError("CPU prefix requires allocating cache concatenation")
        return empty
    except (TypeError, AttributeError) as error:
        raise ValueError("CPU prefix requires maintained DynamicCache APIs") from error


class PrefixScorer:
    """Eager immutable prefix snapshots; each call owns a retained stock cache.

    Keep the base weights, head, config, device and encoder fixed for this
    resource's lifetime. No suffix state survives a call. Matching batches retain
    all layers' extended K/V through readout; this explicitly trades memory for
    prefix reuse. A mixed or overlapping-readout batch falls back intact.
    """

    def __init__(self, base: KaiScorer, prefix: tuple[int, ...]) -> None:
        if not isinstance(base, KaiScorer):
            raise TypeError("CPU prefix requires a KaiScorer base")
        if type(prefix) is not tuple:
            raise TypeError("Prefix must be an exact immutable token tuple")
        self._require_no_autocast()
        config = base.backbone.config
        rotary = config.rope_parameters
        if (
            not isinstance(rotary, dict)
            or rotary.get("rope_type", "default") != "default"
        ):
            raise ValueError("CPU prefix requires default unscaled RoPE")
        if (
            base.device.type != "cpu"
            or config._attn_implementation != "sdpa"
            or any(module.training for module in base.backbone.modules())
            or any(module.training for module in base.head.modules())
            or any(
                value.device.type != "cpu" or value.dtype != torch.float32
                for value in (*base.backbone.parameters(), *base.head.parameters())
            )
            or any(value.device.type != "cpu" for value in base.backbone.buffers())
            or base.backbone.has_sliding_layers
        ):
            raise ValueError("CPU prefix requires eval FP32 CPU SDPA full attention")
        self._require_no_hooks(base)
        if (
            not prefix
            or len(prefix) >= base.max_length
            or any(
                type(token) is not int or not 0 <= token < config.vocab_size
                for token in prefix
            )
        ):
            raise ValueError("Prefix requires valid nonempty tokens below max_length")
        self.base = base
        self.model_id = base.model_id
        self._prefix = prefix
        with torch.inference_mode():
            cache = _prepared_cache(config)
            base.backbone(
                input_ids=torch.tensor([prefix], dtype=torch.long, device="cpu"),
                past_key_values=cache,
                use_cache=True,
            )
            if len(cache.layers) != config.num_hidden_layers:
                raise ValueError("Maintained cache omitted a prefix layer")
            states: list[KVState] = []
            for raw_layer in cache.layers:
                if type(raw_layer) is not DynamicLayer:
                    raise ValueError("CPU prefix requires stock DynamicLayer state")
                layer = raw_layer
                keys, values = layer.keys, layer.values
                if (
                    keys is None
                    or values is None
                    or keys.shape != values.shape
                    or tuple(keys.shape)
                    != (1, config.num_key_value_heads, len(prefix), config.head_dim)
                    or keys.device.type != "cpu"
                    or values.device.type != "cpu"
                    or keys.dtype != torch.float32
                    or values.dtype != torch.float32
                    or keys.stride(-1) != 1
                    or values.stride(-1) != 1
                    or not torch.isfinite(keys).all().item()
                    or not torch.isfinite(values).all().item()
                ):
                    raise ValueError(
                        "Maintained cache did not retain exact FP32 prefix"
                    )
                states.append(
                    (
                        keys.detach().clone().contiguous(),
                        values.detach().clone().contiguous(),
                    )
                )
            self._states = tuple(states)

    @staticmethod
    def _require_no_autocast() -> None:
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("CPU prefix requires disabled CPU autocast")

    @staticmethod
    def _require_no_hooks(base: KaiScorer) -> None:
        if any(
            module._forward_hooks or module._forward_pre_hooks
            for module in base.backbone.modules()
        ):
            raise ValueError("CPU prefix does not support backbone forward hooks")

    @property
    def prefix(self) -> tuple[int, ...]:
        return self._prefix

    @property
    def prefix_state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for pair in self._states
            for value in pair
        )

    def _admits(self, row: EncodedRequest) -> bool:
        length = len(self.prefix)
        return (
            len(row.ids) > length
            and row.ids[:length] == self.prefix
            and row.query_position >= length
            and all(position >= length for position in row.candidate_positions)
        )

    def _fork(self, batch_size: int) -> DynamicCache:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Cache batch size must be a positive integer")
        states = tuple(
            (keys.expand(batch_size, -1, -1, -1), values.expand(batch_size, -1, -1, -1))
            for keys, values in self._states
        )
        return _cache(self.base.backbone.config, states)

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        self._require_no_autocast()
        self._require_no_hooks(self.base)
        if not requests:
            return []
        # Admission belongs to complete original rows before removing any tokens.
        encoded = [self.base.encode_request(request) for request in requests]
        if not all(self._admits(row) for row in encoded):
            return self.base._score_encoded(encoded)
        with torch.inference_mode():
            return self.base._score_encoded(
                encoded, prefix_length=len(self.prefix), cache=self._fork(len(encoded))
            )
