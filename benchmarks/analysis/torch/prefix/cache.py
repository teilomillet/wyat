"""Exact CPU FP32 prefix state; maintained cached model and full final block."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer
from wyat.core import Request

type KVState = tuple[torch.Tensor, torch.Tensor]


class CacheConstructor(Protocol):
    def __call__(
        self,
        *,
        config: Qwen3Config,
        ddp_cache_data: tuple[KVState, ...] | None = None,
    ) -> DynamicCache: ...


# The maintained cache constructor is untyped; narrow this external boundary.
new_cache = cast(CacheConstructor, DynamicCache)


class PrefixScorer:
    """Fixed weights/config/device/encoder for one immutable prefix lifetime.

    Each invocation owns its DynamicCache. Seeding and suffix updates concatenate
    into private storage; no suffix state survives the invocation. Callers must
    keep the base in eval mode and never mutate its weights, config or encoder.
    Mismatched/overlapping batches fall back intact, including admission checks.
    """

    def __init__(self, base: WyatScorer, prefix: tuple[int, ...]) -> None:
        if not isinstance(base, WyatScorer):
            raise TypeError("CPU prefix requires a WyatScorer base")
        if type(prefix) is not tuple:
            raise TypeError("Prefix must be an exact immutable token tuple")
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
            cache = new_cache(config=config)
            base.backbone(
                input_ids=torch.tensor([prefix], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            states: list[KVState] = []
            for layer in cache.layers:
                if not isinstance(layer, DynamicLayer):
                    raise ValueError(
                        "CPU prefix requires maintained DynamicLayer state"
                    )
                keys, values = layer.keys, layer.values
                if (
                    keys is None
                    or values is None
                    or keys.shape != values.shape
                    or tuple(keys.shape)
                    != (1, config.num_key_value_heads, len(prefix), config.head_dim)
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
            if len(states) != config.num_hidden_layers:
                raise ValueError("Maintained cache omitted a prefix layer")
            self._states = tuple(states)

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

    def admits(self, row: EncodedRequest) -> bool:
        length = len(self.prefix)
        return (
            len(row.ids) > length
            and row.ids[:length] == self.prefix
            and row.query_position >= length
            and all(position >= length for position in row.candidate_positions)
        )

    def fork(self, batch_size: int) -> DynamicCache:
        """Maintained update/cat allocates batch-private K/V from prefix views."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Cache batch size must be a positive integer")
        states = tuple(
            (keys.expand(batch_size, -1, -1, -1), values.expand(batch_size, -1, -1, -1))
            for keys, values in self._states
        )
        return new_cache(config=self.base.backbone.config, ddp_cache_data=states)

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        # Admission applies to the complete original input before prefix removal.
        encoded = [self.base.encode_request(request) for request in requests]
        if not all(self.admits(row) for row in encoded):
            return self.base.score_many(requests)
        prefix_length = len(self.prefix)
        batch_size = len(encoded)
        length = max(len(row.ids) for row in encoded) - prefix_length
        width = max(len(row.candidate_positions) for row in encoded)
        ids = torch.full((batch_size, length), self.base.pad_id, dtype=torch.long)
        endpoints = torch.zeros((batch_size, width), dtype=torch.long)
        for index, row in enumerate(encoded):
            suffix = row.ids[prefix_length:]
            ids[index, : len(suffix)] = torch.tensor(suffix, dtype=torch.long)
            endpoints[index, : len(row.candidate_positions)] = torch.tensor(
                [position - prefix_length for position in row.candidate_positions]
            )
        rows = torch.arange(batch_size)
        positions = torch.tensor(
            [row.query_position - prefix_length for row in encoded]
        )
        with torch.inference_mode():
            # Keep maintained offset mask/RoPE: rectangular is_causal alone is wrong.
            output = cast(
                BaseModelOutputWithPast,
                self.base.backbone(
                    input_ids=ids, past_key_values=self.fork(batch_size), use_cache=True
                ),
            )
            hidden = output.last_hidden_state
            if hidden is None:
                raise ValueError("Kai backbone did not return suffix hidden states")
            logits = cast(
                torch.Tensor,
                self.base.head(
                    hidden[rows[:, None], endpoints], hidden[rows, positions]
                ),
            )
            values = cast(list[list[float]], logits.float().cpu().tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]
