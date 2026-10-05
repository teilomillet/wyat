"""Private one-pass CPU cache feasibility; no generation or memory claim."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache, DynamicLayer

type KVState = tuple[torch.Tensor, torch.Tensor]


class OneUseCache(DynamicCache):
    """Seed through maintained layers, then release each updated cache reference.

    Only an eager, eval-mode CPU model under inference_mode may consume this
    object, once, in layer order. Qwen3 must prepare mask/positions before updates.
    The caller supplies use_cache=False and discards this consumed cache. This
    deliberately does not implement the reusable generation-cache contract.
    Returned tensors own their storage until attention releases them; reset only
    drops the cache's reference and does not establish physical RSS reclamation.
    """

    def __init__(self, config: Qwen3Config, states: tuple[KVState, ...]) -> None:
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("One-use cache requires synchronous CPU inference_mode")
        if (
            config.model_type != "qwen3"
            or config._attn_implementation != "sdpa"
            or config.layer_types != ["full_attention"] * config.num_hidden_layers
            or not isinstance(config.rope_parameters, dict)
            or config.rope_parameters.get("rope_type", "default") != "default"
        ):
            raise ValueError("One-use cache requires full Qwen3 SDPA/default RoPE")
        if (
            not states
            or type(states) is not tuple
            or len(states) != config.num_hidden_layers
        ):
            raise ValueError("One-use cache requires exactly one seed per layer")
        shape: tuple[int, ...] | None = None
        for keys, values in states:
            if (
                keys.ndim != 4
                or keys.shape != values.shape
                or keys.shape[0] < 1
                or keys.shape[1] != config.num_key_value_heads
                or keys.shape[2] < 1
                or keys.shape[3] != config.head_dim
                or any(
                    tensor.device.type != "cpu"
                    or tensor.dtype != torch.float32
                    or tensor.requires_grad
                    or tensor.stride(-1) != 1
                    for tensor in (keys, values)
                )
                or (shape is not None and tuple(keys.shape) != shape)
            ):
                raise ValueError("One-use seeds require equal FP32 CPU KV geometry")
            shape = tuple(keys.shape)
        initialize = cast(Callable[..., None], super().__init__)
        initialize(config=config, ddp_cache_data=states, offloading=False)
        if any(type(layer) is not DynamicLayer for layer in self.layers):
            raise ValueError("One-use cache requires maintained DynamicLayer")
        self._next_layer = 0
        self._suffix_length: int | None = None
        self._seed_shape = cast(tuple[int, ...], shape)

    @property
    def consumed(self) -> bool:
        return self._next_layer == len(self.layers)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> KVState:
        # Reject a reused/out-of-order invocation before any layer is changed.
        if type(layer_idx) is not int or self.consumed or layer_idx != self._next_layer:
            raise RuntimeError("One-use cache requires exactly one ordered layer pass")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("One-use cache requires synchronous CPU inference_mode")
        if args or kwargs:
            raise ValueError("One-use cache accepts the maintained Qwen3 update only")
        if (
            key_states.ndim != 4
            or key_states.shape != value_states.shape
            or key_states.shape[0:2] != self._seed_shape[0:2]
            or key_states.shape[2] < 1
            or key_states.shape[3] != self._seed_shape[3]
            or (
                self._suffix_length is not None
                and key_states.shape[2] != self._suffix_length
            )
            or any(
                tensor.device.type != "cpu"
                or tensor.dtype != torch.float32
                or tensor.requires_grad
                or tensor.stride(-1) != 1
                for tensor in (key_states, value_states)
            )
        ):
            raise ValueError("One-use update requires consistent FP32 CPU suffix KV")
        maintained_update = cast(Callable[..., KVState], super().update)
        returned = maintained_update(key_states, value_states, layer_idx)
        # DynamicLayer.reset drops references, without zeroing returned tensors.
        self.layers[layer_idx].reset()
        self._suffix_length = key_states.shape[2]
        self._next_layer += 1
        return returned
