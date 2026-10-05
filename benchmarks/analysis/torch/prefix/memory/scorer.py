"""One common prefix path with retained or one-use maintained cache owners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from benchmarks.analysis.torch.prefix.cache import PrefixScorer
from benchmarks.analysis.torch.prefix.lifetime import OneUseCache
from wyat.backends.kai.torch import KaiScorer
from wyat.core import Request

type Arm = Literal["retained", "one-use"]
type KVState = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class StorageObservation:
    layer: int | None
    cache_owned_bytes: int
    returned_logical_bytes: int
    combined_unique_storage_bytes: int


def storage_bytes(tensors: Sequence[torch.Tensor]) -> int:
    """Deduplicate actual underlying CPU allocations, including aliased views."""
    allocations = {
        (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
        for tensor in tensors
    }
    return sum(size for _, size in allocations)


def observe_storage(
    cache: DynamicCache,
    seeds: tuple[KVState, ...],
    returned: KVState | None,
    layer: int | None,
) -> StorageObservation:
    owned: list[torch.Tensor] = []
    for state in cache.layers:
        if not isinstance(state, DynamicLayer):
            raise ValueError("Observer requires maintained DynamicLayer")
        owned.extend(
            tensor for tensor in (state.keys, state.values) if tensor is not None
        )
    current = list(returned or ())
    return StorageObservation(
        layer,
        storage_bytes(owned),
        sum(tensor.numel() * tensor.element_size() for tensor in current),
        storage_bytes(
            [*owned, *current, *(tensor for pair in seeds for tensor in pair)]
        ),
    )


type Observer = Callable[[StorageObservation], None]


class ObservedRetained(DynamicCache):
    """Intrusive memory-cycle owner only; ordinary calls use stock DynamicCache."""

    def __init__(
        self,
        config: Qwen3Config,
        states: tuple[KVState, ...],
        seeds: tuple[KVState, ...],
        observer: Observer,
    ) -> None:
        initialize = cast(Callable[..., None], super().__init__)
        initialize(config=config, ddp_cache_data=states)
        self.seeds, self.observer = seeds, observer
        observer(observe_storage(self, seeds, None, None))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> KVState:
        update = cast(Callable[..., KVState], super().update)
        returned = update(key_states, value_states, layer_idx, *args, **kwargs)
        self.observer(observe_storage(self, self.seeds, returned, layer_idx))
        return returned


class ObservedOneUse(OneUseCache):
    """Observe after maintained concatenation and one-use reference release."""

    def __init__(
        self,
        config: Qwen3Config,
        states: tuple[KVState, ...],
        seeds: tuple[KVState, ...],
        observer: Observer,
    ) -> None:
        super().__init__(config, states)
        self.seeds, self.observer = seeds, observer
        observer(observe_storage(self, seeds, None, None))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: object,
        **kwargs: object,
    ) -> KVState:
        returned = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        self.observer(observe_storage(self, self.seeds, returned, layer_idx))
        return returned


class MemoryScorer(PrefixScorer):
    """Both arms suppress output cache, but keep their private owner through head.

    Prefix validation, snapshots and fallback reuse the frozen prefix scorer.
    Only cache factory/layer reference lifetime differs; neither returns a cache.
    Weights/config/device/encoder stay fixed and eval for this resource lifetime.
    """

    def __init__(self, base: KaiScorer, prefix: tuple[int, ...], arm: Arm) -> None:
        if arm not in ("retained", "one-use"):
            raise ValueError("Cache arm must be retained or one-use")
        super().__init__(base, prefix)
        self.arm = arm

    def fork(self, batch_size: int) -> DynamicCache:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Cache batch size must be a positive integer")
        if self.arm == "retained":
            return super().fork(batch_size)
        states = tuple(
            (keys.expand(batch_size, -1, -1, -1), values.expand(batch_size, -1, -1, -1))
            for keys, values in self._states
        )
        return OneUseCache(self.base.backbone.config, states)

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        return self.score_observed(requests)

    def score_observed(
        self,
        requests: Sequence[Request],
        observer: Observer | None = None,
    ) -> list[list[float]]:
        """Observer includes only cache/returned KV/immutable seeds, not all tensors.

        It runs only in a separate intrusive memory cycle. Returned KV remains
        counted after OneUse reset; callbacks retain numbers, never tensors.
        """
        if not requests:
            return []
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
            if observer is None:
                owner = self.fork(batch_size)
            else:
                states = tuple(
                    (
                        keys.expand(batch_size, -1, -1, -1),
                        values.expand(batch_size, -1, -1, -1),
                    )
                    for keys, values in self._states
                )
                factory = ObservedRetained if self.arm == "retained" else ObservedOneUse
                owner = factory(
                    self.base.backbone.config, states, self._states, observer
                )
            output = cast(
                BaseModelOutputWithPast,
                self.base.backbone(
                    input_ids=ids, past_key_values=owner, use_cache=False
                ),
            )
            if output.past_key_values is not None or (
                isinstance(owner, OneUseCache) and not owner.consumed
            ):
                raise ValueError(
                    "Private cache must be consumed without returned state"
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
