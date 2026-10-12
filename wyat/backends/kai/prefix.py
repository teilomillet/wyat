"""Opt-in eager immutable prefix reuse for the MLX Kai scorer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import mlx.core as mx
from mlx_lm.models.cache import ConcatenateKVCache

from wyat.core import Request

from .encoding import EncodedRequest
from .mlx import MLXKaiScorer, _pack
from .readout import EndpointCache, cached_endpoint_hidden

type KVState = tuple[mx.array, mx.array]


class PrefixCache(EndpointCache, Protocol):
    @property
    def state(self) -> KVState: ...

    @state.setter
    def state(self, value: KVState) -> None: ...


type CachedForward = Callable[[mx.array, list[PrefixCache]], mx.array]
# The maintained cache/model APIs are untyped; narrow only those boundaries.
new_cache = cast(Callable[[], PrefixCache], ConcatenateKVCache)


class PrefixScorer:
    """One immutable prefix for fixed weights/device; fresh cache owners per call.

    Each fresh cache concatenates the suffix once at its exact logical length;
    it never writes into the broadcast immutable prefix views or reserves
    decoding capacity. A mixed or overlapping-readout batch falls back intact
    to the original scorer.
    Weights, device and encoder must remain fixed for this resource lifetime.
    The admitted path is eager; the final block visits only actual readouts.
    """

    def __init__(self, base: MLXKaiScorer, prefix: tuple[int, ...]) -> None:
        if not isinstance(base, MLXKaiScorer):
            raise TypeError("Prefix requires a native MLXKaiScorer base")
        if type(prefix) is not tuple:
            raise TypeError("Prefix must be an exact immutable token tuple")
        if base.compiled:
            raise ValueError("Prefix scorer requires an eager, uncompiled base")
        if base.backbone.args.rope_scaling is not None:
            raise ValueError("Prefix scorer requires unscaled native RoPE")
        if (
            not prefix
            or len(prefix) >= base.encoder.max_length
            or any(
                type(token) is not int or not 0 <= token < base.backbone.vocab_size
                for token in prefix
            )
        ):
            raise ValueError(
                "Prefix requires nonempty valid token IDs below max_length"
            )
        self.base = base
        self.model_id = base.model_id
        self._prefix = prefix
        self._forward = cast(CachedForward, base.backbone)
        with mx.stream(base.device):
            caches = [new_cache() for _ in base.backbone.layers]
            self._forward(mx.array([list(prefix)], dtype=mx.int32), caches)
            self._states = tuple(
                (mx.contiguous(keys), mx.contiguous(values))
                for keys, values in (cache.state for cache in caches)
            )
            if any(
                cache.offset != len(prefix)
                or keys.shape[0] != 1
                or keys.shape[2] != len(prefix)
                or values.shape[:3] != keys.shape[:3]
                for cache, (keys, values) in zip(caches, self._states, strict=True)
            ):
                raise ValueError("Maintained cache did not retain the exact prefix")
            mx.eval(self._states)

    @property
    def prefix(self) -> tuple[int, ...]:
        return self._prefix

    @property
    def prefix_state_bytes(self) -> int:
        return sum(keys.nbytes + values.nbytes for keys, values in self._states)

    def _admits(self, row: EncodedRequest) -> bool:
        length = len(self.prefix)
        return (
            len(row.ids) > length
            and row.ids[:length] == self.prefix
            and row.query_position >= length
            and all(position >= length for position in row.candidate_positions)
        )

    def _fork(self, batch_size: int) -> list[PrefixCache]:
        caches: list[PrefixCache] = []
        for keys, values in self._states:
            cache = new_cache()
            cache.state = (
                mx.broadcast_to(keys, (batch_size, *keys.shape[1:])),
                mx.broadcast_to(values, (batch_size, *values.shape[1:])),
            )
            caches.append(cache)
        return caches

    def _logits(
        self,
        ids: mx.array,
        endpoints: mx.array,
        query_positions: mx.array,
        caches: list[PrefixCache],
    ) -> mx.array:
        positions = mx.concatenate([endpoints, query_positions[:, None]], axis=1)
        hidden = cached_endpoint_hidden(self.base.backbone, ids, positions, caches)
        return self.base.head(hidden[:, :-1], hidden[:, -1])

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        encoded = [self.base.encode_request(request) for request in requests]
        if not all(self._admits(row) for row in encoded):
            return self.base.score_many(requests)
        prefix_length = len(self.prefix)
        with mx.stream(self.base.device):
            ids, endpoints, positions = _pack(
                encoded, self.base.encoder.pad_id, prefix_length=prefix_length
            )
            logits = self._logits(ids, endpoints, positions, self._fork(len(encoded)))
            mx.eval(logits)
            values = cast(list[list[float]], logits.tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]
