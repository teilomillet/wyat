# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Local MLX execution of Wyat's native typed artifact and unmerged adapters.

The native task encoder, artifact contract and readouts are shared with CPU/CUDA.
This module adds a runtime without changing the trained architecture or weights.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, Protocol, cast

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ConcatenateKVCache
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from wyat import tasks
from wyat.backends.wyat.artifact import normalize_config, positive_integer
from wyat.backends.wyat.encoding import TokenizerProtocol, load_tokenizer
from wyat.backends.wyat.mlx import MLXCandidateHead, _weights
from wyat.core import ModelInputError, Request
from wyat.input_rendering import render_record, validate_input_rendering_policy
from wyat.wire import JsonObject, read_object

from .native_artifact import budget_groups, validate_artifact

type _MLXKVPair = tuple[mx.array, mx.array]


class _MLXPrefixCache(Protocol):
    @property
    def state(self) -> _MLXKVPair: ...

    @state.setter
    def state(self, value: _MLXKVPair) -> None: ...


_new_mlx_cache = cast(Callable[[], _MLXPrefixCache], ConcatenateKVCache)


def _lcp_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    matched = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        matched += 1
    return matched


def _common_prefix(encoded: Sequence[tasks.EncodedQuestion]) -> tuple[int, ...]:
    limit = min(row.candidate_positions[0] for row in encoded)
    if limit <= 0:
        return ()
    first = encoded[0].ids[:limit]
    if len(encoded) == 1:
        return first
    matched = 0
    for index, token in enumerate(first):
        if all(row.ids[index] == token for row in encoded):
            matched = index + 1
        else:
            break
    return first[:matched]


def _request_record(request: Request) -> dict[str, object]:
    if request.task not in ("choice", "score", "noul"):
        raise ModelInputError(
            f"native model does not support task {request.task!r}",
            reason="unsupported-task",
        )
    if not 2 <= len(request.candidates) <= 255:
        raise ModelInputError(
            "native tasks require 2 to 255 candidates",
            reason="candidate-count",
        )
    if request.task == "choice":
        question: dict[str, object] = {
            "type": "choice",
            "instructions": request.query,
            "criteria": {
                candidate.id: candidate.text for candidate in request.candidates
            },
        }
    elif request.task == "score":
        question = {
            "type": "score",
            "instructions": request.query,
            "criteria": [candidate.text for candidate in request.candidates],
        }
    else:
        if len(request.candidates) != 2:
            raise ModelInputError(
                "noul task requires 2 candidates",
                reason="candidate-count",
            )
        if {candidate.id for candidate in request.candidates} != {"false", "true"}:
            raise ModelInputError(
                "noul task requires 'false' and 'true' candidate IDs",
                reason="input-segment",
            )
        question = {
            "type": "noul",
            "instructions": request.query,
            "criteria": {
                candidate.id: candidate.text for candidate in request.candidates
            },
        }
    return {"state": request.context, "questions": {"decision": question}}


class _RoPE(Module):  # type: ignore[misc, unused-ignore]
    """Native FP32 rotary arithmetic, with one shared bounded position table.

    Materialize the reference's frequency and product steps: fused fast RoPE
    changes a near-tied complete ranking in the trained native stress fixture.
    """

    def __init__(self, dims: int, base: float) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        if type(dims) is not int or dims < 2 or dims % 2:
            raise ValueError("native rotary width must be a positive even integer")
        if isinstance(base, bool) or not math.isfinite(base) or base <= 0:
            raise ValueError("native rotary base must be finite and positive")
        self.dims = dims
        self.base = base
        self.scale = 1.0
        self.traditional = False
        self._tables: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}
        exponents = mx.arange(0, dims, 2, dtype=mx.float32) / dims
        mx.eval(exponents)
        powers = mx.power(mx.array(base, dtype=mx.float32), exponents)
        mx.eval(powers)
        self._inverse = mx.reciprocal(powers)
        mx.eval(self._inverse)

    def __call__(self, value: mx.array, offset: int = 0) -> mx.array:
        if (
            type(offset) is not int
            or offset < 0
            or value.ndim < 2
            or value.shape[-1] != self.dims
            or value.shape[-2] < 1
            or value.dtype != mx.float32
        ):
            raise ValueError("native rotary input requires full FP32 width and offset")
        width = value.shape[-2]
        key = (width, offset)
        if key not in self._tables:
            # One table per scorer, shared across all layers, replaced on width
            # changes. Normal inference holds the scorer's lock throughout.
            self._tables.clear()
            positions = mx.arange(offset, offset + width, dtype=mx.float32)
            angles = positions[:, None] * self._inverse[None, :]
            mx.eval(angles)
            angles = mx.concatenate([angles, angles], axis=-1)
            cosine, sine = mx.cos(angles), mx.sin(angles)
            mx.eval(cosine, sine)
            self._tables[key] = (cosine, sine)
        cosine, sine = self._tables[key]
        half = self.dims // 2
        rotated = mx.concatenate([-value[..., half:], value[..., :half]], axis=-1)
        original_product = value * cosine
        rotated_product = rotated * sine
        mx.eval(original_product, rotated_product)
        return original_product + rotated_product


class _Adapter(Module):  # type: ignore[misc, unused-ignore]
    """Keep both FP32 factors and the original two matmuls separate."""

    def __init__(self, base: Module, down: mx.array, up: mx.array) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.base = base
        self.down = down
        self.up = up

    def __call__(self, value: mx.array) -> mx.array:
        return cast(
            mx.array,
            self.base(value)
            + 2 * ((value.astype(mx.float32) @ self.down.T) @ self.up.T),
        )


def _config(root: Path) -> JsonObject:
    config = normalize_config(read_object(root / "backbone/config.json"))
    if config.get("attention_bias", False) is not False:
        raise ValueError("native MLX requires bias-free attention projections")
    if config.get("attention_dropout", 0.0) != 0.0:
        raise ValueError("native MLX requires zero attention dropout")
    dimensions = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    )
    sizes = {name: positive_integer(config.get(name), name) for name in dimensions}
    if sizes["num_attention_heads"] % sizes["num_key_value_heads"]:
        raise ValueError("attention heads must be divisible by KV heads")
    layers = config.get("layer_types")
    if (
        layers is not None
        and len(cast(list[object], layers)) != sizes["num_hidden_layers"]
    ):
        raise ValueError("attention layer inventory differs from num_hidden_layers")
    return config


def _install_adapters(backbone: Qwen3Model, state: dict[str, mx.array]) -> None:
    expected: dict[str, tuple[int, int]] = {}
    for index, layer in enumerate(backbone.layers):
        for name in ("q_proj", "v_proj"):
            base = getattr(layer.self_attn, name)
            output_size, input_size = base.weight.shape
            prefix = f"layers.{index}.self_attn.{name}"
            expected[prefix + ".down"] = (16, input_size)
            expected[prefix + ".up"] = (output_size, 16)
    if set(state) != set(expected):
        raise ValueError("native adapter tensor inventory differs")
    for name, shape in expected.items():
        if state[name].dtype != mx.float32 or state[name].shape != shape:
            raise ValueError("invalid native adapter tensor: " + name)
    for index, layer in enumerate(backbone.layers):
        for name in ("q_proj", "v_proj"):
            prefix = f"layers.{index}.self_attn.{name}"
            setattr(
                layer.self_attn,
                name,
                _Adapter(
                    getattr(layer.self_attn, name),
                    state[prefix + ".down"],
                    state[prefix + ".up"],
                ),
            )


class NativeMLXScorer:
    """Independent causal question rows with native Choice, Score and Noul."""

    def __init__(
        self,
        backbone: Qwen3Model,
        head: MLXCandidateHead,
        tokenizer: TokenizerProtocol,
        manifest: dict[str, Any],
        *,
        device: mx.Device,
        max_length: int,
        metal_enabled: bool = False,
        responsive: bool = True,
        prefix_cache_size: int = 16,
    ) -> None:
        if type(responsive) is not bool:
            raise ValueError("responsive must be a boolean")
        if type(prefix_cache_size) is not int or prefix_cache_size < 0:
            raise ValueError("prefix_cache_size must be a nonnegative integer")
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.input_rendering = validate_input_rendering_policy(
            manifest.get("input_rendering")
        )
        self.device = device
        self.max_length = max_length
        self.max_batch_tokens: int = manifest["max_batch_tokens"]
        self.model_id: str = manifest["model_id"]
        self.metal_enabled = metal_enabled
        self.responsive = responsive
        self.prefix_cache_size = prefix_cache_size
        self._prefix_cache: OrderedDict[tuple[int, ...], tuple[_MLXKVPair, ...]] = (
            OrderedDict()
        )
        self._prefix_hits = 0
        self._prefix_misses = 0
        self._inference_lock = Lock()
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if type(pad) is not int or pad < 0:
            raise ValueError("tokenizer requires a nonnegative pad or EOS token")
        self.pad_id = pad

    @property
    def prefix_cache_currsize(self) -> int:
        return len(self._prefix_cache)

    @property
    def prefix_cache_hits(self) -> int:
        return self._prefix_hits

    @property
    def prefix_cache_misses(self) -> int:
        return self._prefix_misses

    def clear_prefix_cache(self) -> None:
        """Clear all retained prefix KV states and reset cache counters."""
        with self._inference_lock:
            self._prefix_cache.clear()
            self._prefix_hits = 0
            self._prefix_misses = 0

    def encode_record(self, record: object) -> tuple[tasks.EncodedQuestion, ...]:
        return tasks.encode_record(
            self.tokenizer,
            render_record(record, self.input_rendering),
            max_length=self.max_length,
        )

    def encode_request(self, request: Request) -> tasks.EncodedQuestion:
        record = _request_record(request)
        try:
            return self.encode_record(record)[0]
        except ValueError as error:
            message = str(error)
            if "max_length" in message or "byte limit" in message:
                raise ModelInputError(message, reason="input-length") from error
            raise ModelInputError(message, reason="input-segment") from error

    def _hidden(self, ids: mx.array) -> mx.array:
        if not self.responsive or self.device.type != mx.gpu:
            return cast(mx.array, self.backbone(ids))
        # Match maintained Qwen3 exactly, with synchronous execution boundaries.
        # Short idle gaps let other applications submit GPU work; they are not
        # a hard QoS guarantee, particularly for one long attention kernel.
        hidden = self.backbone.embed_tokens(ids)
        mask = create_attention_mask(hidden, None)
        count = len(self.backbone.layers)
        for index, layer in enumerate(self.backbone.layers, 1):
            hidden = layer(hidden, mask, None)
            if index % 4 == 0 and index < count:
                mx.eval(hidden)
                sleep(0.001)
        return cast(mx.array, self.backbone.norm(hidden))

    def _hidden_cached(self, ids: mx.array, caches: list[_MLXPrefixCache]) -> mx.array:
        if not self.responsive or self.device.type != mx.gpu:
            return cast(mx.array, self.backbone(ids, cache=caches))
        hidden = self.backbone.embed_tokens(ids)
        mask = create_attention_mask(hidden, caches[0])
        count = len(self.backbone.layers)
        for index, (layer, cache) in enumerate(
            zip(self.backbone.layers, caches, strict=True), 1
        ):
            hidden = layer(hidden, mask, cache)
            if index % 4 == 0 and index < count:
                mx.eval(hidden)
                sleep(0.001)
        return cast(mx.array, self.backbone.norm(hidden))

    def _can_use_prefix_cache(self) -> bool:
        return (
            self.prefix_cache_size > 0
            and isinstance(self.backbone, Qwen3Model)
            and "_hidden" not in self.__dict__
            and "__call__" in Qwen3Model.__dict__
            and type(self.backbone).__call__ is Qwen3Model.__call__
        )

    def _match_cached_prefix(
        self, target: tuple[int, ...]
    ) -> tuple[tuple[int, ...] | None, int, tuple[_MLXKVPair, ...] | None]:
        if target in self._prefix_cache:
            self._prefix_cache.move_to_end(target)
            return target, len(target), self._prefix_cache[target]
        best_key: tuple[int, ...] | None = None
        best_len = 0
        best_states: tuple[_MLXKVPair, ...] | None = None
        for key, states in self._prefix_cache.items():
            common = _lcp_length(key, target)
            if common > best_len:
                best_key = key
                best_len = common
                best_states = states
        if best_key is not None and best_len > 0:
            self._prefix_cache.move_to_end(best_key)
            return best_key, best_len, best_states
        return None, 0, None

    def _store_prefix(
        self,
        key: tuple[int, ...],
        states: tuple[_MLXKVPair, ...],
        *,
        replace_key: tuple[int, ...] | None = None,
    ) -> None:
        if not key or self.prefix_cache_size <= 0:
            return
        if (
            replace_key is not None
            and replace_key != key
            and replace_key in self._prefix_cache
            and key[: len(replace_key)] == replace_key
        ):
            del self._prefix_cache[replace_key]
        self._prefix_cache[key] = states
        self._prefix_cache.move_to_end(key)
        while len(self._prefix_cache) > self.prefix_cache_size:
            self._prefix_cache.popitem(last=False)

    @staticmethod
    def _extract_states(
        caches: list[_MLXPrefixCache], length: int
    ) -> tuple[_MLXKVPair, ...]:
        states = tuple(
            (
                mx.contiguous(cache.state[0][:, :, :length, :]),
                mx.contiguous(cache.state[1][:, :, :length, :]),
            )
            for cache in caches
        )
        mx.eval(states)
        return states

    @staticmethod
    def _fork_cache(
        states: tuple[_MLXKVPair, ...], batch_size: int, length: int
    ) -> list[_MLXPrefixCache]:
        caches: list[_MLXPrefixCache] = []
        for keys, values in states:
            cache = _new_mlx_cache()
            cache.state = (
                mx.broadcast_to(
                    keys[:, :, :length, :],
                    (batch_size, keys.shape[1], length, keys.shape[3]),
                ),
                mx.broadcast_to(
                    values[:, :, :length, :],
                    (batch_size, values.shape[1], length, values.shape[3]),
                ),
            )
            caches.append(cache)
        return caches

    def _logits(self, encoded: Sequence[tasks.EncodedQuestion]) -> list[list[float]]:
        if not encoded:
            return []
        length = max(len(row.ids) for row in encoded)
        if length > self.max_length:
            raise ValueError("complete question exceeds max_length; no truncation")
        if length * len(encoded) > self.max_batch_tokens:
            raise ValueError("complete padded question batch exceeds explicit budget")
        target_prefix = _common_prefix(encoded) if self._can_use_prefix_cache() else ()
        if not target_prefix:
            # Retain the reference's exact width and row order: changing GEMM shapes
            # is a separate numerical/performance experiment, even for causal padding.
            tokens = [
                list(row.ids) + [self.pad_id] * (length - len(row.ids))
                for row in encoded
            ]
            choices = max(len(row.candidate_positions) for row in encoded)
            positions = [
                list(row.candidate_positions)
                + [row.candidate_positions[0]]
                * (choices - len(row.candidate_positions))
                for row in encoded
            ]
            with self._inference_lock, mx.stream(self.device):
                ids = mx.array(tokens, dtype=mx.int32)
                endpoints = mx.array(positions, dtype=mx.int32)
                queries = mx.array(
                    [row.query_position for row in encoded], dtype=mx.int32
                )
                hidden = self._hidden(ids)
                rows = mx.arange(len(encoded))
                logits = self.head(
                    hidden[rows[:, None], endpoints], hidden[rows, queries]
                )
                mx.eval(logits)
                values = cast(list[list[float]], logits.tolist())
            return [
                row[: len(item.candidate_positions)]
                for row, item in zip(values, encoded, strict=True)
            ]

        with self._inference_lock, mx.stream(self.device):
            matched_key, matched_len, matched_states = self._match_cached_prefix(
                target_prefix
            )
            target_len = len(target_prefix)
            if len(encoded) == 1:
                row = encoded[0]
                if matched_states is not None and matched_len > 0:
                    self._prefix_hits += 1
                    caches = self._fork_cache(matched_states, 1, matched_len)
                    suffix_ids = mx.array([list(row.ids[matched_len:])], dtype=mx.int32)
                    hidden = self._hidden_cached(suffix_ids, caches)
                    offset = matched_len
                    if matched_len < target_len:
                        self._store_prefix(
                            target_prefix,
                            self._extract_states(caches, target_len),
                            replace_key=matched_key,
                        )
                else:
                    self._prefix_misses += 1
                    caches = [_new_mlx_cache() for _ in self.backbone.layers]
                    full_ids = mx.array([list(row.ids)], dtype=mx.int32)
                    hidden = self._hidden_cached(full_ids, caches)
                    offset = 0
                    self._store_prefix(
                        target_prefix, self._extract_states(caches, target_len)
                    )
                endpoints = mx.array(
                    [[pos - offset for pos in row.candidate_positions]],
                    dtype=mx.int32,
                )
                queries = mx.array([row.query_position - offset], dtype=mx.int32)
                rows = mx.arange(1)
                logits = self.head(
                    hidden[rows[:, None], endpoints], hidden[rows, queries]
                )
                mx.eval(logits)
                values = cast(list[list[float]], logits.tolist())
                return [values[0][: len(row.candidate_positions)]]

            if matched_states is not None and matched_len == target_len:
                self._prefix_hits += 1
                prefix_states = matched_states
            elif matched_states is not None and matched_len > 0:
                self._prefix_hits += 1
                caches = self._fork_cache(matched_states, 1, matched_len)
                delta_ids = mx.array(
                    [list(target_prefix[matched_len:])], dtype=mx.int32
                )
                self._hidden_cached(delta_ids, caches)
                prefix_states = self._extract_states(caches, target_len)
                self._store_prefix(
                    target_prefix, prefix_states, replace_key=matched_key
                )
            else:
                self._prefix_misses += 1
                caches = [_new_mlx_cache() for _ in self.backbone.layers]
                prefix_ids = mx.array([list(target_prefix)], dtype=mx.int32)
                self._hidden_cached(prefix_ids, caches)
                prefix_states = self._extract_states(caches, target_len)
                self._store_prefix(target_prefix, prefix_states)

            batch_caches = self._fork_cache(prefix_states, len(encoded), target_len)
            suffix_len = max(len(row.ids) - target_len for row in encoded)
            tokens = [
                list(row.ids[target_len:])
                + [self.pad_id] * (suffix_len - (len(row.ids) - target_len))
                for row in encoded
            ]
            choices = max(len(row.candidate_positions) for row in encoded)
            positions = [
                [pos - target_len for pos in row.candidate_positions]
                + [row.candidate_positions[0] - target_len]
                * (choices - len(row.candidate_positions))
                for row in encoded
            ]
            ids = mx.array(tokens, dtype=mx.int32)
            endpoints = mx.array(positions, dtype=mx.int32)
            queries = mx.array(
                [row.query_position - target_len for row in encoded], dtype=mx.int32
            )
            hidden = self._hidden_cached(ids, batch_caches)
            rows = mx.arange(len(encoded))
            logits = self.head(hidden[rows[:, None], endpoints], hidden[rows, queries])
            mx.eval(logits)
            values = cast(list[list[float]], logits.tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        """Score one batch of public Requests in request and candidate order."""
        if not requests:
            return []
        encoded = [self.encode_request(request) for request in requests]
        return self._refusing_logits(encoded)

    def _refusing_logits(
        self, encoded: Sequence[tasks.EncodedQuestion]
    ) -> list[list[float]]:
        try:
            return self._logits(encoded)
        except ValueError as error:
            if isinstance(error, ModelInputError):
                raise
            message = str(error)
            if "max_length" in message or "budget" in message:
                raise ModelInputError(message, reason="input-length") from error
            raise

    def _records_logits(
        self, records: Sequence[object]
    ) -> list[tuple[tuple[tasks.EncodedQuestion, ...], list[list[float]]]]:
        """Encode every record, then score each record in budgeted groups.

        Every record is encoded before any scoring. A malformed record keeps its
        ValueError. Encoder and budget refusals raise ModelInputError with the
        same reasons as ``decide``.
        """
        encoded_records = []
        for record in records:
            rendered = render_record(record, self.input_rendering)
            tasks.public_questions(rendered)
            try:
                encoded_records.append(self.encode_record(record))
            except ValueError as error:
                message = str(error)
                if "max_length" in message or "byte limit" in message:
                    raise ModelInputError(message, reason="input-length") from error
                raise ModelInputError(message, reason="input-segment") from error
        # Groups never cross records: a record's questions share its state
        # prefix, which batched prefix reuse computes once. Mixing records
        # shrinks that prefix to a few tokens (measured 11x slower on 4 records
        # of 58 questions with a long review).
        result = []
        for encoded in encoded_records:
            scored = []
            for group in budget_groups(encoded, self.max_batch_tokens):
                scored.extend(self._refusing_logits(group))
            result.append((encoded, scored))
        return result

    def logits(self, record: object) -> list[list[float]]:
        """One complete ordered FP32 logit vector per public question."""
        return self._records_logits((record,))[0][1]

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        encoded, rows = self._records_logits((record,))[0]
        return {
            row.question.id: tasks.interpret(row.question, logits)
            for row, logits in zip(encoded, rows, strict=True)
        }


def load_native_mlx(
    directory: str | Path,
    *,
    device: str = "gpu",
    max_length: int | None = None,
    metal: bool = False,
    responsive: bool = True,
    prefix_cache_size: int = 16,
) -> NativeMLXScorer:
    """Load a validated native artifact locally; optional Metal is explicit.

    Ordinary MLX is the baseline. Metal uses the existing qualified projection
    owner and falls back on unsupported runtimes/devices without altering math.
    Adapters and the candidate head always retain their separate FP32 weights.
    """
    if device not in {"cpu", "gpu"}:
        raise ValueError("MLX device must be cpu or gpu")
    if type(metal) is not bool:
        raise ValueError("metal must be a boolean")
    if type(responsive) is not bool:
        raise ValueError("responsive must be a boolean")
    if type(prefix_cache_size) is not int or prefix_cache_size < 0:
        raise ValueError("prefix_cache_size must be a nonnegative integer")
    if device == "gpu" and not mx.metal.is_available():
        raise ValueError("MLX GPU inference requires available Apple Metal")
    root = Path(directory).resolve()
    manifest = validate_artifact(root)
    maximum = (
        manifest["max_length"]
        if max_length is None
        else positive_integer(max_length, "max_length")
    )
    if maximum > manifest["max_length"]:
        raise ValueError("max_length exceeds the artifact contract")
    config = _config(root)
    if maximum > cast(int, config["max_position_embeddings"]):
        raise ValueError("max_length exceeds the backbone context")
    arguments = cast(Callable[[JsonObject], ModelArgs], ModelArgs.from_dict)
    args = arguments(config)
    target = mx.Device(mx.gpu if device == "gpu" else mx.cpu)
    enabled = False
    with mx.stream(target):
        backbone = Qwen3Model(args)
        head = MLXCandidateHead(args.hidden_size, manifest["head_dim"])
        operation = None
        prepare_weight = None
        if metal and device == "gpu":
            from wyat.backends.wyat.metal import (
                _checkpoint_weights,
                _prepare,
                optimize_projections,
            )

            operation = _prepare(device=target, compiled=False)
            if operation is not None:
                prepare_weight = _checkpoint_weights(backbone)
        base = _weights(
            root / "backbone/model.safetensors",
            head=False,
            projection_weight=prepare_weight,
        )
        backbone.load_weights(list(base.items()), strict=True)
        mx.eval(base)
        del base
        if operation is not None:
            enabled = optimize_projections(
                backbone, device=target, compiled=False, operation=operation
            )
        # Install only base projections, then wrap q/v; adapter matrices can
        # never enter the shape-qualified base-projection optimization.
        adapters = _weights(root / "native-adapters.safetensors", head=True)
        _install_adapters(backbone, adapters)
        head_state = _weights(root / "decision_head.safetensors", head=True)
        head.load_weights(list(head_state.items()), strict=True)
        # Share positional constants without changing any trained tensor. This
        # applies only to the native scorer; maintained Wyat/CUDA paths are intact.
        rope = _RoPE(args.head_dim, args.rope_theta)
        for layer in backbone.layers:
            layer.self_attn.rope = rope
        backbone.eval()
        head.eval()
        mx.eval(adapters, head_state)
    return NativeMLXScorer(
        backbone,
        head,
        load_tokenizer(root),
        manifest,
        device=target,
        max_length=maximum,
        metal_enabled=enabled,
        responsive=responsive,
        prefix_cache_size=prefix_cache_size,
    )
