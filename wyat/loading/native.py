# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Local typed Qwen inference and the canonical full-artifact contract.

Axes writes this format; Wyat owns validation, execution and typed readouts.
Adapters remain separate: merging would change the trained floating-point math.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicCache, DynamicLayer

from wyat import tasks
from wyat.backends.wyat.encoding import load_tokenizer
from wyat.backends.wyat.torch import CandidateHead
from wyat.core import ModelInputError, Request
from wyat.input_rendering import render_record, validate_input_rendering_policy

from .native_artifact import (
    DOCUMENTATION_FILES as DOCUMENTATION_FILES,
)
from .native_artifact import (
    DOWNLOAD_METADATA as DOWNLOAD_METADATA,
)
from .native_artifact import (
    FORMAT as FORMAT,
)
from .native_artifact import (
    MANIFEST as MANIFEST,
)
from .native_artifact import (
    TOKENIZER_FILES as TOKENIZER_FILES,
)
from .native_artifact import (
    _positive_integer,
    budget_groups,
)
from .native_artifact import (
    artifact_contract as artifact_contract,
)
from .native_artifact import (
    file_sha256 as file_sha256,
)
from .native_artifact import (
    validate_artifact as validate_artifact,
)

type _TorchKVState = tuple[torch.Tensor, torch.Tensor]


class _CacheConstructor(Protocol):
    def __call__(
        self,
        *,
        config: Qwen3Config,
        ddp_cache_data: tuple[_TorchKVState, ...] | None = None,
    ) -> DynamicCache: ...


_new_cache = cast(_CacheConstructor, DynamicCache)


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


def _finite_state(state: dict[str, torch.Tensor]) -> None:
    for name, value in state.items():
        if not value.is_floating_point():
            raise ValueError("native weights must be floating point: " + name)
        flattened = value.reshape(-1)
        for start in range(0, flattened.numel(), 1 << 20):
            if not bool(torch.isfinite(flattened[start : start + (1 << 20)]).all()):
                raise ValueError("nonfinite native weight: " + name)


class _Adapter(nn.Module):
    """Execute the exact unmerged FP32 rank-16 training projection."""

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(16, base.in_features))
        self.up = nn.Parameter(torch.empty(base.out_features, 16))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor,
            self.base(value)
            + 2 * F.linear(F.linear(value.float(), self.down), self.up),
        )


class NativeModel:
    """One shared predictor with Wyat's Choice, Score and Noul readouts."""

    def __init__(
        self,
        backbone: Qwen3Model,
        head: CandidateHead,
        tokenizer: Any,
        manifest: dict[str, Any],
        device: torch.device,
        max_length: int,
        *,
        prefix_cache_size: int = 16,
    ) -> None:
        if type(prefix_cache_size) is not int or prefix_cache_size < 0:
            raise ValueError("prefix_cache_size must be a nonnegative integer")
        self.backbone = backbone
        cast(nn.Module, backbone).eval()
        self.head = head.eval()
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.input_rendering = validate_input_rendering_policy(
            manifest.get("input_rendering")
        )
        self.device = device
        self.max_length = max_length
        self.max_batch_tokens = manifest["max_batch_tokens"]
        self.model_id = manifest["model_id"]
        self.prefix_cache_size = prefix_cache_size
        self._prefix_cache: OrderedDict[tuple[int, ...], tuple[_TorchKVState, ...]] = (
            OrderedDict()
        )
        self._prefix_hits = 0
        self._prefix_misses = 0
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("tokenizer requires a nonnegative pad or EOS token")

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

    def _can_use_prefix_cache(self) -> bool:
        return (
            self.prefix_cache_size > 0
            and isinstance(self.backbone, Qwen3Model)
            and "__call__" not in type(self.backbone).__dict__
            and "forward" not in self.backbone.__dict__
        )

    def _match_cached_prefix(
        self, target: tuple[int, ...]
    ) -> tuple[tuple[int, ...] | None, int, tuple[_TorchKVState, ...] | None]:
        if target in self._prefix_cache:
            self._prefix_cache.move_to_end(target)
            return target, len(target), self._prefix_cache[target]
        best_key: tuple[int, ...] | None = None
        best_len = 0
        best_states: tuple[_TorchKVState, ...] | None = None
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
        states: tuple[_TorchKVState, ...],
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
    def _extract_states(cache: DynamicCache, length: int) -> tuple[_TorchKVState, ...]:
        extracted: list[_TorchKVState] = []
        for raw_layer in cache.layers:
            layer = cast(DynamicLayer, raw_layer)
            assert layer.keys is not None and layer.values is not None
            extracted.append(
                (
                    layer.keys[:, :, :length, :].detach().clone().contiguous(),
                    layer.values[:, :, :length, :].detach().clone().contiguous(),
                )
            )
        return tuple(extracted)

    def _fork_cache(
        self,
        states: tuple[_TorchKVState, ...],
        batch_size: int,
        length: int,
    ) -> DynamicCache:
        return _new_cache(
            config=self.backbone.config,
            ddp_cache_data=tuple(
                (
                    keys[:, :, :length, :].expand(batch_size, -1, -1, -1),
                    values[:, :, :length, :].expand(batch_size, -1, -1, -1),
                )
                for keys, values in states
            ),
        )

    def _logits(self, encoded: Sequence[tasks.EncodedQuestion]) -> list[list[float]]:
        if not encoded:
            return []
        length = max(len(row.ids) for row in encoded)
        if length > self.max_length:
            raise ValueError("complete question exceeds max_length; no truncation")
        if length * len(encoded) > self.max_batch_tokens:
            raise ValueError("complete padded question batch exceeds explicit budget")
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        target_prefix = _common_prefix(encoded) if self._can_use_prefix_cache() else ()
        if not target_prefix:
            ids = torch.full(
                (len(encoded), length),
                self.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for index, row in enumerate(encoded):
                ids[index, : len(row.ids)] = torch.tensor(row.ids, device=self.device)
            with torch.inference_mode(), autocast:
                hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
                result = []
                for index, row in enumerate(encoded):
                    candidates = hidden[index : index + 1, row.candidate_positions, :]
                    query = hidden[index : index + 1, row.query_position, :]
                    result.append(
                        self.head(candidates, query)[0].float().cpu().tolist()
                    )
            return result

        with torch.inference_mode(), autocast:
            matched_key, matched_len, matched_states = self._match_cached_prefix(
                target_prefix
            )
            target_len = len(target_prefix)
            if len(encoded) == 1:
                row = encoded[0]
                if matched_states is not None and matched_len > 0:
                    self._prefix_hits += 1
                    cache = self._fork_cache(matched_states, 1, matched_len)
                    suffix = torch.tensor(
                        [row.ids[matched_len:]], dtype=torch.long, device=self.device
                    )
                    hidden = self.backbone(
                        input_ids=suffix, past_key_values=cache, use_cache=True
                    ).last_hidden_state
                    offset = matched_len
                    if matched_len < target_len:
                        self._store_prefix(
                            target_prefix,
                            self._extract_states(cache, target_len),
                            replace_key=matched_key,
                        )
                else:
                    self._prefix_misses += 1
                    cache = _new_cache(config=self.backbone.config)
                    full_ids = torch.tensor(
                        [row.ids], dtype=torch.long, device=self.device
                    )
                    hidden = self.backbone(
                        input_ids=full_ids, past_key_values=cache, use_cache=True
                    ).last_hidden_state
                    offset = 0
                    self._store_prefix(
                        target_prefix, self._extract_states(cache, target_len)
                    )
                cpos = [pos - offset for pos in row.candidate_positions]
                qpos = row.query_position - offset
                candidates = hidden[0:1, cpos, :]
                query = hidden[0:1, qpos, :]
                return [self.head(candidates, query)[0].float().cpu().tolist()]

            if matched_states is not None and matched_len == target_len:
                self._prefix_hits += 1
                prefix_states = matched_states
            elif matched_states is not None and matched_len > 0:
                self._prefix_hits += 1
                cache = self._fork_cache(matched_states, 1, matched_len)
                delta = torch.tensor(
                    [target_prefix[matched_len:]], dtype=torch.long, device=self.device
                )
                self.backbone(input_ids=delta, past_key_values=cache, use_cache=True)
                prefix_states = self._extract_states(cache, target_len)
                self._store_prefix(
                    target_prefix, prefix_states, replace_key=matched_key
                )
            else:
                self._prefix_misses += 1
                cache = _new_cache(config=self.backbone.config)
                prefix_ids = torch.tensor(
                    [target_prefix], dtype=torch.long, device=self.device
                )
                self.backbone(
                    input_ids=prefix_ids, past_key_values=cache, use_cache=True
                )
                prefix_states = self._extract_states(cache, target_len)
                self._store_prefix(target_prefix, prefix_states)

            batch_cache = self._fork_cache(prefix_states, len(encoded), target_len)
            suffix_len = max(len(row.ids) - target_len for row in encoded)
            suffix_ids = torch.full(
                (len(encoded), suffix_len),
                self.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for index, row in enumerate(encoded):
                suffix_ids[index, : len(row.ids) - target_len] = torch.tensor(
                    row.ids[target_len:], device=self.device
                )
            hidden = self.backbone(
                input_ids=suffix_ids, past_key_values=batch_cache, use_cache=True
            ).last_hidden_state
            result = []
            for index, row in enumerate(encoded):
                cpos = [pos - target_len for pos in row.candidate_positions]
                qpos = row.query_position - target_len
                candidates = hidden[index : index + 1, cpos, :]
                query = hidden[index : index + 1, qpos, :]
                result.append(self.head(candidates, query)[0].float().cpu().tolist())
        return result

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
        """Return one complete ordered FP32 logit vector per public question."""
        return self._records_logits((record,))[0][1]

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        encoded, rows = self._records_logits((record,))[0]
        return {
            row.question.id: tasks.interpret(row.question, logits)
            for row, logits in zip(encoded, rows, strict=True)
        }


def load_native(
    directory: str | Path,
    *,
    device: str = "cpu",
    max_length: int | None = None,
    prefix_cache_size: int = 16,
) -> NativeModel:
    """Load an Axes export entirely locally; no remote code, downloads or resume."""
    if type(prefix_cache_size) is not int or prefix_cache_size < 0:
        raise ValueError("prefix_cache_size must be a nonnegative integer")
    root = Path(directory).resolve()
    manifest = validate_artifact(root)
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("native inference supports cpu and cuda")
    if target.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("native CUDA inference requires BF16 support")
    maximum = (
        manifest["max_length"]
        if max_length is None
        else _positive_integer(max_length, "max_length")
    )
    if maximum > manifest["max_length"]:
        raise ValueError("max_length exceeds the artifact contract")
    raw = json.loads((root / "backbone/config.json").read_text())
    if not isinstance(raw, dict) or raw.get("model_type") != "qwen3":
        raise ValueError("native artifact requires a Qwen3 backbone")
    config = Qwen3Config.from_dict(raw)
    config._attn_implementation = "sdpa"
    backbone = Qwen3Model(config)
    base = load_file(str(root / "backbone/model.safetensors"))
    _finite_state(base)
    if any(
        value.dtype not in {torch.float32, torch.bfloat16} for value in base.values()
    ):
        raise ValueError("native backbone requires FP32 or BF16 storage")
    # CUDA preserves the historical mixed BF16/FP32 storage. CPU promotes BF16
    # values losslessly, as the Wyat CPU loader does; GPU arithmetic
    # equivalence is a separate validation and is never implied by this load.
    if target.type == "cpu":
        base = {name: value.float() for name, value in base.items()}
    backbone.load_state_dict(base, strict=True, assign=True)
    for layer in backbone.layers:
        for name in ("q_proj", "v_proj"):
            setattr(layer.self_attn, name, _Adapter(getattr(layer.self_attn, name)))
    state = load_file(str(root / "native-adapters.safetensors"))
    _finite_state(state)
    adapters = {
        name: parameter
        for name, parameter in backbone.named_parameters()
        if name.endswith((".down", ".up"))
    }
    if set(state) != set(adapters):
        raise ValueError("native adapter tensor inventory differs")
    with torch.no_grad():
        for name, value in state.items():
            if value.dtype != torch.float32 or value.shape != adapters[name].shape:
                raise ValueError("invalid native adapter tensor: " + name)
            adapters[name].copy_(value)
    head = CandidateHead(config.hidden_size, manifest["head_dim"])
    head_state = load_file(str(root / "decision_head.safetensors"))
    _finite_state(head_state)
    if any(value.dtype != torch.float32 for value in head_state.values()):
        raise ValueError("native shared head requires FP32 weights")
    head.load_state_dict(head_state, strict=True)
    backbone.config.use_cache = False
    backbone.requires_grad_(False)
    head.requires_grad_(False)
    return NativeModel(
        cast(Qwen3Model, cast(nn.Module, backbone).to(target)),
        head.to(target),
        load_tokenizer(root),
        manifest,
        target,
        maximum,
        prefix_cache_size=prefix_cache_size,
    )
