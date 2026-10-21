# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch inference for option-fork Wyat artifacts (Qwen3.5 hybrid backbone).

Each question is encoded as one canonical prefix holding the state, the
instructions and every option, then one isolated branch per option in caller
order. The prefix runs once; every branch continues from a copy of its cache
(attention keys/values and gated-delta-net convolution/recurrent state), and a
shared FP32 scalar head reads the branch's last token after the final norm.
This reproduces the training recipe exactly; golden tests compare it with the
trainer's own forward pass.

Optionally, per-layer state after a question's rendered state block is kept
across calls, keyed by that exact token prefix: consecutive questions of one
record state then continue from it instead of re-prefilling the state. Reuse is
token-exact and disabled by default (``prefix_cache_size=0``), which takes no
extra tokenization or forward pass and returns the historical logits.

CPU threads: without the optional flash-linear-attention and causal-conv1d
kernels, the hybrid layers run as many small PyTorch operations, and extra
intra-op threads mostly add synchronization. Measured on an Apple Silicon CPU
(2026-10-17): one thread was 7-26x faster than eight for 190-4,200-token
prompts, with identical logits. This module does not change torch's global
thread setting; callers on CPU should consider ``torch.set_num_threads(1)``
and measure on their own hardware.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, Qwen3_5Tokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from transformers.utils import logging as transformers_logging

from wyat import tasks
from wyat.core import ModelInputError, Request
from wyat.input_rendering import render_record, validate_input_rendering_policy
from wyat.loading import fork_artifact as contract
from wyat.loading.fork_artifact import (
    ForkQuestion,
    adapter_name,
    branch_groups,
    encode_record,
    reuse_prefix,
)
from wyat.loading.native import _request_record

# Per-layer prefix state: attention keys/values or gated-delta-net convolution
# and recurrent tensors, each defensively cloned so later forwards cannot mutate
# a stored snapshot.
type _ForkLayerState = tuple[torch.Tensor, torch.Tensor]


def _checked_prefix_cache_size(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("prefix_cache_size must be a nonnegative integer")
    return value


class _DropMessages(logging.Filter):
    def __init__(self, *markers: str) -> None:
        super().__init__()
        self.markers = markers

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(marker in message for marker in self.markers)


@contextmanager
def _dropping(logger: str, *markers: str) -> Iterator[None]:
    """Drop one Transformers logger's records that contain a marker, briefly."""
    drop = _DropMessages(*markers)
    logging.getLogger(logger).addFilter(drop)
    try:
        yield
    finally:
        logging.getLogger(logger).removeFilter(drop)


@contextmanager
def _quiet_base_loading() -> Iterator[None]:
    """Silence Transformers' load report and progress bar for the base weights.

    The report lists the checkpoint's vision and multi-token-prediction weights
    as unexpected; ``from_directory`` checks the same key sets itself and
    refuses every other missing, mismatched or unexpected weight. Other
    Transformers messages, and settings outside this load, are unchanged.
    """
    progress = transformers_logging.is_progress_bar_enabled()
    transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    try:
        with _dropping("transformers.modeling_utils", "LOAD REPORT"):
            yield
    finally:
        if progress:
            transformers_logging.enable_progress_bar()  # type: ignore[no-untyped-call]


def _without_cuda_kernel_advice() -> AbstractContextManager[None]:
    """Off CUDA, drop Transformers' advice to install its CUDA-only kernels.

    ``causal-conv1d`` and ``flash-linear-attention`` need CUDA or Triton; on
    CPU and MPS the reference PyTorch path is the runtime, not a fallback.
    """
    return _dropping(
        "transformers.integrations.hub_kernels",
        "is falling back to its reference PyTorch implementation",
    )


class _ForkAdapter(nn.Module):
    """Unmerged rank-16 LoRA; the update stays FP32 outside autocast."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Parameter(torch.empty(rank, base.in_features))
        self.up = nn.Parameter(torch.empty(base.out_features, rank))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        with torch.autocast(value.device.type, enabled=False):
            update = 2 * F.linear(F.linear(value.float(), self.down), self.up)
        return cast(torch.Tensor, base + update)


class _ScalarHead(nn.Module):
    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.autocast(hidden.device.type, enabled=False):
            return cast(torch.Tensor, self.net(hidden.float()).squeeze(-1))


class ForkScorer:
    """Score public Requests or records with an option-fork artifact.

    With a nonnegative ``prefix_cache_size``, the per-layer state after the
    rendered state block of a question is retained across calls, keyed by the
    exact token prefix it covers. A later question whose prefix begins with a
    stored key (consecutive questions of one record state, or an exact repeat)
    continues the backbone from that state instead of re-prefilling it. Only
    token prefixes are reused and the disabled default takes no extra
    tokenization or forward pass, so results never depend on earlier requests.
    The cache is owned by a single inference owner: the Wyat server serializes
    active inference, so no lock is taken here.
    """

    def __init__(
        self,
        backbone: Qwen3_5TextModel,
        head: _ScalarHead,
        tokenizer: Any,
        *,
        model_id: str,
        max_length: int,
        max_batch_tokens: int,
        input_rendering: object,
        prefix_cache_size: int = 0,
    ) -> None:
        self.backbone, self.head, self.tokenizer = backbone, head, tokenizer
        self.model_id = model_id
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens
        self.input_rendering = validate_input_rendering_policy(input_rendering)
        self.prefix_cache_size = _checked_prefix_cache_size(prefix_cache_size)
        # Longest-first matching keeps the most prefix tokens cached; an entry's
        # state is the per-layer state after exactly its key tokens.
        self._prefix_cache: OrderedDict[
            tuple[int, ...], tuple[_ForkLayerState, ...]
        ] = OrderedDict()
        self._prefix_hits = 0
        self._prefix_misses = 0
        pad = tokenizer.pad_token_id
        self.pad_id = tokenizer.eos_token_id if pad is None else pad

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

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
        """Clear every retained prefix state and reset the cache counters."""
        self._prefix_cache.clear()
        self._prefix_hits = 0
        self._prefix_misses = 0

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        max_length: int | None = None,
        prefix_cache_size: int = 0,
    ) -> ForkScorer:
        prefix_cache_size = _checked_prefix_cache_size(prefix_cache_size)
        root = Path(path)
        manifest = contract.validate_artifact(root)
        config = AutoConfig.from_pretrained(
            root / "base", local_files_only=True, trust_remote_code=False
        )
        text = getattr(config, "text_config", config)
        if text.model_type != manifest["backbone"]:
            raise ValueError("fork backbone configuration differs from manifest")
        if manifest["max_length"] > text.max_position_embeddings:
            raise ValueError("fork max_length exceeds the backbone context")
        with _quiet_base_loading():
            backbone, loading = cast(
                tuple[Qwen3_5TextModel, dict[str, Any]],
                Qwen3_5TextModel.from_pretrained(
                    root / "base",
                    config=text,
                    dtype=torch.float32,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    attn_implementation="sdpa",
                    output_loading_info=True,
                ),
            )
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError("fork base text weights incomplete")
        stray = [
            key
            for key in loading.get("unexpected_keys", ())
            if not key.startswith(("model.visual.", "mtp."))
        ]
        if stray:
            raise ValueError("unexpected fork base weights: " + ", ".join(stray[:5]))
        target = torch.device(device)
        if target.type == "cuda":
            # Same storage as training: BF16 autocast rounds these frozen
            # weights to BF16 on every call, so storing them so is identical.
            for module in backbone.layers.modules():
                if isinstance(module, nn.Linear):
                    module.to(torch.bfloat16)
        rank = manifest["adapter"]["rank"]
        adapters = load_file(root / "adapters.safetensors")
        names = {
            adapter_name(index, owner, name, factor)
            for index, kind in enumerate(text.layer_types)
            for owner, name in contract.QWEN3_5_ADAPTER_TARGETS[kind]
            for factor in ("down", "up")
        }
        if set(adapters) != names:
            raise ValueError("fork adapter inventory differs")
        expected = set()
        for index, kind in enumerate(text.layer_types):
            block = backbone.layers[index]
            for owner_name, name in contract.QWEN3_5_ADAPTER_TARGETS[kind]:
                owner = getattr(block, owner_name)
                adapter = _ForkAdapter(getattr(owner, name), rank)
                for factor in ("down", "up"):
                    key = adapter_name(index, owner_name, name, factor)
                    expected.add(key)
                    value = adapters[key]
                    if (
                        value.shape != getattr(adapter, factor).shape
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError("invalid fork adapter tensor: " + key)
                    getattr(adapter, factor).data.copy_(value.float())
                setattr(owner, name, adapter)
        if set(adapters) != expected:
            raise ValueError("fork adapter inventory differs")
        head = _ScalarHead(text.hidden_size, manifest["head_width"])
        state = load_file(root / "head.safetensors")
        shapes = {k: v.shape for k, v in head.state_dict().items()}
        if {k: v.shape for k, v in state.items()} != shapes or not all(
            torch.isfinite(value).all() for value in state.values()
        ):
            raise ValueError("invalid fork head tensors")
        head.load_state_dict({k: v.float() for k, v in state.items()})
        backbone.requires_grad_(False)
        head.requires_grad_(False)
        # The trainer's AutoTokenizer resolves Qwen3_5Tokenizer from the base
        # config. Here the tokenizer files sit beside the manifest, without that
        # config, and AutoTokenizer would pick tokenizer_config's Qwen2Tokenizer,
        # whose pre-tokenizer splits combining marks (Hindi, Thai) differently.
        tokenizer = Qwen3_5Tokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
        limit = manifest["max_length"] if max_length is None else max_length
        if not 0 < limit <= manifest["max_length"]:
            raise ValueError("max_length must be within the artifact's limit")
        return cls(
            cast(Qwen3_5TextModel, cast(nn.Module, backbone).to(target).eval()),
            head.to(target).eval(),
            tokenizer,
            model_id=manifest["model_id"],
            max_length=limit,
            max_batch_tokens=manifest["max_batch_tokens"],
            input_rendering=manifest.get("input_rendering"),
            prefix_cache_size=prefix_cache_size,
        )

    def _amp(self) -> AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def encode_record(self, record: object) -> tuple[ForkQuestion, ...]:
        return encode_record(
            self.tokenizer,
            render_record(record, self.input_rendering),
            max_length=self.max_length,
        )

    def _rows(
        self, sequences: Sequence[tuple[int, ...]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), length), self.pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, : len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[index, : len(sequence)] = 1
        return ids, mask

    @staticmethod
    def _snapshot(cache: DynamicCache, length: int) -> tuple[_ForkLayerState, ...]:
        """Defensive clones of every layer's state after ``length`` tokens.

        Gated-delta-net convolution/recurrent tensors are updated in place by
        later forwards and attention keys/values are concatenated onto, so a
        stored snapshot must not share storage with the cache it came from.
        Attention states are sliced to the cached token count.
        """
        states: list[_ForkLayerState] = []
        for raw in cache.layers:
            if isinstance(raw, LinearAttentionLayer):
                linear = raw
                conv = cast(torch.Tensor, linear.conv_states[0])
                recurrent = cast(torch.Tensor, linear.recurrent_states[0])
                states.append((conv.detach().clone(), recurrent.detach().clone()))
            else:
                attention = cast(DynamicLayer, raw)
                keys = cast(torch.Tensor, attention.keys)[:, :, :length, :]
                values = cast(torch.Tensor, attention.values)[:, :, :length, :]
                states.append((keys.detach().clone(), values.detach().clone()))
        return tuple(states)

    def _restore(self, states: tuple[_ForkLayerState, ...]) -> DynamicCache:
        """A fresh prefix cache holding the given per-layer states."""
        cache = DynamicCache(config=self.backbone.config)
        for (first, second), raw in zip(states, cache.layers, strict=True):
            if isinstance(raw, LinearAttentionLayer):
                linear = raw
                conv, recurrent = first.clone(), second.clone()
                linear.lazy_initialization(conv_states=conv, recurrent_states=recurrent)
                linear.conv_states[0] = conv
                linear.recurrent_states[0] = recurrent
                linear.is_conv_states_initialized[0] = True
                linear.is_recurrent_states_initialized[0] = True
                linear.has_previous_state[0] = True
            else:
                attention = cast(DynamicLayer, raw)
                keys, values = first.clone(), second.clone()
                attention.lazy_initialization(keys, values)
                attention.keys, attention.values = keys, values
        return cache

    def _match_prefix(self, prefix_ids: tuple[int, ...]) -> tuple[int, ...] | None:
        """The longest stored key that is an exact token prefix, or None."""
        best: tuple[int, ...] | None = None
        for key in self._prefix_cache:
            if len(key) <= (0 if best is None else len(best)):
                continue
            if prefix_ids[: len(key)] == key:
                best = key
        if best is not None:
            self._prefix_cache.move_to_end(best)
        return best

    def _store_prefix(
        self, key: tuple[int, ...], states: tuple[_ForkLayerState, ...]
    ) -> None:
        if not key or self.prefix_cache_size <= 0:
            return
        self._prefix_cache[key] = states
        self._prefix_cache.move_to_end(key)
        while len(self._prefix_cache) > self.prefix_cache_size:
            self._prefix_cache.popitem(last=False)

    def _run_prefix_tail(
        self, cache: DynamicCache, row: ForkQuestion, start: int
    ) -> DynamicCache:
        """Advance ``cache`` (holding ``start`` prefix tokens) to the full prefix."""
        prefix = len(row.prefix_ids)
        if start >= prefix:
            return cache
        ids, mask = self._rows([row.prefix_ids[start:]])
        # The cached tokens are all non-padding, and the tail is one sequence.
        seen = torch.ones(
            (1, len(row.prefix_ids)), dtype=mask.dtype, device=self.device
        )
        positions = torch.arange(start, len(row.prefix_ids), device=self.device)[
            None, :
        ]
        return cast(
            DynamicCache,
            self.backbone(
                input_ids=ids,
                attention_mask=seen,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
            ).past_key_values,
        )

    def _prefix_cursor(self, row: ForkQuestion) -> DynamicCache:
        """A cache advanced over every token of ``row.prefix_ids``.

        With the cache disabled, this is the historical single prefix forward.
        Otherwise a stored token prefix is continued when one matches; on a miss
        the forward is split at the state-block token boundary so the resulting
        state can be retained for later questions of the same record state.
        """
        if self.prefix_cache_size <= 0:
            ids, mask = self._rows([row.prefix_ids])
            return cast(
                DynamicCache,
                self.backbone(
                    input_ids=ids, attention_mask=mask, use_cache=True
                ).past_key_values,
            )
        key = self._match_prefix(row.prefix_ids)
        if key is not None:
            self._prefix_hits += 1
            return self._run_prefix_tail(
                self._restore(self._prefix_cache[key]), row, len(key)
            )
        self._prefix_misses += 1
        boundary = reuse_prefix(self.tokenizer, row)
        if not boundary or len(boundary) >= len(row.prefix_ids):
            ids, mask = self._rows([row.prefix_ids])
            return cast(
                DynamicCache,
                self.backbone(
                    input_ids=ids, attention_mask=mask, use_cache=True
                ).past_key_values,
            )
        ids, mask = self._rows([boundary])
        cache = cast(
            DynamicCache,
            self.backbone(
                input_ids=ids, attention_mask=mask, use_cache=True
            ).past_key_values,
        )
        self._store_prefix(boundary, self._snapshot(cache, len(boundary)))
        return self._run_prefix_tail(cache, row, len(boundary))

    def _branch_logits(self, cache: DynamicCache, row: ForkQuestion) -> torch.Tensor:
        """Score every option branch from a cache holding the full prefix."""
        prefix = len(row.prefix_ids)
        values = []
        for group in branch_groups(row.option_ids, prefix, self.max_batch_tokens):
            width = len(group)
            fork = DynamicCache(config=self.backbone.config)
            for source, layer in zip(cache.layers, fork.layers, strict=True):
                if isinstance(source, LinearAttentionLayer):
                    linear = cast(LinearAttentionLayer, layer)
                    conv = cast(torch.Tensor, source.conv_states[0])
                    state = cast(torch.Tensor, source.recurrent_states[0])
                    conv = conv.expand(width, -1, -1).clone()
                    state = state.expand(width, -1, -1, -1).clone()
                    linear.lazy_initialization(conv_states=conv, recurrent_states=state)
                    linear.conv_states[0] = conv
                    linear.recurrent_states[0] = state
                    linear.is_conv_states_initialized[0] = True
                    linear.is_recurrent_states_initialized[0] = True
                    linear.has_previous_state[0] = True
                else:
                    attention = cast(DynamicLayer, layer)
                    keys = cast(torch.Tensor, cast(DynamicLayer, source).keys)
                    vals = cast(torch.Tensor, cast(DynamicLayer, source).values)
                    keys = keys.expand(width, -1, -1, -1).clone()
                    vals = vals.expand(width, -1, -1, -1).clone()
                    attention.lazy_initialization(keys, vals)
                    attention.keys, attention.values = keys, vals
            suffix_ids, suffix_mask = self._rows(group)
            positions = torch.arange(
                prefix, prefix + suffix_ids.shape[1], device=self.device
            )[None, :].expand(width, -1)
            output = self.backbone(
                input_ids=suffix_ids,
                attention_mask=torch.cat(
                    [
                        torch.ones(
                            (width, prefix),
                            dtype=suffix_mask.dtype,
                            device=self.device,
                        ),
                        suffix_mask,
                    ],
                    dim=1,
                ),
                position_ids=positions,
                past_key_values=fork,
                use_cache=True,
            )
            last = output.last_hidden_state[
                torch.arange(width, device=self.device),
                torch.tensor([len(s) - 1 for s in group], device=self.device),
            ]
            values.append(self.head(last))
        return torch.cat(values).float()

    def question_logits(self, row: ForkQuestion) -> list[float]:
        advice = (
            nullcontext()
            if self.device.type == "cuda"
            else _without_cuda_kernel_advice()
        )
        with torch.inference_mode(), self._amp(), advice:
            logits = self._branch_logits(self._prefix_cursor(row), row)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite fork logits")
        return cast(list[float], logits.cpu().tolist())

    def _encode_request(self, request: Request) -> ForkQuestion:
        try:
            return self.encode_record(_request_record(request))[0]
        except ModelInputError:
            raise
        except ValueError as error:
            raise contract.input_error(error) from error

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        """Score public Requests in request and candidate order.

        Every request is encoded before any forward pass, so a refused request
        costs no model work."""
        rows = [self._encode_request(r) for r in requests]
        return [self.question_logits(row) for row in rows]

    def _encode_public(self, record: object) -> tuple[ForkQuestion, ...]:
        """A malformed record keeps its ValueError; encoder refusals of a valid
        record raise ModelInputError with the native loader's reasons."""
        tasks.public_questions(render_record(record, self.input_rendering))
        try:
            return self.encode_record(record)
        except ValueError as error:
            raise contract.input_error(error) from error

    def logits(self, record: object) -> list[list[float]]:
        return [self.question_logits(row) for row in self._encode_public(record)]

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        return {
            row.question.id: tasks.interpret(row.question, self.question_logits(row))
            for row in self._encode_public(record)
        }


def load_fork(
    path: str | Path,
    *,
    device: str = "cpu",
    max_length: int | None = None,
    prefix_cache_size: int = 0,
) -> ForkScorer:
    return ForkScorer.from_directory(
        path,
        device=device,
        max_length=max_length,
        prefix_cache_size=prefix_cache_size,
    )


def write_artifact(
    output: str | Path,
    *,
    base: str | Path,
    tokenizer_files: Sequence[str | Path],
    adapters: dict[str, torch.Tensor],
    head: dict[str, torch.Tensor],
    max_length: int,
    max_batch_tokens: int,
    head_width: int,
    input_rendering: object,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write a complete artifact: base files copied, tensors saved, manifest last."""
    import shutil

    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "base").mkdir()
    # One weights file, stored as base/model.safetensors: a shard index is not
    # admitted (its weight map could name files outside the artifact).
    weights = sorted(Path(base).glob("*.safetensors"))
    if len(weights) != 1:
        raise ValueError("fork export requires exactly one base safetensors file")
    shutil.copy2(Path(base) / "config.json", root / "base" / "config.json")
    shutil.copy2(weights[0], root / "base" / "model.safetensors")
    for name in tokenizer_files:
        shutil.copy2(name, root / Path(name).name)
    save_file(
        {k: v.detach().float().contiguous().cpu() for k, v in adapters.items()},
        root / "adapters.safetensors",
    )
    save_file(
        {k: v.detach().float().contiguous().cpu() for k, v in head.items()},
        root / "head.safetensors",
    )
    manifest = contract.manifest_for(
        root,
        max_length=max_length,
        max_batch_tokens=max_batch_tokens,
        head_width=head_width,
        input_rendering=input_rendering,
        provenance=provenance,
    )
    (root / contract.MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    contract.validate_artifact(root)
    return manifest
