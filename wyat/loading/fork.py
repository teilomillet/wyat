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
)
from wyat.loading.native import _request_record


class _DropLoadReport(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "LOAD REPORT" not in record.getMessage()


@contextmanager
def _quiet_base_loading() -> Iterator[None]:
    """Silence Transformers' load report and progress bar for the base weights.

    The report lists the checkpoint's vision and multi-token-prediction weights
    as unexpected; ``from_directory`` checks the same key sets itself and
    refuses every other missing, mismatched or unexpected weight. Other
    Transformers messages, and settings outside this load, are unchanged.
    """
    report = logging.getLogger("transformers.modeling_utils")
    drop = _DropLoadReport()
    progress = transformers_logging.is_progress_bar_enabled()
    report.addFilter(drop)
    transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
    try:
        yield
    finally:
        report.removeFilter(drop)
        if progress:
            transformers_logging.enable_progress_bar()  # type: ignore[no-untyped-call]


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
    """Score public Requests or records with an option-fork artifact."""

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
    ) -> None:
        self.backbone, self.head, self.tokenizer = backbone, head, tokenizer
        self.model_id = model_id
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens
        self.input_rendering = validate_input_rendering_policy(input_rendering)
        pad = tokenizer.pad_token_id
        self.pad_id = tokenizer.eos_token_id if pad is None else pad

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    @classmethod
    def from_directory(
        cls, path: str | Path, *, device: str = "cpu", max_length: int | None = None
    ) -> ForkScorer:
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

    def question_logits(self, row: ForkQuestion) -> list[float]:
        with torch.inference_mode(), self._amp():
            ids, mask = self._rows([row.prefix_ids])
            cache = self.backbone(
                input_ids=ids, attention_mask=mask, use_cache=True
            ).past_key_values
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
                        linear.lazy_initialization(
                            conv_states=conv, recurrent_states=state
                        )
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
            logits = torch.cat(values).float()
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
    path: str | Path, *, device: str = "cpu", max_length: int | None = None
) -> ForkScorer:
    return ForkScorer.from_directory(path, device=device, max_length=max_length)


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
