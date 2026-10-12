"""Local typed Qwen inference and the canonical full-artifact contract.

Axes writes this format; Wyat owns validation, execution and typed readouts.
Adapters remain separate: merging would change the trained floating-point math.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model

from wyat import tasks
from wyat.backends.kai.encoding import load_tokenizer
from wyat.backends.kai.torch import CandidateHead

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
    ) -> None:
        self.backbone = backbone
        cast(nn.Module, backbone).eval()
        self.head = head.eval()
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.device = device
        self.max_length = max_length
        self.max_batch_tokens = manifest["max_batch_tokens"]
        self.model_id = manifest["model_id"]
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise ValueError("tokenizer requires a nonnegative pad or EOS token")

    def encode_record(self, record: object) -> tuple[tasks.EncodedQuestion, ...]:
        return tasks.encode_record(self.tokenizer, record, max_length=self.max_length)

    def _logits(self, encoded: Sequence[tasks.EncodedQuestion]) -> list[list[float]]:
        length = max(len(row.ids) for row in encoded)
        if length * len(encoded) > self.max_batch_tokens:
            raise ValueError("complete padded question batch exceeds explicit budget")
        ids = torch.full(
            (len(encoded), length), self.pad_id, dtype=torch.long, device=self.device
        )
        for index, row in enumerate(encoded):
            ids[index, : len(row.ids)] = torch.tensor(row.ids, device=self.device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
            result = []
            for index, row in enumerate(encoded):
                candidates = hidden[index : index + 1, row.candidate_positions, :]
                query = hidden[index : index + 1, row.query_position, :]
                result.append(self.head(candidates, query)[0].float().cpu().tolist())
        return result

    def logits(self, record: object) -> list[list[float]]:
        """Return one complete ordered FP32 logit vector per public question."""
        return self._logits(self.encode_record(record))

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        encoded = self.encode_record(record)
        return {
            row.question.id: tasks.interpret(row.question, logits)
            for row, logits in zip(encoded, self._logits(encoded), strict=True)
        }


def load_native(
    directory: str | Path, *, device: str = "cpu", max_length: int | None = None
) -> NativeModel:
    """Load an Axes export entirely locally; no remote code, downloads or resume."""
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
    # values losslessly, as the released Kai CPU loader does; GPU arithmetic
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
    )
