"""PyTorch CPU/CUDA inference for the released candidate-aware Kai contract.

Adapted from Decision 2.0 at cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764.
Artifact validation and segmented input belong to artifact and encoding. This
module owns tensor loading, the FP32 candidate head, and causal execution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from wyat.core import Request
from wyat.wire import JsonObject

from .artifact import ARCHITECTURE as ARCHITECTURE
from .artifact import PROMPT_VERSION as PROMPT_VERSION
from .artifact import load_artifact, normalize_config, positive_integer
from .encoding import EncodedRequest as EncodedRequest
from .encoding import KaiEncoder, TokenizerProtocol, load_tokenizer


def _config(raw: JsonObject) -> Qwen3Config:
    return Qwen3Config.from_dict(normalize_config(raw))


class CandidateHead(nn.Module):
    """The released FP32 bilinear plus nonlinear per-candidate readout."""

    def __init__(self, hidden_size: int, head_dim: int = 256) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(query.float())
            bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(
                -1
            ) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(
                F.gelu(
                    self.candidate_mlp(candidate)
                    + self.query_mlp(global_query)[:, None, :]
                )
            ).squeeze(-1)
            return cast(torch.Tensor, bilinear + nonlinear)


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    # Keep the temporary boolean allocation below 1 MiB, including embeddings.
    flattened = value.detach().reshape(-1)
    for start in range(0, flattened.numel(), 1 << 20):
        chunk = flattened[start : start + (1 << 20)]
        if not torch.isfinite(chunk).all().item():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")


class KaiScorer(KaiEncoder):
    """Immutable CPU/CUDA weights with the shared entry-bounded input encoder."""

    def __init__(
        self,
        backbone: Qwen3Model,
        head: CandidateHead,
        tokenizer: TokenizerProtocol,
        *,
        device: torch.device,
        max_length: int,
        max_options: int,
        token_cache_size: int,
        model_id: str,
    ) -> None:
        super().__init__(
            tokenizer,
            max_length=max_length,
            max_options=max_options,
            token_cache_size=token_cache_size,
        )
        self.backbone = backbone
        self.head = head
        self.device = device
        self.model_id = model_id

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        max_length: int = 2048,
        token_cache_size: int = 0,
        attention: str = "sdpa",
    ) -> KaiScorer:
        if attention not in {"sdpa", "eager"}:
            raise ValueError("attention must be sdpa or eager")
        target = torch.device(device)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("Kai currently supports cpu and cuda devices")
        if target.type == "cuda" and (
            not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("CUDA/ROCm inference requires a BF16-capable GPU")
        artifact = load_artifact(path, max_length)
        tokenizer = load_tokenizer(artifact.root)
        config = _config(artifact.config)
        try:
            backbone, loading = cast(
                tuple[Qwen3Model, dict[str, object]],
                Qwen3Model.from_pretrained(
                    artifact.root / "backbone",
                    config=config,
                    dtype=torch.float32,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    attn_implementation=attention,
                    output_loading_info=True,
                ),
            )
        except SafetensorError as error:
            raise ValueError("Malformed Kai backbone Safetensors weights") from error
        if any(
            loading.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(
                f"Backbone weights do not match their configuration: {loading}"
            )
        for name, parameter in backbone.named_parameters():
            _finite_tensor(parameter, name)
        for name, buffer in backbone.named_buffers():
            _finite_tensor(buffer, name)
        head = CandidateHead(
            positive_integer(artifact.config.get("hidden_size"), "hidden_size"),
            artifact.head_dim,
        )
        try:
            head_state = load_file(artifact.root / "decision_head.safetensors")
        except SafetensorError as error:
            raise ValueError("Malformed Kai head Safetensors weights") from error
        for name, tensor in head_state.items():
            _finite_tensor(tensor, name)
        head.load_state_dict(head_state, strict=True)
        # Restore only losslessly representable Linear weights to the released
        # BF16 storage recipe; embeddings, norms, residuals and head stay FP32.
        if target.type == "cuda":
            for layer in backbone.modules():
                if isinstance(layer, nn.Linear):
                    for linear_parameter in (layer.weight, layer.bias):
                        if linear_parameter is not None:
                            rounded = linear_parameter.detach().to(torch.bfloat16)
                            if torch.equal(rounded.float(), linear_parameter.detach()):
                                linear_parameter.data = rounded
        backbone.config.use_cache = False
        cast(nn.Module, backbone).to(target)
        cast(nn.Module, backbone).eval()
        return cls(
            backbone,
            head.to(target).eval(),
            tokenizer,
            device=target,
            max_length=max_length,
            max_options=artifact.max_options,
            token_cache_size=token_cache_size,
            model_id=artifact.model_id,
        )

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        encoded = [self.encode_request(request) for request in requests]
        return self._score_encoded(encoded)

    def _score_encoded(
        self,
        encoded: Sequence[EncodedRequest],
        *,
        prefix_length: int = 0,
        cache: DynamicCache | None = None,
    ) -> list[list[float]]:
        """One tensor/readout owner for fully admitted original request rows."""
        if not encoded:
            return []
        if cache is None:
            if prefix_length != 0:
                raise ValueError("Removing prefix tokens requires a seeded cache")
        elif (
            self.device.type != "cpu"
            or type(cache) is not DynamicCache
            or type(prefix_length) is not int
            or prefix_length <= 0
            or cache.get_seq_length() != prefix_length
        ):
            raise ValueError("Cached scoring requires a seeded CPU DynamicCache")
        length = max(len(row.ids) for row in encoded) - prefix_length
        width = max(len(row.candidate_positions) for row in encoded)
        batch_size = len(encoded)
        ids = torch.full((batch_size, length), self.pad_id, dtype=torch.long)
        endpoints = torch.zeros((batch_size, width), dtype=torch.long)
        for index, row in enumerate(encoded):
            suffix = row.ids[prefix_length:]
            ids[index, : len(suffix)] = torch.tensor(suffix, dtype=torch.long)
            endpoints[index, : len(row.candidate_positions)] = torch.tensor(
                [position - prefix_length for position in row.candidate_positions]
            )
        ids, endpoints = (value.to(self.device) for value in (ids, endpoints))
        rows = torch.arange(batch_size, device=self.device)
        query_positions = torch.tensor(
            [row.query_position - prefix_length for row in encoded], device=self.device
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            # Every readout precedes right padding. Causality excludes those
            # future pads, so a separate padding mask cannot affect the readout.
            if cache is None:
                output = cast(
                    BaseModelOutputWithPast,
                    self.backbone(input_ids=ids, use_cache=False),
                )
            else:
                # HF owns the absolute RoPE positions and rectangular offset
                # mask. An unmasked rectangular is_causal call is not equivalent.
                output = cast(
                    BaseModelOutputWithPast,
                    self.backbone(input_ids=ids, past_key_values=cache, use_cache=True),
                )
            hidden = output.last_hidden_state
            if hidden is None:
                raise ValueError("Kai backbone did not return hidden states")
            logits = cast(
                torch.Tensor,
                self.head(
                    hidden[rows[:, None], endpoints], hidden[rows, query_positions]
                ),
            )
        values = cast(list[list[float]], logits.float().cpu().tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]
