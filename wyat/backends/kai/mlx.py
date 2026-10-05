"""Native MLX choice scoring using MLX-LM's maintained Qwen3 backbone.

The checkpoint, segmented tokenizer recipe and FP32 head are shared with the
Torch backend. No language-model vocabulary projection or decoding is executed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx.nn.layers.activations import gelu
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from wyat.core import Request
from wyat.wire import JsonObject

from .artifact import load_artifact
from .encoding import EncodedRequest, KaiEncoder

Forward = Callable[[mx.array, mx.array, mx.array], mx.array]


# MLX's base type is unavailable on CPU-only installations.
class MLXCandidateHead(Module):  # type: ignore[misc, unused-ignore]
    """Released exact-GELU bilinear + MLP readout, entirely in FP32."""

    def __init__(self, hidden_size: int, head_dim: int) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.head_dim = head_dim
        self.candidate_norm = LayerNorm(hidden_size, eps=1e-5)
        self.query_norm = LayerNorm(hidden_size, eps=1e-5)
        self.key = Linear(hidden_size, head_dim, bias=False)
        self.query = Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = Linear(hidden_size, head_dim)
        self.query_mlp = Linear(hidden_size, head_dim, bias=False)
        self.scalar = Linear(head_dim, 1, bias=False)

    def __call__(self, candidates: mx.array, query: mx.array) -> mx.array:
        candidate = cast(mx.array, self.candidate_norm(candidates.astype(mx.float32)))
        global_query = cast(mx.array, self.query_norm(query.astype(mx.float32)))
        bilinear = mx.sum(
            self.key(candidate) * self.query(global_query)[:, None, :], axis=-1
        ) / math.sqrt(self.head_dim)
        nonlinear = self.scalar(
            gelu(
                self.candidate_mlp(candidate) + self.query_mlp(global_query)[:, None, :]
            )
        ).squeeze(-1)
        return bilinear + nonlinear


def _weights(path: Path, *, head: bool) -> dict[str, mx.array]:
    loaded: object = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected named checkpoint tensors: {path.name}")
    weights = cast(dict[str, mx.array], loaded)
    for name, value in weights.items():
        allowed = {mx.float32} if head else {mx.float32, mx.bfloat16}
        if value.dtype not in allowed:
            raise ValueError(
                f"Unsupported {'head' if head else 'backbone'} dtype: {name}"
            )
        # Keep validation temporary arrays bounded for large embedding tables.
        flat = value.reshape(-1)
        for start in range(0, flat.size, 1 << 20):
            if not bool(mx.all(mx.isfinite(flat[start : start + (1 << 20)])).item()):
                raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        weights[name] = value.astype(mx.float32)
    return weights


class MLXKaiScorer:
    """Immutable local weights; causal batched prefill and endpoint readout."""

    def __init__(
        self,
        backbone: Qwen3Model,
        head: MLXCandidateHead,
        encoder: KaiEncoder,
        *,
        model_id: str,
        device: mx.Device,
        compile: bool = False,
    ) -> None:
        if type(compile) is not bool:
            raise ValueError("compile must be a boolean")
        self.backbone = backbone
        self.head = head
        self.encoder = encoder
        self.model_id = model_id
        self.device = device
        self._compiled = compile
        self._forward: Forward = (
            cast(Forward, mx.compile(self._logits)) if self._compiled else self._logits
        )

    @property
    def compiled(self) -> bool:
        return self._compiled

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        max_length: int = 2048,
        token_cache_size: int = 0,
        device: str = "gpu",
        compile: bool = False,
    ) -> MLXKaiScorer:
        if device not in {"cpu", "gpu"}:
            raise ValueError("MLX device must be cpu or gpu")
        if device == "gpu" and not mx.metal.is_available():
            raise ValueError("MLX GPU inference requires available Apple Metal")
        if type(compile) is not bool:
            raise ValueError("compile must be a boolean")
        artifact = load_artifact(path, max_length=max_length)
        encoder = KaiEncoder.from_artifact(
            artifact,
            max_length=max_length,
            token_cache_size=token_cache_size,
        )
        target = mx.Device(mx.gpu if device == "gpu" else mx.cpu)
        with mx.stream(target):
            arguments = cast(Callable[[JsonObject], ModelArgs], ModelArgs.from_dict)
            args = arguments(artifact.config)
            backbone = Qwen3Model(args)
            head = MLXCandidateHead(args.hidden_size, artifact.head_dim)
            backbone_weights = _weights(
                artifact.root / "backbone/model.safetensors", head=False
            )
            head_weights = _weights(
                artifact.root / "decision_head.safetensors", head=True
            )
            backbone.load_weights(list(backbone_weights.items()), strict=True)
            head.load_weights(list(head_weights.items()), strict=True)
            backbone.eval()
            head.eval()
            mx.eval(backbone_weights, head_weights)
        return cls(
            backbone,
            head,
            encoder,
            model_id=artifact.model_id,
            device=target,
            compile=compile,
        )

    def encode_request(self, request: Request) -> EncodedRequest:
        return self.encoder.encode_request(request)

    def _logits(
        self, ids: mx.array, endpoints: mx.array, query_positions: mx.array
    ) -> mx.array:
        # Right padding cannot influence an earlier endpoint in a causal model.
        hidden = cast(mx.array, self.backbone(ids))
        rows = mx.arange(ids.shape[0])
        return self.head(
            hidden[rows[:, None], endpoints], hidden[rows, query_positions]
        )

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        encoded = [self.encode_request(request) for request in requests]
        with mx.stream(self.device):
            ids, endpoints, positions = _pack(encoded, self.encoder.pad_id)
            logits = self._forward(ids, endpoints, positions)
            mx.eval(logits)
            values = cast(list[list[float]], logits.tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]


def _pack(
    encoded: Sequence[EncodedRequest], pad_id: int, *, prefix_length: int = 0
) -> tuple[mx.array, mx.array, mx.array]:
    """One native packing owner; retain full padded width when removing prefix."""
    length = -(-max(len(row.ids) for row in encoded) // 8) * 8
    width = max(len(row.candidate_positions) for row in encoded)
    ids = mx.array(
        [
            list(row.ids[prefix_length:]) + [pad_id] * (length - len(row.ids))
            for row in encoded
        ],
        dtype=mx.int32,
    )
    endpoints = mx.array(
        [
            [position - prefix_length for position in row.candidate_positions]
            + [0] * (width - len(row.candidate_positions))
            for row in encoded
        ],
        dtype=mx.int32,
    )
    positions = mx.array(
        [row.query_position - prefix_length for row in encoded], dtype=mx.int32
    )
    return ids, endpoints, positions
