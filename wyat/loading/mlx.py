"""Local MLX execution of Wyat's native typed artifact and unmerged adapters.

The native task encoder, artifact contract and readouts are shared with CPU/CUDA.
This module adds a runtime without changing the trained architecture or weights.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, cast

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from wyat import tasks
from wyat.backends.kai.artifact import normalize_config, positive_integer
from wyat.backends.kai.encoding import TokenizerProtocol, load_tokenizer
from wyat.backends.kai.mlx import MLXCandidateHead, _weights
from wyat.wire import JsonObject, read_object

from .native_artifact import validate_artifact


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
    ) -> None:
        if type(responsive) is not bool:
            raise ValueError("responsive must be a boolean")
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.device = device
        self.max_length = max_length
        self.max_batch_tokens: int = manifest["max_batch_tokens"]
        self.model_id: str = manifest["model_id"]
        self.metal_enabled = metal_enabled
        self.responsive = responsive
        self._inference_lock = Lock()
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if type(pad) is not int or pad < 0:
            raise ValueError("tokenizer requires a nonnegative pad or EOS token")
        self.pad_id = pad

    def encode_record(self, record: object) -> tuple[tasks.EncodedQuestion, ...]:
        return tasks.encode_record(self.tokenizer, record, max_length=self.max_length)

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

    def _logits(self, encoded: Sequence[tasks.EncodedQuestion]) -> list[list[float]]:
        if not encoded:
            return []
        length = max(len(row.ids) for row in encoded)
        if length > self.max_length:
            raise ValueError("complete question exceeds max_length; no truncation")
        if length * len(encoded) > self.max_batch_tokens:
            raise ValueError("complete padded question batch exceeds explicit budget")
        # Retain the reference's exact width and row order: changing GEMM shapes
        # is a separate numerical/performance experiment, even for causal padding.
        tokens = [
            list(row.ids) + [self.pad_id] * (length - len(row.ids)) for row in encoded
        ]
        choices = max(len(row.candidate_positions) for row in encoded)
        positions = [
            list(row.candidate_positions)
            + [row.candidate_positions[0]] * (choices - len(row.candidate_positions))
            for row in encoded
        ]
        with self._inference_lock, mx.stream(self.device):
            ids = mx.array(tokens, dtype=mx.int32)
            endpoints = mx.array(positions, dtype=mx.int32)
            queries = mx.array([row.query_position for row in encoded], dtype=mx.int32)
            hidden = self._hidden(ids)
            rows = mx.arange(len(encoded))
            logits = self.head(hidden[rows[:, None], endpoints], hidden[rows, queries])
            mx.eval(logits)
            values = cast(list[list[float]], logits.tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]

    def logits(self, record: object) -> list[list[float]]:
        """One complete ordered FP32 logit vector per public question."""
        return self._logits(self.encode_record(record))

    def predict(
        self, record: object
    ) -> dict[str, tasks.ChoiceAnswer | tasks.ScoreAnswer | tasks.NoulAnswer]:
        encoded = self.encode_record(record)
        return {
            row.question.id: tasks.interpret(row.question, logits)
            for row, logits in zip(encoded, self._logits(encoded), strict=True)
        }


def load_native_mlx(
    directory: str | Path,
    *,
    device: str = "gpu",
    max_length: int | None = None,
    metal: bool = False,
    responsive: bool = True,
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
            from wyat.backends.kai.metal import (
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
        # applies only to the native scorer; maintained Kai/CUDA paths are intact.
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
    )
