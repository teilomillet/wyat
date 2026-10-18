# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""MLX inference for option-fork Wyat artifacts (Qwen3.5 hybrid backbone).

The artifact, prompt encoding, branch grouping and readout are those of the
PyTorch runtime (``wyat.loading.fork``); only execution differs. MLX-LM's
Qwen3.5 text model runs the backbone with weights expanded from BF16 storage to
FP32, on Apple GPU or CPU, with the artifact's unmerged FP32 adapters and FP32
scalar head. The prefix runs once; every branch group continues from broadcast
copies of its attention keys/values and gated-delta-net convolution/recurrent
states.

Branch rows are right-padded without a mask: every layer is causal, so padding
after a branch's last token cannot change the hidden state read there.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock
from time import sleep
from typing import Any, cast

import mlx.core as mx
from mlx.nn.layers.activations import gelu, silu
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.normalization import LayerNorm
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Model, ModelArgs, Qwen3_5TextModel
from transformers import Qwen3_5Tokenizer

from wyat import tasks
from wyat.backends.wyat.mlx import _weights
from wyat.core import ModelInputError, Request
from wyat.input_rendering import render_record, validate_input_rendering_policy
from wyat.loading import fork_artifact as contract
from wyat.loading.fork_artifact import (
    ForkQuestion,
    adapter_name,
    branch_groups,
    encode_record,
)
from wyat.loading.mlx import _Adapter, _request_record

type _Cache = KVCache | ArraysCache

_new_kv_cache = cast(Callable[[], KVCache], KVCache)


def _l2norm(value: mx.array) -> mx.array:
    return value * mx.rsqrt((value * value).sum(axis=-1, keepdims=True) + 1e-6)


class _GatedDeltaNet(GatedDeltaNet):  # type: ignore[misc, unused-ignore]
    """MLX-LM's gated delta net with the training runtime's q/k normalization.

    MLX-LM computes ``rms_norm(x, eps=1e-6) / sqrt(d)``, which adds epsilon to
    the mean rather than the sum of squares; Transformers uses
    ``x * rsqrt(sum(x * x) + 1e-6)``. The two differ materially for small-norm
    vectors. Branch rows need no padding mask, so none is accepted.
    """

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        if mask is not None:
            raise ValueError("option-fork linear attention takes no padding mask")
        batch, length, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, length, self.num_v_heads, self.head_v_dim
        )
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        keep = self.conv_kernel_size - 1
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((batch, keep, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -keep:, :])
        conv_out = silu(self.conv1d(conv_input))
        q, k, v = (
            part.reshape(batch, length, heads, width)
            for part, heads, width in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                (self.num_k_heads, self.num_k_heads, self.num_v_heads),
                (self.head_k_dim, self.head_k_dim, self.head_v_dim),
                strict=True,
            )
        )
        q = self.head_k_dim**-0.5 * _l2norm(q)
        k = _l2norm(k)
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            None if cache is None else cache[1],
            None,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
        out = self.norm(out, z)
        return cast(mx.array, self.out_proj(out.reshape(batch, length, -1)))


# PyTorch ``nn.Sequential`` indices of the trained head's parameterized layers.
_HEAD_NAMES = {"net.0.": "norm.", "net.1.": "project.", "net.3.": "score."}


class _ScalarHead(Module):  # type: ignore[misc, unused-ignore]
    """The FP32 head: LayerNorm (epsilon 1e-5), Linear, exact GELU, Linear."""

    def __init__(self, hidden: int, width: int) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.norm = LayerNorm(hidden, eps=1e-5)
        self.project = Linear(hidden, width)
        self.score = Linear(width, 1)

    def __call__(self, hidden: mx.array) -> mx.array:
        normalized = cast(mx.array, self.norm(hidden.astype(mx.float32)))
        return self.score(gelu(self.project(normalized))).squeeze(-1)


def _text_config(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config = json.loads((root / "base/config.json").read_text())
    text = dict(config.get("text_config", config))
    if text.get("model_type") != manifest["backbone"]:
        raise ValueError("fork backbone configuration differs from manifest")
    if manifest["max_length"] > text["max_position_embeddings"]:
        raise ValueError("fork max_length exceeds the backbone context")
    # MLX-LM places full attention at a fixed interval; Transformers names the
    # type of every layer. Accept only layouts both describe identically.
    count = text["num_hidden_layers"]
    kinds = text.get("layer_types")
    interval = text.get("full_attention_interval", 4)
    if kinds is not None and "full_attention" in kinds:
        interval = kinds.index("full_attention") + 1
    expected = [
        "full_attention" if (index + 1) % interval == 0 else "linear_attention"
        for index in range(count)
    ]
    if kinds is not None and list(kinds) != expected:
        raise ValueError("fork layer layout is not supported by MLX")
    if (
        text.get("hidden_act", "silu") != "silu"
        or text.get("attn_output_gate", True) is not True
        or text.get("num_experts", 0)
    ):
        raise ValueError("fork backbone configuration is not supported by MLX")
    text["full_attention_interval"] = interval
    # Scoring reads final-norm hidden states: build no separate LM head.
    text["tie_word_embeddings"] = True
    return text


def _text_weights(base: dict[str, mx.array]) -> dict[str, mx.array]:
    """Text-model weights under MLX-LM names, from either Transformers layout.

    A full Qwen3.5 checkpoint nests the text model under ``model.language_model``
    beside vision and multi-token-prediction weights, which scoring ignores; a
    bare text-model export has unprefixed names.
    """
    weights: dict[str, mx.array] = {}
    for key, value in base.items():
        if key.startswith(("model.visual.", "mtp.")):
            continue
        name = "language_model.model." + key.removeprefix("model.language_model.")
        if key.split(".", 1)[0] not in {"model", "embed_tokens", "layers", "norm"} or (
            key.startswith("model.") and not key.startswith("model.language_model.")
        ):
            raise ValueError("unexpected fork base weight: " + key)
        if name in weights:
            raise ValueError("duplicate fork base weight: " + key)
        weights[name] = value
    return weights


def _install_adapters(
    backbone: Qwen3_5TextModel, adapters: dict[str, mx.array], rank: int
) -> None:
    expected: dict[str, tuple[int, int]] = {}
    for index, layer in enumerate(backbone.layers):
        kind = "linear_attention" if layer.is_linear else "full_attention"
        for owner, name in contract.QWEN3_5_ADAPTER_TARGETS[kind]:
            output_size, input_size = getattr(getattr(layer, owner), name).weight.shape
            expected[adapter_name(index, owner, name, "down")] = (rank, input_size)
            expected[adapter_name(index, owner, name, "up")] = (output_size, rank)
    if set(adapters) != set(expected):
        raise ValueError("fork adapter inventory differs")
    for key, shape in expected.items():
        if adapters[key].shape != shape:
            raise ValueError("invalid fork adapter tensor: " + key)
    for index, layer in enumerate(backbone.layers):
        kind = "linear_attention" if layer.is_linear else "full_attention"
        for owner_name, name in contract.QWEN3_5_ADAPTER_TARGETS[kind]:
            owner = getattr(layer, owner_name)
            down = adapters[adapter_name(index, owner_name, name, "down")]
            up = adapters[adapter_name(index, owner_name, name, "up")]
            setattr(owner, name, _Adapter(getattr(owner, name), down, up))


class ForkMLXScorer:
    """Score public Requests or records with an option-fork artifact on MLX."""

    def __init__(
        self,
        backbone: Qwen3_5TextModel,
        head: _ScalarHead,
        tokenizer: Any,
        *,
        device: mx.Device,
        model_id: str,
        max_length: int,
        max_batch_tokens: int,
        input_rendering: object,
        responsive: bool = True,
    ) -> None:
        if type(responsive) is not bool:
            raise ValueError("responsive must be a boolean")
        self.backbone, self.head, self.tokenizer = backbone, head, tokenizer
        self.device = device
        self.model_id = model_id
        self.max_length = max_length
        self.max_batch_tokens = max_batch_tokens
        self.input_rendering = validate_input_rendering_policy(input_rendering)
        self.responsive = responsive
        self._inference_lock = Lock()
        pad = tokenizer.pad_token_id
        self.pad_id = tokenizer.eos_token_id if pad is None else pad

    def encode_record(self, record: object) -> tuple[ForkQuestion, ...]:
        return encode_record(
            self.tokenizer,
            render_record(record, self.input_rendering),
            max_length=self.max_length,
        )

    def _hidden(self, ids: mx.array, caches: Sequence[_Cache]) -> mx.array:
        """Final-norm hidden states, advancing ``caches`` past ``ids``."""
        hidden = self.backbone.embed_tokens(ids)
        mask = create_attention_mask(hidden, None)
        # As in native MLX: short idle gaps let other applications submit GPU
        # work; they are not a hard QoS guarantee.
        pace = self.responsive and self.device.type == mx.gpu
        count = len(self.backbone.layers)
        for index, (layer, cache) in enumerate(
            zip(self.backbone.layers, caches, strict=True), 1
        ):
            hidden = layer(hidden, None if layer.is_linear else mask, cache)
            if pace and index % 4 == 0 and index < count:
                mx.eval(hidden)
                sleep(0.001)
        return cast(mx.array, self.backbone.norm(hidden))

    def _branch_caches(
        self, states: Sequence[Sequence[mx.array]], width: int
    ) -> list[_Cache]:
        caches: list[_Cache] = []
        for layer, state in zip(self.backbone.layers, states, strict=True):
            rows = [
                mx.broadcast_to(array, (width, *array.shape[1:])) for array in state
            ]
            if layer.is_linear:
                linear = ArraysCache(size=2)
                linear.state = rows
                caches.append(linear)
            else:
                attention = _new_kv_cache()
                attention.state = (rows[0], rows[1])
                caches.append(attention)
        return caches

    def question_logits(self, row: ForkQuestion) -> list[float]:
        with self._inference_lock, mx.stream(self.device):
            caches: list[_Cache] = [
                ArraysCache(size=2) if layer.is_linear else _new_kv_cache()
                for layer in self.backbone.layers
            ]
            # Only the caches are needed; lazy evaluation skips the last block.
            self._hidden(mx.array([row.prefix_ids]), caches)
            states = [tuple(cache.state) for cache in caches]
            mx.eval(states)
            prefix = len(row.prefix_ids)
            values = []
            for group in branch_groups(row.option_ids, prefix, self.max_batch_tokens):
                width = len(group)
                length = max(map(len, group))
                ids = mx.array(
                    [list(s) + [self.pad_id] * (length - len(s)) for s in group]
                )
                hidden = self._hidden(ids, self._branch_caches(states, width))
                last = hidden[mx.arange(width), mx.array([len(s) - 1 for s in group])]
                values.append(self.head(last))
                mx.eval(values[-1])
            logits = mx.concatenate(values)
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise ValueError("nonfinite fork logits")
            return cast(list[float], logits.tolist())

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


def load_fork_mlx(
    path: str | Path,
    *,
    device: str = "gpu",
    max_length: int | None = None,
    responsive: bool = True,
) -> ForkMLXScorer:
    """Load a validated option-fork artifact for MLX on Apple GPU or CPU."""
    if device not in {"cpu", "gpu"}:
        raise ValueError("MLX device must be cpu or gpu")
    if type(responsive) is not bool:
        raise ValueError("responsive must be a boolean")
    if device == "gpu" and not mx.metal.is_available():
        raise ValueError("MLX GPU inference requires available Apple Metal")
    root = Path(path).resolve()
    manifest = contract.validate_artifact(root)
    limit = manifest["max_length"] if max_length is None else max_length
    if not 0 < limit <= manifest["max_length"]:
        raise ValueError("max_length must be within the artifact's limit")
    text = _text_config(root, manifest)
    if device == "gpu" and text["linear_key_head_dim"] % 32:
        raise ValueError(
            "MLX GPU fork inference requires a linear key head width divisible by 32"
        )
    target = mx.Device(mx.gpu if device == "gpu" else mx.cpu)
    with mx.stream(target):
        model = Model(
            cast(Callable[[dict[str, Any]], ModelArgs], ModelArgs.from_dict)(
                {"model_type": "qwen3_5", "text_config": text}
            )
        )
        for layer in model.language_model.model.layers:
            if layer.is_linear:
                layer.linear_attn = _GatedDeltaNet(model.language_model.args)
        weights = _text_weights(_weights(root / "base/model.safetensors", head=False))
        # MLX-LM's sanitize moves the depthwise convolution axis and adds the
        # Transformers zero-centred RMSNorm offset only for checkpoints still
        # in the Transformers layout, which every fork base must be.
        if not any(
            key.endswith("conv1d.weight") and value.shape[1] == 1
            for key, value in weights.items()
        ):
            raise ValueError("fork base weights must use the Transformers layout")
        sanitize = cast(
            Callable[[dict[str, mx.array]], dict[str, mx.array]], model.sanitize
        )
        base = sanitize(weights)
        del weights
        model.load_weights(list(base.items()), strict=True)
        backbone = model.language_model.model
        adapters = _weights(root / "adapters.safetensors", head=True)
        _install_adapters(backbone, adapters, manifest["adapter"]["rank"])
        head = _ScalarHead(text["hidden_size"], manifest["head_width"])
        state = _weights(root / "head.safetensors", head=True)
        head.load_weights(
            [
                (_HEAD_NAMES.get(key[:6], key[:6]) + key[6:], value)
                for key, value in state.items()
            ],
            strict=True,
        )
        model.eval()
        head.eval()
        mx.eval(base, adapters, state)
    # Same tokenizer class as the PyTorch runtime; see wyat.loading.fork.
    tokenizer = Qwen3_5Tokenizer.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
    return ForkMLXScorer(
        backbone,
        head,
        tokenizer,
        device=target,
        model_id=manifest["model_id"],
        max_length=limit,
        max_batch_tokens=manifest["max_batch_tokens"],
        input_rendering=manifest.get("input_rendering"),
        responsive=responsive,
    )
