# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""PyTorch inference and artifact writer for Ouro decision artifacts.

Each question is encoded as one canonical shared context (state, instructions,
and UTF-8-sorted option menu) followed by one isolated candidate continuation
per option in caller order:

1. Stage 1 (Shared Context Prefill, ``context_steps=4``):
   The shared context runs through four unadapted Universal Transformer
   recurrent passes with all step-LoRA adapters disabled. Only the fourth
   recurrent pass's K/V cache (``ContextKVCache``) is retained for candidate
   scoring.
2. Stage 2 (Per-Candidate Continuation Scoring, ``candidate_steps=2``):
   Each candidate continuation runs through two recurrent passes reading the
   fourth context K/V bank at fixed positions ``P .. P+S-1``, with step-indexed
   rank-16/alpha-32 attention LoRA adapters (``StepLoRARouter``,
   ``StepLoRALinear``) active on pass 0 and pass 1 respectively.
3. Readout (``ScalarScoreHead``):
   An FP32 LayerNorm-Linear-GELU-Linear scalar head scores the final normalized
   hidden state of each candidate's last token.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import math
import os
import shutil
import sys
import types
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
    redirect_stderr,
    redirect_stdout,
)
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right
from transformers import AttentionInterface, PreTrainedTokenizerFast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from wyat import tasks
from wyat.core import ModelInputError, Request
from wyat.input_rendering import (
    input_rendering_policy,
    render_record,
    validate_input_rendering_policy,
)
from wyat.loading import ouro_artifact as contract
from wyat.loading.native import _request_record
from wyat.loading.native_artifact import file_sha256
from wyat.loading.ouro_artifact import (
    CandidateBatch,
    encode_record,
    reuse_context_prefix,
)
from wyat.resources import (
    ResourceGuard,
    checkpoint,
    current_guard,
)

CachePolicy = Literal["matched", "last"]

# Target device types each declared execution policy permits. CUDA artifacts
# also run on CPU or Apple MPS with an FP32 backbone. Cross-device agreement
# must be qualified per checkpoint; bitwise identity is not guaranteed.
# Other policy mismatches require explicit ``allow_device_policy_override``.
_DEVICE_POLICY_DEVICES: dict[str, frozenset[str]] = {
    "cuda-bf16-autocast": frozenset({"cuda", "cpu", "mps"}),
    "cpu-fp32": frozenset({"cpu"}),
    "mps-fp32": frozenset({"mps"}),
}


def _default_rope_init(
    config: Any, device: torch.device | None = None
) -> tuple[torch.Tensor, float]:
    """Standard RoPE inverse frequencies for Transformers 5.x compatibility."""
    base = float(config.rope_theta)
    dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(
                dtype=torch.float32
            )
            / dim
        )
    )
    return inv_freq, 1.0


@contextmanager
def _quiet_ouro_source_exec() -> Iterator[None]:
    """Silence Transformers 5.x auto_docstring and rope deprecation warnings."""
    root_logger = logging.getLogger()
    tx_logger = logging.getLogger("transformers")
    prev_root = root_logger.level
    prev_tx = tx_logger.level
    root_logger.setLevel(logging.CRITICAL)
    tx_logger.setLevel(logging.CRITICAL)
    try:
        with (
            warnings.catch_warnings(),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            yield
    finally:
        root_logger.setLevel(prev_root)
        tx_logger.setLevel(prev_tx)


def load_ouro_classes(
    directory: str | Path,
    expected_sha256: Mapping[str, str] | None = None,
) -> tuple[type[Any], type[Any]]:
    """Compile authenticated Ouro configuration and modeling source bytes."""
    root = Path(directory).resolve()
    sources: dict[str, bytes] = {}
    for name in contract.SOURCE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular Ouro source file: {name}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.get(name):
            raise ValueError(f"Ouro source SHA-256 mismatch: {name}")
        sources[name] = data

    identity = hashlib.sha256(
        b"".join(sources[name] for name in contract.SOURCE_FILES[1:])
    ).hexdigest()[:16]
    package_name = f"_wyat_ouro_{identity}"
    configuration_name = f"{package_name}.configuration_ouro"
    modeling_name = f"{package_name}.modeling_ouro"
    if modeling_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package
        try:
            with _quiet_ouro_source_exec():
                for name in contract.SOURCE_FILES[1:]:
                    module_name = f"{package_name}.{Path(name).stem}"
                    spec = importlib.util.spec_from_file_location(
                        module_name, root / name
                    )
                    if spec is None:
                        raise ValueError(f"unable to create module spec for {name}")
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    exec(
                        compile(sources[name], str(root / name), "exec"),
                        module.__dict__,
                    )
                    if name == "modeling_ouro.py":
                        module.__dict__["ROPE_INIT_FUNCTIONS"] = {
                            "default": _default_rope_init,
                            **ROPE_INIT_FUNCTIONS,
                        }
                        module.OuroRotaryEmbedding.compute_default_rope_parameters = (
                            staticmethod(_default_rope_init)
                        )
        except BaseException:
            for mod_name in (modeling_name, configuration_name, package_name):
                sys.modules.pop(mod_name, None)
            raise
    config_cls = sys.modules[configuration_name].OuroConfig
    if not hasattr(config_cls, "pad_token_id"):
        config_cls.pad_token_id = None
    return (
        config_cls,
        sys.modules[modeling_name].OuroForCausalLM,
    )


class StepLoRARouter:
    """Route a model's LoRA-wrapped projections to a zero-based recurrent step.

    Adapters are disabled (``active_step is None``) during shared-context prefill
    and enabled for step ``0 .. steps - 1`` during candidate evaluation.
    """

    def __init__(self, owner: nn.Module, steps: int) -> None:
        if type(steps) is not int or steps <= 0:
            raise ValueError("adapter steps must be a positive integer")
        self._owner = weakref.ref(owner)
        self.steps = steps
        self._route: ContextVar[int | None] = ContextVar(
            f"wyat_ouro_step_lora_{id(self)}", default=None
        )

    @property
    def loops(self) -> int:
        return self.steps

    @property
    def owner(self) -> nn.Module:
        owner = self._owner()
        if owner is None:
            raise RuntimeError("adapter owner no longer exists")
        return owner

    @property
    def active_step(self) -> int | None:
        return self._route.get()

    @property
    def active_loop(self) -> int | None:
        return self._route.get()

    @contextmanager
    def activate(self, step: int) -> Iterator[None]:
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError(f"adapter step must be an integer in [0, {self.steps})")
        token = self._route.set(step)
        try:
            yield
        finally:
            self._route.reset(token)

    @contextmanager
    def disabled(self) -> Iterator[None]:
        token = self._route.set(None)
        try:
            yield
        finally:
            self._route.reset(token)


class StepLoRALinear(nn.Module):
    """Frozen linear projection with independent FP32 LoRA factors per step."""

    def __init__(
        self,
        base: nn.Linear,
        router: StepLoRARouter,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.router = router
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.ParameterList(
            nn.Parameter(
                torch.empty(
                    rank,
                    base.in_features,
                    device=base.weight.device,
                    dtype=torch.float32,
                )
            )
            for _ in range(router.steps)
        )
        self.lora_B = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    base.out_features,
                    rank,
                    device=base.weight.device,
                    dtype=torch.float32,
                )
            )
            for _ in range(router.steps)
        )
        for factor in self.lora_A:
            nn.init.kaiming_uniform_(factor, a=math.sqrt(5))

    @property
    def controller(self) -> StepLoRARouter:
        return self.router

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden)
        step = self.router.active_step
        if step is None:
            return cast(torch.Tensor, base_output)
        # The guard keeps the factors FP32 under autocast. Without autocast it
        # changes no op, but entering it on every projection costs dispatch.
        guard = (
            torch.autocast(device_type=hidden.device.type, enabled=False)
            if torch.is_autocast_enabled(hidden.device.type)
            else nullcontext()
        )
        with guard:
            update = F.linear(
                F.linear(hidden.float(), self.lora_A[step]), self.lora_B[step]
            )
        return cast(
            torch.Tensor, base_output + (update * self.scaling).to(base_output.dtype)
        )


def attach_step_lora(
    model: nn.Module,
    *,
    rank: int = contract.ADAPTER_RANK,
    alpha: float = contract.ADAPTER_ALPHA,
    steps: int = contract.CANDIDATE_STEPS,
    targets: Sequence[str] = contract.ADAPTER_TARGETS,
) -> StepLoRARouter:
    """Freeze base parameters and wrap target projections with ``StepLoRALinear``."""
    if type(rank) is not int or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("alpha must be finite and positive")
    if type(steps) is not int or steps <= 0:
        raise ValueError("adapter steps must be a positive integer")
    if getattr(model, "_step_lora_router", None) is not None or any(
        isinstance(module, StepLoRALinear) for module in model.modules()
    ):
        raise ValueError("StepLoRA is already attached")
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError("Ouro model has no decoder layers")
    parsed_targets = [tuple(t.split(".")) for t in targets]
    selected: list[tuple[nn.Module, str, nn.Linear]] = []
    for index, layer in enumerate(layers):
        for owner_name, proj_name in parsed_targets:
            owner = getattr(layer, owner_name, None)
            if not isinstance(owner, nn.Module):
                raise TypeError(f"model.layers.{index}.{owner_name} must be a module")
            projection = getattr(owner, proj_name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"model.layers.{index}.{owner_name}.{proj_name} must be nn.Linear"
                )
            selected.append((owner, proj_name, projection))
    router = StepLoRARouter(model, steps)
    replacements = [
        (owner, proj_name, StepLoRALinear(projection, router, rank, alpha))
        for owner, proj_name, projection in selected
    ]
    model.requires_grad_(False)
    for owner, proj_name, replacement in replacements:
        setattr(owner, proj_name, replacement)
    model._step_lora_router = router  # type: ignore[assignment]
    return router


class ScalarScoreHead(nn.Module):
    """Shared FP32 candidate score head applied after the final candidate pass."""

    def __init__(self, hidden_size: int, width: int = contract.HEAD_WIDTH) -> None:
        super().__init__()
        if (
            type(hidden_size) is not int
            or type(width) is not int
            or hidden_size <= 0
            or width <= 0
        ):
            raise ValueError("scalar head dimensions must be positive integers")
        self.hidden_size = hidden_size
        self.width = width
        self.norm = nn.LayerNorm(hidden_size, dtype=torch.float32)
        self.projection = nn.Linear(hidden_size, width, dtype=torch.float32)
        self.output = nn.Linear(width, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("scalar head expects hidden[K, hidden_size]")
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return cast(
                torch.Tensor,
                self.output(F.gelu(self.projection(self.norm(hidden.float())))),
            )


@dataclass(frozen=True)
class LayerKVCache:
    """Post-RoPE key and value tensors for one decoder layer."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class ContextKVCache:
    """Retained per-step or final-step K/V banks for a shared question context."""

    banks: tuple[tuple[LayerKVCache, ...], ...]
    length: int
    policy: CachePolicy
    owner: nn.Module


def _native_backbone(model: nn.Module) -> Any:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("Ouro model must expose .model")
    if backbone.total_ut_steps != 4 or backbone.config.total_ut_steps != 4:
        raise ValueError("Ouro requires exactly four native recurrent steps")
    if backbone.has_sliding_layers or any(
        layer.attention_type != "full_attention" for layer in backbone.layers
    ):
        raise ValueError("Ouro decision scoring requires full attention layers")
    if backbone.config.attention_dropout != 0:
        raise ValueError("Ouro decision scoring requires zero attention dropout")
    if backbone.config._attn_implementation not in {"eager", "sdpa"}:
        raise ValueError("Ouro decision scoring supports eager or SDPA attention")
    return backbone


def _validate_token_tensor(backbone: Any, ids: torch.Tensor) -> None:
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("expected nonempty unpadded token IDs with shape [1, S]")
    if ids.dtype != torch.long:
        raise ValueError("token IDs must have dtype torch.long")
    if ids.device != backbone.embed_tokens.weight.device:
        raise ValueError("token IDs and backbone must share a device")
    if torch.any((ids < 0) | (ids >= backbone.config.vocab_size)):
        raise ValueError("token IDs are outside the vocabulary")
    if ids.shape[1] > backbone.config.max_position_embeddings:
        raise ValueError("input exceeds max_position_embeddings")


def _causal_mask(hidden: torch.Tensor, context_length: int) -> torch.Tensor:
    length = hidden.shape[1]
    query = torch.arange(context_length, context_length + length, device=hidden.device)
    keys = torch.arange(context_length + length, device=hidden.device)
    blocked = keys[None, :] > query[:, None]
    mask = torch.zeros(
        (length, context_length + length), device=hidden.device, dtype=hidden.dtype
    )
    return mask.masked_fill(blocked, torch.finfo(hidden.dtype).min)[None, None]


_StepGeometry = tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]


class _GeometryCache:
    """Positions, rotary embeddings, and causal mask per step geometry.

    The geometry of a recurrent pass only depends on ``(context_length,
    sequence_length, dtype, device)``, not on the recurrent step or candidate.
    Sharing one cache across the steps of a prefill and across every candidate
    of a question removes the duplicate mask/rotary construction (and the
    duplicate allocations it caused) without changing any value fed to the
    native decoder layers.
    """

    def __init__(self, backbone: Any) -> None:
        self._backbone = backbone
        self._entries: dict[
            tuple[int, int, torch.dtype, torch.device], _StepGeometry
        ] = {}

    def get(self, hidden: torch.Tensor, context_length: int) -> _StepGeometry:
        length = int(hidden.shape[1])
        key = (context_length, length, hidden.dtype, hidden.device)
        entry = self._entries.get(key)
        if entry is None:
            positions = torch.arange(
                context_length, context_length + length, device=hidden.device
            )
            rotary = self._backbone.rotary_emb(hidden, positions[None])
            entry = (positions, rotary, _causal_mask(hidden, context_length))
            self._entries[key] = entry
        return entry


class _CaptureStepKV:
    """Capture K/V tensors during one recurrent context pass."""

    def __init__(self, layers: int, step: int, retain: bool) -> None:
        self.layers = layers
        self.step = step
        self.retain = retain
        self.entries: list[LayerKVCache] = []
        self.count = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.step * self.layers + self.count:
            raise ValueError("unexpected native cache update order")
        self.count += 1
        if self.retain:
            self.entries.append(LayerKVCache(key, value))
        return key, value


class _ReadAndCaptureStepKV:
    """Continue from retained state-block K/V and capture the extended K/V bank."""

    def __init__(
        self, prior_bank: tuple[LayerKVCache, ...], step: int, retain: bool
    ) -> None:
        self.prior_bank = prior_bank
        self.step = step
        self.retain = retain
        self.entries: list[LayerKVCache] = []
        self.count = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.step * len(self.prior_bank) + self.count:
            raise ValueError("unexpected native cache update order")
        prior = self.prior_bank[self.count]
        self.count += 1
        full_key = torch.cat((prior.key, key), dim=2)
        full_value = torch.cat((prior.value, value), dim=2)
        if self.retain:
            self.entries.append(LayerKVCache(full_key, full_value))
        return full_key, full_value


class _KVWorkspace:
    """Preallocated combined context+candidate K/V buffer for one recurrent pass.

    One buffer pair is reused for every layer of a pass because a layer's
    attention consumes it before the next layer runs, which keeps the workspace
    at two ``[batch, heads, context + candidate, head_dim]`` tensors instead of
    one pair per layer. Each layer pass copies its own retained context prefix
    and overwrites the candidate suffix in place, so no per-candidate
    ``torch.cat`` allocation is needed.
    """

    def __init__(self, bank: tuple[LayerKVCache, ...], step: int) -> None:
        self.bank = bank
        self.step = step
        self.count = 0
        self.context_length = int(bank[0].key.shape[2])
        self.key_buffer: torch.Tensor | None = None
        self.value_buffer: torch.Tensor | None = None

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.step * len(self.bank) + self.count:
            raise ValueError("unexpected native cache update order")
        total = self.context_length + int(key.shape[2])
        key_buffer = self.key_buffer
        if key_buffer is None or key_buffer.shape[2] != total:
            shape = (key.shape[0], key.shape[1], total, key.shape[3])
            key_buffer = torch.empty(shape, device=key.device, dtype=key.dtype)
            self.key_buffer = key_buffer
            self.value_buffer = torch.empty(
                shape, device=value.device, dtype=value.dtype
            )
        value_buffer = self.value_buffer
        if value_buffer is None:
            raise RuntimeError("K/V workspace buffers are inconsistent")
        context = self.bank[self.count]
        key_buffer[:, :, : self.context_length].copy_(context.key)
        value_buffer[:, :, : self.context_length].copy_(context.value)
        key_buffer[:, :, self.context_length :].copy_(key)
        value_buffer[:, :, self.context_length :].copy_(value)
        self.count += 1
        return (
            key_buffer[:, :, :total],
            value_buffer[:, :, :total],
        )


class _WorkspacePool:
    """Reuse preallocated K/V workspaces across candidate passes."""

    def __init__(self) -> None:
        self._workspaces: dict[tuple[int, int, int], _KVWorkspace] = {}

    def acquire(
        self,
        bank: tuple[LayerKVCache, ...],
        step: int,
        batch: int,
        candidate_length: int,
    ) -> _KVWorkspace:
        key = (step, batch, candidate_length)
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = _KVWorkspace(bank, step)
            self._workspaces[key] = workspace
        else:
            workspace.count = 0
        return workspace


class _ReadContextKV:
    """Provide retained context K/V with the current candidate suffix appended."""

    def __init__(
        self,
        bank: tuple[LayerKVCache, ...],
        step: int,
        pool: _WorkspacePool | None = None,
    ) -> None:
        self.bank = bank
        self.step = step
        self.pool = pool
        self.workspace: _KVWorkspace | None = None

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self.workspace
        if workspace is None:
            pool = self.pool
            if pool is None:
                workspace = _KVWorkspace(self.bank, self.step)
            else:
                workspace = pool.acquire(
                    self.bank,
                    self.step,
                    int(key.shape[0]),
                    int(key.shape[2]),
                )
            self.workspace = workspace
        return workspace.update(key, value, layer_idx)


def _run_recurrent_step(
    backbone: Any,
    hidden: torch.Tensor,
    context_length: int,
    step: int,
    cache: _CaptureStepKV | _ReadAndCaptureStepKV | _ReadContextKV | _OwnCandidateKV,
    geometry: _GeometryCache,
) -> torch.Tensor:
    positions, rotary, mask = geometry.get(hidden, context_length)
    for layer in backbone.layers:
        checkpoint()
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=positions[None],
            past_key_value=cache,
            use_cache=True,
            cache_position=positions,
            position_embeddings=rotary,
            current_ut=step,
        )
    return cast(torch.Tensor, backbone.norm(hidden))


# Chunked context prefill: the context execution of a v2 artifact
# (``execution.context_prefill``). The same arithmetic as keme's canonical
# ``prefill_context``, which trained and judged those artifacts.
ContextAttention = Literal["native", "lean", "flash"]


class _ChunkedContextKV:
    """Preallocated per-layer K/V written one query chunk at a time."""

    def __init__(self, layer_idx: int, length: int) -> None:
        self.layer_idx = layer_idx
        self.length = length
        self.key: torch.Tensor | None = None
        self.value: torch.Tensor | None = None
        self.key32: torch.Tensor | None = None
        self.value32: torch.Tensor | None = None
        self.start = 0
        self.end = 0

    def begin(self, start: int, end: int) -> None:
        self.start, self.end = start, end

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.layer_idx:
            raise ValueError("unexpected native cache update order")
        if key.shape[2] != self.end - self.start:
            raise ValueError("chunk K/V length differs from the scheduled chunk")
        if self.key is None:
            shape = (*key.shape[:2], self.length, key.shape[3])
            self.key, self.value = key.new_empty(shape), value.new_empty(shape)
        assert self.value is not None
        self.key[:, :, self.start : self.end].copy_(key)
        self.value[:, :, self.start : self.end].copy_(value)
        return self.key[:, :, : self.end], self.value[:, :, : self.end]

    def fp32(self, root: float) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 K (pre-scaled by sqrt(scale)) and V, converting only this chunk."""
        assert self.key is not None and self.value is not None
        if self.key32 is None:
            self.key32 = self.key.new_empty(self.key.shape, dtype=torch.float32)
            self.value32 = self.value.new_empty(self.value.shape, dtype=torch.float32)
        assert self.value32 is not None
        new = slice(self.start, self.end)
        torch.mul(self.key[:, :, new].float(), root, out=self.key32[:, :, new])
        self.value32[:, :, new].copy_(self.value[:, :, new])
        return self.key32[:, :, : self.end], self.value32[:, :, : self.end]


_ACTIVE_CONTEXT_WRITER: ContextVar[_ChunkedContextKV | None] = ContextVar(
    "_ACTIVE_CONTEXT_WRITER", default=None
)


def _lean_context_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """PyTorch's MATH SDPA for one context chunk, without its per-call waste.

    MATH upcasts BF16 to FP32, scales q and K by sqrt(scale), adds the mask,
    applies ``_safe_softmax`` and multiplies by V. Here K/V are converted and
    scaled once per new chunk instead of over the whole context on every call;
    the mask is added only on the diagonal block (adding 0.0 elsewhere is the
    identity); plain softmax equals ``_safe_softmax`` because a causal row is
    never fully masked. The same FP32 matmuls run on the same shapes, so
    results are bit-identical to MATH.
    """
    writer = _ACTIVE_CONTEXT_WRITER.get()
    if writer is None or dropout or attention_mask is None or scaling is None:
        raise ValueError("lean attention runs only inside chunked context prefill")
    if key.shape[2] != writer.end or query.shape[2] != writer.end - writer.start:
        raise ValueError("lean attention received another chunk's tensors")
    root = math.sqrt(scaling)
    with torch.autocast(query.device.type, enabled=False):
        key32, value32 = writer.fp32(root)
        scores = torch.matmul(query.float() * root, key32.transpose(-2, -1))
        block = slice(writer.start, writer.end)
        scores[..., block].add_(attention_mask[..., block])
        output = torch.matmul(torch.softmax(scores, dim=-1), value32)
    return output.to(query.dtype).transpose(1, 2).contiguous(), None


def _flash_context_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Fused BF16 causal attention for a chunked frozen context: not MATH-exact."""
    if dropout:
        raise ValueError("context attention has no dropout")
    mask = causal_lower_right(query.shape[2], key.shape[2])
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scaling
        )
    return output.transpose(1, 2).contiguous(), None


AttentionInterface.register("wyat_context_lean", _lean_context_attention)
AttentionInterface.register("wyat_context_flash", _flash_context_attention)


@contextmanager
def _context_attention(native: Any, attention: ContextAttention) -> Iterator[None]:
    if attention not in ("native", "lean", "flash"):
        raise ValueError("context attention must be native, lean or flash")
    if attention != "native" and native.config._attn_implementation != "sdpa":
        raise ValueError("lean and flash context attention replace SDPA attention")
    previous = native.config._attn_implementation
    if attention != "native":
        native.config._attn_implementation = f"wyat_context_{attention}"
    try:
        yield
    finally:
        native.config._attn_implementation = previous


@contextmanager
def _without_uninitialized_fill() -> Iterator[None]:
    """Disable deterministic mode's fill of the K/V buffers before they are written."""
    previous = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]
    torch.utils.deterministic.fill_uninitialized_memory = False  # type: ignore[attr-defined]
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous  # type: ignore[attr-defined]


def _place_context_layer(
    key: torch.Tensor, value: torch.Tensor, device: str | torch.device | None
) -> LayerKVCache:
    target = key.device if device is None else torch.device(device)
    if target == key.device:
        return LayerKVCache(key, value)
    pinned = target.type == "cpu" and key.is_cuda
    host = [
        torch.empty(t.shape, dtype=t.dtype, device=target, pin_memory=pinned)
        for t in (key, value)
    ]
    host[0].copy_(key, non_blocking=pinned)
    host[1].copy_(value, non_blocking=pinned)
    return LayerKVCache(host[0], host[1])


def _chunked_prefill_context(
    native: Any,
    owner: nn.Module,
    context_ids: torch.Tensor,
    *,
    chunk: int,
    bank_device: str | torch.device | None,
    attention: ContextAttention,
) -> ContextKVCache:
    length = int(context_ids.shape[1])
    layers = len(native.layers)
    router = getattr(owner, "_step_lora_router", None)
    spans = [(start, min(start + chunk, length)) for start in range(0, length, chunk)]
    bank: list[LayerKVCache] = []
    with (
        torch.no_grad(),
        _without_uninitialized_fill(),
        router.disabled() if router is not None else nullcontext(),
        _context_attention(native, attention),
    ):
        hidden = native.embed_tokens(context_ids)
        positions = [
            torch.arange(start, end, device=hidden.device) for start, end in spans
        ]
        rotary = [native.rotary_emb(hidden, position[None]) for position in positions]
        # Flash attention constructs its own causal bias; retaining these full
        # masks would waste about 2 GiB for a 32K-token FP32 context.
        masks: list[torch.Tensor | None] = (
            [None] * len(spans)
            if attention == "flash"
            else [_causal_mask(hidden[:, start:end], start) for start, end in spans]
        )
        for step in range(contract.CONTEXT_STEPS):
            for layer_index, layer in enumerate(native.layers):
                checkpoint()
                writer = _ChunkedContextKV(step * layers + layer_index, length)
                output = torch.empty_like(hidden)
                for (start, end), position, rope, mask in zip(
                    spans, positions, rotary, masks, strict=True
                ):
                    writer.begin(start, end)
                    checkpoint()
                    token = _ACTIVE_CONTEXT_WRITER.set(writer)
                    try:
                        output[:, start:end] = layer(
                            hidden[:, start:end],
                            attention_mask=mask,
                            position_ids=position[None],
                            past_key_value=writer,
                            use_cache=True,
                            cache_position=position,
                            position_embeddings=rope,
                            current_ut=step,
                        )
                    finally:
                        _ACTIVE_CONTEXT_WRITER.reset(token)
                if writer.key is None or writer.value is None:
                    raise ValueError("native layer skipped its cache update")
                if step == contract.CONTEXT_STEPS - 1:
                    bank.append(
                        _place_context_layer(writer.key, writer.value, bank_device)
                    )
                hidden = output
                del writer
            hidden = native.norm(hidden)
    return ContextKVCache((tuple(bank),), length, "last", owner)


def prefill_context(
    backbone: nn.Module,
    context_ids: torch.Tensor,
    *,
    policy: CachePolicy = "last",
    context_steps: int = contract.CONTEXT_STEPS,
    chunk: int | None = None,
    bank_device: str | torch.device | None = None,
    attention: ContextAttention = "native",
) -> ContextKVCache:
    """Run shared-context recurrent steps with all step-LoRA adapters disabled.

    A context longer than ``chunk`` is prefilled in ``chunk``-token pieces with
    ``attention``: ``native``, its bit-identical ``lean`` rewrite, or fused
    ``flash`` (BF16 rounding differs), retaining only the last step's K/V, in
    ``bank_device`` memory when given. A context that fits in one chunk always
    takes the exact single pass.
    """
    if policy not in {"matched", "last"}:
        raise ValueError("unknown cache policy")
    if type(context_steps) is not int or context_steps not in {1, 2, 3, 4}:
        raise ValueError("context_steps must be 1, 2, 3, or 4")
    if policy == "last" and context_steps != 4:
        raise ValueError("last policy requires four context steps")
    if chunk is not None and (type(chunk) is not int or chunk <= 0):
        raise ValueError("chunk must be a positive integer or None")
    native = _native_backbone(backbone)
    _validate_token_tensor(native, context_ids)
    if chunk is not None and context_ids.shape[1] > chunk:
        if policy != "last":
            raise ValueError("chunked context prefill requires the last policy")
        return _chunked_prefill_context(
            native,
            backbone,
            context_ids,
            chunk=chunk,
            bank_device=bank_device,
            attention=attention,
        )
    banks: list[tuple[LayerKVCache, ...]] = []
    router = getattr(backbone, "_step_lora_router", None)
    geometry = _GeometryCache(native)
    with router.disabled() if router is not None else nullcontext():
        hidden = native.embed_tokens(context_ids)
        for step in range(context_steps):
            cache = _CaptureStepKV(
                len(native.layers), step, policy == "matched" or step == 3
            )
            hidden = _run_recurrent_step(native, hidden, 0, step, cache, geometry)
            if cache.entries:
                banks.append(tuple(cache.entries))
    return ContextKVCache(tuple(banks), int(context_ids.shape[1]), policy, backbone)


def _prepare_candidate_pass(
    native: Any,
    backbone: nn.Module,
    context_cache: ContextKVCache,
    candidate_ids: torch.Tensor,
    candidate_steps: int,
) -> Any:
    """Validate one candidate continuation and return its step-LoRA router."""
    if context_cache.owner is not backbone:
        raise ValueError("context cache belongs to another backbone")
    if context_cache.policy == "matched" and len(context_cache.banks) < candidate_steps:
        raise ValueError("matched context cache has fewer banks than candidate_steps")
    router = getattr(backbone, "_step_lora_router", None)
    if router is not None and (
        router.owner is not backbone or router.steps < candidate_steps
    ):
        raise ValueError("step LoRA router does not cover this backbone and step count")
    _validate_token_tensor(native, candidate_ids)
    if (
        context_cache.length + candidate_ids.shape[1]
        > native.config.max_position_embeddings
    ):
        raise ValueError("complete candidate sequence exceeds max_position_embeddings")
    return router


def _candidate_hidden_steps(
    native: Any,
    router: Any,
    context_length: int,
    candidate_ids: torch.Tensor,
    context_cache: ContextKVCache,
    candidate_steps: int,
    geometry: _GeometryCache,
    pool: _WorkspacePool,
) -> torch.Tensor:
    """Run validated candidate recurrent passes with shared geometry/workspace."""
    hidden = native.embed_tokens(candidate_ids)
    for step in range(candidate_steps):
        bank = context_cache.banks[step if context_cache.policy == "matched" else 0]
        with router.activate(step) if router is not None else nullcontext():
            hidden = _run_recurrent_step(
                native,
                hidden,
                context_length,
                step,
                _ReadContextKV(bank, step, pool),
                geometry,
            )
    return cast(torch.Tensor, hidden)


def run_candidate_steps(
    backbone: nn.Module,
    context_cache: ContextKVCache,
    candidate_ids: torch.Tensor,
    *,
    candidate_steps: int = contract.CANDIDATE_STEPS,
) -> torch.Tensor:
    """Evaluate one candidate continuation across ``candidate_steps`` passes."""
    if type(candidate_steps) is not int or candidate_steps not in {1, 2, 3, 4}:
        raise ValueError("candidate_steps must be 1, 2, 3, or 4")
    native = _native_backbone(backbone)
    router = _prepare_candidate_pass(
        native, backbone, context_cache, candidate_ids, candidate_steps
    )
    return _candidate_hidden_steps(
        native,
        router,
        context_cache.length,
        candidate_ids,
        context_cache,
        candidate_steps,
        _GeometryCache(native),
        _WorkspacePool(),
    )


CandidateBatching = Literal["shared_context", "per_candidate"]
# Inference scores candidates in shared-context batches by default; the
# per-candidate path stays bitwise identical to ``run_candidate_steps``.
CANDIDATE_BATCHING: CandidateBatching = "shared_context"
# A batch grows while its FP32 scores and their softmax,
# 8 · heads · packed · (context + packed) bytes, stay within this budget.
SHARED_SCORE_BUDGET_BYTES = 512 * 2**20
_SHARED_CONTEXT_KIND = "wyat_shared_context_batch"
_SHARED_CONTEXT_LAYER: ContextVar[LayerKVCache | None] = ContextVar(
    "wyat_shared_context_layer", default=None
)
# The block-diagonal causal mask of the batch the layers are running. It is
# built once per batch instead of once per decoder-layer visit; its values are
# identical to the mask that computation produced.
_SHARED_CONTEXT_BLOCK: ContextVar[torch.Tensor | None] = ContextVar(
    "wyat_shared_context_block", default=None
)


def _candidate_batching(value: object) -> CandidateBatching:
    """``"shared_context"`` or ``"per_candidate"``.

    ``True``/``False``, the earlier opt-in flag the profiling scripts in
    ``scripts/ouro_speed`` pass, map to those two modes.
    """
    if value is True or value == "shared_context":
        return "shared_context"
    if value is False or value == "per_candidate":
        return "per_candidate"
    raise ValueError("candidate_batching must be 'shared_context' or 'per_candidate'")


def _shared_context_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Attention of a candidate batch over one shared context and its own tokens.

    Per head, the batch's candidate tokens follow the context in one packed
    sequence; a block-diagonal causal mask lets each token see the whole
    context and its own candidate's earlier tokens only. The context K/V is
    read once for the batch instead of copied into every row. FP32 softmax;
    equal to attention over the concatenated K/V in exact arithmetic, with
    different rounding than the native kernels. A context layer held in host
    memory (a v2 artifact's long context) moves to the device here, one layer
    at a time; the copy changes no value.
    """
    context = _SHARED_CONTEXT_LAYER.get()
    blocked = _SHARED_CONTEXT_BLOCK.get()
    if (
        context is None
        or blocked is None
        or dropout
        or scaling is None
        or attention_mask is None
    ):
        raise ValueError("shared-context attention runs only inside its own batches")
    batch, heads, length, dim = query.shape
    if key.shape[1] != heads or context.key.shape[:2] != (1, heads):
        raise ValueError("shared-context attention needs one K/V head per query head")
    if context.key.device != query.device:
        context = LayerKVCache(
            context.key.to(query.device, non_blocking=True),
            context.value.to(query.device, non_blocking=True),
        )
    packed = batch * length
    if tuple(blocked.shape) != (packed, packed) or blocked.device != query.device:
        raise ValueError("shared-context mask does not match the packed batch")
    with torch.autocast(query.device.type, enabled=False):

        def pack(states: torch.Tensor) -> torch.Tensor:
            # [batch, heads, length, dim] -> [heads, batch · length, dim]
            return states.float().transpose(0, 1).reshape(heads, packed, dim)

        keys = torch.cat((context.key[0].float(), pack(key)), dim=1)
        values = torch.cat((context.value[0].float(), pack(value)), dim=1)
        scores = torch.matmul(pack(query) * scaling, keys.transpose(-1, -2))
        scores[..., context.key.shape[2] :].masked_fill_(blocked, float("-inf"))
        output = torch.matmul(scores.softmax(-1), values)
    # [heads, batch · length, dim] -> [batch, length, heads, dim], as SDPA returns.
    output = output.reshape(heads, batch, length, dim).permute(1, 2, 0, 3)
    return output.to(query.dtype).contiguous(), None


AttentionInterface.register(_SHARED_CONTEXT_KIND, _shared_context_attention)


class _OwnCandidateKV:
    """Cache stand-in for a shared-context batch.

    Each layer keeps its own candidate K/V; the layer's context K/V is published
    for the attention instead of being concatenated into every row.
    """

    def __init__(self, bank: tuple[LayerKVCache, ...], step: int) -> None:
        self.bank = bank
        self.step = step
        self.count = 0

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        cache_kwargs: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx != self.step * len(self.bank) + self.count:
            raise ValueError("unexpected native cache update order")
        layer = self.bank[self.count]
        if layer.key.device != key.device:
            layer = LayerKVCache(
                layer.key.to(key.device, non_blocking=True),
                layer.value.to(key.device, non_blocking=True),
            )
        _SHARED_CONTEXT_LAYER.set(layer)
        self.count += 1
        return key, value


@contextmanager
def _shared_context_layers(
    native: Any, blocked: torch.Tensor | None = None
) -> Iterator[None]:
    previous = native.config._attn_implementation
    native.config._attn_implementation = _SHARED_CONTEXT_KIND
    token = _SHARED_CONTEXT_LAYER.set(None)
    block_token = _SHARED_CONTEXT_BLOCK.set(blocked)
    try:
        yield
    finally:
        native.config._attn_implementation = previous
        _SHARED_CONTEXT_LAYER.reset(token)
        _SHARED_CONTEXT_BLOCK.reset(block_token)


def shared_context_batches(
    lengths: Sequence[int],
    context_length: int,
    heads: int,
    budget_bytes: int = SHARED_SCORE_BUDGET_BYTES,
) -> list[list[int]]:
    """Candidate indices per batch, shortest first, within the score budget.

    Shortest first keeps right padding small. A batch takes the next candidate
    while its packed scores fit ``budget_bytes``; a candidate that alone exceeds
    the budget still gets a batch of its own.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        packed = (len(current) + 1) * lengths[index]
        if current and 8 * heads * packed * (context_length + packed) > budget_bytes:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


def _shared_context_padded(
    candidates: Sequence[torch.Tensor],
    batch: Sequence[int],
    lengths: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Right-padded token IDs for one shared-context batch.

    The real tokens keep their order; padding is zero, which the block mask
    keeps every real token blind to.
    """
    width = max(lengths[index] for index in batch)
    padded = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    columns = torch.arange(width, device=device)
    padded_lengths = torch.tensor(
        [lengths[index] for index in batch], dtype=torch.long, device=device
    )
    padded[columns[None, :] < padded_lengths[:, None]] = torch.cat(
        [candidates[index][0] for index in batch]
    )
    return padded


def _shared_context_block(count: int, width: int, device: torch.device) -> torch.Tensor:
    """Block-diagonal causal mask of a packed shared-context batch.

    ``True`` marks a forbidden query/key pair: another candidate's tokens and
    a token's own later positions.
    """
    packed = count * width
    token = torch.arange(packed, device=device)
    row, position = token // width, token % width
    return (row[:, None] != row[None, :]) | (position[None, :] > position[:, None])


def score_candidates_shared_context(
    backbone: nn.Module,
    head: nn.Module,
    context_cache: ContextKVCache,
    candidates: Sequence[torch.Tensor],
    *,
    candidate_steps: int = contract.CANDIDATE_STEPS,
    score_budget_bytes: int = SHARED_SCORE_BUDGET_BYTES,
) -> torch.Tensor:
    """Score candidates in right-padded batches that share one context read.

    The default path of ``OuroScorer``. Every candidate is validated before any
    runs. Every linear runs once per batch instead of once per candidate, and
    attention reads the context K/V once per layer for the whole batch
    (``_shared_context_attention``). It is not bitwise identical to the
    per-candidate path: it rounds differently, and is held to the same ranking
    within the execution tolerance instead.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if type(candidate_steps) is not int or candidate_steps not in {1, 2, 3, 4}:
        raise ValueError("candidate_steps must be 1, 2, 3, or 4")
    native = _native_backbone(backbone)
    heads = int(native.config.num_attention_heads)
    if int(native.config.num_key_value_heads) != heads:
        raise ValueError("shared-context attention needs one K/V head per query head")
    routers = [
        _prepare_candidate_pass(
            native, backbone, context_cache, candidate, candidate_steps
        )
        for candidate in candidates
    ]
    router = routers[-1]
    lengths = [int(candidate.shape[1]) for candidate in candidates]
    geometry = _GeometryCache(native)
    endpoints: list[torch.Tensor | None] = [None] * len(candidates)
    batches = shared_context_batches(
        lengths, context_cache.length, heads, score_budget_bytes
    )
    device = candidates[0].device
    for batch in batches:
        padded = _shared_context_padded(candidates, batch, lengths, device)
        blocked = _shared_context_block(len(batch), padded.shape[1], device)
        # Causal masking keeps every real token blind to the padding after it.
        hidden = native.embed_tokens(padded)
        with _shared_context_layers(native, blocked):
            for step in range(candidate_steps):
                bank = context_cache.banks[
                    step if context_cache.policy == "matched" else 0
                ]
                with router.activate(step) if router is not None else nullcontext():
                    hidden = _run_recurrent_step(
                        native,
                        hidden,
                        context_cache.length,
                        step,
                        _OwnCandidateKV(bank, step),
                        geometry,
                    )
        for slot, index in enumerate(batch):
            endpoints[index] = hidden[slot : slot + 1, lengths[index] - 1]
    scores = head(torch.cat([e for e in endpoints if e is not None], dim=0))
    if scores.shape != (len(candidates), 1):
        raise ValueError("scalar score head must return shape [K, 1]")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite Ouro candidate logits")
    return cast(torch.Tensor, scores[:, 0])


# CUDA-graph replay of one short shared-context candidate batch. The graph
# records exactly the kernels and fixed buffers the eager batch uses, so a
# replay changes no arithmetic; every geometry is captured once and kept only
# after its replay matches the eager logits bit for bit. Capture is attempted
# on the second sighting of a geometry, so a shape that never repeats pays
# nothing. Any capture error, replay error or bit mismatch disables the
# feature for the scorer and leaves the eager path in charge.
GRAPH_MAX_CONTEXT = 1024
GRAPH_MAX_CANDIDATES = 32
GRAPH_CACHE_ENTRIES = 64
# Capture is attempted only after this many eager sightings of the same
# geometry, so a shape that never repeats pays nothing. The measured capture
# cost is ~158 ms against ~20 ms saved per replay, so a geometry needs several
# guaranteed replays before a capture pays for itself.
GRAPH_CAPTURE_SIGHTINGS = 3
# A host-cached context bank may stay on the GPU instead when it fits with this
# much free-memory headroom for the prefill and candidate workspaces. Placement
# is a memory policy, not arithmetic: values are identical either way.
DEVICE_BANK_RESERVE_BYTES = 3 * 1024**3


class _SharedBatchGraph:
    """One captured candidate batch and the static buffers it replays on."""

    # Every tensor below is read by the captured kernels; each one is retained
    # here so its storage cannot be freed to the allocator and reused before a
    # replay (a captured graph holds raw pointers, not tensor references).
    __slots__ = (
        "graph",
        "padded",
        "mask",
        "blocked",
        "geometry",
        "scores",
        "lengths",
        "batch",
        "bank",
    )

    def __init__(
        self,
        graph: Any,
        padded: torch.Tensor,
        mask: torch.Tensor,
        blocked: torch.Tensor,
        geometry: _GeometryCache,
        scores: torch.Tensor,
        lengths: list[int],
        batch: list[int],
        bank: tuple[LayerKVCache, ...],
    ) -> None:
        self.graph = graph
        self.padded = padded
        self.mask = mask
        self.blocked = blocked
        self.geometry = geometry
        self.scores = scores
        self.lengths = lengths
        self.batch = batch
        self.bank = bank


class _ExactCandidateGraphs:
    """Capture and replay short shared-context candidate batches on CUDA."""

    def __init__(self, backbone: Any, head: Any) -> None:
        self.backbone = backbone
        self.head = head
        self.native = _native_backbone(backbone)
        self.enabled = True
        self.replays = 0
        self.captures = 0
        self.mismatches = 0
        self.errors = 0
        self.evictions = 0
        self._graphs: OrderedDict[tuple[Any, ...], _SharedBatchGraph] = OrderedDict()
        self._seen: OrderedDict[tuple[Any, ...], int] = OrderedDict()
        self._keys: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    def _plan(
        self, candidates: Sequence[torch.Tensor], context_cache: ContextKVCache
    ) -> tuple[tuple[Any, ...], int, list[int], list[int], Any] | None:
        """The capture key and batch layout, or None when eager must run."""
        if not self.enabled or not candidates:
            return None
        if context_cache.owner is not self.backbone:
            return None
        if context_cache.policy != "last" or len(context_cache.banks) != 1:
            return None
        bank = context_cache.banks[0]
        if not bank or len(bank) != len(self.native.layers):
            return None
        if bank[0].key.device.type != "cuda":
            return None
        length = context_cache.length
        if length <= 0 or length > GRAPH_MAX_CONTEXT:
            return None
        lengths: list[int] = []
        device = None
        for candidate in candidates:
            if candidate.ndim != 2 or candidate.shape[0] != 1:
                return None
            lengths.append(int(candidate.shape[1]))
            if device is None:
                device = candidate.device
            elif candidate.device != device:
                return None
        if len(lengths) > GRAPH_MAX_CANDIDATES:
            return None
        if any(value <= 0 for value in lengths):
            return None
        # Multi-batch rows keep their exact batching and use the eager path.
        batches = shared_context_batches(
            lengths, length, int(self.native.config.num_attention_heads)
        )
        if len(batches) != 1:
            return None
        key = (
            length,
            contract.CANDIDATE_STEPS,
            bank[0].key.dtype,
            bank[0].key.device.index,
            tuple(lengths),
        )
        return key, length, lengths, batches[0], device

    def scores(
        self, candidates: Sequence[torch.Tensor], context_cache: ContextKVCache
    ) -> torch.Tensor | None:
        """Replay a captured geometry, or return None and leave eager in charge."""
        if not self._graphs:
            return None
        plan = self._plan(candidates, context_cache)
        if plan is None:
            return None
        key, length, _lengths, _batch, _device = plan
        entry = self._graphs.get(key)
        if entry is None:
            return None
        # The eager path validates before any pass; replay keeps that contract.
        for candidate in candidates:
            _prepare_candidate_pass(
                self.native,
                self.backbone,
                context_cache,
                candidate,
                contract.CANDIDATE_STEPS,
            )
        try:
            self._fill(entry, candidates)
            for index, layer in enumerate(entry.bank):
                layer.key.copy_(context_cache.banks[0][index].key)
                layer.value.copy_(context_cache.banks[0][index].value)
            entry.graph.replay()
        except Exception:
            self.errors += 1
            self.enabled = False
            return None
        self.replays += 1
        self._graphs.move_to_end(key)
        return entry.scores.clone()

    def capture(
        self,
        candidates: Sequence[torch.Tensor],
        context_cache: ContextKVCache,
        eager_logits: torch.Tensor,
    ) -> None:
        """Capture a geometry on its second sighting and verify it bitwise."""
        if not self.enabled:
            return
        plan = self._plan(candidates, context_cache)
        if plan is None:
            return
        key, length, lengths, batch, device = plan
        if key in self._graphs:
            return
        sightings = self._seen.get(key)
        if sightings is None:
            self._seen[key] = 1
            limit = GRAPH_CACHE_ENTRIES * 4
            while len(self._seen) > limit:
                self._seen.popitem(last=False)
            return
        if sightings < GRAPH_CAPTURE_SIGHTINGS:
            self._seen[key] = sightings + 1
            return
        self._seen.pop(key, None)
        try:
            entry = self._build(
                candidates, context_cache, length, lengths, batch, device
            )
        except Exception:
            self.errors += 1
            self.enabled = False
            return
        reference = eager_logits.detach()
        if (
            entry.scores.shape != reference.shape
            or entry.scores.dtype != torch.float32
            or reference.dtype != torch.float32
            or not torch.equal(
                entry.scores.view(torch.int32), reference.view(torch.int32)
            )
        ):
            self.mismatches += 1
            self.enabled = False
            return
        self._graphs[key] = entry
        self.captures += 1
        while len(self._graphs) > GRAPH_CACHE_ENTRIES:
            self._graphs.popitem(last=False)
            self.evictions += 1

    def _build(
        self,
        candidates: Sequence[torch.Tensor],
        context_cache: ContextKVCache,
        length: int,
        lengths: list[int],
        batch: list[int],
        device: torch.device,
    ) -> _SharedBatchGraph:
        native = self.native
        heads = int(native.config.num_attention_heads)
        dim = int(
            getattr(native.config, "head_dim", 0) or native.config.hidden_size // heads
        )
        layers = len(native.layers)
        if self._keys is None or self._values is None:
            dtype = context_cache.banks[0][0].key.dtype
            self._keys = torch.empty(
                (layers, heads, GRAPH_MAX_CONTEXT, dim), dtype=dtype, device=device
            )
            self._values = torch.empty_like(self._keys)
        static_bank = tuple(
            LayerKVCache(
                self._keys[index, :, :length, :].unsqueeze(0),
                self._values[index, :, :length, :].unsqueeze(0),
            )
            for index in range(layers)
        )
        for index, layer in enumerate(static_bank):
            layer.key.copy_(context_cache.banks[0][index].key)
            layer.value.copy_(context_cache.banks[0][index].value)
        width = max(lengths[index] for index in batch)
        padded = _shared_context_padded(candidates, batch, lengths, device)
        mask = (
            torch.arange(width, device=device)[None, :]
            < torch.tensor(
                [lengths[index] for index in batch], dtype=torch.long, device=device
            )[:, None]
        )
        blocked = _shared_context_block(len(batch), width, device)
        static_context = ContextKVCache((static_bank,), length, "last", self.backbone)
        router = getattr(self.backbone, "_step_lora_router", None)
        geometry = _GeometryCache(native)
        # Warm the per-geometry positions/rotary/mask before capture: they are
        # deterministic functions of the geometry, not of the replay.
        geometry.get(native.embed_tokens(padded), length)
        graph = torch.cuda.CUDAGraph()
        # One pool per captured graph: the cache replays graphs in LRU order,
        # and a shared pool is only safe when replay order matches capture
        # order, so sharing would let one replay overwrite another's buffers.
        with torch.cuda.graph(graph):
            hidden = native.embed_tokens(padded)
            with _shared_context_layers(native, blocked):
                for step in range(contract.CANDIDATE_STEPS):
                    bank = static_context.banks[0]
                    with router.activate(step) if router is not None else nullcontext():
                        hidden = _run_recurrent_step(
                            native,
                            hidden,
                            length,
                            step,
                            _OwnCandidateKV(bank, step),
                            geometry,
                        )
            # Exactly the eager path's head input: one endpoint per caller
            # candidate, in caller order (the batch itself is length-sorted).
            endpoints: list[Any] = [None] * len(lengths)
            for slot, index in enumerate(batch):
                endpoints[index] = hidden[slot : slot + 1, lengths[index] - 1]
            scores = self.head(
                torch.cat([entry for entry in endpoints if entry is not None], dim=0)
            )[:, 0]
        graph.replay()
        return _SharedBatchGraph(
            graph,
            padded,
            mask,
            blocked,
            geometry,
            scores,
            lengths,
            batch,
            static_bank,
        )

    @staticmethod
    def _fill(entry: _SharedBatchGraph, candidates: Sequence[torch.Tensor]) -> None:
        entry.padded.zero_()
        entry.padded[entry.mask] = torch.cat(
            [candidates[index][0] for index in entry.batch]
        )


def candidate_kv_workspace_bytes(
    backbone: nn.Module,
    context_length: int,
    candidate_length: int,
    *,
    batch: int = 1,
) -> int:
    """Size in bytes of one preallocated candidate K/V workspace pair.

    Counting key and value for every decoder layer, the exact scoring path holds
    one pair per recurrent step and candidate length (reused across all
    candidates of that length). The default shared-context path copies no
    context K/V: its transient attention scores stay within
    ``SHARED_SCORE_BUDGET_BYTES`` per batch.
    """
    if type(context_length) is not int or context_length < 0:
        raise ValueError("context_length must be a nonnegative integer")
    if type(candidate_length) is not int or candidate_length <= 0:
        raise ValueError("candidate_length must be a positive integer")
    if type(batch) is not int or batch <= 0:
        raise ValueError("batch must be a positive integer")
    native = _native_backbone(backbone)
    total = context_length + candidate_length
    if total > native.config.max_position_embeddings:
        raise ValueError("complete candidate sequence exceeds max_position_embeddings")
    head_dim = getattr(native.config, "head_dim", None) or (
        native.config.hidden_size // native.config.num_attention_heads
    )
    elements = (
        2
        * batch
        * len(native.layers)
        * native.config.num_key_value_heads
        * total
        * head_dim
    )
    element_size = native.layers[0].input_layernorm.weight.element_size()
    return int(elements * element_size)


def score_candidates(
    backbone: nn.Module,
    head: nn.Module,
    context_ids: torch.Tensor,
    candidates: Sequence[torch.Tensor],
    *,
    policy: CachePolicy = "last",
    context_steps: int = contract.CONTEXT_STEPS,
    candidate_steps: int = contract.CANDIDATE_STEPS,
    candidate_batching: CandidateBatching | bool = "per_candidate",
) -> torch.Tensor:
    """Prefill shared context once and score all candidate continuations."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    context_cache = prefill_context(
        backbone, context_ids, policy=policy, context_steps=context_steps
    )
    return score_candidates_from_cache(
        backbone,
        head,
        context_cache,
        candidates,
        candidate_steps=candidate_steps,
        candidate_batching=candidate_batching,
    )


def score_candidates_from_cache(
    backbone: nn.Module,
    head: nn.Module,
    context_cache: ContextKVCache,
    candidates: Sequence[torch.Tensor],
    *,
    candidate_steps: int = contract.CANDIDATE_STEPS,
    candidate_batching: CandidateBatching | bool = "per_candidate",
) -> torch.Tensor:
    """Score candidate continuations from an already-computed ``ContextKVCache``.

    ``"per_candidate"``, the default of this building block, evaluates
    candidates one at a time and is bitwise identical to the per-candidate
    ``run_candidate_steps`` path. Every candidate is validated before any is
    evaluated: the token-range checks read device values, so on asynchronous
    devices (MPS, CUDA) checking between passes would make the host wait for
    each pass to drain before issuing the next. ``"shared_context"`` (the
    default of ``OuroScorer`` and ``load_ouro``) runs
    ``score_candidates_shared_context``: candidates batched, the context read
    once per layer, rounding different from the per-candidate path.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if type(candidate_steps) is not int or candidate_steps not in {1, 2, 3, 4}:
        raise ValueError("candidate_steps must be 1, 2, 3, or 4")
    if _candidate_batching(candidate_batching) == "shared_context":
        return score_candidates_shared_context(
            backbone,
            head,
            context_cache,
            candidates,
            candidate_steps=candidate_steps,
        )
    native = _native_backbone(backbone)
    routers = [
        _prepare_candidate_pass(
            native, backbone, context_cache, candidate, candidate_steps
        )
        for candidate in candidates
    ]
    geometry = _GeometryCache(native)
    pool = _WorkspacePool()
    hidden = torch.cat(
        [
            _candidate_hidden_steps(
                native,
                router,
                context_cache.length,
                candidate,
                context_cache,
                candidate_steps,
                geometry,
                pool,
            )[:, -1]
            for router, candidate in zip(routers, candidates, strict=True)
        ],
        dim=0,
    )
    scores = head(hidden)
    if scores.shape != (len(candidates), 1):
        raise ValueError("scalar score head must return shape [K, 1]")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite Ouro candidate logits")
    return cast(torch.Tensor, scores[:, 0])


def _clone_banks(
    banks: tuple[tuple[LayerKVCache, ...], ...],
) -> tuple[tuple[LayerKVCache, ...], ...]:
    return tuple(
        tuple(
            LayerKVCache(entry.key.detach().clone(), entry.value.detach().clone())
            for entry in bank
        )
        for bank in banks
    )


class OuroScorer:
    """Score public ``Request``s or records with an Ouro decision artifact.

    Supports optional exact-token state-block context caching across questions
    sharing the same rendered ``State:\\n<state>\\n`` block. Cache is disabled by
    default (``prefix_cache_size=0``), which executes the single-pass context
    prefill and is bitwise identical to training/evaluation without caching.

    A v2 artifact's ``context_prefill`` replaces that prefill for contexts
    longer than its ``single_chunk_max_tokens`` with the chunked execution it
    was trained with, and its ``numerics`` restrict SDPA to the exact MATH
    kernel while scoring; prefix caching is not part of that execution.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: ScalarScoreHead,
        tokenizer: Any,
        *,
        model_id: str,
        max_length: int,
        input_rendering: object = None,
        prefix_cache_size: int = 0,
        state_block_tokens: int = contract.CACHE_MAX_STATE_BLOCK_TOKENS,
        candidate_batching: CandidateBatching | bool = CANDIDATE_BATCHING,
        device_policy: str | None = None,
        context_prefill: Mapping[str, Any] | None = None,
        numerics: str | None = None,
        max_resource_fraction: float | None = None,
    ) -> None:
        if type(prefix_cache_size) is not int or prefix_cache_size < 0:
            raise ValueError("prefix_cache_size must be a nonnegative integer")
        if type(state_block_tokens) is not int or state_block_tokens < 0:
            raise ValueError("state_block_tokens must be a nonnegative integer")
        batching = _candidate_batching(candidate_batching)
        if device_policy is not None and device_policy not in contract.DEVICE_POLICIES:
            raise ValueError(f"unknown artifact device_policy: {device_policy!r}")
        if context_prefill is not None and prefix_cache_size > 0:
            raise ValueError("prefix reuse is not part of a v2 context execution")
        if numerics is not None and numerics not in contract.NUMERICS_POLICIES:
            raise ValueError(f"unknown execution numerics: {numerics!r}")
        self.resource_guard = current_guard() or ResourceGuard(max_resource_fraction)
        self.context_prefill = (
            None if context_prefill is None else dict(context_prefill)
        )
        self.numerics = numerics
        self.backbone: Any = backbone
        self.head = head
        self.resource_guard.configure_torch(torch, self.device)
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_length = max_length
        self.input_rendering = _normalize_input_rendering(input_rendering)
        self.prefix_cache_size = prefix_cache_size
        self.device_policy = device_policy
        # Shared-context batches by default; "per_candidate" is bitwise exact.
        self.candidate_batching: CandidateBatching = batching
        self.state_block_tokens = (
            contract.CACHE_MAX_STATE_BLOCK_TOKENS
            if prefix_cache_size > 0 and state_block_tokens == 0
            else state_block_tokens
        )
        # Retained state-block cache stores all 4 recurrent steps' K/V banks so
        # a question-specific tail can continue through all 4 steps before
        # producing the final step-4 bank for candidate scoring.
        self._prefix_cache: OrderedDict[
            tuple[int, ...], tuple[tuple[LayerKVCache, ...], ...]
        ] = OrderedDict()
        self._prefix_hits = 0
        self._prefix_misses = 0
        # Exact CUDA-graph replay of short candidate batches (CUDA inference
        # only). It is disabled automatically off CUDA or when capture fails.
        self.candidate_graphs: _ExactCandidateGraphs | None = (
            _ExactCandidateGraphs(backbone, head)
            if self.device.type == "cuda" and batching == "shared_context"
            else None
        )

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
        """Clear every retained state-block context entry and reset counters."""
        self._prefix_cache.clear()
        self._prefix_hits = 0
        self._prefix_misses = 0

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        max_length: int | None = None,
        prefix_cache_size: int | None = None,
        attention: str | None = None,
        context_attention: str | None = None,
        verify_release_base: bool = False,
        candidate_batching: CandidateBatching | bool = CANDIDATE_BATCHING,
        allow_device_policy_override: bool = False,
        max_resource_fraction: float | None = None,
    ) -> OuroScorer:
        """Load under the same macOS resource policy as inference."""
        requested_guard = ResourceGuard(max_resource_fraction)
        guard = current_guard() or requested_guard
        with guard.activate():
            guard.check(operation="load Ouro")
            scorer = cls._from_directory(
                path,
                device=device,
                max_length=max_length,
                prefix_cache_size=prefix_cache_size,
                attention=attention,
                context_attention=context_attention,
                verify_release_base=verify_release_base,
                candidate_batching=candidate_batching,
                allow_device_policy_override=allow_device_policy_override,
                max_resource_fraction=max_resource_fraction,
            )
            guard.check(operation="loaded Ouro model")
            return scorer

    @classmethod
    def _from_directory(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        max_length: int | None = None,
        prefix_cache_size: int | None = None,
        attention: str | None = None,
        context_attention: str | None = None,
        verify_release_base: bool = False,
        candidate_batching: CandidateBatching | bool = CANDIDATE_BATCHING,
        allow_device_policy_override: bool = False,
        max_resource_fraction: float | None = None,
    ) -> OuroScorer:
        root = Path(path).resolve()
        manifest = contract.validate_artifact(
            root, verify_release_base=verify_release_base
        )
        execution = manifest["execution"]
        if execution.get("numerics") is not None:
            # Before any CUDA matmul: deterministic cuBLAS reads its workspace
            # setting once.
            apply_numerics(execution["numerics"])
        declared_device_policy = execution["device_policy"]
        permitted_devices = _DEVICE_POLICY_DEVICES[declared_device_policy]
        if device is None:
            if declared_device_policy == "cuda-bf16-autocast":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif declared_device_policy == "mps-fp32":
                if not torch.backends.mps.is_available():
                    raise ValueError(
                        "artifact device_policy is 'mps-fp32' but Apple Silicon "
                        "MPS is unavailable on this host"
                    )
                device = "mps"
            else:
                device = "cpu"
        target = torch.device(device)
        if target.type not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported Ouro execution device: {device!r}")
        if target.type not in permitted_devices and not allow_device_policy_override:
            raise ValueError(
                f"artifact device_policy {declared_device_policy!r} does not permit "
                f"{target.type!r} execution; pass "
                "allow_device_policy_override=True to accept the artifact's "
                "declared execution tolerance on this device"
            )

        declared_attention = execution["attention"]
        if attention is not None:
            normalized_attn = (
                "sdpa-math" if attention in {"sdpa", "sdpa-math"} else attention
            )
            if normalized_attn != declared_attention:
                raise ValueError(
                    f"requested attention {attention!r} conflicts with manifest "
                    f"attention {declared_attention!r}"
                )
        hf_attn = "sdpa" if declared_attention == "sdpa-math" else "eager"

        cache_cfg = manifest["cache"]
        if prefix_cache_size is None:
            resolved_cache_size = 0
        else:
            if type(prefix_cache_size) is not int or prefix_cache_size < 0:
                raise ValueError("prefix_cache_size must be a nonnegative integer")
            if prefix_cache_size > contract.CACHE_MAX_ENTRIES:
                raise ValueError(
                    f"prefix_cache_size exceeds maximum {contract.CACHE_MAX_ENTRIES}"
                )
            resolved_cache_size = prefix_cache_size

        limit = execution["max_length"] if max_length is None else max_length
        if type(limit) is not int or not 0 < limit <= execution["max_length"]:
            raise ValueError("max_length must be within the artifact's execution limit")

        # Validate the small adapter/head files before materializing the model;
        # the base checkpoint is validated per tensor while streaming (below).
        _verify_safetensors_file(
            root / contract.ADAPTERS_FILE,
            expected_keys=set(contract.adapter_tensor_names()),
            expected_dtype=torch.float32,
            label="adapter",
        )
        _verify_safetensors_file(
            root / contract.HEAD_FILE,
            expected_keys=set(contract.HEAD_TENSORS),
            expected_dtype=torch.float32,
            label="head",
        )

        config_cls, model_cls = load_ouro_classes(
            root, manifest["base"]["source_sha256"]
        )
        config_data = json.loads((root / "config.json").read_text(encoding="utf-8"))
        with _quiet_ouro_source_exec():
            config = config_cls(**config_data)
        config._attn_implementation = hf_attn
        if config.num_hidden_layers != contract.NATIVE_LAYERS:
            raise ValueError("Ouro config num_hidden_layers differs from contract")
        if limit > config.max_position_embeddings:
            raise ValueError("max_length exceeds Ouro max_position_embeddings")

        base_dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
        # Materialize an empty model directly on the target device and stream the
        # base checkpoint into its parameters. Building on "meta" skips 5.7 GiB
        # of random weight initialization, and streaming avoids holding the
        # 2.9 GiB BF16 checkpoint as a host dictionary beside the parameters
        # (peak RSS 8.43 GiB -> about 6 GiB). BF16 -> FP32 is exact per value,
        # so this is bitwise identical to a load_state_dict of the same file.
        with _quiet_ouro_source_exec(), torch.device("meta"):
            backbone = model_cls(config)
        resource_guard = current_guard() or ResourceGuard(max_resource_fraction)
        resource_guard.configure_torch(torch, target.type)
        if resource_guard.enabled:
            # Parameters are still on meta: refuse before physical allocation.
            parameter_bytes = sum(
                p.numel() * p.element_size() for p in backbone.parameters()
            )
            largest_tensor = max(
                p.numel() * p.element_size() for p in backbone.parameters()
            )
            resource_guard.check(
                parameter_bytes + 2 * largest_tensor + (256 << 20),
                operation="load Ouro",
            )
        backbone.to_empty(device=target)
        _stream_base_weights(backbone, root / contract.BASE_WEIGHT_FILE)
        if base_dtype != torch.float32:
            backbone.to(dtype=base_dtype)
        # Restore after the weight cast: casting this buffer to BF16 rounds the
        # position frequencies and changes scores, even if forward uses .float().
        # It is non-persistent, so to_empty also leaves it uninitialized.
        inv_freq, _ = _default_rope_init(config)
        backbone.model.rotary_emb.inv_freq = inv_freq.to(
            device=target, dtype=torch.float32
        )
        backbone.model.rotary_emb.original_inv_freq = backbone.model.rotary_emb.inv_freq
        attach_step_lora(
            backbone,
            rank=contract.ADAPTER_RANK,
            alpha=contract.ADAPTER_ALPHA,
            steps=contract.CANDIDATE_STEPS,
            targets=contract.ADAPTER_TARGETS,
        )
        adapter_tensors = load_file(str(root / contract.ADAPTERS_FILE))
        params = dict(backbone.named_parameters())
        with torch.no_grad():
            for key, tensor in adapter_tensors.items():
                if not key.startswith("backbone."):
                    raise ValueError(f"invalid adapter key: {key}")
                param_name = key.removeprefix("backbone.")
                target_param = params.get(param_name)
                if target_param is None or target_param.shape != tensor.shape:
                    raise ValueError(f"adapter shape mismatch for {key}")
                target_param.copy_(tensor.to(device=target, dtype=torch.float32))

        head = ScalarScoreHead(config.hidden_size, contract.HEAD_WIDTH)
        head_tensors = load_file(str(root / contract.HEAD_FILE))
        head_state = {
            k.removeprefix("head."): v.to(dtype=torch.float32)
            for k, v in head_tensors.items()
        }
        if {k: v.shape for k, v in head_state.items()} != {
            k: v.shape for k, v in head.state_dict().items()
        }:
            raise ValueError("invalid Ouro scalar head tensor shapes")
        head.load_state_dict(head_state)
        head.to(device=target, dtype=torch.float32)
        backbone.requires_grad_(False)
        head.requires_grad_(False)
        backbone.eval()
        head.eval()

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        context_prefill = execution.get("context_prefill")
        if context_attention is not None:
            if context_attention not in {"lean", "flash"}:
                raise ValueError(
                    "context_attention must be 'lean' or 'flash' for a v2 artifact"
                )
            if context_prefill is None:
                raise ValueError(
                    "context_attention requires a v2 artifact with context_prefill"
                )
            # ``flash`` is a throughput mode: it is not bit-identical to the
            # MATH path. ``lean`` is the exact path; this opt-in lets a caller
            # choose parity without mutating the artifact or its model_id.
            context_prefill = {
                **context_prefill,
                "chunked_attention": context_attention,
            }
        state_tokens = (
            cache_cfg["state_block_tokens"]
            if cache_cfg["state_block_tokens"] > 0
            else contract.CACHE_MAX_STATE_BLOCK_TOKENS
        )
        scorer = cls(
            backbone,
            head,
            tokenizer,
            model_id=manifest["model_id"],
            max_length=limit,
            input_rendering=manifest["prompt"]["input_rendering"],
            prefix_cache_size=resolved_cache_size,
            state_block_tokens=state_tokens,
            candidate_batching=candidate_batching,
            device_policy=declared_device_policy,
            context_prefill=context_prefill,
            numerics=execution.get("numerics"),
            max_resource_fraction=max_resource_fraction,
        )
        scorer.resource_guard = resource_guard
        return scorer

    def _context_execution(
        self, tokens: int
    ) -> tuple[int | None, ContextAttention, str | None]:
        """Chunk, prefill attention and context-cache device for ``tokens``.

        A v1 artifact, and a v2 context of at most ``single_chunk_max_tokens``,
        take the exact single pass. A longer v2 context is chunked; above
        ``host_cache_above_tokens`` its cache lives in host memory, which only
        the shared-context batches read, and never changes a result.
        """
        prefill = self.context_prefill
        if prefill is None or tokens <= prefill["single_chunk_max_tokens"]:
            return None, "native", None
        host = prefill["host_cache_above_tokens"]
        offload = (
            host is not None
            and tokens > host
            and self.device.type == "cuda"
            and self.candidate_batching == "shared_context"
        )
        return (
            prefill["chunk"],
            prefill["chunked_attention"],
            "cpu" if offload else None,
        )

    def _context_bank_bytes(self, tokens: int) -> int:
        """Bytes of the retained step-4 K/V tail for a ``tokens`` context."""
        native = _native_backbone(self.backbone)
        config = native.config
        query_heads = int(config.num_attention_heads)
        kv_heads = int(
            getattr(config, "num_key_value_heads", query_heads) or query_heads
        )
        head_dim = int(
            getattr(config, "head_dim", 0) or config.hidden_size // query_heads
        )
        element = native.layers[0].input_layernorm.weight.element_size()
        return int(2 * len(native.layers) * kv_heads * tokens * head_dim * element)

    def _context_bank_fits_device(self, tokens: int) -> bool:
        """Whether this bank can stay on the GPU with safe headroom left.

        The artifact's host-cache placement is a memory policy, not arithmetic;
        when the bank fits, keeping it resident removes the per-layer transfers
        and never changes a value. A short-on-memory device keeps the artifact's
        host placement.
        """
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            return False
        return self._context_bank_bytes(tokens) + DEVICE_BANK_RESERVE_BYTES <= free

    def _attention_kernels(self) -> AbstractContextManager[Any]:
        """The exact MATH SDPA kernel under a v2 artifact's numerics."""
        if self.numerics is None:
            return nullcontext()
        return sdpa_kernel(SDPBackend.MATH)

    def _amp(self) -> AbstractContextManager[Any]:
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def encode_record(self, record: object) -> tuple[CandidateBatch, ...]:
        return encode_record(
            self.tokenizer,
            render_record(record, self.input_rendering),
            max_length=self.max_length,
        )

    def _match_prefix(self, context_ids: tuple[int, ...]) -> tuple[int, ...] | None:
        best: tuple[int, ...] | None = None
        for key in self._prefix_cache:
            if len(key) <= (0 if best is None else len(best)):
                continue
            if context_ids[: len(key)] == key:
                best = key
        if best is not None:
            self._prefix_cache.move_to_end(best)
        return best

    def _store_prefix(
        self,
        key: tuple[int, ...],
        banks: tuple[tuple[LayerKVCache, ...], ...],
    ) -> None:
        if not key or self.prefix_cache_size <= 0:
            return
        self._prefix_cache[key] = _clone_banks(banks)
        self._prefix_cache.move_to_end(key)
        while len(self._prefix_cache) > self.prefix_cache_size:
            self._prefix_cache.popitem(last=False)

    def _continue_context_from_banks(
        self,
        prior_banks: tuple[tuple[LayerKVCache, ...], ...],
        tail_ids: tuple[int, ...],
        start: int,
    ) -> ContextKVCache:
        """Continue 4 recurrent context passes from retained state-block banks."""
        if not tail_ids:
            return ContextKVCache((prior_banks[-1],), start, "last", self.backbone)
        native = _native_backbone(self.backbone)
        ids = torch.tensor([tail_ids], dtype=torch.long, device=self.device)
        _validate_token_tensor(native, ids)
        router = getattr(self.backbone, "_step_lora_router", None)
        geometry = _GeometryCache(native)
        final_bank: tuple[LayerKVCache, ...] = ()
        with router.disabled() if router is not None else nullcontext():
            hidden = native.embed_tokens(ids)
            for step in range(contract.CONTEXT_STEPS):
                reader = _ReadAndCaptureStepKV(
                    prior_banks[step], step, retain=(step == contract.CONTEXT_STEPS - 1)
                )
                hidden = _run_recurrent_step(
                    native, hidden, start, step, reader, geometry
                )
                if reader.entries:
                    final_bank = tuple(reader.entries)
        return ContextKVCache(
            (final_bank,), start + len(tail_ids), "last", self.backbone
        )

    def _context_cache_for(self, row: CandidateBatch) -> ContextKVCache:
        """Compute or reuse the step-4 ``ContextKVCache`` for ``row``."""
        if self.context_prefill is not None:
            chunk, attention, bank = self._context_execution(len(row.context_ids))
            if bank is not None:
                # Return cached blocks to the driver first: ``mem_get_info``
                # excludes allocator-cached memory, so without this a bank
                # freed by the previous row can make an identical-size bank
                # look not to fit. The placement decision is the only memory
                # policy here; the values are the same either way.
                torch.cuda.empty_cache()
                if self._context_bank_fits_device(len(row.context_ids)):
                    bank = None
            ids = torch.tensor([row.context_ids], dtype=torch.long, device=self.device)
            return prefill_context(
                self.backbone,
                ids,
                policy="last",
                context_steps=contract.CONTEXT_STEPS,
                chunk=chunk,
                bank_device=bank,
                attention=attention,
            )
        if self.prefix_cache_size <= 0:
            ids = torch.tensor([row.context_ids], dtype=torch.long, device=self.device)
            return prefill_context(
                self.backbone,
                ids,
                policy="last",
                context_steps=contract.CONTEXT_STEPS,
            )
        key = self._match_prefix(row.context_ids)
        if key is not None:
            self._prefix_hits += 1
            return self._continue_context_from_banks(
                self._prefix_cache[key],
                row.context_ids[len(key) :],
                len(key),
            )
        self._prefix_misses += 1
        boundary = reuse_context_prefix(self.tokenizer, row)
        if len(boundary) > self.state_block_tokens:
            boundary = boundary[: self.state_block_tokens]
        if not boundary or len(boundary) >= len(row.context_ids):
            ids = torch.tensor([row.context_ids], dtype=torch.long, device=self.device)
            return prefill_context(
                self.backbone,
                ids,
                policy="last",
                context_steps=contract.CONTEXT_STEPS,
            )
        boundary_ids = torch.tensor([boundary], dtype=torch.long, device=self.device)
        matched_state = prefill_context(
            self.backbone,
            boundary_ids,
            policy="matched",
            context_steps=contract.CONTEXT_STEPS,
        )
        self._store_prefix(boundary, matched_state.banks)
        return self._continue_context_from_banks(
            matched_state.banks,
            row.context_ids[len(boundary) :],
            len(boundary),
        )

    def _request_memory_bytes(self, row: CandidateBatch) -> int:
        """Conservative extra live bytes; retained model/cache memory is already used.

        Account for the final context bank, recurrent K/V workspaces, attention
        intermediates and chunk masks. This is admission evidence, not an exact
        allocator peak prediction (framework caches and other apps can grow).
        """
        native = _native_backbone(self.backbone)
        config = native.config
        length = len(row.context_ids)
        suffix = max(map(len, row.candidate_ids))
        width = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        size = native.layers[0].input_layernorm.weight.element_size()
        kv = (
            2
            * len(native.layers)
            * config.num_key_value_heads
            * width
            * (length + suffix)
            * size
        )
        chunk, attention, _ = self._context_execution(length)
        query = length if chunk is None else min(chunk, length)
        # MATH can retain score and softmax buffers. Older MPS flash fallbacks
        # do too; estimate conservatively without guessing the installed kernel.
        attention_bytes = 3 * config.num_attention_heads * query * length * size
        masks = 0 if attention == "flash" else length * (length + query) * size
        hidden = 4 * length * config.hidden_size * size
        lengths = [len(ids) for ids in row.candidate_ids]
        if self.candidate_batching == "per_candidate":
            # The workspace pool retains both recurrent steps for EACH distinct
            # suffix length, not just the longest candidate.
            candidate = sum(
                2 * candidate_kv_workspace_bytes(self.backbone, length, n)
                for n in set(lengths)
            )
            candidate += (
                3 * config.num_attention_heads * suffix * (length + suffix) * size
            )
        else:
            batches = shared_context_batches(
                lengths, length, config.num_attention_heads
            )
            packed = [len(batch) * max(lengths[i] for i in batch) for batch in batches]
            # A singleton may exceed the batching budget. Count its real score
            # matrix rather than assuming the budget is an allocation ceiling.
            candidate = max(
                8 * config.num_attention_heads * n * (length + n) for n in packed
            )
            candidate += 4 * sum(packed) * config.hidden_size * size
        prefix = 0
        if self.prefix_cache_size > 0:
            boundary = min(length, self.state_block_tokens)
            # A prefix miss can hold four matched banks and their four clones
            # while continuing the context. Existing retained entries are
            # already included in system-wide usage.
            prefix = (
                8
                * 2
                * len(native.layers)
                * config.num_key_value_heads
                * width
                * boundary
                * size
            )
        return int(
            kv
            + 2 * kv // len(native.layers)
            + attention_bytes
            + masks
            + hidden
            + candidate
            + prefix
        )

    def question_logits(self, row: CandidateBatch) -> list[float]:
        with self.resource_guard.activate():
            if self.resource_guard.enabled:
                self.resource_guard.check(
                    self._request_memory_bytes(row), operation="score Ouro request"
                )
            return self._question_logits(row)

    def _question_logits(self, row: CandidateBatch) -> list[float]:
        # One device copy for the whole row, split into [1, S] views. The
        # token values are the same as one tensor per candidate.
        flat = torch.tensor(
            [token for ids in row.candidate_ids for token in ids],
            dtype=torch.long,
            device=self.device,
        )
        candidates = []
        offset = 0
        for ids in row.candidate_ids:
            end = offset + len(ids)
            candidates.append(flat[offset:end].view(1, -1))
            offset = end
        with torch.inference_mode(), self._amp(), self._attention_kernels():
            context_cache = self._context_cache_for(row)
            graphs = self.candidate_graphs
            logits = (
                graphs.scores(candidates, context_cache) if graphs is not None else None
            )
            if logits is None:
                logits = score_candidates_from_cache(
                    self.backbone,
                    self.head,
                    context_cache,
                    candidates,
                    candidate_steps=contract.CANDIDATE_STEPS,
                    candidate_batching=self.candidate_batching,
                )
                if graphs is not None:
                    graphs.capture(candidates, context_cache, logits)
        values = cast(list[float], logits.detach().float().cpu().tolist())
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite Ouro candidate logits")
        return values

    def _encode_request(self, request: Request) -> CandidateBatch:
        try:
            return self.encode_record(_request_record(request))[0]
        except ModelInputError:
            raise
        except ValueError as error:
            raise contract.input_error(error) from error

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        """Score public ``Request``s in request and candidate order."""
        rows = [self._encode_request(r) for r in requests]
        return [self.question_logits(row) for row in rows]

    def _encode_public(self, record: object) -> tuple[CandidateBatch, ...]:
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


def apply_numerics(policy: str) -> None:
    """Set the process-wide numerics a v2 artifact was trained and judged with.

    ``deterministic-no-tf32-v1``: TF32 off for matmuls and cuDNN, FP32 matmul
    precision ``highest``, deterministic algorithms with cuBLAS workspace
    ``:4096:8`` (kept if already set), and no fill of uninitialized memory.
    These settings stay in effect for the rest of the process.
    """
    if policy not in contract.NUMERICS_POLICIES:
        raise ValueError(f"unknown execution numerics: {policy!r}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False  # type: ignore[attr-defined]


def _normalize_input_rendering(policy: object) -> Any:
    """Accept either wyat's or keme's authenticated rendering source hash."""
    if policy is None:
        return None
    if type(policy) is not dict or set(policy) != {"mode", "source_sha256"}:
        raise ValueError("input rendering policy requires mode and source_sha256")
    mode = policy["mode"]
    if type(mode) is not str or mode not in contract.INPUT_RENDERING_MODES:
        raise ValueError(f"unsupported input rendering mode: {mode!r}")
    # Return wyat's local authenticated policy object for the validated mode.
    return validate_input_rendering_policy(input_rendering_policy(mode))


def _stream_base_weights(model: nn.Module, path: Path) -> None:
    """Verify and copy BF16 base checkpoint tensors into ``model`` one at a time.

    Combines the dtype/finiteness validation of ``_verify_safetensors_file`` with
    the copy so the file is read once and never materialized as a full host-side
    dictionary. Every BF16 value is converted to the destination dtype exactly.
    """
    state = dict(model.state_dict())
    expected = set(state)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = expected - keys
        if missing - {"lm_head.weight"}:
            raise ValueError("Ouro base weights do not match native architecture")
        if keys - expected:
            raise ValueError("Ouro base weights do not match native architecture")
        with torch.no_grad():
            for key in handle.keys():
                checkpoint()
                tensor = handle.get_tensor(key)
                destination = state[key]
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        f"Ouro base tensor {key} has dtype {tensor.dtype}, "
                        "expected torch.bfloat16"
                    )
                if destination.shape != tensor.shape:
                    raise ValueError(f"Ouro base tensor {key} has unexpected shape")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"nonfinite Ouro base tensor: {key}")
                destination.copy_(tensor)


def _verify_safetensors_file(
    path: Path,
    *,
    expected_keys: set[str] | None,
    expected_dtype: torch.dtype,
    label: str,
) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if expected_keys is not None and keys != expected_keys:
            raise ValueError(f"Ouro {label} tensor inventory differs")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Ouro {label} tensor {key} has dtype {tensor.dtype}, "
                    f"expected {expected_dtype}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite Ouro {label} tensor: {key}")


def load_ouro(
    path: str | Path,
    *,
    device: str | None = None,
    max_length: int | None = None,
    prefix_cache_size: int | None = None,
    attention: str | None = None,
    context_attention: str | None = None,
    verify_release_base: bool = False,
    candidate_batching: CandidateBatching | bool = CANDIDATE_BATCHING,
    allow_device_policy_override: bool = False,
    max_resource_fraction: float | None = None,
) -> OuroScorer:
    """Load an Ouro decision artifact directory.

    The artifact's ``device_policy`` selects the default device: ``cuda`` when
    available for ``cuda-bf16-autocast``, ``mps`` for ``mps-fp32`` on Apple
    Silicon, and ``cpu`` for ``cpu-fp32``. A CUDA artifact also accepts explicit
    ``cpu`` or ``mps`` execution with an FP32 backbone; its CUDA BF16 outputs
    need not be bitwise identical. An explicit ``device`` may only select a
    device type the policy permits; pass
    ``allow_device_policy_override=True`` to execute elsewhere under the
    artifact's declared execution tolerance. ``mps-fp32`` runs FP32 math on the
    Apple GPU and matches the CPU FP32 reference within
    ``contract.MPS_EXECUTION_TOLERANCE_ATOL``/``RTOL``. Candidates are scored in
    shared-context batches by default; ``candidate_batching="per_candidate"``
    scores them one at a time, bitwise identical to the reference path.
    ``context_attention="lean"`` overrides a v2 artifact's chunked context
    attention with the exact MATH-equivalent path; ``"flash"`` is the fast,
    non-MATH-exact throughput path.
    """
    return OuroScorer.from_directory(
        path,
        device=device,
        max_length=max_length,
        prefix_cache_size=prefix_cache_size,
        attention=attention,
        context_attention=context_attention,
        verify_release_base=verify_release_base,
        candidate_batching=candidate_batching,
        allow_device_policy_override=allow_device_policy_override,
        max_resource_fraction=max_resource_fraction,
    )


def write_ouro_artifact(
    output: str | Path,
    *,
    contract_prompt: str,
    source_dir: str | Path,
    tokenizer_dir: str | Path,
    base_weights: str | Path | Mapping[str, torch.Tensor],
    adapters: Mapping[str, torch.Tensor],
    head: Mapping[str, torch.Tensor],
    max_length: int,
    attention: str = "sdpa-math",
    device_policy: str = "cuda-bf16-autocast",
    cache_entries: int = contract.CACHE_DEFAULT_ENTRIES,
    cache_state_block_tokens: int = contract.CACHE_DEFAULT_STATE_BLOCK_TOKENS,
    input_rendering: Mapping[str, object] | None = None,
    encoder_sha256: str | None = None,
    context_prefill: Mapping[str, object] | None = None,
    numerics: str | None = None,
) -> dict[str, Any]:
    """Assemble and validate a complete Ouro artifact directory.

    With ``context_prefill`` and ``numerics`` (the context execution the model
    was trained and judged with) it is a ``wyat-ouro-decision-v2`` artifact,
    otherwise ``wyat-ouro-decision-v1``.
    """
    if contract_prompt != contract.PROMPT:
        raise ValueError(
            "write_ouro_artifact only writes the frozen current-prompt layout; "
            "quoted/preview contracts must use a v2 artifact writer"
        )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    src_root = Path(source_dir)
    tok_root = Path(tokenizer_dir)
    for name in contract.SOURCE_FILES:
        shutil.copy2(src_root / name, root / name)
    for name in contract.TOKENIZER_FILES:
        shutil.copy2(tok_root / name, root / name)
    if isinstance(base_weights, (str, Path)):
        shutil.copy2(Path(base_weights), root / contract.BASE_WEIGHT_FILE)
    else:
        save_file(
            {
                k: v.detach().to(dtype=torch.bfloat16).contiguous().cpu()
                for k, v in base_weights.items()
            },
            str(root / contract.BASE_WEIGHT_FILE),
        )
    adapter_tensors = {
        k: v.detach().to(dtype=torch.float32).contiguous().cpu()
        for k, v in adapters.items()
    }
    head_tensors = {
        (k if k.startswith("head.") else f"head.{k}"): v.detach()
        .to(dtype=torch.float32)
        .contiguous()
        .cpu()
        for k, v in head.items()
    }
    save_file(adapter_tensors, str(root / contract.ADAPTERS_FILE))
    save_file(head_tensors, str(root / contract.HEAD_FILE))

    files_sha256 = {
        name: file_sha256(root / name) for name in sorted(contract.REQUIRED_FILES)
    }
    resolved_encoder_sha256 = (
        encoder_sha256
        if encoder_sha256 is not None
        else file_sha256(Path(contract.__file__).resolve())
    )
    manifest = contract.build_manifest(
        files_sha256=files_sha256,
        encoder_sha256=resolved_encoder_sha256,
        adapter_dtypes={k: "float32" for k in adapter_tensors},
        head_dtypes={k: "float32" for k in head_tensors},
        max_length=max_length,
        attention=attention,
        device_policy=device_policy,
        cache_entries=cache_entries,
        cache_state_block_tokens=cache_state_block_tokens,
        input_rendering=input_rendering,
        context_prefill=context_prefill,
        numerics=numerics,
    )
    (root / contract.MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    contract.validate_artifact(root)
    return manifest


__all__ = [
    "CANDIDATE_BATCHING",
    "SHARED_SCORE_BUDGET_BYTES",
    "CandidateBatch",
    "CandidateBatching",
    "ContextKVCache",
    "ContextAttention",
    "LayerKVCache",
    "OuroScorer",
    "ScalarScoreHead",
    "StepLoRALinear",
    "StepLoRARouter",
    "apply_numerics",
    "attach_step_lora",
    "candidate_kv_workspace_bytes",
    "load_ouro",
    "load_ouro_classes",
    "prefill_context",
    "run_candidate_steps",
    "score_candidates",
    "score_candidates_from_cache",
    "score_candidates_shared_context",
    "shared_context_batches",
    "write_ouro_artifact",
]
