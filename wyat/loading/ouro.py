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
from transformers import PreTrainedTokenizerFast
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

CachePolicy = Literal["matched", "last"]

# Target device types each declared execution policy permits. A request outside
# this set requires the explicit ``allow_device_policy_override`` opt-in and
# then runs under the artifact's declared execution tolerance.
_DEVICE_POLICY_DEVICES: dict[str, frozenset[str]] = {
    "cuda-bf16-autocast": frozenset({"cuda", "cpu"}),
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
    if torch.any(ids < 0) or torch.any(ids >= backbone.config.vocab_size):
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
    cache: _CaptureStepKV | _ReadAndCaptureStepKV | _ReadContextKV,
    geometry: _GeometryCache,
) -> torch.Tensor:
    positions, rotary, mask = geometry.get(hidden, context_length)
    for layer in backbone.layers:
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


def prefill_context(
    backbone: nn.Module,
    context_ids: torch.Tensor,
    *,
    policy: CachePolicy = "last",
    context_steps: int = contract.CONTEXT_STEPS,
) -> ContextKVCache:
    """Run shared-context recurrent steps with all step-LoRA adapters disabled."""
    if policy not in {"matched", "last"}:
        raise ValueError("unknown cache policy")
    if type(context_steps) is not int or context_steps not in {1, 2, 3, 4}:
        raise ValueError("context_steps must be 1, 2, 3, or 4")
    if policy == "last" and context_steps != 4:
        raise ValueError("last policy requires four context steps")
    native = _native_backbone(backbone)
    _validate_token_tensor(native, context_ids)
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


def _batched_candidate_last_hidden(
    backbone: nn.Module,
    context_cache: ContextKVCache,
    candidates: Sequence[torch.Tensor],
    *,
    candidate_steps: int,
    geometry: _GeometryCache,
    pool: _WorkspacePool,
) -> torch.Tensor:
    """Run equal-length candidate groups as one batched forward per step.

    Candidates are grouped by token length, so no padding is involved: every
    real token keeps exactly the positions and causal mask it has in the
    sequential path, and each frozen backbone weight is read once per recurrent
    step for the whole group. Rows are written back in caller order.
    """
    native: Any = None
    router: Any = None
    groups: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        native = _native_backbone(backbone)
        router = _prepare_candidate_pass(
            native, backbone, context_cache, candidate, candidate_steps
        )
        groups.setdefault(int(candidate.shape[1]), []).append(index)
    if native is None:
        raise ValueError("at least one candidate is required")
    rows: torch.Tensor | None = None
    for indices in groups.values():
        batched_ids = torch.cat([candidates[index] for index in indices], dim=0)
        hidden = native.embed_tokens(batched_ids)
        for step in range(candidate_steps):
            bank = context_cache.banks[step if context_cache.policy == "matched" else 0]
            with router.activate(step) if router is not None else nullcontext():
                hidden = _run_recurrent_step(
                    native,
                    hidden,
                    context_cache.length,
                    step,
                    _ReadContextKV(bank, step, pool),
                    geometry,
                )
        last = hidden[:, -1]
        if rows is None:
            rows = torch.empty(
                (len(candidates), last.shape[1]),
                dtype=last.dtype,
                device=last.device,
            )
        rows[indices] = last
    if rows is None:
        raise ValueError("at least one candidate is required")
    return rows


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
    candidates of that length). The opt-in ``candidate_batching`` fast path
    holds one pair per (step, candidate length, batch-size) group.
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
    candidate_batching: bool = False,
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
    candidate_batching: bool = False,
) -> torch.Tensor:
    """Score candidate continuations from an already-computed ``ContextKVCache``.

    The default path evaluates candidates sequentially and is bitwise identical
    to the per-candidate ``run_candidate_steps`` path. Every candidate is
    validated before any is evaluated: the token-range checks read device
    values, so on asynchronous devices (MPS, CUDA) checking between passes
    would make the host wait for each pass to drain before issuing the next.
    ``candidate_batching`` is
    an opt-in, tolerance-gated fast path: candidates are grouped by equal token
    length and each group runs as one batched forward per recurrent step, so
    each frozen backbone weight is read once per step instead of once per
    candidate. Batched FP32 GEMMs can legitimately shift results by a few ULPs,
    so keep the default when logits must match bitwise (measured drift on the
    1.4B artifact is reported by the fast-path gate script).
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if type(candidate_steps) is not int or candidate_steps not in {1, 2, 3, 4}:
        raise ValueError("candidate_steps must be 1, 2, 3, or 4")
    if type(candidate_batching) is not bool:
        raise ValueError("candidate_batching must be a bool")
    if candidate_batching:
        native: Any = _native_backbone(backbone)
        geometry = _GeometryCache(native)
        pool = _WorkspacePool()
        hidden = _batched_candidate_last_hidden(
            backbone,
            context_cache,
            candidates,
            candidate_steps=candidate_steps,
            geometry=geometry,
            pool=pool,
        )
    else:
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
        candidate_batching: bool = False,
        device_policy: str | None = None,
    ) -> None:
        if type(prefix_cache_size) is not int or prefix_cache_size < 0:
            raise ValueError("prefix_cache_size must be a nonnegative integer")
        if type(state_block_tokens) is not int or state_block_tokens < 0:
            raise ValueError("state_block_tokens must be a nonnegative integer")
        if type(candidate_batching) is not bool:
            raise ValueError("candidate_batching must be a bool")
        if device_policy is not None and device_policy not in contract.DEVICE_POLICIES:
            raise ValueError(f"unknown artifact device_policy: {device_policy!r}")
        self.backbone: Any = backbone
        self.head = head
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.max_length = max_length
        self.input_rendering = _normalize_input_rendering(input_rendering)
        self.prefix_cache_size = prefix_cache_size
        self.device_policy = device_policy
        # Opt-in, tolerance-gated candidate-batching fast path (default off).
        self.candidate_batching = candidate_batching
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
        verify_release_base: bool = False,
        candidate_batching: bool = False,
        allow_device_policy_override: bool = False,
    ) -> OuroScorer:
        root = Path(path).resolve()
        manifest = contract.validate_artifact(
            root, verify_release_base=verify_release_base
        )
        execution = manifest["execution"]
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
        backbone.to_empty(device=target)
        # inv_freq is a non-persistent buffer: to_empty leaves it uninitialized,
        # so restore the exact FP32 frequencies the native module would compute.
        inv_freq, _ = _default_rope_init(config)
        backbone.model.rotary_emb.inv_freq = inv_freq.to(
            device=target, dtype=torch.float32
        )
        backbone.model.rotary_emb.original_inv_freq = backbone.model.rotary_emb.inv_freq
        _stream_base_weights(backbone, root / contract.BASE_WEIGHT_FILE)
        if base_dtype != torch.float32:
            backbone.to(dtype=base_dtype)
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
        state_tokens = (
            cache_cfg["state_block_tokens"]
            if cache_cfg["state_block_tokens"] > 0
            else contract.CACHE_MAX_STATE_BLOCK_TOKENS
        )
        return cls(
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
        )

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

    def question_logits(self, row: CandidateBatch) -> list[float]:
        candidates = [
            torch.tensor([ids], dtype=torch.long, device=self.device)
            for ids in row.candidate_ids
        ]
        with torch.inference_mode(), self._amp():
            context_cache = self._context_cache_for(row)
            logits = score_candidates_from_cache(
                self.backbone,
                self.head,
                context_cache,
                candidates,
                candidate_steps=contract.CANDIDATE_STEPS,
                candidate_batching=self.candidate_batching,
            )
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite Ouro candidate logits")
        return cast(list[float], logits.detach().float().cpu().tolist())

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
    verify_release_base: bool = False,
    candidate_batching: bool = False,
    allow_device_policy_override: bool = False,
) -> OuroScorer:
    """Load an Ouro decision artifact directory.

    The artifact's ``device_policy`` selects the default device: ``cuda`` when
    available for ``cuda-bf16-autocast``, ``mps`` for ``mps-fp32`` on Apple
    Silicon, and ``cpu`` for ``cpu-fp32``. An explicit ``device`` may only
    select a device type the policy permits; pass
    ``allow_device_policy_override=True`` to execute elsewhere under the
    artifact's declared execution tolerance. ``mps-fp32`` runs FP32 math on the
    Apple GPU and matches the CPU FP32 reference within
    ``contract.MPS_EXECUTION_TOLERANCE_ATOL``/``RTOL``.
    """
    return OuroScorer.from_directory(
        path,
        device=device,
        max_length=max_length,
        prefix_cache_size=prefix_cache_size,
        attention=attention,
        verify_release_base=verify_release_base,
        candidate_batching=candidate_batching,
        allow_device_policy_override=allow_device_policy_override,
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
) -> dict[str, Any]:
    """Assemble and validate a complete ``wyat-ouro-decision-v1`` artifact directory."""
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
    )
    (root / contract.MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    contract.validate_artifact(root)
    return manifest


__all__ = [
    "CandidateBatch",
    "ContextKVCache",
    "LayerKVCache",
    "OuroScorer",
    "ScalarScoreHead",
    "StepLoRALinear",
    "StepLoRARouter",
    "attach_step_lora",
    "candidate_kv_workspace_bytes",
    "load_ouro",
    "load_ouro_classes",
    "prefill_context",
    "run_candidate_steps",
    "score_candidates",
    "score_candidates_from_cache",
    "write_ouro_artifact",
]
