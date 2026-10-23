"""Exact padded-candidate batch prototype for Apple GPU (MPS) timing studies.

The production serial path scores candidates one at a time; the loader's
``candidate_batching=True`` fast path groups candidates by *exact* token length
and runs each group as one batched forward. This module prototypes the
complementary variant for heterogeneous lengths: pad every candidate of a
question to ``S_max`` and score all of them in one batched forward, with an exact
additive ``-inf`` causal mask so padding can never influence a real token.

Exactness argument
------------------
Each batch row carries its own K/V buffer of shape ``[1, heads, P + S_max, d]``:
the retained shared-context bank in positions ``0 .. P-1`` (broadcast over the
batch) and that row's own candidate suffix in positions ``P .. P + S_max - 1``.
No row can attend another row's suffix (the batch dimension isolates them), so
only a causal mask is needed - every real token keeps exactly the positions and
causal mask it has in the serial path. Padded suffix positions sit causally
*after* every real query token of that row, so they are masked out for all real
queries; the readout gathers hidden state at ``length - 1`` per row. Results are
therefore exact up to floating-point reduction order (measured, not assumed).

This file is a study prototype only; nothing here is wired into the runtime
loader. ``scripts/ouro_speed/mps/probe_batching.py`` measures it against the
serial path and the loader's grouped fast path.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mps_common as common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT))

import torch  # noqa: E402


def padded_workspace_bytes(
    backbone: Any,
    context_length: int,
    candidate_count: int,
    padded_length: int,
    element_size: int = 4,
) -> int:
    """Bytes of the single reused K/V buffer pair for one batched step."""
    native = backbone.model
    head_dim = getattr(native.config, "head_dim", None) or (
        native.config.hidden_size // native.config.num_attention_heads
    )
    elements = (
        2
        * candidate_count
        * native.config.num_key_value_heads
        * (context_length + padded_length)
        * head_dim
    )
    return int(elements * element_size)


def _padded_causal_mask(
    context_length: int, padded_length: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Additive causal mask ``[1, 1, S_max, P + S_max]`` (shared by every row)."""
    query = torch.arange(context_length, context_length + padded_length, device=device)
    keys = torch.arange(context_length + padded_length, device=device)
    blocked = keys[None, :] > query[:, None]
    mask = torch.zeros(
        (padded_length, context_length + padded_length), device=device, dtype=dtype
    )
    return mask.masked_fill(blocked, torch.finfo(dtype).min)[None, None]


class _BatchedReadContextKV:
    """Per-row ``context + own padded suffix`` K/V, one reused buffer pair/step."""

    def __init__(
        self,
        bank: tuple[Any, ...],
        step: int,
        batch: int,
        suffix_length: int,
        workspace: dict[str, torch.Tensor | None],
    ) -> None:
        self.bank = bank
        self.step = step
        self.batch = batch
        self.suffix_length = suffix_length
        self.workspace = workspace
        self.context_length = int(bank[0].key.shape[2])
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
        context = self.bank[self.count]
        total = self.context_length + self.suffix_length
        key_buffer = self.workspace.get("key")
        if key_buffer is None or key_buffer.shape[2] != total:
            shape = (self.batch, context.key.shape[1], total, context.key.shape[3])
            key_buffer = torch.empty(shape, device=key.device, dtype=key.dtype)
            value_buffer = torch.empty(shape, device=value.device, dtype=value.dtype)
            self.workspace["key"] = key_buffer
            self.workspace["value"] = value_buffer
        value_buffer = self.workspace["value"]
        assert value_buffer is not None
        assert key_buffer is not None
        # Broadcast the shared context bank over the batch rows; overwrite the
        # per-row candidate suffix in place (the layer consumes it immediately).
        key_buffer[:, :, : self.context_length].copy_(context.key)
        value_buffer[:, :, : self.context_length].copy_(context.value)
        key_buffer[:, :, self.context_length :].copy_(key)
        value_buffer[:, :, self.context_length :].copy_(value)
        self.count += 1
        return key_buffer[:, :, :total], value_buffer[:, :, :total]


def _batched_recurrent_step(
    backbone: Any,
    hidden: torch.Tensor,
    context_length: int,
    step: int,
    cache: _BatchedReadContextKV,
    mask: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(
        context_length, context_length + hidden.shape[1], device=hidden.device
    )
    rotary = backbone.rotary_emb(hidden, positions[None])
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
    return backbone.norm(hidden)


@torch.inference_mode()
def score_candidates_batched_padded(
    backbone: Any,
    head: Any,
    context_cache: Any,
    candidates: Sequence[torch.Tensor],
    *,
    candidate_steps: int = 2,
) -> torch.Tensor:
    """Score all candidates in one padded forward per recurrent step."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    native = backbone.model
    device = candidates[0].device
    batch = len(candidates)
    lengths = [int(candidate.shape[1]) for candidate in candidates]
    padded_length = max(lengths)
    padded_ids = torch.zeros((batch, padded_length), dtype=torch.long, device=device)
    for index, candidate in enumerate(candidates):
        if candidate.shape[0] != 1:
            raise ValueError("each candidate must be shape [1, S]")
        if candidate.dtype != torch.long:
            raise ValueError("candidate token ids must be torch.long")
        padded_ids[index, : lengths[index]] = candidate[0]
    hidden = native.embed_tokens(padded_ids)
    mask = _padded_causal_mask(
        int(context_cache.length), padded_length, hidden.dtype, hidden.device
    )
    router = getattr(backbone, "_step_lora_router", None)
    workspace: dict[str, torch.Tensor | None] = {"key": None, "value": None}
    for step in range(candidate_steps):
        bank = context_cache.banks[step if context_cache.policy == "matched" else 0]
        cache = _BatchedReadContextKV(bank, step, batch, padded_length, workspace)
        with router.activate(step) if router is not None else _nullcontext():
            hidden = _batched_recurrent_step(
                native, hidden, int(context_cache.length), step, cache, mask
            )
    rows = torch.arange(batch, device=device)
    last_hidden = hidden[rows, torch.tensor(lengths, device=device) - 1]
    scores = head(last_hidden)
    if scores.shape != (batch, 1):
        raise ValueError("scalar score head must return shape [K, 1]")
    return scores[:, 0]


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
