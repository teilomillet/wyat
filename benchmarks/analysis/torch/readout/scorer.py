"""Temporary CPU endpoint candidate; production retained-prefix scoring is unchanged."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from wyat.backends.wyat.cpu.prefix import PrefixScorer
from wyat.backends.wyat.torch import WyatScorer
from wyat.core import Request

from .block import selected_last_block


def selected_hidden(
    base: WyatScorer,
    ids: torch.Tensor,
    positions: torch.Tensor,
    cache: DynamicCache,
    prefix_length: int,
) -> torch.Tensor:
    """Run stock early blocks; capture positions/mask before any cache append."""
    backbone = base.backbone
    if cache.get_seq_length() != prefix_length:
        raise ValueError("Selected scoring requires a fresh complete prefix owner")
    hidden = cast(torch.Tensor, backbone.embed_tokens(ids))
    position_ids = (torch.arange(ids.shape[1]) + prefix_length)[None]
    mask = create_causal_mask(
        config=backbone.config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=cache,
        position_ids=position_ids,
    )
    embeddings = cast(
        tuple[torch.Tensor, torch.Tensor], backbone.rotary_emb(hidden, position_ids)
    )
    for layer in backbone.layers[:-1]:
        hidden = cast(
            torch.Tensor,
            layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=embeddings,
            ),
        )
    return selected_last_block(
        cast(Qwen3DecoderLayer, backbone.layers[-1]),
        backbone.norm,
        hidden,
        positions,
        embeddings,
        cache,
        prefix_length=prefix_length,
    )


class SelectedScorer(PrefixScorer):
    """Independent eager snapshots and per-call disposable caches; no routing change.

    This experimental copy owns only admitted tensor packing/forward/readout.
    Whole-batch fallback uses the original production tensor owner. It earns a
    production boundary only after separately reviewed full77 confirmation.
    """

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        self._require_no_autocast()
        self._require_no_hooks(self.base)
        if not requests:
            return []
        encoded = [self.base.encode_request(request) for request in requests]
        if not all(self._admits(row) for row in encoded):
            return self.base._score_encoded(encoded)
        prefix_length = len(self.prefix)
        batch = len(encoded)
        length = max(len(row.ids) for row in encoded) - prefix_length
        width = max(len(row.candidate_positions) for row in encoded)
        ids = torch.full((batch, length), self.base.pad_id, dtype=torch.long)
        positions = torch.zeros((batch, width + 1), dtype=torch.long)
        for index, row in enumerate(encoded):
            suffix = row.ids[prefix_length:]
            ids[index, : len(suffix)] = torch.tensor(suffix, dtype=torch.long)
            positions[index, : len(row.candidate_positions)] = torch.tensor(
                [value - prefix_length for value in row.candidate_positions]
            )
            positions[index, -1] = row.query_position - prefix_length
        with torch.inference_mode():
            hidden = selected_hidden(
                self.base, ids, positions, self._fork(batch), prefix_length
            )
            logits = cast(torch.Tensor, self.base.head(hidden[:, :-1], hidden[:, -1]))
        values = cast(list[list[float]], logits.float().cpu().tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]
