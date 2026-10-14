"""An explicit FP32 control, independent of the production CUDA autocast path."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol, cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_outputs import BaseModelOutputWithPast

from wyat.backends.wyat.torch import WyatScorer
from wyat.core import Request


class PrecisionOwner(Protocol):
    fp32_precision: str


@contextmanager
def ieee_precision() -> Iterator[None]:
    """This owned process requires modern controls; never mix legacy TF32 flags.

    https://docs.pytorch.org/docs/2.12/notes/cuda.html
    """
    global_owner = cast(PrecisionOwner, torch.backends)
    old_global = global_owner.fp32_precision
    old_matmul = torch.backends.cuda.matmul.fp32_precision
    try:
        global_owner.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_matmul
        global_owner.fp32_precision = old_global


def require_fp32(scorer: WyatScorer) -> None:
    for owner in (cast(nn.Module, scorer.backbone), scorer.head):
        for name, value in (*owner.named_parameters(), *owner.named_buffers()):
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError(f"FP32 control has non-FP32 tensor: {name}")


class FP32Scorer:
    """Same encoder/head and endpoint contract; no nested BF16-enabled scorer."""

    def __init__(self, base: WyatScorer) -> None:
        require_fp32(base)
        self.base = base

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        if not requests:
            return []
        base = self.base
        encoded = [base.encode_request(request) for request in requests]
        length = max(len(row.ids) for row in encoded)
        width = max(len(row.candidate_positions) for row in encoded)
        batch = len(encoded)
        ids = torch.full((batch, length), base.pad_id, dtype=torch.long)
        endpoints = torch.zeros((batch, width), dtype=torch.long)
        for index, row in enumerate(encoded):
            ids[index, : len(row.ids)] = torch.tensor(row.ids, dtype=torch.long)
            endpoints[index, : len(row.candidate_positions)] = torch.tensor(
                row.candidate_positions, dtype=torch.long
            )
        ids, endpoints = ids.to(base.device), endpoints.to(base.device)
        rows = torch.arange(batch, device=base.device)
        query = torch.tensor(
            [row.query_position for row in encoded], device=base.device
        )
        with (
            torch.inference_mode(),
            torch.autocast(device_type=base.device.type, enabled=False),
            sdpa_kernel(SDPBackend.MATH),
        ):
            output = cast(
                BaseModelOutputWithPast,
                base.backbone(input_ids=ids, use_cache=False),
            )
            hidden = output.last_hidden_state
            if hidden is None or hidden.dtype != torch.float32:
                raise ValueError("FP32 control requires FP32 hidden states")
            logits = cast(
                torch.Tensor,
                base.head(hidden[rows[:, None], endpoints], hidden[rows, query]),
            )
            if logits.dtype != torch.float32:
                raise ValueError("FP32 control requires FP32 head output")
        values = cast(list[list[float]], logits.cpu().tolist())
        return [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ]
