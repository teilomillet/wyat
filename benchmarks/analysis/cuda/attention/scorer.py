"""Explicit causal mask admits maintained KV repetition and efficient SDPA.

Torch 2.12.1 uses CUTLASS OpMultiplyAddFastF32 for float/SM80 attention.
FP32 tensors and process IEEE GEMM settings do not make this arithmetic IEEE.
https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from benchmarks.analysis.cuda.scorer import FP32Scorer, require_fp32
from wyat.backends.kai.encoding import EncodedRequest
from wyat.backends.kai.torch import CandidateHead, KaiScorer
from wyat.core import Request


class MaskedBackbone(nn.Module):
    """One existing backbone owner; fresh square full-prefill mask per call.

    The original model/config is unchanged. No cache, registry mutation or
    backend fallback is permitted. HF performs the KV repetition itself.
    """

    def __init__(self, model: Qwen3Model) -> None:
        super().__init__()
        if (
            any(module.training for module in model.modules())
            or model.has_sliding_layers
            or model.config._attn_implementation != "sdpa"
        ):
            raise ValueError("Efficient candidate requires eval full attention")
        self.model = model
        self.training = False

    def forward(
        self, *, input_ids: torch.Tensor, use_cache: bool
    ) -> BaseModelOutputWithPast:
        if (
            use_cache is not False
            or input_ids.ndim != 2
            or input_ids.dtype != torch.long
        ):
            raise ValueError("Efficient candidate requires uncached 2D token IDs")
        length = input_ids.shape[1]
        if length == 0:
            raise ValueError("Efficient candidate requires nonempty token IDs")
        positions = torch.arange(length, device=input_ids.device)
        mask = (positions[:, None] >= positions[None, :])[None, None]
        # This nested context deliberately overrides FP32Scorer's math control.
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return cast(
                BaseModelOutputWithPast,
                self.model(input_ids=input_ids, attention_mask=mask, use_cache=False),
            )


@dataclass(frozen=True)
class ScorerView:
    """Only the fields consumed by the immutable FP32Scorer boundary."""

    base: KaiScorer
    backbone: MaskedBackbone

    @property
    def head(self) -> CandidateHead:
        return self.base.head

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def pad_id(self) -> int:
        return self.base.pad_id

    def encode_request(self, request: Request) -> EncodedRequest:
        return self.base.encode_request(request)


def efficient_scorer(base: KaiScorer) -> FP32Scorer:
    require_fp32(base)
    view = ScorerView(base, MaskedBackbone(base.backbone))
    # The frozen research API names KaiScorer nominally but consumes only this
    # typed view's encoder/device/pad/head/module-call contract. Do not change
    # its old source or misrepresent the adapter as a Qwen3Model subclass.
    return FP32Scorer(cast(KaiScorer, view))
