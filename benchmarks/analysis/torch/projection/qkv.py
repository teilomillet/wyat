"""One packed allocation shared by separate, KV and QKV linear operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch.nn import functional as F

type Arm = Literal["separate", "kv", "qkv"]
type Outputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def require_fp32(value: torch.Tensor) -> None:
    if (
        value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.requires_grad
        or torch.is_autocast_enabled("cpu")
    ):
        raise ValueError("Projection requires CPU FP32 without gradients/autocast")


@dataclass(frozen=True)
class PackedQKV:
    weight: torch.Tensor
    query_width: int
    kv_width: int

    @classmethod
    def from_weights(
        cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> PackedQKV:
        for item in (query, key, value):
            require_fp32(item)
            if item.ndim != 2 or not torch.isfinite(item).all().item():
                raise ValueError("Projection weights require finite matrices")
        if (
            key.shape != value.shape
            or query.shape[1] != key.shape[1]
            or query.shape[0] != 2 * key.shape[0]
            or min(key.shape) <= 0
        ):
            raise ValueError("Packing requires biasless Q16/KV8 width geometry")
        return cls(torch.cat((query, key, value)), query.shape[0], key.shape[0])

    @property
    def views(self) -> Outputs:
        return cast(
            Outputs,
            torch.split(self.weight, [self.query_width, self.kv_width, self.kv_width]),
        )

    def project(self, value: torch.Tensor, arm: Arm) -> Outputs:
        require_fp32(value)
        if value.ndim not in (2, 3) or value.shape[-1] != self.weight.shape[1]:
            raise ValueError("Projection input width/rank differs")
        if arm == "separate":
            query, key, values = self.views
            return F.linear(value, query), F.linear(value, key), F.linear(value, values)
        if arm == "kv":
            query = F.linear(value, self.views[0])
            key, values = torch.split(
                F.linear(value, self.weight[self.query_width :]), self.kv_width, dim=-1
            )
            return query, key, values
        if arm == "qkv":
            return cast(
                Outputs,
                torch.split(
                    F.linear(value, self.weight),
                    [self.query_width, self.kv_width, self.kv_width],
                    dim=-1,
                ),
            )
        raise ValueError("Unknown projection arm")


def quality(reference: Outputs, actual: Outputs) -> dict[str, object]:
    """Symmetric max tolerance; projection evidence is not full-model quality."""
    failures = 0
    maximum = 0.0
    relative = 0.0
    for expected, observed in zip(reference, actual, strict=True):
        require_fp32(expected)
        require_fp32(observed)
        if expected.shape != observed.shape:
            raise ValueError("Projection output geometry differs")
        # FP64 computes the gate without rounding its FP32 operands/tolerance.
        left, right = expected.double(), observed.double()
        error = (left - right).abs()
        scale = torch.maximum(left.abs(), right.abs())
        finite = torch.isfinite(left) & torch.isfinite(right)
        failures += int((~finite | (error > (scale * 1e-3).clamp_min(1e-3))).sum())
        if finite.all().item():
            maximum = max(maximum, float(error.max()))
            relative = max(relative, float((error / scale.clamp_min(1e-300)).max()))
        else:
            return {
                "passed": False,
                "finite": False,
                "failed_elements": failures,
                "max_absolute_error": None,
                "max_relative_error": None,
            }
    return {
        "passed": failures == 0,
        "finite": True,
        "failed_elements": failures,
        "max_absolute_error": maximum,
        "max_relative_error": relative,
        "atol": 1e-3,
        "rtol": 1e-3,
        "rule": "abs(a-b)<=max(atol,rtol*max(abs(a),abs(b)))",
    }
