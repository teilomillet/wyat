"""Tiny maintained masked-GQA dispatch oracle; no model or timing claim."""

from __future__ import annotations

import inspect
import json
import platform
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import torch
from torch.nn import functional as functional
from transformers import cache_utils, masking_utils
from transformers.integrations import sdpa_attention as maintained
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.evidence import freeze_sources

type Attention = Callable[..., tuple[torch.Tensor, None]]
attention = cast(Attention, maintained.sdpa_attention_forward)


class Event(Protocol):
    name: str
    input_shapes: list[list[int]]


@dataclass(frozen=True, slots=True)
class Observation:
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    last_strides: tuple[int, int, int]
    mask_shape: tuple[int, ...] | None
    operators: tuple[str, ...]
    attention_events: tuple[dict[str, object], ...]
    oracle_max_abs_error: float


def observe(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    expected: torch.Tensor,
) -> tuple[torch.Tensor, Observation]:
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
        ) as profile,
    ):
        output, _ = attention(module, query, key, value, mask, scaling=8**-0.5)
    events = cast(Sequence[Event], profile.events())
    return output, Observation(
        tuple(query.shape),
        tuple(key.shape),
        tuple(value.shape),
        (query.stride(-1), key.stride(-1), value.stride(-1)),
        None if mask is None else tuple(mask.shape),
        tuple(sorted({event.name for event in events})),
        tuple(
            {"operator": event.name, "input_shapes": event.input_shapes}
            for event in events
            if "scaled_dot_product" in event.name
        ),
        float((output - expected).abs().max().item()),
    )


def probe(output: Path) -> dict[str, object]:
    """Fixed deterministic FP32 arrays challenge alignment and grouped head use."""
    if output.exists():
        raise FileExistsError("Preserve existing probe receipts")
    root = Path(__file__).resolve().parents[4]
    sources = freeze_sources(
        {
            "probe": Path(__file__),
            "evidence": root / "benchmarks/analysis/evidence.py",
            "hf_sdpa": Path(inspect.getfile(maintained)),
            "hf_mask": Path(inspect.getfile(masking_utils)),
            "hf_cache": Path(inspect.getfile(cache_utils)),
            "hf_qwen3": Path(inspect.getfile(modeling_qwen3)),
        },
        output,
    )
    result: dict[str, object] = {
        "completed": False,
        "passed": False,
        "scope": "tiny CPU maintained-dispatch and causal oracle; no speed claim",
        "sources_sha256": sources.hashes(),
        "sources_archive": sources.archive(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "transformers": version("transformers"),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "hf_mps_snapshot": maintained._is_torch_mps_available,
            "hf_torch_ge_2_13": maintained._is_torch_greater_or_equal_than_2_13,
            "torch_threads": torch.get_num_threads(),
        },
        "protocol": {
            "prefix": 5,
            "suffix": 6,
            "query_heads": 4,
            "kv_heads": 2,
            "head_dimension": 8,
            "dtype": "float32",
            "mask": "key_index <= prefix_length + suffix_query_index",
            "global_flags_changed": False,
            "oracle": "uniform mean key index plus 100 times KV-head index",
        },
        "limits": [
            "No real weights or model execution; this is one tiny attention shape.",
            "Profiler input shapes establish actual flash Q/K/V head counts.",
            "Source archives identify disk sources, not compiled binary identity.",
            "HF import-time availability is process/context dependent.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    try:
        prefix, suffix, heads, dimension = 5, 6, 4, 8
        query = torch.zeros(1, heads, suffix, dimension)
        key = torch.zeros(1, 2, prefix + suffix, dimension)
        broadcast_value = (
            torch.arange(prefix + suffix, dtype=torch.float32)[None, None, :, None]
            + 100 * torch.arange(2, dtype=torch.float32)[None, :, None, None]
        ).expand(1, 2, prefix + suffix, dimension)
        value = broadcast_value.contiguous()
        mask = (
            torch.arange(prefix + suffix)[None, None, None, :]
            <= (prefix + torch.arange(suffix))[None, None, :, None]
        )
        expected = (
            (prefix + torch.arange(suffix, dtype=torch.float32))[None, :, None, None]
            / 2
            + 100 * (torch.arange(heads) // 2)[None, None, :, None]
        ).expand(1, suffix, heads, dimension)
        actual, contiguous = observe(query, key, value, mask, expected)
        _, zero_stride = observe(query, key, broadcast_value, mask, expected)
        _, upper_left = observe(query, key, value, None, expected)
        with torch.inference_mode():
            full = functional.scaled_dot_product_attention(
                torch.zeros(1, heads, prefix + suffix, dimension),
                key,
                value,
                is_causal=True,
                enable_gqa=True,
            )[:, :, prefix:].transpose(1, 2)
        full_difference = float((actual - full).abs().max().item())
        flash = tuple(
            event
            for event in contiguous.attention_events
            if event["operator"] == "aten::_scaled_dot_product_flash_attention_for_cpu"
        )
        expected_shapes = [[1, 4, 6, 8], [1, 2, 11, 8], [1, 2, 11, 8]]
        unexpanded = any(
            cast(list[list[int]], event["input_shapes"])[:3] == expected_shapes
            for event in flash
        )
        checks = {
            "actual_flash_uses_unexpanded_kv_heads": unexpanded,
            "contiguous_matches_uniform_oracle": contiguous.oracle_max_abs_error
            <= 1e-5,
            "suffix_matches_full_causal": full_difference <= 1e-5,
            "wrong_upper_left_fails_oracle": upper_left.oracle_max_abs_error > 1,
            "stride_zero_forces_math": "aten::_scaled_dot_product_attention_math"
            in zero_stride.operators,
        }
        result.update(
            completed=True,
            passed=all(checks.values()),
            checks=checks,
            full_vs_suffix_max_abs_error=full_difference,
            contiguous_masked=asdict(contiguous),
            stride_zero_masked=asdict(zero_stride),
            no_mask_upper_left=asdict(upper_left),
        )
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}")
        raise
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = probe(arguments.output)
    print(json.dumps(receipt["checks"], indent=2))
    if receipt["completed"] is not True or receipt["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
