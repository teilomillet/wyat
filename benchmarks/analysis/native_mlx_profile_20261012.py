"""Bounded, fenced native MLX attribution; diagnostic fractions are not speedups.

Uses complete native encoded rows and the scorer's exact padded width. Every
adapter retains its two separate FP32 products and original scale/add order.
Run this standalone process with reduced scheduling priority on an idle Mac.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models import activations, base, qwen3
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from benchmarks.analysis.native_mlx_benchmark_20261012 import (
    check_frozen,
    compare_rows,
    digest,
    memory,
    metadata,
    prepare,
    read,
    snapshot,
)
from wyat.loading.mlx import _Adapter, load_native_mlx


class Recorder:
    def __init__(self, device: mx.Device) -> None:
        self.device = device
        self.samples: list[dict[str, Any]] = []

    def call(
        self, phase: str, operation: Callable[[], Any], layer: int | None = None
    ) -> Any:
        mx.synchronize(self.device)
        start = time.perf_counter_ns()
        result = operation()
        mx.eval(result)
        mx.synchronize(self.device)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        self.samples.append({"phase": phase, "layer": layer, "elapsed_ms": elapsed})
        return result


def projection(
    recorder: Recorder, owner: Any, value: mx.array, name: str, layer: int
) -> mx.array:
    if not isinstance(owner, _Adapter):
        return cast(
            mx.array, recorder.call(name + ".base", lambda: owner(value), layer)
        )
    base = recorder.call(name + ".base", lambda: owner.base(value), layer)
    down = recorder.call(
        name + ".adapter_down", lambda: value.astype(mx.float32) @ owner.down.T, layer
    )
    up = recorder.call(name + ".adapter_up", lambda: down @ owner.up.T, layer)
    scaled = recorder.call(name + ".adapter_scale", lambda: 2 * up, layer)
    return cast(
        mx.array, recorder.call(name + ".adapter_add", lambda: base + scaled, layer)
    )


def block(
    recorder: Recorder, layer: Any, hidden: mx.array, mask: Any, index: int
) -> mx.array:
    attention, mlp = layer.self_attn, layer.mlp
    normalized = recorder.call(
        "input_norm", lambda: layer.input_layernorm(hidden), index
    )
    batch, width, _ = normalized.shape
    queries = projection(recorder, attention.q_proj, normalized, "q", index)
    keys = projection(recorder, attention.k_proj, normalized, "k", index)
    values = projection(recorder, attention.v_proj, normalized, "v", index)
    queries = recorder.call(
        "q_norm_layout",
        lambda: attention.q_norm(
            queries.reshape(batch, width, attention.n_heads, -1)
        ).transpose(0, 2, 1, 3),
        index,
    )
    keys = recorder.call(
        "k_norm_layout",
        lambda: attention.k_norm(
            keys.reshape(batch, width, attention.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3),
        index,
    )
    values = recorder.call(
        "v_layout",
        lambda: values.reshape(batch, width, attention.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        ),
        index,
    )
    queries, keys = recorder.call(
        "qk_rope", lambda: (attention.rope(queries), attention.rope(keys)), index
    )
    output = recorder.call(
        "sdpa",
        lambda: scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=attention.scale, mask=mask
        ),
        index,
    )
    output = recorder.call(
        "attention_output_layout",
        lambda: output.transpose(0, 2, 1, 3).reshape(batch, width, -1),
        index,
    )
    output = projection(recorder, attention.o_proj, output, "o", index)
    residual = recorder.call("attention_residual", lambda: hidden + output, index)
    normalized = recorder.call(
        "post_attention_norm", lambda: layer.post_attention_layernorm(residual), index
    )
    gate = projection(recorder, mlp.gate_proj, normalized, "gate", index)
    up = projection(recorder, mlp.up_proj, normalized, "up", index)
    activation = recorder.call("swiglu", lambda: swiglu(gate, up), index)
    output = projection(recorder, mlp.down_proj, activation, "down", index)
    return cast(
        mx.array, recorder.call("mlp_residual", lambda: residual + output, index)
    )


def diagnostic(model: Any, encoded: Any, pause_ms: float) -> dict[str, Any]:
    length = max(len(row.ids) for row in encoded)
    if length > model.max_length or length * len(encoded) > model.max_batch_tokens:
        raise ValueError("native complete padded batch exceeds artifact budget")
    choices = max(len(row.candidate_positions) for row in encoded)
    recorder = Recorder(model.device)
    slept_ms = 0.0
    start = time.perf_counter_ns()
    with mx.stream(model.device):
        ids, endpoints, queries = recorder.call(
            "packing",
            lambda: (
                mx.array(
                    [
                        list(row.ids) + [model.pad_id] * (length - len(row.ids))
                        for row in encoded
                    ],
                    dtype=mx.int32,
                ),
                mx.array(
                    [
                        list(row.candidate_positions)
                        + [row.candidate_positions[0]]
                        * (choices - len(row.candidate_positions))
                        for row in encoded
                    ],
                    dtype=mx.int32,
                ),
                mx.array([row.query_position for row in encoded], dtype=mx.int32),
            ),
        )
        hidden = recorder.call("embedding", lambda: model.backbone.embed_tokens(ids))
        mask = create_attention_mask(hidden, None)
        for index, layer in enumerate(model.backbone.layers):
            hidden = block(recorder, layer, hidden, mask, index)
            # The fence above has completed the block. Sleep is outside all
            # attributed phases and reported separately, leaving the GPU idle.
            if pause_ms:
                sleep_start = time.perf_counter_ns()
                time.sleep(pause_ms / 1000)
                slept_ms += (time.perf_counter_ns() - sleep_start) / 1e6
        hidden = recorder.call("final_norm", lambda: model.backbone.norm(hidden))
        rows = mx.arange(len(encoded))
        candidates, query = recorder.call(
            "endpoint_gather",
            lambda: (hidden[rows[:, None], endpoints], hidden[rows, queries]),
        )
        logits = recorder.call("head", lambda: model.head(candidates, query))
        values = logits.tolist()
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    outputs = snapshot(
        encoded,
        [
            row[: len(item.candidate_positions)]
            for row, item in zip(values, encoded, strict=True)
        ],
    )
    return {
        "outputs": outputs,
        "samples": recorder.samples,
        "wall_ms_including_idle": elapsed_ms,
        "block_idle_ms_observed": slept_ms,
        "wall_ms_excluding_block_idle": elapsed_ms - slept_ms,
        "attribution": attribution(recorder.samples),
    }


def attribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, float] = defaultdict(float)
    for sample in samples:
        elapsed = sample["elapsed_ms"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid stage duration")
        phases[sample["phase"]] += elapsed
    total = sum(phases.values())
    if total <= 0:
        raise ValueError("empty stage profile")
    groups = {
        "mlp": ["gate.base", "up.base", "down.base", "swiglu", "mlp_residual"],
        "attention": [
            name
            for name in phases
            if name.startswith(("q.", "k.", "v.", "o."))
            or name
            in {
                "q_norm_layout",
                "k_norm_layout",
                "v_layout",
                "qk_rope",
                "sdpa",
                "attention_output_layout",
                "attention_residual",
            }
        ],
        "adapters": [name for name in phases if ".adapter_" in name],
        "base_projections": [name for name in phases if name.endswith(".base")],
        "norms": ["input_norm", "post_attention_norm", "final_norm"],
    }
    result = {}
    for name, members in groups.items():
        elapsed = sum(phases[member] for member in members)
        fraction = elapsed / total
        result[name] = {
            "phases": members,
            "elapsed_ms": elapsed,
            "diagnostic_fraction": fraction,
            "conditional_overall_if_component_2x": 1 / (1 - fraction / 2),
            "conditional_removal_ceiling": 1 / (1 - fraction) if fraction < 1 else None,
        }
    return {
        "stage_total_ms": total,
        "phase_total_ms": dict(phases),
        "overlapping_groups": result,
    }


def timed(model: Any, rows: Any) -> dict[str, Any]:
    mx.synchronize(model.device)
    start = time.perf_counter_ns()
    values = model._logits(rows)
    mx.synchronize(model.device)
    return {
        "elapsed_ms": (time.perf_counter_ns() - start) / 1e6,
        "outputs": snapshot(rows, values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--ordinary",
        action="store_true",
        help="Use ordinary MLX instead of admitted Metal",
    )
    parser.add_argument("--block-idle-ms", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.block_idle_ms) or args.block_idle_ms < 1:
        raise ValueError("diagnostic requires at least 1 ms GPU idle between blocks")
    if args.output.exists():
        raise FileExistsError(
            "choose fresh output; retained evidence cannot be replaced"
        )
    frozen = read(args.frozen)
    check_frozen(frozen)
    result: dict[str, Any] = {
        "completed": False,
        "passed": False,
        "schema": "native-mlx-profile-v1",
        "frozen_sha256": digest(args.frozen),
        "artifact_manifest_sha256": digest(args.artifact / "native_manifest.json"),
        "protocol": {
            "one_model": True,
            "cache_limit_bytes": 256 << 20,
            "mlx_memory_limit_bytes": 6144 << 20,
            "block_idle_ms_requested": args.block_idle_ms,
            "case_idle_ms_requested": 100,
            "ordinary_calls": (
                "one warmup, one before, one after each fenced diagnostic"
            ),
            "interpretation": (
                "fences alter scheduling and fusion; Amdahl fractions are "
                "conditional diagnostics, not ordinary-path decomposition or speedups"
            ),
            "memory_limit_scope": (
                "MLX allocator guideline in this standalone process; "
                "not a total process RSS guarantee"
            ),
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the output before model work; a failed run remains observable.
    with args.output.open("x") as stream:
        stream.write(json.dumps(result) + "\n")
    previous_cache = mx.set_cache_limit(256 << 20)
    previous_memory = mx.set_memory_limit(6144 << 20)
    try:
        model = load_native_mlx(args.artifact, metal=not args.ordinary)
        if not args.ordinary and not model.metal_enabled:
            raise RuntimeError(
                "requested Metal not admitted; profile outcome is unknown"
            )
        batches = prepare(model, frozen)
        names = args.case
        if not names:
            records = {item["id"]: item for item in frozen["records"]}
            candidates = [
                case
                for case in frozen["cases"]
                if len(case["records"]) == 1
                and len(records[case["records"][0]]["encoded"]) == 1
                and len(
                    records[case["records"][0]]["encoded"][0]["candidate_positions"]
                )
                == 77
            ]
            if not candidates:
                raise ValueError(
                    "no singleton 77-option frozen case; select --case explicitly"
                )
            longest = max(candidates, key=lambda case: len(batches[case["id"]][0].ids))
            names = [longest["id"], "batch8-mixed"]
        if len(set(names)) != len(names) or any(name not in batches for name in names):
            raise ValueError("duplicate or unknown frozen case")
        result.update(
            metadata=metadata(),
            model_id=model.model_id,
            metal_enabled=model.metal_enabled,
        )
        result["sources_sha256"] = {
            str(path): digest(path)
            for path in [
                Path(__file__).resolve(),
                Path(inspect.getfile(activations)),
                Path(inspect.getfile(base)),
                Path(inspect.getfile(qwen3)),
                Path(inspect.getfile(prepare)),
                Path(inspect.getfile(type(model))),
                Path(__file__).resolve().parents[2] / "wyat/backends/wyat/metal.py",
                Path(__file__).resolve().parents[2] / "wyat/backends/wyat/mlx.py",
                args.frozen.resolve(),
                (args.artifact / "native_manifest.json").resolve(),
            ]
        }
        result["memory_after_load"] = memory("metal")
        for name in names:
            entry: dict[str, Any] = {"id": name, "passed": False}
            result["cases"].append(entry)
            try:
                rows = batches[name]
                entry.update(
                    lengths=[len(row.ids) for row in rows],
                    candidate_counts=[len(row.candidate_positions) for row in rows],
                )
                entry["warmup"] = timed(model, rows)
                time.sleep(0.1)
                entry["before"] = timed(model, rows)
                time.sleep(0.1)
                entry["diagnostic"] = diagnostic(model, rows, args.block_idle_ms)
                time.sleep(0.1)
                entry["after"] = timed(model, rows)
                gates = frozen["quality_gates"]
                entry["quality"] = {
                    "before_vs_diagnostic": compare_rows(
                        entry["before"]["outputs"],
                        entry["diagnostic"]["outputs"],
                        gates,
                    ),
                    "before_vs_after": compare_rows(
                        entry["before"]["outputs"], entry["after"]["outputs"], gates
                    ),
                    "warmup_vs_before": compare_rows(
                        entry["warmup"]["outputs"], entry["before"]["outputs"], gates
                    ),
                }
                ordinary_ms = statistics.mean(
                    [entry["before"]["elapsed_ms"], entry["after"]["elapsed_ms"]]
                )
                entry["diagnostic_to_ordinary_ratio_excluding_idle"] = (
                    entry["diagnostic"]["wall_ms_excluding_block_idle"] / ordinary_ms
                )
                entry["passed"] = all(
                    gate["pass"] for gate in entry["quality"].values()
                )
            except Exception as error:
                entry["error"] = repr(error)
            entry["memory_after_case"] = memory("metal")
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            time.sleep(0.1)
        result["completed"] = True
        result["passed"] = all(case["passed"] for case in result["cases"])
    except Exception as error:
        result["error"] = repr(error)
    finally:
        result["memory_at_end"] = memory("metal")
        mx.synchronize()
        mx.clear_cache()
        mx.set_cache_limit(previous_cache)
        mx.set_memory_limit(previous_memory)
        result["sources_at_end_sha256"] = {
            name: digest(Path(name)) for name in result.get("sources_sha256", {})
        }
        result["source_fence_pass"] = (
            result.get("sources_sha256") == result["sources_at_end_sha256"]
        )
        result["passed"] = result["passed"] and result["source_fence_pass"]
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
