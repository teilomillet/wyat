"""Quality-gated FP32 projection packing; experimental prefill only.

The attention post-projection recipe follows MLX-LM 0.31.3 Qwen3 (Apple,
MIT license). Its norms, RoPE, SDPA and output projection are reused unchanged.
Only projection grouping changes; the maintained Transformer and head remain.
"""

from __future__ import annotations

import gc
import hashlib
import json
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.utils import tree_flatten
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3 import MLP, Attention, Qwen3Model

from benchmarks.analysis.mlx.profile import ArrayMap, quality
from benchmarks.analysis.reference import read_reference
from wyat import Model, Request
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.benchmark import BenchmarkEngine, BenchmarkProtocol, benchmark_pair
from wyat.evaluation import EvaluationCase, case_from_dict
from wyat.wire import decode_json


class AttentionMap(Protocol):
    def __call__(
        self, x: mx.array, mask: mx.array | str | None, cache: object | None = None
    ) -> mx.array: ...


class MutableBlock(Protocol):
    self_attn: AttentionMap
    mlp: ArrayMap


def packed_linear(projections: Sequence[Linear]) -> Linear:
    """Concatenate exact stored FP32 values once, outside every timed call."""
    weights = [projection.weight for projection in projections]
    if any(value.dtype != mx.float32 for value in weights):
        raise ValueError("Packing requires FP32 source weights")
    packed = Linear(
        weights[0].shape[1], sum(value.shape[0] for value in weights), False
    )
    packed.weight = mx.concatenate(weights, axis=0)
    mx.eval(packed.weight)
    start = 0
    for original in weights:
        end = start + original.shape[0]
        if not bool(mx.all(packed.weight[start:end] == original).item()):
            raise ValueError("Projection packing changed a source value")
        start = end
    return packed


# MLX's base type is unavailable on CPU-only installations.
class PackedAttention(Module):  # type: ignore[misc, unused-ignore]
    """Same full causal prefill; grouped query width need not equal hidden width."""

    def __init__(self, original: Attention) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.n_heads = original.n_heads
        self.n_kv_heads = original.n_kv_heads
        self.scale = original.scale
        self.query_width = original.q_proj.weight.shape[0]
        self.key_width = original.k_proj.weight.shape[0]
        self.qkv = packed_linear([original.q_proj, original.k_proj, original.v_proj])
        self.q_norm = cast(ArrayMap, original.q_norm)
        self.k_norm = cast(ArrayMap, original.k_norm)
        self.rope = cast(ArrayMap, original.rope)
        self.o_proj = original.o_proj
        # No reference to original or its three linears is retained. Baseline
        # owns its original weights for the paired comparison.

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: object | None = None,
    ) -> mx.array:
        if cache is not None:
            raise ValueError("Experimental packed attention supports prefill only")
        batch, length, _ = x.shape
        queries, keys, values = mx.split(
            self.qkv(x), [self.query_width, self.query_width + self.key_width], axis=-1
        )
        queries = self.q_norm(
            queries.reshape(batch, length, self.n_heads, -1)
        ).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(batch, length, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch, length, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        output = mx.fast.scaled_dot_product_attention(
            self.rope(queries), self.rope(keys), values, scale=self.scale, mask=mask
        )
        return cast(
            mx.array,
            self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1)),
        )


class PackedMLP(Module):  # type: ignore[misc, unused-ignore]
    def __init__(self, original: MLP) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.gate_up = packed_linear([original.gate_proj, original.up_proj])
        self.down_proj = original.down_proj

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up(x), 2, axis=-1)
        activation = cast(Callable[[mx.array, mx.array], mx.array], swiglu)
        return cast(mx.array, self.down_proj(activation(gate, up)))


def packed_backbone(original: Qwen3Model, *, qkv: bool, gate_up: bool) -> Qwen3Model:
    """Share immutable arrays; replaced modules drop their original references."""
    backbone = Qwen3Model(original.args)
    parameters = cast(Callable[[], object], original.parameters)()
    flattened = cast(list[tuple[str, mx.array]], tree_flatten(parameters))
    backbone.load_weights(flattened, strict=True)
    blocks = cast(Sequence[MutableBlock], backbone.layers)
    for block in blocks:
        if qkv:
            block.self_attn = PackedAttention(cast(Attention, block.self_attn))
        if gate_up:
            block.mlp = PackedMLP(cast(MLP, block.mlp))
    backbone.eval()
    return backbone


def variant(scorer: MLXWyatScorer, *, qkv: bool, gate_up: bool) -> MLXWyatScorer:
    with mx.stream(scorer.device):
        backbone = packed_backbone(scorer.backbone, qkv=qkv, gate_up=gate_up)
    return MLXWyatScorer(
        backbone,
        scorer.head,
        scorer.encoder,
        model_id=scorer.model_id,
        device=scorer.device,
    )


def score(
    scorer: MLXWyatScorer, requests: Sequence[Request], batch_size: int
) -> list[list[float]]:
    values: list[list[float]] = []
    for start in range(0, len(requests), batch_size):
        values.extend(scorer.score_many(requests[start : start + batch_size]))
    return values


class Arguments(Namespace):
    batch_size: int
    repeats: int
    output: Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks/analysis/results/mlx-fusion77-b8.json",
    )
    options = parser.parse_args(namespace=Arguments())
    if options.batch_size < 1 or options.repeats < 3:
        raise ValueError("Batch size must be positive and repeats at least three")
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases: tuple[EvaluationCase, ...] = tuple(
        case_from_dict(decode_json(row)) for row in dataset.read_text().splitlines()
    )
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    scorer = MLXWyatScorer.from_directory(
        root / ".models/Decision-2.0-Kai-0.6B",
        max_length=512,
        token_cache_size=0,
        device="gpu",
    )
    frozen = read_reference(
        root / "benchmarks/real/torch-fp32-reference77.json",
        cases,
        scorer,
        dataset_sha256=dataset_hash,
    )
    saved: list[list[float]] = []
    for row in frozen.rows:
        if row.prediction is None:
            raise ValueError("Frozen reference must contain every prediction")
        saved.append(list(row.prediction.scores))
    requests = tuple(case.request for case in cases)
    baseline_values = score(scorer, requests, options.batch_size)
    baseline_gate = quality(cases, saved, baseline_values, scorer.model_id)
    if not baseline_gate.passed:
        raise ValueError("Unchanged FP32 baseline failed reference gate")
    payload: dict[str, object] = {
        "scope": "77-case training-source runtime assay; checkpoint exposure unknown",
        "hypothesis": (
            "Exact FP32 projection packing preserves strict quality and lowers latency"
        ),
        "model_id": scorer.model_id,
        "dataset_sha256": dataset_hash,
        "case_ids": [case.id for case in cases],
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                root / "wyat/backends/wyat/mlx.py",
                root / "wyat/backends/wyat/encoding.py",
                root / "wyat/benchmark.py",
                root / "benchmarks/analysis/reference.py",
            )
        },
        "versions": {name: version(name) for name in ("mlx", "mlx-lm", "transformers")},
        "device_info": mx.device_info(scorer.device),
        "microbatch_size": options.batch_size,
        "token_cache_size": 0,
        "weight_dtype": "float32",
        "concatenation_exact": True,
        "synchronization": (
            "score_many mx.eval then host tolist; benchmark explicit device fence"
        ),
        "baseline_screen": asdict(baseline_gate),
        "baseline_logits": baseline_values,
        "arms": {},
    }
    arms: dict[str, object] = {}
    payload["arms"] = arms
    for name, qkv, gate_up in (
        ("qkv", True, False),
        ("gate-up", False, True),
        ("both", True, True),
    ):
        candidate = variant(scorer, qkv=qkv, gate_up=gate_up)
        values = score(candidate, requests, options.batch_size)
        gate = quality(cases, saved, values, scorer.model_id)
        arm: dict[str, object] = {
            "screen": asdict(gate),
            "logits": values,
            "timed": False,
            "active_memory_bytes": mx.get_active_memory(),
            "peak_memory_bytes": mx.get_peak_memory(),
        }
        arms[name] = arm
        options.output.write_text(json.dumps(payload, indent=2) + "\n")
        if gate.passed:
            report = benchmark_pair(
                BenchmarkEngine(
                    "unpacked",
                    Model(scorer, scorer.model_id),
                    options.batch_size,
                    synchronize=lambda: mx.synchronize(scorer.device),
                ),
                BenchmarkEngine(
                    name,
                    Model(candidate, scorer.model_id),
                    options.batch_size,
                    synchronize=lambda: mx.synchronize(scorer.device),
                ),
                cases,
                protocol=BenchmarkProtocol(repeats=options.repeats, warmup=1),
            )
            arm["timed"] = True
            arm["paired"] = report.to_dict()
            del report
        options.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "arm": name,
                    "screen_passed": gate.passed,
                    "speed": cast(dict[str, object], arm.get("paired", {})).get(
                        "speed"
                    ),
                }
            ),
            flush=True,
        )
        del candidate
        gc.collect()  # Bound scorer forward methods form cycles; collect between arms.
        mx.clear_cache()


if __name__ == "__main__":
    main()
