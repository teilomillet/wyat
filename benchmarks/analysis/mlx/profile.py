"""Synchronize real Kai stages; report diagnostic overhead and preserved outputs."""

from __future__ import annotations

import hashlib
import json
import time
from argparse import ArgumentParser
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import median
from typing import Protocol, TypeVar, cast

import mlx.core as mx

from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.scaling import amdahl_ceiling
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.benchmark import BenchmarkCycle, DriftReport, QualityGate, compare_cycles
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict
from wyat.wire import decode_json

T = TypeVar("T")


class ArrayMap(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class AttentionMap(Protocol):
    def __call__(self, x: mx.array, mask: str, cache: None = None) -> mx.array: ...


class Block(Protocol):
    input_layernorm: ArrayMap
    self_attn: AttentionMap
    post_attention_layernorm: ArrayMap
    mlp: ArrayMap


@dataclass(frozen=True, slots=True)
class StageSample:
    repeat: int
    batch: int
    layer: int | None
    phase: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class PackedBatch:
    ids: mx.array
    endpoints: mx.array
    queries: mx.array
    encoded: tuple[EncodedRequest, ...]


class Recorder:
    def __init__(self, device: mx.Device, repeat: int, batch: int) -> None:
        self.device = device
        self.repeat = repeat
        self.batch = batch
        self.samples: list[StageSample] = []

    def cpu(self, phase: str, operation: Callable[[], T]) -> T:
        start = time.perf_counter_ns()
        result = operation()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        self.samples.append(StageSample(self.repeat, self.batch, None, phase, elapsed))
        return result

    def gpu(
        self, phase: str, operation: Callable[[], T], layer: int | None = None
    ) -> T:
        # Fences intentionally expose execution ownership. They change graph
        # scheduling and fusion opportunities, so these are diagnostic timings.
        mx.synchronize(self.device)
        start = time.perf_counter_ns()
        result = operation()
        mx.eval(result)
        mx.synchronize(self.device)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        self.samples.append(StageSample(self.repeat, self.batch, layer, phase, elapsed))
        return result


def pack(encoded: tuple[EncodedRequest, ...], pad_id: int) -> PackedBatch:
    length = -(-max(len(row.ids) for row in encoded) // 8) * 8
    width = max(len(row.candidate_positions) for row in encoded)
    ids = mx.array(
        [list(row.ids) + [pad_id] * (length - len(row.ids)) for row in encoded],
        dtype=mx.int32,
    )
    endpoints = mx.array(
        [
            list(row.candidate_positions) + [0] * (width - len(row.candidate_positions))
            for row in encoded
        ],
        dtype=mx.int32,
    )
    queries = mx.array([row.query_position for row in encoded], dtype=mx.int32)
    mx.eval(ids, endpoints, queries)
    return PackedBatch(ids, endpoints, queries, encoded)


def diagnostic_batch(
    scorer: MLXWyatScorer,
    cases: Sequence[EvaluationCase],
    repeat: int,
    batch: int,
) -> tuple[list[list[float]], list[StageSample], dict[str, object]]:
    recorder = Recorder(scorer.device, repeat, batch)
    encoded = recorder.cpu(
        "encoding", lambda: tuple(scorer.encode_request(case.request) for case in cases)
    )
    with mx.stream(scorer.device):
        # PackedBatch contains arrays inside a dataclass, so pack itself
        # evaluates them; the recorder's final fence still measures completion.
        packed = recorder.gpu("packing", lambda: pack(encoded, scorer.encoder.pad_id))
        hidden = recorder.gpu(
            "embedding", lambda: scorer.backbone.embed_tokens(packed.ids)
        )
        blocks = cast(Sequence[Block], scorer.backbone.layers)
        for index, block in enumerate(blocks):
            normalized = recorder.gpu(
                "input_norm", partial(block.input_layernorm, hidden), index
            )
            attention = recorder.gpu(
                "attention", partial(block.self_attn, normalized, "causal"), index
            )
            residual = recorder.gpu(
                "attention_residual", partial(mx.add, hidden, attention), index
            )
            normalized = recorder.gpu(
                "post_attention_norm",
                partial(block.post_attention_layernorm, residual),
                index,
            )
            output = recorder.gpu("mlp", partial(block.mlp, normalized), index)
            hidden = recorder.gpu(
                "mlp_residual", partial(mx.add, residual, output), index
            )
        norm = cast(ArrayMap, scorer.backbone.norm)
        hidden = recorder.gpu("final_norm", lambda: norm(hidden))
        rows = mx.arange(packed.ids.shape[0])
        candidates, query = recorder.gpu(
            "endpoint_gather",
            lambda: (
                hidden[rows[:, None], packed.endpoints],
                hidden[rows, packed.queries],
            ),
        )
        logits = recorder.gpu("head", lambda: scorer.head(candidates, query))
        values = recorder.cpu(
            "materialization", lambda: cast(list[list[float]], logits.tolist())
        )
    scores = [
        row[: len(item.candidate_positions)]
        for row, item in zip(values, encoded, strict=True)
    ]
    shape: dict[str, object] = {
        "batch": batch,
        "rows": len(cases),
        "padded_length": packed.ids.shape[1],
        "useful_tokens": sum(len(row.ids) for row in encoded),
        "padded_tokens": packed.ids.size,
        "candidate_counts": [len(row.candidate_positions) for row in encoded],
        "lengths": [len(row.ids) for row in encoded],
    }
    return scores, recorder.samples, shape


def quality_rows(
    cases: tuple[EvaluationCase, ...],
    logits: Sequence[Sequence[float]],
    model_id: str,
) -> tuple[EvaluationRow, ...]:
    rows: list[EvaluationRow] = []
    for case, values in zip(cases, logits, strict=True):
        prediction = decision_from_logits(case.request, values, model_id)
        rows.append(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                prediction,
                prediction.selected in case.expected,
                0.0,
                None,
            )
        )
    return tuple(rows)


def quality(
    cases: tuple[EvaluationCase, ...],
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
    model_id: str,
) -> DriftReport:
    return compare_cycles(
        [BenchmarkCycle(0.0, quality_rows(cases, reference, model_id), 0, 0)],
        [BenchmarkCycle(0.0, quality_rows(cases, candidate, model_id), 0, 0)],
        QualityGate(),
    )


def operand_bounds(
    scorer: MLXWyatScorer, batches: list[dict[str, object]]
) -> dict[str, object]:
    args = scorer.backbone.args
    hidden, intermediate = args.hidden_size, args.intermediate_size
    query = args.num_attention_heads * args.head_dim
    kv = args.num_key_value_heads * args.head_dim
    operations: list[dict[str, object]] = []
    total_flops = total_bytes = 0
    for shape in batches:
        batch, length = (
            int(cast(int, shape["rows"])),
            int(cast(int, shape["padded_length"])),
        )
        tokens = batch * length
        for name, inner, columns in (
            ("q_proj", hidden, query),
            ("k_proj", hidden, kv),
            ("v_proj", hidden, kv),
            ("o_proj", query, hidden),
            ("gate_proj", hidden, intermediate),
            ("up_proj", hidden, intermediate),
            ("down_proj", intermediate, hidden),
        ):
            flops = 2 * tokens * inner * columns
            operand_bytes = 4 * (tokens * inner + inner * columns + tokens * columns)
            operations.append(
                {
                    "batch": shape["batch"],
                    "operator": name,
                    "per_layer": True,
                    "input": [tokens, inner],
                    "weight": [columns, inner],
                    "output": [tokens, columns],
                    "flops": flops,
                    "operand_bytes_once": operand_bytes,
                    "arithmetic_intensity_flops_per_operand_byte": flops
                    / operand_bytes,
                }
            )
            total_flops += flops * args.num_hidden_layers
            total_bytes += operand_bytes * args.num_hidden_layers
        attention_dense = (
            4 * batch * args.num_attention_heads * length * length * args.head_dim
        )
        attention_causal = (
            2 * batch * args.num_attention_heads * length * (length + 1) * args.head_dim
        )
        attention_bytes = 4 * tokens * (2 * query + 2 * kv)
        operations.append(
            {
                "batch": shape["batch"],
                "operator": "fused_sdpa",
                "per_layer": True,
                "query": [batch, args.num_attention_heads, length, args.head_dim],
                "key_value": [batch, args.num_key_value_heads, length, args.head_dim],
                "dense_flops": attention_dense,
                "causal_useful_flops": attention_causal,
                "operand_bytes_once": attention_bytes,
                "score_matrix_materialized": False,
            }
        )
        total_flops += attention_dense * args.num_hidden_layers
        total_bytes += attention_bytes * args.num_hidden_layers
    return {
        "operators": operations,
        "dense_linear_plus_sdpa_flops": total_flops,
        "operand_bytes_once": total_bytes,
        "bytes_definition": (
            "Each operation reads input and weight once and writes output once; "
            "excludes inter-operation cache reuse; not measured DRAM traffic"
        ),
        "flops_exclude": ["norms", "RoPE", "softmax", "SiLU", "head", "packing"],
        "embedding": {
            "storage_bytes": args.vocab_size * hidden * 4,
            "lookup_does_not_read_entire_table": True,
        },
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ordinary-repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    batch_size = int(options.batch_size)
    ordinary_repeats, profile_repeats = (
        int(options.ordinary_repeats),
        int(options.profile_repeats),
    )
    if min(batch_size, ordinary_repeats, profile_repeats) < 1:
        raise ValueError("batch size and repetitions must be positive")
    root = Path(__file__).resolve().parents[3]
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    scorer = MLXWyatScorer.from_directory(
        root / ".models/Decision-2.0-Kai-0.6B", max_length=512, token_cache_size=0
    )
    frozen = read_reference(
        root / "benchmarks/real/torch-fp32-reference77.json",
        cases,
        scorer,
        dataset_sha256=dataset_hash,
    )
    reference: list[list[float]] = []
    for row in frozen.rows:
        if row.prediction is None:
            raise ValueError("Frozen reference must contain every prediction")
        reference.append(list(row.prediction.scores))
    case_limit = cast(int | None, options.case_limit)
    if case_limit is not None:
        if not 1 <= case_limit <= len(cases):
            raise ValueError("case limit must select an existing nonempty prefix")
        cases, reference = cases[:case_limit], reference[:case_limit]
    requests = tuple(case.request for case in cases)

    def ordinary() -> list[list[float]]:
        result: list[list[float]] = []
        for offset in range(0, len(requests), batch_size):
            result.extend(scorer.score_many(requests[offset : offset + batch_size]))
        return result

    def diagnostic(
        repeat: int,
    ) -> tuple[list[list[float]], list[StageSample], list[dict[str, object]]]:
        result: list[list[float]] = []
        samples: list[StageSample] = []
        shapes: list[dict[str, object]] = []
        for offset in range(0, len(cases), batch_size):
            values, stages, shape = diagnostic_batch(
                scorer,
                cases[offset : offset + batch_size],
                repeat,
                offset // batch_size,
            )
            result.extend(values)
            samples.extend(stages)
            shapes.append(shape)
        return result, samples, shapes

    warmup = ordinary()
    assert quality(cases, reference, warmup, scorer.model_id).passed
    if bool(options.trace_only):
        # This branch supplies a short owned process to Xcode Instruments.
        # Its timings include trace instrumentation and are not a speed claim.
        cycles: list[dict[str, object]] = []
        for repeat in range(ordinary_repeats):
            start = time.perf_counter_ns()
            scores = ordinary()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            cycles.append(
                {
                    "repeat": repeat,
                    "latency_ms": elapsed,
                    "quality": asdict(
                        quality(cases, reference, scores, scorer.model_id)
                    ),
                    "logits": scores,
                }
            )
        payload: dict[str, object] = {
            "scope": "Owned short Metal trace workload; no performance claim",
            "model_id": scorer.model_id,
            "dataset_sha256": dataset_hash,
            "case_ids": [case.id for case in cases],
            "batch_size": batch_size,
            "cycles": cycles,
        }
        output = cast(Path | None, options.output)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps({"trace_cycles": len(cycles), "output": str(output)}), flush=True
        )
        return
    diagnostic(-1)
    ordinary_cycles: list[dict[str, object]] = []
    diagnostic_cycles: list[dict[str, object]] = []
    all_stages: list[StageSample] = []
    shapes: list[dict[str, object]] = []
    for repeat in range(max(ordinary_repeats, profile_repeats)):

        def run_ordinary(repeat: int = repeat) -> None:
            mx.synchronize(scorer.device)
            start = time.perf_counter_ns()
            scores = ordinary()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            drift = quality(cases, reference, scores, scorer.model_id)
            ordinary_cycles.append(
                {
                    "repeat": repeat,
                    "latency_ms": elapsed,
                    "quality": asdict(drift),
                    "passed": drift.passed,
                    "logits": scores,
                }
            )

        def run_diagnostic(repeat: int = repeat) -> None:
            mx.synchronize(scorer.device)
            start = time.perf_counter_ns()
            scores, stages, batch_shapes = diagnostic(repeat)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            drift = quality(cases, reference, scores, scorer.model_id)
            diagnostic_cycles.append(
                {
                    "repeat": repeat,
                    "latency_ms": elapsed,
                    "quality": asdict(drift),
                    "passed": drift.passed,
                    "logits": scores,
                }
            )
            all_stages.extend(stages)
            shapes[:] = batch_shapes

        operations = []
        if repeat < ordinary_repeats:
            operations.append(run_ordinary)
        if repeat < profile_repeats:
            operations.append(run_diagnostic)
        if repeat % 2:
            operations.reverse()
        for operation in operations:
            operation()
    groups: dict[str, list[float]] = {}
    for repeat in range(profile_repeats):
        for stage in (
            "encoding",
            "packing",
            "embedding",
            "input_norm",
            "attention",
            "attention_residual",
            "post_attention_norm",
            "mlp",
            "mlp_residual",
            "final_norm",
            "endpoint_gather",
            "head",
            "materialization",
        ):
            groups.setdefault(stage, []).append(
                sum(
                    sample.latency_ms
                    for sample in all_stages
                    if sample.repeat == repeat and sample.phase == stage
                )
            )
    total_stage_median = sum(median(values) for values in groups.values())
    phases = {
        phase: {
            "cycle_totals_ms": values,
            "median_ms": median(values),
            "diagnostic_fraction": median(values) / total_stage_median,
            "conditional_amdahl_ceiling": amdahl_ceiling(
                median(values) / total_stage_median
            ),
        }
        for phase, values in groups.items()
    }
    ordinary_ms = [float(cast(float, cycle["latency_ms"])) for cycle in ordinary_cycles]
    diagnostic_ms = [
        float(cast(float, cycle["latency_ms"])) for cycle in diagnostic_cycles
    ]
    work = operand_bounds(scorer, shapes)
    payload = {
        "schema_version": 1,
        "model_id": scorer.model_id,
        "dataset_sha256": dataset_hash,
        "scope": (
            f"{len(cases)} real BANKING77 development requests; "
            "no claimed unseen model-quality evaluation"
        ),
        "versions": {"mlx": version("mlx"), "mlx_lm": version("mlx-lm")},
        "device": mx.device_info(scorer.device),
        "batch_size": batch_size,
        "token_cache_size": 0,
        "dtype": "float32",
        "quality_gate": asdict(QualityGate()),
        "tokens_and_endpoints_match_reference": True,
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
        "passed": all(
            cycle["passed"] is True for cycle in ordinary_cycles + diagnostic_cycles
        ),
        "warmup": "one complete ordinary cohort and one diagnostic cohort",
        "ordinary_cycles": ordinary_cycles,
        "diagnostic_cycles": diagnostic_cycles,
        "ordinary_latency_median_ms": median(ordinary_ms),
        "diagnostic_latency_median_ms": median(diagnostic_ms),
        "instrumentation_ratio": median(diagnostic_ms) / median(ordinary_ms),
        "stage_fraction_denominator_ms": total_stage_median,
        "stage_fraction_denominator": "sum of per-phase cycle-total medians",
        "outer_overhead_median_difference_ms": (
            median(diagnostic_ms) - total_stage_median
        ),
        "phases": phases,
        "raw_stage_samples": [asdict(sample) for sample in all_stages],
        "batch_shapes": shapes,
        "work": work,
        "derived_dense_flops_per_second": cast(
            int, work["dense_linear_plus_sdpa_flops"]
        )
        / (median(ordinary_ms) / 1000),
        "metal_active_memory_bytes": mx.get_active_memory(),
        "metal_peak_memory_bytes": mx.get_peak_memory(),
        "limitations": [
            "Per-stage fences alter graph scheduling and fusion; "
            "Amdahl fractions are diagnostic conditional bounds",
            "Operand bytes are algorithmic bounds; "
            "external memory bandwidth was not measured",
            "One model and frozen cohort; "
            "deployment traffic behavior is not established",
        ],
    }
    output = (
        cast(Path | None, options.output)
        or root / "benchmarks/analysis/results/mlx-profile77-b8.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "ordinary_median_ms": median(ordinary_ms),
                "diagnostic_median_ms": median(diagnostic_ms),
                "phases": phases,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
