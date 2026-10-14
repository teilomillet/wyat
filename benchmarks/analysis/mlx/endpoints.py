"""Experimental final-block endpoint selection with original causal positions.

First layers remain maintained upstream blocks. The last attention recipe follows
MLX-LM 0.31.3 Qwen3, but query/output/MLP work visits only requested endpoints.
Every K/V token remains; selected queries require an explicit original-position
mask rather than the lower-right aligned short-query causal shortcut.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from mlx.nn.layers.positional_encoding import RoPE
from mlx_lm.models.qwen3 import Qwen3Model, TransformerBlock

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.mlx.profile import ArrayMap, quality
from benchmarks.analysis.reference import read_reference
from wyat import Model, Request
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.backends.wyat.readout import RotaryMap as RotaryMap
from wyat.backends.wyat.readout import endpoint_last_block
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    _run,
    compare_cycles,
)
from wyat.evaluation import EvaluationCase, _input_digest, case_from_dict
from wyat.wire import decode_json


class BlockMap(Protocol):
    def __call__(self, x: mx.array, mask: str, cache: None = None) -> mx.array: ...


def endpoint_hidden(
    backbone: Qwen3Model, ids: mx.array, positions: mx.array
) -> mx.array:
    if backbone.args.rope_scaling is not None:
        raise ValueError("Endpoint experiment requires unscaled native RoPE")
    hidden = backbone.embed_tokens(ids)
    blocks = cast(Sequence[BlockMap], backbone.layers[:-1])
    for block in blocks:
        hidden = block(hidden, "causal")
    hidden = endpoint_last_block(backbone.layers[-1], hidden, positions)
    normalize = cast(ArrayMap, backbone.norm)
    return normalize(hidden)


class EndpointOnly(MLXWyatScorer):
    def _logits(
        self, ids: mx.array, endpoints: mx.array, query_positions: mx.array
    ) -> mx.array:
        positions = mx.concatenate([endpoints, query_positions[:, None]], axis=1)
        hidden = endpoint_hidden(self.backbone, ids, positions)
        return self.head(hidden[:, :-1], hidden[:, -1])


def score(
    scorer: MLXWyatScorer, requests: Sequence[Request], batch_size: int
) -> list[list[float]]:
    values: list[list[float]] = []
    for start in range(0, len(requests), batch_size):
        values.extend(scorer.score_many(requests[start : start + batch_size]))
    return values


def memory() -> dict[str, int]:
    return {
        "active": mx.get_active_memory(),
        "cache": mx.get_cache_memory(),
        "peak_active": mx.get_peak_memory(),
        "active_plus_cache": mx.get_active_memory() + mx.get_cache_memory(),
    }


@contextmanager
def bounded_cache(limit: int) -> Iterator[int]:
    """Fresh experiment owns the process-wide cap; restore it on every exit."""
    previous = mx.set_cache_limit(limit)
    try:
        mx.clear_cache()
        yield previous
    finally:
        try:
            mx.synchronize()
        finally:
            mx.set_cache_limit(previous)
            mx.clear_cache()


def paired_with_telemetry(
    baseline: BenchmarkEngine,
    candidate: BenchmarkEngine,
    cases: tuple[EvaluationCase, ...],
    *,
    incremental: Path,
) -> tuple[BenchmarkReport, list[dict[str, object]]]:
    """Same canonical timing/alternation; memory/process reads stay outside it."""
    plan = BenchmarkProtocol(repeats=5, warmup=1)
    incremental.write_text("")
    left_warmup, right_warmup = (_run(baseline, cases),), (_run(candidate, cases),)
    left: list[BenchmarkCycle] = []
    right: list[BenchmarkCycle] = []
    telemetry: list[dict[str, object]] = []
    for repeat in range(plan.repeats):
        arms = [(baseline, left), (candidate, right)]
        if repeat % 2:
            arms.reverse()
        for engine, cycles in arms:
            mx.synchronize()
            mx.reset_peak_memory()
            before = process_diagnostic(device_memory_bytes=memory())
            cycle = _run(engine, cases)
            after = process_diagnostic(device_memory_bytes=memory())
            cycles.append(cycle)
            observation: dict[str, object] = {
                "repeat": repeat,
                "arm": engine.name,
                "before": asdict(before),
                "after": asdict(after),
            }
            telemetry.append(observation)
            with incremental.open("a") as stream:
                json.dump(
                    {**observation, "cycle": cycle.to_dict()}, stream, allow_nan=False
                )
                stream.write("\n")
                stream.flush()
    report = BenchmarkReport(
        _input_digest(cases),
        cases,
        plan,
        baseline,
        candidate,
        tuple(left),
        tuple(right),
        compare_cycles(left, right, plan.quality_gate),
        left_warmup,
        right_warmup,
    )
    return report, telemetry


class Arguments(Namespace):
    batch_size: int
    output: Path


def study(output: Path, *, batch_size: int = 8) -> dict[str, object]:
    payload: dict[str, object] = {
        "completed": False,
        "passed": False,
        "started_wall_time_ns": time.time_ns(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be positive")
        root = Path(__file__).resolve().parents[3]
        frozen_sources = freeze_sources(
            {
                name: root / name
                for name in (
                    "benchmarks/analysis/mlx/endpoints.py",
                    "benchmarks/analysis/mlx/profile.py",
                    "benchmarks/analysis/reference.py",
                    "benchmarks/analysis/evidence.py",
                    "wyat/backends/wyat/mlx.py",
                    "wyat/backends/wyat/readout.py",
                    "wyat/backends/wyat/encoding.py",
                    "wyat/benchmark.py",
                )
            }
            | {
                "upstream_qwen3": Path(inspect.getfile(TransformerBlock)),
                "upstream_rope": Path(inspect.getfile(RoPE)),
            },
            output,
        )
        dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        cases: tuple[EvaluationCase, ...] = tuple(
            case_from_dict(decode_json(row)) for row in dataset.read_text().splitlines()
        )
        dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
        reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        payload.update(
            {
                "dataset_sha256": dataset_hash,
                "reference_sha256": reference_hash,
                "sources_archive": frozen_sources.archive(),
                "sources_sha256": frozen_sources.hashes(),
            }
        )
        output.write_text(json.dumps(payload, indent=2) + "\n")
        cache_cap = 512 * 1024**2
        with bounded_cache(cache_cap) as previous_cap:
            baseline = MLXWyatScorer.from_directory(
                root / ".models/Decision-2.0-Kai-0.6B",
                max_length=512,
                token_cache_size=0,
                device="gpu",
            )
            frozen = read_reference(
                root / "benchmarks/real/torch-fp32-reference77.json",
                cases,
                baseline,
                dataset_sha256=dataset_hash,
            )
            saved: list[list[float]] = []
            for row in frozen.rows:
                if row.prediction is None:
                    raise ValueError("Frozen reference must contain every prediction")
                saved.append(list(row.prediction.scores))
            candidate = EndpointOnly(
                baseline.backbone,
                baseline.head,
                baseline.encoder,
                model_id=baseline.model_id,
                device=baseline.device,
            )
            requests = tuple(case.request for case in cases)
            reference_values = score(baseline, requests, batch_size)
            values = score(candidate, requests, batch_size)
            baseline_gate = quality(cases, saved, reference_values, baseline.model_id)
            gate = quality(cases, saved, values, baseline.model_id)
            payload.update(
                {
                    "scope": (
                        "77-case training-source runtime assay; "
                        "checkpoint exposure unknown"
                    ),
                    "hypothesis": (
                        "Endpoint-only final block preserves strict quality "
                        "and improves complete latency"
                    ),
                    "conditional_whole_final_block_ceiling": 1.036,
                    "model_id": baseline.model_id,
                    "dataset_sha256": dataset_hash,
                    "case_ids": [case.id for case in cases],
                    "sources_sha256": frozen_sources.hashes(),
                    "sources_archive": frozen_sources.archive(),
                    "versions": {
                        name: version(name)
                        for name in ("mlx", "mlx-lm", "transformers")
                    },
                    "batch_size": batch_size,
                    "token_cache_size": 0,
                    "dtype": "float32",
                    "device": mx.device_info(baseline.device),
                    "baseline_screen": asdict(baseline_gate),
                    "candidate_screen": asdict(gate),
                    "baseline_logits": reference_values,
                    "candidate_logits": values,
                    "memory_guideline_bytes": mx.get_memory_limit(),
                    "allocator_cache_cap_bytes": cache_cap,
                    "previous_allocator_cache_cap_bytes": previous_cap,
                    "cache_history": (
                        "cap before load; clear once before load; "
                        "screen and warm both arms; "
                        "no per-arm clearing"
                    ),
                    "promotion_gate": (
                        "median>=1.02 and paired_p05>1 plus scalar confirmation "
                        "and strict quality"
                    ),
                    "synchronization": (
                        "host-materialized native outputs plus "
                        "explicit benchmark fences"
                    ),
                    "diagnostic_before_pairs": asdict(
                        process_diagnostic(device_memory_bytes=memory())
                    ),
                    "timed": False,
                    "passed": False,
                    "completed": False,
                    "performance_gate_passed": False,
                }
            )
            output.write_text(json.dumps(payload, indent=2) + "\n")
            if baseline_gate.passed and gate.passed:
                report, telemetry = paired_with_telemetry(
                    BenchmarkEngine(
                        "whole-final-block",
                        Model(baseline, baseline.model_id),
                        batch_size,
                        synchronize=lambda: mx.synchronize(baseline.device),
                    ),
                    BenchmarkEngine(
                        "endpoint-final-block",
                        Model(candidate, candidate.model_id),
                        batch_size,
                        synchronize=lambda: mx.synchronize(baseline.device),
                    ),
                    cases,
                    incremental=output.with_suffix(".cycles.jsonl"),
                )
                payload["timed"] = True
                payload["paired"] = report.to_dict()
                payload["paired_untimed_telemetry"] = telemetry
                payload["passed"] = report.passed
                speed = cast(dict[str, object], report.to_dict()["speed"])
                payload["performance_gate_passed"] = (
                    report.passed
                    and cast(float, speed["paired_speedup_p50"]) >= 1.02
                    and cast(float, speed["paired_speedup_p05"]) > 1
                )
            payload["completed"] = True
            payload["diagnostic_after_pairs"] = asdict(
                process_diagnostic(device_memory_bytes=memory())
            )
            output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "screen": asdict(gate),
                        "paired_speed": cast(
                            dict[str, object], payload.get("paired", {})
                        ).get("speed"),
                    }
                ),
                flush=True,
            )

    except BaseException as error:
        payload["passed"] = False
        payload["completed"] = False
        payload["error"] = f"{type(error).__name__}: {error}"
        output.write_text(json.dumps(payload, indent=2) + "\n")
        raise
    return payload


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "benchmarks/analysis/results/mlx-endpoints77-b8.json",
    )
    options = parser.parse_args(namespace=Arguments())
    receipt = study(options.output, batch_size=options.batch_size)
    if receipt.get("passed") is not True or receipt.get("completed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
