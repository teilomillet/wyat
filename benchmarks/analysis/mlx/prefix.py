"""Reuse one exact frozen token prefix through maintained Qwen3 cache state."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import mlx.core as mx
from mlx.nn.layers.positional_encoding import RoPE
from mlx_lm.models import base as maintained_base
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Qwen3Model

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.mlx.cache import DATASET_SHA256, memory
from benchmarks.analysis.mlx.endpoints import bounded_cache, paired_with_telemetry
from benchmarks.analysis.reference import read_reference
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.backends.wyat.prefix import PrefixCache
from wyat.backends.wyat.prefix import PrefixScorer as ProductionPrefixScorer
from wyat.benchmark import BenchmarkEngine, QualityGate, _run, compare_cycles
from wyat.core import Model
from wyat.evaluation import case_from_dict, percentile
from wyat.wire import decode_json


class PrefixScorer(ProductionPrefixScorer):
    """Pure prefix ablation: retain the maintained full cached final block."""

    def _logits(
        self,
        ids: mx.array,
        endpoints: mx.array,
        query_positions: mx.array,
        caches: list[PrefixCache],
    ) -> mx.array:
        hidden = self._forward(ids, caches)
        rows = mx.arange(ids.shape[0])
        return self.base.head(
            hidden[rows[:, None], endpoints], hidden[rows, query_positions]
        )


def study(
    output: Path,
    *,
    batch_size: int = 8,
    candidate_type: type[ProductionPrefixScorer] = PrefixScorer,
    strategy: Literal["prefix25", "prefix25-endpoints"] = "prefix25",
) -> dict[str, object]:
    """One frozen25-token resource; fixed512MiB allocator cap for both arms."""
    root = Path(__file__).resolve().parents[3]
    payload: dict[str, object] = {
        "completed": False,
        "passed": False,
        "performance_gate_passed": False,
        "strategy": strategy,
        "candidate_type": candidate_type.__qualname__,
        "started_wall_time_ns": time.time_ns(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload) + "\n")
    try:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be positive")
        if strategy not in ("prefix25", "prefix25-endpoints"):
            raise ValueError("Unknown experiment strategy")
        sources = freeze_sources(
            {
                name: root / name
                for name in (
                    "benchmarks/analysis/mlx/prefix.py",
                    "benchmarks/analysis/mlx/combined.py",
                    "benchmarks/analysis/mlx/endpoints.py",
                    "benchmarks/analysis/mlx/cache.py",
                    "benchmarks/analysis/mlx/profile.py",
                    "benchmarks/analysis/reference.py",
                    "benchmarks/analysis/evidence.py",
                    "wyat/backends/wyat/mlx.py",
                    "wyat/backends/wyat/prefix.py",
                    "wyat/backends/wyat/readout.py",
                    "wyat/backends/wyat/encoding.py",
                    "wyat/benchmark.py",
                    "wyat/core.py",
                    "wyat/evaluation.py",
                )
            }
            | {
                "maintained_cache": Path(inspect.getfile(KVCache)),
                "maintained_qwen3": Path(inspect.getfile(Qwen3Model)),
                "maintained_attention": Path(inspect.getfile(maintained_base)),
                "maintained_rope": Path(inspect.getfile(RoPE)),
                "candidate_strategy": Path(inspect.getfile(candidate_type)),
            },
            output,
        )
        payload.update(
            sources_sha256=sources.hashes(), sources_archive=sources.archive()
        )
        dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        dataset_bytes = dataset.read_bytes()
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        if dataset_hash != DATASET_SHA256:
            raise ValueError("Frozen development cohort hash changed")
        cases = tuple(
            case_from_dict(decode_json(row))
            for row in dataset_bytes.decode("utf-8").splitlines()
        )
        if len(cases) != 77:
            raise ValueError("Experiment requires all77 frozen cases")
        reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
        reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        with bounded_cache(512 * 1024**2) as previous_cap:
            base = MLXWyatScorer.from_directory(
                root / ".models/Decision-2.0-Kai-0.6B",
                max_length=512,
                token_cache_size=0,
                device="gpu",
                compile=False,
            )
            reference = read_reference(
                reference_path, cases, base, dataset_sha256=dataset_hash
            )
            encoded = tuple(base.encode_request(case.request) for case in cases)
            prefix = encoded[0].ids[:25]
            before = memory()
            build_start = time.perf_counter_ns()
            candidate = candidate_type(base, prefix)
            mx.synchronize(base.device)
            build_ms = (time.perf_counter_ns() - build_start) / 1e6
            after = memory()
            if len(prefix) != 25 or not all(candidate._admits(row) for row in encoded):
                raise ValueError(
                    "Not every frozen case admits the declared25-token prefix"
                )
            baseline_engine = BenchmarkEngine(
                "full-prefill", Model(base, base.model_id), batch_size
            )
            candidate_engine = BenchmarkEngine(
                strategy, Model(candidate, base.model_id), batch_size
            )
            screens = (_run(baseline_engine, cases), _run(candidate_engine, cases))
            screen_gate = compare_cycles((reference,) * 2, screens, QualityGate())
            payload.update(
                scope=f"real full77 training-source {strategy} experiment",
                model_id=base.model_id,
                dataset_sha256=dataset_hash,
                reference_sha256=reference_hash,
                versions={
                    name: version(name) for name in ("mlx", "mlx-lm", "transformers")
                },
                device=mx.device_info(base.device),
                dtype="float32",
                compile=False,
                batch_size=batch_size,
                token_cache_size=0,
                cache_cap_bytes=512 * 1024**2,
                original_cache_cap_bytes=previous_cap,
                prefix_tokens=list(prefix),
                prefix_length=25,
                prefix_logical_state_bytes=candidate.prefix_state_bytes,
                prefix_build_ms=build_ms,
                memory_before_prefix_build=before,
                memory_after_prefix_build=after,
                token_lengths=[len(row.ids) for row in encoded],
                screen_cycles=[cycle.to_dict() for cycle in screens],
                screen_quality=asdict(screen_gate),
                hypothesis=(
                    "Exact prefix reuse preserves strict quality "
                    "and improves full cohort latency"
                    if strategy == "prefix25"
                    else "Combined prefix and endpoint reuse preserves strict "
                    "quality and improves full cohort latency; separate gains "
                    "are not multiplied"
                ),
                promotion_gate=(
                    "median>=1.02 and paired_p05>1 plus strict quality "
                    "and scalar confirmation"
                ),
                lifecycle=(
                    "immutable fixed-weight prefix states; fresh perbatch "
                    "KVCache owners; no cross-request suffix state"
                ),
                limits=[
                    "Prefix build excluded from warm/measured timing "
                    "and reported separately.",
                    "Prefix_state_bytes is logical; array views may retain "
                    "backing allocation.",
                    "One initial warm each; continuous512MiB allocator history, "
                    "no perarm clearing.",
                    "Any mismatch or overlapping readout falls back whole batch; "
                    "duplicate encoding is charged.",
                    "Original rounded fullwidth preserved; maintained causal "
                    "mask/RoPE offsets; true readouts translated once.",
                    "Same77 examples repeated; released-model exposure unknown, "
                    "no held-out quality claim.",
                    "Source snapshots attest setup-time disk bytes, "
                    "not imported/external execution.",
                ],
            )
            output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            if screen_gate.passed:
                report, telemetry = paired_with_telemetry(
                    baseline_engine,
                    candidate_engine,
                    cases,
                    incremental=output.with_suffix(".cycles.jsonl"),
                )
                cycles = (
                    report.baseline_warmup_cycles
                    + report.candidate_warmup_cycles
                    + report.baseline_cycles
                    + report.candidate_cycles
                )
                gate = compare_cycles((reference,) * len(cycles), cycles, QualityGate())
                speedups = tuple(
                    left.latency_ms / right.latency_ms
                    for left, right in zip(
                        report.baseline_cycles, report.candidate_cycles, strict=True
                    )
                )
                payload.update(
                    paired=report.to_dict(),
                    untimed_telemetry=telemetry,
                    all_cycles_cpu_quality=asdict(gate),
                    passed=report.passed and gate.passed,
                    performance_gate_passed=(
                        report.passed
                        and gate.passed
                        and percentile(speedups, 0.5) >= 1.02
                        and percentile(speedups, 0.05) > 1
                    ),
                )
            payload["completed"] = True
    except BaseException as error:
        payload.update(
            completed=False,
            passed=False,
            performance_gate_passed=False,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


class Arguments(Namespace):
    batch_size: int
    output: Path


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analysis/results/mlx-prefix77-b8.json"),
    )
    options = parser.parse_args(namespace=Arguments())
    receipt = study(options.output, batch_size=options.batch_size)
    if receipt["completed"] is not True or receipt["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
