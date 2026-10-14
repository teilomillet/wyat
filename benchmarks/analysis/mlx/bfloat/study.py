"""Declared three-arm screen; timing is admitted only after full-family screens."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlx.core as mx
from mlx.nn.layers.linear import Linear
from mlx_lm.models import activations, qwen3

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.mlx.bfloat.products import ARMS, Arm, candidate_backbone
from benchmarks.analysis.mlx.bfloat.run import (
    DATASET_SHA256,
    MODEL_ID,
    REFERENCE_SHA256,
    validate_prior,
)
from benchmarks.analysis.mlx.endpoints import (
    bounded_cache,
    memory,
    paired_with_telemetry,
)
from benchmarks.analysis.reference import read_reference
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.core import Model
from wyat.evaluation import case_from_dict, percentile
from wyat.wire import decode_json


def study(
    output: Path, *, arm: str | None = None, prior_screen: Path | None = None
) -> int:
    if arm not in (None, *ARMS) or ((arm is None) != (prior_screen is None)):
        raise ValueError("Timing requires one declared survivor and family screen")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("Worker must start with TF32 disabled")
    if output.exists():
        raise FileExistsError("Preserve existing BF16 receipt")
    root = Path(__file__).resolve().parents[4]
    payload: dict[str, object] = {
        "completed": False,
        "passed": False,
        "performance_gate_passed": False,
        "phase": "screen" if arm is None else "confirmation",
        "arms": {},
        "tf32": "0",
        "cache_cap_bytes": 512 << 20,
        "token_cache_size": 0,
        "compile": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    try:
        files = {
            name: root / path
            for name, path in {
                "worker": "benchmarks/analysis/mlx/bfloat/study.py",
                "mechanics": "benchmarks/analysis/mlx/bfloat/products.py",
                "package": "benchmarks/analysis/mlx/bfloat/__init__.py",
                "launcher": "benchmarks/analysis/mlx/bfloat/run.py",
                "declaration": "docs/analysis/bfloat.md",
                "tests": "tests/analysis/test_mlx_bfloat.py",
                "protocol_tests": "tests/analysis/test_bfloat_protocol.py",
                "telemetry": "benchmarks/analysis/mlx/endpoints.py",
                "profile": "benchmarks/analysis/mlx/profile.py",
                "reference": "benchmarks/analysis/reference.py",
                "cache": "benchmarks/analysis/mlx/cache.py",
                "evidence": "benchmarks/analysis/evidence.py",
                "mlx_backend": "wyat/backends/wyat/mlx.py",
                "artifact": "wyat/backends/wyat/artifact.py",
                "encoding": "wyat/backends/wyat/encoding.py",
                "core": "wyat/core.py",
                "benchmark": "wyat/benchmark.py",
                "evaluation": "wyat/evaluation.py",
            }.items()
        }
        files.update(
            maintained_linear=Path(inspect.getfile(Linear)),
            maintained_qwen3=Path(inspect.getfile(qwen3)),
            maintained_swiglu=Path(inspect.getfile(activations)),
        )
        frozen = freeze_sources(files, output)
        payload.update(source_hashes=frozen.hashes(), source_archive=frozen.archive())
        selected_arm = cast(Arm, arm)
        dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        dataset_bytes = dataset.read_bytes()
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
        if dataset_hash != DATASET_SHA256:
            raise ValueError("Frozen cohort changed")
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in dataset_bytes.decode().splitlines()
        )
        if len(cases) != 77:
            raise ValueError("Full77 cohort required")
        reference_path = root / "benchmarks/real/torch-fp32-reference77.json"
        payload.update(
            dataset_sha256=dataset_hash,
            reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        )
        if payload["reference_sha256"] != REFERENCE_SHA256:
            raise ValueError("Frozen CPU reference changed")
        if arm is not None and prior_screen is not None:
            payload["prior_screen_sha256"] = validate_prior(
                prior_screen,
                selected_arm,
                frozen.hashes(),
                tuple(case.id for case in cases),
            )
        arm_results: dict[str, dict[str, object]] = {}
        with bounded_cache(512 << 20) as previous:
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
            if base.model_id != MODEL_ID:
                raise ValueError("Pinned model identity changed")
            payload.update(
                model_id=base.model_id,
                original_cache_cap_bytes=previous,
                versions={
                    name: version(name) for name in ("mlx", "mlx-lm", "transformers")
                },
                device=mx.device_info(base.device),
            )
            baseline_cycles: dict[int, BenchmarkCycle] = {}
            for batch in (8, 1):
                engine = BenchmarkEngine(
                    "fp32-full-prefill", Model(base, base.model_id), batch
                )
                baseline_cycles[batch] = _run(engine, cases)
            baseline_gate = compare_cycles(
                (reference,) * 2, tuple(baseline_cycles.values()), QualityGate()
            )
            payload.update(
                baseline_screens={
                    str(batch): cycle.to_dict()
                    for batch, cycle in baseline_cycles.items()
                },
                baseline_quality=asdict(baseline_gate),
            )
            if not baseline_gate.passed:
                raise ValueError("FP32 control failed frozen CPU gates")
            for name in ARMS if arm is None else (selected_arm,):
                outcome: dict[str, object] = {"passed": False, "screens": {}}
                arm_results[name] = outcome
                mx.synchronize(base.device)
                before = memory()
                started = time.perf_counter_ns()
                with mx.stream(base.device):
                    backbone = candidate_backbone(base.backbone, name)
                    candidate = MLXWyatScorer(
                        backbone,
                        base.head,
                        base.encoder,
                        model_id=base.model_id,
                        device=base.device,
                    )
                    mx.eval(backbone.parameters())
                mx.synchronize(base.device)
                outcome.update(
                    preparation_ms=(time.perf_counter_ns() - started) / 1e6,
                    memory_before_preparation=before,
                    memory_after_preparation=memory(),
                )
                screens: dict[int, BenchmarkCycle] = {}
                for batch in (8, 1):
                    engine = BenchmarkEngine(
                        name, Model(candidate, base.model_id), batch
                    )
                    screens[batch] = _run(engine, cases)
                    outcome["screens"] = {
                        str(size): cycle.to_dict() for size, cycle in screens.items()
                    }
                    payload["arms"] = arm_results
                    output.write_text(
                        json.dumps(payload, indent=2, allow_nan=False) + "\n"
                    )
                gate = compare_cycles(
                    (reference,) * 2, tuple(screens.values()), QualityGate()
                )
                outcome.update(
                    passed=gate.passed,
                    cpu_quality=asdict(gate),
                    exact_bfloat_parameters=True,
                )
                if arm is not None and gate.passed:
                    confirmations: dict[str, object] = {}
                    all_passed = True
                    for batch in (8, 1):
                        left = BenchmarkEngine(
                            "fp32-full-prefill", Model(base, base.model_id), batch
                        )
                        right = BenchmarkEngine(
                            name, Model(candidate, base.model_id), batch
                        )
                        report, telemetry = paired_with_telemetry(
                            left,
                            right,
                            cases,
                            incremental=output.with_suffix(f".b{batch}.cycles.jsonl"),
                        )
                        cycles = (
                            report.baseline_warmup_cycles
                            + report.candidate_warmup_cycles
                            + report.baseline_cycles
                            + report.candidate_cycles
                        )
                        cpu_gate = compare_cycles(
                            (reference,) * len(cycles), cycles, QualityGate()
                        )
                        ratios = tuple(
                            left.latency_ms / right.latency_ms
                            for left, right in zip(
                                report.baseline_cycles,
                                report.candidate_cycles,
                                strict=True,
                            )
                        )
                        passed = (
                            report.passed
                            and cpu_gate.passed
                            and percentile(ratios, 0.5) >= 1.02
                            and percentile(ratios, 0.05) > 1
                        )
                        confirmations[str(batch)] = {
                            "paired": report.to_dict(),
                            "telemetry": telemetry,
                            "all_cycles_cpu_quality": asdict(cpu_gate),
                            "performance_gate_passed": passed,
                        }
                        all_passed = all_passed and passed
                        outcome["confirmations"] = confirmations
                        output.write_text(
                            json.dumps(payload, indent=2, allow_nan=False) + "\n"
                        )
                        del left, right, report
                    payload["performance_gate_passed"] = all_passed
                del engine, candidate, backbone
                gc.collect()
                mx.synchronize(base.device)
                mx.clear_cache()
                output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            payload.update(
                completed=True,
                passed=any(result["passed"] is True for result in arm_results.values()),
            )
    except Exception as error:
        payload.update(
            completed=False,
            passed=False,
            performance_gate_passed=False,
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    accepted = payload["passed"] if arm is None else payload["performance_gate_passed"]
    return 0 if payload["completed"] and accepted else 1
