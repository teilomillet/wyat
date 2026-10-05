"""Frozen full77 confirmation of exact CPU prefix reuse, separately B8 and B1."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from transformers import cache_utils, masking_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.torch.threads import DATASET_SHA256
from wyat import Model
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict, percentile
from wyat.wire import decode_json, read_object

from .cache import PrefixScorer
from .study import ROOT

# Exact independent first-eight audit; never accept an edited historical pass.
HISTORY = {
    "json": "532fc2d2ae6daf02e10ab872e9091b41cd52d4fb23043d0f144a748797e030e5",
    "cycles.jsonl": "12adcc9373e88bf10a77b7b24d7cd357225370a400f31ff16be0c8bb7b836413",
    "wall.json": "ba18c04ec715f5a96082b96caeb1a3b14bf068a93cd864e1caa9e08fceef26c6",
    "sources-d6910c77d1e9d422.json": (
        "d6910c77d1e9d422be86cbe7ec17d25fa0089ac80dd80f9c8b2875b5aee80e53"
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_history(root: Path = ROOT) -> dict[str, object]:
    """Pin historical execution bytes; a new run freezes its own current source."""
    stem = root / "benchmarks/analysis/results/torch-prefix8"
    for suffix, expected in HISTORY.items():
        if digest(Path(str(stem) + "." + suffix)) != expected:
            raise ValueError("Frozen first-eight history changed")
    history = read_object(stem.with_suffix(".json"))
    archive = read_object(Path(str(stem) + ".sources-d6910c77d1e9d422.json"))
    sources = cast(list[dict[str, object]], archive["files"])
    if (
        len(sources) != len(cast(dict[str, str], history["source_sha256"]))
        or {cast(str, source["name"]): source["sha256"] for source in sources}
        != history["source_sha256"]
    ):
        raise ValueError("Archived first-eight source inventory changed")
    for source in sources:
        if (
            hashlib.sha256(cast(str, source["text"]).encode()).hexdigest()
            != source["sha256"]
        ):
            raise ValueError("Archived first-eight source identity changed")
    dataset = root / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    reference = root / "benchmarks/real/torch-fp32-reference77.json"
    cases = tuple(
        case_from_dict(decode_json(line)) for line in dataset.read_text().splitlines()
    )
    cycles = cast(list[dict[str, object]], history["cycles"])
    if (
        len(cases) != 77
        or digest(dataset) != history["dataset_sha256"]
        or digest(reference) != history["reference_sha256"]
        or history["input_sha256"] != _input_digest(cases[:8])
        or history["case_ids"] != [case.id for case in cases[:8]]
        or history["completed"] is not True
        or history["screen_passed"] is not True
        or len(cycles) != 8
        or any(
            len(cast(list[object], cast(dict[str, object], row["cycle"])["rows"])) != 8
            or row["passed"] is not True
            for row in cycles
        )
    ):
        raise ValueError("First-eight admission history is incomplete or mismatched")
    return {
        "receipt_sha256": HISTORY["json"],
        "all_history_sha256": HISTORY,
        "model_id": history["model_id"],
        "prefix_tokens": history["prefix_tokens"],
        "dataset_sha256": history["dataset_sha256"],
        "reference_sha256": history["reference_sha256"],
        "raw_outputs": 64,
    }


def study(model: Path, output: Path, *, batch_size: int) -> bool:
    if type(batch_size) is not int or batch_size not in (1, 8):
        raise ValueError("CPU prefix confirmation requires batch size 1 or 8")
    gate = QualityGate()
    plan = BenchmarkProtocol(repeats=5, warmup=1, quality_gate=gate)
    payload: dict[str, object] = {
        "scope": "fixed full77 training-source development; exposure unknown",
        "completed": False,
        "quality_passed": False,
        "performance_passed": False,
        "accepted": False,
        "phase": "admission",
        "batch_size": batch_size,
        "quality_gate": asdict(gate),
        "performance_gate": {
            "paired_median_minimum": 1.02,
            "paired_p05_minimum_exclusive": 1.0,
            "aggregate_throughput_ratio_minimum_exclusive": 1.0,
            "candidate_cohort_p95_maximum_ratio": 1.0,
            "declaration": "CPU confirmation strengthened before full77 results",
        },
        "protocol": {
            "dtype": "float32",
            "device": "cpu",
            "threads": 8,
            "max_length": 512,
            "token_cache_size": 0,
            "prefix_length": 25,
            "warmup": 1,
            "pairs": 5,
            "order": "alternating baseline/candidate",
            "context": "native Mac, unchanged HF flags",
            "full_final_block": True,
            "timing": "prefix build excluded; all fresh cache copying inside inference",
            "promotion": "both B8 and separate B1 must pass; this receipt is one shape",
            "memory": "shared cumulative RSS high-water, not per-engine peak",
        },
        "cycles": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    previous_threads = torch.get_num_threads()
    rows: list[dict[str, object]] = []
    try:
        write()
        payload["first8_admission"] = history = admit_history()
        sources = freeze_sources(
            {
                "pairs": Path(__file__),
                "candidate": Path(__file__).with_name("cache.py"),
                "runner": Path(__file__).with_name("run.py"),
                "screen": Path(__file__).with_name("study.py"),
                "launcher": ROOT / "benchmarks/analysis/torch/process.py",
                "reference": ROOT / "benchmarks/analysis/reference.py",
                "evidence": ROOT / "benchmarks/analysis/evidence.py",
                "benchmark": ROOT / "wyat/benchmark.py",
                "core": ROOT / "wyat/core.py",
                "evaluation": ROOT / "wyat/evaluation.py",
                "artifact": ROOT / "wyat/backends/kai/artifact.py",
                "encoding": ROOT / "wyat/backends/kai/encoding.py",
                "backend": ROOT / "wyat/backends/kai/torch.py",
                "qwen3": Path(modeling_qwen3.__file__),
                "cache": Path(cache_utils.__file__),
                "mask": Path(masking_utils.__file__),
                "sdpa": Path(sdpa_attention.__file__),
            },
            output,
        )
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in (ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl")
            .read_text()
            .splitlines()
        )
        flags = {
            name: vars(sdpa_attention)[name]
            for name in (
                "_is_torch_mps_available",
                "_is_torch_greater_or_equal_than_2_13",
            )
        }
        payload.update(
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            sdpa_flags=flags,
            torch_mps_built=torch.backends.mps.is_built(),
            torch_mps_available=torch.backends.mps.is_available(),
            torch_build=torch.__config__.show(),
            input_sha256=_input_digest(cases),
            case_ids=[case.id for case in cases],
            interop_threads=torch.get_num_interop_threads(),
        )
        if not all(value is True for value in flags.values()):
            raise ValueError("Native masked-GQA flags differ from admitted screen")
        torch.set_num_threads(8)
        payload["phase"] = "loading"
        write()
        base = KaiScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != history["model_id"]:
            raise ValueError("Model identity differs from admitted first-eight screen")
        reference = read_reference(
            ROOT / "benchmarks/real/torch-fp32-reference77.json",
            cases,
            base,
            dataset_sha256=DATASET_SHA256,
        )
        encoded = tuple(base.encode_request(case.request) for case in cases)
        prefix = tuple(cast(list[int], history["prefix_tokens"]))
        before = process_diagnostic()
        started = time.perf_counter_ns()
        candidate = PrefixScorer(base, prefix)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        if len(prefix) != 25 or not all(candidate.admits(row) for row in encoded):
            raise ValueError(
                "Not every frozen full77 case admits original25-token prefix"
            )
        payload.update(
            model_id=base.model_id,
            dataset_sha256=history["dataset_sha256"],
            reference_sha256=history["reference_sha256"],
            prefix_tokens=list(prefix),
            prefix_build_ms=build_ms,
            prefix_state_bytes=candidate.prefix_state_bytes,
            token_lengths=[len(row.ids) for row in encoded],
            phase="cycles",
        )
        engines = (
            BenchmarkEngine("cpu-full-prefill", Model(base, base.model_id), batch_size),
            BenchmarkEngine(
                "cpu-exact-prefix25", Model(candidate, base.model_id), batch_size
            ),
        )
        measured: tuple[list[BenchmarkCycle], list[BenchmarkCycle]] = ([], [])
        with output.with_suffix(".cycles.jsonl").open("w") as journal:

            def run(index: int, stage: str, repeat: int) -> BenchmarkCycle:
                diagnostic = process_diagnostic()
                cycle = _run(engines[index], cases)
                quality = compare_cycles((reference,), (cycle,), gate)
                row: dict[str, object] = {
                    "arm": engines[index].name,
                    "stage": stage,
                    "repeat": repeat,
                    "cycle": cycle.to_dict(),
                    "reference_quality": asdict(quality),
                    "before": asdict(diagnostic),
                    "after": asdict(process_diagnostic()),
                }
                rows.append(row)
                payload["cycles"] = rows
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                write()
                return cycle

            screens = (run(0, "screen", 0), run(1, "screen", 0))
            if not compare_cycles((reference,) * 2, screens, gate).passed:
                payload.update(completed=True, phase="rejected screen")
                write()
                return False
            warmup = (run(0, "warmup", 0), run(1, "warmup", 0))
            for repeat in range(plan.repeats):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    measured[index].append(run(index, "measured", repeat))
        cycles = screens + warmup + tuple(measured[0]) + tuple(measured[1])
        reference_quality = compare_cycles((reference,) * len(cycles), cycles, gate)
        report = BenchmarkReport(
            _input_digest(cases),
            cases,
            plan,
            *engines,
            tuple(measured[0]),
            tuple(measured[1]),
            compare_cycles(*measured, gate),
            (warmup[0],),
            (warmup[1],),
        )
        ratios = [
            left.latency_ms / right.latency_ms
            for left, right in zip(*measured, strict=True)
        ]
        left_times = [cycle.latency_ms for cycle in measured[0]]
        right_times = [cycle.latency_ms for cycle in measured[1]]
        observed = {
            "paired_median": percentile(ratios, 0.5),
            "paired_p05": percentile(ratios, 0.05),
            "aggregate_throughput_ratio": sum(left_times) / sum(right_times),
            "candidate_cohort_p95_ratio": percentile(right_times, 0.95)
            / percentile(left_times, 0.95),
        }
        performance = (
            observed["paired_median"] >= 1.02
            and observed["paired_p05"] > 1
            and observed["aggregate_throughput_ratio"] > 1
            and observed["candidate_cohort_p95_ratio"] <= 1
        )
        quality_passed = report.passed and reference_quality.passed
        payload.update(
            report=report.to_dict(),
            reference_quality=asdict(reference_quality),
            quality_passed=quality_passed,
            performance_passed=performance,
            performance_observed=observed,
            accepted=performance and quality_passed,
            completed=True,
            phase="complete",
            diagnostics={
                "period": "prefix build plus all screen/warm/measured cycles",
                "before": asdict(before),
                "after": asdict(process_diagnostic()),
            },
        )
        write()
        return quality_passed
    except BaseException as error:
        payload.update(
            completed=False,
            quality_passed=False,
            performance_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        raise
    finally:
        torch.set_num_threads(previous_threads)
