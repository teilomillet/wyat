"""Reuse canonical pairs; strict legacy gates apply to every returned cycle."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import torch
from transformers.integrations import sdpa_attention

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.torch.prefix.memory.scorer import MemoryScorer
from benchmarks.analysis.torch.prefix.memory.study import (
    admit_confirmation,
    evidence_path,
    source_paths,
)
from benchmarks.analysis.torch.prefix.pairs import admit_history, digest
from benchmarks.analysis.torch.threads import DATASET_SHA256
from wyat import Model
from wyat.backends.wyat.artifact import load_artifact
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    benchmark_pair,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict, percentile
from wyat.wire import decode_json, read_object

ROOT = Path(__file__).resolve().parents[5]
MEMORY_REVIEW_SHA256 = (
    "1bd3339589db488fa36cd175493efa1e665772dbb6c56585d063c15f3b4692c3"
)
type Comparator = Literal["retained", "full"]
MEDIAN_MINIMUM = {"retained": 1.0, "full": 1.02}


def admit_resource(root: Path = ROOT) -> tuple[dict[str, object], dict[str, Path]]:
    """Validate immutable history; return paths to freeze for this new execution."""
    history = admit_history(root)
    full77 = admit_confirmation(history, root)
    path = root / "benchmarks/analysis/results/torch-prefix-memory-review.json"
    if digest(path) != MEMORY_REVIEW_SHA256:
        raise ValueError("Independent six-child memory review changed")
    review = read_object(path)
    children = cast(list[dict[str, object]], review["children"])
    order = [[row["arm"], row["repeat"]] for row in children]
    if (
        order
        != [
            ["retained", 0],
            ["one-use", 0],
            ["one-use", 1],
            ["retained", 1],
            ["retained", 2],
            ["one-use", 2],
        ]
        or review["resource_screen_supported"] is not True
    ):
        raise ValueError("Six-child memory evidence incomplete/unsupported")
    paths = source_paths(root)
    for child in children:
        evidence = cast(dict[str, dict[str, str]], child["evidence"])
        for item in evidence.values():
            if digest(evidence_path(item["path"], root)) != item["sha256"]:
                raise ValueError(
                    "Confirmed memory receipt/journal/wall/archive changed"
                )
        receipt = read_object(evidence_path(evidence["receipt"]["path"], root))
        if (
            any(
                receipt[name] != history[name]
                for name in ("model_id", "dataset_sha256", "reference_sha256")
            )
            or receipt["input_sha256"] != child["input_sha256"]
        ):
            raise ValueError("Memory canonical identity differs from admitted history")
        if receipt["completed"] is not True or receipt["screen_passed"] is not True:
            raise ValueError("Memory child did not complete passing screen")
        archive = read_object(evidence_path(evidence["archive"]["path"], root))
        sources = cast(list[dict[str, str]], archive["files"])
        if (
            len(sources) != len(cast(dict[str, str], receipt["source_sha256"]))
            or {source["name"]: source["sha256"] for source in sources}
            != receipt["source_sha256"]
            or {source["name"] for source in sources} != paths.keys()
        ):
            raise ValueError("Confirmed memory source inventory changed")
        for source in sources:
            if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
                raise ValueError("Confirmed memory archived mechanics changed")
    return {
        "first8": history,
        "full77": full77,
        "memory_review_sha256": MEMORY_REVIEW_SHA256,
        "memory_children": [child["evidence"] for child in children],
        "model_id": history["model_id"],
        "prefix_tokens": history["prefix_tokens"],
    }, paths


def performance(report: BenchmarkReport, comparator: Comparator) -> dict[str, object]:
    """Selection differs from full-prefill promotion; declare both before data."""
    left = [cycle.latency_ms for cycle in report.baseline_cycles]
    right = [cycle.latency_ms for cycle in report.candidate_cycles]
    if (
        not left
        or len(left) != len(right)
        or any(not math.isfinite(t) or t <= 0 for t in left + right)
    ):
        raise ValueError("Performance requires complete finite positive paired times")
    ratios = [a / b for a, b in zip(left, right, strict=True)]
    median, p05 = percentile(ratios, 0.5), percentile(ratios, 0.05)
    aggregate = math.fsum(left) / math.fsum(right)
    p95 = percentile(right, 0.95) / percentile(left, 0.95)
    return {
        "paired_ratios": ratios,
        "paired_median": median,
        "paired_p05": p05,
        "aggregate_throughput_ratio": aggregate,
        "candidate_cohort_p95_ratio": p95,
        "passed": median >= MEDIAN_MINIMUM[comparator]
        and p05 > 1
        and aggregate > 1
        and p95 <= 1,
    }


def study(
    model: Path, output: Path, *, comparator: Comparator, batch_size: int
) -> bool:
    if (
        comparator not in MEDIAN_MINIMUM
        or type(batch_size) is not int
        or batch_size not in (1, 8)
    ):
        raise ValueError("Latency requires retained/full comparator and batch1/8")
    plan = BenchmarkProtocol(repeats=5, warmup=1)
    payload: dict[str, object] = {
        "scope": "pinned77 training-source cases; exposure unknown; native Mac FP32",
        "comparator": comparator,
        "batch_size": batch_size,
        "phase": "admission",
        "completed": False,
        "quality_passed": False,
        "performance_passed": False,
        "selection_passed": False,
        "accepted": False,
        "benchmark_returned": False,
        "raw_partial_cycles_available": False,
        "protocol": {
            "dtype": "float32",
            "device": "cpu",
            "threads": 8,
            "token_cache_size": 0,
            "max_length": 512,
            "prefix_length": 25,
            "full_final_block": True,
            "benchmark": asdict(plan),
            "median_minimum_by_comparator": MEDIAN_MINIMUM,
            "paired_p05_minimum_exclusive": 1.0,
            "aggregate_throughput_minimum_exclusive": 1.0,
            "cohort_p95_ratio_maximum": 1.0,
            "owner_cap_seconds": 180,
            "partial_limit": (
                "benchmark_pair returns cycles only at completion; timeout/exception "
                "inside it retains setup, partial timing/quality unknown"
            ),
            "timing": (
                "all per-call cache copies included; cold prefix build outside; "
                "no observers/profilers"
            ),
            "promotion": "none automatic; four separate comparator/shape confirmations",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    prior_threads = torch.get_num_threads()
    try:
        write()
        admission, paths = admit_resource()
        paths.update(
            latency_study=Path(__file__),
            latency_runner=Path(__file__).with_name("run.py"),
        )
        sources = freeze_sources(paths, output)
        dataset = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in dataset.read_text().splitlines()
        )
        full_review = read_object(
            ROOT / "benchmarks/analysis/results/torch-prefix77-review.json"
        )
        expected_input = cast(list[dict[str, object]], full_review["shape_results"])[0][
            "canonical_input_sha256"
        ]
        if (
            len(cases) != 77
            or digest(dataset) != DATASET_SHA256
            or _input_digest(cases) != expected_input
        ):
            raise ValueError("Frozen full77 canonical inputs changed")
        flags = {
            name: vars(sdpa_attention)[name]
            for name in (
                "_is_torch_mps_available",
                "_is_torch_greater_or_equal_than_2_13",
            )
        }
        payload.update(
            admission=admission,
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            input_sha256=_input_digest(cases),
            dataset_sha256=DATASET_SHA256,
            reference_sha256=digest(
                ROOT / "benchmarks/real/torch-fp32-reference77.json"
            ),
            sdpa_flags=flags,
            torch_mps_built=torch.backends.mps.is_built(),
            torch_mps_available=torch.backends.mps.is_available(),
            torch_build=torch.__config__.show(),
            torch_git_version=torch.version.git_version,
            prior_threads=prior_threads,
            interop_threads=torch.get_num_interop_threads(),
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError("Native masked-GQA eligibility flags unavailable")
        if load_artifact(model, 512).model_id != admission["model_id"]:
            raise ValueError(
                "Model artifact differs from confirmed history before load"
            )
        torch.set_num_threads(8)
        payload["phase"] = "loading"
        write()
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != admission["model_id"]:
            raise ValueError("Loaded model differs from admitted artifact")
        reference = read_reference(
            ROOT / "benchmarks/real/torch-fp32-reference77.json",
            cases,
            base,
            dataset_sha256=DATASET_SHA256,
        )
        prefix = tuple(cast(list[int], admission["prefix_tokens"]))
        start = time.perf_counter_ns()
        candidate = MemoryScorer(base, prefix, "one-use")
        candidate_build_ms = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns()
        baseline = (
            MemoryScorer(base, prefix, "retained") if comparator == "retained" else base
        )
        baseline_build_ms = (
            (time.perf_counter_ns() - start) / 1e6 if comparator == "retained" else None
        )
        if not all(
            candidate.admits(base.encode_request(case.request)) for case in cases
        ):
            raise ValueError("Full77 does not admit exact25-token prefix")
        payload.update(
            phase="benchmarking",
            model_id=base.model_id,
            prefix_tokens=list(prefix),
            candidate_prefix_build_ms=candidate_build_ms,
            baseline_prefix_build_ms=baseline_build_ms,
            benchmark_before=asdict(process_diagnostic()),
        )
        write()  # Explicit unsupported partial boundary before canonical pair call.
        left = BenchmarkEngine(comparator, Model(baseline, base.model_id), batch_size)
        right = BenchmarkEngine("one-use", Model(candidate, base.model_id), batch_size)
        report = benchmark_pair(left, right, cases, protocol=plan)
        payload.update(
            phase="returned",
            benchmark_returned=True,
            report=report.to_dict(),
            benchmark_after=asdict(process_diagnostic()),
        )
        write()  # Preserve every returned raw output before further gate calculations.
        groups = {
            "baseline_warmup": report.baseline_warmup_cycles,
            "candidate_warmup": report.candidate_warmup_cycles,
            "baseline_measured": report.baseline_cycles,
            "candidate_measured": report.candidate_cycles,
        }
        if (
            report.protocol != plan
            or report.baseline is not left
            or report.candidate is not right
            or report.cases != cases
            or report.input_sha256 != expected_input
            or [len(cycles) for cycles in groups.values()] != [1, 1, 5, 5]
            or any(
                len(cycle.rows) != 77 for cycles in groups.values() for cycle in cycles
            )
        ):
            raise ValueError("Canonical benchmark returned inconsistent contract")
        checks = {
            name: [
                asdict(compare_cycles((reference,), (cycle,), plan.quality_gate))
                for cycle in cycles
            ]
            for name, cycles in groups.items()
        }
        quality = report.passed and all(
            item["passed"] is True for rows in checks.values() for item in rows
        )
        metrics = performance(report, comparator)
        payload.update(
            reference_checks=checks,
            performance=metrics,
            quality_passed=quality,
            performance_passed=metrics["passed"],
            selection_passed=quality and metrics["passed"] is True,
            completed=True,
            phase="complete",
        )
        write()
        return payload["selection_passed"] is True
    except BaseException as error:
        payload.update(
            completed=False,
            quality_passed=False,
            performance_passed=False,
            selection_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        raise
    finally:
        torch.set_num_threads(prior_threads)
