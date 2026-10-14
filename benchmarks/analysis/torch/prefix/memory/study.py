"""Fresh-process first-eight resource screen; sampling and profiling are untimed."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from torch.autograd.profiler_util import FunctionEventAvg
from transformers import cache_utils, masking_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.torch.prefix.pairs import admit_history, digest
from benchmarks.analysis.torch.threads import DATASET_SHA256
from wyat import Model, Request
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.evaluation import _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

from .measure import MemorySampler, memory_snapshot
from .scorer import Arm, MemoryScorer, Observer, StorageObservation

ROOT = Path(__file__).resolve().parents[5]

# Prospective repeated resource screen, not an automatic production decision.
POLICY: dict[str, object] = {
    "pairs": 3,
    "order": ["retained/one-use", "one-use/retained", "retained/one-use"],
    "owned_child_cap_seconds": 180,
    "dtype": "float32",
    "device": "cpu",
    "threads": 8,
    "batch_size": 8,
    "max_length": 512,
    "token_cache_size": 0,
    "prefix_length": 25,
    "full_final_block": True,
    "cycles": ["screen", "warmup", "ordinary", "ordinary", "memory", "profile"],
    "timing": "prefix build excluded; fresh cache copying included; ordinary only",
    "memory_observer": "own-PID RSS via ps; separate memory cycle; profile LAST",
    "sample_interval_ms": 20,
    "minimum_interior_valid_samples": 3,
    "maximum_body_gap_ms": 100,
    "resource_gate": (
        "all three pairs reduction >= max(64MiB,5% retained sampled peak), "
        "plus lower observed selected live KV storage"
    ),
    "promotion": (
        "none; ordinary timings descriptive; failed/missing sampling unsupported"
    ),
    "common_path_change": (
        "both use_cache=False; same private owner kept through head; "
        "only cache lifetime differs"
    ),
}


@dataclass(frozen=True)
class ObservedScorer:
    scorer: MemoryScorer
    observer: Observer

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        return self.scorer.score_observed(requests, self.observer)


def dispatch(events: list[dict[str, object]], suffix_length: int) -> dict[str, object]:
    flash = [
        event
        for event in events
        if event["operator"] == "aten::_scaled_dot_product_flash_attention_for_cpu"
    ]
    geometry = bool(flash) and all(
        cast(list[list[int]], event["input_shapes"])[0:3]
        == [
            [8, 16, suffix_length, 128],
            [8, 8, suffix_length + 25, 128],
            [8, 8, suffix_length + 25, 128],
        ]
        and cast(list[list[int]], event["input_shapes"])[5]
        == [8, 1, suffix_length, suffix_length + 25]
        for event in flash
    )
    flash_calls = sum(cast(int, event["count"]) for event in flash)
    math_calls = sum(
        cast(int, event["count"])
        for event in events
        if event["operator"] == "aten::_scaled_dot_product_attention_math"
    )
    return {
        "flash_calls": flash_calls,
        "math_calls": math_calls,
        "gqa_offset_mask_geometry": geometry,
        "passed": flash_calls == 28 and math_calls == 0 and geometry,
    }


def inventory_complete(rows: Sequence[StorageObservation], prefix_bytes: int) -> bool:
    """Only selected owner-pool storage; concatenation transients are excluded."""
    return (
        [row.layer for row in rows] == [None, *range(28)]
        and rows[0].cache_owned_bytes == 8 * prefix_bytes
        and rows[0].combined_unique_storage_bytes == 9 * prefix_bytes
        and all(row.combined_unique_storage_bytes > 0 for row in rows)
        and all(row.returned_logical_bytes > 0 for row in rows[1:])
    )


REVIEW_SHA256 = "48055f286b2a2443da7e194d794258dbfae69bfd1514df7900fca40693443065"
RESULTS_DIRECTORY = Path("benchmarks/analysis/results")
LEGACY_RESULTS_DIRECTORY = Path("/Users/teilomillet/Code/wyat") / RESULTS_DIRECTORY


def evidence_path(value: str, root: Path = ROOT) -> Path:
    """Relocate a pinned review's result file; never follow its old checkout."""
    recorded = Path(value)
    directory = (
        LEGACY_RESULTS_DIRECTORY if recorded.is_absolute() else RESULTS_DIRECTORY
    )
    try:
        relative = recorded.relative_to(directory)
    except ValueError as error:
        raise ValueError(
            "Historical evidence must remain in the results folder"
        ) from error
    if len(relative.parts) != 1 or relative.name in {".", ".."} or "\\" in value:
        raise ValueError("Historical evidence must name one results file")
    results = root / RESULTS_DIRECTORY
    path = results / relative
    if path.resolve().parent != results.resolve():
        raise ValueError("Historical evidence escapes the results folder")
    return path


def source_paths(root: Path = ROOT) -> dict[str, Path]:
    """Current execution owners, including the actually imported HF modules."""
    prefix = root / "benchmarks/analysis/torch/prefix"
    memory = prefix / "memory"
    return {
        "scorer": memory / "scorer.py",
        "study": memory / "study.py",
        "runner": memory / "run.py",
        "measure": memory / "measure.py",
        "prefix": prefix / "cache.py",
        "one_use": prefix / "lifetime.py",
        "admission": prefix / "pairs.py",
        "first8_study": prefix / "study.py",
        "launcher": root / "benchmarks/analysis/torch/process.py",
        "reference": root / "benchmarks/analysis/reference.py",
        "evidence": root / "benchmarks/analysis/evidence.py",
        "benchmark": root / "wyat/benchmark.py",
        "core": root / "wyat/core.py",
        "evaluation": root / "wyat/evaluation.py",
        "backend": root / "wyat/backends/wyat/torch.py",
        "encoding": root / "wyat/backends/wyat/encoding.py",
        "artifact": root / "wyat/backends/wyat/artifact.py",
        "qwen3": Path(modeling_qwen3.__file__),
        "maintained_cache": Path(cache_utils.__file__),
        "maintained_mask": Path(masking_utils.__file__),
        "sdpa": Path(sdpa_attention.__file__),
    }


def admit_confirmation(
    history: dict[str, object], root: Path = ROOT
) -> dict[str, object]:
    """Pin the independently reconstructed full77 receipts and complete histories."""
    review_path = root / "benchmarks/analysis/results/torch-prefix77-review.json"
    if digest(review_path) != REVIEW_SHA256:
        raise ValueError("Full77 independent review changed")
    review = read_object(review_path)
    results = cast(list[dict[str, object]], review["shape_results"])
    if [row["batch_size"] for row in results] != [8, 1]:
        raise ValueError("Full77 shape confirmations incomplete")
    for row in results:
        evidence = cast(dict[str, dict[str, str]], row["evidence"])
        for item in evidence.values():
            if digest(evidence_path(item["path"], root)) != item["sha256"]:
                raise ValueError("Full77 confirmed history changed")
        receipt = read_object(evidence_path(evidence["receipt"]["path"], root))
        if any(
            receipt[name] != history[name]
            for name in ("model_id", "dataset_sha256", "reference_sha256")
        ):
            raise ValueError("Full77 canonical identity differs from first8")
        if receipt["input_sha256"] != row["canonical_input_sha256"]:
            raise ValueError("Full77 canonical inputs changed")
        if not all(
            receipt[name] is True
            for name in (
                "completed",
                "quality_passed",
                "performance_passed",
                "accepted",
            )
        ):
            raise ValueError("Full77 confirmation not accepted")
        archive = read_object(evidence_path(evidence["source_archive"]["path"], root))
        sources = cast(list[dict[str, str]], archive["files"])
        if (
            len(sources) != len(cast(dict[str, str], receipt["source_sha256"]))
            or {source["name"]: source["sha256"] for source in sources}
            != receipt["source_sha256"]
        ):
            raise ValueError("Confirmed full77 source inventory changed")
        for source in sources:
            if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
                raise ValueError("Confirmed full77 archived source changed")
    return {
        "review_sha256": REVIEW_SHA256,
        "evidence": [row["evidence"] for row in results],
        "raw_outputs": 2156,
    }


def study(model: Path, output: Path, *, arm: Arm, repeat: int) -> bool:
    if (
        arm not in ("retained", "one-use")
        or type(repeat) is not int
        or repeat not in (0, 1, 2)
    ):
        raise ValueError("Memory study requires retained/one-use and repeat0/1/2")
    gate = QualityGate()
    payload: dict[str, object] = {
        "scope": ("first8 fixed training-source BANKING77 cases; exposure unknown"),
        "hypothesis": (
            "One-use cache reduces selected live KV ownership and observed peak "
            "current RSS without numerical/flash drift"
        ),
        "completed": False,
        "quality_passed": False,
        "screen_passed": False,
        "accepted": False,
        "phase": "setup",
        "arm": arm,
        "repeat": repeat,
        "protocol": POLICY,
        "quality_gate": asdict(gate),
        "cycles": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    prior_threads = torch.get_num_threads()
    cycles: list[dict[str, object]] = []
    try:
        write()
        history = admit_history()
        payload["full77_admission"] = admit_confirmation(history)
        sources = freeze_sources(source_paths(), output)
        dataset = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        reference_path = ROOT / "benchmarks/real/torch-fp32-reference77.json"
        all_cases = tuple(
            case_from_dict(decode_json(line))
            for line in dataset.read_text().splitlines()
        )
        cases = all_cases[:8]
        flags = {
            name: vars(sdpa_attention)[name]
            for name in (
                "_is_torch_mps_available",
                "_is_torch_greater_or_equal_than_2_13",
            )
        }
        payload.update(
            first8_admission=history,
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            dataset_sha256=digest(dataset),
            reference_sha256=digest(reference_path),
            input_sha256=_input_digest(cases),
            case_ids=[case.id for case in cases],
            sdpa_flags=flags,
            torch_mps_built=torch.backends.mps.is_built(),
            torch_mps_available=torch.backends.mps.is_available(),
            versions={name: version(name) for name in ("torch", "transformers")},
            torch_git_version=torch.version.git_version,
            torch_build=torch.__config__.show(),
            prior_threads=prior_threads,
            interop_threads=torch.get_num_interop_threads(),
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError(
                "Native maintained masked-GQA eligibility flags unavailable"
            )
        torch.set_num_threads(8)
        payload.update(phase="loading", before_load=asdict(memory_snapshot()))
        write()
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != history["model_id"]:
            raise ValueError("Model differs from admitted history")
        saved = read_reference(
            reference_path, all_cases, base, dataset_sha256=DATASET_SHA256
        )
        reference = BenchmarkCycle(0.0, saved.rows[:8], 0, 0)
        encoded = tuple(base.encode_request(case.request) for case in cases)
        prefix = encoded[0].ids[:25]
        payload["after_load"] = asdict(memory_snapshot())
        started = time.perf_counter_ns()
        scorer = MemoryScorer(base, prefix, arm)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        if list(prefix) != history["prefix_tokens"] or not all(
            scorer.admits(row) for row in encoded
        ):
            raise ValueError("First8 does not admit frozen exact25-token prefix")
        payload.update(
            model_id=base.model_id,
            prefix_tokens=list(prefix),
            token_lengths=[len(row.ids) for row in encoded],
            prefix_state_bytes=scorer.prefix_state_bytes,
            prefix_build_ms=build_ms,
            after_prefix_build=asdict(memory_snapshot()),
            phase="cycles",
        )
        write()
        engine = BenchmarkEngine(
            f"CPU FP32 prefix25 {arm} private cache", Model(scorer, base.model_id), 8
        )
        with output.with_suffix(".cycles.jsonl").open("w") as journal:
            for stage in cast(list[str], POLICY["cycles"]):
                events: list[dict[str, object]] = []
                storage: list[StorageObservation] = []
                extra: dict[str, object] = {}
                if stage == "memory":
                    # Selected owner-pool enumeration and ps sampling only here.
                    memory_engine = BenchmarkEngine(
                        engine.name,
                        Model(ObservedScorer(scorer, storage.append), base.model_id),
                        8,
                    )
                    with MemorySampler() as sampler:
                        cycle = _run(memory_engine, cases)
                    report = sampler.report()
                    inventory_passed = inventory_complete(
                        storage, scorer.prefix_state_bytes
                    )
                    extra.update(
                        memory=asdict(report),
                        selected_kv_storage=[asdict(item) for item in storage],
                        memory_reliable=report.completed
                        and report.sampled_peak_rss_bytes is not None
                        and report.interior_valid_samples >= 3
                        and max(report.body_gaps_ns, default=0) <= 100_000_000,
                        inventory_complete=inventory_passed,
                        selected_live_storage_peak_bytes=(
                            max(item.combined_unique_storage_bytes for item in storage)
                            if inventory_passed
                            else None
                        ),
                    )
                elif stage == "profile":
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=True,
                        with_stack=False,
                        profile_memory=False,
                    ) as profiler:
                        cycle = _run(engine, cases)
                    averages = cast(
                        Sequence[FunctionEventAvg],
                        profiler.key_averages(group_by_input_shape=True),
                    )
                    events = [
                        {
                            "operator": event.key,
                            "count": event.count,
                            "input_shapes": event.input_shapes,
                            "self_cpu_time_us": float(event.self_cpu_time_total),
                        }
                        for event in averages
                    ]
                    extra["dispatch"] = dispatch(
                        events, max(len(row.ids) for row in encoded) - 25
                    )
                else:
                    cycle = _run(engine, cases)
                quality = compare_cycles((reference,), (cycle,), gate)
                row: dict[str, object] = {
                    "stage": stage,
                    "cycle": cycle.to_dict(),
                    "quality": asdict(quality),
                    "passed": quality.passed,
                    "operator_self_times": events,
                    **extra,
                }
                cycles.append(row)
                payload["cycles"] = cycles
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                if stage == "warmup":
                    payload["after_warmup"] = asdict(memory_snapshot())
                write()
        memory = next(row for row in cycles if row["stage"] == "memory")
        profile = next(row for row in cycles if row["stage"] == "profile")
        ordinary = [
            cast(float, cast(dict[str, object], row["cycle"])["latency_ms"])
            for row in cycles
            if row["stage"] == "ordinary"
        ]
        quality_passed = all(row["passed"] is True for row in cycles)
        payload.update(
            phase="complete",
            completed=True,
            quality_passed=quality_passed,
            screen_passed=quality_passed
            and memory["memory_reliable"] is True
            and memory["inventory_complete"] is True
            and cast(dict[str, object], profile["dispatch"])["passed"] is True,
            ordinary_times_ms=ordinary,
            ordinary_median_ms=statistics.median(ordinary),
            observer_over_ordinary_ratio={
                str(row["stage"]): cast(
                    float, cast(dict[str, object], row["cycle"])["latency_ms"]
                )
                / statistics.median(ordinary)
                for row in (memory, profile)
            },
            after_all_cycles=asdict(memory_snapshot()),
            unsupported_reasons=[
                reason
                for condition, reason in (
                    (
                        quality_passed,
                        "One or more cycles failed frozen CPU quality gates",
                    ),
                    (
                        memory["memory_reliable"] is True,
                        "RSS sampling incomplete/unavailable or exceeds "
                        "frozen coverage bound",
                    ),
                    (
                        memory["inventory_complete"] is True,
                        "Selected KV inventory missing/duplicated/invalid",
                    ),
                    (
                        cast(dict[str, object], profile["dispatch"])["passed"] is True,
                        "Actual CPU masked flash GQA geometry/count failed",
                    ),
                )
                if not condition
            ],
            limits=[
                "Sampled current RSS can miss peaks; selected KV storage excludes "
                "activations/transient projection inputs/allocator arenas.",
                "Cumulative HWM includes cold model load and is not current live RSS.",
                "Single child supports no paired resource contrast; "
                "all three fresh pairs required.",
                "Observed factories instrument memory cycle only; "
                "ordinary times descriptive; no production promotion.",
            ],
        )
        write()
        return payload["screen_passed"] is True
    except BaseException as error:
        payload.update(
            completed=False,
            quality_passed=False,
            screen_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        raise
    finally:
        torch.set_num_threads(prior_threads)
