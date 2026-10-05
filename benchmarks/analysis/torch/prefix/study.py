"""Frozen first-eight CPU prefix screen; two controls and two CPU profiles."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from torch.autograd.profiler_util import FunctionEventAvg
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
    QualityGate,
    _run,
    compare_cycles,
    process_peak_rss_bytes,
)
from wyat.evaluation import _input_digest, case_from_dict
from wyat.wire import decode_json

from .cache import PrefixScorer

ROOT = Path(__file__).resolve().parents[4]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def study(model: Path, output: Path) -> bool:
    """Native Mac context required; no patching backend flags or attention policy."""
    gate = QualityGate()
    payload: dict[str, object] = {
        "scope": "first8 fixed training-source BANKING77 cases; exposure unknown",
        "hypothesis": (
            "Exact prefix removes linear work while preserving masked CPU flash"
        ),
        "completed": False,
        "quality_passed": False,
        "screen_passed": False,
        "accepted": False,
        "phase": "setup",
        "protocol": {
            "context": "native escalated Mac; unchanged maintained HF backend flags",
            "dtype": "float32",
            "device": "cpu",
            "intraop_threads": 8,
            "batch_size": 8,
            "max_length": 512,
            "token_cache_size": 0,
            "prefix_length": 25,
            "warmup_per_arm": 1,
            "ordinary_per_arm": 2,
            "profiles_per_arm": 1,
            "profiler": "CPU only, shapes, no stacks/memory; self times",
            "quality_gate": asdict(gate),
            "screen_gate": (
                "all strict gates; 28 CPU flash calls each profile, no math; "
                "actual Q16/KV8 head shapes; two ordinary timings descriptive"
            ),
            "timing": "fresh cache inside inference; eager prefix build separately",
            "promotion": (
                "none; passing screen justifies reviewed full77 five pairs then B1"
            ),
        },
        "cycles": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    prior_threads = torch.get_num_threads()
    cycles: list[dict[str, object]] = []
    try:
        write()
        sources = freeze_sources(
            {
                "candidate": Path(__file__).with_name("cache.py"),
                "study": Path(__file__),
                "runner": Path(__file__).with_name("run.py"),
                "launcher": ROOT / "benchmarks/analysis/torch/process.py",
                "reference": ROOT / "benchmarks/analysis/reference.py",
                "evidence": ROOT / "benchmarks/analysis/evidence.py",
                "benchmark": ROOT / "wyat/benchmark.py",
                "evaluation": ROOT / "wyat/evaluation.py",
                "backend": ROOT / "wyat/backends/kai/torch.py",
                "encoding": ROOT / "wyat/backends/kai/encoding.py",
                "artifact": ROOT / "wyat/backends/kai/artifact.py",
                "core": ROOT / "wyat/core.py",
                "qwen3": Path(modeling_qwen3.__file__),
                "maintained_cache": Path(cache_utils.__file__),
                "maintained_mask": Path(masking_utils.__file__),
                "sdpa_integration": Path(sdpa_attention.__file__),
                "cpu_sdpa_guards": Path(torch.__file__).parent
                / "include/ATen/native/transformers/sdp_utils_cpp.h",
            },
            output,
        )
        dataset = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
        reference_path = ROOT / "benchmarks/real/torch-fp32-reference77.json"
        if digest(dataset) != DATASET_SHA256:
            raise ValueError("Frozen development dataset changed")
        all_cases = tuple(
            case_from_dict(decode_json(line))
            for line in dataset.read_text().splitlines()
        )
        if len(all_cases) != 77:
            raise ValueError("Expected fixed 77-case development cohort")
        cases = all_cases[:8]
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
            dataset_sha256=DATASET_SHA256,
            reference_sha256=digest(reference_path),
            input_sha256=_input_digest(cases),
            case_ids=[case.id for case in cases],
            sdpa_flags=flags,
            torch_mps_built=torch.backends.mps.is_built(),
            torch_mps_available=torch.backends.mps.is_available(),
            versions={name: version(name) for name in ("torch", "transformers")},
            torch_git_version=torch.version.git_version,
            torch_build=torch.__config__.show(),
            prior_intraop_threads=prior_threads,
            interop_threads=torch.get_num_interop_threads(),
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError(
                "Frozen native HF masked-GQA eligibility flags unavailable"
            )
        torch.set_num_threads(8)
        payload["phase"] = "loading"
        write()
        base = KaiScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        saved = read_reference(
            reference_path, all_cases, base, dataset_sha256=DATASET_SHA256
        )
        reference = BenchmarkCycle(0.0, saved.rows[:8], 0, 0)
        encoded = tuple(base.encode_request(case.request) for case in cases)
        prefix = encoded[0].ids[:25]
        before_build = process_diagnostic()
        rss_before_build = process_peak_rss_bytes()
        build_start = time.perf_counter_ns()
        candidate = PrefixScorer(base, prefix)
        build_ms = (time.perf_counter_ns() - build_start) / 1e6
        after_build = process_diagnostic()
        if len(prefix) != 25 or not all(candidate.admits(row) for row in encoded):
            raise ValueError("First8 cases do not all admit exact25-token prefix")
        payload.update(
            model_id=base.model_id,
            prefix_tokens=list(prefix),
            token_lengths=[len(row.ids) for row in encoded],
            prefix_state_bytes=candidate.prefix_state_bytes,
            prefix_state_shapes=[
                {
                    "key": list(keys.shape),
                    "value": list(values.shape),
                    "key_stride": list(keys.stride()),
                    "value_stride": list(values.stride()),
                }
                for keys, values in candidate._states
            ],
            prefix_build_ms=build_ms,
            prefix_build_before=asdict(before_build),
            prefix_build_after=asdict(after_build),
            prefix_build_peak_rss_before_bytes=rss_before_build,
            prefix_build_peak_rss_after_bytes=process_peak_rss_bytes(),
            phase="cycles",
        )
        write()
        engines = {
            "baseline": BenchmarkEngine(
                "CPU FP32 full prefill", Model(base, base.model_id), 8
            ),
            "prefix": BenchmarkEngine(
                "CPU FP32 maintained prefix25", Model(candidate, base.model_id), 8
            ),
        }
        schedule = (
            ("warmup", "baseline"),
            ("warmup", "prefix"),
            ("ordinary", "baseline"),
            ("ordinary", "prefix"),
            ("ordinary", "prefix"),
            ("ordinary", "baseline"),
            ("profile", "baseline"),
            ("profile", "prefix"),
        )
        with output.with_suffix(".cycles.jsonl").open("w") as journal:
            for stage, arm in schedule:
                before = process_diagnostic()
                events: list[dict[str, object]] = []
                started = time.perf_counter_ns()
                if stage == "profile":
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU],
                        record_shapes=True,
                        with_stack=False,
                        profile_memory=False,
                    ) as profiler:
                        cycle = _run(engines[arm], cases)
                    outer_ms = (time.perf_counter_ns() - started) / 1e6
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
                else:
                    cycle = _run(engines[arm], cases)
                    outer_ms = (time.perf_counter_ns() - started) / 1e6
                quality = compare_cycles((reference,), (cycle,), gate)
                row: dict[str, object] = {
                    "stage": stage,
                    "arm": arm,
                    "cycle": cycle.to_dict(),
                    "quality": asdict(quality),
                    "passed": quality.passed,
                    "outer_wall_ms": outer_ms,
                    "before": asdict(before),
                    "after": asdict(process_diagnostic()),
                    "operator_self_times": events,
                    "self_cpu_sum_us": math.fsum(
                        cast(float, event["self_cpu_time_us"]) for event in events
                    ),
                }
                cycles.append(row)
                payload["cycles"] = cycles
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                write()
        times = {
            arm: [
                cast(float, cast(dict[str, object], row["cycle"])["latency_ms"])
                for row in cycles
                if row["arm"] == arm and row["stage"] == "ordinary"
            ]
            for arm in engines
        }
        profiles = [row for row in cycles if row["stage"] == "profile"]
        dispatch: list[dict[str, object]] = []
        for row in profiles:
            events = cast(list[dict[str, object]], row["operator_self_times"])
            flash = sum(
                cast(int, event["count"])
                for event in events
                if event["operator"]
                == "aten::_scaled_dot_product_flash_attention_for_cpu"
            )
            math_calls = sum(
                cast(int, event["count"])
                for event in events
                if event["operator"] == "aten::_scaled_dot_product_attention_math"
            )
            flash_shapes = [
                event["input_shapes"]
                for event in events
                if event["operator"]
                == "aten::_scaled_dot_product_flash_attention_for_cpu"
            ]
            dispatch.append(
                {
                    "arm": row["arm"],
                    "flash_calls": flash,
                    "math_calls": math_calls,
                    "flash_shapes": flash_shapes,
                    "native_gqa_heads": bool(flash_shapes)
                    and all(
                        len(shapes) >= 3
                        and len(shapes[0]) == len(shapes[1]) == len(shapes[2]) == 4
                        and shapes[0][1] == base.backbone.config.num_attention_heads
                        and shapes[1][1]
                        == shapes[2][1]
                        == base.backbone.config.num_key_value_heads
                        for shapes in cast(list[list[list[int]]], flash_shapes)
                    ),
                    "calculated_retained_call_kv_bytes": (
                        sum(
                            cast(int, event["count"])
                            * (math.prod(shapes[1]) + math.prod(shapes[2]))
                            * 4  # Frozen FP32 inputs; four bytes per element.
                            for event in events
                            if event["operator"]
                            == "aten::_scaled_dot_product_flash_attention_for_cpu"
                            for shapes in [cast(list[list[int]], event["input_shapes"])]
                        )
                        if row["arm"] == "prefix"
                        else None
                    ),
                }
            )
        ratio = statistics.median(times["baseline"]) / statistics.median(
            times["prefix"]
        )
        quality_passed = all(row["passed"] is True for row in cycles)
        payload.update(
            phase="complete",
            completed=True,
            quality_passed=quality_passed,
            screen_passed=(
                quality_passed
                and all(
                    row["flash_calls"] == 28
                    and row["math_calls"] == 0
                    and row["native_gqa_heads"] is True
                    for row in dispatch
                )
            ),
            ordinary_times_ms=times,
            ordinary_median_ratio=ratio,
            profile_over_ordinary_ratio={
                cast(str, row["arm"]): cast(
                    float, cast(dict[str, object], row["cycle"])["latency_ms"]
                )
                / statistics.median(times[cast(str, row["arm"])])
                for row in profiles
            },
            profile_dispatch=dispatch,
            limits=[
                "Two ordinary observations screen; five pairs required for promotion.",
                "Profiles follow controls; observer/order effects are unresolved.",
                "RSS is cumulative process high water, not live cache allocation.",
                "Cached suffix retains all28 extended KV layers; baseline does not.",
                "KV bytes derive from FP32 flash shapes; transient copies excluded.",
                "Repeated same eight cases do not establish held-out model quality.",
                "Source snapshots freeze setup disk bytes, not external execution.",
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
