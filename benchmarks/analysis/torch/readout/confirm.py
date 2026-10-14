"""Scalar full77 confirmation against retained prefix; no production promotion."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
from transformers.integrations import sdpa_attention

from benchmarks.analysis.evidence import freeze_sources, process_diagnostic
from benchmarks.analysis.reference import read_reference
from benchmarks.analysis.torch.prefix.production import study as original
from benchmarks.analysis.torch.process import launch
from wyat import Model
from wyat.backends.wyat.cpu.prefix import PrefixScorer
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, QualityGate, compare_cycles
from wyat.benchmark import _run as _run
from wyat.evaluation import _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

from . import study as screen
from .scorer import SelectedScorer

SCREEN_HISTORY = {
    "torch-readout-first8-review.json": (
        "b42ef790a0de67ddddc4265946e3d478e11dafcc8b05fb6b55eda2653a4040cd"
    ),
    "torch-readout-first8-review.execution.json": (
        "93c802f6f471008d990eac1cd7cff754fb19774fe2f9839a2822328d2a55213b"
    ),
    "torch-readout-first8.json": (
        "a57e8b52ae727ca954dc8f943d1eb6cdbbd622fb4a3cc6f9304e67158f249858"
    ),
    "torch-readout-first8.wall.json": (
        "ba5f7f489848a2ca891ac3c4e3852839fa3b08a336ef7197a1638e2d7a4cb7c3"
    ),
    "torch-readout-first8.cycles.jsonl": (
        "60edf2a08ecbd71f3adbc747eb28a7a7c2969363d9f79149e43f482d8ef30e0d"
    ),
    "torch-readout-first8.sources-7a31a5ea873ab0e2.json": (
        "7a31a5ea873ab0e2199fbe9b8fc0f7f41fcd3c1502703407f22a66930c85fdac"
    ),
}
SCHEDULE = (
    ("screen", "retained"),
    ("screen", "selected"),
    ("warmup", "retained"),
    ("warmup", "selected"),
    *(
        ("measured", arm)
        for pair in range(5)
        for arm in (
            ("retained", "selected") if pair % 2 == 0 else ("selected", "retained")
        )
    ),
)


def admit_screen() -> tuple[int, ...]:
    """Old capture identity comes from its archive, not expanded current globs."""
    prefix = screen.admit_history()
    original.verify_files(
        {original.RESULTS / name: sha for name, sha in SCREEN_HISTORY.items()}
    )
    saved = read_object(original.RESULTS / "torch-readout-first8.json")
    archive = read_object(
        original.RESULTS / next(name for name in SCREEN_HISTORY if ".sources-" in name)
    )
    files = cast(list[dict[str, object]], archive["files"])
    table: dict[str, str] = {}
    for item in files:
        sha = hashlib.sha256(cast(str, item["text"]).encode()).hexdigest()
        if sha != item["sha256"] or item["name"] in table:
            raise ValueError("Executed screen archive identity differs")
        table[cast(str, item["name"])] = sha
    if (
        saved["source_sha256"] != table
        or saved["completed"] is not True
        or saved["quality_passed"] is not True
        or saved["dispatch_passed"] is not True
        or saved["accepted"] is not False
        or saved["performance_passed"] is not False
        or saved["model_id"] != original.MODEL_ID
        or saved["dataset_sha256"] != original.DATA_SHA
        or saved["reference_sha256"] != original.REFERENCE_SHA
        or saved["full_input_sha256"] != original.INPUT_SHA
        or tuple(cast(list[int], saved["prefix_tokens"])) != prefix
    ):
        raise ValueError("Required first8 quality/dispatch evidence differs")
    review = read_object(original.RESULTS / "torch-readout-first8-review.json")
    if (
        review["audit_passed"] is not True
        or review["independent_quality_passed"] is not True
        or review["independent_dispatch_passed"] is not True
        or review["raw_outcomes"] != 88
        or review["main_sha256"] != SCREEN_HISTORY["torch-readout-first8.json"]
        or review["source_archive_sha256"]
        != SCREEN_HISTORY["torch-readout-first8.sources-7a31a5ea873ab0e2.json"]
    ):
        raise ValueError("Independent first8 raw audit did not pass")
    return prefix


def source_paths() -> dict[str, Path]:
    paths = screen.source_paths()
    for name in (
        "docs/analysis/cpu/readout/confirm.md",
        "tests/analysis/cpu/readout/test_confirm.py",
    ):
        paths[name] = original.ROOT / name
    return paths


def study(model: Path, output: Path) -> bool:
    if (
        output.exists()
        or output.with_suffix(".cycles.jsonl").exists()
        or any(output.parent.glob(output.stem + ".sources-*.json"))
    ):
        raise FileExistsError("Preserve existing receipts; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = QualityGate()
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "scope": (
            "Scalar full77 selected-last-block versus retained prefix; "
            "development exposure unknown"
        ),
        "batch_size": 1,
        "completed": False,
        "quality_passed": False,
        "performance_passed": False,
        "accepted": False,
        "threads_restored": False,
        "dispatch_observed": False,
        "phase": "admission",
        "cycles": rows,
        "pending_cohorts": list(range(14)),
        "quality_gate": asdict(gate),
        "performance_gate": {
            "paired_median_minimum": 1.02,
            "paired_p05_minimum_exclusive": 1.0,
            "aggregate_throughput_ratio_minimum_exclusive": 1.0,
            "candidate_cohort_p95_maximum_ratio": 1.0,
        },
        "protocol": {
            "schedule": [list(item) for item in SCHEDULE],
            "outcomes": 1078,
            "device": "cpu",
            "dtype": "float32",
            "threads": 8,
            "prefix_length": 25,
            "max_length": 512,
            "token_cache_size": 0,
            "timing": (
                "Complete ordinary cohorts; fresh cache inside calls; "
                "setup/hash/journal outside metric"
            ),
            "dispatch_scope": "Prior first8 B8 profiles only; scalar calls unprofiled",
            "promotion": (
                "Scalar pass requires separate B8 confirmation; no production change"
            ),
        },
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    with output.open("x") as initial:
        initial.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    previous = torch.get_num_threads()
    restored = False

    def restore() -> None:
        nonlocal restored
        restored = True
        try:
            torch.set_num_threads(previous)
            if torch.get_num_threads() != previous:
                raise ValueError("Original process thread count was not restored")
        except BaseException as error:
            payload["thread_restoration_error"] = f"{type(error).__name__}: {error}"
            raise
        payload["threads_restored"] = True

    try:
        prefix = admit_screen()
        inputs = {
            original.DATA: original.DATA_SHA,
            original.REFERENCE: original.REFERENCE_SHA,
        }
        inputs.update({model / name: sha for name, sha in original.MODEL_FILES.items()})
        original.verify_files(inputs)
        sources = freeze_sources(source_paths(), output)
        policy = original.process_policy()
        flags = {
            name: vars(sdpa_attention)[name]
            for name in (
                "_is_torch_mps_available",
                "_is_torch_greater_or_equal_than_2_13",
            )
        }
        history = (
            original.HISTORY
            | screen.HISTORY
            | SCREEN_HISTORY
            | {screen.CORE: screen.CORE_SHA}
        )
        payload.update(
            source_sha256=sources.hashes(),
            source_archive=sources.archive(),
            historical_sha256=history,
            asset_sha256={str(path): sha for path, sha in inputs.items()},
            process_policy=policy,
            sdpa_flags=flags,
            runtime={
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch_git_version": torch.version.git_version,
                "torch_build": torch.__config__.show(),
                "prior_threads": previous,
                "interop_threads": torch.get_num_interop_threads(),
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "tokenizers",
                        "safetensors",
                        "numpy",
                    )
                },
            },
            source_scope=(
                "On-disk Python/dependency source and runtime metadata; "
                "native binaries/imported objects not independently attested"
            ),
            phase="loading",
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError(
                "Native masked-GQA flags differ; no policy mutation allowed"
            )
        torch.set_num_threads(8)
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in original.DATA.read_text().splitlines()
        )
        if len(cases) != 77 or _input_digest(cases) != original.INPUT_SHA:
            raise ValueError("All original ordered77 requests required")
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != original.MODEL_ID:
            raise ValueError("Released model identity differs")
        reference = read_reference(
            original.REFERENCE, cases, base, dataset_sha256=original.DATA_SHA
        )
        encoded = tuple(base.encode_request(case.request) for case in cases)
        resources: list[PrefixScorer] = []
        build: list[float] = []
        for kind in (PrefixScorer, SelectedScorer):
            started = time.perf_counter_ns()
            resources.append(kind(base, prefix))
            build.append((time.perf_counter_ns() - started) / 1e6)
        if not all(resource._admits(row) for resource in resources for row in encoded):
            raise ValueError(
                "Every original encoding must admit the frozen exact prefix"
            )
        payload.update(
            model_id=base.model_id,
            dataset_sha256=original.DATA_SHA,
            reference_sha256=original.REFERENCE_SHA,
            input_sha256=original.INPUT_SHA,
            case_ids=[case.id for case in cases],
            prefix_tokens=list(prefix),
            token_lengths=[len(row.ids) for row in encoded],
            prefix_state_bytes=[resource.prefix_state_bytes for resource in resources],
            prefix_build_ms=build,
            phase="cycles",
        )
        write()
        engines = {
            "retained": BenchmarkEngine(
                "retained", Model(resources[0], base.model_id), 1
            ),
            "selected": BenchmarkEngine(
                "selected", Model(resources[1], base.model_id), 1
            ),
        }
        measured: dict[str, list[BenchmarkCycle]] = {"retained": [], "selected": []}
        anchors: dict[str, BenchmarkCycle] = {}
        observed: list[BenchmarkCycle] = []

        def verify() -> None:
            original.verify_sources(sources)
            original.verify_files(
                {original.RESULTS / name: sha for name, sha in history.items()}
            )
            if original.process_policy() != policy or torch.get_num_threads() != 8:
                raise ValueError("Owned process policy or threads changed")

        with output.with_suffix(".cycles.jsonl").open("x") as journal:
            for index, (stage, arm) in enumerate(SCHEDULE):
                verify()
                before = process_diagnostic()
                cycle = _run(engines[arm], cases)
                row: dict[str, object] = {
                    "index": index,
                    "stage": stage,
                    "arm": arm,
                    "cycle": original.json_cycle(cycle),
                    "before": asdict(before),
                    "after": asdict(process_diagnostic()),
                    "unknown_case_ids": [
                        case.id
                        for case in cases
                        if not any(
                            item.id == case.id
                            and item.status == "ok"
                            and item.prediction is not None
                            for item in cycle.rows
                        )
                    ],
                }
                rows.append(row)
                payload["pending_cohorts"] = list(range(index + 1, 14))
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                write()
                original.require_timing(cycle)
                quality = compare_cycles((reference,), (cycle,), gate)
                reports = [
                    compare_cycles((anchor,), (cycle,), gate)
                    for anchor in anchors.values()
                ]
                row["reference_quality"] = asdict(quality)
                row["anchor_quality"] = [asdict(report) for report in reports]
                write()
                if (
                    not quality.passed
                    or any(not report.passed for report in reports)
                    or cycle.failed
                    or cycle.correct != reference.correct
                ):
                    raise ValueError(
                        "Cohort numerical/rank/failure/accuracy gate failed"
                    )
                anchors.setdefault(arm, cycle)
                observed.append(cycle)
                if stage == "measured":
                    measured[arm].append(cycle)
                verify()
        quality = compare_cycles((reference,) * 14, observed, gate)
        paired = compare_cycles(measured["retained"], measured["selected"], gate)
        performance = original.performance(measured["retained"], measured["selected"])
        verify()
        original.verify_files(inputs)
        restore()
        passed = quality.passed and paired.passed
        speed = original.speed_passed(performance)
        payload.update(
            completed=True,
            quality_passed=passed,
            performance_passed=speed,
            accepted=passed and speed,
            phase="complete",
            reference_quality=asdict(quality),
            paired_quality=asdict(paired),
            performance_observed=performance,
            source_verified_after=True,
            asset_verified_after=True,
            limits=(
                "Development77 scalar assay; sample p05 is not a confidence bound. "
                "No timed dispatch, peak-memory, held-out quality "
                "or production promotion claim."
            ),
        )
        write()
        # Performance negatives are complete scientific observations, not crashes.
        return passed
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
        if not restored:
            try:
                restore()
                write()
            except BaseException:
                payload.update(
                    completed=False,
                    quality_passed=False,
                    performance_passed=False,
                    accepted=False,
                    phase="failed",
                )
                write()
                raise


class Arguments(Namespace):
    model: Path
    output: Path
    worker: bool


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help="internal owned worker")
    args = parser.parse_args(namespace=Arguments())
    if args.worker:
        raise SystemExit(0 if study(args.model.resolve(), args.output.resolve()) else 1)
    raise SystemExit(
        launch(
            __package__ + ".confirm",
            args.model.resolve(),
            args.output.resolve(),
            timeout=180,
        )
    )


if __name__ == "__main__":
    main()
