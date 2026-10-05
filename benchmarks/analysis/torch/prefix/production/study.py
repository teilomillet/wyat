"""Fixed full77 final-source CPU prefix assay; historical archives remain intact."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import cast

import torch
import transformers.tokenization_utils_tokenizers as tokenization_utils_tokenizers
from transformers import (
    activations,
    cache_utils,
    configuration_utils,
    masking_utils,
    modeling_rope_utils,
    modeling_utils,
    tokenization_utils_base,
)
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import configuration_qwen3, modeling_qwen3

from benchmarks.analysis.evidence import (
    FrozenSources,
    freeze_sources,
    process_diagnostic,
)
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.kai.cpu.prefix import PrefixScorer
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    QualityGate,
    compare_cycles,
)
from wyat.benchmark import (
    _run as _run,
)
from wyat.evaluation import _input_digest, case_from_dict, percentile
from wyat.wire import decode_json, read_object

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT / "benchmarks/analysis/results"
DATA = ROOT / "benchmarks/real/banking77-train-dev-menu6.jsonl"
REFERENCE = ROOT / "benchmarks/real/torch-fp32-reference77.json"
DATA_SHA = "0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03"
REFERENCE_SHA = "b8a90b31941b60e793357a8cb360767520da70ec33d2161c1242606962b22664"
INPUT_SHA = "fd5dd942250c6416c4ab50c59a5164a20909d11ead12bf83c505efff6a5556e8"
MODEL_ID = "kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee"
MODEL_FILES = {
    "LICENSE": ("cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "MODEL_MANIFEST.json": (
        "ca7bf2cff1494a5bfc51f4271f52557da0f3ce3b6f0f6e7a2b4118cb4965e92e"
    ),
    "backbone/config.json": (
        "fcf62f425b626755c6df66630215aed57c70b38bf779e6ac9aeff376224310ac"
    ),
    "backbone/model.safetensors": (
        "43369a2bfddb61495beb7cea6bf30d7564a5c5caff31f4dbf2e6cfce48423b69"
    ),
    "config.json": ("35ce70886f7642ec5eacf969e27da404915824aa25f6a30a19a1107cb2ec99b3"),
    "decision_config.json": (
        "4778b8818f8510bf761eebba48ec5ffd5941c8f95ef2b228a97162fb0c814a6e"
    ),
    "decision_head.safetensors": (
        "508dfe385c8fde8eb0b4eec5c199f5fe8363efeb247de024a08765b53183409d"
    ),
    "tokenizer.json": (
        "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506"
    ),
    "tokenizer_config.json": (
        "cbea7bca9904d56693d2226b00fd564e3be053609d1e50f8f1885510f0f6e790"
    ),
}
HISTORY = {
    "torch-prefix77-review.json": (
        "48055f286b2a2443da7e194d794258dbfae69bfd1514df7900fca40693443065"
    ),
    "torch-prefix77-b8.json": (
        "3d333417639b8f7cc5f5d49a4f2ccf4f2e9d3654d8df39b448aa62fb0d375d61"
    ),
    "torch-prefix77-b1.json": (
        "525cfce1066c6c1d3064838de85e9bbf880f71031596e56662c8c1d61badc915"
    ),
    "torch-prefix77-b8.cycles.jsonl": (
        "7aec4c6d8d779b133322df320f9598e1b512687b4f5332bfed2228aae9cb3fc1"
    ),
    "torch-prefix77-b1.cycles.jsonl": (
        "466d812155eaddf8f1b6aaf7f7808868986c47e7abb2dc59039c13b3dac383f0"
    ),
    "torch-prefix77-b8.wall.json": (
        "cc6fd605d90bf9cf0c7e853a1f0b6e1d02dd8e30d6986dcf61397a1e1ddc89cc"
    ),
    "torch-prefix77-b1.wall.json": (
        "b5e5be5fcf07fac2c0bee01070c55affc0aefe24cce2a66a47e5acaed0a9f79e"
    ),
    "torch-prefix77-b8.sources-d86eb841ae184bbb.json": (
        "d86eb841ae184bbb6413418a86c542dc668eb14a310292b9cecfa1cfc3faeb31"
    ),
    "torch-prefix77-b1.sources-d86eb841ae184bbb.json": (
        "d86eb841ae184bbb6413418a86c542dc668eb14a310292b9cecfa1cfc3faeb31"
    ),
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_files(pins: dict[Path, str]) -> None:
    if any(digest(path) != expected for path, expected in pins.items()):
        raise ValueError("Pinned input, source or historical evidence changed")


def admit_history(results: Path = RESULTS) -> tuple[int, ...]:
    """Historical source identity comes from archives, never the live checkout."""
    verify_files({results / name: expected for name, expected in HISTORY.items()})
    prefixes: list[tuple[int, ...]] = []
    for batch in (8, 1):
        saved = read_object(results / f"torch-prefix77-b{batch}.json")
        archive = read_object(
            results / f"torch-prefix77-b{batch}.sources-d86eb841ae184bbb.json"
        )
        files = cast(list[dict[str, object]], archive["files"])
        table = cast(dict[str, str], saved["source_sha256"])
        if len(files) != len(table) or len({item["name"] for item in files}) != len(
            files
        ):
            raise ValueError("Historical source inventory differs")
        for item in files:
            checksum = hashlib.sha256(cast(str, item["text"]).encode()).hexdigest()
            if checksum != item["sha256"] or table[cast(str, item["name"])] != checksum:
                raise ValueError("Historical archived source bytes differ")
        if (
            saved["completed"] is not True
            or saved["quality_passed"] is not True
            or saved["accepted"] is not True
            or saved["model_id"] != MODEL_ID
            or saved["dataset_sha256"] != DATA_SHA
            or saved["reference_sha256"] != REFERENCE_SHA
            or saved["input_sha256"] != INPUT_SHA
        ):
            raise ValueError("Historical full77 confirmation is unsupported")
        tokens = cast(list[int], saved["prefix_tokens"])
        if len(tokens) != 25 or any(type(token) is not int for token in tokens):
            raise ValueError("Historical exact25-token prefix differs")
        prefixes.append(tuple(tokens))
    if prefixes[0] != prefixes[1]:
        raise ValueError("Historical loads must use the same prefix")
    return prefixes[0]


def source_paths() -> dict[str, Path]:
    paths = {
        str(path.relative_to(ROOT)): path for path in (ROOT / "wyat").rglob("*.py")
    }
    for relative in (
        "benchmarks/__init__.py",
        "benchmarks/analysis/__init__.py",
        "benchmarks/analysis/evidence.py",
        "benchmarks/analysis/reference.py",
        "benchmarks/analysis/torch/process.py",
        "benchmarks/analysis/torch/prefix/__init__.py",
        "docs/analysis/cpu/prefix.md",
        "tests/backends/cpu/test_reuse.py",
        "tests/analysis/test_cpu_prefix_production.py",
        "pyproject.toml",
    ):
        paths[relative] = ROOT / relative
    for path in Path(__file__).parent.glob("*.py"):
        paths[str(path.relative_to(ROOT))] = path
    for module in (
        modeling_qwen3,
        configuration_qwen3,
        cache_utils,
        masking_utils,
        sdpa_attention,
        modeling_rope_utils,
        activations,
        configuration_utils,
        modeling_utils,
        tokenization_utils_base,
        tokenization_utils_tokenizers,
        torch.nn.modules.module,
        torch.nn.modules.linear,
        torch.nn.modules.normalization,
        torch.nn.functional,
        torch.backends,
        torch.backends.cuda,
        torch.autograd.grad_mode,
        torch.amp.autocast_mode,
    ):
        paths[module.__name__] = Path(cast(str, module.__file__))
    return paths


def verify_sources(sources: FrozenSources) -> None:
    verify_files({Path(item.path): item.sha256 for item in sources.files})
    verify_files({Path(sources.archive_path): sources.archive_sha256})


def process_policy() -> dict[str, object]:
    """Observe shared settings; this assay never changes attention/precision policy."""
    return {
        "flash_sdpa": cast(Callable[[], bool], torch.backends.cuda.flash_sdp_enabled)(),
        "math_sdpa": cast(Callable[[], bool], torch.backends.cuda.math_sdp_enabled)(),
        "efficient_sdpa": cast(
            Callable[[], bool], torch.backends.cuda.mem_efficient_sdp_enabled
        )(),
        "cudnn_sdpa": cast(Callable[[], bool], torch.backends.cuda.cudnn_sdp_enabled)(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_autocast": torch.is_autocast_enabled("cpu"),
    }


def performance(
    baseline: list[BenchmarkCycle], candidate: list[BenchmarkCycle]
) -> dict[str, float]:
    left = [cycle.latency_ms for cycle in baseline]
    right = [cycle.latency_ms for cycle in candidate]
    if (
        len(left) != 5
        or len(right) != 5
        or any(not math.isfinite(value) or value <= 0 for value in (*left, *right))
    ):
        raise ValueError("All five finite positive paired cohort timings required")
    ratios = [a / b for a, b in zip(left, right, strict=True)]
    return {
        "paired_median": percentile(ratios, 0.5),
        "paired_p05": percentile(ratios, 0.05),
        "aggregate_throughput_ratio": math.fsum(left) / math.fsum(right),
        "candidate_cohort_p95_ratio": percentile(right, 0.95) / percentile(left, 0.95),
    }


def speed_passed(observed: dict[str, float]) -> bool:
    return (
        observed["paired_median"] >= 1.02
        and observed["paired_p05"] > 1
        and observed["aggregate_throughput_ratio"] > 1
        and observed["candidate_cohort_p95_ratio"] <= 1
    )


def json_cycle(cycle: BenchmarkCycle) -> dict[str, object]:
    """Retain raw outcomes while making invalid clock evidence JSON-safe."""
    saved = cycle.to_dict()
    records = [saved, *cast(list[dict[str, object]], saved["rows"])]
    for record in records:
        value = cast(float, record["latency_ms"])
        if not math.isfinite(value):
            record.update(latency_ms=None, invalid_timing=repr(value))
    return saved


def require_timing(cycle: BenchmarkCycle) -> None:
    if (
        not math.isfinite(cycle.latency_ms)
        or cycle.latency_ms <= 0
        or any(
            not math.isfinite(row.latency_ms) or row.latency_ms < 0
            for row in cycle.rows
        )
    ):
        raise ValueError("Finite positive cohort and nonnegative row timings required")


def study(model: Path, output: Path, batch_size: int, *, screen: bool = False) -> bool:
    if type(batch_size) is not int or batch_size not in (1, 8):
        raise ValueError("Final-source confirmation requires batch size1 or8")
    if screen and batch_size != 8:
        raise ValueError("First8 numerical screen requires batch size8")
    if (
        output.exists()
        or output.with_suffix(".cycles.jsonl").exists()
        or any(output.parent.glob(output.stem + ".sources-*.json"))
    ):
        raise FileExistsError("Preserve existing receipts; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = QualityGate()
    plan = BenchmarkProtocol(repeats=5, warmup=1, quality_gate=gate)
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "scope": "Final-source CPU prefix; fixed77 development, exposure unknown",
        "completed": False,
        "quality_passed": False,
        "performance_passed": False,
        "accepted": False,
        "screen_passed": False,
        "mode": "screen" if screen else "confirmation",
        "dispatch_observed": False,
        "threads_restored": False,
        "phase": "admission",
        "batch_size": batch_size,
        "quality_gate": asdict(gate),
        "performance_gate": {
            "paired_median_minimum": 1.02,
            "paired_p05_minimum_exclusive": 1.0,
            "aggregate_throughput_ratio_minimum_exclusive": 1.0,
            "candidate_cohort_p95_maximum_ratio": 1.0,
        },
        "protocol": {
            "device": "cpu",
            "dtype": "float32",
            "threads": 8,
            "max_length": 512,
            "token_cache_size": 0,
            "prefix_length": 25,
            "screens": 0 if screen else 1,
            "warmup": 1,
            "pairs": 2 if screen else 5,
            "order": "warm AB; ordinary ABBA"
            if screen
            else "AB,BA,AB,BA,AB; screens/warmups AB",
            "attention": "sdpa; default process policies and native HF flags unchanged",
            "timing": "whole cohort; setup separate; fresh cache inside calls",
            "memory": "Logical prefix bytes; cumulative RSS; no memory ceiling",
            "promotion": "Both final-source B8 and B1 must pass all original gates",
        },
        "cycles": rows,
    }

    def write() -> None:
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    with output.open("x") as initial:
        initial.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    previous_threads = torch.get_num_threads()
    restoration_attempted = False

    def restore_threads() -> None:
        nonlocal restoration_attempted
        restoration_attempted = True
        try:
            torch.set_num_threads(previous_threads)
            if torch.get_num_threads() != previous_threads:
                raise ValueError("Original process thread count was not restored")
        except BaseException as error:
            payload["thread_restoration_error"] = f"{type(error).__name__}: {error}"
            raise
        payload["threads_restored"] = True

    try:
        prefix = admit_history()
        inputs = {DATA: DATA_SHA, REFERENCE: REFERENCE_SHA}
        inputs.update(
            {model / name: checksum for name, checksum in MODEL_FILES.items()}
        )
        verify_files(inputs)
        sources = freeze_sources(source_paths(), output)
        policy = process_policy()
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
            historical_sha256=HISTORY,
            asset_sha256={str(path): checksum for path, checksum in inputs.items()},
            sdpa_flags=flags,
            process_policy=policy,
            runtime={
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_git_version": torch.version.git_version,
                "torch_build": torch.__config__.show(),
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
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
                "prior_threads": previous_threads,
                "interop_threads": torch.get_num_interop_threads(),
            },
            source_scope=(
                "On-disk Python/dependency source and package/build observations; "
                "not independent attestation of every native binary/imported object"
            ),
            phase="loading",
        )
        write()
        if not all(value is True for value in flags.values()):
            raise ValueError(
                "Native masked-GQA flags differ from original admitted context"
            )
        torch.set_num_threads(8)
        cases = tuple(
            case_from_dict(decode_json(line)) for line in DATA.read_text().splitlines()
        )
        if len(cases) != 77 or _input_digest(cases) != INPUT_SHA:
            raise ValueError("All original ordered77 requests required")
        base = KaiScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        if base.model_id != MODEL_ID:
            raise ValueError("Released model identity differs")
        reference = read_reference(REFERENCE, cases, base, dataset_sha256=DATA_SHA)
        encoded = tuple(base.encode_request(case.request) for case in cases)
        started = time.perf_counter_ns()
        candidate = PrefixScorer(base, prefix)
        build_ms = (time.perf_counter_ns() - started) / 1e6
        if not all(candidate._admits(row) for row in encoded):
            raise ValueError("Every original input must admit the frozen exact prefix")
        if screen:
            cases = cases[:8]
            reference = BenchmarkCycle(0, reference.rows[:8], 0, 0)
            encoded = encoded[:8]
        payload.update(
            model_id=MODEL_ID,
            dataset_sha256=DATA_SHA,
            reference_sha256=REFERENCE_SHA,
            input_sha256=_input_digest(cases),
            full_input_sha256=INPUT_SHA,
            case_ids=[case.id for case in cases],
            prefix_tokens=list(prefix),
            token_lengths=[len(row.ids) for row in encoded],
            prefix_build_ms=build_ms,
            prefix_state_bytes=candidate.prefix_state_bytes,
            phase="cycles",
        )
        write()
        engines = (
            BenchmarkEngine("cpu-full-prefill", Model(base, MODEL_ID), batch_size),
            BenchmarkEngine(
                "cpu-exact-prefix25", Model(candidate, MODEL_ID), batch_size
            ),
        )
        measured: tuple[list[BenchmarkCycle], list[BenchmarkCycle]] = ([], [])

        def verify_final() -> None:
            verify_sources(sources)
            verify_files(inputs)
            verify_files({RESULTS / name: sha for name, sha in HISTORY.items()})
            if process_policy() != policy or torch.get_num_threads() != 8:
                raise ValueError(
                    "Owned process attention, precision or threads changed"
                )

        with output.with_suffix(".cycles.jsonl").open("x") as journal:

            def run(index: int, stage: str, repeat: int) -> BenchmarkCycle:
                verify_sources(sources)
                verify_files({RESULTS / name: sha for name, sha in HISTORY.items()})
                if process_policy() != policy or torch.get_num_threads() != 8:
                    raise ValueError(
                        "Owned process attention, precision or threads changed"
                    )
                before = process_diagnostic()
                cycle = _run(engines[index], cases)
                successful = {
                    item.id
                    for item in cycle.rows
                    if item.status == "ok" and item.prediction is not None
                }
                row: dict[str, object] = {
                    "arm": engines[index].name,
                    "stage": stage,
                    "repeat": repeat,
                    "cycle": json_cycle(cycle),
                    "unknown_case_ids": [
                        case.id for case in cases if case.id not in successful
                    ],
                    "before": asdict(before),
                    "after": asdict(process_diagnostic()),
                }
                rows.append(row)
                try:
                    row["reference_quality"] = asdict(
                        compare_cycles((reference,), (cycle,), gate)
                    )
                finally:
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    journal.flush()
                    write()
                require_timing(cycle)
                verify_sources(sources)
                return cycle

            if screen:
                warmup = (run(0, "warmup", 0), run(1, "warmup", 0))
                ordinary = (
                    run(0, "ordinary", 0),
                    run(1, "ordinary", 0),
                    run(1, "ordinary", 1),
                    run(0, "ordinary", 1),
                )
                quality = compare_cycles((reference,) * 6, warmup + ordinary, gate)
                paired = compare_cycles(
                    (ordinary[0], ordinary[3]), (ordinary[1], ordinary[2]), gate
                )
                verify_final()
                restore_threads()
                passed = quality.passed and paired.passed
                payload.update(
                    completed=True,
                    quality_passed=passed,
                    screen_passed=passed,
                    phase="complete",
                    reference_quality=asdict(quality),
                    paired_quality=asdict(paired),
                    source_verified_after=True,
                    asset_verified_after=True,
                    limits="First8 numeric screen; no dispatch or speed claim",
                )
                write()
                return passed
            screens = (run(0, "screen", 0), run(1, "screen", 0))
            if not compare_cycles((reference,) * 2, screens, gate).passed:
                verify_final()
                restore_threads()
                payload.update(completed=True, phase="rejected screen")
                write()
                return False
            payload["screen_passed"] = True
            warmup = (run(0, "warmup", 0), run(1, "warmup", 0))
            for repeat in range(5):
                for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                    measured[index].append(run(index, "measured", repeat))
        cycles = screens + warmup + tuple(measured[0]) + tuple(measured[1])
        quality = compare_cycles((reference,) * 14, cycles, gate)
        report = BenchmarkReport(
            INPUT_SHA,
            cases,
            plan,
            *engines,
            tuple(measured[0]),
            tuple(measured[1]),
            compare_cycles(*measured, gate),
            (warmup[0],),
            (warmup[1],),
        )
        observed = performance(*measured)
        verify_final()
        restore_threads()
        passed = report.passed and quality.passed
        payload.update(
            completed=True,
            quality_passed=passed,
            performance_passed=speed_passed(observed),
            accepted=passed and speed_passed(observed),
            phase="complete",
            report=report.to_dict(),
            reference_quality=asdict(quality),
            performance_observed=observed,
            source_verified_after=True,
            asset_verified_after=True,
        )
        write()
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
        if not restoration_attempted:
            try:
                restore_threads()
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
