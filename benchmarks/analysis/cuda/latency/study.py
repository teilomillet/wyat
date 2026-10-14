"""All outputs gated; synchronized batch intervals exclude telemetry and I/O."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn
from transformers import masking_utils
from transformers.integrations import sdpa_attention
from transformers.models.qwen3 import modeling_qwen3

from benchmarks.analysis.cuda.attention.scorer import efficient_scorer
from benchmarks.analysis.cuda.attention.study import SOURCES as ATTENTION_SOURCES
from benchmarks.analysis.cuda.attention.study import admit_sources as admit_attention
from benchmarks.analysis.cuda.attention.study import check_cycle
from benchmarks.analysis.cuda.attention.trace import (
    efficient_dispatch_gate,
    profile_batch,
)
from benchmarks.analysis.cuda.manifest import (
    DATA,
    DATA_SHA,
    INPUT_SHA,
    MODEL_ID,
    REFERENCE,
    ROOT,
    asset_paths,
    digest,
)
from benchmarks.analysis.cuda.scorer import FP32Scorer, ieee_precision, require_fp32
from benchmarks.analysis.cuda.study import (
    admit,
    admit_control,
    control_cycle,
    occupancy,
)
from benchmarks.analysis.evidence import freeze_sources
from benchmarks.analysis.reference import read_reference
from wyat import Model
from wyat.backends.wyat.artifact import load_artifact
from wyat.backends.wyat.encoding import EncodedRequest, WyatEncoder
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.core import Request, Scorer, decision_from_logits
from wyat.evaluation import (
    EvaluationCase,
    EvaluationRow,
    _input_digest,
    case_from_dict,
    percentile,
)
from wyat.wire import decode_json, read_object

type Mode = Literal["confirm", "timing"]
SOURCES = tuple(
    Path(name)
    for name in (
        "benchmarks/analysis/cuda/latency/__init__.py",
        "benchmarks/analysis/cuda/latency/study.py",
        "benchmarks/analysis/cuda/latency/run.py",
        "tests/analysis/test_cuda_latency.py",
        "docs/analysis/latency.md",
    )
)
MANIFEST_SHA = "f7d9d7a4755a8df37e7e55d577489a1f9597760a68928b2a8213a0d19d3c6866"
ATTENTION_DECLARATION = Path(
    "benchmarks/analysis/results/cuda-attention-source-declaration-v2-20261005.json"
)
ATTENTION_SHA = "589a45fe5f941d13311ec2668889186fd14557cc173071c0de3e0cea333dfe8f"
SCREEN_PINS = {
    1: (
        "e2149d7d4bb6c51827b68798c28d016387be89354b153ce521b10247d5bca562",
        "2ed8a63c92f7cc41dff3cf0ce698441c057bfaabc991af3949a685a0b457ac69",
    ),
    8: (
        "dda922388386aa04a8deb072fc9da782afd0662a25865dedd64d5bbfafbf0e49",
        "fd4c0d48c6e876fda60e7ff51e14b02818c86f26c4d4afa16943c531de4a6a2c",
    ),
}
ARCHIVE_SHA = "686a18493cecf0930c788160cb7032c5f559575ec5c725f780a707449d38a97a"
CONTROL_SHA = "d528a44e19ceb6097ea81081d7e57fd97454a26777e45f3a284cb52d044c5167"
CONTROL_WALL_SHA = "cb00f5f065a04e5c598da04cb3e57a7b75bbaedeb1407dc6ebbb0dd806f4dc8e"


@dataclass(slots=True)
class ReferenceInput:
    """Validate the exact tokenizer contract before loading checkpoint tensors."""

    model_id: str
    encoder: WyatEncoder

    def encode_request(self, request: Request) -> EncodedRequest:
        return self.encoder.encode_request(request)


def source_paths() -> dict[str, Path]:
    paths = {
        str(path): ROOT / path
        for path in (*asset_paths(), *ATTENTION_SOURCES, *SOURCES)
        if path.suffix in (".py", ".md")
    }
    paths.update(
        external_qwen3=Path(modeling_qwen3.__file__),
        external_sdpa=Path(sdpa_attention.__file__),
        external_masking=Path(masking_utils.__file__),
    )
    return paths


def admit_sources(path: Path, sha256: str) -> dict[str, str]:
    if digest(path) != sha256:
        raise ValueError("Latency source declaration changed")
    saved = cast(dict[str, str], read_object(path))
    if set(saved) != {str(item) for item in SOURCES} or any(
        digest(ROOT / name) != value for name, value in saved.items()
    ):
        raise ValueError("Every exact latency source is required unchanged")
    return saved


def read_cycle(value: object, cases: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
    saved = cast(dict[str, object], value)
    raw = cast(list[dict[str, object]], saved["rows"])
    if len(raw) != len(cases):
        raise ValueError("History cycle must cover its entire canonical cohort")
    rows: list[EvaluationRow] = []
    for item, case in zip(raw, cases, strict=True):
        prediction = cast(dict[str, object], item["prediction"])
        decision = decision_from_logits(
            case.request, cast(list[float], prediction["scores"]), MODEL_ID
        )
        correct = decision.selected in case.expected
        if (
            item["id"] != case.id
            or item["expected"] != list(case.expected)
            or item["status"] != "ok"
            or item["error"] is not None
            or prediction != decision.to_dict()
            or item["correct"] != correct
        ):
            raise ValueError("History output is failed, reordered or inconsistent")
        rows.append(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                decision,
                correct,
                float(cast(float, item["latency_ms"])),
                None,
            )
        )
    return BenchmarkCycle(
        float(cast(float, saved["latency_ms"])),
        tuple(rows),
        int(cast(int, saved["peak_rss_before_bytes"])),
        int(cast(int, saved["peak_rss_after_bytes"])),
    )


def require_parent(path: Path) -> dict[str, object]:
    wall = read_object(path.with_suffix(".wall.json"))
    if (
        wall.get("exit_code") != 0
        or wall.get("completed") is not True
        or wall.get("timed_out") is not False
        or wall.get("timeout_seconds") != 180
    ):
        raise ValueError("History parent failed terminal cleanup or its owned cap")
    return cast(dict[str, object], wall)


def admit_screens(
    reference: BenchmarkCycle, cuda: BenchmarkCycle, cases: tuple[EvaluationCase, ...]
) -> dict[str, str]:
    pins = {str(ATTENTION_DECLARATION): ATTENTION_SHA}
    admit_attention(ROOT / ATTENTION_DECLARATION, ATTENTION_SHA)
    for batch, (main_sha, wall_sha) in SCREEN_PINS.items():
        path = (
            ROOT / f"benchmarks/analysis/results/cuda-attention-observer-b{batch}.json"
        )
        archive = path.with_suffix(".sources-686a18493cecf093.json")
        for item, expected in (
            (path, main_sha),
            (path.with_suffix(".wall.json"), wall_sha),
            (archive, ARCHIVE_SHA),
        ):
            if digest(item) != expected:
                raise ValueError("Reviewed attention screen history changed")
            pins[str(item.relative_to(ROOT))] = expected
        saved = read_object(path)
        require_parent(path)
        current = source_paths()
        if any(
            name not in current or digest(current[name]) != sha
            for name, sha in cast(dict[str, str], saved["source_sha256"]).items()
        ):
            raise ValueError("Consumed screen sources, including external HF, changed")
        if (
            saved.get("completed") is not True
            or saved.get("quality_passed") is not True
            or saved.get("dispatch_passed") is not True
            or saved.get("accepted") is not False
            or saved.get("batch_size") != batch
            or saved.get("source_declaration_sha256") != ATTENTION_SHA
            or saved.get("case_ids") != [case.id for case in cases[:8]]
            or any(cast(dict[str, list[str]], saved["unknown_case_ids"]).values())
        ):
            raise ValueError("Reviewed attention screen is incomplete")
        for arm in ("math", "efficient"):
            for phase, cohort in (
                ("cycle", cases[:8]),
                ("profile_cycle", cases[:batch]),
            ):
                cycle = read_cycle(saved[f"{arm}_{phase}"], cohort)
                passed, _ = check_cycle(
                    cycle,
                    replace(reference, rows=reference.rows[: len(cohort)]),
                    replace(cuda, rows=cuda.rows[: len(cohort)]),
                )
                if not passed:
                    raise ValueError("Reviewed screen fails unchanged CPU/CUDA gates")
        if not efficient_dispatch_gate(
            cast(dict[str, object], saved["efficient_profile"]),
            batch_size=batch,
            length=172 if batch == 1 else 192,
        ):
            raise ValueError("Reviewed representative screen dispatch is unsupported")
    return pins


def admit_confirmation(
    path: Path | None,
    sha256: str | None,
    *,
    wall_sha256: str | None,
    batch_size: int,
    source_hashes: dict[str, str],
    reference: BenchmarkCycle,
    cuda: BenchmarkCycle,
    cases: tuple[EvaluationCase, ...],
) -> dict[str, str]:
    if (
        path is None
        or sha256 is None
        or wall_sha256 is None
        or digest(path) != sha256
        or digest(path.with_suffix(".wall.json")) != wall_sha256
    ):
        raise ValueError(
            "Timing requires an independently reviewed confirmation identity"
        )
    saved = read_object(path)
    require_parent(path)
    if (
        saved.get("mode") != "confirm"
        or saved.get("batch_size") != batch_size
        or saved.get("completed") is not True
        or saved.get("quality_passed") is not True
        or saved.get("dispatch_passed") is not True
        or saved.get("accepted") is not False
        or saved.get("case_ids") != [case.id for case in cases]
        or saved.get("input_sha256") != INPUT_SHA
        or cast(dict[str, str], saved["new_source_hashes"]) != source_hashes
        or any(cast(dict[str, list[str]], saved["unknown_case_ids"]).values())
    ):
        raise ValueError(
            "Confirmation is incomplete or differs from the timing contract"
        )
    current = source_paths()
    hashes = cast(dict[str, str], saved["source_sha256"])
    if set(hashes) != set(current) or any(
        digest(current[name]) != sha for name, sha in hashes.items()
    ):
        raise ValueError("Confirmation source identity differs from current execution")
    archive = cast(dict[str, str], saved["source_archive"])
    archive_path = path.with_suffix(f".sources-{archive['sha256'][:16]}.json")
    if digest(archive_path) != archive["sha256"]:
        raise ValueError("Confirmation source archive changed")
    cycles = cast(list[dict[str, object]], saved["cycles"])
    if [item.get("key") for item in cycles] != ["confirm_math", "confirm_efficient"]:
        raise ValueError("Confirmation must contain both complete ordinary arms")
    paired: list[BenchmarkCycle] = []
    for item in cycles:
        cycle = read_cycle(item["cycle"], cases)
        if not check_cycle(cycle, reference, cuda)[0]:
            raise ValueError("Confirmation fails unchanged CPU/CUDA numerical gates")
        paired.append(cycle)
    if not compare_cycles((paired[0],), (paired[1],), QualityGate()).passed:
        raise ValueError("Confirmation arms fail paired quality")
    profiled = cases[:batch_size]
    for arm in ("math", "efficient"):
        cycle = read_cycle(saved[f"{arm}_profile_cycle"], profiled)
        if not check_cycle(
            cycle,
            replace(reference, rows=reference.rows[:batch_size]),
            replace(cuda, rows=cuda.rows[:batch_size]),
        )[0]:
            raise ValueError("Confirmation profile output fails unchanged quality")
    if not efficient_dispatch_gate(
        cast(dict[str, object], saved["efficient_profile"]),
        batch_size=batch_size,
        length=172 if batch_size == 1 else 192,
    ):
        raise ValueError("Confirmation profile is not the reviewed efficient dispatch")
    return {
        str(path): sha256,
        str(path.with_suffix(".wall.json")): wall_sha256,
        str(archive_path): archive["sha256"],
    }


def cycle_data(cycle: BenchmarkCycle) -> dict[str, object]:
    """Keep unsupported timing evidence JSON-safe without losing returned rows."""
    saved = cycle.to_dict()
    invalid: dict[str, str] = {}
    if not math.isfinite(cycle.latency_ms):
        invalid["latency_ms"] = repr(cycle.latency_ms)
        saved["latency_ms"] = None
    for row in cast(list[dict[str, object]], saved["rows"]):
        value = cast(float, row["latency_ms"])
        if not math.isfinite(value):
            invalid[f"row:{row['id']}"] = repr(value)
            row["latency_ms"] = None
    if invalid:
        saved["invalid_timing"] = invalid
    return saved


def collect(
    engine: BenchmarkEngine,
    cases: tuple[EvaluationCase, ...],
    reference: BenchmarkCycle,
    cuda: BenchmarkCycle,
    *,
    key: str,
    payload: dict[str, object],
    write: Callable[[], None],
    verify: Callable[[], None] | None = None,
) -> BenchmarkCycle:
    """Ownership and persistence surround, never enter, canonical batch timers."""
    started = time.perf_counter_ns()
    chunks: list[BenchmarkCycle] = []
    observations: list[dict[str, object]] = []
    record: dict[str, object] = {
        "key": key,
        "chunks": [],
        "ownership": observations,
        "drift": [],
    }
    cast(list[dict[str, object]], payload["cycles"]).append(record)
    unknown = cast(dict[str, list[str]], payload["unknown_case_ids"])
    for offset in range(0, len(cases), engine.batch_size):
        if verify is not None:
            verify()
        observations.append(
            {
                "utc": datetime.now(UTC).isoformat(),
                "offset": offset,
                "snapshot": occupancy(own_pid=os.getpid()),
            }
        )
        cohort = cases[offset : offset + engine.batch_size]
        chunk = _run(engine, cohort)
        chunks.append(chunk)
        cast(list[dict[str, object]], record["chunks"]).append(cycle_data(chunk))
        unknown[key] = [
            case.id
            for case in cases
            if case.id not in {row.id for item in chunks for row in item.rows}
        ]
        try:
            passed, drift = check_cycle(
                chunk,
                replace(reference, rows=reference.rows[offset : offset + len(cohort)]),
                replace(cuda, rows=cuda.rows[offset : offset + len(cohort)]),
            )
            cast(list[dict[str, object]], record["drift"]).append(drift)
        finally:
            # One retained update per returned batch, also on gate exceptions.
            write()
        if chunk.failed or not passed:
            raise ValueError(f"{key} batch failed unchanged quality gates")
        if (
            not math.isfinite(chunk.latency_ms)
            or chunk.latency_ms <= 0
            or any(
                not math.isfinite(row.latency_ms) or row.latency_ms < 0
                for row in chunk.rows
            )
        ):
            raise ValueError("Canonical batch interval must be finite and positive")
    cycle = BenchmarkCycle(
        math.fsum(item.latency_ms for item in chunks),
        tuple(row for item in chunks for row in item.rows),
        chunks[0].peak_rss_before_bytes,
        chunks[-1].peak_rss_after_bytes,
    )
    record.update(
        cycle=cycle.to_dict(), cohort_wall_ms=(time.perf_counter_ns() - started) / 1e6
    )
    write()
    return cycle


def performance(
    baseline: tuple[BenchmarkCycle, ...], candidate: tuple[BenchmarkCycle, ...]
) -> dict[str, object]:
    if len(baseline) != 6 or len(candidate) != 6:
        raise ValueError("Six complete paired cohorts are required")
    left, right = (
        [item.latency_ms for item in baseline],
        [item.latency_ms for item in candidate],
    )
    if any(not math.isfinite(value) or value <= 0 for value in (*left, *right)):
        raise ValueError("Paired intervals must be finite and positive")
    ratios = [a / b for a, b in zip(left, right, strict=True)]
    median, low = percentile(ratios, 0.5), percentile(ratios, 0.05)
    aggregate = math.fsum(left) / math.fsum(right)
    tail = percentile(right, 0.95) / percentile(left, 0.95)
    return {
        "paired_ratios": ratios,
        "median": median,
        "p05": low,
        "aggregate": aggregate,
        "cohort_p95_ratio": tail,
        "passed": median >= 1.05 and low > 1 and aggregate > 1 and tail <= 1,
    }


def study(
    model: Path,
    output: Path,
    *,
    manifest: Path,
    manifest_sha256: str,
    sources: Path,
    sources_sha256: str,
    control: Path,
    control_sha256: str,
    mode: Mode,
    batch_size: int,
    confirmation: Path | None = None,
    confirmation_sha256: str | None = None,
    confirmation_wall_sha256: str | None = None,
) -> bool:
    if (
        mode not in ("confirm", "timing")
        or type(batch_size) is not int
        or batch_size not in (1, 8)
    ):
        raise ValueError("Use a declared confirmation/timing mode and batch1/8")
    if mode == "confirm" and (
        confirmation is not None
        or confirmation_sha256 is not None
        or confirmation_wall_sha256 is not None
    ):
        raise ValueError("Confirmation must not consume another confirmation")
    started = time.perf_counter_ns()
    payload: dict[str, object] = {
        "mode": mode,
        "batch_size": batch_size,
        "completed": False,
        "quality_passed": False,
        "dispatch_passed": False,
        "dispatch_evidence_scope": (
            "prior representative confirmation profile; timed calls unprofiled"
            if mode == "timing"
            else (
                "current representative first-batch profile; "
                "ordinary cohorts unprofiled"
            )
        ),
        "performance_passed": False,
        "accepted": False,
        "cycles": [],
        "scope": "Pinned77 BANKING77 training-source menus; exposure unknown",
        "gate": asdict(QualityGate()),
        "input_sha256": INPUT_SHA,
        "timing_scope": (
            "Sum synchronized _run(batch).latency_ms including row bookkeeping; "
            "telemetry-paced; excludes ownership/I/O/quality/gaps"
        ),
        "limitations": [
            "No online capacity/tail or universal speed claim",
            "Instantaneous ownership is not a reservation",
            "Unstarted outcomes unknown; profiler is diagnostic only",
            "FP32 tensors do not imply IEEE efficient-attention arithmetic",
        ],
    }

    def write() -> None:
        output.write_text(json.dumps(payload, allow_nan=False) + "\n")

    # The worker owns this new receipt; never replace another trial's history.
    with output.open("x") as stream:
        stream.write(json.dumps(payload, allow_nan=False) + "\n")
    try:
        if manifest_sha256 != MANIFEST_SHA or digest(manifest) != MANIFEST_SHA:
            raise ValueError(
                "Only the original immutable 44-asset manifest is admitted"
            )
        admission = admit(manifest)
        new_hashes = admit_sources(sources, sources_sha256)
        admit_attention(ROOT / ATTENTION_DECLARATION, ATTENTION_SHA)
        if (
            control_sha256 != CONTROL_SHA
            or digest(control.with_suffix(".wall.json")) != CONTROL_WALL_SHA
        ):
            raise ValueError(
                "Only the reviewed original CUDA control/main wall is admitted"
            )
        pinned = admit_control(
            control, control_sha256, admission, batch_size=batch_size
        )
        payload.update(
            admission=admission,
            control=pinned,
            new_source_hashes=new_hashes,
            source_declaration_sha256=sources_sha256,
        )
        if any(
            (
                sdpa_attention._is_torch_mps_available,
                sdpa_attention._is_torch_xpu_available,
                sdpa_attention._is_torch_npu_available,
            )
        ):
            raise ValueError(
                "This experiment requires frozen Linux CUDA HF availability"
            )
        frozen = freeze_sources(source_paths(), output)
        payload.update(source_sha256=frozen.hashes(), source_archive=frozen.archive())
        artifact = load_artifact(model, 512)
        if artifact.model_id != MODEL_ID:
            raise ValueError("Checkpoint identity differs from the frozen reference")
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in (ROOT / DATA).read_text().splitlines()
        )
        if len(cases) != 77 or _input_digest(cases) != INPUT_SHA:
            raise ValueError("Every exact canonical case must remain ordered")
        payload.update(
            case_ids=[case.id for case in cases],
            expected_ordinary_rows=154 if mode == "confirm" else 1078,
            expected_profile_rows=2 * batch_size if mode == "confirm" else 0,
        )
        schedule = (
            [("confirm", None, ("math", "efficient"))]
            if mode == "confirm"
            else [("warmup", None, ("math", "efficient"))]
            + [
                (
                    "pair",
                    repeat,
                    ("math", "efficient") if repeat % 2 == 0 else ("efficient", "math"),
                )
                for repeat in range(6)
            ]
        )
        keys = [
            f"{phase}{'' if repeat is None else repeat}_{arm}"
            for phase, repeat, order in schedule
            for arm in order
        ]
        payload["unknown_case_ids"] = {key: [case.id for case in cases] for key in keys}
        if mode == "confirm":
            cast(dict[str, list[str]], payload["unknown_case_ids"]).update(
                {
                    f"{arm}_profile": [case.id for case in cases[:batch_size]]
                    for arm in ("math", "efficient")
                }
            )
        write()
        inputs = ReferenceInput(
            artifact.model_id,
            WyatEncoder.from_artifact(artifact, max_length=512, token_cache_size=0),
        )
        reference = read_reference(
            ROOT / REFERENCE, cases, inputs, dataset_sha256=DATA_SHA
        )
        cuda = control_cycle(control, cases)
        if not compare_cycles((reference,), (cuda,), QualityGate()).passed:
            raise ValueError("Reviewed CUDA control fails the original CPU reference")
        history = admit_screens(reference, cuda, cases)
        if mode == "timing":
            confirmed = admit_confirmation(
                confirmation,
                confirmation_sha256,
                wall_sha256=confirmation_wall_sha256,
                batch_size=batch_size,
                source_hashes=new_hashes,
                reference=reference,
                cuda=cuda,
                cases=cases,
            )
            history.update(confirmed)
            payload.update(confirmation=confirmed, dispatch_passed=True)
        payload["history_sha256"] = history

        def verify() -> None:
            if any(digest(Path(item.path)) != item.sha256 for item in frozen.files):
                raise ValueError("Consumed source changed between inference batches")

        payload["occupancy_before_load"] = occupancy()
        write()
        torch.set_num_threads(8)
        base = WyatScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        read_reference(ROOT / REFERENCE, cases, base, dataset_sha256=DATA_SHA)
        payload["occupancy_before_allocation"] = occupancy()
        with ieee_precision():
            base.device = torch.device("cuda:0")
            cast(nn.Module, base.backbone).to(base.device)
            base.head.to(base.device)
            require_fp32(base)
            scorers: dict[str, Scorer] = {
                "math": FP32Scorer(base),
                "efficient": efficient_scorer(base),
            }
            engines = {
                arm: BenchmarkEngine(
                    arm,
                    Model(scorer, model_id=MODEL_ID),
                    batch_size=batch_size,
                    synchronize=torch.cuda.synchronize,
                )
                for arm, scorer in scorers.items()
            }
            measured: dict[str, list[BenchmarkCycle]] = {"math": [], "efficient": []}
            warm: dict[str, list[BenchmarkCycle]] = {"math": [], "efficient": []}
            for phase, repeat, order in schedule:
                current: dict[str, BenchmarkCycle] = {}
                for arm in order:
                    key = f"{phase}{'' if repeat is None else repeat}_{arm}"
                    payload["phase"] = key
                    write()
                    current[arm] = collect(
                        engines[arm],
                        cases,
                        reference,
                        cuda,
                        key=key,
                        payload=payload,
                        write=write,
                        verify=verify,
                    )
                    if phase == "warmup":
                        warm[arm].append(current[arm])
                    elif phase == "pair":
                        measured[arm].append(current[arm])
                drift = compare_cycles(
                    (current["math"],), (current["efficient"],), QualityGate()
                )
                payload[f"{phase}{'' if repeat is None else repeat}_paired_drift"] = (
                    asdict(drift)
                )
                write()
                if not drift.passed:
                    raise ValueError("Paired arms fail unchanged quality gates")
            if mode == "confirm":
                profiled = cases[:batch_size]
                for arm in ("math", "efficient"):
                    payload["phase"] = f"{arm}_profile"
                    occupancy(own_pid=os.getpid())
                    write()
                    cycle, observed = profile_batch(scorers[arm], profiled, name=arm)
                    payload[f"{arm}_profile_cycle"] = cycle_data(cycle)
                    payload[f"{arm}_profile"] = observed
                    cast(dict[str, list[str]], payload["unknown_case_ids"])[
                        f"{arm}_profile"
                    ] = [
                        case.id
                        for case in profiled
                        if case.id not in {row.id for row in cycle.rows}
                    ]
                    passed, profile_drift = check_cycle(
                        cycle,
                        replace(reference, rows=reference.rows[:batch_size]),
                        replace(cuda, rows=cuda.rows[:batch_size]),
                    )
                    payload[f"{arm}_profile_drift"] = profile_drift
                    write()
                    if not passed:
                        raise ValueError("Representative profile quality failed")
                payload["dispatch_passed"] = efficient_dispatch_gate(
                    cast(dict[str, object], payload["efficient_profile"]),
                    batch_size=batch_size,
                    length=max(
                        len(base.encode_request(case.request).ids) for case in profiled
                    ),
                )
                if payload["dispatch_passed"] is not True:
                    raise ValueError("Representative efficient dispatch is unsupported")
            else:
                left, right = tuple(measured["math"]), tuple(measured["efficient"])
                report = BenchmarkReport(
                    INPUT_SHA,
                    cases,
                    BenchmarkProtocol(repeats=6, warmup=1),
                    engines["math"],
                    engines["efficient"],
                    left,
                    right,
                    compare_cycles(left, right, QualityGate()),
                    tuple(warm["math"]),
                    tuple(warm["efficient"]),
                ).to_dict()
                cast(dict[str, object], report["protocol"])["timing"] = payload[
                    "timing_scope"
                ]
                payload.update(report=report, performance=performance(left, right))
                payload["performance_passed"] = cast(
                    dict[str, object], payload["performance"]
                )["passed"]
            payload["occupancy_after"] = occupancy(own_pid=os.getpid())
            payload["cuda_memory"] = {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        admit_attention(ROOT / ATTENTION_DECLARATION, ATTENTION_SHA)
        if (
            admit(manifest) != admission
            or admit_sources(sources, sources_sha256) != new_hashes
            or any(digest(ROOT / path) != sha for path, sha in history.items())
            or digest(control) != pinned["sha256"]
            or digest(Path(cast(str, pinned["wall_path"]))) != pinned["wall_sha256"]
            or any(digest(Path(item.path)) != item.sha256 for item in frozen.files)
        ):
            raise ValueError(
                "Consumed source/reference/history changed during execution"
            )
        payload.update(
            completed=True,
            quality_passed=True,
            phase="completed",
            worker_body_wall_ms=(time.perf_counter_ns() - started) / 1e6,
            torch_build=torch.__config__.show(),
        )
        write()
        return mode == "confirm" or payload["performance_passed"] is True
    except Exception as error:
        payload.update(
            completed=False,
            quality_passed=False,
            dispatch_passed=False,
            performance_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        return False
