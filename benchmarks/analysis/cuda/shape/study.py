"""Twelve strictly gated outcomes and four independent intrusive Chrome traces."""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn
from torch._C._autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.integrations import sdpa_attention

from benchmarks.analysis.cuda.attention.scorer import efficient_scorer
from benchmarks.analysis.cuda.attention.study import check_cycle
from benchmarks.analysis.cuda.attention.trace import Dispatch, efficient_dispatch_gate
from benchmarks.analysis.cuda.latency import study as latency
from benchmarks.analysis.cuda.manifest import (
    DATA,
    DATA_SHA,
    INPUT_SHA,
    MODEL_ID,
    REFERENCE,
    ROOT,
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
from wyat.backends.kai.artifact import load_artifact
from wyat.backends.kai.encoding import KaiEncoder
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    QualityGate,
    _run,
    compare_cycles,
)
from wyat.core import Scorer
from wyat.evaluation import EvaluationCase, _input_digest, case_from_dict
from wyat.wire import decode_json, read_object

type Arm = Literal["math", "efficient"]
type Phase = Literal["warmup", "ordinary", "profile"]
SOURCES = tuple(
    Path(name)
    for name in (
        "benchmarks/analysis/cuda/shape/__init__.py",
        "benchmarks/analysis/cuda/shape/study.py",
        "benchmarks/analysis/cuda/shape/run.py",
        "tests/analysis/test_cuda_shape.py",
        "docs/analysis/shape.md",
    )
)
CASE_SELECTION = (
    (7, "banking77-train-03357", 192),
    (18, "banking77-train-00566", 197),
)
HISTORY_PINS = {
    "cuda-latency-source-declaration-20261005.json": (
        "a26d5be45598c29e9fe6f1a0d0e620c22d0101f060eda730274d404498c7a0a4"
    ),
    "cuda-latency-confirm-b1.json": (
        "c41d4d08665c9089f1e279814d6caf47a5dbe1c4fe31df2d5ffacd9763a7ed4b"
    ),
    "cuda-latency-confirm-b1.wall.json": (
        "29d7fe3a350713ad2698a5205f4df385eb57f7f40be3b2c3fb2c23a91a22d350"
    ),
    "cuda-latency-timing-b1.json": (
        "21ae0cc5b07571969ebcc26315fe0cc49533b43b3de92c4fbbc8c2741edb3fb6"
    ),
    "cuda-latency-timing-b1.wall.json": (
        "dd01e353e74061afb4b353cfcc4cbc5179cddae222a97c7ff284ebd63d06ca49"
    ),
    "cuda-latency-shape-review.json": (
        "0d3f003aec12460c47b3b97c59ca3e948b0824d096cb79ee01e448017bbf5f68"
    ),
    "cuda-latency-timing-b1-review.json": (
        "790c0e5f6efad23e4db8594cf068199320c43a40864719168901f61e89178d58"
    ),
}
RESULTS = Path("benchmarks/analysis/results")


@dataclass(frozen=True)
class Probe:
    phase: Phase
    offset: int
    arm: Arm

    @property
    def key(self) -> str:
        return f"{self.phase}:{self.offset}:{self.arm}"


def probe_schedule() -> tuple[Probe, ...]:
    phases: tuple[Phase, ...] = ("warmup", "ordinary", "profile")
    order: tuple[tuple[int, Arm], ...] = (
        (7, "math"),
        (7, "efficient"),
        (18, "efficient"),
        (18, "math"),
    )
    return tuple(Probe(phase, offset, arm) for phase in phases for offset, arm in order)


def admit_sources(path: Path, sha256: str) -> dict[str, str]:
    saved = cast(dict[str, str], read_object(path))
    if (
        digest(path) != sha256
        or set(saved) != {str(item) for item in SOURCES}
        or any(digest(ROOT / name) != sha for name, sha in saved.items())
    ):
        raise ValueError("Every exact five-file shape identity is required")
    return saved


def source_paths() -> dict[str, Path]:
    return {**latency.source_paths(), **{str(path): ROOT / path for path in SOURCES}}


def admit_history(
    reference: BenchmarkCycle,
    cuda: BenchmarkCycle,
    cases: tuple[EvaluationCase, ...],
) -> dict[str, str]:
    pins = {str(RESULTS / name): sha for name, sha in HISTORY_PINS.items()}
    if any(digest(ROOT / name) != sha for name, sha in pins.items()):
        raise ValueError("Independently reviewed B1 history changed")
    declaration = ROOT / RESULTS / "cuda-latency-source-declaration-20261005.json"
    hashes = latency.admit_sources(declaration, HISTORY_PINS[declaration.name])
    pins.update(latency.admit_screens(reference, cuda, cases))
    confirm = ROOT / RESULTS / "cuda-latency-confirm-b1.json"
    pins.update(
        latency.admit_confirmation(
            confirm,
            HISTORY_PINS[confirm.name],
            wall_sha256=HISTORY_PINS["cuda-latency-confirm-b1.wall.json"],
            batch_size=1,
            source_hashes=hashes,
            reference=reference,
            cuda=cuda,
            cases=cases,
        )
    )
    timing = ROOT / RESULTS / "cuda-latency-timing-b1.json"
    saved = read_object(timing)
    latency.require_parent(timing)
    if (
        saved.get("mode") != "timing"
        or saved.get("batch_size") != 1
        or saved.get("completed") is not True
        or saved.get("quality_passed") is not True
        or saved.get("case_ids") != [case.id for case in cases]
        or saved.get("input_sha256") != INPUT_SHA
        or saved.get("new_source_hashes") != hashes
        or any(cast(dict[str, list[str]], saved["unknown_case_ids"]).values())
    ):
        raise ValueError("Prior timing must retain complete canonical B1 outcomes")
    chronological = cast(list[dict[str, object]], saved["cycles"])
    expected = ["warmup_math", "warmup_efficient"] + [
        f"pair{pair}_{arm}"
        for pair in range(6)
        for arm in (("math", "efficient") if pair % 2 == 0 else ("efficient", "math"))
    ]
    if [item.get("key") for item in chronological] != expected:
        raise ValueError("All prior six paired cycles are required in order")
    reconstructed = []
    for item in chronological:
        cycle = latency.read_cycle(item["cycle"], cases)
        if not check_cycle(cycle, reference, cuda)[0]:
            raise ValueError("Prior timing violates the unchanged numerical gates")
        reconstructed.append(cycle)
    for start in range(0, 14, 2):
        if not compare_cycles(
            (reconstructed[start],), (reconstructed[start + 1],), QualityGate()
        ).passed:
            raise ValueError("Prior timing paired quality failed")
    current = latency.source_paths()
    if saved["source_sha256"] != {name: digest(path) for name, path in current.items()}:
        raise ValueError("Prior timing execution sources changed")
    archive = cast(dict[str, str], saved["source_archive"])
    archive_path = timing.with_suffix(f".sources-{archive['sha256'][:16]}.json")
    if digest(archive_path) != archive["sha256"]:
        raise ValueError("Prior timing source archive changed")
    pins[str(archive_path.relative_to(ROOT))] = archive["sha256"]
    return pins


def trace_metadata(path: Path) -> dict[str, object]:
    """Archive native bytes; interpret the observed schema independently later."""
    return {
        "path": str(path),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "schema_interpreted": False,
        "scope": "Raw Chrome evidence; launch geometry/linkage/clock domains unknown",
    }


def require_finite_timing(cycle: BenchmarkCycle) -> None:
    if (
        not math.isfinite(cycle.latency_ms)
        or cycle.latency_ms <= 0
        or any(
            not math.isfinite(row.latency_ms) or row.latency_ms < 0
            for row in cycle.rows
        )
    ):
        raise ValueError("Completed intervals must be finite and nonnegative")


def math_dispatch_gate(evidence: dict[str, object]) -> bool:
    operators = cast(Sequence[dict[str, object]], evidence.get("operators", ()))
    counts: dict[str, int] = {}
    for event in operators:
        name, count = event.get("name"), event.get("count")
        if not isinstance(name, str) or type(count) is not int or count < 0:
            return False
        counts[name] = counts.get(name, 0) + count
    return counts.get("aten::_scaled_dot_product_attention_math") == 28 and not any(
        count > 0
        and any(
            other in name
            for other in ("efficient_attention", "flash_attention", "cudnn_attention")
        )
        for name, count in counts.items()
    )


def profile_one(
    scorer: Scorer,
    case: EvaluationCase,
    *,
    name: str,
    trace_path: Path,
    retain: Callable[[BenchmarkCycle, dict[str, object]], None],
) -> dict[str, object]:
    """One exported capture; retain output even if cleanup/export later fails."""
    if trace_path.exists():
        raise FileExistsError("A native trace must never overwrite prior evidence")
    observer = Dispatch()
    cycle: BenchmarkCycle | None = None
    try:
        torch.cuda.synchronize()
        with (
            warnings.catch_warnings(record=True) as caught,
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as profiler,
            observer,
        ):
            with record_function(name):
                cycle = _run(
                    BenchmarkEngine(
                        name,
                        Model(scorer, model_id=MODEL_ID),
                        batch_size=1,
                        synchronize=torch.cuda.synchronize,
                    ),
                    (case,),
                )
            torch.cuda.synchronize()
        observed: dict[str, object] = {
            "case_ids": [case.id],
            "dispatch": observer.events,
            "operators": [
                {
                    "name": event.key,
                    "count": event.count,
                    "shapes": event.input_shapes,
                    "self_cpu_time_us": event.self_cpu_time_total,
                    "self_device_time_us": event.self_device_time_total,
                }
                for event in profiler.key_averages(group_by_input_shape=True)
            ],
            "cuda_kernel_events": [
                {
                    "name": event.name,
                    "duration_us": event.time_range.elapsed_us(),
                    "device_index": event.device_index,
                }
                for event in profiler.events()
                if event.device_type == DeviceType.CUDA
            ],
            "warnings": [str(item.message) for item in caught],
            "timing_scope": "Intrusive one-call profile; no ordinary-speed claim",
            "cleanup_completed": True,
        }
    except Exception as error:
        if cycle is not None:
            retain(
                cycle,
                {
                    "cleanup_completed": False,
                    "error": str(error),
                    "dispatch": observer.events,
                },
            )
        raise
    observed["cuda_kernel_names_available"] = bool(observed["cuda_kernel_events"])
    retain(cycle, observed)
    if trace_path.exists():
        raise FileExistsError("A native trace must never overwrite prior evidence")
    try:
        profiler.export_chrome_trace(str(trace_path))
    except Exception as error:
        observed["chrome_export_error"] = f"{type(error).__name__}: {error}"
        if trace_path.exists():
            observed["chrome"] = trace_metadata(trace_path)
        raise
    observed["chrome"] = trace_metadata(trace_path)
    if trace_path.stat().st_size <= 0:
        raise ValueError("Native Chrome export produced no bytes")
    return observed


def study(model: Path, output: Path, *, sources: Path, sources_sha256: str) -> bool:
    started = time.perf_counter_ns()
    active_phase: Phase | None = None
    phase_starts: dict[Phase, int] = {}
    phase_wall_ms: dict[str, float] = {}
    selection_ids = {offset: case_id for offset, case_id, _ in CASE_SELECTION}
    pending = {probe.key: [selection_ids[probe.offset]] for probe in probe_schedule()}
    payload: dict[str, object] = {
        "completed": False,
        "quality_passed": False,
        "dispatch_passed": False,
        "accepted": False,
        "cycles": [],
        "profiles": {},
        "unknown_case_ids": pending,
        "phase_wall_ms": phase_wall_ms,
        "selection": [list(item) for item in CASE_SELECTION],
        "expected_rows": 12,
        "scope": "Exploratory real B1 L192/L197; content/order/governor confounded",
        "gate": asdict(QualityGate()),
        "limits": [
            "Missing native launch fields unknown",
            "Profile durations do not establish ordinary speed or critical path",
            "No routing threshold, synthetic input or production promotion",
        ],
    }

    def write() -> None:
        output.write_text(json.dumps(payload, allow_nan=False) + "\n")

    with output.open("x") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")
    try:
        new_hashes = admit_sources(sources, sources_sha256)
        manifest = ROOT / RESULTS / "cuda-transfer-manifest-v2-20261005.json"
        if digest(manifest) != latency.MANIFEST_SHA:
            raise ValueError("Original 44-asset manifest changed")
        admission = admit(manifest)
        control = ROOT / RESULTS / "cuda-fp32-b1.json"
        if digest(control.with_suffix(".wall.json")) != latency.CONTROL_WALL_SHA:
            raise ValueError("Original control parent identity changed")
        pinned = admit_control(control, latency.CONTROL_SHA, admission, batch_size=1)
        if any(
            (
                sdpa_attention._is_torch_mps_available,
                sdpa_attention._is_torch_xpu_available,
                sdpa_attention._is_torch_npu_available,
            )
        ):
            raise ValueError("Frozen Linux CUDA HF availability required")
        frozen = freeze_sources(source_paths(), output)
        payload.update(
            admission=admission,
            control=pinned,
            new_source_hashes=new_hashes,
            source_declaration_sha256=sources_sha256,
            source_sha256=frozen.hashes(),
            source_archive=frozen.archive(),
        )
        cases = tuple(
            case_from_dict(decode_json(line))
            for line in (ROOT / DATA).read_text().splitlines()
        )
        if len(cases) != 77 or _input_digest(cases) != INPUT_SHA:
            raise ValueError("All original ordered77 cases required")
        for offset, case_id, _ in CASE_SELECTION:
            if cases[offset].id != case_id:
                raise ValueError("Fixed real case identity changed")
        write()
        artifact = load_artifact(model, 512)
        if artifact.model_id != MODEL_ID:
            raise ValueError("Original checkpoint identity changed")
        inputs = latency.ReferenceInput(
            artifact.model_id,
            KaiEncoder.from_artifact(artifact, max_length=512, token_cache_size=0),
        )
        reference = read_reference(
            ROOT / REFERENCE, cases, inputs, dataset_sha256=DATA_SHA
        )
        cuda = control_cycle(control, cases)
        if not compare_cycles((reference,), (cuda,), QualityGate()).passed:
            raise ValueError("Original CUDA/CPU control quality changed")
        for offset, _, length in CASE_SELECTION:
            if len(inputs.encode_request(cases[offset].request).ids) != length:
                raise ValueError("Fixed exact token length changed")
        history = admit_history(reference, cuda, cases)
        payload["history_sha256"] = history
        traces = output.with_suffix(".trace")
        traces.mkdir()

        def verify() -> None:
            if (
                any(digest(Path(item.path)) != item.sha256 for item in frozen.files)
                or any(digest(ROOT / name) != sha for name, sha in history.items())
                or digest(control) != latency.CONTROL_SHA
                or digest(control.with_suffix(".wall.json")) != latency.CONTROL_WALL_SHA
            ):
                raise ValueError("Consumed source/history/control changed")

        payload["occupancy_before_load"] = occupancy()
        torch.set_num_threads(8)
        base = KaiScorer.from_directory(
            model, device="cpu", max_length=512, token_cache_size=0
        )
        read_reference(ROOT / REFERENCE, cases, base, dataset_sha256=DATA_SHA)
        payload["occupancy_before_allocation"] = occupancy()
        with ieee_precision():
            base.device = torch.device("cuda:0")
            cast(nn.Module, base.backbone).to(base.device)
            base.head.to(base.device)
            require_fp32(base)
            scorers: dict[Arm, Scorer] = {
                "math": FP32Scorer(base),
                "efficient": efficient_scorer(base),
            }
            previous: dict[tuple[Phase, int, Arm], BenchmarkCycle] = {}
            profiles = cast(dict[str, dict[str, object]], payload["profiles"])
            completed = cast(list[dict[str, object]], payload["cycles"])
            for probe in probe_schedule():
                active_phase = probe.phase
                phase_starts.setdefault(probe.phase, time.perf_counter_ns())
                verify()
                snapshot = occupancy(own_pid=os.getpid())
                case = cases[probe.offset]
                payload["phase"] = probe.key

                def retain(
                    cycle: BenchmarkCycle,
                    observed: dict[str, object],
                    *,
                    probe: Probe = probe,
                    case: EvaluationCase = case,
                    snapshot: dict[str, object] = snapshot,
                ) -> None:
                    completed.append(
                        {
                            "key": probe.key,
                            "cycle": latency.cycle_data(cycle),
                            "ownership": snapshot,
                        }
                    )
                    if observed:
                        profiles[probe.key] = observed
                    pending[probe.key] = (
                        [] if case.id in {row.id for row in cycle.rows} else [case.id]
                    )
                    write()

                if probe.phase == "profile":
                    length = len(base.encode_request(case.request).ids)
                    observed = profile_one(
                        scorers[probe.arm],
                        case,
                        name=f"{case.id}:{probe.arm}:L{length}",
                        trace_path=traces / f"{case.id}-{probe.arm}.json",
                        retain=retain,
                    )
                    profiles[probe.key] = observed
                    cycle = latency.read_cycle(completed[-1]["cycle"], (case,))
                    if probe.arm == "efficient" and not efficient_dispatch_gate(
                        observed, batch_size=1, length=length
                    ):
                        raise ValueError("Reviewed efficient dispatch unsupported")
                    if probe.arm == "math" and not math_dispatch_gate(observed):
                        raise ValueError("Profiled comparator is not the math backend")
                else:
                    cycle = _run(
                        BenchmarkEngine(
                            probe.arm,
                            Model(scorers[probe.arm], MODEL_ID),
                            batch_size=1,
                            synchronize=torch.cuda.synchronize,
                        ),
                        (case,),
                    )
                    retain(cycle, {})
                require_finite_timing(cycle)
                passed, drift = check_cycle(
                    cycle,
                    replace(reference, rows=(reference.rows[probe.offset],)),
                    replace(cuda, rows=(cuda.rows[probe.offset],)),
                )
                completed[-1]["drift"] = drift
                write()
                if not passed:
                    raise ValueError(
                        "Real-shape output fails unchanged numerical gates"
                    )
                previous[probe.phase, probe.offset, probe.arm] = cycle
                other = previous.get(
                    (
                        probe.phase,
                        probe.offset,
                        "efficient" if probe.arm == "math" else "math",
                    )
                )
                if (
                    other is not None
                    and not compare_cycles((other,), (cycle,), QualityGate()).passed
                ):
                    raise ValueError("Real-shape paired quality failed")
                phase_wall_ms[probe.phase] = (
                    time.perf_counter_ns() - phase_starts[probe.phase]
                ) / 1e6
                write()
            require_fp32(base)
            payload["occupancy_after"] = occupancy(own_pid=os.getpid())
        verify()
        if (
            [row["key"] for row in completed] != [p.key for p in probe_schedule()]
            or any(pending.values())
            or set(profiles)
            != {p.key for p in probe_schedule() if p.phase == "profile"}
        ):
            raise ValueError("Every fixed output and capture must be retained")
        for observed in profiles.values():
            chrome = cast(dict[str, object], observed["chrome"])
            trace = Path(cast(str, chrome["path"]))
            if trace_metadata(trace) != chrome:
                raise ValueError("Native trace bytes changed after export")
        if (
            admit(manifest) != admission
            or admit_sources(sources, sources_sha256) != new_hashes
        ):
            raise ValueError("Final immutable admission changed")
        payload.update(
            completed=True,
            quality_passed=True,
            dispatch_passed=True,
            phase="completed",
            worker_body_wall_ms=(time.perf_counter_ns() - started) / 1e6,
        )
        write()
        return True
    except Exception as error:
        for phase, phase_started in phase_starts.items():
            if phase == active_phase or phase not in phase_wall_ms:
                phase_wall_ms[phase] = (time.perf_counter_ns() - phase_started) / 1e6
        payload["worker_body_wall_ms"] = (time.perf_counter_ns() - started) / 1e6
        payload.update(
            completed=False,
            quality_passed=False,
            dispatch_passed=False,
            accepted=False,
            phase="failed",
            error=f"{type(error).__name__}: {error}",
        )
        write()
        return False
