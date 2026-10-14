"""CPU-only challenges to exact real-case capture, ownership and retained evidence."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import TracebackType
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from torch import nn
from torch._C._autograd import DeviceType
from transformers.integrations import sdpa_attention

from benchmarks.analysis.cuda.latency import study as latency
from benchmarks.analysis.cuda.manifest import DATA, MODEL_ID, ROOT, digest
from benchmarks.analysis.cuda.shape import run as parent
from benchmarks.analysis.cuda.shape import study
from benchmarks.analysis.evidence import FrozenSources, SourceFile
from wyat import Request
from wyat.backends.wyat.encoding import EncodedRequest, WyatEncoder
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine
from wyat.core import Scorer, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict
from wyat.wire import decode_json

RAW_CHROME = (
    b'{"traceEvents":[{"cat":"kernel","name":'
    b'"fmha_cutlassF_f32_aligned_64x128_rf_sm80(Params)"}]}\n'
)
SELECTED = ((7, "banking77-train-03357", 192), (18, "banking77-train-00566", 197))
ORDER = ((7, "math"), (7, "efficient"), (18, "efficient"), (18, "math"))


@pytest.fixture
def cases() -> tuple[EvaluationCase, ...]:
    result = tuple(
        case_from_dict(decode_json(line))
        for line in (ROOT / DATA).read_text().splitlines()
    )
    assert len(result) == 77
    assert [(offset, result[offset].id) for offset, _, _ in SELECTED] == [
        (offset, identifier) for offset, identifier, _ in SELECTED
    ]
    return result


def cycle(cases: tuple[EvaluationCase, ...], elapsed_ms: float = 1.0) -> BenchmarkCycle:
    rows: list[EvaluationRow] = []
    for case in cases:
        scores = [
            2.0 if candidate.id in case.expected else 0.0
            for candidate in case.request.candidates
        ]
        decision = decision_from_logits(case.request, scores, MODEL_ID)
        rows.append(
            EvaluationRow(case.id, case.expected, "ok", decision, True, 0.5, None)
        )
    return BenchmarkCycle(elapsed_ms, tuple(rows), 0, 0)


class NeverScorer:
    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        raise AssertionError("Controlled runner must never allocate or execute a model")


def test_exact_case_order_and_twelve_outcomes_are_predeclared() -> None:
    assert study.CASE_SELECTION == SELECTED
    schedule = study.probe_schedule()
    expected = [
        (phase, offset, arm)
        for phase in ("warmup", "ordinary", "profile")
        for offset, arm in ORDER
    ]
    assert [(probe.phase, probe.offset, probe.arm) for probe in schedule] == expected
    assert len({probe.key for probe in schedule}) == 12


@pytest.mark.parametrize("changed", ["missing", "efficient", "flash", "cudnn", "count"])
def test_math_profile_requires_actual_math_and_no_fallback(changed: str) -> None:
    evidence: dict[str, object] = {
        "operators": [{"name": "aten::_scaled_dot_product_attention_math", "count": 28}]
    }
    assert study.math_dispatch_gate(evidence)
    operators = cast(list[dict[str, object]], evidence["operators"])
    if changed == "missing":
        operators.clear()
    elif changed == "count":
        operators[0]["count"] = 27
    else:
        operators.append(
            {"name": f"aten::_scaled_dot_product_{changed}_attention", "count": 1}
        )
    assert not study.math_dispatch_gate(evidence)


def test_native_trace_metadata_does_not_invent_launch_fields(tmp_path: Path) -> None:
    path = tmp_path / "trace.json"
    path.write_bytes(RAW_CHROME)
    saved = study.trace_metadata(path)
    assert saved["path"] == str(path)
    assert saved["bytes"] == len(RAW_CHROME) and saved["sha256"] == digest(path)
    assert saved["schema_interpreted"] is False
    assert not {"grid", "block", "stream", "correlation", "tile"}.intersection(saved)
    assert path.read_bytes() == RAW_CHROME


@dataclass
class TimeRange:
    def elapsed_us(self) -> float:
        return 1.0


@dataclass
class Kernel:
    name: str = "fmha_cutlassF_f32_aligned_64x128_rf_sm80(Params)"
    time_range: TimeRange = field(default_factory=TimeRange)
    device_index: int = 0
    device_type: object = DeviceType.CUDA


@dataclass
class Operator:
    key: str = "aten::_scaled_dot_product_efficient_attention"
    count: int = 28
    input_shapes: list[object] = field(default_factory=list)
    self_cpu_time_total: float = 1.0
    self_device_time_total: float = 1.0


@dataclass
class Profiler:
    events_log: list[str]
    active: bool = False
    exported: list[str] = field(default_factory=list)
    fail_export: bool = False
    fail_close: bool = False
    on_close: Callable[[], None] | None = None

    def __enter__(self) -> Profiler:
        self.active = True
        self.events_log.append("profiler_enter")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.active = False
        self.events_log.append("profiler_exit")
        if self.on_close is not None:
            self.on_close()
        if self.fail_close:
            raise RuntimeError("profiler cleanup failed")

    def key_averages(self, *, group_by_input_shape: bool) -> list[Operator]:
        assert group_by_input_shape is True
        return [Operator()]

    def events(self) -> list[Kernel]:
        return [Kernel() for _ in range(28)]

    def export_chrome_trace(self, path: str) -> None:
        assert not self.active
        assert "observer_exit" in self.events_log
        self.events_log.append("export")
        self.exported.append(path)
        if self.fail_export:
            raise OSError("native export failed")
        Path(path).write_bytes(RAW_CHROME)


@dataclass
class Observer:
    events_log: list[str]
    events: list[dict[str, object]] = field(default_factory=list)
    active: bool = False

    def __enter__(self) -> Observer:
        self.active = True
        self.events_log.append("observer_enter")
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.active = False
        self.events_log.append("observer_exit")


@dataclass
class ProfileSetup:
    case: EvaluationCase
    path: Path
    profiler: Profiler
    observer: Observer
    events: list[str]
    retained: list[tuple[BenchmarkCycle, dict[str, object]]]

    def run(self) -> dict[str, object]:
        def retain(result: BenchmarkCycle, evidence: dict[str, object]) -> None:
            self.retained.append((result, evidence))
            self.events.append("retain")
            json.dumps(evidence, allow_nan=False)

        return study.profile_one(
            NeverScorer(),
            self.case,
            name=f"{self.case.id}:efficient:L192",
            trace_path=self.path,
            retain=retain,
        )


@pytest.fixture
def profile_setup(
    tmp_path: Path, cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch
) -> ProfileSetup:
    events: list[str] = []
    profiler, observer = Profiler(events), Observer(events)
    setup = ProfileSetup(
        cases[7], tmp_path / "trace.json", profiler, observer, events, []
    )

    def capture(**kwargs: object) -> Profiler:
        assert kwargs["record_shapes"] is True
        assert kwargs.get("with_stack", False) is False
        assert kwargs.get("profile_memory", False) is False
        return profiler

    def dispatch() -> Observer:
        return observer

    @contextmanager
    def label(name: str) -> Iterator[None]:
        assert setup.case.id in name and "efficient" in name and "192" in name
        events.append("label")
        yield

    def run(
        engine: BenchmarkEngine, selected: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        assert profiler.active and observer.active
        assert selected == (setup.case,) and engine.batch_size == 1
        events.append("run")
        return cycle(selected)

    def sync() -> None:
        events.append("sync")

    monkeypatch.setattr(study, "profile", capture)
    monkeypatch.setattr(study, "Dispatch", dispatch)
    monkeypatch.setattr(study, "record_function", label)
    monkeypatch.setattr(study, "_run", run)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    return setup


def test_one_profile_exports_after_contexts_close_and_retains_native_bytes(
    profile_setup: ProfileSetup,
) -> None:
    observed = profile_setup.run()
    assert len(profile_setup.retained) == 1
    assert profile_setup.events[0] == "sync"
    assert (
        profile_setup.events.index("run")
        < profile_setup.events.index("observer_exit")
        < profile_setup.events.index("export")
    )
    assert (
        profile_setup.events.index("profiler_exit")
        < profile_setup.events.index("retain")
        < profile_setup.events.index("export")
    )
    assert profile_setup.events.count("sync") == 2
    assert profile_setup.profiler.exported == [str(profile_setup.path)]
    chrome = cast(dict[str, object], observed["chrome"])
    assert chrome["sha256"] == digest(profile_setup.path) and chrome["bytes"] == len(
        RAW_CHROME
    )
    assert observed["cleanup_completed"] is True
    assert profile_setup.path.read_bytes() == RAW_CHROME


@pytest.mark.parametrize("failure", ["export", "cleanup", "collision"])
def test_profile_failure_preserves_already_returned_row(
    profile_setup: ProfileSetup, failure: str
) -> None:
    if failure == "export":
        profile_setup.profiler.fail_export = True
    elif failure == "cleanup":
        profile_setup.profiler.fail_close = True
    else:

        def collision() -> None:
            profile_setup.path.write_bytes(b"prior evidence")

        profile_setup.profiler.on_close = collision
    with pytest.raises((RuntimeError, OSError, FileExistsError)):
        profile_setup.run()
    assert len(profile_setup.retained) == 1
    assert profile_setup.retained[0][0].rows[0].id == profile_setup.case.id
    assert not profile_setup.profiler.active and not profile_setup.observer.active
    if failure == "collision":
        assert profile_setup.path.read_bytes() == b"prior evidence"
        assert profile_setup.profiler.exported == []


def test_trace_collision_rejects_before_inference(profile_setup: ProfileSetup) -> None:
    profile_setup.path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        profile_setup.run()
    assert profile_setup.events == [] and profile_setup.retained == []
    assert profile_setup.path.read_bytes() == b"existing"


@dataclass
class Setup:
    path: Path
    output: Path
    cases: tuple[EvaluationCase, ...]
    calls: list[tuple[str, str]]
    milestones: list[str]
    length_delta: int = 0

    def run(self) -> bool:
        return study.study(
            self.path, self.output, sources=self.path, sources_sha256="fixed"
        )

    def saved(self) -> dict[str, object]:
        return cast(dict[str, object], json.loads(self.output.read_text()))


@dataclass
class FakeArtifact:
    model_id: str = MODEL_ID


@dataclass
class FakeBase:
    setup: Setup
    backbone: nn.Module = field(default_factory=nn.Module)
    head: nn.Module = field(default_factory=nn.Module)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def encode_request(self, request: Request) -> EncodedRequest:
        length = next(
            (
                length
                for offset, _, length in SELECTED
                if self.setup.cases[offset].request == request
            ),
            172,
        )
        return EncodedRequest(
            tuple(range(length + self.setup.length_delta)), (0, 1, 2, 3, 4, 5), 0
        )


@pytest.fixture
def setup(
    tmp_path: Path, cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch
) -> Setup:
    data = tmp_path / DATA
    data.parent.mkdir(parents=True)
    data.write_bytes((ROOT / DATA).read_bytes())
    configured = Setup(tmp_path / "identity", tmp_path / "result.json", cases, [], [])
    monkeypatch.setattr(study, "ROOT", tmp_path)
    for name in ("MANIFEST_SHA", "CONTROL_SHA", "CONTROL_WALL_SHA"):
        monkeypatch.setattr(latency, name, "fixed")
    for flag in (
        "_is_torch_mps_available",
        "_is_torch_xpu_available",
        "_is_torch_npu_available",
    ):
        monkeypatch.setattr(sdpa_attention, flag, False)

    def fixed(path: Path) -> str:
        return digest(path) if path.parent.suffix == ".trace" else "fixed"

    def admitted(*args: object, **kwargs: object) -> dict[str, object]:
        return {"admitted": True}

    def hashes(*args: object, **kwargs: object) -> dict[str, str]:
        return {}

    def frozen(*args: object, **kwargs: object) -> FrozenSources:
        return FrozenSources((), "fake", "fixed")

    def artifact(*args: object, **kwargs: object) -> FakeArtifact:
        return FakeArtifact()

    def reference(
        path: Path,
        cohort: tuple[EvaluationCase, ...],
        encoder: object,
        **kwargs: object,
    ) -> BenchmarkCycle:
        assert cohort == cases
        configured.milestones.append(
            "token_validation"
            if isinstance(encoder, latency.ReferenceInput)
            else "loaded_validation"
        )
        return cycle(cohort)

    def control(path: Path, cohort: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
        return cycle(cohort)

    def paths() -> dict[str, Path]:
        return {}

    for name, replacement in (
        ("digest", fixed),
        ("admit", admitted),
        ("admit_sources", hashes),
        ("admit_control", hashes),
        ("admit_history", hashes),
        ("freeze_sources", frozen),
        ("source_paths", paths),
        ("load_artifact", artifact),
        ("occupancy", admitted),
        ("read_reference", reference),
        ("control_cycle", control),
    ):
        monkeypatch.setattr(study, name, replacement)
    base = FakeBase(configured)

    def load(*args: object, **kwargs: object) -> FakeBase:
        assert kwargs == {"device": "cpu", "max_length": 512, "token_cache_size": 0}
        assert configured.milestones == ["token_validation"]
        configured.milestones.append("load")
        return base

    def encoder(*args: object, **kwargs: object) -> FakeBase:
        assert kwargs == {"max_length": 512, "token_cache_size": 0}
        return base

    def move(module: nn.Module, *args: object, **kwargs: object) -> nn.Module:
        assert configured.milestones[:3] == [
            "token_validation",
            "load",
            "loaded_validation",
        ]
        configured.milestones.append("allocation")
        return module

    def noop(*args: object, **kwargs: object) -> None:
        return None

    def forbidden() -> None:
        raise AssertionError("CPU fixtures must never initialize CUDA")

    def factory(*args: object, **kwargs: object) -> NeverScorer:
        return NeverScorer()

    @contextmanager
    def precision() -> Iterator[None]:
        configured.milestones.append("precision_enter")
        try:
            yield
        finally:
            configured.milestones.append("precision_restored")

    def ordinary(
        engine: BenchmarkEngine, cohort: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        assert len(cohort) == engine.batch_size == 1
        configured.calls.append((engine.name, cohort[0].id))
        return cycle(cohort)

    def capture(
        scorer: Scorer,
        case: EvaluationCase,
        *,
        name: str,
        trace_path: Path,
        retain: Callable[[BenchmarkCycle, dict[str, object]], None],
    ) -> dict[str, object]:
        assert len(configured.calls) >= 8, "All ordinary/warm outputs must pass first"
        configured.calls.append((f"profile:{name}", case.id))
        observed: dict[str, object] = {
            "cleanup_completed": True,
            "operators": [
                {"name": "aten::_scaled_dot_product_attention_math", "count": 28}
            ],
        }
        retain(cycle((case,)), observed)
        with trace_path.open("xb") as stream:
            stream.write(RAW_CHROME)
        observed["chrome"] = study.trace_metadata(trace_path)
        return observed

    def dispatch(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(WyatEncoder, "from_artifact", encoder)
    monkeypatch.setattr(WyatScorer, "from_directory", load)
    monkeypatch.setattr(nn.Module, "to", move)
    monkeypatch.setattr(study, "require_fp32", noop)
    monkeypatch.setattr(study, "FP32Scorer", factory)
    monkeypatch.setattr(study, "efficient_scorer", factory)
    monkeypatch.setattr(study, "ieee_precision", precision)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", noop)
    monkeypatch.setattr(study, "_run", ordinary)
    monkeypatch.setattr(study, "profile_one", capture)
    monkeypatch.setattr(study, "efficient_dispatch_gate", dispatch)
    return configured


def pending(saved: dict[str, object]) -> dict[str, list[str]]:
    return cast(dict[str, list[str]], saved["unknown_case_ids"])


def records(saved: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], saved["cycles"])


def assert_failed(setup: Setup) -> dict[str, object]:
    saved = setup.saved()
    assert all(
        saved[key] is False
        for key in ("completed", "quality_passed", "dispatch_passed", "accepted")
    )
    assert saved["phase"] == "failed"
    return saved


def test_real_cases_complete_all_twelve_outputs_and_four_distinct_exports(
    setup: Setup,
) -> None:
    assert setup.run()
    saved = setup.saved()
    assert (
        saved["completed"] is True
        and saved["quality_passed"] is True
        and saved["dispatch_passed"] is True
    )
    assert saved["accepted"] is False
    assert setup.milestones[:3] == ["token_validation", "load", "loaded_validation"]
    assert setup.milestones[-1] == "precision_restored"
    assert [record["key"] for record in records(saved)] == [
        probe.key for probe in study.probe_schedule()
    ]
    assert setup.calls[:8] == [
        (arm, setup.cases[offset].id) for _ in range(2) for offset, arm in ORDER
    ]
    assert len(setup.calls) == 12 and not any(pending(saved).values())
    profiles = cast(dict[str, dict[str, object]], saved["profiles"])
    assert set(profiles) == {
        probe.key for probe in study.probe_schedule() if probe.phase == "profile"
    }
    exported = [cast(dict[str, object], entry["chrome"]) for entry in profiles.values()]
    assert len({entry["path"] for entry in exported}) == 4
    for chrome in exported:
        path = Path(cast(str, chrome["path"]))
        assert path.read_bytes() == RAW_CHROME and chrome["sha256"] == digest(path)
        assert (
            chrome["bytes"] == len(RAW_CHROME) and chrome["schema_interpreted"] is False
        )
    assert set(cast(dict[str, object], saved["phase_wall_ms"])) == {
        "warmup",
        "ordinary",
        "profile",
    }


@settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(delta=st.sampled_from([-2, -1, 1, 2]))
def test_exact_selected_token_lengths_cannot_drift(setup: Setup, delta: int) -> None:
    setup.output.unlink(missing_ok=True)
    setup.milestones.clear()
    setup.length_delta = delta
    assert not setup.run()
    saved = assert_failed(setup)
    assert setup.calls == [] and "load" not in setup.milestones
    assert all(pending(saved).values()) and len(pending(saved)) == 12


@pytest.mark.parametrize("change", ["order", "count", "request", "tokens"])
def test_all77_reference_admission_precedes_weight_load_and_allocation(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    if change == "tokens":

        def rejected(*args: object, **kwargs: object) -> BenchmarkCycle:
            raise ValueError("Token contract changed")

        monkeypatch.setattr(study, "read_reference", rejected)
    else:
        data = setup.output.parent / DATA
        lines = data.read_text().splitlines()
        if change == "order":
            lines[7], lines[8] = lines[8], lines[7]
        elif change == "count":
            lines.pop()
        else:
            changed = json.loads(lines[0])
            changed["request"]["query"] += " changed"
            lines[0] = json.dumps(changed)
        data.write_text("\n".join(lines) + "\n")
    assert not setup.run()
    assert_failed(setup)
    assert "load" not in setup.milestones and "allocation" not in setup.milestones
    assert len(pending(setup.saved())) == 12 and all(pending(setup.saved()).values())


@pytest.mark.parametrize("where", ["cycle", "row"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_attempt_is_serializable_retained_and_rejected(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, where: str, value: float
) -> None:
    def invalid(
        engine: BenchmarkEngine, cohort: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        result = cycle(cohort)
        return (
            replace(result, latency_ms=value)
            if where == "cycle"
            else replace(result, rows=(replace(result.rows[0], latency_ms=value),))
        )

    monkeypatch.setattr(study, "_run", invalid)
    assert not setup.run()
    saved = assert_failed(setup)
    retained = cast(dict[str, object], records(saved)[0]["cycle"])
    assert "invalid_timing" in retained
    assert len(cast(list[object], retained["rows"])) == 1
    assert (
        pending(saved)["warmup:7:math"] == []
        and sum(bool(ids) for ids in pending(saved).values()) == 11
    )
    json.dumps(saved, allow_nan=False)
    assert setup.milestones[-1] == "precision_restored"


@pytest.mark.parametrize("change", ["score", "missing", "failed"])
def test_invalid_ordinary_output_aborts_before_profiles_and_preserves_unknowns(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    def invalid(
        engine: BenchmarkEngine, cohort: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        result = cycle(cohort)
        if change == "missing":
            return replace(result, rows=())
        row = result.rows[0]
        if change == "failed":
            return replace(
                result,
                rows=(
                    replace(
                        row,
                        status="error",
                        prediction=None,
                        correct=False,
                        error="failed",
                    ),
                ),
            )
        decision = decision_from_logits(cohort[0].request, [0.0] * 6, MODEL_ID)
        return replace(result, rows=(replace(row, prediction=decision),))

    monkeypatch.setattr(study, "_run", invalid)
    assert not setup.run()
    saved = assert_failed(setup)
    assert len(records(saved)) == 1 and saved["profiles"] == {}
    assert pending(saved)["warmup:7:math"] == (
        [setup.cases[7].id] if change == "missing" else []
    )


@pytest.mark.parametrize("boundary", ["manifest", "history", "control", "ownership"])
def test_changed_admission_or_foreign_ownership_cannot_continue(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    if boundary == "ownership":

        def occupied(*args: object, **kwargs: object) -> dict[str, object]:
            if "own_pid" in kwargs:
                raise ValueError("Foreign GPU process appeared")
            return {}

        monkeypatch.setattr(study, "occupancy", occupied)
    elif boundary == "history":

        def changed(*args: object, **kwargs: object) -> dict[str, str]:
            raise ValueError("Independently reviewed history changed")

        monkeypatch.setattr(study, "admit_history", changed)
    else:

        def altered(path: Path) -> str:
            return (
                "changed"
                if (
                    (boundary == "manifest" and "manifest" in path.name)
                    or (boundary == "control" and path.name == "cuda-fp32-b1.json")
                )
                else "fixed"
            )

        monkeypatch.setattr(study, "digest", altered)
    assert not setup.run()
    saved = assert_failed(setup)
    assert records(saved) == [] and all(pending(saved).values())
    if boundary in ("ownership", "control"):
        assert setup.milestones[-1] == "precision_restored"


@pytest.mark.parametrize(
    "boundary", ["dispatch", "export", "empty", "missing_retain", "trace_changed"]
)
def test_profile_failure_retains_observed_attempts_and_never_falls_back(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    original = study.profile_one
    if boundary == "dispatch":

        def unsupported(*args: object, **kwargs: object) -> bool:
            return False

        monkeypatch.setattr(study, "efficient_dispatch_gate", unsupported)
    else:

        def capture(
            scorer: Scorer,
            case: EvaluationCase,
            *,
            name: str,
            trace_path: Path,
            retain: Callable[[BenchmarkCycle, dict[str, object]], None],
        ) -> dict[str, object]:
            if boundary == "export":
                observed: dict[str, object] = {"chrome_export_error": "OSError: failed"}
                retain(cycle((case,)), observed)
                raise OSError("native export failed")
            if boundary == "missing_retain":
                trace_path.write_bytes(RAW_CHROME)
                return {"chrome": study.trace_metadata(trace_path)}
            observed = original(
                scorer, case, name=name, trace_path=trace_path, retain=retain
            )
            if boundary == "empty":
                trace_path.write_bytes(b"")
            else:
                trace_path.write_bytes(b"changed")
            return observed

        monkeypatch.setattr(study, "profile_one", capture)
    assert not setup.run()
    saved = assert_failed(setup)
    assert len(records(saved)) >= 8 and setup.milestones[-1] == "precision_restored"
    if boundary == "export":
        assert len(records(saved)) == 9 and pending(saved)["profile:7:math"] == []
        assert sum(bool(ids) for ids in pending(saved).values()) == 3
        assert "chrome_export_error" in cast(
            dict[str, object],
            cast(dict[str, object], saved["profiles"])["profile:7:math"],
        )
    assert saved["accepted"] is False


@pytest.mark.parametrize("boundary", ["restore", "admission", "final_write"])
def test_final_failures_clear_positive_flags_but_keep_all_outputs(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    if boundary == "restore":

        @contextmanager
        def bad_restore() -> Iterator[None]:
            try:
                yield
            finally:
                setup.milestones.append("precision_restored")
                raise RuntimeError("restore failed")

        monkeypatch.setattr(study, "ieee_precision", bad_restore)
    elif boundary == "admission":
        calls = 0

        def admission(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            return {"admitted": calls == 1}

        monkeypatch.setattr(study, "admit", admission)
    else:
        original = Path.write_text

        def write(path: Path, data: str, *args: object, **kwargs: object) -> int:
            if path == setup.output and json.loads(data).get("completed") is True:
                assert setup.milestones[-1] == "precision_restored"
                raise OSError("positive receipt write failed")
            return original(path, data)

        monkeypatch.setattr(Path, "write_text", write)
    assert not setup.run()
    saved = assert_failed(setup)
    assert (
        len(records(saved)) == 12
        and len(cast(dict[str, object], saved["profiles"])) == 4
    )
    assert not any(pending(saved).values())


def test_source_declaration_rejects_wrong_inventory_changed_bytes_and_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("original\n")
    declaration = tmp_path / "sources.json"
    declaration.write_text(json.dumps({source.name: digest(source)}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "SOURCES", (Path(source.name),))
    sha = digest(declaration)
    assert study.admit_sources(declaration, sha) == {source.name: digest(source)}
    with pytest.raises(ValueError):
        study.admit_sources(declaration, "wrong pin")
    source.write_text("changed\n")
    with pytest.raises(ValueError):
        study.admit_sources(declaration, sha)
    declaration.write_text("{}")
    with pytest.raises(ValueError):
        study.admit_sources(declaration, digest(declaration))


def test_existing_result_is_never_overwritten(setup: Setup) -> None:
    setup.output.write_text("historical evidence")
    with pytest.raises(FileExistsError):
        setup.run()
    assert setup.output.read_text() == "historical evidence" and setup.milestones == []


def test_stdlib_parent_forwards_exact_owned180_cap_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    forwarded: list[tuple[str, Path, Path, int, tuple[str, ...]]] = []

    def launch(
        module: str,
        model: Path,
        output: Path,
        *,
        timeout: int,
        worker_args: tuple[str, ...],
    ) -> int:
        forwarded.append((module, model, output, timeout, worker_args))
        return 124

    model, output, declaration = (
        tmp_path / "model",
        tmp_path / "output.json",
        tmp_path / "source.json",
    )
    monkeypatch.setattr(parent, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shape",
            str(model),
            "--output",
            str(output),
            "--sources",
            str(declaration),
            "--sources-sha256",
            "frozen",
        ],
    )
    assert parent.main() == 124
    assert forwarded == [
        (
            "benchmarks.analysis.cuda.shape.run",
            model,
            output,
            180,
            ("--sources", str(declaration), "--sources-sha256", "frozen"),
        )
    ]
    isolated = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; import benchmarks.analysis.cuda.shape.run; "
                "assert not any(x in sys.modules for x in "
                "('torch','transformers','wyat.backends.wyat.torch'))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert isolated.returncode == 0, isolated.stderr


@pytest.mark.parametrize("kind", ["empty", "partial"])
def test_native_export_failure_preserves_exact_partial_bytes(
    profile_setup: ProfileSetup, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    saved_bytes = b"" if kind == "empty" else b'{"traceEvents":['

    def export(path: str) -> None:
        assert not profile_setup.profiler.active and not profile_setup.observer.active
        Path(path).write_bytes(saved_bytes)
        if kind == "partial":
            raise OSError("native writer failed after partial bytes")

    monkeypatch.setattr(profile_setup.profiler, "export_chrome_trace", export)
    with pytest.raises((OSError, ValueError)):
        profile_setup.run()
    assert len(profile_setup.retained) == 1
    observed = profile_setup.retained[0][1]
    chrome = cast(dict[str, object], observed["chrome"])
    assert profile_setup.path.read_bytes() == saved_bytes
    assert chrome["bytes"] == len(saved_bytes)
    assert chrome["sha256"] == digest(profile_setup.path)
    assert observed["cleanup_completed"] is True
    if kind == "partial":
        assert "chrome_export_error" in observed


def test_source_mutation_between_attempts_stops_with_completed_row_retained(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = setup.output.parent / "declared.py"
    source.write_text("frozen\n")
    frozen = FrozenSources(
        (SourceFile("declared.py", str(source), "fixed", "frozen\n"),),
        "fake",
        "fixed",
    )

    def snapshot(*args: object, **kwargs: object) -> FrozenSources:
        return frozen

    def identity(path: Path) -> str:
        if path == source:
            return "fixed" if source.read_text() == "frozen\n" else "changed"
        return "fixed"

    def mutate(
        engine: BenchmarkEngine, cohort: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        source.write_text("mutated\n")
        return cycle(cohort)

    monkeypatch.setattr(study, "freeze_sources", snapshot)
    monkeypatch.setattr(study, "digest", identity)
    monkeypatch.setattr(study, "_run", mutate)
    assert not setup.run()
    saved = assert_failed(setup)
    assert len(records(saved)) == 1 and saved["profiles"] == {}
    assert sum(bool(ids) for ids in pending(saved).values()) == 11
    assert setup.milestones[-1] == "precision_restored"


@pytest.mark.parametrize("invalid", [True, 28.0, -1, "28"])
def test_math_gate_rejects_malformed_private_operator_counts(invalid: object) -> None:
    assert not study.math_dispatch_gate(
        {
            "operators": [
                {"name": "aten::_scaled_dot_product_attention_math", "count": invalid}
            ]
        }
    )
