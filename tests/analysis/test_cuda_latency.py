"""CPU-only falsification of full-cohort CUDA admission and timing history."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import nn
from transformers.integrations import sdpa_attention

from benchmarks.analysis.cuda.latency import run as parent
from benchmarks.analysis.cuda.latency import study
from benchmarks.analysis.cuda.manifest import DATA, INPUT_SHA, MODEL_ID, ROOT, digest
from benchmarks.analysis.evidence import FrozenSources
from wyat import Model, Request
from wyat.backends.kai.encoding import EncodedRequest, KaiEncoder
from wyat.backends.kai.torch import KaiScorer
from wyat.benchmark import BenchmarkCycle, BenchmarkEngine
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict
from wyat.wire import decode_json


@pytest.fixture
def cases() -> tuple[EvaluationCase, ...]:
    cohort = tuple(
        case_from_dict(decode_json(line))
        for line in (ROOT / DATA).read_text().splitlines()
    )
    assert len(cohort) == 77
    return cohort


def cycle(
    cases: tuple[EvaluationCase, ...], *, elapsed_ms: float = 100.0
) -> BenchmarkCycle:
    rows: list[EvaluationRow] = []
    for case in cases:
        scores = [
            2.0 if candidate.id in case.expected else 0.0
            for candidate in case.request.candidates
        ]
        decision = decision_from_logits(case.request, scores, MODEL_ID)
        rows.append(
            EvaluationRow(case.id, case.expected, "ok", decision, True, 1.0, None)
        )
    # Batch residence intentionally differs from the canonical cycle interval.
    return BenchmarkCycle(elapsed_ms, tuple(rows), 10, 20)


class NeverScorer:
    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        raise AssertionError("Fake runner, never allocate or run an actual model")


def engine(batch_size: int = 8) -> BenchmarkEngine:
    return BenchmarkEngine("efficient", Model(NeverScorer(), MODEL_ID), batch_size)


@pytest.mark.parametrize("batch,expected_chunks", [(1, [1] * 77), (8, [8] * 9 + [5])])
def test_collect_sums_batch_cycles_without_row_or_telemetry_inflation(
    cases: tuple[EvaluationCase, ...],
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    expected_chunks: list[int],
) -> None:
    history: list[str] = []
    sizes: list[int] = []
    clock_ns = 0

    def now() -> int:
        return clock_ns

    def verify() -> None:
        nonlocal clock_ns
        history.append("verify")
        clock_ns += 4_000_000_000

    def occupancy(**kwargs: object) -> dict[str, object]:
        nonlocal clock_ns
        history.append("occupancy")
        clock_ns += 5_000_000_000
        return {"observed": True}

    def run(
        selected: BenchmarkEngine, chunk: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal clock_ns
        assert history[-1] == "occupancy"
        history.append("run")
        sizes.append(len(chunk))
        clock_ns += 100_000_000
        return cycle(chunk)

    def write() -> None:
        nonlocal clock_ns
        history.append("write")
        clock_ns += 2_000_000_000

    monkeypatch.setattr(time, "perf_counter_ns", now)
    monkeypatch.setattr(study, "occupancy", occupancy)
    monkeypatch.setattr(study, "_run", run)
    payload: dict[str, object] = {
        "cycles": [],
        "unknown_case_ids": {"probe": [case.id for case in cases]},
    }
    expected = cycle(cases)
    actual = study.collect(
        engine(batch),
        cases,
        expected,
        expected,
        key="probe",
        payload=payload,
        write=write,
        verify=verify,
    )
    assert sizes == expected_chunks
    assert actual.latency_ms == 100.0 * len(expected_chunks)
    assert len(actual.rows) == 77
    assert [row.id for row in actual.rows] == [case.id for case in cases]
    assert sum(row.latency_ms for row in actual.rows) == 77.0
    assert payload["unknown_case_ids"] == {"probe": []}
    record = cast(list[dict[str, object]], payload["cycles"])[0]
    assert len(cast(list[object], record["chunks"])) == len(expected_chunks)
    assert cast(float, record["cohort_wall_ms"]) > actual.latency_ms * 50
    assert (
        history.count("verify")
        == history.count("occupancy")
        == history.count("run")
        == len(expected_chunks)
    )


@pytest.mark.parametrize(
    "failure", ["backend", "drift", "missing", "ownership", "source"]
)
def test_collect_failure_preserves_completed_chunks_and_pending_cases(
    cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    calls = 0
    ownership_calls = 0
    source_calls = 0

    def verify() -> None:
        nonlocal source_calls
        source_calls += 1
        if failure == "source" and source_calls == 2:
            raise ValueError("source changed between batches")

    def occupancy(**kwargs: object) -> dict[str, object]:
        nonlocal ownership_calls
        ownership_calls += 1
        if failure == "ownership" and ownership_calls == 2:
            raise ValueError("foreign compute owner")
        return {}

    def run(
        selected: BenchmarkEngine, chunk: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal calls
        calls += 1
        current = cycle(chunk)
        if calls == 2:
            if failure == "backend":
                return replace(
                    current,
                    rows=tuple(
                        replace(
                            row,
                            status="error",
                            prediction=None,
                            correct=False,
                            error="unsupported backend",
                        )
                        for row in current.rows
                    ),
                )
            if failure == "missing":
                return replace(current, rows=current.rows[:-1])
            if failure == "drift":
                return replace(
                    current,
                    rows=tuple(
                        replace(
                            row,
                            prediction=decision_from_logits(
                                case.request,
                                [0.0] * len(case.request.candidates),
                                MODEL_ID,
                            ),
                        )
                        for row, case in zip(current.rows, chunk, strict=True)
                    ),
                )
        return current

    writes: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "cycles": [],
        "unknown_case_ids": {"probe": [case.id for case in cases]},
    }

    def write() -> None:
        writes.append(cast(dict[str, object], json.loads(json.dumps(payload))))

    monkeypatch.setattr(study, "occupancy", occupancy)
    monkeypatch.setattr(study, "_run", run)
    reference = cycle(cases)
    with pytest.raises(ValueError):
        study.collect(
            engine(),
            cases,
            reference,
            reference,
            key="probe",
            payload=payload,
            write=write,
            verify=verify,
        )
    expected_calls = 1 if failure in {"ownership", "source"} else 2
    assert calls == expected_calls
    record = cast(list[dict[str, object]], payload["cycles"])[0]
    assert len(cast(list[object], record["chunks"])) == expected_calls
    assert "cycle" not in record
    completed = (
        8 if failure in {"ownership", "source"} else 15 if failure == "missing" else 16
    )
    assert cast(dict[str, list[str]], payload["unknown_case_ids"])["probe"] == [
        case.id for case in cases[completed:]
    ]
    assert writes


@pytest.mark.parametrize("elapsed", [0.0, -1.0, math.inf, math.nan])
def test_collect_rejects_invalid_timing_after_preserving_raw_rows(
    cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch, elapsed: float
) -> None:
    def run(
        selected: BenchmarkEngine, chunk: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        return cycle(chunk, elapsed_ms=elapsed)

    monkeypatch.setattr(study, "_run", run)
    monkeypatch.setattr(study, "occupancy", lambda **kwargs: {})
    payload: dict[str, object] = {"cycles": [], "unknown_case_ids": {"probe": []}}
    reference = cycle(cases)

    def write() -> None:
        json.dumps(payload, allow_nan=False)

    with pytest.raises(ValueError, match="finite and positive"):
        study.collect(
            engine(),
            cases,
            reference,
            reference,
            key="probe",
            payload=payload,
            write=write,
        )
    record = cast(list[dict[str, object]], payload["cycles"])[0]
    assert len(cast(list[object], record["chunks"])) == 1
    assert "cycle" not in record
    raw = cast(list[dict[str, object]], record["chunks"])[0]
    if not math.isfinite(elapsed):
        assert raw["latency_ms"] is None
        assert raw["invalid_timing"] == {"latency_ms": repr(elapsed)}
    assert len(cast(list[object], raw["rows"])) == 8
    json.dumps(payload, allow_nan=False)


@given(
    scale=st.floats(
        min_value=0.01, max_value=10_000, allow_nan=False, allow_infinity=False
    )
)
def test_performance_is_invariant_to_time_units(scale: float) -> None:
    baseline = tuple(BenchmarkCycle(scale * 2, (), 0, 0) for _ in range(6))
    candidate = tuple(BenchmarkCycle(scale, (), 0, 0) for _ in range(6))
    report = study.performance(baseline, candidate)
    assert report["paired_ratios"] == [2.0] * 6
    assert report["median"] == report["p05"] == report["aggregate"] == 2.0
    assert report["cohort_p95_ratio"] == 0.5
    assert report["passed"] is True


@pytest.mark.parametrize("elapsed", [0.0, -1.0, math.inf, math.nan])
def test_performance_rejects_invalid_observations(elapsed: float) -> None:
    baseline = tuple(BenchmarkCycle(2.0, (), 0, 0) for _ in range(6))
    candidate = tuple(BenchmarkCycle(elapsed, (), 0, 0) for _ in range(6))
    with pytest.raises(ValueError, match="finite and positive"):
        study.performance(baseline, candidate)


@pytest.mark.parametrize("count", [0, 1, 5, 7])
def test_performance_requires_exactly_six_pairs(count: int) -> None:
    observations = tuple(BenchmarkCycle(1.0, (), 0, 0) for _ in range(count))
    with pytest.raises(ValueError, match="Six complete"):
        study.performance(observations, observations)


@pytest.mark.parametrize(
    "left,right,passed,failed_gate",
    [
        ([105.0] * 6, [100.0] * 6, True, None),
        ([104.99] * 6, [100.0] * 6, False, "median"),
        ([90.0, 130.0, 130.0, 130.0, 130.0, 130.0], [100.0] * 6, False, "p05"),
        (
            [1000.0, 900.0, 900.0, 900.0, 900.0, 900.0],
            [1300.0, 500.0, 500.0, 500.0, 500.0, 500.0],
            False,
            "cohort_p95_ratio",
        ),
        (
            [10_000.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [11_000.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            False,
            "aggregate",
        ),
    ],
)
def test_all_four_predeclared_speed_conditions_are_required(
    left: list[float], right: list[float], passed: bool, failed_gate: str | None
) -> None:
    report = study.performance(
        tuple(BenchmarkCycle(value, (), 0, 0) for value in left),
        tuple(BenchmarkCycle(value, (), 0, 0) for value in right),
    )
    assert report["passed"] is passed
    if failed_gate == "median":
        assert cast(float, report[failed_gate]) < 1.05
    elif failed_gate in {"p05", "aggregate"}:
        assert cast(float, report[failed_gate]) <= 1.0
    elif failed_gate == "cohort_p95_ratio":
        assert cast(float, report[failed_gate]) > 1.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed", False),
        ("exit_code", 1),
        ("timed_out", True),
        ("timeout_seconds", 179),
    ],
)
def test_parent_success_requires_every_terminal_condition(
    tmp_path: Path, field: str, value: object
) -> None:
    main = tmp_path / "history.json"
    wall: dict[str, object] = {
        "completed": True,
        "exit_code": 0,
        "timed_out": False,
        "timeout_seconds": 180,
    }
    main.with_suffix(".wall.json").write_text(json.dumps(wall))
    assert study.require_parent(main) == wall
    wall[field] = value
    main.with_suffix(".wall.json").write_text(json.dumps(wall))
    with pytest.raises(ValueError, match="terminal cleanup"):
        study.require_parent(main)


@pytest.mark.parametrize("change", ["missing", "order", "score", "gold", "error"])
def test_history_reconstructs_scores_and_rejects_inconsistent_claims(
    cases: tuple[EvaluationCase, ...], change: str
) -> None:
    saved = cycle(cases).to_dict()
    assert len(study.read_cycle(saved, cases).rows) == 77
    rows = cast(list[dict[str, object]], saved["rows"])
    if change == "missing":
        rows.pop()
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "score":
        prediction = cast(dict[str, object], rows[0]["prediction"])
        prediction["scores"] = [0.0] * len(cases[0].request.candidates)
    elif change == "gold":
        rows[0]["expected"] = ["invalid"]
    else:
        rows[0]["status"] = "error"
    with pytest.raises(ValueError):
        study.read_cycle(saved, cases)


def test_latency_source_declaration_preserves_exact_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("initial\n")
    declaration = tmp_path / "source.json"
    declaration.write_text(json.dumps({"source.py": digest(source)}))
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "SOURCES", (Path("source.py"),))
    declared_sha = digest(declaration)
    assert study.admit_sources(declaration, declared_sha) == {
        "source.py": digest(source)
    }
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="exact latency source"):
        study.admit_sources(declaration, declared_sha)
    with pytest.raises(ValueError, match="declaration changed"):
        study.admit_sources(declaration, "unknown")


@dataclass
class Setup:
    path: Path
    output: Path
    cases: tuple[EvaluationCase, ...]
    calls: list[tuple[str, int]]
    milestones: list[str]
    candidate_ms: float = 50.0

    def run(self, mode: study.Mode = "confirm", batch_size: int = 8) -> bool:
        return study.study(
            self.path,
            self.output,
            manifest=self.path,
            manifest_sha256="fixed",
            sources=self.path,
            sources_sha256="fixed",
            control=self.path,
            control_sha256="fixed",
            mode=mode,
            batch_size=batch_size,
            confirmation=self.path if mode == "timing" else None,
            confirmation_sha256="fixed" if mode == "timing" else None,
            confirmation_wall_sha256="fixed" if mode == "timing" else None,
        )


@dataclass
class FakeBase:
    backbone: nn.Module
    head: nn.Module
    device: torch.device

    def encode_request(self, request: Request) -> EncodedRequest:
        return EncodedRequest((1, 2), (0, 1), 0)


@pytest.fixture
def setup(
    tmp_path: Path, cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch
) -> Setup:
    data = tmp_path / DATA
    data.parent.mkdir(parents=True)
    data.write_bytes((ROOT / DATA).read_bytes())
    setup = Setup(tmp_path / "identity", tmp_path / "output.json", cases, [], [])
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "MANIFEST_SHA", "fixed")
    monkeypatch.setattr(study, "CONTROL_SHA", "fixed")
    monkeypatch.setattr(study, "CONTROL_WALL_SHA", "fixed")
    for flag in (
        "_is_torch_mps_available",
        "_is_torch_xpu_available",
        "_is_torch_npu_available",
    ):
        monkeypatch.setattr(sdpa_attention, flag, False)

    def fixed(*args: object, **kwargs: object) -> str:
        return "fixed"

    def admission(*args: object, **kwargs: object) -> dict[str, object]:
        return {"admitted": True}

    def hashes(*args: object, **kwargs: object) -> dict[str, str]:
        return {}

    def control(*args: object, **kwargs: object) -> dict[str, str]:
        return {"sha256": "fixed", "wall_path": str(setup.path), "wall_sha256": "fixed"}

    def frozen(*args: object, **kwargs: object) -> FrozenSources:
        return FrozenSources((), "fake", "fixed")

    def source_paths() -> dict[str, Path]:
        return {}

    def artifact(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(model_id=MODEL_ID)

    def reference(
        path: Path,
        cohort: tuple[EvaluationCase, ...],
        encoder: object,
        **kwargs: object,
    ) -> BenchmarkCycle:
        assert len(cohort) == 77
        setup.milestones.append(
            "token_validation"
            if isinstance(encoder, study.ReferenceInput)
            else "loaded_validation"
        )
        return cycle(cohort)

    def cuda_cycle(path: Path, cohort: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
        return cycle(cohort)

    for name, replacement in (
        ("digest", fixed),
        ("admit", admission),
        ("admit_sources", hashes),
        ("admit_attention", hashes),
        ("admit_control", control),
        ("admit_screens", hashes),
        ("admit_confirmation", hashes),
        ("freeze_sources", frozen),
        ("source_paths", source_paths),
        ("load_artifact", artifact),
        ("occupancy", admission),
        ("read_reference", reference),
        ("control_cycle", cuda_cycle),
    ):
        monkeypatch.setattr(study, name, replacement)
    base = FakeBase(nn.Module(), nn.Module(), torch.device("cpu"))

    def load(*args: object, **kwargs: object) -> FakeBase:
        assert kwargs == {"device": "cpu", "max_length": 512, "token_cache_size": 0}
        assert setup.milestones == ["token_validation"]
        setup.milestones.append("load")
        return base

    def encoder(*args: object, **kwargs: object) -> FakeBase:
        assert kwargs == {"max_length": 512, "token_cache_size": 0}
        return base

    def no_device(module: nn.Module, *args: object, **kwargs: object) -> nn.Module:
        assert setup.milestones[:3] == ["token_validation", "load", "loaded_validation"]
        setup.milestones.append("allocation")
        return module

    def noop(*args: object, **kwargs: object) -> None:
        return None

    def factory(*args: object, **kwargs: object) -> NeverScorer:
        return NeverScorer()

    @contextmanager
    def precision() -> Iterator[None]:
        setup.milestones.append("precision_enter")
        try:
            yield
        finally:
            setup.milestones.append("precision_restored")

    def forbidden() -> None:
        raise AssertionError("No GPU allocation in CPU protocol tests")

    def zero() -> int:
        return 0

    monkeypatch.setattr(KaiScorer, "from_directory", load)
    monkeypatch.setattr(KaiEncoder, "from_artifact", encoder)
    monkeypatch.setattr(nn.Module, "to", no_device)
    monkeypatch.setattr(study, "require_fp32", noop)
    monkeypatch.setattr(study, "FP32Scorer", factory)
    monkeypatch.setattr(study, "efficient_scorer", factory)
    monkeypatch.setattr(study, "ieee_precision", precision)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", noop)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", zero)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", zero)

    def ordinary(
        selected: BenchmarkEngine, cohort: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        setup.calls.append((selected.name, len(cohort)))
        return cycle(
            cohort, elapsed_ms=100 if selected.name == "math" else setup.candidate_ms
        )

    def profile(
        scorer: object, cohort: tuple[EvaluationCase, ...], *, name: str
    ) -> tuple[BenchmarkCycle, dict[str, object]]:
        setup.calls.append((f"profile_{name}", len(cohort)))
        return cycle(cohort), {"representative": True}

    def dispatch(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(study, "_run", ordinary)
    monkeypatch.setattr(study, "profile_batch", profile)
    monkeypatch.setattr(study, "efficient_dispatch_gate", dispatch)
    return setup


@pytest.mark.parametrize("batch", [1, 8])
def test_full77_confirmation_precedes_representative_profiles(
    setup: Setup, batch: int
) -> None:
    assert setup.run(batch_size=batch)
    chunks = [1] * 77 if batch == 1 else [8] * 9 + [5]
    assert setup.calls == [
        (arm, size) for arm in ("math", "efficient") for size in chunks
    ] + [("profile_math", batch), ("profile_efficient", batch)]
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] and saved["quality_passed"] and saved["dispatch_passed"]
    assert saved["accepted"] is False
    assert (
        saved["expected_ordinary_rows"] == 154
        and saved["expected_profile_rows"] == 2 * batch
    )
    assert [record["key"] for record in saved["cycles"]] == [
        "confirm_math",
        "confirm_efficient",
    ]
    assert sum(len(record["cycle"]["rows"]) for record in saved["cycles"]) == 154
    assert all(not ids for ids in saved["unknown_case_ids"].values())
    assert setup.milestones[-1] == "precision_restored"


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("speed_passes", [True, False])
def test_six_pairs_preserve_ab_ba_order_even_when_speed_fails(
    setup: Setup, batch: int, speed_passes: bool
) -> None:
    if not speed_passes:
        setup.candidate_ms = 100
    assert setup.run("timing", batch) is speed_passes
    saved = json.loads(setup.output.read_text())
    expected = ["warmup_math", "warmup_efficient"]
    expected.extend(
        f"pair{index}_{arm}"
        for index in range(6)
        for arm in (("math", "efficient") if index % 2 == 0 else ("efficient", "math"))
    )
    assert [record["key"] for record in saved["cycles"]] == expected
    assert saved["completed"] is True and saved["quality_passed"] is True
    assert saved["performance_passed"] is speed_passes and saved["accepted"] is False
    assert saved["dispatch_passed"] is True
    assert (
        saved["dispatch_evidence_scope"]
        == "prior representative confirmation profile; timed calls unprofiled"
    )
    assert (
        saved["expected_ordinary_rows"] == 1078 and saved["expected_profile_rows"] == 0
    )
    assert sum(len(record["cycle"]["rows"]) for record in saved["cycles"]) == 1078
    assert all(len(record["cycle"]["rows"]) == 77 for record in saved["cycles"])
    assert not any(name.startswith("profile") for name, _ in setup.calls)
    assert all(not ids for ids in saved["unknown_case_ids"].values())
    assert saved["performance"]["paired_ratios"] == [2.0 if speed_passes else 1.0] * 6
    assert len(saved["report"]["baseline"]["cycles"]) == 6
    assert len(saved["report"]["candidate"]["cycles"]) == 6
    assert setup.milestones[-1] == "precision_restored"


@pytest.mark.parametrize("mode", ["confirm", "timing"])
def test_token_validation_failure_precedes_weights_and_device(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, mode: study.Mode
) -> None:
    def invalid(*args: object, **kwargs: object) -> BenchmarkCycle:
        raise ValueError("canonical token endpoints changed")

    monkeypatch.setattr(study, "read_reference", invalid)
    assert not setup.run(mode)
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert "token endpoints" in saved["error"]
    assert setup.milestones == [] and setup.calls == []
    assert saved["unknown_case_ids"]
    assert all(
        ids == [case.id for case in setup.cases[:8]]
        if key.endswith("profile")
        else ids == [case.id for case in setup.cases]
        for key, ids in saved["unknown_case_ids"].items()
    )


@pytest.mark.parametrize("failure", ["source", "precision", "write"])
def test_late_failure_preserves_evidence_without_positive_flags(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "source":

        def changed(*args: object, **kwargs: object) -> dict[str, str]:
            if setup.calls:
                raise ValueError("latency sources changed")
            return {}

        monkeypatch.setattr(study, "admit_sources", changed)
    elif failure == "precision":

        @contextmanager
        def broken() -> Iterator[None]:
            yield
            raise RuntimeError("precision restore failed")

        monkeypatch.setattr(study, "ieee_precision", broken)
    else:
        original = Path.write_text
        seen = False

        def write(path: Path, text: str, *args: object, **kwargs: object) -> int:
            nonlocal seen
            if json.loads(text).get("completed") is True and not seen:
                seen = True
                raise OSError("final write failed")
            return original(path, text)

        monkeypatch.setattr(Path, "write_text", write)
    assert not setup.run("timing")
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert saved["dispatch_passed"] is False and saved["performance_passed"] is False
    assert saved["accepted"] is False and len(saved["cycles"]) == 14
    assert saved["report"]["quality"]["passed"] is True


def test_parent_import_is_framework_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; import benchmarks.analysis.cuda.latency.run; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_parent_forwards_owned_cap_and_both_confirmation_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path, Path, dict[str, object]]] = []

    def launch(module: str, model: Path, output: Path, **kwargs: object) -> int:
        calls.append((module, model, output, kwargs))
        return 124

    monkeypatch.setattr(parent, "launch", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "latency",
            "model",
            "--output",
            str(tmp_path / "result.json"),
            "--mode",
            "timing",
            "--batch-size",
            "8",
            "--manifest",
            "manifest",
            "--manifest-sha256",
            "manifest-pin",
            "--sources",
            "sources",
            "--sources-sha256",
            "sources-pin",
            "--control",
            "control",
            "--control-sha256",
            "control-pin",
            "--confirmation",
            "confirmation",
            "--confirmation-sha256",
            "confirmation-pin",
            "--confirmation-wall-sha256",
            "parent-pin",
        ],
    )
    assert parent.main() == 124
    assert len(calls) == 1
    module, model, output, options = calls[0]
    assert module == "benchmarks.analysis.cuda.latency.run"
    assert model == Path("model") and output == tmp_path / "result.json"
    assert options["timeout"] == 180
    args = cast(list[str], options["worker_args"])
    assert args[args.index("--confirmation-sha256") + 1] == "confirmation-pin"
    assert args[args.index("--confirmation-wall-sha256") + 1] == "parent-pin"
    assert args[args.index("--mode") + 1] == "timing"


@dataclass
class Confirmation:
    path: Path
    cases: tuple[EvaluationCase, ...]
    saved: dict[str, object]

    def admit(self, wall_pin: str | None = "current") -> dict[str, str]:
        self.path.write_text(json.dumps(self.saved))
        reference = cycle(self.cases)
        return study.admit_confirmation(
            self.path,
            digest(self.path),
            wall_sha256=digest(self.path.with_suffix(".wall.json"))
            if wall_pin == "current"
            else wall_pin,
            batch_size=8,
            source_hashes={},
            reference=reference,
            cuda=reference,
            cases=self.cases,
        )


@pytest.fixture
def confirmation(
    tmp_path: Path, cases: tuple[EvaluationCase, ...], monkeypatch: pytest.MonkeyPatch
) -> Confirmation:
    path = tmp_path / "confirm.json"
    source = tmp_path / "source.py"
    source.write_text("frozen\n")
    archive_data = b'{"files": []}\n'

    archive_sha = hashlib.sha256(archive_data).hexdigest()
    path.with_suffix(f".sources-{archive_sha[:16]}.json").write_bytes(archive_data)
    path.with_suffix(".wall.json").write_text(
        json.dumps(
            {
                "completed": True,
                "exit_code": 0,
                "timed_out": False,
                "timeout_seconds": 180,
            }
        )
    )
    saved: dict[str, object] = {
        "mode": "confirm",
        "batch_size": 8,
        "completed": True,
        "quality_passed": True,
        "dispatch_passed": True,
        "accepted": False,
        "case_ids": [case.id for case in cases],
        "input_sha256": INPUT_SHA,
        "new_source_hashes": {},
        "unknown_case_ids": {"confirm_math": [], "confirm_efficient": []},
        "source_sha256": {"source": digest(source)},
        "source_archive": {"sha256": archive_sha},
        "cycles": [
            {"key": f"confirm_{arm}", "cycle": cycle(cases).to_dict()}
            for arm in ("math", "efficient")
        ],
        "math_profile_cycle": cycle(cases[:8]).to_dict(),
        "efficient_profile_cycle": cycle(cases[:8]).to_dict(),
        "efficient_profile": {"reviewed": True},
    }

    def paths() -> dict[str, Path]:
        return {"source": source}

    def dispatch(*args: object, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(study, "source_paths", paths)
    monkeypatch.setattr(study, "efficient_dispatch_gate", dispatch)
    return Confirmation(path, cases, saved)


@pytest.mark.parametrize("wall_pin", [None, "changed"])
def test_timing_requires_independently_pinned_confirmation_wall(
    confirmation: Confirmation, wall_pin: str | None
) -> None:
    assert len(confirmation.admit()) == 3
    with pytest.raises(ValueError, match="reviewed confirmation"):
        confirmation.admit(wall_pin)


@pytest.mark.parametrize(
    "change",
    [
        "batch",
        "completion",
        "unknown",
        "cohort",
        "missing_row",
        "source",
        "profile",
        "parent",
    ],
)
def test_confirmation_flags_do_not_replace_full77_replay(
    confirmation: Confirmation, change: str
) -> None:
    assert len(confirmation.admit()) == 3
    saved = confirmation.saved
    if change == "batch":
        saved["batch_size"] = 1
    elif change == "completion":
        saved["quality_passed"] = False
    elif change == "unknown":
        saved["unknown_case_ids"] = {"confirm_efficient": [confirmation.cases[-1].id]}
    elif change == "cohort":
        saved["case_ids"] = list(reversed([case.id for case in confirmation.cases]))
    elif change == "missing_row":
        records = cast(list[dict[str, object]], saved["cycles"])
        rows = cast(list[object], cast(dict[str, object], records[1]["cycle"])["rows"])
        rows.pop()
    elif change == "source":
        saved["source_sha256"] = {"source": "different"}
    elif change == "profile":
        saved["efficient_profile_cycle"] = cycle(confirmation.cases[:7]).to_dict()
    else:
        wall = confirmation.path.with_suffix(".wall.json")
        value = json.loads(wall.read_text())
        value["timed_out"] = True
        wall.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        confirmation.admit()


@pytest.mark.parametrize("mode", ["confirm", "timing"])
def test_ordinary_backend_failure_aborts_later_phases_and_restores_context(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, mode: study.Mode
) -> None:
    efficient_calls = 0

    def failed(
        selected: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        nonlocal efficient_calls
        setup.calls.append((selected.name, len(cases)))
        current = cycle(cases)
        if selected.name == "efficient":
            efficient_calls += 1
            if efficient_calls == 2:
                return replace(
                    current,
                    rows=tuple(
                        replace(
                            row,
                            status="error",
                            prediction=None,
                            correct=False,
                            error="backend failed",
                        )
                        for row in current.rows
                    ),
                )
        return current

    monkeypatch.setattr(study, "_run", failed)
    assert not setup.run(mode)
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert saved["dispatch_passed"] is False and saved["performance_passed"] is False
    assert len(saved["cycles"]) == 2 and len(saved["cycles"][1]["chunks"]) == 2
    phase = "confirm" if mode == "confirm" else "warmup"
    assert saved["unknown_case_ids"][f"{phase}_efficient"] == [
        case.id for case in setup.cases[16:]
    ]
    assert not any(name.startswith("profile") for name, _ in setup.calls)
    assert setup.milestones[-1] == "precision_restored"


def test_nonfinite_batch_interval_emits_json_safe_negative_receipt(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid(
        selected: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        return cycle(cases, elapsed_ms=math.nan)

    monkeypatch.setattr(study, "_run", invalid)
    assert not setup.run("timing")
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert saved["performance_passed"] is False
    first = saved["cycles"][0]["chunks"][0]
    assert first["latency_ms"] is None and first["invalid_timing"] == {
        "latency_ms": "nan"
    }
    assert len(first["rows"]) == 8 and "finite and positive" in saved["error"]
    assert saved["unknown_case_ids"]["warmup_math"] == [
        case.id for case in setup.cases[8:]
    ]


@pytest.mark.parametrize("failure", ["quality", "dispatch"])
def test_representative_profile_failure_cannot_publish_confirmation_success(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    def profiled(
        scorer: object, cases: tuple[EvaluationCase, ...], *, name: str
    ) -> tuple[BenchmarkCycle, dict[str, object]]:
        setup.calls.append((f"profile_{name}", len(cases)))
        return cycle(cases[:-1] if failure == "quality" else cases), {}

    def unsupported(*args: object, **kwargs: object) -> bool:
        return False

    monkeypatch.setattr(study, "profile_batch", profiled)
    monkeypatch.setattr(study, "efficient_dispatch_gate", unsupported)
    assert not setup.run()
    saved = json.loads(setup.output.read_text())
    assert saved["completed"] is False and saved["quality_passed"] is False
    assert saved["dispatch_passed"] is False and saved["accepted"] is False
    assert sum(len(record["cycle"]["rows"]) for record in saved["cycles"]) == 154
    if failure == "quality":
        assert saved["unknown_case_ids"]["math_profile"] == [setup.cases[7].id]
        assert saved["unknown_case_ids"]["efficient_profile"] == [
            case.id for case in setup.cases[:8]
        ]
        assert not any(name == "profile_efficient" for name, _ in setup.calls)
    assert setup.milestones[-1] == "precision_restored"
