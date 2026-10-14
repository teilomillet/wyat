"""Pure history, benchmark reuse and failure-boundary tests; no model/timing."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from transformers.integrations import sdpa_attention

from benchmarks.analysis.torch.prefix.latency import study as runner
from benchmarks.analysis.torch.prefix.latency.study import Comparator
from benchmarks.analysis.torch.prefix.memory.study import evidence_path
from benchmarks.analysis.torch.prefix.pairs import admit_history, digest
from wyat import Request
from wyat.backends.wyat.artifact import WyatArtifact
from wyat.backends.wyat.encoding import EncodedRequest
from wyat.backends.wyat.torch import WyatScorer
from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    BenchmarkReport,
    benchmark_pair,
)
from wyat.core import decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow
from wyat.wire import read_object


@pytest.fixture
def fake_study(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model outputs and controlled times test policy, never learned quality."""
    history = admit_history()
    identifier = cast(str, history["model_id"])
    prefix = tuple(cast(list[int], history["prefix_tokens"]))

    class FakeBase:
        model_id = identifier

        def encode_request(self, request: Request) -> EncodedRequest:
            width = len(request.candidates)
            return EncodedRequest(
                prefix + tuple(range(width + 2)),
                tuple(range(25, 25 + width)),
                26 + width,
            )

        def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
            return [
                [float(-i) for i in range(len(request.candidates))]
                for request in requests
            ]

    class FakePrefix(FakeBase):
        def __init__(self, base: object, tokens: tuple[int, ...], arm: str) -> None:
            assert tokens == prefix and arm in ("retained", "one-use")
            self.arm = arm

        def admits(self, row: EncodedRequest) -> bool:
            return row.ids[:25] == prefix

    def load(*args: object, **kwargs: object) -> FakeBase:
        return FakeBase()

    def artifact(path: Path, max_length: int) -> WyatArtifact:
        return WyatArtifact(path, {}, 8, 255, identifier)

    def reference(
        path: Path,
        cases: tuple[EvaluationCase, ...],
        encoder: object,
        *,
        dataset_sha256: str,
    ) -> BenchmarkCycle:
        rows = tuple(
            EvaluationRow(
                case.id,
                case.expected,
                "ok",
                decision,
                decision.selected in case.expected,
                0,
                None,
            )
            for case in cases
            for decision in [
                decision_from_logits(
                    case.request,
                    tuple(float(-i) for i in range(len(case.request.candidates))),
                    identifier,
                )
            ]
        )
        return BenchmarkCycle(0, rows, 0, 0)

    def pair(
        baseline: BenchmarkEngine,
        candidate: BenchmarkEngine,
        cases: Sequence[EvaluationCase],
        *,
        protocol: BenchmarkProtocol,
    ) -> BenchmarkReport:
        actual = benchmark_pair(baseline, candidate, cases, protocol=protocol)
        return replace(
            actual,
            baseline_cycles=tuple(
                replace(cycle, latency_ms=1000.0) for cycle in actual.baseline_cycles
            ),
            candidate_cycles=tuple(
                replace(cycle, latency_ms=970.0) for cycle in actual.candidate_cycles
            ),
        )

    monkeypatch.setattr(WyatScorer, "from_directory", load)
    monkeypatch.setattr(runner, "load_artifact", artifact)
    monkeypatch.setattr(runner, "MemoryScorer", FakePrefix)
    monkeypatch.setattr(runner, "read_reference", reference)
    monkeypatch.setattr(runner, "benchmark_pair", pair)
    monkeypatch.setattr(sdpa_attention, "_is_torch_mps_available", True)
    monkeypatch.setattr(sdpa_attention, "_is_torch_greater_or_equal_than_2_13", True)


@pytest.mark.parametrize("comparator", ["retained", "full"])
@pytest.mark.parametrize("batch", [1, 8])
def test_complete_report_keeps_all_cycles_and_legacy_reference_checks(
    tmp_path: Path,
    fake_study: None,
    comparator: Comparator,
    batch: int,
) -> None:
    output = tmp_path / "out.json"
    assert runner.study(
        tmp_path / "model", output, comparator=comparator, batch_size=batch
    )
    result = read_object(output)
    assert (
        result["completed"]
        is result["quality_passed"]
        is result["selection_passed"]
        is True
    )
    assert result["accepted"] is False and result["benchmark_returned"] is True
    checks = cast(dict[str, list[dict[str, object]]], result["reference_checks"])
    assert [len(rows) for rows in checks.values()] == [1, 1, 5, 5]
    assert (
        sum(cast(int, item["total"]) for rows in checks.values() for item in rows)
        == 924
    )
    assert all(item["passed"] is True for rows in checks.values() for item in rows)
    report = cast(dict[str, object], result["report"])
    assert (
        cast(dict[str, object], report["protocol"])["paired_order"]
        == "alternating baseline/candidate, candidate/baseline"
    )
    assert (
        result["baseline_prefix_build_ms"] is None
        if comparator == "full"
        else result["baseline_prefix_build_ms"] is not None
    )


@pytest.mark.parametrize("kind", ["pair-exception", "final-write"])
def test_unreturned_pair_or_late_failure_never_claims_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
    kind: str,
) -> None:
    output = tmp_path / "out.json"
    prior = torch.get_num_threads()
    if kind == "pair-exception":

        def fail(*args: object, **kwargs: object) -> None:
            current = read_object(output)
            assert (
                current["phase"] == "benchmarking"
                and current["benchmark_returned"] is False
            )
            assert (
                current["raw_partial_cycles_available"] is False
                and "report" not in current
            )
            raise RuntimeError("pair failed before returned cycles")

        monkeypatch.setattr(runner, "benchmark_pair", fail)
    else:
        original = Path.write_text

        def write(path: Path, text: str) -> int:
            if path.name == "out.json" and json.loads(text).get("completed") is True:
                raise RuntimeError("final write failed")
            return original(path, text)

        monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(RuntimeError, match="failed"):
        runner.study(tmp_path / "model", output, comparator="retained", batch_size=8)
    result = read_object(output)
    assert all(
        result[name] is False
        for name in (
            "completed",
            "quality_passed",
            "performance_passed",
            "selection_passed",
            "accepted",
        )
    )
    assert torch.get_num_threads() == prior
    if kind == "final-write":
        assert result["benchmark_returned"] is True and "report" in result
    else:
        assert result["benchmark_returned"] is False and "report" not in result


def test_two_arms_drifting_together_fail_frozen_cpu_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
) -> None:
    original = cast(type[object], vars(runner)["MemoryScorer"])

    def changed(self: object, requests: Sequence[Request]) -> list[list[float]]:
        return [
            [float(-i) + 0.25 for i in range(len(request.candidates))]
            for request in requests
        ]

    monkeypatch.setattr(original, "score_many", changed)
    output = tmp_path / "out.json"
    assert not runner.study(
        tmp_path / "model", output, comparator="retained", batch_size=8
    )
    result = read_object(output)
    assert cast(dict[str, object], result["report"])["passed"] is True  # Pair agrees.
    assert result["completed"] is True and result["quality_passed"] is False
    assert result["selection_passed"] is False


@pytest.mark.parametrize("evidence", ["receipt", "journal", "wall", "archive"])
def test_any_confirmed_memory_history_drift_rejected(
    evidence: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = read_object(
        runner.ROOT / "benchmarks/analysis/results/torch-prefix-memory-review.json"
    )
    child = cast(list[dict[str, object]], review["children"])[0]
    target = evidence_path(
        cast(dict[str, dict[str, str]], child["evidence"])[evidence]["path"]
    )
    monkeypatch.setattr(
        runner, "digest", lambda path: "0" * 64 if path == target else digest(path)
    )
    with pytest.raises(ValueError, match="memory receipt/journal/wall/archive"):
        runner.admit_resource()


def test_independent_memory_review_drift_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = runner.ROOT / "benchmarks/analysis/results/torch-prefix-memory-review.json"
    monkeypatch.setattr(
        runner, "digest", lambda path: "0" * 64 if path == target else digest(path)
    )
    with pytest.raises(ValueError, match="six-child memory review"):
        runner.admit_resource()


def test_artifact_identity_blocks_model_allocation(
    tmp_path: Path,
    fake_study: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def artifact(path: Path, max_length: int) -> WyatArtifact:
        return WyatArtifact(path, {}, 8, 255, "changed-model")

    def load(*args: object, **kwargs: object) -> None:
        pytest.fail("Mismatched artifact must never allocate model")

    monkeypatch.setattr(runner, "load_artifact", artifact)
    monkeypatch.setattr(WyatScorer, "from_directory", load)
    with pytest.raises(ValueError, match="before load"):
        runner.study(
            tmp_path / "model", tmp_path / "out.json", comparator="full", batch_size=1
        )


@pytest.mark.parametrize(
    "comparator,batch", [("bad", 8), ("retained", True), ("full", 2)]
)
def test_invalid_plan_has_no_effects(
    tmp_path: Path, comparator: str, batch: int
) -> None:
    with pytest.raises(ValueError, match="requires"):
        runner.study(
            tmp_path / "model",
            tmp_path / "out.json",
            comparator=cast(Comparator, comparator),
            batch_size=batch,
        )
    assert not (tmp_path / "out.json").exists()


def test_parent_import_is_torch_free() -> None:
    command = (
        "import sys; import benchmarks.analysis.torch.prefix.latency.run; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_only_warmup_drift_is_not_hidden_by_passing_measured_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
) -> None:
    from collections.abc import Callable

    original = cast(Callable[..., BenchmarkReport], vars(runner)["benchmark_pair"])

    def shifted(*args: object, **kwargs: object) -> BenchmarkReport:
        report = original(*args, **kwargs)
        warm = report.candidate_warmup_cycles[0]
        rows = []
        for row in warm.rows:
            assert row.prediction is not None
            prediction = replace(
                row.prediction,
                scores=tuple(value + 0.25 for value in row.prediction.scores),
            )
            rows.append(replace(row, prediction=prediction))
        return replace(
            report, candidate_warmup_cycles=(replace(warm, rows=tuple(rows)),)
        )

    monkeypatch.setattr(runner, "benchmark_pair", shifted)
    output = tmp_path / "out.json"
    assert not runner.study(tmp_path / "model", output, comparator="full", batch_size=1)
    result = read_object(output)
    assert cast(dict[str, object], result["report"])["passed"] is True
    checks = cast(dict[str, list[dict[str, object]]], result["reference_checks"])
    assert checks["candidate_warmup"][0]["passed"] is False
    assert all(item["passed"] is True for item in checks["candidate_measured"])
    assert result["quality_passed"] is False and result["selection_passed"] is False


@pytest.mark.parametrize(
    "left,right,retained,full",
    [
        ([101.0] * 5, [100.0] * 5, True, False),
        ([102.0] * 5, [100.0] * 5, True, True),
        ([100.0] * 5, [100.0] * 5, False, False),
        (
            [100.0, 100.0, 100.0, 100.0, 200.0],
            [50.0, 50.0, 50.0, 50.0, 250.0],
            False,
            False,
        ),
    ],
)
def test_comparator_threshold_boundary_and_tail_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
    left: list[float],
    right: list[float],
    retained: bool,
    full: bool,
) -> None:
    from collections.abc import Callable

    original = cast(Callable[..., BenchmarkReport], vars(runner)["benchmark_pair"])
    observed: list[BenchmarkReport] = []

    def capture(*args: object, **kwargs: object) -> BenchmarkReport:
        report = original(*args, **kwargs)
        observed.append(report)
        return report

    monkeypatch.setattr(runner, "benchmark_pair", capture)
    assert runner.study(
        tmp_path / "model", tmp_path / "out.json", comparator="retained", batch_size=8
    )
    report = replace(
        observed[0],
        baseline_cycles=tuple(
            replace(cycle, latency_ms=t)
            for cycle, t in zip(observed[0].baseline_cycles, left, strict=True)
        ),
        candidate_cycles=tuple(
            replace(cycle, latency_ms=t)
            for cycle, t in zip(observed[0].candidate_cycles, right, strict=True)
        ),
    )
    assert runner.performance(report, "retained")["passed"] is retained
    assert runner.performance(report, "full")["passed"] is full
    if right[-1] == 250:
        result = runner.performance(report, "retained")
        assert (
            cast(float, result["paired_p05"]) > 1
            and cast(float, result["aggregate_throughput_ratio"]) > 1
        )
        assert cast(float, result["candidate_cohort_p95_ratio"]) > 1


@pytest.mark.parametrize(
    "defect", ["warmup", "pairs", "rows", "plan", "engine", "input"]
)
def test_unsupported_return_keeps_raw_cycles_and_never_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
    defect: str,
) -> None:
    from collections.abc import Callable

    original = cast(Callable[..., BenchmarkReport], vars(runner)["benchmark_pair"])

    def corrupt(*args: object, **kwargs: object) -> BenchmarkReport:
        report = original(*args, **kwargs)
        if defect == "warmup":
            return replace(report, candidate_warmup_cycles=())
        if defect == "pairs":
            return replace(
                report,
                baseline_cycles=report.baseline_cycles[:3],
                candidate_cycles=report.candidate_cycles[:3],
            )
        if defect == "rows":
            cycle = report.candidate_warmup_cycles[0]
            return replace(
                report, candidate_warmup_cycles=(replace(cycle, rows=cycle.rows[:76]),)
            )
        if defect == "plan":
            return replace(report, protocol=BenchmarkProtocol(repeats=3, warmup=1))
        if defect == "engine":
            return replace(report, baseline=replace(report.baseline, batch_size=1))
        return replace(report, input_sha256="0" * 64)

    monkeypatch.setattr(runner, "benchmark_pair", corrupt)
    output = tmp_path / "out.json"
    with pytest.raises(ValueError, match="inconsistent contract"):
        runner.study(tmp_path / "model", output, comparator="retained", batch_size=8)
    result = read_object(output)
    assert result["benchmark_returned"] is True and "report" in result
    assert (
        result["completed"]
        is result["quality_passed"]
        is result["selection_passed"]
        is False
    )


@pytest.mark.parametrize("comparator", ["retained", "full"])
def test_actual_alternating_call_order_and_comparator_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_study: None,
    comparator: Comparator,
) -> None:
    import wyat.benchmark as benchmark

    original = benchmark._run
    observed: list[tuple[str, str, int]] = []

    def run(
        engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]
    ) -> BenchmarkCycle:
        owner = vars(engine.model)["_scorer"]
        kind = vars(owner).get("arm", "full")
        observed.append((engine.name, cast(str, kind), engine.batch_size))
        return original(engine, cases)

    monkeypatch.setattr(benchmark, "_run", run)
    assert runner.study(
        tmp_path / "model", tmp_path / "out.json", comparator=comparator, batch_size=8
    )
    expected = [
        comparator,
        "one-use",
        comparator,
        "one-use",
        "one-use",
        comparator,
        comparator,
        "one-use",
        "one-use",
        comparator,
        comparator,
        "one-use",
    ]
    assert observed == [(name, name, 8) for name in expected]


@pytest.fixture
def report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_study: None
) -> BenchmarkReport:
    from collections.abc import Callable

    original = cast(Callable[..., BenchmarkReport], vars(runner)["benchmark_pair"])
    captured: list[BenchmarkReport] = []

    def capture(*args: object, **kwargs: object) -> BenchmarkReport:
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(runner, "benchmark_pair", capture)
    assert runner.study(
        tmp_path / "model", tmp_path / "out.json", comparator="retained", batch_size=8
    )
    return captured[0]


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    left=st.lists(
        st.floats(
            min_value=100,
            max_value=1000,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=5,
        max_size=5,
    ),
    factor=st.sampled_from((0.95, 1.0, 1.01, 1.05)),
)
def test_uniform_scaling_has_correct_gate_direction_and_finite_ratios(
    report: BenchmarkReport,
    left: list[float],
    factor: float,
) -> None:
    scaled = replace(
        report,
        baseline_cycles=tuple(
            replace(cycle, latency_ms=t)
            for cycle, t in zip(report.baseline_cycles, left, strict=True)
        ),
        candidate_cycles=tuple(
            replace(cycle, latency_ms=t / factor)
            for cycle, t in zip(report.candidate_cycles, left, strict=True)
        ),
    )
    retained = runner.performance(scaled, "retained")
    full = runner.performance(scaled, "full")
    assert cast(list[float], retained["paired_ratios"]) == pytest.approx(
        [factor] * 5, rel=1e-12
    )
    assert retained["passed"] is (factor > 1)
    assert full["passed"] is (factor > 1.02)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_nonfinite_or_nonpositive_times_never_yield_performance_pass(
    report: BenchmarkReport,
    invalid: float,
) -> None:
    cycle = replace(report.candidate_cycles[0], latency_ms=invalid)
    changed = replace(report, candidate_cycles=(cycle, *report.candidate_cycles[1:]))
    with pytest.raises(ValueError, match="finite positive"):
        runner.performance(changed, "retained")
