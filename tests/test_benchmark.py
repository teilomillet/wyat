from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from itertools import count
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from wyat.benchmark import (
    BenchmarkCycle,
    BenchmarkEngine,
    BenchmarkProtocol,
    QualityGate,
    benchmark_pair,
    compare_cycles,
)
from wyat.core import Candidate, Model, Request, decision_from_logits
from wyat.evaluation import EvaluationCase, EvaluationRow, case_from_dict


class FixedScorer:
    def __init__(
        self,
        name: str = "fixed",
        calls: list[str] | None = None,
        scores: tuple[float, float] = (1.0, 2.0),
    ) -> None:
        self.name = name
        self.calls = calls if calls is not None else []
        self.scores = scores

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(self.name)
        if any(request.query == "failure" for request in requests):
            raise RuntimeError("deliberate failure")
        return tuple(self.scores for _ in requests)


def case(identifier: str = "one", query: str = "question") -> EvaluationCase:
    return EvaluationCase(
        identifier,
        Request(query, (Candidate("a", "first"), Candidate("b", "second"))),
        ("b",),
    )


def engine(name: str, scorer: FixedScorer, *, batch_size: int = 1) -> BenchmarkEngine:
    return BenchmarkEngine(name, Model(scorer, name), batch_size)


def cycle(scores: tuple[float, ...]) -> BenchmarkCycle:
    sample = case()
    prediction = decision_from_logits(sample.request, scores, "fixture")
    row = EvaluationRow(
        sample.id,
        sample.expected,
        "ok",
        prediction,
        prediction.selected in sample.expected,
        1.0,
        None,
    )
    return BenchmarkCycle(1.0, (row,), 100, 101)


def test_alternating_pairs_warmups_and_barriers_are_observable() -> None:
    calls: list[str] = []
    barriers: list[str] = []
    left = replace(
        engine("baseline", FixedScorer("left", calls)),
        synchronize=lambda: barriers.append("left"),
    )
    right = replace(
        engine("candidate", FixedScorer("right", calls)),
        synchronize=lambda: barriers.append("right"),
    )
    report = benchmark_pair(
        left, right, (case(),), protocol=BenchmarkProtocol(repeats=3)
    )
    assert calls == ["left", "right", "left", "right", "right", "left", "left", "right"]
    assert barriers == [arm for arm in calls for _ in range(2)]
    assert len(report.baseline_cycles) == len(report.candidate_cycles) == 3
    assert report.drift.passed
    encoded = json.dumps(report.to_dict(), allow_nan=False)
    assert "cumulative process high-water" in encoded
    assert len(report.input_sha256) == 64


def test_failures_keep_every_case_and_invalidate_faster_candidate() -> None:
    samples = (case(), case("bad", "failure"), case("three"))
    report = benchmark_pair(
        engine("serial", FixedScorer()),
        engine("batch", FixedScorer(), batch_size=3),
        samples,
        protocol=BenchmarkProtocol(repeats=3),
    )
    result = report.to_dict()
    assert report.drift.total == 9
    assert report.drift.comparable == 0
    assert not report.drift.passed
    speed = result["speed"]
    assert isinstance(speed, dict)
    assert isinstance(speed["median_speedup_positive"], bool)
    assert not result["passed"]
    assert all(len(measurement.rows) == 3 for measurement in report.candidate_cycles)
    assert all(measurement.failed == 3 for measurement in report.candidate_cycles)
    assert all(measurement.correct == 2 for measurement in report.baseline_cycles)
    candidate = result["candidate"]
    assert isinstance(candidate, dict)
    assert candidate["successful_requests_per_second"] == 0


def test_cycle_units_and_throughput_follow_measured_complete_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = count(step=1_000_000)
    monkeypatch.setattr("wyat.benchmark.time.perf_counter_ns", lambda: next(ticks))
    report = benchmark_pair(
        engine("serial", FixedScorer()),
        engine("batch", FixedScorer(), batch_size=2),
        (case(), case("two")),
        protocol=BenchmarkProtocol(repeats=3),
    )
    assert [measured.latency_ms for measured in report.baseline_cycles] == [5.0] * 3
    assert [measured.latency_ms for measured in report.candidate_cycles] == [3.0] * 3
    result = report.to_dict()
    candidate = result["candidate"]
    speed = result["speed"]
    assert isinstance(candidate, dict) and isinstance(speed, dict)
    assert candidate["attempted_requests_per_second"] == pytest.approx(2 / 0.003)
    assert candidate["batch_residence_p50_ms"] == 1.0
    assert speed["paired_speedup_p50"] == pytest.approx(5 / 3)
    assert speed["median_speedup_positive"]
    assert result["passed"]


def test_batch_size_does_not_change_workload_identity() -> None:
    samples = (case(), case("two"), case("three"))
    report = benchmark_pair(
        engine("serial", FixedScorer()),
        engine("batch", FixedScorer(), batch_size=2),
        samples,
        protocol=BenchmarkProtocol(repeats=3),
    )
    assert report.drift.passed
    assert report.drift.total == report.drift.baseline_correct == 9
    for measured in report.candidate_cycles:
        assert [row.id for row in measured.rows] == [sample.id for sample in samples]
        assert measured.rows[0].latency_ms == measured.rows[1].latency_ms


def test_warmup_failure_remains_visible_and_blocks_acceptance() -> None:
    class ColdFailure(FixedScorer):
        def score_many(
            self, requests: Sequence[Request]
        ) -> tuple[tuple[float, ...], ...]:
            if not self.calls:
                self.calls.append("cold")
                raise RuntimeError("cold inference failed")
            return super().score_many(requests)

    report = benchmark_pair(
        engine("serial", FixedScorer()),
        engine("cold", ColdFailure()),
        (case(),),
        protocol=BenchmarkProtocol(repeats=3),
    )
    assert report.drift.passed
    assert report.warmup_failed == 1
    assert not report.passed
    assert report.candidate_warmup_cycles[0].rows[0].error is not None


def test_empty_cycle_is_not_a_vacuous_quality_pass() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        compare_cycles(
            (BenchmarkCycle(1.0, (), 0, 0),),
            (BenchmarkCycle(1.0, (), 0, 0),),
            QualityGate(),
        )


@pytest.mark.parametrize("warmup,repeats", [(0, 3), (1, 2), (True, 3), (1, True)])
def test_invalid_protocol_fails_early(warmup: int, repeats: int) -> None:
    with pytest.raises(ValueError):
        BenchmarkProtocol(warmup=warmup, repeats=repeats)


@pytest.mark.parametrize("samples", [(), (case(), case())])
def test_empty_and_duplicate_cohorts_never_call_model(
    samples: tuple[EvaluationCase, ...],
) -> None:
    scorer = FixedScorer()
    arm = engine("arm", scorer)
    with pytest.raises(ValueError):
        benchmark_pair(arm, arm, samples)
    assert not scorer.calls


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.001, True])
def test_gate_rejects_invalid_tolerances(value: float) -> None:
    with pytest.raises(ValueError):
        QualityGate(logit_atol=value)


def test_common_logit_shift_preserves_ranking_but_fails_numerical_gate() -> None:
    drift = compare_cycles((cycle((1.0, 2.0)),), (cycle((2.0, 3.0)),), QualityGate())
    assert drift.selected_equal == drift.ranking_equal == 1
    assert drift.max_abs_probability_difference == pytest.approx(0)
    assert not drift.logits_within_tolerance
    assert not drift.passed


def test_small_logit_perturbation_cannot_hide_selected_and_quality_regression() -> None:
    drift = compare_cycles(
        (cycle((0.0, 0.00001)),), (cycle((0.00001, 0.0)),), QualityGate()
    )
    assert drift.logits_within_tolerance
    assert drift.selected_equal == 0
    assert drift.candidate_correct < drift.baseline_correct
    assert not drift.passed


@given(st.tuples(st.floats(-50, 50), st.floats(-50, 50)))
def test_self_comparison_always_passes_for_finite_logits(
    scores: tuple[float, float],
) -> None:
    measured = cycle(scores)
    drift = compare_cycles((measured,), (measured,), QualityGate())
    assert drift.passed
    assert drift.max_abs_logit_difference == 0
    assert drift.max_abs_probability_difference == 0


@given(st.floats(min_value=0.01, max_value=10))
def test_large_common_offset_fails_even_with_equal_selections(offset: float) -> None:
    drift = compare_cycles(
        (cycle((0.0, 1.0)),), (cycle((offset, 1.0 + offset)),), QualityGate()
    )
    assert not drift.passed
    assert drift.selected_equal == 1


def test_misaligned_case_ids_are_rejected() -> None:
    measured = cycle((1.0, 2.0))
    changed = replace(measured, rows=(replace(measured.rows[0], id="other"),))
    with pytest.raises(ValueError, match="IDs"):
        compare_cycles((measured,), (changed,), QualityGate())


def test_pinned_real_development_cohort_has_one_independent_case_per_intent() -> None:
    path = Path(__file__).parents[1] / "benchmarks/real/banking77-train-dev-menu6.jsonl"
    cases = tuple(
        case_from_dict(json.loads(line)) for line in path.read_text().splitlines()
    )
    assert len(cases) == len({sample.id for sample in cases}) == 77
    assert len({sample.expected for sample in cases}) == 77
    assert len({sample.request.query for sample in cases}) == 77
    assert all(len(sample.request.candidates) == 6 for sample in cases)
    assert all(sample.id.startswith("banking77-train-") for sample in cases)
