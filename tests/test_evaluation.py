# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal, cast

import pytest

from wyat.core import Candidate, Decision, Model, Request
from wyat.evaluation import (
    EvaluationCase,
    EvaluationReport,
    EvaluationRow,
    case_from_dict,
    evaluate,
    fit_temperature,
    percentile,
)


def _reference_metrics(rows: tuple[EvaluationRow, ...]) -> dict[str, float | None]:
    """Pre-optimization dictionary-based metrics, including summation order."""
    confidences: list[float] = []
    briers: list[float] = []
    losses: list[float] = []
    bins: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        if row.status != "ok" or row.prediction is None:
            continue
        probs = row.prediction.probabilities_by_id
        conf = probs[row.prediction.selected]
        confidences.append(conf)
        expected = set(row.expected)
        briers.append(
            math.fsum(
                (p - (1.0 if cid in expected else 0.0)) ** 2 for cid, p in probs.items()
            )
        )
        mass = math.fsum(p for cid, p in probs.items() if cid in expected)
        losses.append(-math.log(max(min(mass, 1.0), 1e-15)))
        bins.setdefault(min(int(conf * 10), 9), []).append(
            (conf, 1.0 if row.correct else 0.0)
        )
    n = len(confidences)
    ece = 0.0
    for items in bins.values():
        size = len(items)
        confidence = math.fsum(c for c, _ in items) / size
        accuracy = math.fsum(a for _, a in items) / size
        ece += (size / n) * abs(confidence - accuracy)
    return {
        "mean_confidence": math.fsum(confidences) / n if n else None,
        "brier_score": math.fsum(briers) / n if n else None,
        "log_loss": math.fsum(losses) / n if n else None,
        "expected_calibration_error": ece if n else None,
    }


@pytest.mark.parametrize("seed", range(12))
def test_report_metrics_match_dictionary_reference_exactly(seed: int) -> None:
    import random

    rng = random.Random(seed)
    rows: list[EvaluationRow] = []
    for index in range(80):
        width = rng.choice([1, 2, 3, 14, 77])
        ids = tuple(f"é-日本-{i}" for i in range(width))
        scores = tuple(rng.choice([-1000.0, -1.0, 0.0, 1.0, 1000.0]) for _ in ids)
        order = sorted(range(width), key=scores.__getitem__, reverse=True)
        # Include nonmonotonic probabilities, zeros, and ties: legacy direct
        # Decisions are not restricted to softmax distributions.
        probabilities = tuple(rng.choice([0.0, 0.1, 0.2, 0.5, 1.0]) for _ in ids)
        prediction = Decision(
            "fixture", scores, probabilities, tuple(ids[i] for i in order)
        )
        expected = tuple(rng.sample(ids, rng.randint(1, width)))
        status = "error" if index % 9 == 0 else "ok"
        rows.append(
            EvaluationRow(
                str(index),
                expected,
                status,
                None if index % 13 == 0 else prediction,
                status == "ok" and prediction.selected in expected,
                rng.random() * 100,
                "failure" if status == "error" else None,
            )
        )
    report = EvaluationReport("fixture", tuple(rows), "digest")
    serialized = report.to_dict()
    for name, expected_metric in _reference_metrics(report.rows).items():
        assert getattr(report, name) == expected_metric
        assert serialized[name] == expected_metric
        if expected_metric is not None:
            assert getattr(report, name).hex() == expected_metric.hex()
            assert cast(float, serialized[name]).hex() == expected_metric.hex()
    for name in ("total", "successful", "failed", "correct", "accuracy", "coverage"):
        assert serialized[name] == getattr(report, name)
    latencies = [row.latency_ms for row in rows]
    assert serialized["latency_p50_ms"] == percentile(latencies, 0.50)
    assert serialized["latency_p95_ms"] == percentile(latencies, 0.95)
    assert serialized["rows"] == [row.to_dict() for row in rows]


@pytest.mark.parametrize(
    "prediction",
    [
        Decision("fixture", (), (), ()),
        Decision("fixture", (0.0, float("nan")), (0.5, 0.5), ("a", "b")),
        Decision("fixture", (0.0, 1.0), (0.5,), ("a", "b")),
        Decision("fixture", (0.0, 1.0), (0.5, float("inf")), ("a", "b")),
        Decision("fixture", (0.0, 1.0), (0.5, 0.5), ("a", "a")),
        Decision("fixture", (0.0, 1.0), (0.5, 0.5), ("a", "\ud800")),
        Decision("fixture", (0.0, 1.0), (0.5, 0.5), ("a",)),
    ],
)
def test_optimized_metrics_still_reject_malformed_decisions(
    prediction: Decision,
) -> None:
    report = EvaluationReport(
        "fixture",
        (EvaluationRow("row", ("a",), "ok", prediction, False, 1.0, None),),
        "digest",
    )
    with pytest.raises(ValueError):
        _ = prediction.probabilities_by_id
    for metric in (
        "mean_confidence",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    ):
        with pytest.raises(ValueError):
            getattr(report, metric)
    with pytest.raises(ValueError):
        report.to_dict()


@pytest.mark.parametrize("latency", [-1.0, float("nan"), float("inf")])
def test_serialization_rejects_invalid_latencies(latency: float) -> None:
    report = EvaluationReport(
        "fixture",
        (EvaluationRow("row", ("a",), "error", None, False, latency, "failure"),),
        "digest",
    )
    with pytest.raises(ValueError, match="latencies must be finite and nonnegative"):
        report.to_dict()


def test_empty_report_serialization_preserves_division_error() -> None:
    with pytest.raises(ZeroDivisionError):
        EvaluationReport("fixture", (), "digest").to_dict()


def _reference_temperature(
    rows: tuple[EvaluationRow, ...],
    metric: str,
    bounds: tuple[float, float],
    iterations: int,
) -> float:
    """Original fitting objective and golden-section search as a parity oracle."""
    data = []
    for row in rows:
        if row.status == "ok" and row.prediction is not None:
            pred = row.prediction
            expected = set(row.expected)
            data.append(
                (
                    pred.scores,
                    tuple(cid in expected for cid in pred.candidate_ids),
                    max(pred.scores),
                )
            )

    def loss(temp: float) -> float:
        total = 0.0
        for scores, targets, peak in data:
            weights = [math.exp((s - peak) / temp) for s in scores]
            denominator = math.fsum(weights)
            if metric == "log_loss":
                mass = (
                    math.fsum(w for w, hit in zip(weights, targets, strict=True) if hit)
                    / denominator
                )
                total -= math.log(max(min(mass, 1.0), 1e-15))
            else:
                probs = [w / denominator for w in weights]
                total += math.fsum(
                    (p - (1.0 if hit else 0.0)) ** 2
                    for p, hit in zip(probs, targets, strict=True)
                )
        return total / len(data)

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    lower, upper = math.log(bounds[0]), math.log(bounds[1])
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_loss, right_loss = loss(math.exp(left)), loss(math.exp(right))
    for _ in range(iterations):
        if left_loss < right_loss:
            upper, right, right_loss = right, left, left_loss
            left = upper - ratio * (upper - lower)
            left_loss = loss(math.exp(left))
        else:
            lower, left, left_loss = left, right, right_loss
            right = lower + ratio * (upper - lower)
            right_loss = loss(math.exp(right))
    return math.exp((lower + upper) / 2.0)


@pytest.mark.parametrize("metric", ["log_loss", "brier_score"])
@pytest.mark.parametrize("iterations", [1, 7, 50])
@pytest.mark.parametrize("seed", range(8))
def test_temperature_fit_matches_original_search_exactly(
    metric: Literal["log_loss", "brier_score"], iterations: int, seed: int
) -> None:
    import random

    rng = random.Random(seed)
    rows = []
    for index in range(12):
        width = rng.choice([1, 2, 14])
        ids = tuple(f"é-{i}" for i in range(width))
        scores = tuple(
            rng.choice([-1e308, -1000.0, 0.0, 0.0, 1.0, 1000.0, 1e308])
            if seed % 2
            else rng.uniform(-10, 10)
            for _ in ids
        )
        order = sorted(range(width), key=scores.__getitem__, reverse=True)
        prediction = Decision(
            "fixture",
            scores,
            tuple(1.0 / width for _ in ids),
            tuple(ids[i] for i in order),
        )
        expected = tuple(rng.sample(ids, rng.randint(0, width)))
        rows.append(
            EvaluationRow(
                str(index),
                expected,
                "error" if index % 7 == 0 else "ok",
                None if index % 11 == 0 else prediction,
                prediction.selected in expected,
                1.0,
                None,
            )
        )
    report = EvaluationReport("fixture", tuple(rows), "digest")
    bounds = (0.01, 10.0) if seed % 3 else (0.4, 0.400001)
    expected_temperature = _reference_temperature(
        report.rows, metric, bounds, iterations
    )
    actual = fit_temperature(
        report, metric=metric, bounds=bounds, max_iterations=iterations
    )
    assert actual.hex() == expected_temperature.hex()
    calibrated_temperature, calibrated = report.calibrate(
        metric=metric, bounds=bounds, max_iterations=iterations
    )
    assert calibrated_temperature.hex() == expected_temperature.hex()
    assert calibrated.to_dict() == report.rescale(expected_temperature).to_dict()


@pytest.mark.parametrize("metric", ["log_loss", "brier_score"])
@pytest.mark.parametrize("bounds", [(5e-324, 5e-323), (1e307, 1e308)])
def test_temperature_fit_preserves_extreme_finite_bounds(
    metric: Literal["log_loss", "brier_score"], bounds: tuple[float, float]
) -> None:
    prediction = Decision("fixture", (5e-324, 0.0), (0.5, 0.5), ("a", "b"))
    rows = (EvaluationRow("row", ("b",), "ok", prediction, False, 1.0, None),)
    assert fit_temperature(rows, metric=metric, bounds=bounds).hex() == (
        _reference_temperature(rows, metric, bounds, 50).hex()
    )


def test_fitting_preserves_score_only_validation() -> None:
    prediction = Decision("fixture", (0.0, 1.0), (float("nan"),), ("b", "a"))
    rows = (EvaluationRow("row", ("a",), "ok", prediction, True, 1.0, None),)
    assert (
        fit_temperature(rows).hex()
        == _reference_temperature(rows, "log_loss", (0.01, 10.0), 50).hex()
    )


@pytest.mark.parametrize(
    "prediction",
    [
        Decision("fixture", (), (), ()),
        Decision("fixture", (0.0, float("nan")), (0.5, 0.5), ("a", "b")),
        Decision("fixture", (0.0, 1.0), (0.5, 0.5), ("a", "a")),
        Decision("fixture", (0.0, 1.0), (0.5, 0.5), ("a", "\ud800")),
    ],
)
def test_temperature_fitting_rejects_malformed_scores_and_ranking(
    prediction: Decision,
) -> None:
    rows = (EvaluationRow("row", ("a",), "ok", prediction, False, 1.0, None),)
    with pytest.raises(ValueError):
        fit_temperature(rows)


def request(query: str = "question") -> Request:
    return Request(query, (Candidate("a", "First"), Candidate("b", "Second")))


class Scorer:
    def __init__(self) -> None:
        self.calls: list[Request] = []

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        self.calls.extend(requests)
        if any(value.query == "failure" for value in requests):
            raise RuntimeError("fixture failure")
        return [(1.0, 2.0) for _ in requests]


def test_multiple_expected_ids_and_failed_predictions_stay_in_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0, 10_000_000, 20_000_000, 40_000_000, 50_000_000, 80_000_000])
    monkeypatch.setattr("wyat.evaluation.time.perf_counter_ns", lambda: next(clock))
    scorer = Scorer()
    cases = [
        EvaluationCase("correct", request(), ("a", "b")),
        EvaluationCase("wrong", request(), ("a",)),
        EvaluationCase("failed", request("failure"), ("b",)),
    ]
    report = evaluate(Model(scorer, model_id="fixture"), cases).to_dict()
    assert report["total"] == 3
    assert report["correct"] == 1
    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["coverage"] == pytest.approx(2 / 3)
    assert report["latency_p50_ms"] == 20
    assert report["latency_p95_ms"] == pytest.approx(29)
    rows = cast(list[dict[str, object]], report["rows"])
    assert len(rows) == 3
    failure = rows[2]
    assert failure["status"] == "error"
    assert failure["prediction"] is None
    assert cast(float, failure["latency_ms"]) >= 0


def test_empty_evaluation_is_rejected_without_inference() -> None:
    scorer = Scorer()
    with pytest.raises(ValueError):
        evaluate(Model(scorer, model_id="fixture"), [])
    assert scorer.calls == []


def test_duplicate_case_ids_fail_before_any_inference() -> None:
    scorer = Scorer()
    cases = [
        EvaluationCase("same", request(), ("a",)),
        EvaluationCase("same", request(), ("b",)),
    ]
    with pytest.raises(ValueError):
        evaluate(Model(scorer, model_id="fixture"), cases)
    assert scorer.calls == []


@pytest.mark.parametrize("identifier", ["\ud800", "case-\udfff"])
def test_invalid_unicode_case_id_is_rejected_at_construction(identifier: str) -> None:
    with pytest.raises(ValueError, match="case ID must contain valid Unicode"):
        EvaluationCase(identifier, request(), ("b",))


def test_unicode_case_and_label_ids_remain_exact_with_failures_in_denominator() -> None:
    candidates = (Candidate("é", "First"), Candidate("e\u0301", "Second"))
    first_id = "é-日本-العربية-🧪"
    second_id = "e\u0301-日本-العربية-🧪"
    cases = [
        EvaluationCase(
            first_id, Request("Choisir / 選択 / اختر", candidates), ("e\u0301",)
        ),
        EvaluationCase(second_id, Request("failure", candidates), ("é",)),
    ]
    report = evaluate(Model(Scorer(), model_id="fixture"), cases).to_dict()
    rows = cast(list[dict[str, object]], report["rows"])
    assert [row["id"] for row in rows] == [first_id, second_id]
    assert rows[0]["expected"] == ["e\u0301"]
    prediction = cast(dict[str, object], rows[0]["prediction"])
    assert prediction["selected"] == "e\u0301"
    assert prediction["ranking"] == ["e\u0301", "é"]
    assert rows[1]["status"] == "error"
    assert report["total"] == 2
    assert report["accuracy"] == report["coverage"] == 0.5


@pytest.mark.parametrize("expected", [(), ("unknown",), ("a", "a")])
def test_invalid_expected_labels_fail_before_any_inference(
    expected: tuple[str, ...],
) -> None:
    scorer = Scorer()
    with pytest.raises((ValueError, TypeError)):
        evaluate(
            Model(scorer, model_id="fixture"),
            [EvaluationCase("case", request(), expected)],
        )
    assert scorer.calls == []


def test_case_parser_preserves_gold_labels_outside_public_request() -> None:
    case = case_from_dict(
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [
                    {"id": "a", "text": "First"},
                    {"id": "b", "text": "Second"},
                ],
            },
            "expected": ["b"],
        }
    )
    assert case == EvaluationCase("case", request(), ("b",))


@pytest.mark.parametrize(
    "value",
    [
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [{"id": "a", "text": "First"}],
            },
            "expected": ["a"],
            "extra": True,
        },
        {
            "id": "case",
            "request": {
                "query": "question",
                "candidates": [{"id": "a", "text": "First"}],
            },
            "expected": "a",
        },
    ],
)
def test_case_parser_rejects_unknown_fields_and_string_labels(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        case_from_dict(value)


def test_evaluation_digest_is_stable_across_outputs_and_timings() -> None:
    class OtherScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(2.0, 1.0) for _ in requests]

    case = EvaluationCase("case", request(), ("b",))
    first = evaluate(Model(Scorer(), model_id="fixture"), [case]).to_dict()
    second = evaluate(Model(OtherScorer(), model_id="fixture"), [case]).to_dict()
    assert first["correct"] != second["correct"]
    assert first["schema_version"] == 1
    assert len(cast(str, first["input_sha256"])) == 64
    assert first["input_sha256"] == second["input_sha256"]


@pytest.mark.parametrize(
    "change",
    ["id", "gold", "query", "context", "task", "candidate_text", "candidate_order"],
)
def test_evaluation_digest_binds_each_public_input_and_gold_field(change: str) -> None:
    original = EvaluationCase("case", request(), ("b",))
    if change == "id":
        changed = replace(original, id="different")
    elif change == "gold":
        changed = replace(original, expected=("a",))
    elif change == "query":
        changed = replace(
            original, request=replace(original.request, query="different")
        )
    elif change == "context":
        changed = replace(
            original, request=replace(original.request, context="different")
        )
    elif change == "task":
        changed = replace(original, request=replace(original.request, task="different"))
    elif change == "candidate_text":
        changed = replace(
            original,
            request=replace(
                original.request,
                candidates=(Candidate("a", "Changed"), Candidate("b", "Second")),
            ),
        )
    else:
        assert change == "candidate_order"
        changed = replace(
            original,
            request=replace(
                original.request,
                candidates=tuple(reversed(original.request.candidates)),
            ),
        )
    model = Model(Scorer(), model_id="fixture")
    assert (
        evaluate(model, [original]).input_sha256
        != evaluate(model, [changed]).input_sha256
    )


def test_gold_labels_never_enter_scorer_request() -> None:
    scorer = Scorer()
    public = request()
    evaluate(
        Model(scorer, model_id="fixture"), [EvaluationCase("case", public, ("b",))]
    )
    assert scorer.calls == [public]
    assert not hasattr(scorer.calls[0], "expected")


def test_report_binds_model_identity_separately_from_input_digest() -> None:
    case = EvaluationCase("case", request(), ("b",))
    first = evaluate(Model(Scorer(), model_id="model:first"), [case]).to_dict()
    second = evaluate(Model(Scorer(), model_id="model:second"), [case]).to_dict()
    assert first["model_id"] == "model:first"
    assert second["model_id"] == "model:second"
    assert first["input_sha256"] == second["input_sha256"]


def test_calibration_metrics_calculation() -> None:
    class FixedScorer:
        def __init__(self, logits_seq: Sequence[tuple[float, float]]) -> None:
            self._logits = list(logits_seq)
            self._idx = 0

        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            res = [self._logits[self._idx + i] for i in range(len(requests))]
            self._idx += len(requests)
            return res

    # math.exp(2.0) / (math.exp(2.0) + math.exp(0.0)) ≈ 0.8808
    # math.exp(0.0) / (math.exp(2.0) + math.exp(0.0)) ≈ 0.1192
    p_high = 7.38905609893065 / 8.38905609893065
    p_low = 1.0 - p_high

    # Case 1: logits (2.0, 0.0) -> selects 'a' with p_high ≈ 0.8808 (correct).
    # Case 2: logits (0.0, 2.0) -> selects 'b' with p_high ≈ 0.8808 (wrong).
    scorer = FixedScorer([(2.0, 0.0), (0.0, 2.0)])
    cases = [
        EvaluationCase("c1", request(), ("a",)),
        EvaluationCase("c2", request(), ("a",)),
    ]
    report = evaluate(Model(scorer, model_id="fixed"), cases)
    report_dict = report.to_dict()

    assert report.total == 2
    assert report.correct == 1
    assert report.accuracy == 0.5
    assert report.mean_confidence == pytest.approx(p_high)

    # Brier score:
    # row 1 (target 'a'): (p_high - 1)^2 + (p_low - 0)^2 = 2 * (p_low^2)
    # row 2 (target 'a'): (p_low - 1)^2 + (p_high - 0)^2 = 2 * (p_high^2)
    expected_brier = (2 * (p_low**2) + 2 * (p_high**2)) / 2
    assert report.brier_score == pytest.approx(expected_brier)

    # Log loss: row 1: -ln(p_high), row 2: -ln(p_low)
    expected_log_loss = (-math.log(p_high) - math.log(p_low)) / 2
    assert report.log_loss == pytest.approx(expected_log_loss)

    # ECE: both predictions fall into bin 8 (p_high ≈ 0.88, 0.88 * 10 = 8.8 -> bin 8)
    # bin_conf = p_high, bin_acc = 0.5 (1 out of 2 is correct), ECE = |p_high - 0.5|
    assert report.expected_calibration_error == pytest.approx(abs(p_high - 0.5))

    assert report_dict["mean_confidence"] == report.mean_confidence
    assert report_dict["brier_score"] == report.brier_score
    assert report_dict["log_loss"] == report.log_loss
    assert report_dict["expected_calibration_error"] == (
        report.expected_calibration_error
    )


def test_calibration_metrics_none_when_all_fail() -> None:
    scorer = Scorer()
    cases = [EvaluationCase("c1", request("failure"), ("a",))]
    report = evaluate(Model(scorer, model_id="failing"), cases)
    assert report.total == 1
    assert report.failed == 1
    assert report.successful == 0
    assert report.mean_confidence is None
    assert report.brier_score is None
    assert report.log_loss is None
    assert report.expected_calibration_error is None
    assert report.to_dict()["mean_confidence"] is None


def test_evaluate_with_explicit_temperature() -> None:
    class FixedScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(2.0, 0.0) for _ in requests]

    cases = [EvaluationCase("c1", request(), ("a",))]
    model = Model(FixedScorer(), model_id="fixture")
    report = evaluate(model, cases, temperature=2.0)
    p0 = report.rows[0].prediction.probabilities[0]  # type: ignore[union-attr]
    expected_p0 = math.exp(1.0) / (math.exp(1.0) + math.exp(0.0))
    assert p0 == pytest.approx(expected_p0)


def test_evaluation_report_rescale() -> None:
    class FixedScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(2.0, 0.0) for _ in requests]

    cases = [EvaluationCase("c1", request(), ("a",))]
    report = evaluate(Model(FixedScorer(), model_id="fixture"), cases)
    rescaled = report.rescale(0.5)

    assert rescaled.total == report.total
    assert rescaled.correct == report.correct
    assert rescaled.accuracy == report.accuracy
    pred_0 = rescaled.rows[0].prediction
    assert pred_0 is not None
    assert pred_0.scores == report.rows[0].prediction.scores  # type: ignore[union-attr]

    # At T=0.5, exp(4.0) / (exp(4.0) + 1)
    p0_half = math.exp(4.0) / (math.exp(4.0) + math.exp(0.0))
    assert pred_0.probabilities[0] == pytest.approx(p0_half)

    with pytest.raises(
        ValueError, match="temperature must be a positive finite number"
    ):
        report.rescale(-1.0)


def test_selective_counts_failures_and_applies_margin_inclusively() -> None:
    def row(
        identifier: str, probabilities: tuple[float, ...], correct: bool
    ) -> EvaluationRow:
        ids = tuple("abc"[: len(probabilities)])
        prediction = Decision("fixture", probabilities, probabilities, ids)
        return EvaluationRow(identifier, ("a",), "ok", prediction, correct, 1.0, None)

    failed = EvaluationRow("failed", ("a",), "error", None, False, 1.0, "Boom")
    report = EvaluationReport(
        "fixture",
        (
            row("sure-right", (0.75, 0.25), True),  # margin exactly 0.5
            row("sure-wrong", (0.875, 0.125), False),  # margin 0.75
            row("unsure", (0.5, 0.5), True),  # margin 0
            row("single", (1.0,), True),  # no alternative
            failed,
        ),
        "0" * 64,
    )

    everything = report.selective(0.0)
    assert (everything.automated, everything.correct, everything.total) == (4, 3, 5)
    assert everything.automation_rate == 0.8
    assert everything.accuracy == 0.75

    at_half = report.selective(0.5)
    assert (at_half.min_margin, at_half.automated, at_half.correct) == (0.5, 3, 2)
    above_half = report.selective(0.51)
    assert (above_half.automated, above_half.correct) == (2, 1)
    only_single = report.selective(1.0)
    assert (only_single.automated, only_single.accuracy) == (1, 1.0)

    all_failed = EvaluationReport("fixture", (failed,), "0" * 64).selective(0.0)
    assert all_failed.accuracy is None and all_failed.automation_rate == 0.0

    for invalid in (-0.1, 1.1, math.nan, math.inf, True, "0.5"):
        with pytest.raises(ValueError, match="min_margin"):
            report.selective(invalid)  # type: ignore[arg-type]


def test_selective_uses_the_report_probabilities_after_rescale() -> None:
    class FixedScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(1.0, 0.0) for _ in requests]

    cases = [EvaluationCase("c1", request(), ("a",))]
    report = evaluate(Model(FixedScorer(), model_id="fixture"), cases)
    # T=1: margin = tanh(0.5) ≈ 0.46. T=0.25: margin = tanh(2) ≈ 0.96.
    assert report.selective(0.9).automated == 0
    assert report.rescale(0.25).selective(0.9).automated == 1


def test_fit_temperature_calibrates_overconfident_and_underconfident_models() -> None:
    # 1. Overconfident model: logits (5.0, 0.0) -> p_a ≈ 0.993
    # In reality, 'a' is only correct in 60 out of 100 cases.
    class OverconfidentScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(5.0, 0.0) for _ in requests]

    over_cases = [
        EvaluationCase(f"case_{i}", request(), ("a",) if i < 60 else ("b",))
        for i in range(100)
    ]
    report_over = evaluate(Model(OverconfidentScorer(), model_id="over"), over_cases)
    assert report_over.mean_confidence is not None
    assert report_over.mean_confidence > 0.99
    assert report_over.accuracy == 0.60
    assert report_over.expected_calibration_error is not None
    assert report_over.expected_calibration_error > 0.35

    opt_t, calibrated_over = report_over.calibrate()
    assert opt_t > 1.0  # Temperature must increase to soften overconfidence
    assert calibrated_over.log_loss is not None and report_over.log_loss is not None
    assert calibrated_over.log_loss < report_over.log_loss
    assert (
        calibrated_over.expected_calibration_error is not None
        and report_over.expected_calibration_error is not None
    )
    assert (
        calibrated_over.expected_calibration_error
        < report_over.expected_calibration_error
    )

    # 2. Underconfident model: logits (0.05, 0.0) -> p_a ≈ 0.512
    # In reality, 'a' is correct in 95 out of 100 cases.
    class UnderconfidentScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(0.05, 0.0) for _ in requests]

    under_cases = [
        EvaluationCase(f"case_{i}", request(), ("a",) if i < 95 else ("b",))
        for i in range(100)
    ]
    report_under = evaluate(
        Model(UnderconfidentScorer(), model_id="under"), under_cases
    )
    assert report_under.mean_confidence is not None
    assert report_under.mean_confidence < 0.55
    assert report_under.accuracy == 0.95

    opt_t_under, calibrated_under = report_under.calibrate()
    assert opt_t_under < 1.0  # Temperature must decrease to sharpen probabilities
    assert calibrated_under.log_loss is not None and report_under.log_loss is not None
    assert calibrated_under.log_loss < report_under.log_loss

    # Calibration with Brier score
    opt_t_brier = fit_temperature(report_over.rows, metric="brier_score")
    assert opt_t_brier > 1.0
    rescaled_brier = report_over.rescale(opt_t_brier)
    assert rescaled_brier.brier_score is not None
    assert report_over.brier_score is not None
    assert rescaled_brier.brier_score < report_over.brier_score


def test_fit_temperature_validation_and_failures() -> None:
    cases = [EvaluationCase("c1", request("failure"), ("a",))]
    report_failed = evaluate(Model(Scorer(), model_id="failing"), cases)

    with pytest.raises(ValueError, match="at least one successful prediction"):
        fit_temperature(report_failed)

    with pytest.raises(ValueError, match="at least one successful prediction"):
        fit_temperature([])

    with pytest.raises(ValueError, match="evaluation must be an EvaluationReport"):
        fit_temperature(cast(EvaluationReport, "not-a-report"))

    valid_report = evaluate(
        Model(Scorer(), model_id="fixture"), [EvaluationCase("c1", request(), ("a",))]
    )

    with pytest.raises(ValueError, match="metric must be either"):
        fit_temperature(
            valid_report, metric=cast(Literal["log_loss"], "invalid_metric")
        )

    with pytest.raises(ValueError, match="bounds must be a tuple"):
        fit_temperature(valid_report, bounds=(10.0, 1.0))

    with pytest.raises(ValueError, match="bounds must be a tuple"):
        fit_temperature(valid_report, bounds=(-1.0, 2.0))

    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        fit_temperature(valid_report, max_iterations=0)
