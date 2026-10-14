"""Failure-inclusive evaluation of a declared choice scorer on supplied cases."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

from .core import (
    Decision,
    Model,
    Request,
    _text,
    _validate_temperature,
    request_from_dict,
)


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    request: Request
    expected: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.id, "case ID", nonempty=True)
        if not isinstance(self.request, Request):
            raise ValueError("case request must be a Request")
        if (
            not isinstance(self.expected, tuple)
            or not self.expected
            or any(not isinstance(label, str) or not label for label in self.expected)
            or len(set(self.expected)) != len(self.expected)
        ):
            raise ValueError("expected IDs must be a nonempty tuple of unique text IDs")
        candidates = {candidate.id for candidate in self.request.candidates}
        if not set(self.expected) <= candidates:
            raise ValueError("expected IDs must belong to the submitted candidates")


def case_from_dict(value: object) -> EvaluationCase:
    """Parse labels separately from the request sent to the model."""
    if not isinstance(value, Mapping) or set(value) != {"id", "request", "expected"}:
        raise ValueError("evaluation case requires exactly id, request and expected")
    identifier, expected = value["id"], value["expected"]
    if not isinstance(identifier, str) or not isinstance(expected, list):
        raise ValueError("case ID must be text and expected must be a list of IDs")
    if any(not isinstance(label, str) for label in expected):
        raise ValueError("expected IDs must be text")
    return EvaluationCase(
        identifier, request_from_dict(value["request"]), tuple(expected)
    )


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    id: str
    expected: tuple[str, ...]
    status: str
    prediction: Decision | None
    correct: bool
    latency_ms: float
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "expected": list(self.expected),
            "status": self.status,
            "prediction": None
            if self.prediction is None
            else self.prediction.to_dict(),
            "correct": self.correct,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile of finite measured values."""
    if not values or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("nonempty values and a percentile fraction in [0, 1] required")
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies must be finite and nonnegative")
    ordered = sorted(values)
    return _ordered_percentile(ordered, fraction)


def _ordered_percentile(ordered: Sequence[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _confidence(prediction: Decision) -> float:
    prediction._validate_ranking()
    prediction._validate_probabilities()
    return prediction.probabilities[prediction.scores.index(max(prediction.scores))]


def _expected_mask(prediction: Decision, expected: tuple[str, ...]) -> list[bool]:
    # Rank by scores, not probabilities: directly constructed Decisions need
    # not have monotonic probabilities. Keep sums in original candidate order.
    order = sorted(
        range(len(prediction.scores)), key=prediction.scores.__getitem__, reverse=True
    )
    expected_set = set(expected)
    mask = [False] * len(order)
    for index, candidate_id in zip(order, prediction.ranking, strict=True):
        mask[index] = candidate_id in expected_set
    return mask


def _brier_value(probabilities: Sequence[float], expected: Sequence[bool]) -> float:
    return math.fsum(
        (p - (1.0 if hit else 0.0)) ** 2
        for p, hit in zip(probabilities, expected, strict=True)
    )


def _log_loss_value(probabilities: Sequence[float], expected: Sequence[bool]) -> float:
    mass = math.fsum(p for p, hit in zip(probabilities, expected, strict=True) if hit)
    return -math.log(max(min(mass, 1.0), 1e-15))


def _calibration_error(bins: Mapping[int, list[tuple[float, float]]], n: int) -> float:
    ece = 0.0
    for items in bins.values():
        bin_size = len(items)
        bin_conf = math.fsum(c for c, _ in items) / bin_size
        bin_acc = math.fsum(a for _, a in items) / bin_size
        ece += (bin_size / n) * abs(bin_conf - bin_acc)
    return ece


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    model_id: str
    rows: tuple[EvaluationRow, ...]
    input_sha256: str

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def successful(self) -> int:
        return sum(row.status == "ok" for row in self.rows)

    @property
    def correct(self) -> int:
        return sum(row.correct for row in self.rows)

    @property
    def failed(self) -> int:
        return self.total - self.successful

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def coverage(self) -> float:
        return self.successful / self.total

    @property
    def latency_p50_ms(self) -> float:
        return percentile([row.latency_ms for row in self.rows], 0.50)

    @property
    def latency_p95_ms(self) -> float:
        return percentile([row.latency_ms for row in self.rows], 0.95)

    @property
    def probability_rows(self) -> tuple[EvaluationRow, ...]:
        return tuple(
            row
            for row in self.rows
            if row.status == "ok" and row.prediction is not None
        )

    @property
    def mean_confidence(self) -> float | None:
        valid = self.probability_rows
        if not valid:
            return None
        return math.fsum(
            _confidence(row.prediction) for row in valid if row.prediction is not None
        ) / len(valid)

    @property
    def brier_score(self) -> float | None:
        valid = self.probability_rows
        if not valid:
            return None
        brier_values: list[float] = []
        for row in valid:
            if row.prediction is None:
                continue
            row.prediction._validate_ranking()
            row.prediction._validate_probabilities()
            brier_values.append(
                _brier_value(
                    row.prediction.probabilities,
                    _expected_mask(row.prediction, row.expected),
                )
            )
        return math.fsum(brier_values) / len(brier_values) if brier_values else None

    @property
    def log_loss(self) -> float | None:
        valid = self.probability_rows
        if not valid:
            return None
        losses: list[float] = []
        for row in valid:
            if row.prediction is None:
                continue
            row.prediction._validate_ranking()
            row.prediction._validate_probabilities()
            losses.append(
                _log_loss_value(
                    row.prediction.probabilities,
                    _expected_mask(row.prediction, row.expected),
                )
            )
        return math.fsum(losses) / len(losses) if losses else None

    @property
    def expected_calibration_error(self) -> float | None:
        valid = self.probability_rows
        if not valid:
            return None
        bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in valid:
            if row.prediction is None:
                continue
            conf = _confidence(row.prediction)
            hit = 1.0 if row.correct else 0.0
            bin_idx = min(int(conf * 10), 9)
            bins[bin_idx].append((conf, hit))
        return _calibration_error(bins, len(valid))

    def rescale(self, temperature: float) -> EvaluationReport:
        """Return a new report with probabilities rescaled by temperature."""
        temp = _validate_temperature(temperature)
        new_rows: list[EvaluationRow] = []
        for row in self.rows:
            if row.prediction is None or row.status != "ok":
                new_rows.append(row)
                continue
            pred = row.prediction
            peak = max(pred.scores)
            weights = tuple(math.exp((value - peak) / temp) for value in pred.scores)
            total = math.fsum(weights)
            probabilities = tuple(weight / total for weight in weights)
            new_pred = Decision(
                pred.model_id,
                pred.scores,
                probabilities,
                pred.ranking,
            )
            new_rows.append(
                EvaluationRow(
                    row.id,
                    row.expected,
                    row.status,
                    new_pred,
                    row.correct,
                    row.latency_ms,
                    row.error,
                )
            )
        return EvaluationReport(self.model_id, tuple(new_rows), self.input_sha256)

    def fit_temperature(
        self,
        *,
        metric: Literal["log_loss", "brier_score"] = "log_loss",
        bounds: tuple[float, float] = (0.01, 10.0),
        max_iterations: int = 50,
    ) -> float:
        """Find the optimal temperature minimizing log-loss or Brier score."""
        return fit_temperature(
            self,
            metric=metric,
            bounds=bounds,
            max_iterations=max_iterations,
        )

    def calibrate(
        self,
        *,
        metric: Literal["log_loss", "brier_score"] = "log_loss",
        bounds: tuple[float, float] = (0.01, 10.0),
        max_iterations: int = 50,
    ) -> tuple[float, EvaluationReport]:
        """Fit temperature and return (optimal_temperature, rescaled_report)."""
        optimal_temp = self.fit_temperature(
            metric=metric, bounds=bounds, max_iterations=max_iterations
        )
        return optimal_temp, self.rescale(optimal_temp)

    def to_dict(self) -> dict[str, object]:
        successful = correct = 0
        latencies: list[float] = []
        rows: list[dict[str, object]] = []
        confidences: list[float] = []
        brier_values: list[float] = []
        losses: list[float] = []
        bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for row in self.rows:
            successful += row.status == "ok"
            correct += row.correct
            latencies.append(row.latency_ms)
            if row.status == "ok" and row.prediction is not None:
                prediction = row.prediction
                conf = _confidence(prediction)
                confidences.append(conf)
                bins[min(int(conf * 10), 9)].append((conf, 1.0 if row.correct else 0.0))
                expected = _expected_mask(prediction, row.expected)
                brier_values.append(_brier_value(prediction.probabilities, expected))
                losses.append(_log_loss_value(prediction.probabilities, expected))
            rows.append(row.to_dict())
        total = self.total
        # Preserve the empty-report division error and the latency validation
        # performed by the public percentile helper.
        accuracy = correct / total
        if any(not math.isfinite(value) or value < 0 for value in latencies):
            raise ValueError("latencies must be finite and nonnegative")
        latencies.sort()
        n = len(confidences)
        return {
            "schema_version": 1,
            "input_sha256": self.input_sha256,
            "model_id": self.model_id,
            "total": total,
            "successful": successful,
            "failed": total - successful,
            "correct": correct,
            "accuracy": accuracy,
            "coverage": successful / total,
            "mean_confidence": math.fsum(confidences) / n if n else None,
            "expected_calibration_error": _calibration_error(bins, n) if n else None,
            "brier_score": math.fsum(brier_values) / n if n else None,
            "log_loss": math.fsum(losses) / n if n else None,
            "latency_p50_ms": _ordered_percentile(latencies, 0.50),
            "latency_p95_ms": _ordered_percentile(latencies, 0.95),
            "rows": rows,
        }


def fit_temperature(
    evaluation: EvaluationReport | Sequence[EvaluationRow],
    *,
    metric: Literal["log_loss", "brier_score"] = "log_loss",
    bounds: tuple[float, float] = (0.01, 10.0),
    max_iterations: int = 50,
) -> float:
    """Find the optimal temperature minimizing log-loss or Brier score."""
    if isinstance(evaluation, EvaluationReport):
        rows = evaluation.probability_rows
    elif isinstance(evaluation, Sequence) and not isinstance(
        evaluation, (str, bytes, bytearray)
    ):
        rows = tuple(
            row
            for row in evaluation
            if row.status == "ok" and row.prediction is not None
        )
    else:
        raise ValueError(
            "evaluation must be an EvaluationReport or sequence of EvaluationRows"
        )

    if not rows:
        raise ValueError(
            "temperature fitting requires at least one successful prediction"
        )
    if metric not in ("log_loss", "brier_score"):
        raise ValueError("metric must be either 'log_loss' or 'brier_score'")
    if (
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(
            isinstance(b, bool)
            or not isinstance(b, Real)
            or not math.isfinite(b)
            or b <= 0
            for b in bounds
        )
        or bounds[0] >= bounds[1]
    ):
        raise ValueError(
            "bounds must be a tuple of two positive finite numbers (lower, upper)"
        )
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    data: list[tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...]]] = []
    for row in rows:
        pred = cast(Decision, row.prediction)
        pred._validate_ranking()
        scores = pred.scores
        is_expected = tuple(_expected_mask(pred, row.expected))
        peak = max(scores)
        shifted = tuple(s - peak for s in scores)
        targets = tuple(i for i, hit in enumerate(is_expected) if hit)
        data.append((shifted, is_expected, targets))

    if metric == "log_loss":

        def loss_fn(temp: float) -> float:
            total_loss = 0.0
            for shifted, _, targets in data:
                # Retain division: multiplication by a reciprocal can round
                # differently and change the golden-section search path.
                weights = [1.0 if s == 0.0 else math.exp(s / temp) for s in shifted]
                total_weight = math.fsum(weights)
                if len(targets) == 1:
                    target_weight = weights[targets[0]]
                else:
                    target_weight = math.fsum([weights[i] for i in targets])
                target_mass = target_weight / total_weight
                clamped = max(min(target_mass, 1.0), 1e-15)
                total_loss -= math.log(clamped)
            return total_loss / len(data)

    else:

        def loss_fn(temp: float) -> float:
            total_brier = 0.0
            for shifted, is_expected, _ in data:
                weights = [1.0 if s == 0.0 else math.exp(s / temp) for s in shifted]
                total_weight = math.fsum(weights)
                row_brier = math.fsum(
                    [
                        (weights[i] / total_weight - (1.0 if is_expected[i] else 0.0))
                        ** 2
                        for i in range(len(weights))
                    ]
                )
                total_brier += row_brier
            return total_brier / len(data)

    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    u_a = math.log(float(bounds[0]))
    u_b = math.log(float(bounds[1]))

    u_1 = u_b - ratio * (u_b - u_a)
    u_2 = u_a + ratio * (u_b - u_a)

    f_1 = loss_fn(math.exp(u_1))
    f_2 = loss_fn(math.exp(u_2))

    for _ in range(max_iterations):
        if f_1 < f_2:
            u_b = u_2
            u_2 = u_1
            f_2 = f_1
            u_1 = u_b - ratio * (u_b - u_a)
            f_1 = loss_fn(math.exp(u_1))
        else:
            u_a = u_1
            u_1 = u_2
            f_1 = f_2
            u_2 = u_a + ratio * (u_b - u_a)
            f_2 = loss_fn(math.exp(u_2))

    optimal_log_t = (u_a + u_b) / 2.0
    return math.exp(optimal_log_t)


def _input_digest(cases: tuple[EvaluationCase, ...]) -> str:
    cohort = [
        {
            "id": case.id,
            "request": {
                "query": case.request.query,
                "context": case.request.context,
                "task": case.request.task,
                "candidates": [
                    {"id": candidate.id, "text": candidate.text}
                    for candidate in case.request.candidates
                ],
            },
            "expected": list(case.expected),
        }
        for case in cases
    ]
    canonical = json.dumps(
        cohort,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluate(
    model: Model,
    cases: Iterable[EvaluationCase],
    *,
    temperature: float | None = None,
) -> EvaluationReport:
    """Validate the complete cohort, then retain every attempted prediction.

    Latency covers every model call, including failures. The sequence may mix
    cold and warm calls; its percentiles do not establish steady-state throughput.
    Labels affect only the cohort hash and scoring, never a model request.
    """
    cohort = tuple(cases)
    if not cohort or any(not isinstance(case, EvaluationCase) for case in cohort):
        raise ValueError(
            "evaluation requires a nonempty cohort of EvaluationCase values"
        )
    if len({case.id for case in cohort}) != len(cohort):
        raise ValueError("evaluation case IDs must be unique")
    input_sha256 = _input_digest(cohort)
    model_id = model.model_id
    rows: list[EvaluationRow] = []
    for case in cohort:
        start = time.perf_counter_ns()
        try:
            prediction = model.decide(case.request, temperature=temperature)
        except Exception as error:
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            rows.append(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "error",
                    None,
                    False,
                    latency_ms,
                    f"{type(error).__name__}: {error}",
                )
            )
        else:
            latency_ms = (time.perf_counter_ns() - start) / 1_000_000
            rows.append(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "ok",
                    prediction,
                    prediction.selected in case.expected,
                    latency_ms,
                    None,
                )
            )
    return EvaluationReport(model_id, tuple(rows), input_sha256)
