"""Failure-inclusive evaluation of a declared choice scorer on supplied cases."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .core import Decision, Model, Request, request_from_dict


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    request: Request
    expected: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("case ID must be nonempty text")
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
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


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

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "input_sha256": self.input_sha256,
            "model_id": self.model_id,
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "rows": [row.to_dict() for row in self.rows],
        }


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


def evaluate(model: Model, cases: Iterable[EvaluationCase]) -> EvaluationReport:
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
            prediction = model.decide(case.request)
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
