"""Paired, synchronized engine measurements with failure-inclusive quality gates."""

from __future__ import annotations

import math
import platform
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version

from .core import Model
from .evaluation import EvaluationCase, EvaluationRow, _input_digest, percentile


@dataclass(frozen=True, slots=True)
class QualityGate:
    """Declare tolerances before measuring; default acceptance preserves every rank."""

    logit_atol: float = 1e-3
    logit_rtol: float = 1e-3
    probability_atol: float = 1e-4

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0
            for value in (self.logit_atol, self.logit_rtol, self.probability_atol)
        ):
            raise ValueError("quality tolerances must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    repeats: int = 5
    warmup: int = 1
    quality_gate: QualityGate = field(default_factory=QualityGate)

    def __post_init__(self) -> None:
        if type(self.repeats) is not int or self.repeats < 3:
            raise ValueError("benchmark requires at least three repetitions")
        if type(self.warmup) is not int or self.warmup < 1:
            raise ValueError("benchmark requires at least one warmup cycle")


@dataclass(frozen=True, slots=True)
class BenchmarkEngine:
    name: str
    model: Model
    batch_size: int = 1
    conditions: Mapping[str, str] = field(default_factory=dict)
    synchronize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("engine name must be nonempty")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("engine batch_size must be a positive integer")


def process_peak_rss_bytes() -> int:
    """OS process high-water RSS, cumulative across both loaded engines and arms."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for package in (
        "wyat",
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "mlx",
        "mlx-lm",
    ):
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            continue
    return installed


@dataclass(frozen=True, slots=True)
class BenchmarkCycle:
    latency_ms: float
    rows: tuple[EvaluationRow, ...]
    peak_rss_before_bytes: int
    peak_rss_after_bytes: int

    @property
    def correct(self) -> int:
        return sum(row.correct for row in self.rows)

    @property
    def failed(self) -> int:
        return sum(row.status != "ok" for row in self.rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "latency_ms": self.latency_ms,
            "peak_rss_before_bytes": self.peak_rss_before_bytes,
            "peak_rss_after_bytes": self.peak_rss_after_bytes,
            "correct": self.correct,
            "failed": self.failed,
            "rows": [row.to_dict() for row in self.rows],
        }


def _run(engine: BenchmarkEngine, cases: tuple[EvaluationCase, ...]) -> BenchmarkCycle:
    # Native Wyat scorers return host floats. Other scorers can provide a barrier.
    if engine.synchronize is not None:
        engine.synchronize()
    peak_before = process_peak_rss_bytes()
    start = time.perf_counter_ns()
    rows: list[EvaluationRow] = []
    for offset in range(0, len(cases), engine.batch_size):
        batch = cases[offset : offset + engine.batch_size]
        batch_start = time.perf_counter_ns()
        try:
            predictions = engine.model.decide_many(
                tuple(case.request for case in batch)
            )
            if engine.synchronize is not None:
                engine.synchronize()
        except Exception as error:
            latency = (time.perf_counter_ns() - batch_start) / 1_000_000
            rows.extend(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "error",
                    None,
                    False,
                    latency,
                    f"{type(error).__name__}: {error}",
                )
                for case in batch
            )
        else:
            latency = (time.perf_counter_ns() - batch_start) / 1_000_000
            rows.extend(
                EvaluationRow(
                    case.id,
                    case.expected,
                    "ok",
                    prediction,
                    prediction.selected in case.expected,
                    latency,
                    None,
                )
                for case, prediction in zip(batch, predictions, strict=True)
            )
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    return BenchmarkCycle(elapsed, tuple(rows), peak_before, process_peak_rss_bytes())


@dataclass(frozen=True, slots=True)
class DriftReport:
    total: int
    comparable: int
    selected_equal: int
    ranking_equal: int
    baseline_correct: int
    candidate_correct: int
    max_abs_logit_difference: float | None
    max_abs_probability_difference: float | None
    logits_within_tolerance: bool
    probabilities_within_tolerance: bool
    passed: bool


def compare_cycles(
    baseline: Sequence[BenchmarkCycle],
    candidate: Sequence[BenchmarkCycle],
    gate: QualityGate,
) -> DriftReport:
    """Compare aligned repetitions; every missing or failed result fails the gate."""
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired nonempty cycles with identical counts required")
    total = comparable = selected = ranked = base_correct = cand_correct = 0
    logit_differences: list[float] = []
    probability_differences: list[float] = []
    logits_close = probabilities_close = True
    for left_cycle, right_cycle in zip(baseline, candidate, strict=True):
        if not left_cycle.rows or len(left_cycle.rows) != len(right_cycle.rows):
            raise ValueError("paired cycles require identical nonempty case counts")
        for left, right in zip(left_cycle.rows, right_cycle.rows, strict=True):
            if left.id != right.id or left.expected != right.expected:
                raise ValueError("paired cycle IDs and labels must match")
            total += 1
            base_correct += left.correct
            cand_correct += right.correct
            if left.prediction is None or right.prediction is None:
                continue
            a, b = left.prediction, right.prediction
            if len(a.scores) != len(b.scores):
                raise ValueError("paired candidate counts must match")
            comparable += 1
            selected += a.selected == b.selected
            ranked += a.ranking == b.ranking
            for x, y in zip(a.scores, b.scores, strict=True):
                logit_differences.append(abs(x - y))
                logits_close &= math.isclose(
                    x, y, abs_tol=gate.logit_atol, rel_tol=gate.logit_rtol
                )
            for x, y in zip(a.probabilities, b.probabilities, strict=True):
                difference = abs(x - y)
                probability_differences.append(difference)
                probabilities_close &= difference <= gate.probability_atol
    complete = comparable == total
    return DriftReport(
        total,
        comparable,
        selected,
        ranked,
        base_correct,
        cand_correct,
        max(logit_differences, default=None),
        max(probability_differences, default=None),
        complete and logits_close,
        complete and probabilities_close,
        complete
        and selected == total
        and ranked == total
        and logits_close
        and probabilities_close
        and cand_correct >= base_correct,
    )


def _arm_report(
    engine: BenchmarkEngine, cycles: tuple[BenchmarkCycle, ...]
) -> dict[str, object]:
    timings = [cycle.latency_ms for cycle in cycles]
    attempted = sum(len(cycle.rows) for cycle in cycles)
    failed = sum(cycle.failed for cycle in cycles)
    correct = sum(cycle.correct for cycle in cycles)
    batch_latencies = [row.latency_ms for cycle in cycles for row in cycle.rows]
    elapsed_seconds = math.fsum(timings) / 1000
    return {
        "name": engine.name,
        "model_id": engine.model.model_id,
        "conditions": dict(engine.conditions),
        "batch_size": engine.batch_size,
        "attempted": attempted,
        "failed": failed,
        "correct": correct,
        "accuracy": correct / attempted,
        "coverage": (attempted - failed) / attempted,
        "cycle_p50_ms": percentile(timings, 0.50),
        "cycle_p95_ms": percentile(timings, 0.95),
        "batch_residence_p50_ms": percentile(batch_latencies, 0.50),
        "batch_residence_p95_ms": percentile(batch_latencies, 0.95),
        "attempted_requests_per_second": attempted / elapsed_seconds,
        "successful_requests_per_second": (attempted - failed) / elapsed_seconds,
        "cycles": [cycle.to_dict() for cycle in cycles],
    }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    input_sha256: str
    cases: tuple[EvaluationCase, ...]
    protocol: BenchmarkProtocol
    baseline: BenchmarkEngine
    candidate: BenchmarkEngine
    baseline_cycles: tuple[BenchmarkCycle, ...]
    candidate_cycles: tuple[BenchmarkCycle, ...]
    drift: DriftReport
    baseline_warmup_cycles: tuple[BenchmarkCycle, ...]
    candidate_warmup_cycles: tuple[BenchmarkCycle, ...]

    @property
    def warmup_failed(self) -> int:
        return sum(
            cycle.failed
            for cycle in self.baseline_warmup_cycles + self.candidate_warmup_cycles
        )

    @property
    def passed(self) -> bool:
        return self.drift.passed and self.warmup_failed == 0

    def to_dict(self) -> dict[str, object]:
        baseline_times = [cycle.latency_ms for cycle in self.baseline_cycles]
        candidate_times = [cycle.latency_ms for cycle in self.candidate_cycles]
        speedups = [
            left / right
            for left, right in zip(baseline_times, candidate_times, strict=True)
        ]
        return {
            "schema_version": 1,
            "input_sha256": self.input_sha256,
            "case_ids": [case.id for case in self.cases],
            "case_count": len(self.cases),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "versions": _versions(),
            },
            "protocol": {
                **asdict(self.protocol),
                "paired_order": "alternating baseline/candidate, candidate/baseline",
                "timing": "complete cohort cycles, including tokenization, batching, "
                "model, host results, decision interpretation and row bookkeeping",
                "batch_residence": "full synchronous batch call latency assigned to "
                "each member; excludes admission queue and transport",
                "memory": "OS cumulative process high-water RSS at cycle boundaries; "
                "shared loaded engines, cannot attribute per-arm resident memory",
                "excludes": ["loading", "warmup", "HTTP", "JSON serialization"],
                "native_synchronization": "native scorers materialize host floats; "
                "custom engines may supply an explicit barrier",
            },
            "baseline": _arm_report(self.baseline, self.baseline_cycles),
            "candidate": _arm_report(self.candidate, self.candidate_cycles),
            "quality": asdict(self.drift),
            "warmup": {
                "failed": self.warmup_failed,
                "baseline_cycles": [
                    cycle.to_dict() for cycle in self.baseline_warmup_cycles
                ],
                "candidate_cycles": [
                    cycle.to_dict() for cycle in self.candidate_warmup_cycles
                ],
            },
            "passed": self.passed,
            "speed": {
                "paired_cycle_speedups": speedups,
                "paired_speedup_p50": percentile(speedups, 0.50),
                "paired_speedup_p05": percentile(speedups, 0.05),
                "paired_speedup_p95": percentile(speedups, 0.95),
                "median_speedup_positive": percentile(speedups, 0.50) > 1,
            },
        }


def benchmark_pair(
    baseline: BenchmarkEngine,
    candidate: BenchmarkEngine,
    cases: Sequence[EvaluationCase],
    *,
    protocol: BenchmarkProtocol | None = None,
) -> BenchmarkReport:
    """Freeze inputs and gates, warm both engines, alternate paired measurements."""
    plan = protocol or BenchmarkProtocol()
    cohort = tuple(cases)
    if not cohort or any(not isinstance(case, EvaluationCase) for case in cohort):
        raise ValueError("benchmark requires a nonempty EvaluationCase cohort")
    if len({case.id for case in cohort}) != len(cohort):
        raise ValueError("benchmark case IDs must be unique")
    digest = _input_digest(cohort)
    left_warmup: list[BenchmarkCycle] = []
    right_warmup: list[BenchmarkCycle] = []
    for _ in range(plan.warmup):
        left_warmup.append(_run(baseline, cohort))
        right_warmup.append(_run(candidate, cohort))
    left: list[BenchmarkCycle] = []
    right: list[BenchmarkCycle] = []
    for repeat in range(plan.repeats):
        operations = [(baseline, left), (candidate, right)]
        if repeat % 2:
            operations.reverse()
        for engine, cycles in operations:
            cycles.append(_run(engine, cohort))
    drift = compare_cycles(left, right, plan.quality_gate)
    return BenchmarkReport(
        digest,
        cohort,
        plan,
        baseline,
        candidate,
        tuple(left),
        tuple(right),
        drift,
        tuple(left_warmup),
        tuple(right_warmup),
    )
