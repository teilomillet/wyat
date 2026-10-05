"""Capture an actual maintained gate input, then detach it through host storage."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from wyat.benchmark import BenchmarkCycle, BenchmarkEngine, _run
from wyat.core import Model
from wyat.evaluation import EvaluationCase

if TYPE_CHECKING:
    import mlx.core as mx

    from wyat.backends.kai.mlx import MLXKaiScorer
    from wyat.backends.kai.prefix import PrefixScorer


@dataclass(frozen=True, slots=True)
class CapturedInput:
    values: np.ndarray[tuple[int, int], np.dtype[np.float32]]
    cycle: BenchmarkCycle
    copy_elapsed_ms: float

    def metadata(self) -> dict[str, object]:
        return {
            "copy_materialization_elapsed_ms": self.copy_elapsed_ms,
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "host_strides_bytes": list(self.values.strides),
            "sha256": hashlib.sha256(self.values.tobytes()).hexdigest(),
            "nbytes": self.values.nbytes,
        }


class CaptureFailure(ValueError):
    def __init__(self, message: str, cycle: BenchmarkCycle) -> None:
        super().__init__(message)
        self.cycle = cycle


class CaptureProjection:
    """Call the unchanged projection and observe its exact argument once."""

    def __init__(self, original: Callable[[mx.array], mx.array]) -> None:
        self.original = original
        self.inputs: list[mx.array] = []

    def __call__(self, value: mx.array) -> mx.array:
        self.inputs.append(value)
        return self.original(value)


class ProjectionOwner(Protocol):
    gate_proj: Callable[[mx.array], mx.array]


@contextmanager
def observe_projection(owner: ProjectionOwner) -> Iterator[CaptureProjection]:
    original = owner.gate_proj
    observer = CaptureProjection(original)
    owner.gate_proj = observer
    try:
        yield observer
    finally:
        owner.gate_proj = original


def detached_float32(
    value: mx.array,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """A host-owned copy cannot retain a lazy model graph or checkpoint owner."""
    import mlx.core as mx

    if value.dtype != mx.float32 or len(value.shape) not in (2, 3):
        raise ValueError("Capture requires a rank2/3 FP32 activation")
    mx.eval(value)
    host = np.array(value, dtype=np.float32, copy=True).reshape(-1, value.shape[-1])
    if not np.isfinite(host).all():
        raise ValueError("Actual capture contains nonfinite activations")
    return np.ascontiguousarray(host)


def capture_input(
    base: MLXKaiScorer,
    scorer: MLXKaiScorer | PrefixScorer,
    cases: tuple[EvaluationCase, ...],
    *,
    batch_size: int,
    expected_rows: int,
) -> CapturedInput:
    """One declared batch only; prefix construction must precede this hook."""
    if not cases or len(cases) != batch_size:
        raise ValueError("Capture requires one complete declared batch")
    with observe_projection(
        cast(ProjectionOwner, base.backbone.layers[0].mlp)
    ) as observer:
        cycle = _run(
            BenchmarkEngine(
                "actual-activation-capture", Model(scorer, base.model_id), batch_size
            ),
            cases,
        )
        if cycle.failed or len(observer.inputs) != 1:
            raise CaptureFailure(
                "Capture did not visit exactly one successful gate call", cycle
            )
        started = time.perf_counter_ns()
        values = detached_float32(observer.inputs[0])
        copy_elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if values.shape != (expected_rows, 1024):
            raise CaptureFailure(
                "Actual projection input differs from declared geometry", cycle
            )
    return CapturedInput(values, cycle, copy_elapsed_ms)


def exact_bfloat_bits(
    values: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> bool:
    """BF16 decode preserves FP32 bits, including the sign of zero."""
    return bool(
        np.isfinite(values).all() and np.all((values.view(np.uint32) & 0xFFFF) == 0)
    )


def projection_quality(
    reference: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    candidate: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> dict[str, object]:
    if (
        reference.shape != candidate.shape
        or reference.dtype != candidate.dtype
        or reference.dtype != np.float32
    ):
        raise ValueError("Projection quality requires matching FP32 shapes")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    difference = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    failures = (~np.isfinite(difference)) | (
        difference > 1e-4 + 1e-4 * np.abs(reference.astype(np.float64))
    )
    return {
        "passed": finite and not bool(failures.any()),
        "finite": finite,
        "failing_elements": int(failures.sum()),
        "total_elements": reference.size,
        "max_abs_error": float(difference.max()) if finite else None,
        "atol": 1e-4,
        "rtol": 1e-4,
    }
