"""Small explanatory performance models; observations remain their authority."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _count(value: int, name: str, *, zero: bool = False) -> None:
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(
            f"{name} must be a {'nonnegative' if zero else 'positive'} integer"
        )


def _fraction(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("optimized fraction must be finite and in [0, 1]")


def amdahl_speedup(optimized_fraction: float, local_speedup: float) -> float:
    """Fixed-workload prediction when one measured fraction changes in isolation."""
    _fraction(optimized_fraction)
    _positive(local_speedup, "local_speedup")
    return 1 / ((1 - optimized_fraction) + optimized_fraction / local_speedup)


def amdahl_ceiling(optimized_fraction: float) -> float | None:
    """Ideal removal ceiling; None denotes an unbounded all-work fraction."""
    _fraction(optimized_fraction)
    return None if optimized_fraction == 1 else 1 / (1 - optimized_fraction)


@dataclass(frozen=True, slots=True)
class RooflineBound:
    compute_seconds: float
    memory_seconds: float
    lower_bound_seconds: float
    intensity_flops_per_byte: float | None


def roofline_lower_bound(
    *,
    flops: int,
    bytes_moved: int,
    compute_flops_per_second: float,
    memory_bytes_per_second: float,
) -> RooflineBound:
    """Optimistic max(work/compute, traffic/bandwidth), under supplied assumptions.

    Counts must be justified lower bounds on actual work and traffic, and rates
    must be justified upper bounds for this device and precision. Measurements
    below this bound invalidate an assumption; they do not violate physics.
    """
    _count(flops, "flops", zero=True)
    _count(bytes_moved, "bytes_moved", zero=True)
    _positive(compute_flops_per_second, "compute_flops_per_second")
    _positive(memory_bytes_per_second, "memory_bytes_per_second")
    compute = flops / compute_flops_per_second
    memory = bytes_moved / memory_bytes_per_second
    return RooflineBound(
        compute,
        memory,
        max(compute, memory),
        flops / bytes_moved if bytes_moved else None,
    )


@dataclass(frozen=True, slots=True)
class TransformerShape:
    hidden: int
    intermediate: int
    layers: int
    query_heads: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden", self.hidden),
            ("intermediate", self.intermediate),
            ("layers", self.layers),
            ("query_heads", self.query_heads),
            ("kv_heads", self.kv_heads),
            ("head_dim", self.head_dim),
        ):
            _count(value, name)
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")


@dataclass(frozen=True, slots=True)
class BatchWork:
    examples: int
    useful_tokens: int
    padded_tokens: int
    padded_length: int
    projection_flops: int
    dense_attention_flops: int
    ideal_causal_attention_flops: int

    @property
    def padding_fraction(self) -> float:
        return 1 - self.useful_tokens / self.padded_tokens

    @property
    def projection_plus_dense_attention_flops(self) -> int:
        return self.projection_flops + self.dense_attention_flops


def estimate_batch_work(
    lengths: Sequence[int], shape: TransformerShape, *, pad_multiple: int = 8
) -> BatchWork:
    """Backbone matmul arithmetic for GQA, explicitly including right padding.

    Multiply-add counts as two FLOPs. Attention reports both dense and ideal
    triangular arithmetic; neither identifies a fused kernel's actual schedule.
    Norms, RoPE, softmax, activations and decision head are excluded.
    """
    if not lengths:
        raise ValueError("at least one sequence length required")
    for length in lengths:
        _count(length, "sequence length")
    _count(pad_multiple, "pad_multiple")
    width = ((max(lengths) + pad_multiple - 1) // pad_multiple) * pad_multiple
    examples = len(lengths)
    tokens = examples * width
    query = shape.query_heads * shape.head_dim
    kv = shape.kv_heads * shape.head_dim
    projection_parameters = (
        shape.hidden * (query + 2 * kv)
        + query * shape.hidden
        + 3 * shape.hidden * shape.intermediate
    )
    projection = 2 * tokens * shape.layers * projection_parameters
    attention = examples * shape.layers * shape.query_heads * shape.head_dim
    return BatchWork(
        examples,
        sum(lengths),
        tokens,
        width,
        projection,
        4 * attention * width * width,
        2 * attention * width * (width + 1),
    )


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    load: int
    throughput: float

    def __post_init__(self) -> None:
        _count(self.load, "load")
        _positive(self.throughput, "throughput")


def usl_capacity(load: float, alpha: float, beta: float) -> float:
    """Normalized capacity N/(1 + alpha*(N-1) + beta*N*(N-1))."""
    _positive(load, "load")
    if (
        isinstance(alpha, bool)
        or load < 1
        or not math.isfinite(alpha)
        or not 0 <= alpha < 1
    ):
        raise ValueError("USL requires load >= 1 and alpha in [0, 1)")
    if isinstance(beta, bool) or not math.isfinite(beta) or beta < 0:
        raise ValueError("USL beta must be finite and nonnegative")
    return load / (1 + alpha * (load - 1) + beta * load * (load - 1))


@dataclass(frozen=True, slots=True)
class USLResidual:
    load: int
    observed_throughput: float
    predicted_throughput: float | None
    throughput_residual: float | None
    transformed_residual: float


@dataclass(frozen=True, slots=True)
class USLFit:
    alpha: float
    beta: float
    parameters_valid: bool
    baseline_throughput: float
    rows: tuple[USLResidual, ...]
    transformed_sse: float
    capacity_rmse: float | None
    capacity_r_squared: float | None
    max_relative_error: float | None
    normalized_gram_condition: float
    predicted_peak_load: float | None
    warnings: tuple[str, ...]


def fit_usl(points: Sequence[ScalingPoint]) -> USLFit:
    """Transformed least squares with explicit physical parameter acceptance.

    Normalize by the measured N=1 throughput and regress N/C(N)-1 against
    N-1 and N*(N-1). Accept a curve only for 0 <= alpha < 1, beta >= 0;
    invalid estimates are retained, never clipped into an apparently valid fit.
    Raw throughput residuals reveal distortions introduced by this transform;
    a fitted curve is never a causal diagnosis.
    """
    ordered = tuple(sorted(points, key=lambda point: point.load))
    if len(ordered) < 6 or ordered[0].load != 1:
        raise ValueError("USL fit requires at least six points including load 1")
    if len({point.load for point in ordered}) != len(ordered):
        raise ValueError("USL fit requires unique loads")
    baseline = ordered[0].throughput
    design = tuple(
        (
            point.load - 1,
            point.load * (point.load - 1),
            point.load * baseline / point.throughput - 1,
        )
        for point in ordered[1:]
    )
    xx = math.fsum(x * x for x, _, _ in design)
    xy = math.fsum(x * y for x, y, _ in design)
    yy = math.fsum(y * y for _, y, _ in design)
    xz = math.fsum(x * z for x, _, z in design)
    yz = math.fsum(y * z for _, y, z in design)
    correlation = xy / math.sqrt(xx * yy)
    if 1 - correlation <= 1e-12:
        raise ValueError("USL parameter columns are numerically unidentifiable")
    condition = (1 + correlation) / (1 - correlation)
    determinant = xx * yy - xy * xy
    alpha = (xz * yy - yz * xy) / determinant
    beta = (yz * xx - xz * xy) / determinant
    valid = 0 <= alpha < 1 and beta >= 0
    rows_list: list[USLResidual] = []
    for point in ordered:
        predicted = baseline * usl_capacity(point.load, alpha, beta) if valid else None
        rows_list.append(
            USLResidual(
                point.load,
                point.throughput,
                predicted,
                point.throughput - predicted if predicted is not None else None,
                point.load * baseline / point.throughput
                - 1
                - alpha * (point.load - 1)
                - beta * point.load * (point.load - 1),
            )
        )
    rows = tuple(rows_list)
    capacities = [point.throughput / baseline for point in ordered]
    mean = math.fsum(capacities) / len(capacities)
    sse = math.fsum(
        (row.throughput_residual / baseline) ** 2
        for row in rows
        if row.throughput_residual is not None
    )
    sst = math.fsum((value - mean) ** 2 for value in capacities)
    r_squared = 1 - sse / sst if valid and sst else None
    relative_error = (
        max(
            abs(row.throughput_residual) / row.observed_throughput
            for row in rows
            if row.throughput_residual is not None
        )
        if valid
        else None
    )
    peak = max(1.0, math.sqrt((1 - alpha) / beta)) if valid and beta else None
    warnings = [
        "Normalized fit inherits uncertainty from the measured load-1 baseline.",
        "No confidence intervals or mechanistic identifiability established.",
        "Transformed least squares weights raw throughput errors unequally.",
    ]
    if not valid:
        warnings.append(
            "Estimated parameters violate USL bounds; no curve or peak admitted."
        )
    if condition > 100:
        warnings.append(
            "Correlated parameter columns weaken numerical identifiability."
        )
    if peak is not None and not ordered[0].load <= peak <= ordered[-1].load:
        warnings.append(
            "Predicted peak lies outside observed loads and is an extrapolation."
        )
    if valid and (
        (r_squared is not None and r_squared < 0.9)
        or (relative_error is not None and relative_error > 0.1)
    ):
        warnings.append(
            "Weak descriptive fit: capacity R-squared < 0.9 or maximum relative "
            "error > 10%; illustrative diagnostics, not an acceptance gate."
        )
    return USLFit(
        alpha,
        beta,
        valid,
        baseline,
        rows,
        math.fsum(row.transformed_residual**2 for row in rows),
        math.sqrt(sse / len(rows)) if valid else None,
        r_squared,
        relative_error,
        condition,
        peak,
        tuple(warnings),
    )
