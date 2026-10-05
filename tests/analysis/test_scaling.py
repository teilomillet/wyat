from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from benchmarks.analysis.scaling import (
    ScalingPoint,
    TransformerShape,
    amdahl_ceiling,
    amdahl_speedup,
    estimate_batch_work,
    fit_usl,
    roofline_lower_bound,
    usl_capacity,
)


def test_amdahl_matches_independently_accounted_stage_seconds() -> None:
    # A 60-second operation: 45 eligible seconds become 15, plus 15 unchanged.
    assert amdahl_speedup(45 / 60, 3) == pytest.approx(60 / 30)
    assert amdahl_ceiling(45 / 60) == 4
    assert amdahl_ceiling(1) is None
    assert amdahl_speedup(0, 100) == 1


@given(st.floats(0, 0.999), st.floats(1, 1e6))
def test_amdahl_bounds_whole_system_gain(fraction: float, local: float) -> None:
    gain = amdahl_speedup(fraction, local)
    ceiling = amdahl_ceiling(fraction)
    assert ceiling is not None
    assert 1 <= gain <= min(local, ceiling) * (1 + 1e-12)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), True])
def test_invalid_measured_fraction_rejected(fraction: float) -> None:
    with pytest.raises(ValueError):
        amdahl_ceiling(fraction)


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan"), True])
def test_invalid_local_speedup_rejected(factor: float) -> None:
    with pytest.raises(ValueError):
        amdahl_speedup(0.5, factor)


def test_roofline_uses_slower_lower_bound_without_adding_overlapping_terms() -> None:
    bound = roofline_lower_bound(
        flops=300_000_000_000,
        bytes_moved=20_000_000_000,
        compute_flops_per_second=3_000_000_000_000,
        memory_bytes_per_second=100_000_000_000,
    )
    assert bound.compute_seconds == 0.1
    assert bound.memory_seconds == 0.2
    assert bound.lower_bound_seconds == 0.2
    assert bound.intensity_flops_per_byte == 15


@given(st.integers(0, 10**12), st.integers(0, 10**12))
def test_roofline_work_and_traffic_monotonicity(flops: int, traffic: int) -> None:
    first = roofline_lower_bound(
        flops=flops,
        bytes_moved=traffic,
        compute_flops_per_second=1e12,
        memory_bytes_per_second=1e11,
    )
    doubled = roofline_lower_bound(
        flops=2 * flops,
        bytes_moved=2 * traffic,
        compute_flops_per_second=1e12,
        memory_bytes_per_second=1e11,
    )
    assert doubled.lower_bound_seconds == 2 * first.lower_bound_seconds
    assert first.lower_bound_seconds >= first.compute_seconds
    assert first.lower_bound_seconds >= first.memory_seconds


def test_zero_traffic_does_not_serialize_infinite_intensity() -> None:
    bound = roofline_lower_bound(
        flops=1,
        bytes_moved=0,
        compute_flops_per_second=1,
        memory_bytes_per_second=1,
    )
    assert bound.intensity_flops_per_byte is None
    assert bound.lower_bound_seconds == 1


def test_gqa_arithmetic_counts_projection_width_independently_from_hidden() -> None:
    shape = TransformerShape(2, 4, 1, 2, 1, 3)
    # Q width 6, KV width 3. Projection weights: 24 QKV + 12 O + 24 MLP.
    single = estimate_batch_work((1,), shape, pad_multiple=1)
    assert single.projection_flops == 120
    assert single.dense_attention_flops == single.ideal_causal_attention_flops == 24
    batch = estimate_batch_work((1, 2), shape, pad_multiple=1)
    assert batch.projection_flops == 480
    assert batch.dense_attention_flops == 192
    assert batch.ideal_causal_attention_flops == 144
    assert batch.useful_tokens == 3 and batch.padded_tokens == 4
    assert batch.padding_fraction == 0.25


@given(st.lists(st.integers(1, 512), min_size=1, max_size=32))
def test_padding_never_removes_useful_work(lengths: list[int]) -> None:
    shape = TransformerShape(16, 32, 2, 4, 2, 8)
    work = estimate_batch_work(lengths, shape)
    assert work.useful_tokens == sum(lengths)
    assert work.padded_tokens >= work.useful_tokens
    assert work.padded_length >= max(lengths)
    assert work.padded_length % 8 == 0
    assert 0 <= work.padding_fraction < 1
    assert work.ideal_causal_attention_flops <= work.dense_attention_flops


@pytest.mark.parametrize("lengths", [(), (0,), (True,)])
def test_invalid_sequence_lengths_rejected(lengths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        estimate_batch_work(lengths, TransformerShape(2, 4, 1, 2, 1, 3))


def test_usl_reduces_to_linear_and_amdahl_special_cases() -> None:
    for load in (1, 2, 4, 8, 16, 32):
        assert usl_capacity(load, 0, 0) == load
        assert usl_capacity(load, 0.25, 0) == pytest.approx(amdahl_speedup(0.75, load))


def test_six_load_known_usl_fit_retains_raw_residuals() -> None:
    points = tuple(
        ScalingPoint(load, 100 * usl_capacity(load, 0.08, 0.002))
        for load in (1, 2, 4, 8, 16, 32)
    )
    fitted = fit_usl(points)
    assert fitted.parameters_valid
    assert fitted.alpha == pytest.approx(0.08)
    assert fitted.beta == pytest.approx(0.002)
    assert fitted.predicted_peak_load == pytest.approx(math.sqrt(0.92 / 0.002))
    assert len(fitted.rows) == len(points)
    assert fitted.capacity_r_squared == pytest.approx(1)
    assert fitted.max_relative_error == pytest.approx(0, abs=1e-12)
    assert all(
        row.transformed_residual == pytest.approx(0, abs=1e-12) for row in fitted.rows
    )


def test_superlinear_measurements_do_not_get_clipped_into_physical_parameters() -> None:
    points = tuple(ScalingPoint(load, 100 * load**1.5) for load in (1, 2, 4, 8, 16, 32))
    fitted = fit_usl(points)
    assert not fitted.parameters_valid
    assert fitted.alpha < 0 or fitted.beta < 0
    assert fitted.predicted_peak_load is None
    assert fitted.capacity_rmse is None and fitted.capacity_r_squared is None
    assert all(row.predicted_throughput is None for row in fitted.rows)
    assert all(
        row.observed_throughput == point.throughput
        for row, point in zip(fitted.rows, points, strict=True)
    )
    assert any("no curve or peak" in warning for warning in fitted.warnings)


def test_peak_outside_measured_loads_is_explicitly_extrapolated() -> None:
    fitted = fit_usl(
        tuple(
            ScalingPoint(load, usl_capacity(load, 0.1, 0.00001))
            for load in (1, 2, 4, 8, 16, 32)
        )
    )
    assert fitted.parameters_valid
    assert fitted.predicted_peak_load is not None
    assert fitted.predicted_peak_load > 32
    assert any("extrapolation" in warning for warning in fitted.warnings)


def test_condition_is_normalized_gram_eigenvalue_ratio() -> None:
    loads = (1, 2, 4, 8, 16, 32)
    fitted = fit_usl(
        tuple(ScalingPoint(load, usl_capacity(load, 0.1, 0.001)) for load in loads)
    )
    first = [float(load - 1) for load in loads]
    second = [float(load * (load - 1)) for load in loads]
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    aa = sum((value / first_norm) ** 2 for value in first)
    dd = sum((value / second_norm) ** 2 for value in second)
    bb = sum(
        (left / first_norm) * (right / second_norm)
        for left, right in zip(first, second, strict=True)
    )
    discriminant = math.sqrt((aa - dd) ** 2 + 4 * bb * bb)
    largest = (aa + dd + discriminant) / 2
    smallest = (aa + dd - discriminant) / 2
    assert fitted.normalized_gram_condition == pytest.approx(largest / smallest)
    assert math.sqrt(fitted.normalized_gram_condition) == pytest.approx(7.4201456)


def test_physical_parameters_do_not_hide_a_large_raw_capacity_residual() -> None:
    fitted = fit_usl(
        tuple(
            ScalingPoint(
                load, usl_capacity(load, 0.08, 0.002) * (0.5 if load == 2 else 1)
            )
            for load in (1, 2, 4, 8, 16, 32)
        )
    )
    assert fitted.parameters_valid
    assert fitted.max_relative_error is not None
    assert fitted.max_relative_error > 0.1
    assert fitted.rows[1].observed_throughput == usl_capacity(2, 0.08, 0.002) / 2
    assert any("Weak descriptive fit" in warning for warning in fitted.warnings)
    # A warning cannot change the fitted coefficients or reject a physical curve.
    assert fitted.rows[1].predicted_throughput is not None


@given(st.floats(0.01, 0.9), st.floats(0.00001, 0.02))
def test_exact_usl_parameters_are_recovered_without_boundary_projection(
    alpha: float,
    beta: float,
) -> None:
    fitted = fit_usl(
        tuple(
            ScalingPoint(load, usl_capacity(load, alpha, beta))
            for load in (1, 2, 4, 8, 16, 32)
        )
    )
    assert fitted.parameters_valid
    assert fitted.alpha == pytest.approx(alpha, abs=1e-10)
    assert fitted.beta == pytest.approx(beta, abs=1e-10)


def test_insufficient_and_duplicate_loads_reject_before_fit() -> None:
    with pytest.raises(ValueError, match="six"):
        fit_usl((ScalingPoint(1, 1), ScalingPoint(2, 2), ScalingPoint(4, 4)))
    with pytest.raises(ValueError, match="unique"):
        fit_usl(tuple(ScalingPoint(load, 1) for load in (1, 2, 4, 8, 16, 16)))


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (-0.01, 0.01), (0.1, -0.01)])
def test_usl_physical_domain_is_not_weakened_to_fit_data(
    alpha: float, beta: float
) -> None:
    with pytest.raises(ValueError):
        usl_capacity(2, alpha, beta)
