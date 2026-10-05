"""Tiny device exercises for the fixed mixed-storage Metal projection."""

from collections.abc import Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.mlx.mixed.kernel import ProjectionKernel


@pytest.fixture(scope="module")
def kernel() -> Iterator[ProjectionKernel]:
    if not mx.metal.is_available():
        pytest.skip("The custom projection requires a native Metal device")
    value = ProjectionKernel()
    yield value
    mx.synchronize()
    mx.clear_cache()


def check_projection(
    kernel: ProjectionKernel, rows: int, columns: int, reduction: int
) -> None:
    x = ((mx.arange(rows * reduction) % 23) - 11).astype(mx.float32)
    x = mx.contiguous((x / 16).reshape(rows, reduction))
    weight = ((mx.arange(columns * reduction) % 17) - 8).astype(mx.float32)
    weight = mx.contiguous((weight / 8).reshape(columns, reduction))
    stored = weight.astype(mx.bfloat16)
    # Generated finite dyadic weights are exact BF16, not a precision reduction.
    assert bool(
        mx.array_equal(
            weight.view(mx.uint32), stored.astype(mx.float32).view(mx.uint32)
        ).item()
    )
    control = kernel.prepare(x, weight).run()
    candidate = kernel.prepare(x, stored).run()
    reference = x @ weight.T
    mx.eval(control, candidate, reference)
    assert control.shape == (rows, columns)
    assert control.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(candidate)).item())
    assert bool(
        mx.array_equal(control.view(mx.uint32), candidate.view(mx.uint32)).item()
    )
    assert bool(
        mx.all(mx.abs(candidate - reference) <= 1e-4 + 1e-4 * mx.abs(reference)).item()
    )


@pytest.mark.parametrize(
    ("rows", "columns", "reduction"),
    [
        (1, 1, 1),
        (7, 9, 15),
        (31, 33, 31),
        (32, 32, 32),
        (33, 31, 33),
        (65, 63, 70),
        (17, 49, 64),
    ],
)
def test_tail_rows_columns_and_reduction(
    kernel: ProjectionKernel, rows: int, columns: int, reduction: int
) -> None:
    check_projection(kernel, rows, columns, reduction)


@given(
    rows=st.integers(1, 65),
    columns=st.integers(1, 65),
    reduction=st.integers(1, 70),
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_generated_geometry(
    kernel: ProjectionKernel, rows: int, columns: int, reduction: int
) -> None:
    check_projection(kernel, rows, columns, reduction)


def test_identity_and_repeated_ready_dispatch(kernel: ProjectionKernel) -> None:
    x = mx.arange(33 * 33, dtype=mx.float32).reshape(33, 33)
    weight = mx.eye(33, dtype=mx.float32)
    prepared = kernel.prepare(x, weight.astype(mx.bfloat16))
    first, second = prepared.run(), prepared.run()
    mx.eval(first, second)
    assert bool(mx.array_equal(first, x).item())
    assert bool(mx.array_equal(first.view(mx.uint32), second.view(mx.uint32)).item())


@pytest.mark.parametrize("operand", ["x", "weight"])
@pytest.mark.parametrize("layout", ["transpose", "step", "broadcast", "reverse"])
def test_noncontiguous_is_rejected_without_copy(
    kernel: ProjectionKernel, operand: str, layout: str
) -> None:
    source = mx.arange(8 * 8, dtype=mx.float32).reshape(8, 8)
    if layout == "transpose":
        value = source.T
    elif layout == "step":
        value = source[:, ::2]
    elif layout == "broadcast":
        value = mx.broadcast_to(source[:1], (8, 8))
    else:
        value = source[:, ::-1]
    if operand == "x":
        x, weight = value, mx.ones((3, value.shape[1]), dtype=mx.float32)
    else:
        x, weight = mx.ones((3, value.shape[1]), dtype=mx.float32), value
    with pytest.raises(ValueError, match="row-contiguous"):
        kernel.prepare(x, weight)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.int32])
def test_activation_dtype_rejected(kernel: ProjectionKernel, dtype: mx.Dtype) -> None:
    with pytest.raises(ValueError, match="FP32 X"):
        kernel.prepare(mx.ones((2, 3), dtype=dtype), mx.ones((2, 3), dtype=mx.float32))


@pytest.mark.parametrize("dtype", [mx.float16, mx.int32])
def test_weight_dtype_rejected(kernel: ProjectionKernel, dtype: mx.Dtype) -> None:
    with pytest.raises(ValueError, match="BF16 W"):
        kernel.prepare(mx.ones((2, 3), dtype=mx.float32), mx.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("xshape", [(3,), (1, 2, 3), (0, 3), (2, 0)])
def test_invalid_activation_shape(
    kernel: ProjectionKernel, xshape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="rank-two|positive"):
        kernel.prepare(mx.zeros(xshape), mx.zeros((2, 3)))


@pytest.mark.parametrize("wshape", [(3,), (1, 2, 3), (0, 3), (2, 0)])
def test_invalid_weight_shape(
    kernel: ProjectionKernel, wshape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="rank-two|positive"):
        kernel.prepare(mx.zeros((2, 3)), mx.zeros(wshape))


def test_reduction_mismatch(kernel: ProjectionKernel) -> None:
    with pytest.raises(ValueError, match="reduction dimensions"):
        kernel.prepare(mx.ones((2, 3)), mx.ones((2, 4)))


def test_signed_zero_weights_keep_custom_storage_parity(
    kernel: ProjectionKernel,
) -> None:
    x = mx.array([[1.0, -1.0, 2.0], [-2.0, 4.0, -3.0]])
    weight = mx.array([[0.0, -0.0, 0.0], [-0.0, 0.0, -0.0]])
    stored = weight.astype(mx.bfloat16)
    assert bool(
        mx.array_equal(
            weight.view(mx.uint32), stored.astype(mx.float32).view(mx.uint32)
        ).item()
    )
    first = kernel.prepare(x, weight).run()
    second = kernel.prepare(x, stored).run()
    mx.eval(first, second)
    assert bool(mx.array_equal(first.view(mx.uint32), second.view(mx.uint32)).item())


def test_activation_low_bits_are_preserved(kernel: ProjectionKernel) -> None:
    # The activation differs from 1 by FP32 bits absent in FP16 and BF16.
    x = mx.full((33, 1), 1.0 + 2.0**-20, dtype=mx.float32)
    weight = mx.ones((33, 1), dtype=mx.bfloat16)
    result = kernel.prepare(x, weight).run()
    expected = mx.full((33, 33), 1.0 + 2.0**-20, dtype=mx.float32)
    mx.eval(result, expected)
    assert bool(mx.array_equal(result.view(mx.uint32), expected.view(mx.uint32)).item())


def test_non_bf16_activation_values(kernel: ProjectionKernel) -> None:
    x = mx.contiguous(
        (mx.sin(mx.arange(35 * 67, dtype=mx.float32) * 0.173) * 1.1234567).reshape(
            35, 67
        )
    )
    stored = mx.contiguous(
        mx.cos(mx.arange(41 * 67, dtype=mx.float32) * 0.251)
        .reshape(41, 67)
        .astype(mx.bfloat16)
    )
    weight = stored.astype(mx.float32)
    control = kernel.prepare(x, weight).run()
    candidate = kernel.prepare(x, stored).run()
    reference = x @ weight.T
    mx.eval(control, candidate, reference)
    assert bool(
        mx.array_equal(control.view(mx.uint32), candidate.view(mx.uint32)).item()
    )
    assert bool(
        mx.all(mx.abs(candidate - reference) <= 1e-4 + 1e-4 * mx.abs(reference)).item()
    )
