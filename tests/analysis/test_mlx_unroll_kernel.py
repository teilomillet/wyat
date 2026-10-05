"""The annotated 64 tile must preserve every original numerical boundary."""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("mlx.core", exc_type=ImportError)

import mlx.core as mx

from benchmarks.analysis.mlx.mixed.kernel import ProjectionKernel as FrozenKernel
from benchmarks.analysis.mlx.tile.kernel import SOURCE as TILE_SOURCE
from benchmarks.analysis.mlx.tile.kernel import ProjectionKernel as TileKernel
from benchmarks.analysis.mlx.unroll.kernel import SOURCE, ProjectionKernel


@dataclass(frozen=True)
class Kernels:
    original: TileKernel
    candidate: ProjectionKernel


@pytest.fixture(scope="module")
def kernels() -> Iterator[Kernels]:
    if not mx.metal.is_available():
        pytest.skip("The custom projection requires a native Metal device")
    original = TileKernel(FrozenKernel())
    yield Kernels(original, ProjectionKernel(original))
    mx.synchronize()
    mx.clear_cache()


def test_only_fragment_loop_annotations_change_the_metal_body() -> None:
    # This constrains the experiment to one variable; it is not a compiler proof.
    pragma = "#pragma clang loop unroll(full)"
    without_annotations = "\n".join(
        line for line in SOURCE.splitlines() if line.strip() != pragma
    )
    assert without_annotations == TILE_SOURCE.rstrip("\n")
    assert SOURCE.count(pragma) == 7


def check_projection(kernels: Kernels, rows: int, columns: int, reduction: int) -> None:
    x = mx.contiguous(
        (
            mx.sin(mx.arange(rows * reduction, dtype=mx.float32) * 0.173) * 1.1234567
        ).reshape(rows, reduction)
    )
    stored = mx.contiguous(
        mx.cos(mx.arange(columns * reduction, dtype=mx.float32) * 0.251)
        .reshape(columns, reduction)
        .astype(mx.bfloat16)
    )
    weight = stored.astype(mx.float32)
    control = kernels.candidate.prepare(x, weight).run()
    candidate = kernels.candidate.prepare(x, stored).run()
    original_fp32 = kernels.original.prepare(x, weight).run()
    original_bf16 = kernels.original.prepare(x, stored).run()
    reference = x @ weight.T
    mx.eval(control, candidate, original_fp32, original_bf16, reference)
    assert candidate.shape == (rows, columns)
    assert candidate.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(candidate)).item())
    for original in (control, original_fp32, original_bf16):
        assert bool(
            mx.array_equal(original.view(mx.uint32), candidate.view(mx.uint32)).item()
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
        (63, 65, 31),
        (64, 64, 32),
        (65, 63, 33),
        (129, 127, 97),
        (77, 99, 64),
    ],
)
def test_tail_rows_columns_and_reduction(
    kernels: Kernels, rows: int, columns: int, reduction: int
) -> None:
    check_projection(kernels, rows, columns, reduction)


@given(
    rows=st.integers(1, 129), columns=st.integers(1, 129), reduction=st.integers(1, 99)
)
@settings(max_examples=12, deadline=None, derandomize=True)
def test_generated_geometry(
    kernels: Kernels, rows: int, columns: int, reduction: int
) -> None:
    check_projection(kernels, rows, columns, reduction)


def test_admission_preserves_operand_identity(kernels: Kernels) -> None:
    x = mx.ones((65, 33), dtype=mx.float32)
    weight = mx.ones((63, 33), dtype=mx.bfloat16)
    prepared = kernels.candidate.prepare(x, weight)
    assert prepared.x is x
    assert prepared.weight is weight


def test_identity_and_repeated_ready_dispatch(kernels: Kernels) -> None:
    x = mx.arange(65 * 65, dtype=mx.float32).reshape(65, 65)
    weight = mx.eye(65, dtype=mx.bfloat16)
    prepared = kernels.candidate.prepare(x, weight)
    first, second = prepared.run(), prepared.run()
    mx.eval(first, second)
    assert bool(mx.array_equal(first, x).item())
    assert bool(mx.array_equal(first.view(mx.uint32), second.view(mx.uint32)).item())


@pytest.mark.parametrize("operand", ["x", "weight"])
@pytest.mark.parametrize("layout", ["transpose", "step", "broadcast", "reverse"])
def test_noncontiguous_rejected(kernels: Kernels, operand: str, layout: str) -> None:
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
        kernels.candidate.prepare(x, weight)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.int32])
def test_activation_dtype_rejected(kernels: Kernels, dtype: mx.Dtype) -> None:
    with pytest.raises(ValueError, match="FP32 X"):
        kernels.candidate.prepare(
            mx.ones((2, 3), dtype=dtype), mx.ones((2, 3), dtype=mx.float32)
        )


@pytest.mark.parametrize("dtype", [mx.float16, mx.int32])
def test_weight_dtype_rejected(kernels: Kernels, dtype: mx.Dtype) -> None:
    with pytest.raises(ValueError, match="BF16 W"):
        kernels.candidate.prepare(
            mx.ones((2, 3), dtype=mx.float32), mx.ones((2, 3), dtype=dtype)
        )


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3), (0, 3), (2, 0)])
@pytest.mark.parametrize("operand", ["x", "weight"])
def test_invalid_shape(kernels: Kernels, shape: tuple[int, ...], operand: str) -> None:
    value, valid = mx.zeros(shape), mx.zeros((2, 3))
    x, weight = (value, valid) if operand == "x" else (valid, value)
    with pytest.raises(ValueError, match="rank-two|positive"):
        kernels.candidate.prepare(x, weight)


def test_reduction_mismatch(kernels: Kernels) -> None:
    with pytest.raises(ValueError, match="reduction dimensions"):
        kernels.candidate.prepare(mx.ones((2, 3)), mx.ones((2, 4)))


def test_signed_zero_weights_preserve_storage_and_policy_parity(
    kernels: Kernels,
) -> None:
    x = mx.array([[1.0, -1.0, 2.0], [-2.0, 4.0, -3.0]])
    weight = mx.array([[0.0, -0.0, 0.0], [-0.0, 0.0, -0.0]])
    stored = weight.astype(mx.bfloat16)
    assert bool(
        mx.array_equal(
            weight.view(mx.uint32), stored.astype(mx.float32).view(mx.uint32)
        ).item()
    )
    candidate = kernels.candidate.prepare(x, stored).run()
    controls = [
        kernels.candidate.prepare(x, weight).run(),
        kernels.original.prepare(x, weight).run(),
        kernels.original.prepare(x, stored).run(),
    ]
    mx.eval(candidate, *controls)
    for original in controls:
        assert bool(
            mx.array_equal(original.view(mx.uint32), candidate.view(mx.uint32)).item()
        )


def test_activation_low_bits_preserved(kernels: Kernels) -> None:
    x = mx.full((65, 1), 1.0 + 2.0**-20, dtype=mx.float32)
    weight = mx.ones((65, 1), dtype=mx.bfloat16)
    candidate = kernels.candidate.prepare(x, weight).run()
    expected = mx.full((65, 65), 1.0 + 2.0**-20, dtype=mx.float32)
    mx.eval(candidate, expected)
    assert bool(
        mx.array_equal(candidate.view(mx.uint32), expected.view(mx.uint32)).item()
    )
