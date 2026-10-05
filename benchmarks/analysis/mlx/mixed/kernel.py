"""One fixed tile policy: float SIMD arithmetic with independent weight storage.

Preparation checks actual device strides without asking MLX to copy inputs.
Only the public custom Metal API and Metal SIMD matrix operations are used.
This experiment does not replace production matmul.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import mlx.core as mx

TILE = 32
THREADS = 128
SHARED_BYTES = 3 * TILE * TILE * 4
HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
"""
SOURCE = """
const uint M = X_shape[0];
const uint K = X_shape[1];
const uint N = W_shape[0];
const uint tid = thread_index_in_threadgroup;
const uint group = simdgroup_index_in_threadgroup;
const uint row_start = threadgroup_position_in_grid.y * 32;
const uint col_start = threadgroup_position_in_grid.x * 32;
const uint group_row = (group / 2) * 16;
const uint group_col = (group % 2) * 16;

threadgroup float x_tile[1024];
threadgroup float w_tile[1024];
threadgroup float output_tile[1024];
simdgroup_float8x8 accumulators[4];
for (uint i = 0; i < 4; ++i) {
    accumulators[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

for (uint start = 0; start < K; start += 32) {
    for (uint i = tid; i < 1024; i += 128) {
        const uint local_row = i / 32;
        const uint local_col = i % 32;
        const uint row = row_start + local_row;
        const uint col = col_start + local_row;
        const uint reduction = start + local_col;
        x_tile[i] = (row < M && reduction < K)
            ? float(X[size_t(row) * K + reduction]) : 0.0f;
        // Stage W.T in float; no complete FP32 weight allocation.
        w_tile[local_col * 32 + local_row] = (col < N && reduction < K)
            ? float(W[size_t(col) * K + reduction]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint reduction = 0; reduction < 32; reduction += 8) {
        simdgroup_float8x8 a[2];
        simdgroup_float8x8 b[2];
        for (uint i = 0; i < 2; ++i) {
            simdgroup_load(a[i], x_tile + (group_row + i * 8) * 32
                + reduction, 32);
            simdgroup_load(b[i], w_tile + reduction * 32
                + group_col + i * 8, 32);
        }
        for (uint row = 0; row < 2; ++row) {
            for (uint col = 0; col < 2; ++col) {
                const uint i = row * 2 + col;
                simdgroup_multiply_accumulate(accumulators[i], a[row], b[col],
                    accumulators[i]);
            }
        }
    }
    // No thread overwrites staged operands before every SIMD group finishes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint row = 0; row < 2; ++row) {
    for (uint col = 0; col < 2; ++col) {
        simdgroup_store(accumulators[row * 2 + col], output_tile
            + (group_row + row * 8) * 32 + group_col + col * 8, 32);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 1024; i += 128) {
    const uint row = row_start + i / 32;
    const uint col = col_start + i % 32;
    if (row < M && col < N) {
        Y[size_t(row) * N + col] = output_tile[i];
    }
}
"""

LAYOUT_SOURCE = """
if (thread_position_in_grid.x == 0) {
    layout[0] = X_strides[0];
    layout[1] = X_strides[1];
    layout[2] = W_strides[0];
    layout[3] = W_strides[1];
}
"""


class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[mx.array],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[mx.Dtype],
    ) -> list[mx.array]: ...


@dataclass(frozen=True)
class PreparedProjection:
    """Validated inputs held ready; run excludes preparation and layout dispatch."""

    x: mx.array
    weight: mx.array
    kernel: MetalKernel

    def run(self) -> mx.array:
        rows, _ = self.x.shape
        columns, _ = self.weight.shape
        return self.kernel(
            inputs=[self.x, self.weight],
            grid=(
                THREADS * ((columns + TILE - 1) // TILE),
                (rows + TILE - 1) // TILE,
                1,
            ),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[(rows, columns)],
            output_dtypes=[mx.float32],
        )[0]


class ProjectionKernel:
    """Build once; MLX compiles specializations lazily on evaluated dispatch."""

    def __init__(self) -> None:
        self._projection = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="wyat_mixed_projection",
                input_names=["X", "W"],
                output_names=["Y"],
                source=SOURCE,
                header=HEADER,
                ensure_row_contiguous=False,
                compile_options={"math_mode": "safe"},
            ),
        )
        self._layout = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="wyat_mixed_layout",
                input_names=["X", "W"],
                output_names=["layout"],
                source=LAYOUT_SOURCE,
                ensure_row_contiguous=False,
                compile_options={"math_mode": "safe"},
            ),
        )

    def prepare(self, x: mx.array, weight: mx.array) -> PreparedProjection:
        if x.dtype != mx.float32 or weight.dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("Projection requires FP32 X and FP32 or BF16 W")
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError("Projection requires rank-two X[M,K] and W[N,K]")
        if any(d <= 0 or d > 2**31 - 1 for d in (*x.shape, *weight.shape)):
            raise ValueError("Projection dimensions must be positive int32 values")
        if x.shape[1] != weight.shape[1]:
            raise ValueError("Projection reduction dimensions differ")
        layout = self._layout(
            inputs=[x, weight],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
            output_shapes=[(4,)],
            output_dtypes=[mx.int64],
        )[0]
        mx.eval(layout)
        observed = cast(list[int], layout.tolist())
        expected = [x.shape[1], 1, weight.shape[1], 1]
        if observed != expected:
            raise ValueError(f"Projection requires row-contiguous inputs: {observed}")
        return PreparedProjection(x, weight, self._projection)
