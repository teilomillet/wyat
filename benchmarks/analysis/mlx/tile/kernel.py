"""One 64×64×32 output-reuse hypothesis, with the frozen admission contract.

Sixteen float SIMD accumulators per SIMD group and 32 KiB of shared storage
are explicit costs. The initial 32-tile implementation remains untouched.
"""

from dataclasses import dataclass
from typing import cast

import mlx.core as mx

from benchmarks.analysis.mlx.mixed.kernel import HEADER, MetalKernel
from benchmarks.analysis.mlx.mixed.kernel import ProjectionKernel as Admission

TILE = 64
REDUCTION = 32
THREADS = 128
SHARED_BYTES = (2 * TILE * REDUCTION + TILE * TILE) * 4
SOURCE = """
const uint M = X_shape[0];
const uint K = X_shape[1];
const uint N = W_shape[0];
const uint tid = thread_index_in_threadgroup;
const uint group = simdgroup_index_in_threadgroup;
const uint row_start = threadgroup_position_in_grid.y * 64;
const uint col_start = threadgroup_position_in_grid.x * 64;
const uint group_row = (group / 2) * 32;
const uint group_col = (group % 2) * 32;

threadgroup float x_tile[2048];
threadgroup float w_tile[2048];
threadgroup float output_tile[4096];
simdgroup_float8x8 accumulators[16];
for (uint i = 0; i < 16; ++i) {
    accumulators[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

for (uint start = 0; start < K; start += 32) {
    for (uint i = tid; i < 2048; i += 128) {
        const uint local_row = i / 32;
        const uint local_col = i % 32;
        const uint row = row_start + local_row;
        const uint col = col_start + local_row;
        const uint reduction = start + local_col;
        x_tile[i] = (row < M && reduction < K)
            ? float(X[size_t(row) * K + reduction]) : 0.0f;
        // The weight storage changes; float staged operands do not.
        w_tile[local_col * 64 + local_row] = (col < N && reduction < K)
            ? float(W[size_t(col) * K + reduction]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint reduction = 0; reduction < 32; reduction += 8) {
        simdgroup_float8x8 a[4];
        simdgroup_float8x8 b[4];
        for (uint i = 0; i < 4; ++i) {
            simdgroup_load(a[i], x_tile + (group_row + i * 8) * 32
                + reduction, 32);
            simdgroup_load(b[i], w_tile + reduction * 64
                + group_col + i * 8, 64);
        }
        for (uint row = 0; row < 4; ++row) {
            for (uint col = 0; col < 4; ++col) {
                const uint i = row * 4 + col;
                simdgroup_multiply_accumulate(accumulators[i], a[row], b[col],
                    accumulators[i]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint row = 0; row < 4; ++row) {
    for (uint col = 0; col < 4; ++col) {
        simdgroup_store(accumulators[row * 4 + col], output_tile
            + (group_row + row * 8) * 64 + group_col + col * 8, 64);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = tid; i < 4096; i += 128) {
    const uint row = row_start + i / 64;
    const uint col = col_start + i % 64;
    if (row < M && col < N) {
        Y[size_t(row) * N + col] = output_tile[i];
    }
}
"""


@dataclass(frozen=True)
class PreparedProjection:
    """Ready immutable operands; no admission check or input copy inside run."""

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
    """Share the live 32-tile control's public validator; compile lazily once."""

    def __init__(self, admission: Admission) -> None:
        self._admission = admission
        self._projection = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="wyat_tile_projection",
                input_names=["X", "W"],
                output_names=["Y"],
                source=SOURCE,
                header=HEADER,
                ensure_row_contiguous=False,
                compile_options={"math_mode": "safe"},
            ),
        )

    def prepare(self, x: mx.array, weight: mx.array) -> PreparedProjection:
        admitted = self._admission.prepare(x, weight)
        return PreparedProjection(admitted.x, admitted.weight, self._projection)
