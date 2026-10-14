# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Lossless projection storage and FP32 Metal matmul for qualified M4 Pro shapes.

This optional path follows MLX 0.32.3's split-K partition and addition order.
Only exact BF16-representable finite weights are compressed; activations,
partial sums, outputs and the decision head remain FP32. Other runtimes,
architectures, compiled resources and projection geometries stay native.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
from typing import cast

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.utils import tree_unflatten
from mlx_lm.models.qwen3 import Qwen3Model

Projection = Callable[[mx.array, mx.array], mx.array]
HEADER = """
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""
_GEOMETRIES = frozenset(
    {(1024, 1024), (1024, 2048), (1024, 3072), (2048, 1024), (3072, 1024)}
)


def _direct(tile_m: int = 32, tile_n: int = 32, groups: int = 4) -> Projection:
    """Compute X[M,K] @ W[N,K].T with FP32 activations and output."""
    source = f"""
    const int rows = X_shape[0];
    const int reduction = X_shape[1];
    const int columns = W_shape[0];
    const int offset_m = threadgroup_position_in_grid.y * {tile_m};
    const int offset_n = threadgroup_position_in_grid.x * {tile_n};
    // MPP dispatch traits require an unqualified scalar element type. The
    // operation treats its two source tensors as read-only.
    tensor<device float, dextents<int, 2>, tensor_inline> x(
        const_cast<device float*>(X), dextents<int, 2>(reduction, rows),
        array<int, 2>{{1, reduction}});
    tensor<device Weight, dextents<int, 2>, tensor_inline> w(
        const_cast<device Weight*>(W), dextents<int, 2>(reduction, columns),
        array<int, 2>{{1, reduction}});
    tensor<device float, dextents<int, 2>, tensor_inline> y(
        Y, dextents<int, 2>(columns, rows), array<int, 2>{{1, columns}});
    auto x_tile = x.slice(0, offset_m);
    auto w_tile = w.slice(0, offset_n);
    auto y_tile = y.slice(offset_n, offset_m);
    constexpr auto descriptor = matmul2d_descriptor(
        {tile_m}, {tile_n}, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<descriptor, execution_simdgroups<{groups}>> operation;
    operation.run(x_tile, w_tile, y_tile);
    """
    kernel = cast(
        Callable[..., list[mx.array]],
        mx.fast.metal_kernel(
            name=f"wyat_tensorops_fp32_{tile_m}_{tile_n}_{groups}",
            input_names=["X", "W"],
            output_names=["Y"],
            source=source,
            header=HEADER,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )

    def run(x: mx.array, weight: mx.array) -> mx.array:
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("Requires X[M,K] and W[N,K]")
        if x.dtype != mx.float32 or weight.dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("Requires FP32 input and FP32 or BF16 weights")
        rows, columns = x.shape[0], weight.shape[0]
        return kernel(
            inputs=[x, weight],
            template=[("Weight", weight.dtype)],
            grid=(
                32 * groups * ((columns + tile_n - 1) // tile_n),
                (rows + tile_m - 1) // tile_m,
                1,
            ),
            threadgroup=(32 * groups, 1, 1),
            output_shapes=[(rows, columns)],
            output_dtypes=[mx.float32],
        )[0]

    return run


def _split(partitions: int, partition_size: int) -> Projection:
    """Compute all supplied K partitions in one Metal output-tile dispatch."""
    if type(partitions) is not int or type(partition_size) is not int:
        raise ValueError("Partition count and size must be integers")
    if partitions < 1 or partition_size < 1:
        raise ValueError("Partition count and size must be positive")
    source = f"""
    const int rows = X_shape[0];
    const int reduction = X_shape[1];
    const int columns = W_shape[0];
    const int offset_m = threadgroup_position_in_grid.y * 16;
    const int offset_n = threadgroup_position_in_grid.x * 32;
    tensor<device float, dextents<int, 2>, tensor_inline> x(
        const_cast<device float*>(X), dextents<int, 2>(reduction, rows),
        array<int, 2>{{1, reduction}});
    tensor<device Weight, dextents<int, 2>, tensor_inline> w(
        const_cast<device Weight*>(W), dextents<int, 2>(reduction, columns),
        array<int, 2>{{1, reduction}});
    tensor<device float, dextents<int, 2>, tensor_inline> y(
        Y, dextents<int, 2>(columns, rows), array<int, 2>{{1, columns}});
    auto x_tile = x.slice(0, offset_m);
    auto w_tile = w.slice(0, offset_n);
    auto y_tile = y.slice(offset_n, offset_m);
    constexpr auto descriptor = matmul2d_descriptor(
        16, 32, static_cast<int>(dynamic_extent), false, true, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<descriptor, execution_simdgroups<2>> operation;
    using Left = remove_addrspace_t<decltype(x_tile)>;
    using Right = remove_addrspace_t<decltype(w_tile)>;
    auto partial = operation.get_destination_cooperative_tensor<Left, Right, float>();
    auto total = operation.get_destination_cooperative_tensor<Left, Right, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < total.get_capacity(); ++i) {{
        if (total.is_valid_element(i)) {{
            total[i] = 0.0f;
        }}
    }}
    for (int part = 0; part < {partitions}; ++part) {{
        const int start = part * {partition_size};
        const int length = part == {partitions - 1}
            ? reduction - start : {partition_size};
        // The K extent ends at this partition, while row strides retain the
        // original storage width. The last partition owns the full remainder.
        tensor<device float, dextents<int, 2>, tensor_inline> x_part(
            const_cast<device float*>(X) + start,
            dextents<int, 2>(length, rows), array<int, 2>{{1, reduction}});
        tensor<device Weight, dextents<int, 2>, tensor_inline> w_part(
            const_cast<device Weight*>(W) + start,
            dextents<int, 2>(length, columns), array<int, 2>{{1, reduction}});
        auto x_block = x_part.slice(0, offset_m);
        auto w_block = w_part.slice(0, offset_n);
        operation.run(x_block, w_block, partial);
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < total.get_capacity(); ++i) {{
            if (total.is_valid_element(i)) {{
                total[i] = total[i] + partial[i];
            }}
        }}
    }}
    total.store(y_tile);
    """
    kernel = cast(
        Callable[..., list[mx.array]],
        mx.fast.metal_kernel(
            name=f"wyat_tensorops_split_fused_v2_p{partitions}_k{partition_size}",
            input_names=["X", "W"],
            output_names=["Y"],
            source=source,
            header=HEADER,
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        ),
    )

    def run(x: mx.array, weight: mx.array) -> mx.array:
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("Requires X[M,K] and W[N,K]")
        if x.dtype != mx.float32 or weight.dtype not in (mx.float32, mx.bfloat16):
            raise ValueError("Requires FP32 input and FP32 or BF16 weights")
        if (partitions - 1) * partition_size >= x.shape[1]:
            raise ValueError("Every K partition must contain at least one element")
        rows, columns = x.shape[0], weight.shape[0]
        return kernel(
            inputs=[x, weight],
            template=[("Weight", weight.dtype)],
            grid=(64 * ((columns + 31) // 32), (rows + 15) // 16, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(rows, columns)],
            output_dtypes=[mx.float32],
        )[0]

    return run


def _partitions(rows: int, reduction: int, columns: int) -> tuple[int, int]:
    # Pinned SIMD split-K routing. NAX requires generation >=17; g16s never
    # selects it, independently of MLX_ENABLE_TF32. Never alter that setting.
    if ((rows + 15) // 16) * ((columns + 15) // 16) <= 2048 and reduction >= max(
        rows, columns
    ):
        ratio = (reduction // 16) // (((rows + 31) // 32) * ((columns + 31) // 32))
        count = min(max(2, 1 << max(0, (ratio - 1).bit_length())), 32)
        return count, (reduction // 16 // count) * 16
    return 1, reduction


def _widen(weight: mx.array) -> mx.array:
    return (weight.view(mx.uint16).astype(mx.uint32) << 16).view(mx.float32)


class MetalProjection:
    """One owner for direct/split kernels and native fallback dispatch."""

    def __init__(self) -> None:
        self.direct = _direct(16, 32, 2)
        self.splits: dict[tuple[int, int], Projection] = {}

    def __call__(self, x: mx.array, weight: mx.array) -> mx.array:
        rows, reduction = x.shape
        columns = weight.shape[0]
        if (
            x.dtype != mx.float32
            or weight.dtype != mx.bfloat16
            or (reduction, columns) not in _GEOMETRIES
            or rows < 2
            or rows >= 2**31
        ):
            rhs = _widen(weight) if weight.dtype == mx.bfloat16 else weight
            return mx.matmul(x, rhs.T)
        key = _partitions(rows, reduction, columns)
        if key[0] == 1:
            return self.direct(x, weight)
        operation = self.splits.get(key)
        if operation is None:
            operation = _split(*key)
            self.splits[key] = operation
        return operation(x, weight)


def _storage(weight: mx.array) -> mx.array | None:
    """Preserve every source bit, including signed zero and BF16 subnormals."""
    if weight.dtype == mx.bfloat16:
        flat = weight.reshape(-1)
        for start in range(0, flat.size, 1 << 20):
            if not bool(mx.all(mx.isfinite(flat[start : start + (1 << 20)])).item()):
                raise ValueError("Nonfinite projection weights")
        return weight
    if weight.dtype != mx.float32:
        return None
    words = weight.view(mx.uint32)
    flat = weight.reshape(-1)
    eligible = True
    for start in range(0, flat.size, 1 << 20):
        part = flat[start : start + (1 << 20)]
        finite, exact = cast(
            list[bool],
            mx.stack(
                [
                    mx.all(mx.isfinite(part)),
                    mx.all((part.view(mx.uint32) & 0xFFFF) == 0),
                ]
            ).tolist(),
        )
        if not finite:
            raise ValueError("Nonfinite projection weights")
        eligible = eligible and exact
    if not eligible:
        return None
    stored = (words >> 16).astype(mx.uint16).view(mx.bfloat16)
    mx.eval(stored)
    return stored


class _MetalLinear(Module):  # type: ignore[misc, unused-ignore]
    def __init__(self, weight: mx.array, operation: MetalProjection) -> None:
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.weight = weight
        self.operation = operation

    def __call__(self, value: mx.array) -> mx.array:
        if value.ndim < 1 or value.shape[-1] != self.weight.shape[1]:
            raise ValueError("Expected [..., input_features] projection input")
        if value.dtype != mx.float32:
            raise ValueError("Metal projection activations must remain FP32")
        shape = (*value.shape[:-1], self.weight.shape[0])
        if value.size == 0:
            return mx.zeros(shape, dtype=mx.float32)
        return self.operation(value.reshape(-1, value.shape[-1]), self.weight).reshape(
            shape
        )


def _unavailable(error: RuntimeError) -> bool:
    message = str(error)
    if any(
        word in message.lower()
        for word in (
            "out of memory",
            "allocation failed",
            "resource exhausted",
            "bad_alloc",
            "cannot allocate",
            "failed to allocate",
        )
    ):
        return False
    return message.startswith(
        "[metal::Device] Unable to build metal library from source"
    ) and any(
        marker in message
        for marker in (
            "'metal_tensor' file not found",
            "'MetalPerformancePrimitives/MetalPerformancePrimitives.h' file not found",
            "no template named 'tensor'",
            "use of undeclared identifier 'tensor_inline'",
            "no member named 'tensor_ops' in namespace 'mpp'",
            "unknown type name 'bfloat'",
            "no member named 'is_valid_element'",
        )
    )


def _capability() -> MetalProjection | None:
    if not callable(getattr(mx.fast, "metal_kernel", None)):
        return None
    try:
        operation = MetalProjection()
        x = ((mx.arange(17 * 1024, dtype=mx.float32) % 17 - 8) / 8).reshape(17, 1024)
        # Exercise both native reduction regimes and a partial output tile.
        for columns in (1024, 2048):
            weight = (
                ((mx.arange(columns * 1024, dtype=mx.float32) % 13 - 6) / 4)
                .reshape(columns, 1024)
                .astype(mx.bfloat16)
            )
            output = operation(x, weight)
            reference = mx.matmul(x, _widen(weight).T)
            mx.eval(output, reference)
            if (
                output.shape != reference.shape
                or output.dtype != mx.float32
                or not bool(mx.all(mx.isfinite(output) & (output == reference)).item())
            ):
                raise ValueError(
                    "Metal projection capability probe failed numerical comparison"
                )
    except TypeError as error:
        message = str(error)
        if "compile_options" in message and (
            "unexpected keyword" in message
            or "incompatible function arguments" in message
        ):
            return None
        raise
    except RuntimeError as error:
        if _unavailable(error):
            return None
        raise
    return operation


def _prepare(*, device: mx.Device, compiled: bool) -> MetalProjection | None:
    """Resolve optional capability before deciding checkpoint weight storage."""
    if compiled or device.type != mx.gpu or version("mlx") != "0.32.3":
        return None
    if mx.device_info(device).get("architecture") != "applegpu_g16s":
        return None
    with mx.stream(device):
        return _capability()


def _eligible(name: str, layer: Module) -> bool:
    return (
        isinstance(layer, Linear)
        and "bias" not in layer
        and name.startswith("layers.")
        and layer.weight.ndim == 2
        and (layer.weight.shape[1], layer.weight.shape[0]) in _GEOMETRIES
    )


def _checkpoint_weights(backbone: Qwen3Model) -> Callable[[str, mx.array], mx.array]:
    """Give admitted bias-free owners exact BF16 storage independent of loading."""
    names = frozenset(
        f"{name}.weight"
        for name, layer in cast(list[tuple[str, Module]], backbone.named_modules())
        if _eligible(name, layer)
    )

    def prepare(name: str, weight: mx.array) -> mx.array:
        if name in names and weight.dtype == mx.bfloat16:
            # Give inference independent storage so loading buffers can be
            # released. Evaluate each exact copy to bound temporary ownership.
            stored = mx.asarray(weight, copy=True)
            mx.eval(stored)
            return stored
        return weight.astype(mx.float32)

    return prepare


def optimize_projections(
    backbone: Qwen3Model,
    *,
    device: mx.Device,
    compiled: bool,
    operation: MetalProjection | None = None,
) -> bool:
    """Prepare an owned, newly loaded backbone; update modules only after checks."""
    if compiled or device.type != mx.gpu:
        return False
    if operation is None:
        operation = _prepare(device=device, compiled=compiled)
    if operation is None:
        return False
    with mx.stream(device):
        replacements: list[tuple[str, Module]] = []
        for name, layer in cast(list[tuple[str, Module]], backbone.named_modules()):
            if not _eligible(name, layer):
                continue
            stored = _storage(layer.weight)
            if stored is not None:
                replacements.append((name, _MetalLinear(stored, operation)))
        if not replacements:
            return False
        backbone.update_modules(tree_unflatten(replacements), strict=True)
    return True
