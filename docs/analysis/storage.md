# Exact 16-bit weight storage with FP32 computation

Read-only source audit on 2026-10-05. No maintained MLX 0.32.3 matmul path was
found that directly reads BF16 weights while preserving FP32 activations,
products and accumulation. The ordinary path explicitly promotes the weights
to FP32. This is a concrete blocker to treating smaller stored weights as a
demonstrated DRAM-traffic optimization. No kernel, installation, model execution
or hardware measurement was performed; historical receipts remain unchanged.

## What the source establishes

The inspected version is the `v0.32.3` release tag. In
[`mlx/ops.cpp`, lines 3513–3527](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/ops.cpp#L3513),
`matmul` selects one promoted dtype and inserts `astype` for either operand
that differs. FP32 activations with BF16 weights therefore create an FP32
weight operand before `Matmul`. The retrieved Git blob is
`83094d25b827c0a395b47ce31402fdd048b6fd9e`.

The standard
[`Steel GEMM`](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/gemm/gemm.h#L23)
has input type `T`, output type `U` and accumulation type; both A and B loaders
use the same `T`. Its
[`BlockLoader`](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/gemm/loader.h#L38)
uses that same type for device and threadgroup memory. `U` is not an independent
weight-input dtype. The
[`BlockMMA`](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/steel/gemm/mma.h#L439)
default accumulation type is float; that alone does not preserve FP32 inputs
when both operand loaders use BF16. Retrieved GEMM/loader/MMA blobs are
`bbe1d96cc0a657e729b2be8a0808382886866ba0`,
`c999fdc91acad1903ff824b36a7e9ca9c7b57675` and
`1d9e0a7820f1eee9276f7ff23cf96bcd5b8bdae5`.

`qqmm` does accept a non-quantized weight argument, but its
[`Metal implementation`](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/quantized.cpp#L2107)
quantizes that argument and quantizes/dequantizes activations. It is not a
lossless BF16-storage pass-through. `quantized_matmul` requires encoded weights
and scales: a separate quantization recipe whose exact weight reconstruction
and arithmetic would need verification. Output dtype alone cannot waive that
contract. The retrieved quantized implementation blob is
`97c418e872a5f21a6ae16522727ab8a853036976`.

Apple's [feature tables](https://developer.apple.com/metal/feature-sets/)
list BF16 scalar/vector support and SIMD matrix operations for the M4's Apple9
family. This does not establish a public MLX mixed-input operator or its
numerical parity. A custom BF16 device load followed by FP32 conversion inside
a tile is a plausible kernel technique, not an inspected maintained path.

The existing [tiny graph probe](../../benchmarks/analysis/results/mlx-mixed-storage.executed.py)
stores **FP16** weights built from BF16-representable source values. Its
`AsType`/FP32 matmul observation is consequently FP16-specific. The source
promotion rule above independently establishes the BF16 case; no BF16 GPU
measurement is inferred from that probe.

## Dimensions and conditional traffic ceiling

For the present checkpoint, a projection is `X[M,K] @ W[N,K].T`, where
`M=batch × padded_tokens`. Preserve FP32 X/output, every original weight value,
causal positions and the candidate head.

| Projection | Weight shape | FP32 bytes | BF16 bytes |
| --- | --- | --- | --- |
| Q | 2048 × 1024 | 8 MiB | 4 MiB |
| K and V, each | 1024 × 1024 | 4 MiB | 2 MiB |
| O | 1024 × 2048 | 8 MiB | 4 MiB |
| Gate and up, each | 3072 × 1024 | 12 MiB | 6 MiB |
| Down | 1024 × 3072 | 12 MiB | 6 MiB |

That is 15,728,640 projection elements per layer: 60 MiB FP32 versus 30 MiB
BF16; across 28 layers, 1,680 versus 840 MiB. Embeddings, norms, head and
workspace are excluded. Packed QKV is 4096 × 1024, not 3072 × 1024; packed
gate/up is 6144 × 1024. Packing does not change this element count.

If every projection weight were read once from DRAM per batch, direct 16-bit
loads would halve that **weight** traffic. Real cache reuse, tile rereads,
activations, outputs and other operators invalidate any automatic whole-engine
2× claim. The ideal ceiling is 2× for time entirely due to those weight reads;
with measured fraction `f`, the conditional total ceiling is
`1 / (1 - f/2)`. That fraction and actual DRAM traffic are unknown. Materializing
the existing cast may instead require reading 2 bytes, writing 4 and later
reading 4 per element; this 10-byte account is conditional on materialization
and excludes cache effects, not measured traffic.

## Narrow hypothesis and stopping boundary

A future kernel could decode exact BF16 weights to FP32 in registers or
threadgroup tiles, retain FP32 X and the maintained FP32 MMA/reduction policy,
and eliminate the full FP32 weight intermediate. It would require changing
the loader/kernel and dispatch contract; a Python dtype switch cannot do it.
BF16 storage is admissible only after exact FP32 round-trip checks per tensor.
Changed reduction order still requires strict numerical gates.

Before implementation, establish a measured weight-traffic/launch bottleneck
and an affected-time ceiling. If pursued, first test the actual rectangular
projection shapes and tail M, then the frozen full cohort, retaining rank,
selection, logit/probability drift and synchronized complete-request time.
Reject the hypothesis if cast buffers persist, precision gates fail or measured
complete-request benefit is absent. No maintained implementation or performance
claim is established by this audit.
