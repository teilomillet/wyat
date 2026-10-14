# Direct BF16 weight loads with FP32 projection arithmetic

Read-only orientation, 2026-10-05. Recommend one bounded primitive experiment,
not a production change. No framework import, model execution, kernel/JIT,
device probe or timing occurred during this investigation. Existing receipts
and the reserved CPU timing window remain untouched.

## Supported facility and present blocker

Installed MLX is 0.32.3. Ordinary `matmul` promotes both operands to a common
dtype, so FP32 activations plus BF16 weights materialize an FP32 weight operand
before multiplication. Its source and earlier storage audit are retained in
[storage.md](storage.md). This is the current promoted matmul baseline; it
already has FP32 weights and should not be charged a fictional per-call cast.

The maintained public [custom Metal API](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
generates each input pointer from that input's dtype, independently of output
dtype and template parameters. It can therefore accept FP32 X, BF16 W and
FP32 output. `ensure_row_contiguous` copies noncontiguous operands, preserving
their dtype. Its default safe math mode preserves special-value behavior;
it does not promise the same reduction order or bitwise matmul output.

The [0.32.3 custom evaluator](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/custom_kernel.cpp)
binds those inputs directly and allocates the declared output dtype. Retrieved
Git blob: `b73dd72def76fa3412ef9155749507fe5d5cb48e`. The
[Python binding](https://github.com/ml-explore/mlx/blob/v0.32.3/python/src/fast.cpp#L415)
passes the compile options through; blob:
`f9c472ab991cd156785c5323c0dea1818191e00d`.

Installed `bf16.h` defines `bfloat16_t` as Metal `bfloat`. Installed Steel
`mma.h` uses float SIMD fragments/accumulators and converts loaded elements to
the fragment type. However, its standard `BlockMMA` accepts the same storage
type T for both operands. A custom tiled kernel could convert BF16 weight
loads to float locally, then multiply float fragments, without a complete
FP32 W allocation. This is a feasible source-level hypothesis, not an existing
mixed-input matmul operator or verified fast kernel. Copying private Steel
implementation wholesale would add substantial maintenance surface.

[Apple's feature tables](https://developer.apple.com/metal/feature-sets/)
identify M4 as Apple9 and support BF16 scalar/vector values from Apple6.
They establish compatible loads, not doubled FP32 arithmetic throughput or
mixed-input matrix performance on this device.

Exact inspected installed files, relative to
`.venv/lib/python3.12/site-packages/mlx/`:

| File | SHA256 |
| --- | --- |
| `core/fast.pyi` | `e6ead453b46633174ef101388f4e6b7c02ed837fbee91db16aa3be39ea6f2d4b` |
| `include/mlx/backend/common/metal_kernel.h` | `b64ca4cebcba530d8ec1e0a0f23b57313507a37d05e3615ded3796b563cc83fc` |
| `include/mlx/backend/metal/kernels/bf16.h` | `abd87446a310b77ac530ef52a324feae5cb285d03ec9613e3a88ebb71410fdcb` |
| `include/mlx/backend/metal/kernels/steel/gemm/mma.h` | `9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7` |

The inspected production `wyat/backends/wyat/mlx.py` SHA256 is
`2d56fa59c2fc0f04b6d9ba0fa210af1755ad8b265fead8b87a63505f140c8829`.

## One projection and conditional limit

Select only the first layer's real `gate_proj`: X[M,1024], W[3072,1024],
Y[M,3072], no bias. Capture the actual FP32 activation from the unchanged
checkpoint/encoder; do not substitute random tensors for the real observation.
The prefix job has full padded width 192 and fixed prefix 25, leaving suffix
width 167: M=167 at B1, M=1336 at B8. Full-prefill shapes are separately
M=192 and M=1536. Other batches may have different padded widths.
Before execution, token inspection selects case index 7 of the fixed first-eight
cohort for scalar capture because its encoded length is exactly 192. Batch eight
uses all eight cases, right-padded to 192. The scalar selection fixes geometry;
it is not a quality or timing selection. Observe the actual first-layer gate
input after attention separately for prefix and full-prefill execution.

W occupies 12 MiB in FP32 or 6 MiB in BF16. Admit BF16 storage only if decoding
it to FP32 reproduces every original finite weight's FP32 bits, including
signed zero. Never reduce activation precision. Arithmetic remains 2*M*1024*3072
FLOPs; the ideal one-read operand account is
`4*M*(1024+3072) + weight_bytes`.

| Execution shape | M | FLOPs | FP32 operand minimum, bytes | Mixed minimum, bytes | Traffic ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix B1 | 167 | 1,050,673,152 | 15,319,040 | 9,027,584 | 1.697 |
| Prefix B8 | 1336 | 8,405,385,216 | 34,471,936 | 28,180,480 | 1.223 |
| Full B1 | 192 | 1,207,959,552 | 15,728,640 | 9,437,184 | 1.667 |
| Full B8 | 1536 | 9,663,676,416 | 37,748,736 | 31,457,280 | 1.200 |

These are logical minima, not measured DRAM traffic or speed ceilings under all
cache conditions. Cache hits, tile rereads, workspace, weight-conversion work,
FP32 compute limits and dispatch overhead remain unknown. The likely limiting
tradeoff is reduced weight traffic versus conversion/tile occupancy and already
tuned FP32 compute, especially at B8. A half-sized W does not imply 2x runtime.

## Prospective bounded assay

After explicit resource release, freeze captured activation/weight bytes,
model/source identities, exact strides/shapes, kernel source, one tile policy,
compiler/math options and receipt schema before timing. Keep three arms:

1. Maintained FP32 `X @ W.T`, the current baseline.
2. One custom tiled FP32-input/FP32-weight/FP32-output kernel.
3. The same tile/arithmetic/reduction kernel with BF16 device weight loads
   converted to float locally; FP32 products, accumulator and output.

Use public Metal float SIMD matrix operations, safe math, disjoint output
ownership and no atomic reduction or reduced-precision intermediate. First
require a direct tiny boundary exercise of tail rows/columns, strides and
weight decoding. Reject unsupported layouts rather than silently timing copies.
The two custom storage arms must be bitwise equal on admitted real inputs;
both must be finite and satisfy elementwise
`abs(candidate-reference) <= 1e-4 + 1e-4*abs(reference)` against maintained FP32
matmul. Retain maximum errors and every failing element count. These are
prospective gates: do not loosen them after observing a failure.

Build each kernel once. Record capture, conversion, compilation, first dispatch
and cold allocation separately. After one warm dispatch per arm/shape, measure
six paired blocks, one for each permutation of the three arms, with ten
synchronized dispatches per arm/block, under a 512 MiB process cache cap. This
balances each arm's order position. Restore the original cap on exit;
retain before/after allocator snapshots outside timed intervals. Hold inputs
and weights ready in every arm. Use one owned child with a 180-second parent
bound; preserve incomplete/error receipts and release the model before primitive
timing. Preserve actual captured inputs, weights and cold outputs in an ignored
`*.tensors.safetensors` file. Check returned warm and timed outputs for bitwise
reproducibility against their own numerically admitted cold output after stopping
each interval; retain checks and failures. These intervening observations are
outside dispatch intervals and can perturb their surrounding execution. The
result does not establish unobserved back-to-back performance.
The proposed primitive success gate is median speedup >=1.05 and paired
p05 >1 against maintained matmul at both prefix shapes, with all numerical
gates passing. Full shapes are separately reported controls, not pooled wins.

A primitive pass would justify only a later single-projection integration
assay. That assay must retain the existing all-case CPU-reference gate:
logit atol/rtol 1e-3, probability absolute drift <=1e-4, exact candidate ordering,
selection and rank, failure-inclusive denominators. No end-to-end gain follows
from the primitive result; its production time fraction must be measured before
an Amdahl claim. No production interface, default change or kernel has been
introduced by this proposal.

## Completed fixed32 observation, 2026-10-05

The [real receipt](../../benchmarks/analysis/results/mlx-mixed-gate.json)
and [independent reconstruction](../../benchmarks/analysis/results/mlx-mixed-gate-review.json)
retain a completed negative result on Apple M4 Pro `applegpu_g16s`, MLX 0.32.3,
MLX-LM 0.31.3, FP32 arithmetic and `MLX_ENABLE_TF32=0`. The owned parent exited
1 after 9.816 seconds without reaching its 180-second timeout; the worker
completed the assay and rejected the prospective performance gate. The parent
`completed` flag records successful exit, so its false value does not make this
scientific observation incomplete. No production path changed.

Actual first-layer gate inputs were observed through the unchanged maintained
projection call during production prefix and full-prefill execution. Prefix
construction preceded observation; the exact original gate was restored after
each capture. Scalar captures used declared cohort index 7,
`banking77-train-03357`, whose original token length is 192. Batch captures used
the complete first eight cases. These four runs retained 18 decisions over eight
unique training-source cases, with 16 correct repeated decisions. Every selection,
complete ranking, logit and probability passed the frozen CPU reference gates;
maximum logit and probability errors were 1.45436e-5 and 1.59410e-6. This supports
execution preservation within this cohort, not held-out model quality.

All four captured inputs and the selected weight were materialized as independent
host-owned FP32 arrays before deleting model, head, prefix and observer owners.
The loader and prefix construction took 1,269.865 and 33.693 ms; capture plus
release took 2,062.913 ms. Input copy/materialization took 0.127–1.004 ms per
shape. The exact 12 MiB FP32 gate weight decoded from 6 MiB BF16 storage without
any bit mismatch. Weight reconstruction and BF16 conversion took 1.010 and
1.406 ms outside dispatch timing. The baseline held ready FP32 weights; no
recurring whole-weight conversion was charged to it.

All 12 cold outputs were finite and bitwise identical across the maintained,
custom FP32 and custom BF16 arms. Both custom arms passed 19,851,264 total
candidate-versus-reference element comparisons with zero maximum absolute error.
The separate independent FP64 `math.fsum` check covered 256 scalar dot products;
its largest absolute discrepancy was 3.91652e-6, within the declared bound. All
12 warm and 720 timed returned outputs reproduced their own cold output hashes.
The input/weight and cold-output Safetensors artifacts contain 25,817,488 and
119,108,704 bytes and remain ignored; hashes are retained in both receipts.

The table reports median ten-dispatch block means for each arm and the median
and p05 of six paired maintained/custom-BF16 block ratios. These quantities have
different denominators; the paired median is not a ratio of the displayed means.

| Actual shape | Maintained, ms | Custom FP32, ms | Custom BF16, ms | Paired median | Paired p05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix B1, M=167 | 0.494 | 1.015 | 0.905 | 0.5459 | 0.5092 |
| Prefix B8, M=1336 | 2.380 | 6.137 | 5.934 | 0.4010 | 0.3801 |
| Full B1, M=192 | 0.499 | 1.020 | 1.018 | 0.4963 | 0.4718 |
| Full B8, M=1536 | 2.680 | 7.644 | 7.419 | 0.3571 | 0.3493 |

Both prefix shapes failed the original median>=1.05 and paired-p05>1 gates.
Full-prefill shapes remain separate controls. Every raw dispatch, six balanced
arm permutations, block allocator snapshot and returned-output hash is retained.
Host copies, hashes and receipt writes occurred outside the dispatch intervals
and separate successive observations. Their surrounding cache/thermal effects
are unknown; these measurements do not establish unobserved steady throughput,
statistical significance, measured DRAM traffic or complete-inference speed.

After model release, MLX reported 20 active bytes and zero cached bytes. During
primitive blocks it reported 32,112,660 active and 52,297,760 cached bytes. Its
3,972,358,024-byte peak-active counter is cumulative across capture and projection,
not a primitive-body memory peak or physical RSS measurement. The 512 MiB
inactive allocator cache cap was restored to its original 24,481,313,587 bytes;
that cap is not a total-memory limit.

The [fixed source archive](../../benchmarks/analysis/results/mlx-mixed-gate.sources-94d1d0e80f80623c.json)
retains all 43 source files used by the assay. All current bytes matched at the
independent audit, before this result was appended. The archived prospective
`mixed.md` bytes intentionally precede this completed observation. The original
source archive, model/reference identities, raw inputs, outputs and timing
receipts remain unchanged. Independent audit SHA256:
`08d8b0bd6f60b8f24a2374a1ead6a43cc32d0228d10c77b3f714379314c9e985`.

Both custom storage arms are slower, so the observations do not isolate BF16
load conversion as the cause. The source-derived [tile work account](mixed-tile.md)
supports one separately frozen 64×64×32 reuse hypothesis: halve logical operand
rereads while accepting greater register and shared-memory requirements. Current
prefix-B8 BF16 latency would need approximately 2.62× improvement to satisfy the
1.05 gate. A wider tile is an unmeasured candidate, not an expected pass; any
follow-up must preserve these fixed32 artifacts and the numerical and performance
thresholds. No Amdahl or end-to-end speedup follows from this negative primitive.
