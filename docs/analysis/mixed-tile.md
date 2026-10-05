# Fixed projection tile: source-level work account

Read-only audit, 2026-10-05. The initial custom tile remains frozen at
32×32×32, 128 threads. No device execution, model load or timing occurred in
this audit. These counts describe issued logical work, not measured DRAM
traffic, latency or an observed baseline kernel dispatch.

## Maintained comparison

The existing production receipt records Apple M4 Pro,
`applegpu_g16s`, MLX 0.32.3. In pinned
[matmul.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp),
ordinary Steel begins with BM=64, BN=64, BK=16, WM=2, WN=2. The
[architecture selection](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)
retains these settings for the medium `s` device. `X @ W.T` selects the
nontransposed/transposed layout, with ready FP32 weights.

The [routing guard](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)
requires NAX availability and either enabled TF32 or a non-FP32 input. The
assay sets `MLX_ENABLE_TF32=0` before import. Independently,
[device.cpp](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)
requires generation ≥17 for the `s` architecture, excluding recorded g16s.
SIMD split-K also requires K≥max(M,N), false here because 1024<3072.
The source therefore predicts regular 64×64×16 Steel for these inputs.
An actual dispatch trace would be stronger evidence than this inference.

Installed header identities:

| Header | SHA256 |
| --- | --- |
| `steel/gemm/gemm.h` | `c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0` |
| `steel/gemm/mma.h` | `9b37709de40c3acaa5411875e52c8d26e6de093261741f671fbdab9c39a852b7` |
| `steel/gemm/loader.h` | `c1b82153670b1e18371a6e88fdbcf440af692957775c2f837700315ab8a5112e` |

## Logical rereads and padded computation

For BM×BN output tiles, N=3072, K=1024 and exact tail admission, define
`gm=ceil(M/BM)`, `gn=ceil(N/BN)`. Valid operand-load bytes are
`4*M*K*gn` for X and `weight_itemsize*N*K*gm` for W; output stores are
`4*M*N`. This counts tile rereads, excluding cache effects, shared-memory
traffic, instruction overhead and allocator behavior. It differs from the
one-read minimum in [mixed.md](mixed.md).

| Shape | M | Custom 32 groups | Predicted Steel 64 groups | Custom BF16 X reads, bytes | Predicted Steel FP32 X reads, bytes | W reads, bytes, both paths |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefix B1 | 167 | 576 | 144 | 65,667,072 | 32,833,536 | 37,748,736 |
| Prefix B8 | 1336 | 4032 | 1008 | 525,336,576 | 262,668,288 | 264,241,152 |
| Full B1 | 192 | 576 | 144 | 75,497,472 | 37,748,736 | 37,748,736 |
| Full B8 | 1536 | 4608 | 1152 | 603,979,776 | 301,989,888 | 301,989,888 |

Our smaller output tile doubles both operand rereads. BF16 storage halves W
bytes, canceling only W's extra rereads. X remains FP32 and doubles. The
custom FP32 control has the same X rereads and twice the custom BF16 W bytes.
This gives the three-arm experiment a concrete way to separate storage
effects from tile-policy costs; none establishes physical traffic without
measurement.

Both tile policies pad prefix B1 to 192 rows and prefix B8 to 1344 rows.
Their mathematical padded FLOPs therefore match: 1,207,959,552 and
8,455,716,864. Full shapes need no row padding. Custom 32 uses four float8×8
accumulators per SIMD group; predicted Steel 64 uses sixteen. The total
number of 8×8×8 matrix accumulate operations is consequently equal for
each shape, despite four times as many custom threadgroups.

## One conditional follow-up

If the real screen admits all numerical gates but shows a slow primitive,
test output-tile reuse as one separately frozen hypothesis: 64×64×32,
128 threads, the same float arithmetic and BF16 local decoding. Keeping
M/N tile size equal to predicted Steel would halve our logical operand
rereads. K=1024 would use 32 reduction iterations rather than Steel's 64.
This does not predict a speedup: sixteen accumulators per SIMD group raise
register pressure, and keeping a separate float output staging tile would
raise shared storage from 12 KiB to 32 KiB. Occupancy, bank conflicts and
cache reuse remain unknown. A failed numerical screen requires numerical
diagnosis before any timing-oriented tile change.

The current custom tile executes 64 reduction-loop threadgroup barriers plus
one output barrier per group. Installed ordinary Steel executes 128
reduction-loop barriers for aligned K, additional synchronization and
direct fragment-to-device output stores. Barrier counts alone do not determine
runtime; the initial assay must finish before this follow-up is selected.
