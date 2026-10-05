# Diagnose the 64-tile regression before selecting another optimization

Research note, 2026-10-05. No Metal compiler, GPU dispatch, capture or replay
ran during this investigation. Existing kernels, protocols and historical
receipts remain unchanged.

## Observation and competing explanations

The [real 64-tile receipt](../../benchmarks/analysis/results/mlx-tile64-gate.json)
preserves numerical agreement but fails performance selection. Median stock
over 64-tile ratios are 0.181669 at prefix B1 and 0.148303 at prefix B8.
The corresponding 32-tile over 64-tile ratios are 0.372923 and 0.363842.
Output reuse therefore did not compensate for other costs in this implementation.

The source changed several resource demands together: four accumulators per
SIMD group became sixteen; shared storage increased from 12 KiB to 32 KiB;
shared W scatter stride increased from 32 to 64; fewer threadgroups perform
more work each. Both bodies use constant-trip fragment loops with
source-variable array indices and no explicit unroll annotation. The compiler
may already turn these into static accesses. Timing does not establish spills,
occupancy loss, bank conflicts or the executed instruction sequence.

## Public compiler boundary and one source perturbation

Pinned MLX 0.32.3's
[Python binding](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/fast.cpp)
accepts only `math_mode` in `compile_options`; unknown keys raise an error.
The [custom evaluator](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/custom_kernel.cpp)
gets a pipeline through Metal and checks its maximum threadgroup size.
Neither Python surface exposes register counts, spill bytes, an unroll option
or Apple's library optimization level. The underlying
[MTLCompileOptions](https://developer.apple.com/documentation/metal/mtlcompileoptions)
supports optimizer settings, but that does not make them available through MLX.

Installed maintained Steel uses `_Pragma("clang loop unroll(full)")` in
`.venv/lib/python3.12/site-packages/mlx/include/mlx/backend/metal/kernels/steel/defines.h`,
line 6, SHA256
`b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a`.
Its `steel/gemm/mma.h` applies this to fragment operations. Apple's
[Metal Compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/)
connects dynamic stack-array indexing with possible spills and recommends
compile-time indices. This supports a hypothesis, not a diagnosis of our kernel.

If diagnostics justify a perturbation, explicitly request full unrolling only
for the constant fragment initialization, inner K8 iteration, fragment load,
row/column accumulate and fragment store loops. Keep outer K32 traversal,
tile size, shared layout, safe math, dtype, tails and arithmetic order fixed.
[Clang documents](https://clang.llvm.org/docs/LanguageExtensions.html#loop-unrolling)
the pragma as an optimization request, so successful compilation would not
prove static indices or absence of spills. Increased live values or code size
could make it slower. No annotation or generated expansion is implemented here.

## Minimal decisive measurement

Full Xcode is selected locally. Read-only path discovery resolved `metal` and
`xctrace`; their execution and profiler compatibility are untested. The
installed `mlx/core/metal.pyi` exposes `start_capture` and `stop_capture`;
SHA256 `aecdeb33b024ad2096af9e8e584c679bbdd05d375770418351cbea410353b2b9`.

Within a coordinated resource window, reuse the frozen real input
bytes and prepare both existing kernels before capture. Set
`MTL_CAPTURE_ENABLED=1` and `MLX_ENABLE_TF32=0` before importing MLX. Use public
`mx.metal.start_capture` with a fresh `.gputrace` path, evaluate and synchronize
one ready dispatch from each policy, then stop capture in `finally`. Keep
model loading, layout probes, first compilation and capture outside benchmark
timing. Use an owned bounded process; preserve any incomplete trace/error and
release its resources.

[MLX's debugger workflow](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)
supports opening and replaying the trace in Xcode. It describes
`MLX_METAL_DEBUG` builds for source recording and labels; the installed build's
source visibility is unknown. A capture file alone is not profiling evidence.
Replay with profiling enabled and inspect each pipeline's compiler statistics,
including reported spill bytes and threadgroup memory, plus measured occupancy.
[Apple's profiling workflow](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics)
distinguishes captured commands from collected profiling metrics.

To separate explanations, inspect source-correlated memory costs where available.
Spill bytes and stack/register traffic would support the array-index hypothesis;
expensive shared staging with little reported spilling would favor a shared-layout
perturbation. Low occupancy alone is insufficient. Apple's
[family 9 tools](https://developer.apple.com/videos/play/tech-talks/111374/)
expose occupancy-management, L1 eviction, residency and bandwidth counters for
this investigation. These counters can reflect several memory owners, so retain
their breakdown and uncertainty rather than labeling all L1 traffic as spills.
Availability, replay permissions, metric export and a usable source view remain
unverified; missing observations stay unknown.

Any subsequent unroll or transpose assay needs its own prospective source and
protocol freeze. Preserve the original 32 and 64 paths as controls, repeat
bitwise storage/control checks and the existing elementwise numerical gate,
and keep the same paired performance thresholds and failure denominators.
Profiled timings must not replace the uninstrumented selection measurements.
