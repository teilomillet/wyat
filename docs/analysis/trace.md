# Observe the compiled projection

Prospective diagnostic, 2026-10-05. The Wyat engine task owns the bounded
capture/replay. This is separate from latency selection. DMBench has standing
compute availability until it reports a conflict; unrelated application load
is unknown.

Reuse the admitted real prefix batch-eight activation `[1336,1024]` and gate
weight `[3072,1024]`. Prepare the frozen32 and frozen64 kernels, decode BF16
weights locally, and validate their cold outputs against the archived exact
outputs before capture. Do not allocate a model or recapture activations.

Set `MTL_CAPTURE_ENABLED=1` and `MLX_ENABLE_TF32=0` before importing MLX. Capture
one ready GPU dispatch for each kernel into separate fresh `.gputrace` bundles.
Synchronize explicitly and stop each started capture in `finally`. Apply a
512 MiB allocator-cache limit and restore its prior value and the parent environment;
retain errors and
partial artifacts, and reap only owned processes under a hard deadline. Keep
first compilation, admission and output observation outside capture. Record the
actual deadline and ownership outcome in the receipt.

The installed `gpudebug` manual documents `--oneshot`, structured JSON output
and `profile run --exec serial --gpu-state default` on M3/A17 or later devices.
Try a separately bounded one-shot replay of each bundle, without embedding
profiling into or changing the original capture. Inspect actual compiler and
performance fields for spill bytes, occupancy and memory costs. Preserve raw
tool output and observed variation; do not invent absent fields or infer
compiler spills from timing alone. A captured bundle is not itself evidence of
spills or occupancy.

The offline `xcrun metal --help` currently fails because the Metal Toolchain
component is absent. Installed `xctrace` works, but the previous trace explicitly
disabled shader timing and had no counter set. That old shader inventory cannot
answer this diagnostic. No component download is needed to try the maintained
public MLX capture and installed replay interface.

Raw capture bundles contain model/input buffers: keep them Git-ignored and
excluded from the source distribution. Sanitized receipts describe provenance
and diagnostic observations. Replayed/profiled timing does not replace the
existing four-shape, six-order uninstrumented numerical/performance gates. The
explicit fragment-loop-unrolling candidate remains a separate hypothesis under
[unroll.md](unroll.md).
