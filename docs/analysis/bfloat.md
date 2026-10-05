# Maintained BF16 MLP products: prospective protocol

This is a new experiment after [the exact FP16 storage family](precision.md)
failed its parameter eligibility guard. It retains the checkpoint's native
BF16 weights exactly and deliberately approximates selected projection inputs
and product outputs. No production precision policy changes.

Exactly three arms are fixed, in order: all 28 down projections, all 28 gate/up
projections, and all 28 complete MLPs. No layer narrowing or added arm follows a
failure. Each selected bias-free Linear must survive finite exact
FP32 → BF16 → FP32 weight equality before scoring. Nonselected weights and the
eager full-prefill FP32 control remain unchanged.

Gate/up arms share one BF16 input cast, use maintained BF16 Linear products,
and restore each output to FP32 before maintained SwiGLU. Down arms cast the
FP32 SwiGLU result to BF16, use maintained BF16 Linear, and restore its output
to FP32 before residual addition. Embeddings, attention, norms, residuals and
head remain FP32. Maintained accumulation does not undo activation or final
product rounding. Source/product dtype checks reject unintended promotion.

Fresh candidate module owners share immutable unmodified control arrays. Only
one candidate arm is resident alongside the FP32 control; references are deleted,
GC runs, the device synchronizes and inactive buffers are cleared between arms.
No prefix state is constructed or reused. Exact weight checks and synchronized
preparation time/memory are outside timed comparisons and retained separately.
The process owns a temporary 512 MiB inactive-cache cap, restored on all exits,
and both scorers use token cache zero. The fresh child starts with
`MLX_ENABLE_TF32=0` before importing MLX; whole-model compilation is disabled.

Every arm screens all 77 frozen training-source cases at batch eight and scalar,
with original candidate order, segmented rendering and the unchanged CPU FP32
reference. The initial FP32 control must itself pass both shapes. Existing gates
remain logit atol/rtol 0.001, probability atol 0.0001, identical top choices and
full ranks, finite values, complete coverage and no labeled-accuracy loss. All
three arms' raw failures and successes are retained before timing any survivor.
These cases are development evidence; prior released-model exposure is unknown.

The screen parent bounds its owned child to 180 seconds. Per-shape partial
receipts retain rows. `completed=true, passed=false` is a completed numerical
negative, distinct from an incomplete exception or timeout. Screen `passed`
means at least one of the three arms passed both shapes; individual outcomes
remain explicit. Existing receipts are never overwritten. Source/model/data/
reference identities are frozen before model loading.

Only dual-shape numerical survivors may receive timing, in the declared arm
order. Each survivor has a separate 180-second parent, checks the prior full
family's identities/rows/quality history, then independently rescreens itself
and the control on both shapes. It performs five alternating warmed whole-cohort
pairs at batch eight, then scalar. All warm and measured cycles must pass the
CPU reference gates; raw predictions, telemetry and flushed cycle journals are
retained. Acceptance requires median paired speedup ≥1.02 and paired sample p05
>1 at both shapes. No candidate is chosen by its fastest pair; all admitted
survivors are confirmed. Repeating the same cohort is performance confirmation,
not independent quality validation. A timeout, late restoration/write error or
failed numerical cycle cannot become a passing confirmation.

Prepared screen command; the coordinator must release the shared compute window
after independent source/tiny review before execution:

```sh
.venv/bin/python -m benchmarks.analysis.mlx.bfloat.run \
  --output benchmarks/analysis/results/mlx-bfloat-mlp-screen77.json
```

For a survivor, the fixed confirmation uses `--time-arm all-down` (or its
declared surviving name), `--screen` pointing to that completed family receipt,
and a fresh `--output` name. Source, test and declaration bytes stay fixed through
all confirmations. No real screen or timing has run at declaration time.
Production consideration also requires unchanged numerical behavior on the
frozen longer, variable-menu admitted native DMBench cases, plus independent
review; this short six-choice cohort alone cannot promote a precision policy.

## Observed result: all three arms rejected

The real screen completed within its bound: parent elapsed 28.515 seconds,
exit 1, no timeout or unexpected error. The exact BF16 parameter guards passed,
but every arm failed the unchanged numerical gates at both shapes. The FP32
control passed all 154 outputs, with maximum logit difference `1.502e-5` and
probability difference `3.070e-6` against the frozen CPU reference.

| Arm | Batch | Max logit difference | Max probability difference | Equal full ranks |
| --- | ---: | ---: | ---: | ---: |
| All down | 8 | 0.010673 | 0.002195 | 76 / 77 |
| All down | 1 | 0.016596 | 0.003832 | 77 / 77 |
| All gate/up | 8 | 0.018668 | 0.002426 | 74 / 77 |
| All gate/up | 1 | 0.032235 | 0.005736 | 77 / 77 |
| All MLP | 8 | 0.030442 | 0.006597 | 74 / 77 |
| All MLP | 1 | 0.019596 | 0.003965 | 71 / 77 |

All 462 candidate predictions were finite, with matching top choices and 67/77
correct per cycle. Those observations do not compensate for failed logit,
probability and full-ranking constraints. No numerical survivor entered timing,
no five-pair confirmation ran, and no production precision policy changed.
Retained screen latencies are numerical diagnostics, not a supported speed gain.
Parameter checks and candidate preparation took approximately 165.48, 229.00
and 263.43 ms respectively, outside scoring and timing comparisons.

The [raw receipt](../../benchmarks/analysis/results/mlx-bfloat-mlp-screen77.json)
retains the FP32 controls and every declared arm/shape output. Its SHA256 is
`f554d04edfd73a69c4e5fdf417a8dacd752b90a49480eba1f0ce4443029f3504`.
The [parent receipt](../../benchmarks/analysis/results/mlx-bfloat-mlp-screen77.wall.json)
distinguishes a completed numerical negative from timeout or incomplete failure.
The prospective declaration, reviewed source and tests remain in
[the immutable executed archive](../../benchmarks/analysis/results/mlx-bfloat-mlp-screen77.sources-75a93ba0c33c30b6.json),
whose full SHA256 is
`75a93ba0c33c30b6cd23a0c2bbf363d56581e7922c4d7768671d26e81d3d1d99`.

Before execution, 14 pure protocol checks and 13 tiny CPU recipe/ownership/
property checks passed independently, including PyTorch Linear/SwiGLU oracles,
variable candidate counts/lengths, scalar/batch readouts and malformed prior
screen history. Independent final audit reconstructed all 616 raw control and
candidate outcomes and their per-shape CPU gates, verified frozen source and
pinned identities, and confirmed the non-passing parent status. These results
are limited to this pinned checkpoint and development cohort, with no held-out
quality or performance claim. Parent and child processes exited and the coordinated
CPU/RAM/Metal window was released before any subsequent work.

## Current CPU compatibility

The [hosted macOS check](https://github.com/teilomillet/wyat/actions/runs/37357181906)
exposed a CPU rounding difference with MLX 0.32.3. For one nonzero input term,
eight gate/up products disagreed, all at exact BF16 halfway values: the hosted MLX CPU
matmul rounded away from zero, whereas Torch 2.14.1 and an independent
nearest-even calculation agreed. The [captured stages](../../benchmarks/real/ci/bfloat/hosted.json)
retain the tiny seeded fixture, inputs and both implementations' outputs.
The [independent evaluation](../../benchmarks/real/ci/bfloat/independent.json)
retains its source and input hash. It uses an exact one-term product and explicit
BF16 nearest-even rounding; its later SiLU calculations are numerical references.
The failing source was commit `191963ebfe1ef52fb635880a9c5a0a5c865c90f9`.

Current CPU experiments widen the BF16 operand to FP32 for the maintained
Linear operation, then explicitly cast its output to BF16 before restoring
FP32. Exact BF16 weights promote to FP32 without changing their values. This
keeps the declared rounding explicit in the CPU recipe. It is CPU
emulation, with no native BF16 throughput claim. GPU execution retains its
maintained BF16 Linear products. The original Torch oracle, tolerances and
whole-model acceptance gates remain unchanged. The recorded GPU protocol,
source archives and rejected outcomes above remain historical evidence; a new
experiment must freeze its current sources and pass the same gates.
