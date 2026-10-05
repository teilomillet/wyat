# CUDA FP32 efficient-attention results

The pinned Kai reference gains modest complete-body speed under the unchanged
numerical gates: median paired speedup **1.0565x at B1** and **1.0653x at B8**.
The comparison uses forced math attention versus the causal Boolean-mask
efficient-attention adapter, with FP32 weights, disabled autocast and IEEE
global/matmul settings. It is not a comparison against the released BF16 recipe,
an online server benchmark, a universal fastest-engine claim or a 20x result.

The [prospective protocol](latency.md) and five-file declaration
`a26d5be4…` remain unchanged. All four runs use the same 48-source archive
`21b3fa50…`, the original 44 assets, CPU reference, CUDA FP32 control and corrected
first-eight screens. All twenty production sources remain unchanged.

## Numerical confirmation

Each batch mode runs both ordinary arms over all 77 cases before profiling its
first batch. Independently reconstructed confirmations retain 156 outcomes at
B1 and 170 at B8. Both arms preserve every selected choice and full ranking,
with 67 correct of 77 and no failed or missing results. The candidate's maximum
CPU-reference logit/probability differences are `1.2517e-5 / 2.4516e-6` at B1
and `1.2875e-5 / 2.7318e-6` at B8, inside the original `1e-3` absolute/relative
logit and `1e-4` probability gates.

Representative candidate profiles each contain 28 public SDPA calls, 28 private
efficient calls, 28 forward calls and 28 actual FP32 SM80 CUTLASS attention
kernels, with no fallback. Math profiles contain 28 math calls and no efficient
attention kernels. Profiles remain intrusive diagnostics. FP32 tensors and
IEEE matmul settings do not establish the attention kernel's internal arithmetic.

## Warmed paired body measurements

Each timing worker retains one warmup cohort per arm and all six alternating
pairs: 1,078 outcomes, or 2,156 across both batch modes. Every cohort preserves
all 77 choices/rankings and 67 correct predictions, with no failures or unknown
outputs. Including confirmation, the increment retains 2,482 outcomes and
14,892 scores.

| Batch | Math cohort median | Efficient cohort median | Paired median | Paired p05 | Aggregate speedup | Efficient/math cohort p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1,292.955 ms | 1,224.230 ms | 1.056514 | 1.053600 | 1.056016 | 0.946585 |
| 8 | 907.924 ms | 852.400 ms | 1.065318 | 1.061005 | 1.065806 | 0.936997 |

Both modes pass all four prospectively declared speed gates. Six-observation
percentiles are descriptive, not confidence intervals. Canonical time sums
synchronized batch-body intervals including benchmark row bookkeeping;
ownership/source scans, quality checks, I/O and inter-call gaps are outside it.
Those gaps can still affect governor/cache state. Derived active-body service
rates do not establish observed unpaced throughput, queue latency or capacity.

The repeated-row batch-residence p95 has a separate result: **18.8981 to
19.2106 ms at B1** (a 1.6536% regression), versus **104.2367 to 98.7842 ms at B8**.
The exploratory B1 shape receipt retains all 77-by-six paired measurements.
Exactly four cases regress in every pair: the four longest inputs, at 197, 199,
203 and 214 tokens. Their 24 measurements form the entire candidate upper 5%
tail. No lengths 193–196 occur. Length association and a possible tile boundary
are hypotheses; no routing threshold or causal explanation is established.

## Cost observations and next tests

The independent profile-cost receipt preserves all four raw profiles. In the
representative efficient profiles, MLP matrix-operation self-device times account
for about 42.0% at B1 and 40.9% at B8 of captured kernel-duration sums. Packed
gate/up projections target about 29.0% and 27.2%, not all MLP work. Their
numerical behavior, GEMM tiling, transient/retained memory and complete latency
remain unknown. Fractions and ideal Amdahl ceilings apply only to explicitly
assumed serial captured event sums; missing stream information, profiling
interference and representative geometry prevent a cohort-wide bound.

The next narrow tail diagnostic compares real 192- and 197-token inputs with
actual kernel grids/durations and host gaps. A packed gate/up experiment is a
separate candidate. Neither observation releases a production policy. The
reference assay covers six-choice training-source menus with unknown exposure;
it establishes neither held-out Wyat quality nor the eventual export contract.
Production enablement also needs variable-count/length, causal, unsupported
kernel and resource/cancellation evidence. [The exact Torch 2.12.1 source](cuda/policy.md) supports process-wide
precision/backend ownership, including effective-versus-inherited precision
restoration limits. A dedicated fixed-policy worker remains a candidate;
per-application locking cannot isolate unrelated Torch clients.

## Receipts and resource bounds

Raw confirmations: `cuda-latency-confirm-b1.json` (`c41d4d08…`) and
`cuda-latency-confirm-b8.json` (`51a31cdd…`). Independent reviews:
`b25958e3…` and `067e0c36…`.

Raw timing: `cuda-latency-timing-b1.json` (`21ae0cc5…`) and
`cuda-latency-timing-b8.json` (`f72f3b8c…`). Independent reviews:
`790c0e5f…` and `90cf3e32…`. Exploratory shape receipt: `0d3f003a…`;
profile-cost receipt: `25a66296…`. All live in `benchmarks/analysis/results`.

Each independent raw audit runs under a 60-second owned cap without ML
frameworks, GPU access or networking. Executed source is archived unchanged;
the shape-analysis archive retains one harmless Ruff parenthesis finding.
The maintained increment passes 840 native CPU/Metal/DMBench tests with no
skips, strict mypy for 133 sources and Ruff for 139 files. Native regression
receipt: `engine-cuda-latency-verification.json` (`67930b46…`).

The four GPU workers finish in 20.750, 11.838, 71.126 and 25.654 seconds under
their respective 180-second caps. Owned parents/transports exit and reap;
final release observations at `2026-10-05T10:52:53Z` show 128 MiB, zero GPU
utilization, no compute applications and no owned latency worker. These are
instantaneous observations, not a reservation. The original 44 assets, seven
observer files, twenty production files and prior receipts remain unchanged;
no production default or artifact was published.
