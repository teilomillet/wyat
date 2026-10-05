# Native MLX bottlenecks and bounded hypotheses

Observed on an Apple M4 Pro, MLX 0.32.3 and MLX-LM 0.31.3, using the pinned
Decision-2.0-Kai-0.6B artifact and 77 frozen BANKING77 training-source development
requests with six candidates each. This is a runtime and numerical assay;
checkpoint exposure to these examples is unknown. No gold label enters scoring.

The production path remains FP32 causal prefill followed by endpoint gathering
and the FP32 candidate head. Its maintained upstream Transformer is reused.
The model has 28 layers, hidden width 1024, intermediate width 3072, 16 query
heads and eight KV heads with head dimension 128. Query projection width is
2048; each K/V projection width is 1024.

## Stage profile

`benchmarks/analysis/mlx/profile.py` measures complete ordinary scoring separately
from diagnostics that fence and evaluate every stage. Token cache is disabled,
batch size is eight, and original request order and pad-to-eight packing remain.
All token IDs/endpoints match the stored CPU reference. All five ordinary and
three diagnostic repetitions pass logit/probability tolerances, complete rank,
selection and accuracy gates; raw scores and stage repetitions are retained in
`benchmarks/analysis/results/mlx-profile77-b8.json`.

Ordinary complete latency has median **2799.95 ms**; its five repeats are
2799.80, 2798.25, 2814.79, 2799.95 and 2801.07 ms. Diagnostic complete median is
3308.73 ms, **18.17% higher**. Fences change dispatch and fusion opportunities,
so the stage account diagnoses candidates rather than establishing exact hot
production kernel fractions.

| Phase | Median complete-cohort stage ms | Share of summed stage medians |
|---|---:|---:|
| Attention, including projections/RoPE/SDPA | 1314.13 | 40.14% |
| MLP, including projections/SwiGLU | 1628.49 | 49.74% |
| Two layer norms | 140.89 | 4.30% |
| Two residual additions | 158.95 | 4.85% |
| Encoding | 14.45 | 0.44% |
| Embedding | 7.87 | 0.24% |
| Candidate head | 3.94 | 0.12% |
| Packing, final norm, gathering, materialization | 5.49 | 0.17% |

The fractions use **3274.22 ms**, the sum of per-phase medians, as denominator.
The difference from complete diagnostic median is 34.51 ms, or 1.04% of that
complete diagnostic median. Per-phase medians are not an exact additive
decomposition of one particular cycle. Using complete diagnostic median instead
gives attention 39.72% and MLP 49.22%; completely removing either would imply
conditional Amdahl ceilings of 1.659× and 1.969× respectively. These ceilings
assume the remaining work is unaffected and still apply only to the diagnostic
execution account.

An entire final Transformer block accounts for about 3.45% of measured stages.
Even eliminating it would give only about 1.036× under this account. Computing
only final endpoints can remove only part of that block, so it is a modest
candidate, behind projection packing across every layer. Removing encoding or
the head alone cannot approach a large end-to-end gain on this cohort.

## Work estimates and observable kernels

The runner records individual projection matrix shapes, dense FLOPs and a
single-read/write operand-size proxy. Configured projection work per layer is
`2*B*L*(H*(Q+2*KV)+Q*H+3*H*I)`. Attention dense work is
`4*B*query_heads*L²*head_dim`; an ideal triangular causal count is also retained.
Norm, activation, RoPE and softmax operations are excluded from those FLOPs.
The dense projection plus SDPA total is 13.729 TFLOP per 77-request cohort,
equivalent to a configured-work rate of 4.903 TFLOP/s at the ordinary median.
This is an arithmetic proxy, not measured instruction throughput.

The aggregate operand-size proxy is 65.271 GB. It assumes each listed matrix
operand is accessed once and does not account for cross-operation caching.
It is **not measured DRAM traffic** and must not be divided by latency to claim
attained external memory bandwidth. Likewise the entire embedding table is not
read for each lookup. Active/peak MLX allocator memory is 2.388 GB / 3.847 GB.
Hardware utilization and a compute-versus-bandwidth bottleneck remain unknown.

An actual short Xcode Metal System Trace was recorded for an owned Python
process. `mlx-metal-inventory.json` exports only that process's 34 compiled
shader names. It observes FP32 steel GEMM, FP32 causal attention at head
dimension 128, RMSNorm, RoPE and compiled SwiGLU kernels. Initial loading and
validation kernels also appear. Shader Timeline was disabled and Counter Set
was null; reliable warmed-inference markers were absent. Consequently neither
per-shader latency nor aggregate hot-inference utilization is attributed from
this capture.

The raw 67 MB `.trace` bundle and raw XML exports remain ignored locally because
the template also collects incidental host process inventory and GPU rendering
events. The reviewable TOC removes unrelated process inventory and device
name/UUID. The reviewable inventory filters by the owned process PID and records
raw-export hashes without exposing unrelated events.

## Packing experiment

`benchmarks/analysis/mlx/fusion.py` tests QKV only, gate/up only and both against
the unchanged baseline. It concatenates immutable FP32 weights once and checks
every concatenated segment equals its source. Replaced modules retain no
reference to the old projection modules; the separate baseline retains those
weights for paired timing. The upstream norms, RoPE, native GQA SDPA, output and
down projections remain. The experimental attention explicitly rejects caches.

QKV packing reduces three projections to one, and gate/up packing reduces two
to one: 84 fewer projection dispatches across 28 layers, with unchanged configured
matrix arithmetic and stored weight values. Different GEMM shapes can still
change floating-point results, so each arm first receives a whole 77-request
quality gate against the stored CPU reference. Surviving arms receive five
alternating warmed complete-cohort pairs, with raw outputs retained. A batch 8
gain alone does not justify a universal default: scalar and scaling-best shape
confirmation is required. The batch-eight assay rejected this speed hypothesis:

| Arm | Median baseline/candidate speed | Paired p05–p95 | Quality |
|---|---:|---:|---|
| QKV | 0.98381× | 0.98243–0.98578× | pass |
| Gate/up | 0.99590× | 0.99130–0.99925× | pass |
| Both | 0.97610× | 0.81798–1.02156× | pass |

Each arm preserves every one of the 385 paired selections and complete rankings,
with bit-identical paired logits. Each first screen also passes the stored CPU
reference with max logit error 1.2875e-5. The combined arm's large variation is
retained; its cause is unknown. No packing change enters production.

Whole-process active allocator memory is 2.858, 3.563 and 4.737 GB for the
successive arms, with cumulative peak 6.659 GB. These controls retain baseline
weights and experimental objects through Python reference cycles; they do not
measure eventual production packed-model footprint. The retained runner now
releases reports and collects cycles outside timed calls between arms. The exact
executed runner is archived with a matching source hash in the receipt, and a
sidecar records these later verification/lifetime edits. A scalar shape experiment
would require its own declared hypothesis; batch-eight failure provides no
universal statement about all shapes.

Tiny pytest/Hypothesis mechanics checks pass on CPU and Metal (seven tests):
H16/Q32/KV16, varying endpoint count/length/right padding, all packing arms,
scalar-versus-batch causal parity, and explicit rejection of cache use.

## Endpoint-only final block hypothesis

`benchmarks/analysis/mlx/endpoints.py` retains the first maintained Transformer
blocks and every final-layer K/V row. Only Q, attention output, output projection,
MLP and final norm are evaluated at the ordered candidate/query endpoints.
The gathered residual is kept in that same order, including duplicate positions.
Each selected query becomes a one-token batch row for native RoPE, with its
original position supplied through the supported batch-offset array. Keys use
ordinary full-sequence RoPE. An explicit boolean mask allows only key positions
at or before each query's original position; short-query `mask="causal"` would
align from the right and does not express this contract.

All tensors and arithmetic remain FP32. Ten tiny CPU tests pass: a property
test compares the complete selected model/head against full upstream outputs
with H16/Q32/KV16, changing candidate counts, endpoint order, duplicates, sequence
length and padding; another verifies array-offset RoPE exactly equals gathering
from full-sequence rotation and rejects a zero-offset control. A deterministic
uniform-GQA oracle shows that lower-right short-query causal masking gives
the wrong endpoint means. Cache restoration is checked when initial clearing,
scoring or synchronization fails. Failed/incomplete quality receipts exit nonzero,
a simulated load failure preserves an explicit incomplete receipt, and a
setup failure invalidates an older passed receipt at the reused output path. These
checks establish mechanics; the separate real Metal assay below establishes
checkpoint parity and complete latency within its frozen cohort.

The conditional whole-final-block ceiling is only about 1.036×. K/V remains
13.33% of that block's configured linear work; the potentially selected Q/O/MLP
portion is 86.67%. With seven endpoints among roughly 200 tokens, eliminating
unneeded rows saves only about 3% of total configured projection work before
new reshape/mask and dispatch overhead. This limited ceiling must justify any
extra maintenance surface.

The predeclared real assay uses every frozen 77-case request, a 512 MiB inactive
allocator-cache cap for both arms, and identical FP32 weights/tokenization.
The cap is set before loading, cleared once then, and restored on exit; screen
and warmup history is recorded, with no per-arm clearing during measured pairs.
Every arm must first pass the stored CPU numerical/rank/selection/accuracy gate.
Only then run five alternating warmed complete-cohort pairs with raw scores,
repository and installed Qwen3/RoPE source snapshots taken before loading,
and untimed per-cycle allocator/process
telemetry. A candidate needs median speedup at least 1.02× and paired p05 above 1×,
followed by scalar confirmation, before even considering a production change.
The frozen CPU reference hash is captured before loading. Every completed timed
cycle is flushed to a JSONL sidecar; the main receipt remains `passed=false`,
`completed=false` until the full assay finishes. Quality failure exits nonzero
after saving the receipt. Interruption or setup failure stays incomplete.

Both bounded real processes completed successfully: B8 in 41.26 seconds and B1
in 48.85 seconds, including loading, screens, warmups and measurements. Every
screen passed, and a pure post-run audit compared every warmup and measured
cycle independently with the frozen CPU reference. Each arm/load has 462/462
equal selections and rankings, 402/462 correct decisions (67/77 per cycle), and
finite scores within the unchanged tolerances. The endpoint arm's maximum
absolute logit/probability differences are 1.335e-5/2.490e-6 at B8 and
1.454e-5/2.996e-6 at B1. These are runtime equivalence results on training-source
development requests; checkpoint exposure is unknown.

| Load | Five alternating pair speedups | Median | Sample p05 | 2% gate |
| --- | --- | --- | --- | --- |
| B8 | 1.02298, 1.02656, 1.02375, 1.02361, 1.02953 | 1.02375× | 1.02311× | Pass |
| B1 | 1.01913, 1.01929, 1.02627, 1.01619, 1.01764 | 1.01913× | 1.01648× | Fail |

The median complete-cohort latencies are 2819.98/2750.17 ms for baseline/endpoint
at B8 and 3393.38/3333.31 ms at B1. Every observed pair is faster, consistent
with the small Amdahl ceiling, but scalar confirmation falls below the declared
2% threshold. Sample percentiles from five pairs are not confidence bounds or
proof of a universal gain. Production stays on the maintained whole-block
path; the modest result does not justify adding this extra inference path now.

Raw receipts, exact source archives, incremental cycles, allocator/process
telemetry and bounded-parent status are retained as `mlx-endpoints77-b{8,1}.*`
under `benchmarks/analysis/results/`. The parent timeout reproducer is
`mlx-endpoints-bounded.executed.py`; `mlx-endpoints-reference-audit.executed.py`
and its JSON receipt preserve the independent all-cycle CPU comparison.
Numerical `passed` and `performance_gate_passed` are separate fields so a
completed negative speed result cannot be mistaken for a failed experiment.

## Combined prefix and endpoint hypothesis

The independent scalar assays show small positive gains from immutable
25-token prefix reuse (median 1.01073×) and endpoint-only final-block work
(1.01913×). Their combination has a present measured motivation, but multiplying
these speedups would not establish a combined gain. The prospective experiment
must independently pass strict full-cohort parity and median speedup at least
1.02× with sample p05 above 1× at both B8 and B1 before any production proposal.

The originally measured `combined.py` subclassed the experimental prefix scorer's
narrow readout hook. Its retained source archives preserve that version. Prefix
validation, suffix packing, original full padding, caller
order, fresh cache ownership and whole-batch fallback stay with `PrefixScorer`.
Earlier Transformer blocks use maintained cache-aware attention masks and RoPE.
The endpoint helper retains cached prefix K/V plus every suffix K/V row, rotates
suffix K at the prefix offset, rotates selected Q at each original absolute
position, and masks full keys against those absolute positions. Only the last
block's Q/O/MLP and final norm visit selected readouts. No gold enters model calls.

Thirty-five tiny CPU tests across the combined, prefix and endpoint suites pass.
The combined checks challenge H16/Q32/KV16, GQA, changing candidate counts,
lengths, padding and order, repeated/interleaved scalar and batch calls,
immutable prefix state across cache growth, mismatched/overlapping batch fallback,
duplicate readouts and incorrect absolute RoPE or causal masks. Invalid candidate
counts retain Kai's existing rejection. Four CLI checks also preserve declared
strategy selection and nonzero failure/incomplete status. These establish
mechanics separately from the real checkpoint assay. The protocol uses the same
512 MiB allocator cap for
both arms, independent all-cycle CPU gates, five alternating warmed pairs,
raw outputs, source/reference snapshots and a 180-second parent timeout per shape.

Independent source review approved the bounded assay. Both processes completed
without timeout: B8 in 39.55 seconds and B1 in 48.99 seconds. Every full-cohort
screen passed. At each load, all 924 warm/measured rows across both arms pass
the frozen CPU gate: 924 equal selections and rankings, 804 correct decisions
(67/77 per cycle), and unchanged numerical tolerances. Maximum absolute
logit/probability differences are 1.609e-5/4.615e-6 at B8 and 1.889e-5/3.936e-6
at B1. Scores are finite; neither semantic quality improvement nor held-out
quality is established.

| Load | Five alternating pair speedups | Median | Sample p05 | 2% gate |
| --- | --- | --- | --- | --- |
| B8 | 1.10287, 1.11641, 1.10561, 1.11237, 1.10707 | 1.10707× | 1.10342× | Pass |
| B1 | 1.02556, 1.02708, 1.02265, 1.02964, 1.02004 | 1.02556× | 1.02056× | Pass |

Median complete-cohort baseline/candidate latencies are 2823.69/2545.26 ms at
B8 and 3387.18/3301.63 ms at B1. Prefix preparation costs 33.00/32.55 ms, excluded
from warm/measured calls and recorded separately; the logical frozen KV state
is 5,734,400 bytes. Views can retain additional backing allocation. These gains
are independently measured rather than products of the separate assays. Sample
percentiles describe five observed pairs, not confidence bounds.

B8 peak active allocation grows from about 3.853 GB for the baseline to 4.059 GB
for the combined arm; B1 falls from about 3.025 GB to 2.896–2.902 GB. These measured
shape-dependent peaks are separate from logical prefix bytes and inactive cache.

Both loads clear the declared gate, supporting consideration of an explicit
eager immutable prefix resource. The frozen prefix includes this cohort's fixed
banking-support context; an arbitrary deployment does not necessarily share it.
Mismatch or overlapping readouts use the intact original scorer, and no automatic
25-token cache policy is supported. Independent receipt review passed. The
validated mechanics now have an explicit opt-in production scorer, with unchanged
loader/CLI defaults. The final moved-source confirmation passes both gates:
B8 median 1.11334×/p05 1.10827× and B1 median 1.02620×/p05 1.02036×, preserving
all 924 CPU-comparison rankings and selections at each load. It has separate
`mlx-prefix-production77-b{8,1}.*` receipts; the original combined receipts remain
unchanged. [Prefix usage](prefix.md) states its fixed-resource obligations and fallback
behavior.

Raw results, all-cycle CPU comparisons, incremental measured cycles, process
telemetry and exact source archives are `mlx-combined77-b{8,1}.*` in the analysis
results directory. `mlx-combined-bounded.executed.py` preserves the exact parent
reproducer and its hash appears in each `.wall.json` receipt. Historical endpoint
and prefix receipts retain their original bytes and source snapshots.

## Mixed storage probe

A tiny CPU-only MLX graph probe uses FP32 input and losslessly converts weights
whose values are representable in BF16 to FP16 storage. Its output remains FP32,
and the exported graph places an `AsType` node after the half-weight transpose
and before `Matmul`. The tiny output is exact against its FP32 control. This
does not establish a faster FP16-input/FP32-accumulation Metal kernel: graph
conversion and potential temporary storage need investigation before a speed
claim. It also does not establish full-checkpoint quality. The retained probe,
graph and receipt are `mlx-mixed-storage.executed.py`, `.dot` and `.json` under
the analysis results directory.

## Allocator cache hypothesis

MLX distinguishes active buffers from inactive buffers retained for reuse.
`get_cache_memory()` measures the latter, while peak memory measures active
buffers. Active plus cached memory therefore matters when studying process
footprint. A cache limit controls allocator reuse; it is separate from an
application's prompt KV cache. [MLX cache API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)

The pinned 0.32.3 Metal allocator initializes both its memory guideline and
cache cap to `min(1.5*recommended_working_set, 0.95*physical_memory)`, nominally
24.481 GB on this Mac. Its reclamation threshold is
`min(0.95*recommended_working_set, memory_guideline)`, nominally 18.116 GB.
Allocation tries buffer reuse first; a cache miss can trigger reclamation.
The cap is then enforced, while freeing one buffer can temporarily overshoot it.
These are source mechanics, not evidence that retained cache caused the historical
latency outliers. [Pinned allocator source](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/allocator.cpp#L54-L61)

Installed MLX-LM 0.31.3 periodically clears allocator cache during generation and
after evaluation prefill chunks; its server/generator separately manages wired
memory. The installed source has no `set_cache_limit` calls. A bounded inactive
cache could reduce footprint or stability variation without changing math, but
the fresh comparison must record synchronized active/cache/peak snapshots per
cycle and process/system diagnostics outside timed work. Neither a changed
default nor a causal explanation for old outliers is currently supported.
`set_cache_limit` returns the previous cap; there is no corresponding cache-cap
getter in the installed Python API. A fresh experimental process must record
and restore that setting if changed.

## Reproduction and provenance

Run Metal commands with the project's approved GPU permissions. A Torch-free
editable `wyat[mlx]` environment was used at `/tmp/wyat-mlx-only`:

```sh
PYTHONPATH="$PWD" /tmp/wyat-mlx-only/bin/python benchmarks/analysis/mlx/profile.py
PYTHONPATH="$PWD" /tmp/wyat-mlx-only/bin/python benchmarks/analysis/mlx/fusion.py
```

The original profile receipt is preserved byte-for-byte; exact original runner
bytes were not archived and remain unknown. Its provenance sidecar
hashes the receipt and retained reproducer/backend sources and explicitly records
that source hashes were added afterward: trace-only and case-limit runner
branches were introduced after the original timed profile. Fresh runner outputs
include source hashes directly. Synchronization is explicit stage fences for
diagnostics, and output evaluation plus host materialization for normal scoring.
