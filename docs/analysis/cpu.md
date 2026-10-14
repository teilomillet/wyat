# FP32 CPU experiment

Investigation and completed thread sweep on 2026-10-05. Eight configured
intra-op threads gave the highest observed throughput among the six sampled
counts. All 42 warmup/measured cohorts passed the frozen FP32 quality gates.
No production setting changed. A subsequent bounded operator profile confirmed
that this build already dispatches grouped-query attention to CPU flash.

The current Torch Kai path loads local Safetensors into FP32 `Qwen3Model`, uses
`eval()`/`inference_mode()`, disables KV caching, performs one causal SDPA
prefill and applies the FP32 candidate head after gathering endpoints. It pads
to each batch's exact longest sequence, without a redundant padding mask.
There is no vocabulary projection or generation loop to eliminate. See
[the scorer](../../wyat/backends/wyat/torch.py) and
[completed CPU experiments](../torch-inference.md).

## Observed build and prior measurements

A read-only `torch.__config__.show()`/`parallel_info()` check found Torch 2.14.1
at build commit `5c4886908584029761b579af026dcfb627c84070`, Apple Accelerate
BLAS/LAPACK, OpenMP enabled, MKL disabled and MKLDNN disabled.
`torch.backends.mkldnn.is_available()` returned false even though its enabled
flag was true. Defaults were eight intra-operation threads and twelve
inter-operation threads. Enabling oneDNN flags cannot add absent compiled
kernels. The build-reported twelve hardware threads do not establish a uniform
core topology; a sandboxed `sysctl` topology query was unavailable.

The earlier [16-case screen](../../benchmarks/real/torch-causal-screen.json)
recorded these full-cohort causal-engine median times, over three repetitions:

| Intra-op count | Batch 1 | Batch 4 | Batch 8 |
| --- | --- | --- | --- |
| 1 | 2388.6 ms | 2010.0 ms | 2003.3 ms |
| 4 | 1980.6 ms | 1473.0 ms | 1385.5 ms |
| 8 | 1987.1 ms | 1343.1 ms | 1267.8 ms |

This screen suggests an effect on substantial backbone work: batch-eight
throughput at eight threads was about `1.58×` one thread, but only `1.09×` four
threads. Counts were screened in blocks rather than balanced thread comparisons;
these ratios are exploratory, not confirmed effects. This earlier screen did
not establish an optimum above eight threads. All screen ranks/selections
matched the explicit-mask path. The full 77-case reference remains the numerical
authority for the completed experiment.

CPU Inductor was already slower in all five warm pairs (`0.919×` median), and
QNNPACK INT8 was slower and failed numerical/rank gates. Mask removal gave a
simpler causal path without a meaningful CPU speedup. Repeating these unchanged
candidates would provide less information than the thread experiment.

## Completed six-count receipt

The [full receipt](../../benchmarks/analysis/results/torch-threads77.json)
retains every output and before/after diagnostic. It uses batch eight, all 77
canonical development cases, FP32/SDPA, no cache, one warmup and six balanced
measured cycles per count. Inter-op count remained twelve. Python was 3.12.11,
Torch 2.14.1 and Transformers 5.18.0. Throughput divides all 462 measured
requests at a count by their total synchronous cohort time; it is not a
singleton response-latency estimate.

| Configured intra-op count | Equivalent requests/s | Median cohort time |
| --- | --- | --- |
| 1 | 8.364 | 9.209 s |
| 2 | 10.304 | 7.461 s |
| 4 | 11.957 | 6.440 s |
| 6 | 13.035 | 5.897 s |
| 8 | 13.396 | 5.718 s |
| 12 | 13.102 | 5.874 s |

All 36 measured and six warmup cohorts passed: 3,234 outputs retained, no
failed cases, identical selections and complete rankings. Across them, maximum
absolute logit drift was `8.345e-6` and probability drift `1.786e-6` against
the frozen CPU reference. Accuracy remained 67/77 in every cohort; this assay
does not establish held-out model quality. Eight threads achieved `1.602×`
the one-thread measured throughput. Twelve did not improve on eight. Same-round
ratios support this sampled comparison, not an independently confirmed global
optimum.

The [parent wall receipt](../../benchmarks/analysis/results/torch-threads77.wall.json)
reports 287.548 seconds within the 600-second cap, exit zero and no timeout.
The [setup source archive](../../benchmarks/analysis/results/torch-threads77.sources-ee621c18dc3d4e8a.json)
contains eleven declared source files; its digest
`ee621c18dc3d4e8a99c8e7e8837676f5b8e1d32bf2964231d85cc23fade0e10f`
and each stored source hash were independently rechecked. This archives source
at setup, not dependencies or large checkpoints.

The first, excluded one-thread warmup recorded 139 major and 21,167 minor
faults. Every measured cycle had zero major faults; fault counts therefore do
not establish a paging explanation for measured outliers. OS one-minute load
ranged 1.58–3.43 across all snapshots. Own-process context switches and CPU
times are retained, with snapshot collection taking 2.6–17.5 microseconds
outside inference timing. These counters do not reveal exact Accelerate
worker topology, competing jobs, thermal state or DRAM traffic.

The descriptive USL fit gives `alpha=0.54594`, `beta=0.0047148`, throughput
`R²=0.9757`, maximum relative residual 4.38%, and a predicted peak near
9.814 configured threads. There are no confidence intervals or identified
physical mechanism. The fitted parameters do not establish serialization or
cache-coherence causes on this heterogeneous machine.

## Is nine or ten threads worth checking?

The same fitted curve predicts only `1.00251×` at nine and `1.00303×` at ten
versus **fitted** eight-thread throughput. Its ten-thread prediction is 13.197
requests/s, 1.48% below **observed** eight-thread throughput. That 0.30% fitted
increment is smaller than both the 4.38% fit residual and the eight-thread
cycle range of 5.687–5.869 seconds (3.22%). Interpolation is a hypothesis,
not evidence that ten improves performance. Testing both nine and ten adds
little information at this stage.

If a resource window is released, a narrow eight-versus-ten check can falsify
the peak suggestion: unchanged full cohort and batch eight, cache off, one
warmup per count, five adjacent pairs with alternating arm order, all raw
outputs/diagnostics retained and the existing strict quality gates. Freeze the
sources/reference before loading, hold inter-op twelve, restore prior intra-op
state, and retain a 600-second parent cap. Five pairs can detect a substantial
effect; they are not assumed to resolve a 0.30% difference. Do not change the
eight-thread reference on a fitted peak or a single fastest sample. That paired
thread check remains unperformed; the dispatch profile below is complete.

## Observed CPU grouped-query dispatch

The [typed profiler](../../benchmarks/analysis/torch/profile.py) froze the first
eight canonical cases, their numerical reference, and eleven source files before
loading. It used FP32 CPU, eight intra-op threads, inter-op twelve, batch eight,
maximum length 512 and cache zero. One warmup preceded two ordinary calls and
two CPU-only shape profiles, with stack/memory tracing disabled. The
[receipt](../../benchmarks/analysis/results/torch-profile8.json),
[incremental raw cycles](../../benchmarks/analysis/results/torch-profile8.cycles.jsonl)
and [wall receipt](../../benchmarks/analysis/results/torch-profile8.wall.json)
retain outputs, operators, shapes, self times and process observations.
The owned process completed in 6.386 seconds within its 180-second cap.

Both profiles observed exactly 28 calls to
`aten::_scaled_dot_product_flash_attention_for_cpu`, with query shape
`[8,16,192,128]` and key/value shapes `[8,8,192,128]`. Neither recorded a math
SDPA fallback or explicit K/V repetition. This directly establishes grouped
CPU flash dispatch for the sampled workload. The installed Transformers helper
uses availability flags without inspecting the tensors' device: here its MPS
flag was false, and its no-mask/head-dimension branch enabled GQA. The matching
Torch CPU selector supports FP32 and grouped heads; its source eligibility is
consistent with the observed dispatch. An explicit-repeat-K/V alternative has
no demonstrated dispatch problem to fix and was not tested.

| Operator group | Profile 1 self time | Profile 2 self time |
| --- | --- | --- |
| Backbone matrix multiplications | 441.26 ms | 442.80 ms |
| MLP gate/up/down subset | 263.72 ms | 263.38 ms |
| CPU flash SDPA | 50.20 ms | 50.41 ms |
| All recorded operator self times | 582.58 ms | 585.30 ms |

These rows use self time, avoiding nested inclusive-time double counting. The
MLP row is a subset of the backbone row, not an additional cost. Backbone
matrix operations account for about 75.7% of the recorded self-time denominator;
MLP operations account for 45.1%, and flash SDPA accounts for 8.61%. Those are
profile diagnostics, not independently measured ordinary fractions or DRAM
bandwidth. Eliminating attention alone would have only a roughly `1.09x`
conditional Amdahl ceiling under these diagnostic fractions, well below `20x`.

All forty warmup/control/profile outputs matched the frozen logits and
probabilities exactly, preserving every selection and complete ranking.
Accuracy was seven of eight for each call, with unknown Kai training exposure.
Ordinary call times were 617.60 and 638.12 ms; profiled call times were 594.60
and 596.34 ms. The median ratio of `0.948` does not establish a profiler
speedup: controls ran before profiles, and two observations cannot separate
order, warm-state or observer effects. Profile outer times including capture
setup/teardown were 614.91 and 612.47 ms. No production attention switch or
performance improvement claim follows from this diagnostic.

## Completed FP32 gate/up packing

The experiment-only [candidate](../../benchmarks/analysis/torch/packing.py)
concatenates exact gate/up weights into one maintained `Linear`, then preserves
the activation, elementwise product and down projection. It shares unchanged
backbone parameters through distinct module dictionaries; it does not mutate
the baseline or add a production inference path. Both assays used the same
77 cases, eight threads, cache zero, one warmup and five alternating pairs.

| Batch size | Paired median | Paired p05 | Throughput, baseline → packed | Median gate |
| --- | --- | --- | --- | --- |
| 8 | 1.02047x | 1.01334x | 13.439 → 13.695 requests/s | Pass |
| 1 | 1.01494x | 1.01058x | 8.248 → 8.365 requests/s | Fail |

The unchanged performance gate requires median >=1.02, p05 >1, higher
throughput and no cohort p95 regression. All 1,078 outputs per assay passed
the frozen reference gates, with every selection/ranking preserved and 67/77
correct in every cohort. Maximum logit drift was `9.537e-7` at batch eight and
`1.431e-6` at batch one. Batch eight passed locally; scalar inference missed
the original threshold. The candidate is rejected for production promotion
under the required two-shape confirmation. The roughly 2% local result is not
a portable speed guarantee.

The [batch-eight receipt](../../benchmarks/analysis/results/torch-packing77.json)
and [scalar receipt](../../benchmarks/analysis/results/torch-packing77-b1.json)
retain raw outputs, paired cycles, diagnostics, source archives and parent
wall receipts. Both processes exited zero within their caps: 83.216 and
133.715 seconds respectively. Independent review reconstructed the gates and
raw evidence. Historical receipts and executed source archives remain intact.

## Completed frozen one-projection max-autotune probe

The separate [probe](../../benchmarks/analysis/torch/autotune.py) captured the
first real batch's layer-zero gate input `[1536,1024]` and exact FP32 weight
`[3072,1024]`. It compiled only that private maintained `Linear`, with freezing,
one compiler thread and a private cache. Cold compile plus first call took
3.045 seconds; the whole owned process took 7.129 seconds under its 120-second
cap. The local projection-only additive `torch.allclose` gate is distinct
from the complete-model quality gate. All retained projection outputs were
bit-identical, but that establishes neither full-model quality nor a speed gain.

Autotune selected ATen `mm` on the Accelerate-enabled build: 3.1288 ms versus
9.9839 ms for the proposed NEON CPP alternative. Generated code called
`extern_kernels.mm`. Five alternating warm pairs gave median `0.98404x`, with
ratios spanning 0.863–1.161. No full-model max-autotune or production compile
switch follows. These observations do not identify an internal Accelerate or
AMX microkernel.

The [receipt](../../benchmarks/analysis/results/torch-autotune-projection.json)
and [dispatch sidecar](../../benchmarks/analysis/results/torch-autotune-projection.dispatch.json)
retain timings, actual selection/code excerpts, log hash and numerical archive
hash. The raw tensor archive includes model weights and stays local, excluded
from Git and source distributions. The supervisor kills and reaps its owned
process group on completion or timeout; independent testing verified cleanup
of a real child and sleeping descendant. No compiler descendants remained.

## Maintained CPU prefix screen

Read-only inspection of installed Transformers 5.18 supports a narrow
experiment with maintained `Qwen3Model` and `DynamicCache`, full final block
unchanged. Prefill the exact 25 tokens once, retain detached private FP32 K/V
snapshots and construct a fresh `DynamicCache(ddp_cache_data=..., config=...)`
per call from batch-expanded snapshots. Its `DynamicLayer.update` concatenates
into new storage, so suffix writes need not mutate shared prefix tensors.
Do not reuse a growing cache between requests. Check storage independence and
unchanged snapshots in tiny repeated/interleaved-call tests before a real run.

The current model derives RoPE positions from cache length; a seeded prefix
therefore starts suffix positions at 25. Candidate/query readouts become their
original positions minus 25. A mismatched prefix, empty suffix or any readout
inside the prefix requires whole-batch fallback. Preserve full input admission,
candidate order and exact token segmentation.

For multi-token suffixes, maintained mask construction creates an explicit
offset causal mask: key position <= suffix query position + 25. Plain rectangular
`is_causal=True` has the wrong alignment. Removing this mask is unsafe; the
current integration can additionally slice K/V for an unmasked rectangular
causal invocation. [PyTorch SDPA contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

The earlier sandbox MPS availability flag was false. With that flag, the
maintained masked-attention helper repeats K/V heads and does not pass
`enable_gqa=True`. A later read-only native Mac probe found both Torch MPS
availability and the helper's MPS flag true: with Torch >=2.13, maintained
masked CPU GQA can be admitted naturally. Freeze that native execution context
for both arms; do not patch global flags. The pinned CPU selector accepts
supported masks and grouped heads, but real-model masked suffix dispatch,
repetition cost and complete gain remain unknown.
[Maintained integration](https://raw.githubusercontent.com/huggingface/transformers/v5.18.0/src/transformers/integrations/sdpa_attention.py),
[pinned CPU selector](https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/sdp_utils_cpp.cpp)

The prepared prospective screen is first-eight real cases, FP32/eight threads,
batch eight/cache zero, unchanged full-block/head baseline and exact prefix
tokens. Freeze source/reference and log availability flags; retain one warmup
and two ordinary controls per arm, then one CPU shape profile per arm. Strict
original quality gates and actual Q16/KV8 flash head shapes admit the full
experiment; the two-control timing ratio is descriptive. Preserve mask shapes,
flash/math operators and self times.
Include fresh-cache construction in ordinary timing; record eager prefix build
time/memory separately. Only a passing screen justifies full-77 five-pair timing
and separate scalar confirmation under the original performance thresholds.
The typed candidate and screen live in `benchmarks/analysis/torch/prefix/`; tiny tests
challenge full-model parity, future-token causality, exact offset masks,
interleaved cache ownership, ragged order, whole-batch fallback and admission.
The independently reviewed command ran in the released native compute window:

```sh
python -m benchmarks.analysis.torch.prefix.run .models/Decision-2.0-Kai-0.6B \
  --output benchmarks/analysis/results/torch-prefix8.json
```

The [first-eight receipt](../../benchmarks/analysis/results/torch-prefix8.json)
retains eight cycles and 64 raw outputs. Every strict numerical, selection and
full-ranking gate passed against the frozen CPU reference; accuracy was 7/8
in every cycle. Maximum prefix logit drift was `6.080e-6` and probability drift
`7.618e-7`. The owned process exited zero in 7.945 seconds, without timeout;
the previously reviewed source bytes remained unchanged.

Both profiles observed 28 CPU flash calls and no math fallback. Full-prefill
queries were `[8,16,192,128]`; suffix queries were `[8,16,167,128]`. Both used
K/V `[8,8,192,128]`, establishing unexpanded GQA. The suffix profile retained
the explicit mask `[8,1,167,192]`. Native HF MPS availability and Torch >=2.13
flags were true without patching; eligibility alone was not used as dispatch
evidence.

Ordinary full-prefill controls took 607.30/599.84 ms and prefix controls took
555.24/545.24 ms, a descriptive median ratio `1.09692x`. Profile/ordinary ratios
were 1.000 and 1.010 respectively; one profile per arm cannot establish observer
effects. Prefix construction took 67.12 ms and retained 5,734,400 logical bytes.
Observed FP32 K/V shapes imply 352,321,536 bytes for the candidate's 28 extended
cache layers during one batch-eight call, excluding temporary copies. The
uncached baseline does not retain that stack. RSS counters are cumulative shared
process high water, not per-engine live memory or separate peak estimates.

This dispatch/quality screen admitted the separately reviewed confirmations
below. The two-control ratio remains descriptive and is not their verdict.

## Full77 CPU prefix confirmation

Maintained prefix reuse passed the independently reconstructed batch-eight and
scalar confirmations on the native M4 Pro. The
[batch-eight receipt](../../benchmarks/analysis/results/torch-prefix77-b8.json),
[scalar receipt](../../benchmarks/analysis/results/torch-prefix77-b1.json) and
[independent combined audit](../../benchmarks/analysis/results/torch-prefix77-review.json)
retain raw outcomes, all pairs, source identities and terminal wall receipts.
At this historical confirmation, the candidate remained experiment-only while
cache resource lifetime was evaluated. The subsequent comparison below retained
this ownership, and the [final-source opt-in CPU scorer](cpu/prefix.md) now has
its own independently audited confirmation. These earlier receipts remain
unchanged.

| Batch | Paired median | Paired p05 | Throughput, full → prefix | Cohort p95 ratio |
| --- | --- | --- | --- | --- |
| 8 | 1.10097x | 1.09393x | 13.316 → 14.664 requests/s | 0.90220 |
| 1 | 1.07610x | 1.07442x | 8.275 → 8.909 requests/s | 0.92872 |

Both shapes used all 77 frozen cases, FP32 CPU/eight threads, max length 512,
cache zero and the same exact 25-token prefix, with the full final block intact.
Each retained two screens, one warmup per arm and five alternating measured
pairs. Before either full77 result, the CPU confirmation explicitly strengthened
its prospective performance gate: median >=1.02, paired p05 >1, aggregate
throughput ratio >1 and no cohort p95 regression. All four passed for both
shapes. These four conditions do not rewrite the earlier MLX protocol.

All 2,156 screen/warmup/measured outputs passed the frozen CPU reference gates:
every selection and complete ranking agreed, with no failures and 67/77 correct
in every cohort. Maximum prefix logit/probability drift was `9.835e-6`/`1.543e-6`
at batch eight and `9.954e-6`/`1.560e-6` at batch one. Repetitions do not add
independent quality cases; Kai's exposure to this training-source cohort is
unknown. Five-pair quantiles describe these observations, not confidence bounds.

Prefix construction took 69.01 ms and 67.70 ms respectively, outside ordinary
inference. Every fresh per-call cache allocation/copy stayed inside measured
latency. Both runs retained the same 5,734,400-byte immutable prefix snapshots;
extended and temporary K/V allocations are additional. No per-engine physical
memory saving is established by the shared cumulative RSS counters.

Both owned processes exited zero without timeout, in 80.54 and 129.28 seconds
under separate 180-second caps. The identical 17-file source archive digest is
`d86eb841ae184bbb6413418a86c542dc668eb14a310292b9cecfa1cfc3faeb31`.
`cache.py` remained byte-identical to the passing first-eight candidate. The
confirmation admitted that complete audited first-eight history, model/data/
reference/source identities and full token/readout vectors before measurement.
Production code did not change.

The next proposal at confirmation time was a first-eight comparison of this
retained cache against a separately proved one-use owner. That resource screen
is now completed below; the earlier tiny proof established mechanics only.

## One-use CPU cache resource screen

Three fresh serial native M4 Pro process pairs supported the predeclared
resource hypothesis. The [independent audit](../../benchmarks/analysis/results/torch-prefix-memory-review.json)
reconstructs all six child receipts, journals, source archives and owned wall
caps. No production code changed, and no inference speed gain is established.

| Pair | Retained sampled RSS, MB | One-use sampled RSS, MB | Signed reduction, MB | Reduction |
| --- | --- | --- | --- | --- |
| 0 | 4421.55 | 4000.14 | 421.41 | 9.53% |
| 1 | 4422.80 | 3997.79 | 425.00 | 9.61% |
| 2 | 4428.12 | 3999.73 | 428.39 | 9.67% |

MB means 1,000,000 bytes. Each pair exceeded the prospective reduction threshold
`max(64 MiB, 5% retained sampled peak)` and had lower directly observed selected
K/V storage. These are inference-body **sampled current RSS** contrasts, not
whole-run or host peak memory measurements. The two arms shared a new private
scoring path with `use_cache=False` and kept their cache owner through the head;
only the maintained cache factory/reference lifetime differed. Tiny independent
checks also compared this path with the earlier `use_cache=True` retained scorer.

Each child used the frozen first eight cases, FP32 CPU/eight threads, batch eight,
max length 512, token cache zero, default RoPE, exact 25-token prefix and unchanged
full final block. Its screen, warmup, two ordinary controls, separate sampled
memory cycle and final shape profile retained 48 outcomes. All 288 outputs
passed the frozen CPU numerical, probability, full-ranking and selection gates,
with 7/8 correct each cycle and no failures. They represent eight unique
training-source cases; checkpoint exposure remains unknown. Every profile
observed 28 CPU flash calls, zero math calls, Q16/KV8 heads and the correct
`[8,1,167,192]` offset mask, without changing backend flags.

All 174 storage observations matched the independent allocation formulas.
The selected union of persistent prefix snapshots, current cache-owned K/V and
locally returned K/V peaked at 358,055,936 bytes retained versus 62,554,112 bytes
one-use. Aliased views count their backing storage once; returned tensors remain
counted after cache reset. This boundary excludes suffix projection inputs,
in-concatenation overlap, other activations/workspaces and allocator arenas.
Storage bytes are not RSS, and the larger RSS contrast cannot be attributed
solely to these selected tensors.

RSS sampling retained 21–22 valid interior queries per child, with conservative
maximum body gaps 49.28–50.84 ms, below the frozen 100 ms bound; no observer errors,
limits or missing inventories occurred. Brief peaks can still be missed. The
intrusive memory cycle took 1.48–1.54 times the same child's ordinary median;
query latency and all raw gaps remain recorded. Profiles followed sampling,
and post-profile current RSS was higher than the sampled body peak in every
child. Cold load, warm snapshots and cumulative high-water RSS remain distinct;
none substitutes for sampled current RSS. Ordinary timings are descriptive.

Prefix construction took 69.09–72.75 ms outside inference; each call included
fresh cache construction/copying. All six owned children exited zero in
6.76–6.90 seconds under separate 180-second caps and were reaped. Source bytes
and prior first-eight/full77 receipts remained unchanged through the runs.

The proposed next validation was separate full77 batch-eight and scalar
comparison through existing `benchmark_pair`. That confirmation is completed
below; the resource screen alone did not select a CPU cache owner.

## One-use full77 latency confirmation

One-use ownership failed its predeclared selection gates against retained prefix
reuse at both batch sizes. It preserved quality and the existing speed benefit
over full prefill. The [independent four-comparison audit](../../benchmarks/analysis/results/torch-one-use-latency-review.json)
reconstructs every raw result, source/history identity and terminal wall receipt.
Reject one-use selection; production remains unchanged.

| Comparator | Batch | Paired median | Paired p05 | Aggregate throughput ratio | Cohort p95 ratio | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Retained prefix | 8 | 0.999794x | 0.996876x | 1.000906 | 1.001281 | Reject selection |
| Retained prefix | 1 | 1.001429x | 0.998753x | 1.003504 | 0.988835 | Reject selection |
| Original full prefill | 8 | 1.109282x | 1.088292x | 1.103699 | 0.918216 | Pass original gates |
| Original full prefill | 1 | 1.073133x | 1.058034x | 1.068196 | 0.942890 | Pass original gates |

Both comparator policies were declared before any timing: retained-prefix
selection required median >=1, paired p05 >1, aggregate throughput >1 and
cohort p95 ratio <=1. Full-prefill confirmation preserved the original median
>=1.02 with the same other three conditions. Batch eight failed retained median,
p05 and tail conditions; scalar failed retained p05. These observations are near
parity, but do not pass the fixed selection policy. The independently supported
9.53–9.67% sampled RSS reduction does not compensate for a failed latency gate.
Separate full-prefill results cannot establish improvement over retained reuse.

Each fresh native M4 Pro job used all 77 pinned cases, FP32 CPU/eight threads,
max length 512, token cache zero, exact 25-token prefix and full final block.
Canonical `benchmark_pair` retained one warmup per arm and five alternating
pairs: 12 complete cycles and 924 strict outputs per job, with no separate
pre-screen. Every one of the 3,696 outputs passed frozen legacy CPU logit,
probability, complete-ranking and selection checks, with 67/77 correct each
cycle and no failures. There are 77 unique training-source cases, not 3,696
independent quality cases; checkpoint exposure is unknown. Five-pair quantiles
are observations, not confidence bounds.

The shared retained and one-use scoring paths differed only in their private
cache factory/lifetime. The full comparator used the original unwrapped
`WyatScorer`. Cold prefix construction stayed outside timing; fresh per-call
cache construction/copies, encoding, model, host values and decision processing
stayed inside. No profiler, RSS sampler or tensor-storage observer ran.

The jobs completed in 66.53, 110.55, 69.33 and 114.32 seconds under separate
180-second caps. The two retained comparisons exited one for valid completed
negative selections; the full-prefill jobs exited zero. None timed out, and all
owned groups were reaped. All 23 frozen source files, complete memory/prior77
histories and model/data/reference identities passed admission and final audit.
The canonical benchmark returns raw cycles only on completion: setup was saved
before the call and all returned rows before gate calculations; a timeout would
leave internal partial timing and quality unknown. No retry, source or threshold
change followed the negative results.

## Maintained guidance and boundaries

PyTorch's current [`set_num_threads`](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_threads.html)
API controls intra-operation CPU parallelism and recommends configuring it
before eager/JIT/autograd work. [`set_num_interop_threads`](https://docs.pytorch.org/docs/2.14/generated/torch.set_num_interop_threads.html)
is a different process-global control that may be set only once before inter-op
work. The old TorchScript threading page now contains a deprecation warning;
these explicit APIs are better current references. Library scorers should not
change process-wide counts per request. A dedicated benchmark process can
change intra-op counts between arms; keep inter-op fixed and record both values.

The installed-build [OpenMP setter](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/ParallelOpenMP.cpp)
updates OpenMP and, when compiled, MKL/QNNPACK pools. It contains no
Accelerate-specific worker-limit call. Therefore the experiment's axis is the
configured ATen intra-op count, not a claim that every BLAS operation uses
exactly that many workers. Record environment variables and
`torch.__config__.parallel_info()` per arm; do not invent an Accelerate affinity
policy. [PyTorch's environment-variable guidance](https://docs.pytorch.org/docs/2.14/threading_environment_variables.html)
describes OpenMP/MKL controls, while its
[tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
contains platform-specific affinity advice. Linux NUMA binding and Intel runtime
preload examples do not establish a Mac optimization.

On a later oneDNN-capable wheel, availability and actual dispatched operators
must be observed first. Internal BF16/TF32 computation can be selected for
FP32 tensors, so tensor dtype alone does not establish strict FP32 arithmetic.
Keep the supported IEEE computation policy and declare precision metadata.
The official speed examples are Intel SPR observations, not this Mac's ceiling.
[MKLDNN precision controls](https://docs.pytorch.org/docs/2.14/notes/mkldnn.html)

The installed-build [CPU SDPA selector](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/transformers/sdp_utils_cpp.cpp)
prioritizes CPU flash attention when dtype/shape/stride/dropout guards pass,
then math. It includes FP32 and grouped-query support. A profile must establish
which kernel this full-model invocation reaches; generic CUDA FlashAttention
advice is insufficient. Different fused kernels may change reductions.
[SDPA accuracy guidance](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

Shape compilation can specialize guards, recompile on changed shapes or fall
back to eager after cache limits. `TORCH_LOGS=recompiles,graph_breaks` can explain
a *new* concrete compilation hypothesis; dynamic shapes are not a promised
speedup. The completed warm regression is not explained merely by cold compile
cost, and increasing cache limits or adding padding would require fresh evidence.
[Recompilation guidance](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html)

## Completed bounded experiment protocol

The tested hypothesis was: on the unchanged FP32/SDPA Kai artifact, a count among
`1,2,4,6,8,12` improves complete cohort throughput over eight threads without
changing logits beyond `atol=rtol=1e-3`, probabilities beyond `1e-4`, any
selection or any complete ranking. This affects backbone-wide scheduling;
its scope exceeds head-only work without adding an inference implementation.
The sampled counts did not improve on eight; interpolation remains unknown.

Freeze all 77 cases, canonical order, batch size eight, exact padded lengths,
max length 512, no token cache, no compilation, no autocast and one loaded
scorer. Check artifact/reference/dataset hashes and exact tokens/endpoints before
timing. Hold inter-op count and environment constant. Warm every count once,
then collect six cycles per count in rotating orders so every count occupies
each position exactly once. Every cycle includes tokenization, packing, forward,
host materialization and decision interpretation; exclude setup, warmup, receipt
serialization, HTTP and admission queues. Preserve all raw rows and failures.
Intra-op changes are benchmark process state, recorded explicitly per arm.

The typed runner is `benchmarks/analysis/torch/threads.py`:

```sh
python -m benchmarks.analysis.torch.threads .models/Decision-2.0-Kai-0.6B \
  --output benchmarks/analysis/results/torch-threads77.json
```

The thin runner reuses Wyat's synchronous benchmark cycles, failure-inclusive
quality comparison, shared frozen-input checks and pure scaling fit. Focused
pytest/Hypothesis checks establish balanced orders, rejection of input drift,
thread-state restoration after load failure and timeout failure reporting;
these checks do not execute model inference.

Before model loading, the runner captures the declared source files once into
a content-addressed, exclusive-create UTF-8 source archive beside the receipt.
Its final source hashes come from that frozen snapshot, rather than re-reading
files after measurement. This preserves setup source bytes across team edits;
it does not prove external dependency identity or archive model checkpoints.
The frozen numerical-reference hash is also retained from setup.

Each warmup and measured cycle carries before/after wall and monotonic times,
own-process cumulative user/system CPU time, minor/major faults, voluntary and
involuntary context switches, and OS load averages. These lightweight reads and
receipt writes occur outside `_run`'s timed interval. Snapshot collection time
is recorded separately; unavailable OS load is explicit. Counter differences
may challenge an outlier hypothesis but do not identify its cause. OS load
does not reveal a competing task's identity or GPU utilization.
The shared helper accepts optional device-memory counters supplied by the GPU
runner's owner; it imports no GPU framework. Those counters must name active,
cached and peak bytes separately, rather than treating a cumulative peak as
current usage. The GPU probe also belongs outside the measured interval.

Use a 600-second experiment wall cap with incremental flushed cycle receipts;
a timeout retains partial evidence and cannot become a pass. A winning screen
needs a separate paired confirmation against eight threads; report variation
and failed runs instead of selecting one fastest sample. This fixed-batch sweep
is distinct from the concurrently owned MLX batch-size sweep.

After ordinary measurements, capture at most two representative warmed batches
with CPU profiler activity and shape recording, without stack or memory tracing.
Keep an unprofiled control. Look for Linear `mm/addmm`, SDPA CPU flash versus
math, copies/repeated K/V, norms and host encoding. Shape tracing can retain
tensors, create extra copies and add overhead, so profile fractions are
conditional diagnostics rather than ordinary latency evidence.
[Profiler contract](https://docs.pytorch.org/docs/2.14/profiler.html)
Do not sum nested operator inclusive times. External Accelerate worker topology
and actual DRAM traffic remain unknown from an ATen table alone.

USL fits are descriptive here: heterogeneous cores, hidden BLAS worker policies
and finite small load range limit a physical processor interpretation. Keep
unconstrained estimates, residuals and warnings; invalid or weak fits are
inconclusive. An Amdahl ceiling requires a measured affected fraction, and a
roofline bound requires justified traffic and device throughput limits.
Neither a fitted coefficient nor this measured sweep establishes a 20× CPU gain.
