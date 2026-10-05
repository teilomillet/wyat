# Implementation evidence, 2026-10-05

Wyat loads local decision models once, supports native batching and HTTP,
evaluates labeled JSONL, and exposes scalar/batch adapters to DMBench. The eventual
Wyat architecture remains open. Released Kai is a concrete reference contract.

## Newcomer paths

The [onboarding verification receipt](../benchmarks/real/onboarding-verification-complete.json)
records the separate quickstart, examples, CLI help and optional-runtime error
increment. Its checks cover regressions, typing, formatting, an offline build
and the built wheel's core example and CLI help with site packages disabled.
It binds the checked source and retained historical evidence by hash. This is
usability and regression evidence; it adds no inference speed or model-quality
claim.

The [initial regression](../benchmarks/real/onboarding-verification.json) exposed
test fixtures that required current runtime files to match an old experiment's
hashes. The fixture repair stages the exact archived files in a temporary
checkout and retains the real tamper-rejection guard. Historical experiment
execution still rejects changed live source; no old receipt or pin is updated.

The [next regression](../benchmarks/real/onboarding-verification-final.json)
passed all 1,058 tests, then exposed a duplicate `main.py` module name in the
combined type-check command. Checking the standalone example separately preserves
the established test module mapping. The completion receipt reuses that sealed
pytest result after checking source and peer identity; it runs the remaining
type, formatting and package checks separately.

## Latest completed runtime verification

The [native regression snapshot](../benchmarks/real/engine-native-projection-verification.json)
records **1,041 passing tests**, no reported skips and one upstream TestClient
deprecation warning. Strict mypy checked 160 maintained Python files; Ruff lint
and formatting passed. An independent reviewer verified the executed source
archive, all 322 frozen files and all 75 pinned historical evidence files. This
snapshot predates the newcomer documentation and CLI-help increment; it does
not attest later source edits.

The [current native backend guide](analysis/runtime/native.md) records the newer
scalar FP32 CPU/MLX comparison: 2.83972x paired median on the pinned 77-case Kai
development cohort, with all 924 retained outcomes passing numerical and full-
ranking gates. This is a backend choice on the measured host, not a new internal
MLX optimization or a held-out quality result.

The [CPU prefix packaging receipt](../benchmarks/real/package-cpu-prefix-production-verification.json)
verified all 22 runtime Python files, `py.typed` and attribution in the wheel.
The wheel contains runtime resources; the larger source distribution retains
the research tooling, tests and evidence that those tests consume. The following
sections record earlier increments against their own frozen source snapshots.

## Earlier verification: attention observer

The attention-observer engine and scientific-tool increment passed **777 pytest tests**,
including Hypothesis, native MLX CPU/Metal checks and all thirty actual DMBench
integration checks, with no skips. One upstream Starlette TestClient deprecation
warning remained. Independent focused checks and source/receipt audits also
passed. Ruff lint/formatting passed; strict mypy passed all 129 then-maintained production,
test and maintained experiment Python files. Historical executed-source archives are
preserved byte-for-byte rather than treated as maintained modules.
Optional DMBench types were checked against its real source. Wyat ships `py.typed`.
The [attention-observer regression snapshot](../benchmarks/real/engine-attention-observer-verification.json)
retains the outcomes and exact source archive for all 129 maintained files.
The earlier [733-test](../benchmarks/real/engine-attention-verification.json),
[709-test](../benchmarks/real/engine-unroll-cuda-verification.json),
[626-test](../benchmarks/real/mlx-tile-tooling-verification.json),
[568-test](../benchmarks/real/mlx-mixed-tooling-verification.json),
[517-test](../benchmarks/real/cpu-prefix-latency-verification.json),
[482-test](../benchmarks/real/cpu-prefix-memory-verification.json),
[434-test](../benchmarks/real/cpu-prefix-tooling-verification.json)
and [384-test](../benchmarks/real/engine-tooling-verification.json) snapshots
remain unchanged. Formatting also checks six preserved historical Python snapshots.

Wheel and source distribution build successfully. All 20 production Python
files in that wheel match its recorded source bytes; the typed marker and attribution
are included, with no model weights. Isolated core import/CLI help work with
site packages disabled. The earlier
[packaging receipt](../benchmarks/real/package-engine-tooling-verification.json) hashes both
distributions and every production source file. The corrected public MLX loader also completed GPU
scoring in the separate MLX-only environment with Torch absent and unimported.
Source-distribution inspection exposed raw Instruments capture inclusion despite
nested Git ignore rules. Explicit Hatch exclusions now remove trace bundles and
raw XML; distribution inspection checks their absence directly.
The projection experiment retains actual weight/input/output tensors locally.
Git and source-distribution exclusions also remove `*.tensors.safetensors`;
the rebuilt distributions contain no weights. Sanitized receipts remain included.
The preceding distribution identities and capture-exclusion checks remain in
the independently generated [earlier packaging receipt](../benchmarks/real/package-unroll-verification.json).
The observer increment has its separate
[packaging receipt](../benchmarks/real/package-attention-verification.json).
Both verify the production bytes, typed marker and exclusions; packaging does
not establish inference quality or speed. No artifact has been published.

Independent review reproduced activation/tokenizer mismatches and uncaught
corrupt-file errors. The artifact boundary now rejects non-SiLU backbones.
Tokenizer policy delegates to maintained local-only Hugging Face loading with
custom code disabled. Eighteen focused tests cover metadata, Qwen normalization,
spacing, special tokens, custom-code isolation and malformed JSON. Alternate
version-selected tokenizer files are rejected pending a consumed-file contract.
Torch validates tokenization before allocating its backbone; truncated backbone
and head Safetensors failures now become clear loader errors.

Earlier subprocess checks verified JSONL, failure-inclusive evaluation, and a
temporary localhost HTTP server against the offline tiny fixture. The full
suite retains cancellation/admission regressions: client cancellation cannot
admit new inference until the actual worker finishes.

## Real model and data

The pinned Kai release is revision `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764`,
model identity `kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee`.
The [development protocol](real-benchmark.md) fixes 77 independently annotated
BANKING77 training-source queries with deterministic six-choice menus. Prior
model exposure is unknown. This changed task is neither the official 77-way
score nor held-out Wyat quality evidence.

CPU and FP32 MLX retained every full ranking and selected label on all 77 cases;
both scored 67/77. Largest observed runtime drift was `1.29e-5` logits and
`2.46e-6` probability. Full references and repeated raw outputs remain in
[benchmarks/real](../benchmarks/real). Repeating cases does not create additional
independent labeled observations.

The earlier batch-eight CPU/MLX comparison used the public CLI, identical model/requests,
batch eight, cache zero, FP32, eight CPU threads, one warmup and five alternating
pairs. Both initial and post-review measurements are preserved. The
[batch-eight receipt](../benchmarks/real/runtime-cpu-mlx77.json) is authoritative for
that executed source snapshot. Timings exclude loading/HTTP; batch residence excludes
admission queues. Shared process high-water RSS is not per-engine memory.
On the M4 Pro, the final paired median was **2.02x** (5th–95th paired percentiles
1.85x–2.16x); useful throughput was 25.84 requests/s on MLX versus 12.89 on CPU.
The 385 repeated rankings/decisions all agreed; these are 77 unique cases.

## Rejected hypotheses

| Experiment | Observation | Result |
| --- | --- | --- |
| CPU causal-only attention | Parity; about 1.02x variable timing | Simpler logic, no meaningful speed claim |
| MLX compilation | Parity; 0.990x median | Eager default |
| CPU Inductor | Parity; 0.919x median, 30.62 s cold JIT/cohort | Experimental only |
| QNNPACK INT8 Linear | About 3x slower; substantial ranking/score drift | Rejected |
| MLX whole-backbone BF16 | Eight rankings changed; probability drift 0.0191 | Rejected |
| MLX whole-backbone FP16 | One ranking changed; probability drift 0.00241 | Rejected |
| MLX packed QKV/gate-up/both | Strict parity; medians 0.984x/0.996x/0.976x | Maintained projections retained |
| MLX exact padding/length grouping | Less configured work; paired variation crosses regression | Inconclusive; no promotion |
| Six CPU thread budgets | Eight best sampled; all 3,234 outputs pass | Eight retained; fitted peak is descriptive |
| [CPU FP32 gate/up packing](analysis/cpu.md) | Strict parity; B8 median 1.02047x, B1 1.01494x | Scalar misses original 1.02 threshold; experimental only |
| [Frozen CPU projection max-autotune](analysis/cpu.md) | Selected ATen `mm` over slower NEON CPP; warm median 0.98404x | One real projection; no complete-engine compile promotion |
| [MLX exact FP16 MLP storage](analysis/precision.md) | All 84 tensors contain changed values; all six fixed arms ineligible | Rejected before inference; exact guard preserved |
| [MLX BF16 MLP-role products](analysis/bfloat.md) | All three arms fail strict logit/probability gates at B8 and B1; top choices unchanged | Completed numerical negative; no timing or promotion |

Tiny random-model receipts remain explicitly synthetic. No 20x real-model or
upstream-service speedup is supported. Numerical gates were not weakened to
accept a candidate.

## Resource optimization

The [inactive-cache experiment](analysis/cache.md) found that a 512 MiB MLX
allocator-cache cap reduces retained active-plus-cache allocation from 9.32 GB
to 2.93 GB at batch eight, with about 0.7% lower throughput. Scalar allocation
fell from 6.82 GB to 2.93 GB with about 1.9% lower throughput. This is allocator
accounting, not per-engine RSS or a hard bound on live weights/activations.
Every observed ranking/selection passed the unchanged score/probability gates.
The explicit process-owned `--mlx-cache-limit-mib` CLI option preserves the
runtime default when omitted. Eighteen CLI tests and a
[pinned-model smoke check](../benchmarks/analysis/results/mlx-cli-cache-smoke.json)
verify its conversion, eligibility, ordering and output behavior.

The [decomposition protocol](analysis/plan.md) connects Amdahl ceilings, raw
scaling sweeps, roofline work estimates, profiler overhead and rejected fits.
The [CPU operator profile](analysis/cpu.md) independently observed flash GQA
in all 28 layers and no repeated K/V expansion. GEMMs accounted for about
75.7% of profiled self time, versus 8.61% for flash attention. All forty sampled
outputs matched the frozen reference exactly; profile fractions remain
diagnostics rather than ordinary latency measurements.
The completed [endpoint assay](analysis/mlx.md) preserved strict parity and
measured 1.024x batch-eight versus 1.019x scalar median speedup. The completed
[prefix assay](analysis/prefix.md) measured 1.078x versus 1.011x, with about
206 MB greater batch-eight peak active allocation. Both scalar confirmations
missed the predeclared 1.02 threshold, so their standalone paths remain
experimental. A separately measured combined implementation passed both gates:
1.107x batch-eight and 1.026x scalar, with all five pairs faster at each shape.
An independent audit verified raw cycles, source archives, sample percentiles
and every CPU-reference ranking/decision. Prefix build cost was about 33 ms,
excluded from warmed timing, and batch-eight peak active allocation rose about
206 MB. The gain applies to this fixed-prefix cohort and cache policy.

The final opt-in production [prefix scorer](analysis/prefix.md) shares native
batch packing and one typed endpoint primitive with the experiments. It builds
one exact immutable token prefix eagerly, requires fixed weights/device/encoder,
rejects compiled or scaled-RoPE bases, and falls back intact when any request
mismatches or overlaps its readouts. Default loading and CLI behavior stay on
full prefill. Actual `Model` and HTTP tests cover matching and fallback calls;
a Torch-free MLX tiny fixture also preserves the full reference.

Final-source five-pair confirmation measured **1.113x at batch eight** and
**1.026x scalar**, with paired sample p05 values 1.108x and 1.020x. Both original
gates pass; all CPU-reference rankings and decisions agree. The separate
[batch-eight receipt](../benchmarks/analysis/results/mlx-prefix-production77-b8.json)
and [scalar receipt](../benchmarks/analysis/results/mlx-prefix-production77-b1.json)
retain every cycle and identical source archives. Prefix setup took
32.63/33.57 ms outside warmed timing. These sample percentiles are not confidence
intervals, and the experiment does not establish online serving gains.

The separate [CPU prefix experiment](analysis/cpu.md) uses maintained Qwen3
masking, RoPE and DynamicCache with the full final block. Native Mac availability
flags permit masked CPU GQA in the installed Transformers/Torch combination;
the first-eight profiler directly confirmed flash attention and unexpanded K/V.
Its two ordinary controls remained descriptive before full confirmation.
Five alternating pairs then measured **1.101x batch-eight** and **1.076x scalar**
median speedup. Both shapes passed the prospectively strengthened four-condition
CPU policy: median >=1.02, paired p05 >1, greater aggregate throughput and no
cohort p95 regression. All 2,156 screen/warm/measured outputs preserved complete
CPU-reference rankings and selections; maximum logit/probability drift was
`9.954e-6`/`1.560e-6`. Each cycle retained 67/77 correct on the same development
cases. The [independent combined audit](../benchmarks/analysis/results/torch-prefix77-review.json)
retains receipt, journal, executed-source, input and reference identities.
Cold prefix setup took 69.01/67.70 ms outside timing; fresh per-call cache copying
remained inside. This supports a CPU production candidate for the measured
fixed-prefix cohort; CPU production code has not changed.

The maintained cache retains suffix K/V through each invocation. A separate
[one-use lifecycle proof](../benchmarks/analysis/results/torch-prefix-lifetime-feasibility.json)
passed eight tiny tests and an independent rerun, covering ordered updates,
storage lifetime, offset masks and full hidden-state parity. Its first observer
failure is retained. Releasing references supported the subsequent resource
experiment; the lifecycle proof itself established no physical RSS or speed
savings. Calculated logical tensor bytes
and shared cumulative process high-water RSS are distinct observations.

The subsequent [six-process resource screen](../benchmarks/analysis/results/torch-prefix-memory-review.json)
used three fresh serial alternating arm pairs on the frozen first eight cases.
All 288 raw outputs preserved the CPU-reference ranking and numerical gates;
each process directly confirmed masked flash GQA and complete layer inventories.
Observed selected storage capacity, including fixed prefix snapshots and current
returned K/V, fell from 358,055,936 to 62,554,112 bytes. This excludes transient
projection inputs, concatenation overlaps, activations and allocator overhead.
Separately sampled inference-body RSS fell by **421–428 MB (9.53–9.67%)**;
each pair passed the prospectively declared 64 MiB-or-5% effect screen. All
samples met the declared interior-count and conservative coverage-gap gates.
Observer cycles took 1.484–1.541x ordinary duration, and post-profile RSS could
exceed the sampled body peak. The evidence supports this specific body-sampled
process resource behavior, with eight unique development cases. It establishes
neither an unobserved whole-run peak nor a speed or held-out quality gain.

The subsequent observer-free full77 latency experiment completed four fresh
serial processes, with one warmup and five alternating measured pairs per
comparison. Its [independent raw audit](../benchmarks/analysis/results/torch-one-use-latency-review.json)
reconstructs all outcomes, source identities, walls and prospective gates.
Every returned warmup and measured outcome faced the frozen CPU
reference: all **3,696 checks** passed, with 67/77 correct per cycle. These remain
77 unique development cases. Cold prefix construction stayed outside timing;
fresh per-call cache copies remained inside. The original full-prefill scorer
and the common retained-prefix path were distinct controls.

| One-use candidate versus control | Batch | Median ratio | Paired p05 | Cohort p95 ratio | Prospective performance result |
| --- | ---: | ---: | ---: | ---: | --- |
| Retained prefix | 8 | 0.99979 | 0.99688 | 1.00128 | Fail |
| Retained prefix | 1 | 1.00143 | 0.99875 | 0.98884 | Fail |
| Original full prefill | 8 | 1.10928 | 1.08829 | 0.91822 | Pass |
| Original full prefill | 1 | 1.07313 | 1.05803 | 0.94289 | Pass |

Both comparisons require paired p05 >1, aggregate throughput >1 and no cohort
p95 regression; the median threshold is >=1 for retained-owner selection and
the unchanged >=1.02 for full-prefill improvement. The retained comparisons
failed their declared gates despite ratios close to parity. Full-prefill gains
support prefix reuse on this workload; they do not override those negatives.
No timeout occurred under the four owned 180-second caps; all groups were reaped.
The runner preserves raw returned cycles before validation and rejects missing
cycles, wrong identities and warmup-only drift. Internal partial cycles remain
unknown if canonical `benchmark_pair` never returns. One-use selection is
unsupported under the declared policy; CPU production code and defaults remain
unchanged. Future retained-prefix integration remains a separate decision.
The resource and latency gains are not multiplied or pooled.

The [bounded MLX mixed-weight projection experiment](analysis/mixed.md) completed
with real first-layer gate inputs from those eight cases. Separate actual prefix
and full-prefill captures used batch eight and the declared 192-token scalar
case; all 18 full-model decision checks preserved the CPU reference. The model
and cache owners were released before primitive timing, and weights decoded
from BF16 to their original FP32 bits exactly.

The fixed 32×32×32 custom Metal tile used float SIMD products/accumulation and
local BF16 weight decoding. Both custom storage arms matched maintained FP32
matmul bitwise across all four cold projection shapes: 19,851,264 comparison
elements, zero observed error. All 12 warm and 720 timed outputs reproduced their
admitted cold hashes. The [independent audit](../benchmarks/analysis/results/mlx-mixed-gate-review.json)
also checked 256 scalar FP64 dots, with largest absolute error `3.917e-6`.

| Projection shape | Mixed / maintained paired median speed ratio | Paired p05 | Prospective result |
| --- | ---: | ---: | --- |
| Prefix scalar, M=167 | 0.54590 | 0.50923 | Fail |
| Prefix batch eight, M=1336 | 0.40104 | 0.38013 | Fail |
| Full scalar, M=192 | 0.49630 | 0.47184 | Fail |
| Full batch eight, M=1536 | 0.35713 | 0.34930 | Fail |

Ratios are maintained time divided by mixed time; values below one mean slower.
All six order permutations and ten synchronized dispatches per arm/block remain
recorded. Outside-interval output hashing separates dispatches and can perturb
execution, so these are conditional primitive timings, not unobserved throughput.
The owned child completed in 9.816 seconds under its 180-second cap, restored the
allocator-cache cap and was reaped. Raw inputs/cold outputs remain ignored tensor
artifacts. No custom primitive was promoted and no end-to-end speed gain follows.

Both custom FP32 and BF16 storage arms were slow. The
[source-level tile account](analysis/mixed-tile.md) motivated the separately
declared [64×64 output-reuse trial](analysis/tile.md). It reused the exact
captured activations and weights, without recapturing a model. Maintained and
frozen32 cold bits reproduced their previous outputs; both new64 storage arms
also matched maintained FP32 exactly at all four shapes. All 12 warm and 720
timed outputs reproduced their respective cold hashes.

| Projection shape | Maintained / new64 paired median | Paired p05 | Frozen32 / new64 paired median |
| --- | ---: | ---: | ---: |
| Prefix scalar, M=167 | 0.18167 | 0.17653 | 0.37292 |
| Prefix batch eight, M=1336 | 0.14830 | 0.14566 | 0.36384 |
| Full scalar, M=192 | 0.19830 | 0.17211 | 0.38894 |
| Full batch eight, M=1536 | 0.14271 | 0.14020 | 0.39482 |

The unchanged maintained comparison fails at both required prefix shapes;
full-shape controls also fail. The larger tile is slower than the live frozen32
control. Lower logical read counts therefore did not improve this implementation.
The [real receipt](../benchmarks/analysis/results/mlx-tile64-gate.json) retains
every permutation and raw dispatch. The owned child was terminal after 9.465
seconds without timeout, restored the allocator-cache limit and was reaped.
The [independent CPU-only audit](../benchmarks/analysis/results/mlx-tile64-gate-review.json)
verified all 123 current trial sources, historical identities, 16 cold arrays,
732 warm/timed hashes and every paired ratio. All 29,776,896 candidate/reference
comparison elements had zero error. Its 256 scalar FP64 dot anchors passed with
maximum absolute error `3.698e-6`. The prospective declaration remains unchanged.
Production bytes and prior experiments remain unchanged. No kernel was selected.
Register spills, occupancy and shared-memory stalls remain possible explanations,
not observed causes. Output hashing has the same observer limitation as the
initial trial; no whole-engine speed claim follows. The
[next diagnostic](analysis/tile-next.md) identified public capture/compiler
interfaces and a separately testable fragment-loop unrolling hypothesis.

The [explicit fragment-loop unrolling trial](analysis/unroll.md) changed only
seven static-loop directives. All 16 cold arrays and 732 warm/timed hashes still
matched maintained FP32 exactly. Its paired medians improved over frozen64 by
1.927x scalar and 1.995x batch eight at the prefix shapes, but the required
maintained/unrolled ratios were only 0.35280x and 0.29556x (p05 0.33534x and
0.29292x). Both full-shape controls also failed. This improves the prototype and
rejects promotion against the maintained implementation. The
[raw receipt](../benchmarks/analysis/results/mlx-unroll64-gate.json) preserves
every order and dispatch; its owned worker completed in 9.819 seconds without
timeout and restored its cache policy. The
[independent audit](../benchmarks/analysis/results/mlx-unroll64-gate-review.json)
verified 131 consumed sources, all outputs and ratios, and 256 FP64 anchors.
No register-spill or whole-engine speed claim follows.

The [bounded Metal diagnostic](analysis/trace.md) captured one ready batch-eight
dispatch for each frozen32/frozen64 kernel. The
[successful capture receipt](../benchmarks/analysis/results/mlx-ready-capture2.json)
retains output hashes and raw bundle inventories. An earlier inventory check
rejected an internal buffer alias; that incomplete receipt remains intact.
The corrected guard admits only aliases resolving to contained regular files.
The offline Metal compiler was unavailable; one owned headless replay attempt
returned no profiling fields before its 120-second cap. Its client was killed
and reaped, with no matching process in the final snapshot; hidden XPC session
state remains unknown. The
[diagnostic audit](../benchmarks/analysis/results/mlx-ready-capture-review.json)
preserves this failure and verifies the original captures stayed unchanged.
Spills, occupancy and shared-memory stalls remain unmeasured.

## Coordination and limits

DMBench owns benchmark contracts, provenance, grouping and metrics; Wyat owns
inference and deployment. Both projects independently tested the single Wyat
translation helper. A [four-case native pilot](../benchmarks/real/dmbench-native-pilot.json)
ran actual scalar/batch DMBench evaluation through FP32 MLX without errors and
with identical answers. Its empty-context, verbatim-text recipe is explicit.
Single-run pilot timing supports no speed comparison.
The optional `dmbench-native-choice-v1` recipe now uses DMBench's actual role
validator and canonical serializer, preserving supplied choices and separating
instructions from state. Its [integration receipt](../benchmarks/real/dmbench-native-integration.json)
hashes peer and Wyat sources and two real peer fixtures. It establishes adapter
mechanics, not native-recipe model quality or throughput; the old pilot used
`dmbench-verbatim-choice-v1`. Both recipes retain whole-chunk preflight and
encoder limits without truncation or menu reduction.
The separately frozen [30-case native development assay](../benchmarks/real/dmbench-native-dev-assay.json)
completed scalar and batch-six evaluation with all cases retained. Six full
CLINC menus exceeded the declared 2,048-token setting; all 24 authored cases
were admitted, with identical selections and maximum probability drift
`6.27e-8`. Independent raw-output inspection also retained all 24 full rankings
and finite normalized distributions. The released artifact permits 255 choices;
the 64-choice boundary belongs to a configured test fixture. Accuracy was
4/30 including failures in both reports. That modest development observation
must not be confused with passing interoperability checks or representative
model quality. Prior exposure is unknown; no test split or reserved Wyat
confirmation cohort was used.

Earlier RTX 4070 Ti probes sampled 71–99% utilization. A
[fresh observer-timestamped snapshot](../benchmarks/analysis/results/cuda-availability-20261005T062156Z.json)
found zero utilization and 128 MiB allocated. These observations are not a
reservation. Canonical model assets were transferred to an owned private
experiment directory; existing environments remain unchanged. The
[prospective CUDA control](analysis/cuda-run.md) requires exact asset/runtime
admission and uncontended pre-allocation and per-batch observations.
The [real batch-one FP32 cohort](../benchmarks/analysis/results/cuda-fp32-b1.json)
preserved all 77 selections and full rankings, with 67/77 labeled correct and
no failures. Maximum logit drift was `1.723e-5`; probability drift was `2.821e-6`.
All 231 observed projection/head events had FP32 inputs/outputs and disabled
autocast; math SDPA was forced. The parent completed in 11.184 seconds without
timeout; peak allocated/reserved memory was 2.417/2.441 GB. These are cold,
observer-instrumented numerical controls, not speed benchmarks.
The [independent raw audit](../benchmarks/analysis/results/cuda-fp32-b1-review.json)
verified every row, all 44 assets and 33 consumed sources, package/build pins,
parent cleanup and the terminal 128 MiB/zero-utilization/no-compute-owner
snapshot. One independently reconstructed probability differed by one ULP
across hosts; the audit retains its initial exact-serialization failure and
uses a two-ULP derived-softmax check. The original `1e-4` probability gate and
all ranking/logit gates remain unchanged; the precise cause was not established.
The separately run [batch-eight FP32 control](../benchmarks/analysis/results/cuda-fp32-b8.json)
also preserved every selection and full ranking against both CPU and CUDA
batch one. Maximum CPU logit/probability drift was `1.395e-5`/`2.566e-6`;
peak allocated/reserved memory was 2.532/2.691 GB. Its parent completed in
8.580 seconds without timeout. Batch geometry passed this numerical assay;
this cold diagnostic cannot compare speed with batch one. Its
[independent raw audit](../benchmarks/analysis/results/cuda-fp32-b8-review.json)
verified all rows, ten batches, 30 dtype events and immutable control identities.

The [released CUDA BF16 recipe](../benchmarks/analysis/results/cuda-released-b1.json)
returned all 77 finite predictions with the expected observed projection/head
dtypes and unchanged top choices and 67 correct labels. Only 72/77 full ranks
matched CPU; maximum logit/probability drift was `0.077878`/`0.015278`.
The same strict gates also failed against CUDA FP32 batch one. Peak
allocated/reserved memory was 1.529/1.539 GB. Its parent was terminal in
12.593 seconds with exit one and no timeout: a completed negative quality
result, not an omitted failure. The recipe changes arithmetic and attention
selection together; this assay does not isolate a particular cause of drift.
The [independent negative-result audit](../benchmarks/analysis/results/cuda-released-b1-review.json)
confirmed the five changed ranks: `banking77-train-01221`, `-07093`, `-01726`,
`-06546` and `-08199`. All 77 case-level logit gates and 66 probability gates
failed against CPU. No BF16 batch-eight trial, retry or production precision
change followed.
The [final release snapshot](../benchmarks/analysis/results/cuda-final-release-20261005.json)
retains per-command statuses: 128 MiB allocated, zero utilization, no compute
owner and no matching owned runner after all three children exited.
CUDA performance and production promotion remain unestablished.
Production traffic, reserved confirmation, authentication/TLS and the eventual
Wyat export contract also remain outside this evidence. Necessary decisions are
in [open-question.md](../open-question.md).

Kai code is separated into `backends/kai/artifact.py`, `encoding.py`, `torch.py`
and `mlx.py`; loading lives in `loading/local.py`. Experiment runners are grouped by device/task with short names;
package initializers contain no inference logic. Historical receipt bytes stay
unchanged. The original source-hashed pilot is archived as text when its
runnable reproducer moves.

## CUDA efficient-attention experiment preparation

The [prospective first-eight screen](analysis/attention.md) uses the maintained
Transformers causal-mask path to repeat KV heads before PyTorch efficient
attention. Its FP32 tensor interface does not establish IEEE internal arithmetic;
the declared FastF32 candidate must pass the original numerical and full-ranking
gates. Twenty-four focused CPU/fake checks cover causal equivalence, mask
construction, context restoration, source admission, partial failures and final
receipt failures. Independent review passed before GPU release. The full
733-test preparation regression preserves all twenty production sources.
Native execution and actual dispatch require separately recorded observations;
local tests alone support no complete inference speed claim.

The [initial real batch-one screen](../benchmarks/analysis/results/cuda-attention-b1.json)
retains eighteen ordinary/profile outcomes on eight unique development cases.
Both ordinary arms preserve all eight choices/full rankings and seven correct
labels. The efficient arm's largest CPU-reference logit/probability differences
are `1.2040e-5`/`1.3444e-6`. An
[independent raw audit](../benchmarks/analysis/results/cuda-attention-b1-review.json)
verified these unchanged gates and individual terminal release queries.
The original dispatch flag remains false and the 9.63-second parent exits 1:
the public observer did not expose the internal operator schema expected by
its gate. Separate profiler observations contain twenty-eight private efficient
operators and twenty-eight FP32 CUTLASS attention kernels. Additive-bias strides
remain unobserved. These diagnostic observations establish neither ordinary
latency nor full77/held-out quality, and do not override the original failed flag.
A separately declared observation correction will test that boundary directly.

The observer correction passes sixty-eight focused tests and the complete
777-test native regression. Independent AST comparison preserves all fourteen
original non-observer test functions. New checks challenge missing private/kernel
observations, fallback operators, extra FP16 kernels, positional/keyword flags,
Boolean/integer aliases and explicit/nonfinite scales. Source inspection and a
local frozen-HF CPU call establish that Qwen supplies the mathematical scale
`128**-0.5` explicitly; this local Torch version does not establish the remote
binary's dispatch. The corrected diagnostic requires both public metadata and
private profiler/kernel evidence. The
[independent preparation audit](../benchmarks/analysis/results/cuda-attention-observer-local-review.json)
checks all 129 current sources, the three approved Python differences, all
twenty production files, original histories and the seven-file v2 declaration.
Its native rerun remains separately gated.

The [corrected batch-one screen](../benchmarks/analysis/results/cuda-attention-observer-b1.json)
passes quality and dispatch with the separate v2 source identity; its
[independent raw audit](../benchmarks/analysis/results/cuda-attention-observer-b1-review.json)
rebuilds all eighteen outcomes and verifies both observation boundaries.
The remote public call records the explicit scale `0.08838834764831845`.
Twenty-eight private efficient/forward operators and twenty-eight FP32 SM80
CUTLASS kernels execute without observed fallback. The selector cannot represent
that explicit scale; converted-bias strides and precise internal arithmetic stay
unknown. The parent exits zero in 9.58 seconds under its 180-second cap, and
individually attributed queries verify idle GPU and no owned runner. Release
observation UTC was not captured in this sidecar and remains unknown. The
initial negative flags and original archived bytes are unchanged. These repeats
cover the same eight development cases, not additional independent quality data;
loading, occupancy queries and intrusive profiling preclude a speed claim.

The [corrected batch-eight screen](../benchmarks/analysis/results/cuda-attention-observer-b8.json)
and its [independent raw audit](../benchmarks/analysis/results/cuda-attention-observer-b8-review.json)
retain all thirty-two ordinary/profile outcomes over the same eight cases, with
zero failures and unchanged choices/full rankings. Both ordinary arms score
seven correct. Largest candidate CPU-reference logit/probability differences
are `1.1861e-5`/`1.0655e-6`; CUDA-control differences are `9.8944e-6`/`1.3008e-6`.
Public Q/K/V are `[8,16,192,128]` FP32; the Boolean mask is `[1,1,192,192]`.
Twenty-eight private efficient/forward events and twenty-eight FP32 attention
kernels pass the corrected gate without observed fallback. The 9.13-second
parent exits zero; timestamped individual release queries at
`2026-10-05T08:33:01Z` confirm idle GPU and no owned runner. All current sources
and original controls remain exact. Neither job accepts a performance result
or changes production. Full77 confirmation and warmed alternating comparisons
remain necessary before proposing this candidate for deployment.
