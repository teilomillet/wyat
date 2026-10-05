# Kai engine optimization protocol

The target is complete, synchronous decision inference on the pinned Kai model,
not a decoder benchmark. The baseline is current FP32 MLX eager inference,
with unchanged choice rendering, candidate order and relative-softmax semantics.
CPU and CUDA observations are separate runtime baselines. No 20x target is
claimed achieved merely by switching hardware.

Use the existing 77-case BANKING77 training-source development cohort, preserved
raw predictions and model identity. Its six-choice menus are a runtime assay;
prior model exposure is unknown. Reserved confirmation stays closed.

## Decomposition and decision rules

Measure request validation/serialization/tokenization, batch packing, embeddings,
28 attention/MLP blocks, final normalization/gather, candidate head, host
materialization and decision interpretation. Loading is measured separately.
Synchronizing intermediate stages changes execution/fusion: stage sums must be
compared with ordinary whole inference, with diagnostic overhead reported.

Amdahl's law bounds a proposed stage improvement:

    whole speedup = 1 / ((1 - fraction) + fraction / stage_speedup)

Even eliminating a 5% stage caps whole speedup at 1.053x. Eliminating one of 28
uniform-cost layers caps it at 1.037x; uniform cost is an assumption to test.
Optimize dominant measured work before introducing state or specialized kernels.
A 20x whole-engine gain from one stage requires that stage to exceed 95% of
ordinary cost at finite acceleration; exactly 95% permits 20x only if its cost
vanishes. A conditional device-work fraction cannot establish that ordinary
whole-engine fraction.

Roofline diagnostics use explicit shapes and conservative FLOP/byte counts.
Effective rates are measurements; a hardware peak is not invented. Count
padding work and memory traffic separately from useful request work. A lower
bound on bytes is not a measurement of actual device bandwidth.

The user's [Universal Scaling Law collection](https://teilo.xyz/collections/universal-scaling-law/)
and [Gunther's primary paper](https://arxiv.org/abs/0808.1431) inform repeated
six-level sweeps. Retain raw throughput, residuals and invalid/weak fits. Batch
size changes work granularity and padding, so fitted coefficients cannot by
themselves establish cache coherence or processor contention. CPU thread counts
also span heterogeneous performance/efficiency cores on this M4 Pro.

Little's law applies only to a stated steady-state admission/queue system;
offline ready-batch residence times cannot establish online queue latency.

## Candidate queue

These are hypotheses, not promised wins. Profiling and scaling order the queue.

| Candidate | Falsifier or acceptance evidence |
| --- | --- |
| Length grouping or exact padding widths | Reduced padded work and whole-cohort latency; original output order and all quality gates preserved |
| Fused QKV and gate/up projections | Fewer dispatches or better GEMM utilization; no duplicate retained weights or whole-latency regression |
| Lossless narrow weight storage with FP32 computation | Exact stored values, observed kernel dtype/traffic, numerical parity and a measured complete gain |
| Selective precision instead of whole-backbone casts | Independently declared strict gates pass on all 77 cases; no tolerance relaxation after results |
| Endpoint-only final block work | Causal equivalence, actual last-block fraction and enough gain to justify maintenance |
| Shared-prefix prefill | Identical tokens/positions/readouts, explicit cache ownership and resource limits, benefit at measured prefix reuse |
| Maintained vLLM/SGLang mechanisms | Actual candidate endpoint/head contract support; installation/device cost justified by measured benefit |
| Targeted Metal kernels | A dominant primitive is identified; standard kernel oracle and whole-model quality/latency both pass |

Before production edits, retain at least five alternating warmed comparison
pairs with the same artifact and input hash. Every selection and full ranking
must agree, no labeled-accuracy loss or failures are allowed, logits use the
existing 1e-3 absolute/relative gate and probability differences stay <=1e-4.
Failures and rejected ideas remain in receipts. Small numerical fixtures and
Hypothesis challenge connected boundaries; independent review verifies the
result and scope. Typed functions and short role-based files remain required.

Full Xcode/Metal System Trace is available when run with its required cache/device
access. Initial RTX 4070 Ti probes found active compute; no workload was run
against it. A later idle observation opened the separately declared
[CUDA numerical control](cuda-run.md), with asset admission, per-batch ownership
checks and owned process caps. An idle snapshot does not reserve the device.

## Observed decisions

The [stage profile](mlx.md) attributes about 49.7% of fenced stage time to MLPs
and 40.1% to attention. Fencing adds 18.2% diagnostic overhead, so these are
conditional fractions, not ordinary request-latency measurements. Eliminating
the entire final block would cap the conditional gain near 1.036x. The prepared
endpoint experiment tested whether its smaller saving survived dispatch overhead.
An independent read of the frozen token vectors found a common 25-token prefix
across all 77 cases: 1,925 of 13,679 token rows, or 14.07%. This is a configured
linear-work opportunity, not a latency saving. Maintained prefix-cache reuse is
tested with maintained masks, RoPE offsets and fresh per-call state ownership.
The [storage audit](storage.md) separately found no maintained mixed
FP32-activation/BF16-weight direct-load path; halved weight storage alone does
not establish halved DRAM traffic or faster inference.

| Completed assay | Evidence and decision |
| --- | --- |
| [Six CPU thread budgets](cpu.md) | Eight best among sampled counts; all 3,234 outputs pass. Descriptive USL peak near ten predicts only 0.3% over fitted eight, below residuals and observed variation. Keep eight. |
| [CPU operator profile](cpu.md) | All 28 layers already dispatch CPU flash GQA. GEMMs account for about 75.7% of profiled self time; attention accounts for 8.61%. These diagnostic fractions prioritize matrix work without establishing ordinary latency fractions. |
| [CPU FP32 gate/up packing](cpu.md) | Strict quality gates pass at both shapes. Batch-eight median 1.02047x passes locally; scalar 1.01494x misses the original 1.02 threshold. Reject production promotion after two-shape confirmation. |
| [Frozen CPU projection max-autotune](cpu.md) | Exact local projection outputs; ATen mm selected over the slower CPP candidate. Warm paired median 0.98404x, not an improvement. No full-model compile proposal. |
| [CPU exact prefix reuse](cpu.md) | Earlier experimental snapshot: native M4 Pro full77 medians 1.10097x at batch eight and 1.07610x scalar. All 2,156 outputs and four prospectively declared CPU performance gates pass; independent audit agrees. Cold prefix build about 68–69 ms, per-call copying included. Subsequent resource selection retained this ownership; final-source production confirmation follows below. |
| [One-use CPU cache resource screen](cpu.md) | Three fresh native process pairs reduce sampled inference-body RSS by 421–428 MB (9.53–9.67%), passing the frozen resource thresholds. All 288 outputs on eight unique cases and masked flash GQA pass. Selected K/V storage 358.06→62.55 MB; intrusive observer cost 1.48–1.54x. Full77/scalar selection results follow below; no CPU production or speed claim. |
| [One-use CPU full77 latency](cpu.md) | All 3,696 strict outputs pass. Relative to retained prefix, medians 0.99979x/1.00143x fail selection at batch eight/scalar (p05 fails both); reject one-use ownership. Relative to full prefill, 1.10928x/1.07313x pass original gates. Sampled memory gain does not override the latency rejection; production unchanged. |
| [Final opt-in CPU prefix scorer](cpu/prefix.md) | Final-source medians 1.10038x at B8 and 1.07260x at B1; all four original performance gates pass at both shapes. Independent audits preserve all 2,204 strict outcomes; one B1 pair regresses to 0.99801x and remains recorded. Explicit retained prefix, full final block and whole-batch fallback; default loading unchanged. Connected native CPU/Metal/DMBench regression passes 971 tests. |
| [CPU selected final readouts](cpu/readout/confirm.md) | Independent scalar full77 audit preserves all 1,078 strict outcomes and 67/77 correct per cohort. Retained-relative median 1.00853x and p05 0.98916x fail the original selection gates; two of five pairs regress. Reject promotion and skip B8 confirmation. Production keeps the complete final block; scalar timing is unprofiled. |
| [Native current CPU versus MLX](runtime/native.md) | All 924 measured/warmup outcomes preserve gold-reference and cross-backend numerical gates, full ranks and 67/77 correct per cohort. Full-stock MLX GPU versus eight-thread full-stock Torch CPU passes four original performance gates: median 2.83972x, p05 2.79313x, aggregate 2.82992x, p95 ratio 0.349892. Existing explicit backend selection is supported on this host/artifact; this is not an increment over prior MLX or a server/held-out claim. |
| [CPU packed projection primitive](cpu/projection.md) | All 96 calls on four authentic layer-zero operands preserve exact FP32 output hashes. KV medians 1.01678–1.04164x and QKV medians 1.03274–1.07154x miss the predeclared 1.12x triage filter; scalar-192 KV and B8 QKV tails regress. Both variants ineligible for this follow-up screen; no full-model candidate or promotion. Initial supervisor-sidecar failure and corrected auditor singleton-stride assumption remain preserved separately. |
| [Six MLX batch sizes](scaling.md) | All strict gates pass; aggregate USL parameters are invalid. Preserve the stalls and reject a physical optimum claim. |
| [Packed QKV/gate-up](mlx.md) | All rankings pass; paired medians 0.984x, 0.996x and 0.976x. Retain maintained projections. |
| [Exact padding/length grouping](padding.md) | Smaller estimated padded work; paired variation crosses regressions. No production promotion. |
| [512 MiB inactive cache cap](cache.md) | Batch-eight retained active-plus-cache allocation 9.32 to 2.93 GB, roughly 0.7% throughput cost. Expose an explicit process-owned CLI option; default policy unchanged. |
| [Endpoint-only final block](mlx.md) | Batch-eight median 1.024x, scalar 1.019x; strict parity throughout. Scalar misses the declared 1.02 threshold, so keep experimental. |
| [Exact prefix reuse](prefix.md) | Batch-eight median 1.078x, scalar 1.011x; strict parity throughout. Batch-eight peak active memory rises about 206 MB. Scalar misses the same promotion threshold. |
| [Combined prefix and endpoint](mlx.md) | Independently measured batch-eight median 1.107x and scalar 1.026x; all five pairs faster at both shapes, with strict parity. Both original gates pass on this fixed-prefix cohort. |
| [Final opt-in production prefix](prefix.md) | Moved-source confirmation measured 1.113x batch-eight and 1.026x scalar; both original gates pass. Default loading stays on full prefill, with eager explicit prefix construction and whole-batch fallback. |
| [Selective MLP FP16 storage eligibility](precision.md) | All 84 selected tensors fail exact round-trip storage: 43,137 changed values, including 224 rounded to zero. All six declared arms rejected before model allocation or inference; keep FP32 production. |
| [Maintained BF16 MLP products](bfloat.md) | All three fixed role arms fail strict numerical gates at batch eight and scalar despite exact parameters, finite predictions and unchanged top choices. Full ranks and logit/probability drift reject every arm; no timing or production change. |
| [Direct BF16 loads with FP32 projection arithmetic](mixed.md) | One real gate projection preserves exact cold outputs and all 744 output hashes; independent FP64 checks pass. The fixed 32×32 tile is slower at all four shapes (prefix medians 0.546x scalar, 0.401x batch eight). Reject promotion. The [tile work account](mixed-tile.md) supports a separately frozen 64×64 reuse test, without predicting success. |
| [64×64 output reuse](tile.md) | Exact previously captured activations/weights and all cold/warm/timed hashes pass. Maintained/new64 prefix medians 0.18167x scalar and 0.14830x batch eight fail unchanged gates; frozen32/new64 medians 0.37292x/0.36384x also show regression. Reject promotion. Logical read reduction did not translate to speed; register/occupancy/shared-memory causes remain unknown. |
| [Explicit fragment-loop unrolling](unroll.md) | All 748 outputs preserve exact maintained FP32 bits. Prefix medians improve 1.927x/1.995x over frozen64, but maintained/unrolled medians 0.35280x/0.29556x fail unchanged gates. Both full controls fail; reject promotion. No spill diagnosis or whole-engine gain follows. |
| [Ready Metal capture and bounded replay](trace.md) | Two dispatch captures preserve cold bits and raw inventories. Offline compiler unavailable; one 120-second headless replay yielded no profiling fields. Client killed/reaped; hidden XPC state unknown. Register/occupancy hypotheses remain unmeasured. |
| [CUDA FP32 numerical control](cuda-run.md) | Batch one passes all 77 strict CPU-reference selections/ranks/logit/probability gates with 67 correct and no failures. Observed FP32/autocast-off dtypes and forced math SDPA; independent raw audit verifies assets, source and terminal cleanup. Peak reserved memory 2.441 GB. Cold observer-instrumented cohort supports no speed or promotion claim. |
| [CUDA batch geometry and released precision](cuda-run.md) | Batch-eight FP32 passes all 77 strict gates against CPU and CUDA batch one; peak reserved memory 2.691 GB. Released BF16 batch one retains all top choices but changes five full ranks, with maximum probability drift 0.015278 against CPU; reject under unchanged gates. All owned workers exit and GPU returns to idle. No warmed speed comparison or default change. |
| [CUDA FP32 efficient-attention screen](attention.md) | Corrected B1/B8 diagnostics pass all 50 outcomes on eight unique cases under unchanged numerical/ranking gates, with 28 actual FP32 CUTLASS attention kernels per candidate profile. Independent raw audits agree. Original observer failure remains archived. These screen receipts establish no full77 parity or warmed speed; the following full-cohort study supplies that separate evidence. |
| [CUDA full77 and paired body latency](latency-results.md) | Independent audits preserve all 2,482 outcomes and strict gates. Six warmed pairs support median 1.0565x at B1 and 1.0653x at B8 against FP32 math attention. B1 repeated-row residence p95 regresses 1.65%, concentrated in the four longest cases. Representative MLP costs nominate packed gate/up; exact kernel-grid tail diagnostics remain pending. No online, 20x, held-out-quality or production-promotion claim. |
| [CUDA projection storage preparation](cuda/packing.md) | A local CPU probe on real layer-zero weights and cached activations preserves exact outputs and one shared 24 MiB gate/up backing; nine tests including Hypothesis pass. Full-model/CUDA parity, device aliases, peak memory and speed remain unmeasured. The reviewed native L192/L197 diagnostic awaits changed-payload transfer authorization. |
| [Server admission under pool saturation](server/admission.md) | Reproduced concurrent requests returning 200 instead of documented 503 when the shared pool has one token. Admission now precedes pool dispatch, with atomic queued/worker cancellation ownership. Five focused tests including 12 Hypothesis histories and full 894-test native/DMBench regression pass. Model arithmetic is unchanged; no HTTP throughput or latency claim. |

These assays use one pinned release and 77 unique training-source development
cases. Repeated predictions establish numerical/runtime behavior within this
assay, not new independent quality evidence. Raw failures, repeats, telemetry
and declared source archives remain beside each receipt. Native DMBench role
projection is separately [contract-tested](../dmbench-integration.md); the old
full-model pilot used the original verbatim recipe. A separately frozen native
development assay now retains all 30 cases: six expected 2,048-token admission
failures and 24 admitted authored cases, with matching scalar/batch selections.
The released artifact allows 255 choices; the 64-choice limit in the fixture
was deliberately configured for a boundary test. No menu was reduced.

The endpoint and prefix gains were measured separately and must not be
multiplied or added. The subsequent combined assay measured its own complete
latency and passed full quality and five-pair gates at both batch eight and
scalar. Independent inspection verified all cycles, source archives and original
thresholds. These results motivate one explicitly configured, eagerly built
prefix scorer; they do not establish a universal default or online queue gain.
