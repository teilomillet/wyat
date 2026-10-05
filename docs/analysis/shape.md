# Prospective CUDA tail diagnostic

Declaration, 2026-10-05. The independent [ordinary timing results](latency-results.md)
support a modest B1 cohort-body gain but show a 1.6536% repeated-row residence
p95 regression. All six pairs regress for the four longest development inputs
at 197, 199, 203 and 214 tokens; the remaining maximum is 192, with 193–196
unobserved. Those associations do not establish a cause or a routing threshold.

Hypothesis: a launch-geometry boundary between the actual 192- and 197-token
inputs contributes to the observed regression. The diagnostic does not assume
a query tile or grid size from `64x128` in an attention kernel's name.

## Frozen inputs and ownership

Use exactly `banking77-train-03357`, the first canonical 192-token case at
offset 7, and `banking77-train-00566`, the sole 197-token case at offset 18.
Select from the pinned CPU-reference encodings before new captures. Preserve
every token, candidate ID, endpoint, query position and input order across the
full original 77 cases before loading weights or allocating CUDA tensors. No
replacement, text alteration, padding-width search or synthetic input is admitted.

Keep unchanged the original 44 assets (`f7d9d7a4…`), CUDA FP32 main/control wall
(`d528a44e…` / `cb00f5f0…`), corrected observer seven (`589a45fe…`), latency five
(`a26d5be4…`) and 48-source execution archive (`21b3fa50…`). Admit the exact B1
confirmation main/wall (`c41d4d08…` / `29d7fe3a…`), its independently reconstructed
full77 quality/source/dispatch evidence, and the separately pinned timing and
exploratory shape evidence. Old receipts and declarations remain immutable.

Freeze three new `cuda/shape` Python files, `test_cuda_shape.py` and this document
in a separate five-file declaration. Transfer those files, their declaration,
and the two already reviewed timing/shape-association JSONs: eight files total.
The latter two are immutable admission evidence, separately pinned rather than
new execution sources. Use the existing engine workspace only after independent
source/test review.
Retain the new executed source archive separately from prior sources.

One fresh worker owns one FP32 backbone/head, shared by math and the reviewed
efficient adapter. Use threads eight, maximum length 512, token cache zero,
full prefill without KV cache, disabled autocast and IEEE global/matmul settings.
No registration, backbone duplication, production policy or default changes.
The parent caps only its owned worker group at 180 seconds. Fresh ownership
checks before allocation and each call reject foreign jobs; source/history
checks, persistence and ownership scans sit outside ordinary body clocks and
captures. Idle observations are not reservations.

## Fixed schedule and gates

For each phase, use exactly this order: L192 math, L192 efficient, L197 efficient,
L197 math. Run phases warmup, ordinary, then profile: **four warmup, four ordinary
and four profiled outcomes**, twelve total. Profile neither shape before all
ordinary outcomes pass. Compare every completed result with its aligned CPU
and CUDA references and same-case arm: identical choice/full ranking, no failed
or missing result, no accuracy decline, original `1e-3` absolute/relative logit
and `1e-4` probability gates. Low or high latency never selects observations.

Ordinary synchronized body/residence and phase/worker walls are retained
separately. Four ordinary controls cannot establish a speed gain or tail estimate;
the original six-pair results remain the independent speed evidence. No new
speed gate or production admission rule is introduced.

Capture each shape/arm once in its own CPU+CUDA profiler context with
`record_shapes` and the immutable public Dispatch observer. Label each call
explicitly with arm, case and actual length. Synchronize before capture and
after its sole call before closing the context. Export each ended context to a
distinct exclusive Chrome JSON under the output's owned `.trace` directory,
which existing packaging exclusions omit. Preserve exact bytes, SHA, size and
path; exporting a scheduled profiler's last cycle alone is insufficient.

Require unchanged efficient evidence at each actual length: 28 public Q/K/V/mask
and effective-flag observations, 28 private efficient/forward calls, no fallback
and 28 actual FP32 SM80 attention kernels. Retain the math comparator's actual
math/no-efficient evidence. Keep explicit-scale selector limitations, converted
bias strides and internal arithmetic unknowns. IEEE matmul and FP32 tensors do
not establish IEEE attention arithmetic.

## Native trace interpretation

Store the complete native traces before designing a schema interpreter. Preserve
each event's original index, phase/category/name, process/thread IDs, timestamp,
duration and entire arguments. Actual grid, block, stream, device and correlation
fields may be missing or malformed; retain those as unknown, without filling
them from tensor dimensions or kernel names.

Independent review links a kernel to a same-trace runtime launch by a unique,
exactly typed correlation value, then to a CPU operation by unique External ID
where observed. Missing or multiple matches remain unknown. Correlation and
External ID are distinct. Do not match by name, order or timestamp proximity.

Chrome timing, Python performance clocks and UTC remain separate domains unless
an observed clock mapping supports a comparison. Kernel-event sums count each
event once, separate from inclusive CPU/runtime durations. Per-stream gaps,
interval unions and spans are descriptive when metadata is present; they do
not prove a critical path, host-launch bottleneck or device occupancy.

- Identical complete grid/block distributions across all 28 efficient kernels
  challenge the proposed grid/block step for this pair. Other internal loop,
  alignment, layout or algorithm boundaries remain unknown. If multiple geometries
  occur, equal multisets leave a possible per-layer permutation unknown. Stronger
  refutation requires one uniform geometry in both captures or independently
  linked corresponding layers; event order and whole-case labels do not establish
  layer identity.
- Different observed geometry establishes a geometry change only. Attention
  duration changes relative to the math comparator and remaining operations may
  support a contribution; content/menu, capture order, governor state and
  intrusive-observer effects remain confounded.
- Missing geometry or linkage leaves the mechanism unknown. Dominant changes
  in other matrix/copy operations or host-visible spans challenge attention as
  the useful next target, without themselves establishing causation.

Retain all attempted and unstarted IDs, partial rows, warnings, failed exports,
timeouts, context/observer cleanup and final-write failures. Positive completion
requires all source/history rechecks and cleanup; `accepted` remains false.
Independent raw audit and terminal GPU/job-release observations follow capture.
This diagnostic supports no online tail/capacity, held-out model quality, 20x
gain, routing threshold or production promotion. A controlled synthetic length
perturbation, if needed, requires its own declaration.
