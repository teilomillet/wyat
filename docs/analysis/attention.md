# CUDA efficient-attention screen

Prospective declaration, 2026-10-05. The engine task owns this bounded screen.
The preceding FP32 batch-one and batch-eight controls passed all 77 strict
CPU-reference gates; the released BF16 recipe failed. No production dtype or
attention policy changes here.

## Hypothesis and arithmetic boundary

An explicit full causal mask may let the maintained Transformers SDPA path
expand eight KV heads to sixteen query heads and use PyTorch efficient
attention on FP32 tensors. This could avoid the unfused math attention path,
but expansion and mask construction add work. CUDA attention's ordinary time
fraction has not been measured, so no Amdahl gain or whole-engine ceiling is
predicted from the Mac profiles.

The actual frozen Transformers `5.18.0` integration repeats KV heads when a
mask is present on this Linux/CUDA path. Prepared 4D masks are accepted by its
masking integration. The candidate supplies a fresh `[1,1,L,L]` Boolean mask,
with `True` exactly when key position is no later than query position. It uses
full prefill, no past state and `use_cache=False`. Right padding follows all
observed query/candidate endpoints; masking future positions preserves their
causal domain. Scalar `L=1` remains a full one-token causal domain.

Force `SDPBackend.EFFICIENT_ATTENTION` inside the adapter's forward. The inner
context must override the unchanged FP32 scorer's outer math-only context;
unsupported execution must fail visibly rather than silently fall back to math.
Mask construction and maintained KV expansion belong inside inference. There
is no retained mask cache or duplicated model storage. The original scorer's
encoder, head, device and backbone ownership remain unchanged; a typed view
adapts only the backbone call at the frozen nominal type boundary.

FP32 input/output tensors and IEEE process matmul settings do **not** prove
IEEE arithmetic inside this candidate. PyTorch's version-tagged
[GEMM specialization](https://raw.githubusercontent.com/pytorch/pytorch/v2.12.1/aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h)
selects CUTLASS `OpMultiplyAddFastF32` for float on SM80 and later; its
[forward kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.12.1/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h)
uses that type for QK and probability-value products. This is a FastF32 quality
hypothesis, not exact arithmetic. The tag alone does not prove the installed
binary's internal implementation. Record its exact build and actual operator/
kernel observations; retain unobserved details as unknown.

PyTorch's maintained mask preprocessing may pad additive bias row strides for
efficient-kernel alignment. Record actual shapes, strides, mask dtype and
selector evidence rather than inventing a sequence-length restriction. The
[maintained attention implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.12.1/aten/src/ATen/native/transformers/attention.cpp)
and observed dispatch are separate evidence.

## Frozen inputs and staged evidence

Preserve the six original CUDA source/test files, all twenty production files,
the v2 transfer manifest `f7d9d7a4…`, and the reviewed FP32 batch-one receipt
`d528a44e…` plus terminal wall receipt `cb00f5f0…` byte-for-byte. Admit all 44
canonical assets and exact package/build identities before GPU allocation.
Validate every original token, endpoint and query position against the pinned
CPU reference. Freeze the new source/test/declaration bytes separately and
check them before and after execution.

Use only the first eight cases in the original ordered 77-row BANKING77
training-source development cohort. In separate batch-one and batch-eight
jobs, run one ordinary math control cohort and one ordinary candidate cohort.
Retain all sixteen case outcomes per job, including errors and unknown cases.
Compare both arms with the original CPU reference and the admitted CUDA FP32
control: identical choices and full rankings, no accuracy decrease or failures,
logit absolute/relative tolerance `1e-3`, probability tolerance `1e-4`.

Only after ordinary quality passes, profile one control and one candidate
batch: the first case at batch one and all eight cases at batch eight. Retain
their outputs under the same gates. Record actual SDPA Q/K/V shapes, strides,
dtypes, mask and public selector diagnostics without retaining tensor owners.
Require an efficient-attention operator for every candidate layer and preserve
actual kernel names when available. Selector support is not observed execution;
an efficient operator is not proof of the precise arithmetic used internally.
Profile overhead and diagnostic events support no latency or throughput claim.
Failure of an ordinary arm stops before profiling, with attempted failures and
unstarted case IDs retained. No failed or unsupported sample is omitted.

Each job uses one owned fresh worker with a 180-second whole-process cap.
Require uncontended pre-allocation and per-batch observations; stop on foreign
compute ownership and never signal another job. Restore precision/backend
contexts and dispose observers before successful completion. Preserve partial
receipts, timeouts, source drift, write failures and cleanup failures.

Full77 confirmation, warmed alternating performance comparisons and production
selection need separate release after independent review of this screen. Eight
unique development cases cannot gate full77 behavior or held-out Wyat quality.
No timing or production result is accepted by this diagnostic screen.

## Prospective observer correction, 2026-10-05

The first batch-one receipt retains `dispatch_passed=false` and parent exit 1.
Its independent audit confirms all eighteen ordinary/profile outcomes passed
unchanged numerical and ranking gates. The public dispatcher observed twenty-eight
`aten.scaled_dot_product_attention.default` calls with FP32 equal-head Q/K/V
and the Boolean `[1,1,L,L]` causal mask. It does not expose internal C++ dispatch
or the converted additive bias. The profiler separately recorded twenty-eight
private efficient-attention operators, twenty-eight efficient forward operators,
and twenty-eight CUTLASS FP32 attention kernels. Original source text is retained
in the immutable first-run source archive; its declaration and flags remain
unchanged. This is an observation-boundary correction, not a quality-gate change.

For the second, separately hashed seven-file source declaration, require both:

- Exactly twenty-eight public SDPA observations with expected FP32 CUDA Q/K/V
  shapes and innermost strides, plus the fresh Boolean causal-mask shape.
  Require effective dropout zero, causal flag false, GQA flag false and default
  scale: either omitted (`None`) or a finite float exactly `128**-0.5`. The
  frozen HF/Qwen source explicitly supplies that numeric mathematical default.
  Resolve positional/keyword arguments and pinned schema defaults, preserving
  the observed representation rather than requiring omission.
  Label these metadata as preceding PyTorch's bias conversion and padding.
  Causal mask contents are supported by construction and exact mask tests;
  metadata alone observes only shape, dtype and strides.
- Exactly twenty-eight private efficient-attention and forward profiler events,
  no math/Flash/cuDNN fallback events, and twenty-eight actual FP32 CUTLASS
  attention kernel events in the pinned runtime.

Record public selector diagnostics using those same effective call arguments
at their actual boundary. `SDPAParams` cannot encode an explicit scale; label
that API limit even when the observed value is the mathematical default.
Eligibility before
bias preprocessing is diagnostic only; it cannot establish execution or reject
an internally prepared mask that the recorded kernel accepted. Converted bias
strides remain unknown. Kernel names support executed implementation identity,
not proof of IEEE internal arithmetic. Preserve missing kernel observations and
all failures; do not infer a pass from public metadata or forced context alone.

First rerun the eight-case batch-one screen with this corrected observer and
new source identity. Only its independently reviewed success releases the same
eight-case batch-eight screen. Both keep the original numerical/ranking gates,
ordinary/profile outcome counts, occupancy checks and 180-second parent caps.
No original receipt is overwritten. These repeated observations add no unique
quality cases and support no latency, full77 or production claim.
