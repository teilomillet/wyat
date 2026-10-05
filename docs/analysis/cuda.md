# CUDA feasibility and precision contract

Inspected on 2026-10-05. This is a read-only contract audit, not a CUDA
benchmark. No weights were transferred, no model was loaded remotely, and no
GPU workload was launched. The existing CPU FP32 reference remains the quality
denominator.

## Observed local execution policy

[`torch.py`](../../wyat/backends/kai/torch.py) loads the backbone into FP32,
validates its weights, and accepts CPU or CUDA devices. CUDA requires reported
BF16 support. Before moving to CUDA, only Linear weights/biases whose BF16
round-trip is exact are stored in BF16. Every CUDA `score_many` call then
enables BF16 autocast for the backbone. Embeddings, norms, residuals and head
weights remain FP32; the candidate head explicitly disables autocast and casts
its inputs to FP32. Tokens, endpoint positions, causal ordering, choice-only
semantics and host response computation are shared with CPU.

This matches the inspected released Kai GPU precision recipe, rather than an
all-FP32 CUDA policy. Source is pinned to
[`decision2/qwen.py` at cd49ea3](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764/decision2/qwen.py),
SHA-256 `62df8de050d38cd92dc68c12de7d9b65d041a14317feda02ff8ddafc3c69c779`.
Its local retained bytes were inspected; the browser could not reopen that
file in this audit. The pinned
[`decision2/fast.py`](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/blob/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764/decision2/fast.py)
requires HIP and Transformers 5.17 for installation, with fused kernels further
limited to gfx942. Those optimizations do not establish an NVIDIA fast path.

Exact weight storage is not exact activation arithmetic. BF16 autocast rounds
eligible operations even when parameters are FP32. An outer disabled-autocast
context does not override the enabled context inside `score_many`.
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html)
describes these nested regions and operation-specific dtypes. FP32 head tensors
also do not prevent process-global TF32 matmul policy from affecting head
Linear operations. The adapter does not configure that policy. PyTorch 2.14
provides IEEE precision settings and warns against mixing them with legacy
`allow_tf32` controls.
[CUDA precision documentation](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

SDPA backend selection can change rounding. The math implementation supports
FP32, while fused kernels have their own eligibility restrictions and numerical
behavior. Record dispatch rather than equating `sdpa` with FlashAttention.
[SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

## Observed resource boundary and unknowns

Root's fresh read-only probe reported an RTX 4070 Ti, 12,282 MiB total memory,
6,364 MiB used and 98% utilization. The established concurrent training workload
is outside this experiment. Apparent free memory is not permission or evidence
of idle compute. No CUDA inference experiment should contend with it.

Remote PyTorch/Transformers versions, CUDA build, precision settings, actual
BF16 capability check, attention dispatch, workspace peaks and Wyat numerical
parity are unknown. Local CPU fixture tests establish encoding/head mechanics;
the released-weight CPU/MLX assay does not test CUDA. No current evidence
supports or falsifies CUDA passing the strict FP32 quality gates.

## Smallest useful experiment after an idle window

Use an experiment-only forward with the production encoder, backbone, head and
endpoint gather. Load through the CPU path, move **all** floating weights to
CUDA FP32, and execute with CUDA autocast disabled. Do not call the existing
CUDA `score_many` for this control, because it re-enables BF16. Do not change
production dtype policy merely to create the control.

In a separate process, record installed versions/builds and save/restore the
version-appropriate precision settings. For PyTorch 2.14, request IEEE global
and CUDA matmul precision without mixing legacy flags. Force
`sdpa_kernel(SDPBackend.MATH)` for the initial control and verify parameter,
head-input and projection dtypes in an untimed diagnostic. This controls the
main reduced-precision choices; it does not prove CPU/GPU bitwise identity.
PyTorch explicitly permits different results across devices and batched forms.
[Numerical accuracy documentation](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html)

Freeze source bytes, artifact identity and the full 77-case input/reference
receipt before loading. Start with batch one and maximum length 512; retain
every output or failure. Gate every selected candidate and full ranking, logits
at absolute/relative tolerance 1e-3, and probabilities at 1e-4. Then check the
same cohort at batch eight if measured memory permits. Any drift, missing row,
unsupported setting or allocation failure remains a failure/unknown, not an
omitted sample.

Only after the FP32 control passes, compare the ordinary CUDA BF16 policy to
the same frozen reference and separately to the FP32 CUDA control. A failed
BF16 gate rejects that candidate under the current quality contract; matching
upstream's dtype recipe does not waive the gate. Keep shape/backend changes as
separate arms. Run one warmup and three measured repetitions per arm, with a
600-second whole-process cap and early exit after a material quality failure.

Include encoding, transfers and host response in complete-request latency.
Synchronize around any separately reported device interval; GPU submission
alone is not completion time. Capture allocated/reserved/peak CUDA memory and
own-process diagnostics outside the timed region, and require uncontended
occupancy observations before and after. Performance under a busy training
job cannot establish a standalone speedup. This proposal is prepared, not run.
