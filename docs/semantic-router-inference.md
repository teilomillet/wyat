# Semantic Router Decision 2.0 inference investigation

Inspected on 2026-10-05. Wyat's eventual model shape remains open. This document
describes the released Kai candidate head as one concrete, reversible inference
adapter; it does not make that architecture the contract for every Wyat model.

## Source and evidence

The relevant project is [vLLM Semantic Router](https://github.com/vllm-project/semantic-router),
whose [Decision-2.0-Kai-0.6B release](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764)
includes its inference implementation. It is distinct from Aurelio's package
also named semantic-router. The Hugging Face tree and
[release commit](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/commit/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764)
were checked through the browser. Local source bytes were inspected from Kayak's
existing release-inspection directory and checked against their stored receipt
and the packaged manifest. The pin is
`cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764`, whose builder source commit is
`1b977b6af46e3794859f84468c7366f504f2ec32` and runtime source commit is
`fbedfded8ce33c7c25a99b8fb1f71f733b2c2c44`.

The release gives the runtime, head, weights and tokenizer Apache-2.0 licenses.
Wyat's adapted mathematical implementation retains attribution in `NOTICE`.
The inspected source entry points are:

| File, relative to the pinned release | Observed responsibility | SHA-256 |
| --- | --- | --- |
| `decision2/_vendor/dev2model/decision_model.py` | Serialization, segmented tokenization, endpoint gather, head | `1ab1e49bccc8b84632a32565ce666a6db94298372ee891dee5b68e54f80957fb` |
| `decision2/_vendor/dev2model/infer.py` | API validation, stable softmax, response semantics | `7bb9f3cc88f04cd6e4ded62e69f41dd1c7a3d0154a76acdf44c0d73e4163ef58` |
| `decision2/qwen.py` | Local checkpoint loading, precision, per-question batching | `62df8de050d38cd92dc68c12de7d9b65d041a14317feda02ff8ddafc3c69c779` |
| `decision2/fast.py` | ROCm graphs, cast reuse, fused-kernel gates | `d125a98848089ad8be04ef1c799c1daee5ea7704e138c17a874692321a206b54` |
| `decision2/shared_ctx.py` | Opt-in shared-prefix cache and packed attention | `3ea492989258d460b28a5b568e388696a8d07c0b985f960e38d3c99d68620be9` |

## Observed input-to-output path

`state + question + ordered criteria → strict validation → segmented tokens →
causal Qwen3Model → candidate endpoints + final query → FP32 candidate head →
optional Score offsets → temperature + softmax → typed answer`.

Each question is its own row. The prefix renders `Context`, `Task type`,
`Question` and `Options`. Each option is compact, sorted-key JSON containing
`key` and `description`, enclosed by `<option>` tags. Unicode is retained.
Prefix, each entire option and final decision cue are tokenized separately with
no added special tokens. Encoding the visually identical whole string can yield
different tokens at segment boundaries. The candidate representation is the last
token of its complete option, including its closing tag. The query representation
is the final token of the decision cue. Overlength inputs are rejected; no
candidate or context is silently truncated. Rows use right padding to a multiple
of eight; candidate order is preserved. The released prompt allows 2–255 choice
candidates and caps the total encoded input at 8,192 tokens.

The backbone is a causal Qwen3 text model, without vocabulary logits or text
generation. Earlier option endpoints cannot see later options, while the final
query sees all options. Independently cached candidate hidden states therefore
would change this model. Permuting candidates also changes the model input.

For candidate vectors `C[B,N,H]` and query `q[B,H]`, two LayerNorms first compute
`c` and `g`. The shared head computes
`sum(key(c) * query(g), -1) / sqrt(D)` plus
`scalar(GELU(candidate_mlp(c) + query_mlp(g)))`, with `D=256` in the release.
The head disables autocast and calculates in FP32. Invalid padded candidates
are masked. A request's questions run together, subject to a padded-token budget
on the hybrid Qwen3.5 family; this particular Qwen3 model has no such upstream
gated-delta budget.

CPU uses FP32. GPU uses BF16 autocast for the backbone, retaining FP32 residuals,
embeddings, norms and head. Only Linear weights exactly representable as BF16
are stored in BF16. Casting the entire model to BF16 is a different policy.
The release uses SDPA. Its CPU and GPU policies are not claimed numerically equal.

The upstream product response applies a stable host softmax. Choice ties resolve
by original caller order. Confidence is normalized entropy, not accuracy.
Score has model-bound, level-count-specific logit offsets; Noul returns the
probability of the `true` key. Wyat's initial Kai adapter accepts choice only so
the other response semantics cannot be accidentally substituted.

## Optimization hypotheses and their falsifiers

| Hypothesis | Necessary check before accepting it |
| --- | --- |
| Reuse exact option token IDs for repeated menus to reduce host tokenization. | Identical complete IDs and endpoint positions with cache off/on across Unicode, tag-like text, and candidate orders; bounded cache; measure host and complete-request time separately. |
| Batch similar lengths to reduce padding. | Same order restored; batch/singleton decision and probability drift measured, including near ties; serving queue latency counted. |
| Keep BF16-exact Linear weights resident on GPUs to avoid repeated casts. | Exact representability check per parameter; FP32 head/norms remain; compare logits against the existing GPU policy. Upstream already does this. |
| Graph replay/fused elementwise kernels reduce launch overhead on short requests. | Hardware/version gate, exact padded shapes/masks, eager fallback, bounded memory, forced kernel failure, full parity and realistic latency. Upstream already implements this on ROCm. |
| Shared state prefill saves repeated backbone work for multi-question requests. | Isolation between suffixes, original rotary positions, near-tie errors and calibration measured, cache ownership explicit. This changes floating-point order and is not exact by construction. |

The fast path is gated to ROCm and Transformers 5.17; fused kernels are further
gated to gfx942. The graph cache stops admitting shapes after 512 graphs or 4 GiB
of retained outputs and does not evict live captures because upstream observed
GPU faults. Shared context is disabled by default; both tree packing and KV cache
mode can change logits. Upstream explicitly notes that even the normal path is
not strictly batch-invariant.

## Supported conclusions and remaining unknowns

Source inspection supports the data flow and identifies useful controlled
experiments. It does not establish deployed production behavior, reproduce the
released accuracy, or validate upstream performance claims. The original source investigation preceded full-model measurements. Subsequent
released-weight CPU/MLX observations are recorded in the
[implementation receipt](implementation-evidence.md). Tiny random fixtures
exercise mechanics and numerical equivalence only.

The current adapter also rejects duplicate/nonfinite metadata, wrong tensor
dtypes, incomplete backbone/head state and nonfinite weights at load time.
Head tensors must be FP32; backbone storage may be FP32 or BF16. Weight scans use
bounded temporary chunks. Artifact identity hashes include consumed configs,
the optional package manifest, tokenizer and both weight files. The initial
request limit is 2,048 tokens (explicitly configurable up to the package cap);
oversized inputs are errors, not truncations. A per-segment byte ceiling of
`32 × max_length` bounds retained token-cache strings. This physical resource
limit can reject a segment that a particular tokenizer would otherwise encode
within the token cap. When a package declares a `files_sha256` table, consumed
files with declared entries must match their 64-hex digests; local bundles without
that optional table remain supported. This is integrity against the supplied
manifest, not authentication of its publisher. Exact option-token caching
is available explicitly and remains off by default until a target workload
establishes complete-request benefit.

Real BPE and FP32 MLX parity have since passed on the 77-case development assay.
Actual CUDA/BF16 parity remains unknown; the shared GPU host was occupied.

There is no evidence yet for a 20× complete-request improvement. Omitting
generation, SDPA and inference mode are already upstream baseline behavior.
Every claimed speedup must name the denominator, hardware, precision, shapes,
warm/cold policy, tokenization inclusion, latency distribution and decision/
probability drift. A head-only or repeated-menu tokenization result cannot be
reported as whole-engine acceleration. The supported first implementation is a
small, local-only adapter with exact tokens and head math; architecture,
quantization and compilation remain gated experiments; prefix sharing has no
production path.
