# Runtime mechanisms for Kai prefill

Read-only investigation, 2026-10-05. The target is one causal Qwen3 prefill,
followed by the released candidate head over ordered option endpoints and the
final query endpoint. There is no autoregressive decoding. Wyat's future trained
model shape remains open; these findings apply to the currently supported Kai
adapter.

## Evidence and source pins

These are inspected current upstream snapshots, not the versions deployed by
every user. Wyat's measured environment uses MLX 0.32.3 and MLX-LM 0.31.3.

| Repository | Inspected commit | Relevant implementation |
| --- | --- | --- |
| vLLM | `138810056093301f4881050fcf2b1786939da387` | [Qwen3](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/vllm/model_executor/models/qwen3.py), [Qwen2 MLP](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/vllm/model_executor/models/qwen2.py) |
| SGLang | `b792228b35b21565067520857319dfc05e4d134e` | [Qwen3](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/python/sglang/srt/models/qwen3.py), [Qwen2 MLP](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/python/sglang/srt/models/qwen2.py) |
| MLX | `f8aaf49d4a7e4c0ccd31986b45b69d791f925ed8` | [maintained kernel/runtime source](https://github.com/ml-explore/mlx/tree/f8aaf49d4a7e4c0ccd31986b45b69d791f925ed8), [SDPA API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) |
| MLX-LM | `5cfec4cb39deba54210b3ff4d86f2337c7bc10b5` | [Qwen3](https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/mlx_lm/models/qwen3.py), [compiled activation](https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/mlx_lm/models/activations.py), [attention wrapper](https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/mlx_lm/models/base.py) |
| vLLM Metal plugin | `2fd759dbd1171a77f7889d04a3b6d75c78fbeea3` | [pooling scope](https://github.com/vllm-project/vllm-metal/blob/2fd759dbd1171a77f7889d04a3b6d75c78fbeea3/docs/text_embedding_pooling.md) |

Semantic Router provenance and exact input/head semantics are recorded in
[the source investigation](../semantic-router-inference.md). In particular,
segmented tokenization, causal option ordering, original RoPE positions, head
FP32 arithmetic, calibration and rejection of overlength inputs are contracts.
Similar-looking prompts, independent candidate embeddings and normalized
sentence embeddings do not satisfy them.

Observed local baseline: [five paired full-cohort cycles](../../benchmarks/real/runtime-cpu-mlx77.json)
gave FP32 MLX Metal versus Torch CPU/8 threads a median `2.024×` speedup
(paired 5th–95th percentiles `1.850×–2.158×`). All 385 repeated selections and
rankings agreed. Maximum logit/probability differences were `1.287e-5`/
`2.462e-6`. Timing includes tokenization, batching, host results and decision
interpretation; it excludes loading, HTTP and admission queues. This is a
77-case development assay with unknown training exposure, not a quality
generalization result. [Compilation](../../benchmarks/real/mlx-fp32-compile77.json)
was `0.990×`; full BF16 and FP16 failed the declared drift gates. Those are
completed negative observations, not untested suggestions.

The subsequent [synchronized stage profile](../../benchmarks/analysis/results/mlx-profile77-b8.json)
passed the same cohort parity gate. Its diagnostic fractions were MLP `49.74%`
and attention block `40.14%` (the latter includes projections/norm/RoPE/output,
not only SDPA), versus encoding `0.44%` and head `0.12%`. The fenced path took
`3.309s` versus ordinary `2.800s`, an `18.17%` instrumentation increase. These
conditional observations support testing block projections first, but cannot
identify individual kernel time or measured DRAM bandwidth. Removing the whole
MLP diagnostic fraction would cap speedup at `1.990×`; removing head alone at
`1.0012×`. Projection packing affects only part of those blocks.

## Transferable mechanisms and integration boundaries

| Mechanism | Observed upstream behavior | Transfer to Wyat and cost |
| --- | --- | --- |
| Packed projections | vLLM and SGLang combine Q/K/V in one linear projection; combine gate/up in one MLP projection; apply fused SiLU multiplication. | Small, typed MLX projection adapter using maintained linear kernels is plausible. Concatenate original weights once at load, split outputs in their original order, and release duplicate storage. FLOPs and weight bytes are unchanged; fewer launches/input reads and different GEMM shapes are the hypothesis. |
| Fused elementwise work | vLLM carries residual separately through RMSNorm; MLX-LM already uses fast attention/RoPE/norm and compiled SwiGLU. | Profile residual/add/norm launch fraction before introducing a kernel. Counting an already-used fused primitive as a new improvement would overstate the result. |
| Packed variable-length prefill | vLLM/SGLang model forward paths operate on scheduler token batches and reconstruct per-request boundaries. | Can reduce padding versus a rectangular batch. Requires attention metadata, request ordering and endpoint translation, not just concatenating requests. Medium-to-high integration cost compared with length bucketing inside the current engine. |
| Prefix reuse | Semantic Router's opt-in shared-context path reuses the actual causal prefix before independent question suffixes. | Useful only when the token prefix is truly identical. Menus occur after the query in Kai; repeated menu text is not generally a reusable KV prefix. Token-ID caching remains different from hidden-state/KV caching. |
| Graph replay | Semantic Router captures exact padded shapes on ROCm with bounded retained outputs and explicit fallbacks. | CUDA/HIP graph mechanisms do not transfer directly to Metal. Local MLX whole-function compilation already failed to establish benefit. A graph hypothesis needs measured launch overhead and enough shape reuse to amortize capture. |

The first row follows the pinned [vLLM Qwen3 implementation](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/vllm/model_executor/models/qwen3.py)
and [SGLang MLP implementation](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/python/sglang/srt/models/qwen2.py).
The shared-context and graph rows follow inspected released Semantic Router
source; its MI325X break-even thresholds and gfx942/Transformers guards are
hardware-specific observations, not portable Wyat settings.

vLLM has a pooling runner, including token embeddings. Its stock LAST/ALL/STEP
readouts are not Kai's grouped candidate/query head; STEP selects token IDs,
not arbitrary caller endpoint indices. A future adapter must request raw
hidden states, disable unwanted embedding activation/normalization and apply
the original head before transferring small scores to the host. Sending all
token vectors over HTTP would add avoidable transfer and serialization work.
The official pooling documentation explicitly declines a general performance
guarantee over Transformers. [Pooling models](https://docs.vllm.ai/en/latest/models/pooling_models/)

SGLang's standard Qwen3 embedding path uses normalized LAST pooling. However,
the pinned [pooler](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/python/sglang/srt/layers/pooler.py)
also has `token_indices_to_pool`: gather exactly at per-request positions and
optionally return raw pooled hidden states. This is a useful narrow integration
boundary. Its score-head call is flat/per-position; Kai must regroup candidate
rows and each final query row before its shared head. Cache hits/chunked prefill
must not hide requested endpoint states. SGLang's newer
[`/v1/decisions`](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/docs/docs/supported-models/decision_models.mdx)
uses next-token label scoring for chat models; naming alone does not make it a
Kai implementation.

On Mac, the smallest maintained-kernel route remains native MLX. vLLM documents
experimental Apple Silicon **CPU** support requiring a source build, and points
to the community-maintained Metal plugin. The plugin's current experimental
decoder pooling accepts LAST embeddings and a specific Qwen3 reranker; it
explicitly rejects generic heads, decoder ALL/STEP and generic token pooling.
That prevents a drop-in endpoint-aware Kai deployment.
[vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/),
[Metal pooling contract](https://github.com/vllm-project/vllm-metal/blob/2fd759dbd1171a77f7889d04a3b6d75c78fbeea3/docs/text_embedding_pooling.md)

SGLang's inspected CPU recipe targets Intel Xeon AMX. It establishes no Apple
Metal support. Its GPU kernels/scheduler remain relevant to a later supported
GPU comparison, but that comparison has not run here.
[CPU recipe](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/docs/docs/hardware-platforms/cpu_server.mdx)

Batch invariance is a separate contract from equality to a particular reference.
vLLM's beta invariant mode has specific NVIDIA/XPU support and performance
tradeoffs; SGLang's deterministic mode supports FlashInfer/FA3/Triton. Neither
establishes FP32 Metal equality. Packing projections, changing batch shape or
attention kernels can change reductions even when real-number algebra agrees.
Compare raw logits, probabilities, all ranks and tie behavior after each change.
[vLLM invariant mode](https://github.com/vllm-project/vllm/blob/138810056093301f4881050fcf2b1786939da387/docs/features/batch_invariance.md),
[SGLang deterministic mode](https://github.com/sgl-project/sglang/blob/b792228b35b21565067520857319dfc05e4d134e/docs/docs/advanced_features/deterministic_inference.mdx)

## Ranked hypotheses and falsifiers

1. **Projection packing:** fewer launches and larger GEMMs improve the measured
   attention/MLP projection fraction. Test QKV and gate/up separately, then
   together, at fixed FP32/weights/tokens/batches. Reject for failed drift gates,
   extra steady memory, or no repeatable full-cohort improvement. Qwen3's query
   width is `16 × 128 = 2048`, while hidden width is 1024 and each K/V width is
   `8 × 128 = 1024`; packing must not assume query width equals hidden width.
   This affects every layer and has a larger possible ceiling than final-layer
   pruning, but no speedup is established by source inspection.
2. **Batch shape efficiency:** distinguish amortized model weight reads from
   padding waste. Sweep six ready-work batch sizes `1,2,4,8,16,32`, retain token
   lengths and complete cohort wall times, then perturb only length grouping.
   Restore caller order and check ranks. Ready-batch throughput is not serving
   latency; a scheduler must also account for admission delay and fairness.
   Reject if gain comes from a changed workload, dropped failures or unmet
   latency constraints.
3. **Selective final-layer readout:** after all earlier layers, compute final
   K/V for all positions; compute queries/output projection/MLP/final norm only
   for candidate/query endpoints. Token-local norms and MLP permit this in real
   arithmetic. Preserve each endpoint's original RoPE position and use explicit
   masks `key_position <= original_query_position`. MLX's causal string uses
   lower-right alignment when query length differs from key length; using it
   for arbitrary selected rows is incorrect. Retain native GQA K/V grouping.
   Explicit masks may change fused dispatch and erase arithmetic savings.
   [MLX SDPA contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
   RoPE accepts one start offset per batch example, not arbitrary per-token
   positions. Applying full-query RoPE before gathering preserves the maintained
   path but saves less work. Treating each selected query as a length-one batch
   example with its original offset is another testable route, not an established
   performance improvement. [MLX RoPE contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html)
   Reject unless endpoint logits pass and measured full-call speed survives.
   Dropping earlier non-endpoint rows is invalid: later layers need their K/V.
4. **Shared actual prefixes:** only after a workload exhibits repeated prefixes
   long enough to dominate prefill. Preserve causal suffix isolation, rotary
   positions and requested cached endpoint states. Freeze a cache key including
   artifact identity and token IDs, bound lifetime/memory, and perturb near ties.
   A margin-triggered fallback is additional policy, not exact equivalence; it
   needs independent validation before becoming a production contract.

## Amdahl, roofline and scaling interpretation

The explanatory tools in `benchmarks/analysis/scaling.py` are candidate models.
For a measured time fraction `f` accelerated by `s`, predicted whole-call
speedup is `1 / (1-f+f/s)`. Even removing the **entire** last block of 28 equally
costly blocks gives only `28/27 ≈ 1.037×`; selective final-layer work saves less.
Use the actual measured fraction, including host work, to replace this rough
assumption. A 20× fixed-workload improvement requires eliminating at least 95%
of baseline time; finite acceleration requires affecting more than 95%.

For Kai's measured shape, one layer's projection parameter count is
`H(Q+2K) + QH + 3HI = 15,728,640`, with `H=1024,Q=2048,K=1024,I=3072`.
Projection FLOPs are `2 × padded_tokens × layers × parameter_count`.
Dense attention FLOPs are `4 × B × layers × query_heads × head_dim × T²`;
ideal triangular causal work is `2 × B × layers × query_heads × head_dim × T(T+1)`.
These exclude norms/RoPE/softmax/head and do not assert a kernel's actual tiling.
Fused projection packing changes neither projection arithmetic nor weight bytes.
Roofline lower bounds need justified device/dtype throughput upper bounds and
actual traffic lower bounds. Summing synchronized microphase timings can
inflate overhead; compare the instrumented whole call with an ordinary control.

Gunther's USL is `C(N)=N/[1+σ(N-1)+κN(N-1)]`, with an interior maximum
`sqrt((1-σ)/κ)` when `κ>0`. Its queue-theoretic interpretation relates
serialization and state-dependent pairwise costs to a synchronous
machine-repairman bound. The paper also discusses paging/thrashing as retrograde
behavior. [Gunther, arXiv:0808.1431v2](https://arxiv.org/abs/0808.1431v2)
For this single-GPU batch sweep, `N` is examples per call rather than physical
processors: GEMM shape, padding and memory reuse all change. A fitted positive
`κ` therefore cannot identify cache coherence. This application is descriptive,
an inference limitation rather than a new claim from the paper. Retain residuals,
unconstrained estimates, repeated measurements and ill-conditioning warnings;
negative or weak fits are inconclusive. A predicted optimum outside measured
loads is extrapolation. Follow the user's
[collection guidance](https://teilo.xyz/collections/universal-scaling-law/)
for at least six warmed, steady-state loads, then use traces and one-factor
perturbations to distinguish bandwidth, padding, kernel dispatch, launch overhead
and contention. Do not force coefficients into a preferred causal story.

No vLLM/SGLang integration, GPU execution, new precision policy or 20× inference
claim was validated by this investigation. The supported result is a set of
source-grounded experiments with explicit contracts and rejection conditions.
