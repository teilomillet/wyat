# Training and deployment boundaries

Inspected locally on 2026-10-05. These observations describe the current source
and selected artifact bytes; they do not select Wyat's final architecture or
establish deployed model quality. Training remains owned by CLM Lab and Kayak.

## Three distinct recipes

| Path | Observable contract | Deployment consequence |
| --- | --- | --- |
| Kayak public CLM | `clm-qwen3` / `clm-choice-v1`: state and candidates encoded separately, normalized last-token vectors, distinct FP32 projection heads, scaled dot products and menu softmax. Strict manifest pins encoder revision and checkpoint SHA-256. | This is an existing entrypoint pattern, not the recipe for a joint candidate-aware model. |
| Kayak joint research | Qwen3-0.6B with rank-8, alpha-16 q/v adapters; linear, MLP or shared candidate head. Segmented public candidate-line input, final query token and endpoint or mean candidate pooling. | Exact serializer, adapter configuration and pooling must accompany exported weights. Current wrapper admits only two through six candidates. |
| CLM Lab validity diagnosis | Frozen Qwen3-Embedding-0.6B, separately rendered state/action or per-candidate joint text; independent binary-validity logits. | These logits are not a mutually exclusive Choice distribution. Applying menu softmax changes their meaning. The primary model failed its declared quality gate. |

Kayak's public artifact contract is documented in
`/Users/teilomillet/Code/kayak/docs/model-contract.md` and implemented by
`kayak/bundle.py`, `kayak/runtime/_loading.py` and `kayak/runtime/_model.py`.
Its manifest includes family, input recipe, pinned encoder identity, width,
context limit, checkpoint filename and integrity hash. It refuses remote Python
code and unrestricted checkpoint pickle loading. Later recipes need their own
implementation and independent reference fixtures.

## Joint candidate-aware research, exactly as inspected

The source owner is
`/Users/teilomillet/Code/kayak/benchmarks/decision_search/model.py`, with backbone
and adapter ownership in `benchmarks/prerequisites/model.py`.

- Backbone: `Qwen/Qwen3-0.6B` revision
  `c1899de289a04d12100db370d81485cdf75e47ca`, hidden width 1,024.
- Prompt recipe: `public-candidate-lines-segmented-no-special-tokens-v1`.
  Prefix, each complete `A. description` line, newline separators and suffix
  are tokenized separately with no special tokens and no truncation. Concatenated
  segments reproduce every original character. The default cap is 1,536 tokens.
- Candidates: two through six unique contiguous lines labeled A–F in display
  order. Descriptions are single-line; suffix is exactly `\n\nDecision:` or
  `\nReturn only the letter of the next action.`. Other renderings fail.
- Query: final non-padding token under contiguous right padding.
- Candidate readout: LayerNorms, bias-free key/query projections to 256,
  bilinear dot product divided by square-root 256, plus a shared GELU scalar
  branch. Candidate pooling is an explicit `endpoint` or `mean` setting.
- Weights: only q/v adapter matrices and the selected head are saved in
  Safetensors. Restore checks exact tensor names, shape, dtype and finiteness.
  Backbone weights, tokenizer and deployment recipe are not embedded.

The shared head accepts an arbitrary number of already-computed candidate
vectors algebraically. The current serializer, batch spans, training masks and
measurement contract enforce six slots. Larger menus need a versioned input
and result contract plus training/evaluation evidence. A displayed candidate
with unmet prerequisites must remain scored; only padding is masked.

Candidate vectors depend on the complete preceding causal prompt. They cannot
be cached by candidate text alone. Permuting fixed feature vectors permutes
shared-head scores, but permuting prompt lines changes the feature vectors.
End-to-end order invariance is therefore unestablished.

One concrete available checkpoint is
`/Users/teilomillet/Code/kayak/.benchmarks/joint-search-20261003/runtime-quality-v1/seed43_final_repeat_fast/last.safetensors`.
Its directly verified SHA-256 is
`1df83f58cad070efe4adebcfee691ae2897b6f099172e91848ac5fd75c30a10e`.
The header has 112 adapter tensors, `head.weight [6,1024]`, `head.bias [6]` and
no metadata: this is a **linear-slot** artifact. No candidate-head checkpoint
was found among Safetensors files under that research directory in this
inspection. Candidate-head source availability is not a deployable model.

## CLM Lab's retained diagnostic artifact

The primary checkpoint exists at
`/Users/teilomillet/Code/clm-lab/runs/wyat-validity-diagnosis-20261003/training/joint-validity-n2048-s17/step-0512.pt`.
Its directly verified SHA-256 is
`d0b2be9c6bf884b7635555d9a4dfc3ae0685660ebfe98fc522f9685e09a4b68d`.
The retained report records 53.94% novel balanced accuracy and zero of 512
complete novel groups; no checkpoint was promoted. This assay is development
evidence, not an official Wyat or Decision Index score.

The frozen adapter in `lab/tools/wyat_diagnosis/infer.py` binds model spec,
encoder recipe and source hashes. Inputs contain only `id`, `state`, `question`
and candidates `{id,text}`. Rendering is:

```text
state:  Instruct: {question}\nQuery: {state}
action: {candidate_text}
joint:  Instruct: {question}\nQuery: {state}\nDocument: {candidate_text}
```

Candidate IDs and other menu entries are not rendered into the joint pair.
The adapter returns independent logits, with zero as its assay threshold;
sigmoid values are uncalibrated outside that assay. Its archived loader uses
`torch.load(weights_only=False)`, acceptable only within its sealed experiment
trust boundary. Wyat should not expose that path as a general user-model loader.
Exporting tensor-only weights with a strict manifest is the necessary boundary
before admitting this format, even for diagnostic replay.

## Export requirements before admitting another recipe

Keep user-facing requests independent of head architecture, but make each
supported backend's obligations explicit. A deployable export must bind:

1. Format version, model family, input recipe and exact artifact identity.
2. Backbone/tokenizer revisions or verified local files; weight hashes and
   tensor-only head/adapters; exact expected names, shapes and dtypes.
3. Rendering/tokenization, padding, cap and overflow behavior; candidate
   identity/order/span rules; pooling, normalization and adapter parameters.
4. Score semantics: choice ranking versus independent validity, scale and
   normalization, tie policy, supported tasks and maximum candidates.
5. Reference inputs, exact token IDs/spans, raw logits and output identities;
   declared cross-device/precision tolerance and a reproducible capture owner.

Reject unknown recipes instead of inferring architecture from tensor names or
silently falling back. A model-independent `decide(request)` boundary can
support load, evaluation and HTTP serving without committing every future
artifact to candidate heads. The current Kai backend is one explicit recipe.
This document does not promise loading the research artifacts above.

## Evidence needed for speed and quality claims

Measure preprocessing, resident inference and output handling together on
declared hardware, precision, lengths, menu widths and question counts. Separate
model load, cold requests and warmed/cached requests; retain p50/p95, variation,
memory scope and failures. A head-only microbenchmark establishes head cost,
not complete model speed. A 20× claim requires a named vanilla baseline with
identical inputs and supported numerical/quality parity.

Evaluation keeps reference labels outside inference requests, validates labels
against submitted IDs, supports tied acceptable answers and retains failed
predictions in denominators. Native metrics, correlated-group uncertainty,
French results and reserved-confirmation exposure retain their original
protocols. See `/Users/teilomillet/Code/clm-lab/docs/evaluation-contract.md`.
No training, source modification or reserved-data access was performed by this
inspection.
