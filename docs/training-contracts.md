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

## Native typed task conformance

The research-native adapter format is separate from the released Kai bundle.
`wyat.tasks` is the canonical pure parser, segmented encoder and typed interpreter
for `wyat-native-typed-segmented-independent-questions-v1`. See
[task meanings](decision-tasks.md) for the exact public record, Choice/Score/Noul
readouts, FP32 conversion and tolerance. `wyat.load()` remains the Kai Choice
loader; do not change `production_kai_compatible: false` on native exports.

The intended launch binds `artifacts/typed-task-conformance-20261008/CONTRACT.json`
and `REFERENCE-FIXTURES.json` prospectively alongside immutable native/loss
sources, initializer/tokenizer identities, task exposure and resource policy.
Labels and review/supervision metadata stay outside model-facing records.
Synthetic serializer/readout fixtures establish mechanics, not learned quality.
The launcher must verify actual exported-adapter execution through its accepted
loader, including the declared dtype and numerical tolerances; that physical
check is not supplied by the pure module. Typed benchmarks consume the same
interpretation owner and report task-specific metrics and failure coverage.
Historical recipes and their existing receipts remain immutable.

## Strict backward reproducibility for new native training

### Context for v6 and future native training

As requested on 12 October 2026, new native training uses a 32,768-token complete
question limit and a 32,768-token padded microbatch budget. The canonical policy
is `scripts.training_optimization.context`; both the decision training preparer
and optimization training worker consume it. Logical question batches and
microbatch-one optimization remain unchanged. Inputs are never truncated to fit.

The prospective initializer is
`artifacts/training-context-32768-20261012/initializer`. Create it with
`python -m scripts.training_optimization.context`: the builder authenticates and
copies the old initializer, changes only the new manifest context declaration,
and records lineage and unchanged file hashes. It preserves the original package.
New token censuses use this initializer by default; new preparation rejects an
8k census. Bind the new initializer, census, readiness and source identities into
fresh reviewed freezes. Adapter exports inherit both limits and restore them on
reload. Existing checkpoints, historical sources/freezes and the running v5.2
evaluation retain their original limits.

32k is an admitted input ceiling, not padding every input to 32k, a change to the
token exposure budget, or evidence of long-context quality. Existing compute,
memory and ownership limits remain enforced. Profile and smoke the actual new
data before training; exercise a full-length input with measured memory and
finite forward/backward behavior before claiming full-length training support.
Do not silently fall back to 8k after an out-of-memory failure. Long-context
quality requires separate measured evaluation.

For v6 and future cached-source reconsideration, use
`python -m scripts.french_data.training_data_selection --output <fresh-directory>`.
This uses the same context policy and tokenizer package, including full retained
MultiEURLEX TRAIN documents by default, rather than restricting discovery to the
old token-valid bank. The pinned constructor preserves source/ancestry exclusions,
duplicate checks and qualification holds. Each run reports the 8k/32k length
bands and public-input differences against the existing v6 candidate. Those
differences are candidate coverage, not semantic novelty or training admission.
Use fresh candidate, census, rights/quality and custody evidence before merging
newly recovered examples. Historical exclusions are preserved as evidence;
only the length condition is reconsidered. No periodic collection or training
is implicitly launched by this command.

New freezes produced by `scripts.french_data.decision_training_prepare`, including
future combined-v6 and successor datasets, require strict deterministic Torch
algorithms (`warn_only=False`). The supervisor sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the worker environment before Torch/CUDA
initialization. The worker applies the policy before model loading or CUDA work
and records the observed settings in its receipt. Missing or weakened policies
fail validation. Unsupported deterministic operations fail execution rather than
silently falling back. The existing explicit attention backend remains pinned;
CUDA preparation continues to select Flash SDPA.

This carries forward the successful backward reproducibility fix from
`artifacts/physicslm-retention-deterministic-20261012`. It does not enable retention
loss by default: that comparison failed the retained quality gates. Exact replay
was established for that pinned environment; each new runtime/model still needs
fresh profile, smoke and reproducibility evidence. Seeds, resource budgets,
exclusive GPU ownership, data admission, independent freeze review and retention
quality gates remain required. Historical freezes and receipts remain immutable;
new launcher source pins and reviews must be generated for future runs. The
separate optimization launcher's explicit deterministic math-attention policy
retains its own contract.

### Opt-in padded-token microbatches

CUDA qualification on 2026-10-14 rejected packed BF16 execution: all five
synthetic native cases failed the unchanged optimizer-update parity gates.
Separating question backwards did not repair that difference. The opt-in CUDA
path therefore validates the requested padded groups, then executes singleton
forward/backward pairs. Its `microbatch_execution` receipt records the requested
and executed sizes and `cuda-singleton-parity-fallback` policy. Native GPU checks
establish fallback parity only; they do not qualify accelerated CUDA packing.
CPU packing and the omitted-field singleton default retain their existing paths.

For a new single-view native run, add `max_microbatch_padded_tokens = 4096`
under `[optimizer]`. Keep `microbatch_questions = 1`; it selects the historical
singleton policy when the token field is omitted. No existing run definition
opts in automatically. Baseline and online permutation modes are supported;
paired modes are rejected because their view accounting is not a native forward
shape.

The candidate groups consecutive questions without changing their order. A
multi-question group must satisfy `count * longest_encoded_length <= 4096`.
Longer questions run as complete singletons, subject to the native model's hard
`max_length` and `max_batch_tokens` limits. This is padded batching, with no
sequence concatenation or truncation. Input validation precedes gradient reset.

Losses retain the whole logical batch's weight denominator. Gradient clipping,
AdamW, scheduler exposure, and moment policy still advance once per logical
update, including partial batches. Batched floating-point operations can differ
from singleton operations; numerical parity must be measured, not assumed.
New source manifests and prospective run freezes are required. Historical
freezes and receipts are not rewritten to authorize the candidate.

Use `benchmarks/analysis/token_microbatch.py --help` for bounded native timing,
parity, and memory checks. Synthetic shape results do not establish performance
on the allocation distribution. Default promotion requires target CUDA BF16 /
Flash measurements, acceptable loss/gradient/parameter/optimizer-state drift,
and repeated updates within the run's peak allocated and reserved memory cap.
