# Explicit CPU prefix reuse: final-source confirmation

The hypothesis is that exposing the previously tested retained-prefix recipe as
an explicit CPU scorer preserves its numerical behavior and saves at least 2%
of complete inference time at batch one and batch eight. The earlier experimental
results support this test; they do not validate newly moved production code.
Default loading and CLI behavior retain full prefill. The shared tensor core
preserves the ordinary forward arguments and CUDA precision policy.

The programmatic entry point is
`wyat.backends.kai.cpu.prefix.PrefixScorer`; install the `inference` extra for
CPU use and the `server` extra when serving it. It implements the existing
scorer contract:

```python
from wyat import Candidate, Model, Request
from wyat.backends.kai.cpu.prefix import PrefixScorer
from wyat.backends.kai.torch import KaiScorer
from wyat.server import create_app

base = KaiScorer.from_directory("/models/kai", device="cpu")
example = Request(
    context="Classify customer banking support requests into the supplied intents.",
    query="I have a missing refund",
    candidates=(Candidate("refund", "Missing refund"), Candidate("card", "Card problem")),
)
scorer = PrefixScorer(base, base.encode_request(example).ids[:25])
model = Model(scorer, scorer.model_id)
decision = model.decide(example)
app = create_app(model)
```

Choose the exact token prefix shared by the intended inputs; 25 is an explicit
choice in this example, not an automatic policy. Matching text alone does not
establish matching tokens. This CPU path retains the complete final block.

The candidate owns one eagerly prepared immutable token prefix for a fixed
CPU FP32, eval-mode, full-attention Kai model with default unscaled RoPE and SDPA.
Every call gets fresh maintained cache owners; suffix state does not survive the
call. Original complete inputs are encoded and admitted before prefix removal.
A mismatch, empty suffix or readout inside the prefix falls back for the entire
batch. Keep weights, head, configuration, device and encoder fixed for the
resource lifetime. There is no automatic prefix policy or process-wide precision,
thread or attention-policy change in the scorer.
Ambient CPU autocast and backbone forward hooks are outside this resource's
equivalence contract and are rejected. The constructor checks the maintained cache's
capabilities before prefix computation. Required cache and position APIs are
present in the declared minimum
[Transformers 5.17 cache implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/cache_utils.py)
and [Qwen3 forward](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3/modeling_qwen3.py).
Source-level API presence does not establish numerical or performance results on
every allowed runtime version.

## Prospective protocol

Use the existing pinned Kai release and all 77 ordered training-source BANKING77
development menus, with their original six choices, tokens and readout positions.
The dataset SHA256 is
`0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03`;
the legacy CPU reference SHA256 is
`b8a90b31941b60e793357a8cb360767520da70ec33d2161c1242606962b22664`.
The ordered input digest is
`fd5dd942250c6416c4ab50c59a5164a20909d11ead12bf83c505efff6a5556e8`.
Reserved confirmation remains closed; prior model exposure is unknown.

Run two separate fresh native Mac processes, B8 and B1, each under the existing
180-second owned-worker cap. Use eight intraop threads, FP32 CPU SDPA,
max_length=512, token_cache_size=0 and the exact common 25-token prefix.
Retain installed runtime versions, Torch build and unchanged native HF masked-GQA
eligibility flags. The flags are a condition for this performance comparison,
not a universal correctness requirement for CPU prefix caching.

Within each process, share one freshly loaded base between full prefill and the
final prefix scorer. Run a screen and warmup for each arm, then five measured
pairs in AB, BA, AB, BA, AB order. Retain all fourteen chronological cohorts,
1,078 outcomes per shape, including failures and partial histories. B8 retains
nine batches of eight and the final five; B1 retains all 77 scalar calls.

Compare every output with the frozen legacy CPU reference and measured paired
arms using the existing gates: logits satisfy the existing 1e-3 absolute/relative
comparison, probabilities differ by at most 1e-4, every full rank and selection
agrees, and no failure, missing prediction or labeled-accuracy loss occurs.
No tolerance changes follow an observed result.

Use complete `_run` cohort latency, including per-call cache construction,
packing, computation and host output materialization. Preparation is measured
separately. Source hashing, diagnostics, journals and profiling stay outside the
timed body; no intrusive memory sampler is introduced. All four existing speed
gates must pass at both shapes:

- paired median speedup >= 1.02;
- sample p05 paired speedup > 1;
- aggregate baseline/candidate time ratio > 1;
- candidate/baseline cohort p95 time ratio <= 1.

Freeze final source and relevant installed dependency source before execution;
check stability outside each timed cohort and afterward. Validate historical
receipts against their immutable executed archives, rather than requiring the
current checkout to remain identical to old code. Independently reconstruct
raw outcomes, scheduling, quality and performance metrics before promotion.

The connected first-eight, full77 and memory-history admissions now check each
archived source's text hash and complete inventory against its pinned execution
receipt. Their original receipt, journal, wall, review and archive hash constants
remain unchanged. They no longer read live Python files to identify an old run.
When those tools execute a new experiment, they separately archive its current
source. Focused tests challenge the boundary by forbidding live source reads
during historical admission; all numerical and performance rules retain their
original meaning.

## Observed final-source results

The first-eight screen passed 48 outputs, followed by separate full77 B8 and B1
confirmations. Independent standard-library audits reconstructed all 2,204 raw
outcomes, complete ranks, numerical gates and chronological pairs, and validated
source archives and terminal worker receipts. The audits validated frozen
reference token vectors, prefix/readout geometry and recorded input lengths;
exact fresh runtime encoding was attested by the source-frozen runner rather
than independently retokenized. Every full cohort retains 67/77
correct selections, with no missing prediction or failure. The screen carries
no performance verdict.

| Batch | Paired median speedup | Paired p05 | Aggregate speedup | Candidate/baseline cohort p95 |
| --- | --- | --- | --- | --- |
| 8 | 1.10038x | 1.09290x | 1.09870x | 0.91436 |
| 1 | 1.07260x | 1.01150x | 1.06265x | 0.97340 |

All four predeclared performance gates pass at both shapes. The five B8 pair
ratios are 1.09220, 1.09573, 1.10115, 1.10038 and 1.10415. The B1 ratios are
1.07794, 1.06546, 1.07260, **0.99801** and 1.10493: one scalar pair regresses
about 0.20%. The interpolated sample p05 passes the original gate; it is not a
confidence bound or a claim that every future pair improves. These complete
cohort timings include fresh cache owners, packing, model computation and host
output materialization.

Maximum logit/probability drift is 9.835e-6/1.543e-6 at B8 and
9.954e-6/1.560e-6 at B1, within the unchanged strict gates. Prefix construction
took 69.968 ms and 70.082 ms respectively, separately from inference. Both
resources retain 5,734,400 logical prefix bytes; extended and temporary K/V
allocations are additional. The first-eight, B8 and B1 workers exited zero and
were reaped in 7.964, 83.275 and 135.724 seconds under separate 180-second caps.
Thread settings were restored before each successful completion was recorded.

The [first-eight audit](../../../benchmarks/analysis/results/torch-prefix-production-first8-review.json),
[B8 audit](../../../benchmarks/analysis/results/torch-prefix-production77-b8-review.json)
and [B1 audit](../../../benchmarks/analysis/results/torch-prefix-production77-b1-review.json)
retain exact raw receipt hashes and independent calculations. All three runs
share executed source archive SHA256
`22aa511fe5434b21aa6b6d45c0d6cbe42ef71473d53dc0ceddec202bd0e9e409`.
Those immutable archives describe the executed code and protocol; this summary
was updated after the audits.

## Connected checks and limits

The complete native CPU/Metal and actual DMBench regression passed 971 tests,
zero skips and one existing Starlette/httpx deprecation warning. Strict mypy
passed on 146 maintained sources; Ruff lint and formatting passed. The owned
pytest child exited and was reaped in 28.669 seconds under its 120-second cap.
All source bytes stayed stable. The initial full check found a duplicate test
basename at collection; the unchanged test was renamed, and that failed receipt
remains alongside the successful
[final regression](../../../benchmarks/real/engine-cpu-prefix-production-verification-final.json).
Current-source native CUDA execution is unmeasured; the existing CUDA receipts
describe their earlier frozen runtime.
These observations do not establish current peak RSS,
an allocation ceiling or a benefit for another prefix or workload. No HTTP
capacity, online latency, GPU gain, 20x speedup or held-out quality follows from
this assay. The existing scorer interface permits an explicitly constructed
resource to be used by `Model` and the server without a separate transport path.
