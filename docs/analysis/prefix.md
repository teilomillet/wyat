# Explicit native MLX prefix reuse

`PrefixScorer` prepares one exact immutable token prefix eagerly for a loaded Kai
model. Matching requests reuse its fixed K/V state and evaluate the final
Transformer block only at the candidate/query readouts. It implements the normal
Wyat scorer contract, so `Model` and `create_app` need no separate integration.
Existing loading and CLI defaults retain full prefill.
The full server example requires the `mlx` and `server` extras; from this
checkout, install them with `uv pip install -e '.[mlx,server]'`.

```python
from wyat import Candidate, Model, Request
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.backends.kai.prefix import PrefixScorer
from wyat.server import create_app

base = MLXKaiScorer.from_directory("/models/kai", device="gpu", compile=False)
example = Request(
    context="Classify customer banking support requests into the supplied intents.",
    query="I have a missing refund",
    candidates=(Candidate("refund", "Missing refund"), Candidate("card", "Card problem")),
)
exact_prefix = base.encode_request(example).ids[:25]
scorer = PrefixScorer(base, exact_prefix)
model = Model(scorer, scorer.model_id)
decision = model.decide(example)
app = create_app(model)
```

The prefix length here is an explicit application choice, not an automatic
25-token policy. Choose token IDs shared by the intended requests, produced by
this model's exact encoding recipe. A text prefix alone does not establish the
same token prefix. The constructor requires an exact tuple of valid integer IDs,
nonempty and shorter than the encoder's maximum length. Compiled bases and
scaled RoPE are rejected before cache construction; this path has evidence only
for eager inference with the released unscaled Kai contract.

The per-batch array-offset RoPE API is present in the declared minimum
[MLX 0.30.0 source](https://github.com/ml-explore/mlx/blob/v0.30.0/python/src/fast.cpp),
and [MLX-LM 0.30.0](https://github.com/ml-explore/mlx-lm/blob/v0.30.0/mlx_lm/models/rope_utils.py)
selects that native RoPE implementation for the unscaled model. This source check
establishes API presence, not older-version numerical or performance results.
The recorded measurements used MLX 0.32.3 and MLX-LM 0.31.3.

Keep the base weights, decision head, device and encoder fixed for the resource's
lifetime. Create a new resource after replacing any of them. The `prefix` and
base `compiled` properties are read-only. Preparation finishes before the
constructor returns; its cost belongs to setup. `prefix_state_bytes` reports
logical K/V bytes, while views may retain additional backing allocation. The
scorer never changes the process-wide MLX allocator limits.

Each admitted batch gets fresh cache owners from the immutable prefix snapshots.
Every suffix token remains in K/V, with original RoPE positions and an explicit
absolute causal mask for selected final queries. Candidate/query positions are
translated once; full right padding and caller/candidate order are preserved.
If any request differs from the exact prefix or has a readout inside it, the
whole batch uses the original base scorer. Admission encoding and fallback
encoding both count toward that call's work. No suffix state survives between
requests. `create_app` continues to serialize active inference for the shared
model and encoder.

The independently audited experimental combined path preserved the frozen CPU
rankings, selections and numerical tolerances on all 77 development requests.
Five alternating pairs measured median speedups of 1.10707× at B8 and 1.02556×
at B1, clearing the declared 2% gate at both loads. Preparation took about 33 ms
and retained 5,734,400 logical K/V bytes. The prefix contains that cohort's fixed
banking context; these measurements establish neither a universal gain nor
held-out model quality. B8 peak active allocation grew from about 3.853 to
4.059 GB; B1 peak active allocation fell from about 3.025 to 2.90 GB. Peak active
allocation is distinct from logical prefix size and cached inactive allocation.
Raw provenance and limits are in [the MLX analysis](mlx.md). Final production
source confirmation is recorded separately from these original experimental
receipts.

The final production source passed a fresh full-cohort confirmation at both
loads, with the same 512 MiB allocator cap, strict CPU numerical/rank/selection
gate and five alternating warmed pairs. Each load preserved all 924 compared
warm/measured rankings and selections, with accuracy unchanged at 67/77 per
cycle. Maximum absolute logit/probability differences were 1.609e-5/4.615e-6
at B8 and 1.889e-5/3.936e-6 at B1.

| Load | Median paired speedup | Sample p05 | Baseline/cached cohort p50 |
| --- | --- | --- | --- |
| B8 | 1.11334× | 1.10827× | 2834.87/2546.27 ms |
| B1 | 1.02620× | 1.02036× | 3383.95/3301.33 ms |

Both clear the original median >=1.02× and sample p05 >1× gate. B1 retains an
observed pair of 1.01973×; the gate uses the declared median and percentile,
and does not require every pair to exceed 2%. Preparation cost 32.63/33.57 ms
at B8/B1 is separate from timed calls. Final useful throughput was
27.166/30.321 requests/s for baseline/cached B8 and 22.758/23.330 at B1.
The same shape-dependent peak active allocation tradeoff remained: about
3.853/4.059 GB at B8 and 3.025/2.900 GB at B1. These are workload-specific
observations, not a guarantee for another prefix, model or workload.

The new receipts are `benchmarks/analysis/results/mlx-prefix-production77-b8.json`
and `mlx-prefix-production77-b1.json`, with exact source archives, ten-cycle
journals and parent `.wall.json` receipts. Both parent/child pairs exited zero
without their 180-second timeout; the Mac resources were released afterward.
The original `mlx-combined77-*` observations and archives remain unchanged.
Sample percentiles from five pairs are descriptive, not confidence bounds.
