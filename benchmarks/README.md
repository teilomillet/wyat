# Benchmarks and retained evidence

For user-facing measurement, use `wyat benchmark` with your local artifact and
labeled JSONL cases; start with the [measurement guide](../docs/real-benchmark.md).
The runtime implementation lives in `wyat/`. The modules here are research and
verification tools, not additional production backends.

| Directory | Purpose |
| --- | --- |
| `fixtures/` | Seeded tiny models and tokenizers for offline mechanics checks |
| `tiny/` | Synthetic inference experiments; no trained-model quality claim |
| `real/` | Released Kai references, canonical development cases, DMBench tools and verification receipts |
| `analysis/` | Scoped optimization hypotheses and experimental implementations |
| `analysis/results/` | Immutable raw results, independent audits and executed-source snapshots |

Read the [documentation index](../docs/README.md) for user guides and the
[contributor guide](../CONTRIBUTING.md) for maintained checks and optional runtime
requirements. Full research tooling and hash-bound evidence belong to a source
checkout; they are not model weights or public runtime APIs.

## Where to start

- [Current native CPU/MLX configuration](../docs/analysis/runtime/native.md):
  existing load/serve/benchmark options, a bounded full-stock comparison and its
  limits. The observed 2.84x ratio compares MLX GPU with Torch CPU on one host.
- [Opt-in CPU retained prefix](../docs/analysis/cpu/prefix.md) and
  [MLX retained prefix](../docs/analysis/prefix.md): supported opt-in resource
  ownership, fallback behavior, setup cost and final-source measurements.
- [Investigation plan](../docs/analysis/plan.md): accepted, rejected and unknown
  engine hypotheses, with links to exact raw evidence and source archives.
- [DMBench integration](../docs/dmbench-integration.md): public benchmark
  contracts remain distinct from engine timing and parity.

The real development cohort has 77 fixed six-choice BANKING77 training-source
menus. Repetitions do not create independent labeled observations, and the
checkpoint's prior exposure is unknown. These measurements do not establish
held-out quality, HTTP throughput or an unknown future Wyat architecture.

Preserve existing receipt, manifest, audit and archived-source bytes. A
separately authorized experiment uses a fresh output stem. Do not rerun a
historical command into its retained destination or treat an experimental
implementation as deployed behavior. Raw checkpoint and profiler captures stay
local under the declared Git/distribution exclusions.

## Tiny offline mechanics

`tiny/paired.py` measures a seeded, one-layer Qwen3 fixture with hidden width 16,
head width 8 and a local WordLevel tokenizer. Its weights are random. The result
tests mechanics and optimization hypotheses; it establishes no learned quality
and no speedup for the released 0.6B checkpoint or a future Wyat checkpoint.
The workload uses equal-length prompts and two reused options; unknown synthetic
query words map to the fixture tokenizer's unknown token. It is a controlled
batching workload rather than a representative production text distribution.

Run from this checkout after installing the inference and development dependencies:

```sh
trial_dir="$(mktemp -d /tmp/wyat-tiny.XXXXXX)"
uv run --extra inference python -m benchmarks.tiny.paired \
  --output "$trial_dir/result.json"
```

The fresh temporary directory protects the retained historical receipts. This
script writes its requested destination directly; choose a new destination when
preserving another run. The command exercises a tiny random model without
downloading pretrained weights.

Both comparators use identical weights, FP32 CPU inference, SDPA and one PyTorch
thread. Loading, three warmup cycles, HTTP transport and JSON serialization are
excluded. Each measurement contains 25 paired cycles; comparator order alternates.
Option caching is warm and suffix pretokenization is shared by both paths. Every
raw cycle sample, environment, fixture identity and equivalence result is saved.
The engine timings include tokenization, backbone, head, logit validation, ranking
and relative softmax probabilities through `Model.decide` or `Model.decide_many`.

On the recorded macOS ARM host, the two eight-request runs measured option
tokenization at 2.59x, engine caching at 1.031x and 1.035x, and batch-eight engine
inference at 5.88x and 5.75x versus eight singleton calls. A roughly three-percent
engine difference on this fixture does not justify enabling caching for an unknown
target workload. It remains an opt-in setting. The two complete receipts are
`tiny-cpu.json` and `tiny-cpu-repeat2.json`.

The separate batch sweep in `tiny-cpu-batch-sweep.json` measured:

| Ready requests | Batch cycle p50 (ms) | Batch cycle p95 (ms) | Requests/s | Speedup vs singleton cycle |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0.937 | 1.012 | 2,135 | 1.92x |
| 4 | 1.061 | 1.273 | 3,771 | 3.41x |
| 8 | 1.254 | 1.428 | 6,381 | 5.72x |
| 16 | 1.648 | 1.774 | 9,711 | 8.75x |
| 32 | 2.430 | 2.639 | 13,167 | 11.88x |
| 64 | 3.989 | 4.222 | 16,046 | 14.56x |

These are complete-cycle distributions with all requests already available;
they exclude time waiting to form a batch. Throughput is ready requests divided
by cycle p50. They are not HTTP tail latencies or an automatic batching policy.
The two batch-eight repetitions and sweep also show run-to-run variation rather
than a single best result.

Cached token IDs, endpoint positions and singleton logits matched exactly.
Batching retained selected candidates and stayed within `rtol=1e-5, atol=1e-5`;
the eight-request maximum absolute logit difference was `4.47e-8`. The tests
independently exercise the upstream head formula and segmented prompt recipe.
The synthetic tokenizer does not establish parity at actual pretrained BPE
segment boundaries. These tiny receipts do not measure full-checkpoint behavior or trained quality.
The later [real-data protocol](../docs/real-benchmark.md),
[CPU/CUDA](../docs/torch-inference.md), [MLX](../docs/mlx-inference.md) and
[implementation receipt](../docs/implementation-evidence.md) describe released
Kai measurements separately. No 20x claim follows from the tiny evidence.
