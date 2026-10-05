# Measure inactive cache separately from active model memory

The experiment in `benchmarks.analysis.mlx.cache` tests whether bounding MLX's
inactive allocator cache reduces retained memory without changing real model
outputs. It compares the fresh process's original cache cap with 512 MiB and
1 GiB at one fixed engine batch size. It does not change production settings.

MLX's cache retains reusable inactive buffers. It is separate from prompt KV
state and Wyat's token-encoding cache. Active allocations hold the live model
and work; inactive buffers can remain allocated after that work completes.
`get_peak_memory()` reports peak active allocation, so active plus cache is
recorded separately. The cap is a reuse guideline: a freed buffer can cause a
small transient overshoot rather than a strict process-memory ceiling.
The [pinned allocator source](https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/allocator.cpp)
and [cache-limit API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
define these mechanics. The setter returns the previous cap; the installed
API has no corresponding getter.

The real workload remains all 77 pinned BANKING77 training-source development
cases, the same six choices in original order, the same FP32 model and the
same token IDs/endpoints checked against the frozen CPU reference. Cache zero
for token encoding, eager execution and GPU device remain fixed. Released
model exposure to these queries is unknown; this is no held-out quality score.

Each cap receives five paired comparisons against the original setting.
Pair order alternates; the second cap begins in reverse order so original
and capped arms occupy the first position five times each across ten pairs.
Before **every** measured sample, both arms receive the same treatment:

1. Synchronize, set the declared cap, clear inactive cache, reset active peak.
2. Run one full-cohort warmup and retain its outputs, timing and memory.
3. Reset active peak, record synchronized memory, process counters and swap.
4. Time one complete cohort with native host-float output synchronization.
5. Record memory, process counters and swap outside the timed call.

Loading, clearing, warmups, telemetry and JSON persistence are excluded from
measured cohort time. This controlled warm-per-sample design bounds cache
history; it does not represent uninterrupted serving with variable shapes.
It also cannot attribute the earlier scaling delays to inactive buffers.

Every warmup and measured cycle must preserve all selections and full ranks,
logits within absolute/relative tolerance `1e-3`, probabilities within `1e-4`,
no failures and no labeled accuracy decline. Raw outputs and diagnostics are
flushed to JSONL after every sample. Failed gates produce a nonzero CLI exit.
The original cap is restored in `finally`, including model-load/sample errors.
Failed calls are excluded from successful requests/s. Failed-quality arms
retain timings for diagnosis and cannot support equivalent-throughput claims.

The receipt records synchronized active, inactive cache, their sum, and active
peak. It also retains cumulative process faults, CPU times, context switches,
load averages and wall timestamps around warmups and measurements. These
counter brackets include untimed synchronization/snapshot collection and are
not pure GPU counters. `sysctl vm.swapusage` is a global untimed observation;
changes do not establish which process caused them. Exact source bytes are
archived before loading; setup-time disk bytes do not attest already-imported
code or external dependency execution.

```sh
python -m benchmarks.analysis.mlx.cache .models/Decision-2.0-Kai-0.6B
```

Fake-only regression tests check paired ordering, restoration on both model
load and sample failures, and CLI rejection of failed quality. They replace
allocator/model calls; importing the installed MLX module still requires a
Metal-enabled session.

## Observed resource tradeoff

The [batch-8 receipt](../../benchmarks/analysis/results/mlx-cache77-b8.json)
and [singleton receipt](../../benchmarks/analysis/results/mlx-cache77-b1.json)
both completed. Across the two studies, every one of 80 warmup/measured
cohorts preserved all selections and full rankings; each cohort classified
67/77 labels correctly. The largest logit/probability differences against
CPU FP32 were `1.5021e-5` / `3.0703e-6`. Both processes restored the measured
original cache cap of 24481313587 bytes before exiting.

| Engine batch | Inactive cache cap | End-of-cohort active + cache GB | Overall requests/s, original → capped | Median paired speedup | Paired p05–p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | 512 MiB | 9.317 → 2.926 | 27.53 → 27.33 | 0.9932 | 0.9907–0.9943 |
| 8 | 1 GiB | 9.317 → 3.464 | 27.51 → 27.20 | 0.9903 | 0.9841–0.9910 |
| 1 | 512 MiB | 6.817 → 2.926 | 23.24 → 22.79 | 0.9804 | 0.9784–0.9831 |
| 1 | 1 GiB | 6.817 → 3.464 | 23.19 → 22.97 | 0.9902 | 0.9896–0.9922 |

GB uses decimal bytes; configured MiB/GiB use powers of 1024. Active allocation
remained 2.388 GB in all end-of-cohort snapshots. Peak active was 3.847 GB at
batch 8 and 3.018 GB at batch 1, essentially unchanged by the cap. Inactive
cache accounted for the observed allocation reduction: default end-of-cohort
cache was 6.928 GB and 4.429 GB respectively. The bounded cache sometimes
exceeded its nominal cap slightly, consistent with allocator source mechanics.
These snapshots describe MLX allocation accounting, not process RSS or physical
page residency.

512 MiB reduced end-of-cohort active-plus-cache bytes by 68.6% at batch 8 and
57.1% at batch 1, with approximately 0.7% and 1.9% lower overall throughput.
The observed benefit is a memory tradeoff, not faster inference. Five pairs
are insufficient to estimate rare-tail stability. No large stalls occurred
in these controlled-reset samples and every measured process major-fault
delta was zero; neither observation establishes the cause of historical
outliers or guarantees behavior under uninterrupted serving.

For singleton reproduction, use a separate output:

```sh
python -m benchmarks.analysis.mlx.cache .models/Decision-2.0-Kai-0.6B \
  --batch-size 1 --output benchmarks/analysis/results/mlx-cache77-b1.json
```

The separate `benchmarks.analysis.mlx.confirm_batches` experiment uses a fresh
process with unchanged default allocator behavior, one initial warm per shape,
and five alternating batch-2/batch-8 pairs. It never changes the cache cap or
clears between arms. This avoids treating the cache study's per-sample reset
design as evidence for ordinary continuous batching. Its two-shape history
still cannot reproduce the earlier six-shape workload exactly.

The [fresh continuous-default confirmation](../../benchmarks/analysis/results/mlx-batch2-8-confirm77.json)
preserved every output/ranking in both warmups and all ten measured cohorts,
with 67/77 labeled answers correct each time. Batch 8's median paired speedup
over batch 2 was 1.165×, paired p05–p95 1.008–1.355×; one of five pairs was
slower at 0.991×. Overall requests/s was 22.65 versus 26.29. Those five pairs
support batch 8 throughput for this ready cohort while retaining substantial
variation, not an online serving latency recommendation.

Both shapes finished with 14.276 GB inactive cache plus 2.388 GB active,
16.664 GB total allocator accounting. This differs from cleared/warmed batch
8's 6.928 GB inactive cache: loading and continuous variable-shape execution
history matter to allocator retention. Two measured cycles had process
major-fault deltas of one; other measured cycles had zero. They do not
establish a cause for delay. These observations make a controlled bounded
cache experiment with continuous variable-shape history a useful next
hypothesis, without revising the original six-shape evidence.

```sh
python -m benchmarks.analysis.mlx.confirm_batches .models/Decision-2.0-Kai-0.6B
```
