# Measure scaling before choosing an optimization

The pure helpers in `benchmarks.analysis.scaling` summarize performance
hypotheses; observations determine whether those hypotheses hold. They have
no accelerator or optimization-library dependency. DMBench continues to own
general benchmark quality schemas, imports and scoring.

## Amdahl and roofline

For a fixed workload, accelerating a measured fraction `f` by factor `s`
predicts whole-operation gain `1 / (1-f + f/s)`. Removing that fraction gives
ceiling `1/(1-f)`. `amdahl_speedup` and `amdahl_ceiling` expose those quantities;
the latter returns `None` for the unbounded `f=1` case. This assumes unaffected
work and no new coordination cost. A synchronized diagnostic profile changes
execution, so its fractions yield conditional diagnostic ceilings rather than
promised end-to-end gains. The fixed-workload assumption follows the Amdahl
discussion in [Gunther's primary paper](https://arxiv.org/html/0808.1431v2#S2.SS1).

`roofline_lower_bound` returns the maximum of arithmetic work divided by
compute capacity and traffic divided by bandwidth. Its counts must be
justified work/traffic lower bounds and its rates justified hardware upper
bounds for the relevant precision. Operand bytes, tensor sizes and advertised
bandwidth do not establish measured DRAM traffic or sustained bandwidth.
Without those inputs, the runner records a roofline result as unknown.
Measurements below a proposed bound challenge its assumptions. The model
comes from [Williams, Waterman and Patterson's primary Roofline paper](https://aiichironakano.github.io/cs596/Williams-Roofline-CACM09.pdf).

The backbone arithmetic helper distinguishes hidden width `D`, query width
`Q = query_heads * head_dim` and KV width `K = kv_heads * head_dim`. Per-layer
projection FLOPs are `2 * tokens * [D*(Q+2K) + Q*D + 3D*I]`, where `I` is the
MLP intermediate width. A multiply-add counts as two FLOPs. Dense attention
adds `4 * batch * query_heads * length² * head_dim` per layer; ideal causal
triangular arithmetic instead uses
`2 * batch * query_heads * length*(length+1) * head_dim`. A fused causal
attention kernel's actual schedule remains unknown. These counts exclude
normalization, RoPE, softmax, nonlinear activations and the decision head.

The real model has `D=1024`, `Q=2048`, `K=1024`, `I=3072` and 28 layers.
Equating query width with hidden width would materially undercount its work.
Each batch reports original token count, padded count, width and padding
fraction. Counts describe the observed batching policy rather than implying
that padding can be removed from arbitrary model kernels without verification.

## USL is a diagnostic hypothesis

The user's [Universal Scaling Law collection](https://teilo.xyz/collections/universal-scaling-law/)
gives `C(N)=N/[1+sigma*(N-1)+kappa*N*(N-1)]`, with `0<=sigma<1`, `kappa>=0`.
It requires warmed steady-state data at six or more distinct loads and treats
the coefficients as diagnostic hypotheses. Its linked
[Gunther paper](https://arxiv.org/html/0808.1431v2#S2.SS3) derives the rational
capacity model and its peak `sqrt((1-sigma)/kappa)` when `kappa>0`.

`fit_usl` requires six unique loads including `N=1`. It normalizes measured
throughput by the observed load-one value, then fits `N/C(N)-1` to the two
columns `N-1` and `N*(N-1)` using a two-variable least-squares solution.
Parameters must satisfy the physical domain before a curve or peak is
admitted. Invalid estimates remain visible; they are never clipped to make
superlinear or otherwise incompatible measurements appear to fit the model.

The receipt retains raw throughputs, transformed residuals, raw throughput
residuals when a physical curve is admitted, capacity RMSE/R², maximum relative
error and normalized design conditioning. The transform weights errors
unequally and shares baseline uncertainty across every normalized point.
Correlated parameter columns can weaken identifiability. The current field
`normalized_gram_condition` is the eigenvalue ratio of the unit-column Gram
matrix; the normalized design condition is its square root. A Gram condition
above 100 generates a warning. The original receipt called this quantity
`normalized_design_condition`; its stored value is retained and interpreted
as a Gram condition. No confidence interval or mechanistic identification
follows from a small fitted curve.

A peak outside the observed load interval is explicitly marked as
extrapolation. Physical parameters alone cannot establish a useful fit:
capacity R² below 0.9 or maximum relative throughput error above 10% generates
a weak-fit warning. Those thresholds are illustrative diagnostics, not
predeclared quality or promotion gates, and cannot change the fitted
coefficients. Raw residuals remain visible even when the parameters pass.

Batch size changes amortization, matrix shape, padding and resource use; it is
not the processor/load-generator count assumed by the physical USL model.
The batch curve is explicitly descriptive. A decline with larger batches does
not establish cache coherence. Even a CPU thread curve would need to account
for the machine's heterogeneous performance/efficiency cores and investigate
the underlying contention rather than assigning a physical cause from a fit.

## Frozen six-shape development experiment

The first sweep uses every original case in the fixed 77-row BANKING77
training-source development cohort, unchanged caller order and the frozen
CPU FP32 reference. Its source-byte hash is
`0020d9a3c72a21941ba551cf31f404187f69c6a5dc30299cc0a24d303b904a03`.
Reference identity, original choice IDs, token IDs, candidate endpoints and
query positions must match before measurement. Released-model training
exposure remains unknown; this is no official or held-out benchmark score.

The six batch sizes are `1,2,4,8,16,32`. FP32, eager MLX GPU execution and zero
option-token cache remain fixed. Each shape receives one complete warmup and
six measured cohort cycles. Cyclic order rotation places every shape in every
order position once. The runner accepts multiples of six repetitions to
preserve that ordering balance. Setup, warmup, JSON persistence and HTTP are
excluded from timings. Native scorers materialize host floats synchronously.

Every warmup and measured shape must preserve all selections and full
rankings, logit absolute/relative tolerance `1e-3`, probability tolerance
`1e-4`, no failures and no labeled accuracy decrease against the same frozen
reference. All per-case logits, probabilities, rankings, exceptions and
timings remain in the receipt; measured cycles are also flushed to JSONL as
they complete. Failed calls cannot count as useful equivalent throughput.
Successful, equivalent and independently labeled-correct throughputs remain
separate. Repetition repeats the same 77 cases rather than increasing the
number of independent quality observations.

```sh
python -m benchmarks.analysis.run_scaling .models/Decision-2.0-Kai-0.6B
```

Complete-cohort length sorting, exact padding widths, packed projections and
CPU threading are separate hypotheses. They require their own measured
comparison and the same strict quality boundary before any production change.

## Observed first sweep and retained anomalies

The immutable [complete receipt](../../benchmarks/analysis/results/mlx-scaling77.json)
contains six measured cycles per shape, all successful with every selection
and full ranking preserved against CPU FP32. Each cycle classified 67/77 of
the independent labels correctly. Repetitions do not create additional
independent examples. The archived
[executed runner](../../benchmarks/archive/2026-10-05/scaling/run_scaling.txt)
and [executed helper](../../benchmarks/archive/2026-10-05/scaling/scaling.txt)
match its recorded source hashes. Subsequent typing, shared-reference cleanup
and diagnostic warning changes do not revise the measured receipt.

| Batch | Overall successful requests/s | Cohort p50 ms | Cohort p95 ms | Batch residence p95 ms | Padding fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 13.94 | 5425.81 | 7484.21 | 84.02 | 1.67% |
| 2 | 11.41 | 3392.36 | 18767.63 | 158.15 | 3.83% |
| 4 | 23.71 | 3193.55 | 3584.96 | 263.95 | 5.22% |
| 8 | 25.12 | 3027.73 | 3358.76 | 431.99 | 7.82% |
| 16 | 24.32 | 3017.03 | 3564.90 | 926.78 | 10.15% |
| 32 | 25.53 | 2989.58 | 3142.49 | 1364.12 | 10.90% |

Residence describes an already-ready engine batch; it excludes any online
admission queue. Overall requests/s counts every timed cycle, including large
delays. Batch 32 is only 1.6% above batch 8 in overall throughput here and has
much higher batch residence. These observations do not support a general
batching recommendation before confirmation.

Batch 2's fifth measured round took 23817.93 ms; the slowest internal batch
took 18281.10 ms for `banking77-train-04562` and `banking77-train-01221`.
Batch 1's third round took 7999.86 ms; its slowest request,
`banking77-train-08638`, took 4295.68 ms. The former had no failures and 67
correct answers, as did its neighboring cycles. Cumulative process RSS stayed
2052882432 bytes before and after the cycle; that high-water measurement
cannot establish current device residency or diagnose a pause. This run did
not capture cycle wall timestamps, OS pressure or queue telemetry. The cause
of the delays is unknown, and no timing has been removed.

Later read-only system observations reported 15601.44 MB of swap in use out
of 21504 MB allocated and `memory_pressure -Q` free percentages of 16% then
26%. `pmset -g therm` reported no recorded warning; that does not establish
absence of throttling. These later observations cannot attribute the earlier
delays to pressure. Future confirmation should capture current swap, faults
and context switches around each cycle without changing unrelated apps.

The [posthoc sensitivity receipt](../../benchmarks/analysis/results/mlx-scaling77-sensitivity.json)
retains every original timing and binds the unchanged receipt and revised
analysis bytes by hash. The reciprocal-mean throughput fit has negative
`beta=-0.0000327805` and admits no physical USL curve. The reciprocal-median
fit has physical parameters but weak descriptive agreement: R² 0.8312,
maximum raw relative error 16.59%, and a predicted batch peak near 19.87.
Median timing describes a typical cycle rather than total capacity; changing
the summary statistic cannot accept the failed original fit or establish a
physical batching optimum. Gram condition is 55.0586, corresponding to design
condition 7.42015. Reproduce the separate analysis with:

```sh
python -m benchmarks.analysis.scaling_sensitivity \
  benchmarks/analysis/results/mlx-scaling77.json \
  benchmarks/analysis/results/mlx-scaling77-sensitivity.json
```

The [fresh five-pair batch-2/batch-8 confirmation](../../benchmarks/analysis/results/mlx-batch2-8-confirm77.json)
now preserves the same full cohort and strict FP32 quality boundary, with
all raw cycles and untimed system/device diagnostics. Median batch-8 speedup
is 1.165× with paired p05–p95 1.008–1.355× and one slower pair. Overall
requests/s is 22.65 versus 26.29. The default allocator retains 16.664 GB of
active-plus-cache allocation after continuous execution of these two shapes.
The [allocator study](cache.md) documents a separate controlled reset design
and its memory/throughput tradeoff. Neither experiment identifies the cause
of the original anomaly; the failed fit and all observations remain intact.
