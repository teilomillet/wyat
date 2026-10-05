# Padding and ready-cohort scheduling

This experiment uses the same pinned Kai weights and 77 training-source requests
as the [stage profile](mlx.md). Labels are outside scoring, and model exposure to
these examples is unknown. All comparisons retain FP32 weights/activations,
original candidate order, endpoint positions, and the existing strict quality
gates. No production packing or scheduling policy changes.

The [runner](../../benchmarks/analysis/mlx/padding.py) compares ordinary
pad-to-eight caller-order microbatches with exact widths, stable length grouping,
and both together. Microbatch size is eight. Grouping includes an additional
encoding pass to obtain actual lengths; its cost is inside timing. Results are
restored to caller order, including duplicate requests. Hypothesis tests use an
independent scheduler oracle to challenge that restoration.

Length grouping reduces padded projection tokens from 14,840 to 14,072 on this
cohort: an arithmetic work ratio of 1.0546. Dense attention work has ratio 1.1107.
These are work estimates, not timing predictions. Removing padding can change
kernel shapes, tile utilization, compilation, and allocator reuse.

Each arm passed the CPU-reference screen, then five alternating warmed complete
cohort pairs. All 385 repeated selections and full rankings matched its control;
each cycle retained 67/77 independently labeled-correct predictions. Raw logits,
probabilities, timings and failures are in the [immutable receipt](../../benchmarks/analysis/results/mlx-padding77-b8.json).
Its runner hash matches the [executed source snapshot](../../benchmarks/analysis/results/mlx-padding77-b8.executed.txt).

| Arm | Median baseline/candidate speed | Paired p05–p95 |
| --- | ---: | ---: |
| Exact width | 1.0365× | 0.8489–1.0494× |
| Length grouping | 1.0534× | 0.7540–1.3098× |
| Grouping plus exact width | 0.9665× | 0.8217–1.0435× |

The individual timings are unstable, and the combined candidate is slower at
the median. These observations do not support a production speedup claim or
policy change. Paired percentiles describe the five observed ratios; they are
not confidence intervals. The profiler and earlier scaling assay also contain
material timing variation, prompting a separate allocator-cache hypothesis.

Outer benchmark calls contain all 77 ready requests. Their row residence metric
is complete-cohort call residence, rather than internal microbatch residence or
online admission/queue latency. Loading, warmups and output serialization are
outside timed cycles. Host logits are materialized before the scorer returns.
Per-cycle allocator/cache and fault telemetry was not collected in this assay;
the later [system observation](../../benchmarks/analysis/results/system-conditions.json)
cannot establish the cause of its earlier delays.
