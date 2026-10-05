# Selective projection precision: prospective protocol

The current MLX control remains eager full-prefill FP32 with the pinned Kai
weights, segmented tokenizer and candidate head. Whole-backbone BF16 and FP16
previously failed strict output gates. This experiment asks whether limiting
reduced precision to particular MLP projection roles avoids that failure.
No production precision policy changes.

## Eligibility before inference

All 196 serialized projection tensors are BF16. BF16's significand precision
fits FP16's, but its exponent range and subnormal grid do not. For every selected
tensor, first measure finite FP32 → FP16 → FP32 equality on its actual values.
Retain names, shapes, element counts, changed values, maximum errors and
overflow/underflow observations. Derive arm eligibility from its actual MLP
projection set; unrelated attention tensors cannot reject an MLP arm.

A failed exact-storage guard rejects that declared arm before inference. It
does not justify silently relabeling lossy weights as lossless or loosening the
guard. An approximation with changed weights would need a separate prospective
contract. This check can decisively reject the family without loading a model.

## Fixed arm family

| Arm | Selected MLP products |
| --- | --- |
| All down | Down projection in all 28 layers |
| All gate/up | Gate and up projections in all 28 layers |
| All MLP | All three projections in all 28 layers |
| First four | All three projections in layers 0–3 |
| Last four | All three projections in layers 24–27 |
| Last one | All three projections in layer 27; diagnostic only |

For an eligible arm, selected products use maintained FP16 inputs and weights
and restore outputs to FP32. SwiGLU, embeddings, attention, norms, residuals and
the head stay FP32. Input and output rounding remain intentional numerical
changes even with exact stored weights; FP32 accumulation cannot undo them.
The single-layer Amdahl ceiling assumes uniform costs and fenced fractions,
so diagnostic status is a declared budget decision, not a proof of impossibility.

## Evidence and acceptance

Set `MLX_ENABLE_TF32=0` before framework import. Freeze source, parameter and
input identities. Keep the same 512 MiB allocator cap and token-cache-zero
conditions. Screen every eligible arm on all 77 frozen training-source cases at
batch eight and scalar. Preserve every selection and full ranking, no failures
or labeled-accuracy loss, the existing 1e-3 absolute/relative logit rule, and
maximum probability drift 1e-4. Retain failed and ineligible arms.

Only numerical survivors can enter five alternating warmed timing pairs.
Production consideration requires median speedup at least 1.02 and paired
sample p05 above one at both shapes, plus a challenge on the frozen longer,
variable-menu native DMBench cases. These development screens do not establish
held-out quality. Separate bounded parent processes retain timeout/failure
receipts. A changed projection recipe requires freshly built prefix state;
existing FP32 prefixes must never be reused across changed arithmetic.

## Observed eligibility result

The bounded CPU-only scan completed in 2.122 seconds; its parent exited in
2.228 seconds without timeout. All 84 selected MLP tensors failed exact FP16
storage eligibility. Across 264,241,152 elements, 43,137 values changed and 224
rounded to zero. No source nonfinite values, FP16 overflows or BF16-source
subnormals were found. The largest finite absolute change was
`2.9802322387695312e-8`. These failures arise in the FP16 subnormal grid despite
finite, normal BF16 source values. Small absolute errors do not waive the
prospective exact-storage contract.

| Declared arm | Selected elements | Ineligible selected tensors | Result |
| --- | ---: | ---: | --- |
| All down | 88,080,384 | 28 / 28 | Rejected before inference |
| All gate/up | 176,160,768 | 56 / 56 | Rejected before inference |
| All MLP | 264,241,152 | 84 / 84 | Rejected before inference |
| First four | 37,748,736 | 12 / 12 | Rejected before inference |
| Last four | 37,748,736 | 12 / 12 | Rejected before inference |
| Last one, diagnostic | 9,437,184 | 3 / 3 | Rejected before inference |

The [raw receipt](../../benchmarks/analysis/results/mlx-mlp-fp16-eligibility.json)
retains every tensor and mechanically derived arm outcome. The
[parent receipt](../../benchmarks/analysis/results/mlx-mlp-fp16-eligibility.wall.json)
records exit 1 as a completed negative eligibility result, distinct from an
incomplete error or timeout. Checkpoint SHA256 before and after scanning is
`43369a2bfddb61495beb7cea6bf30d7564a5c5caff31f4dbf2e6cfce48423b69`;
the validated pinned Kai identity also matches prior assays. No Torch or MLX
was imported, no model was allocated, and no inference or speed measurement
was performed. The fixed guard rejects all six arms; production is unchanged.

The [scanner](../../benchmarks/analysis/mlx/eligibility.py) reads exact BF16
values in bounded 262,144-element CPU chunks, validates selected shapes and
disjoint byte spans, writes partial receipts after each tensor and bounds its
owned child to 120 seconds. Existing receipts are not overwritten. Fourteen
tiny/property tests passed, including an independent Python half conversion
oracle, BF16/FP16 exponent and subnormal counterexamples, malformed spans,
per-arm role/layer intersections and failure/timeout retention. Independent
source and tiny-test review preceded the real scan.

Both prospective declarations remain archived: the original scanner declaration
and source bytes in
[the executed source archive](../../benchmarks/analysis/results/mlx-mlp-fp16-eligibility.sources-960d68e2ea757b13.json),
and this document's original protocol identity in
[the protocol sidecar](../../benchmarks/analysis/results/mlx-mlp-fp16-eligibility.protocol.json).
The live protocol and results now have this single owner. Eligibility evidence
is limited to the pinned checkpoint's selected parameter values; it establishes
neither model quality nor a claim about other checkpoints.
