# CPU projection packing primitive

The hypothesis is that one maintained FP32 linear operation for K/V or Q/K/V
reduces layer-zero projection cost. This is a primitive test, before a whole-model
candidate. Numerical and full-ranking model parity remain hard constraints.

The existing captured gate inputs are post-attention MLP activations and are not
QKV inputs. Instead, read the exact released checkpoint through Safetensors mmap:
only the required embedding rows, layer-zero input RMSNorm and Q/K/V weights.
Apply maintained Qwen3RMSNorm with epsilon `1e-6` to the authoritative full token
vectors, then take suffix rows after the exact 25-token prefix. This norm is
per-token; preceding attention is absent in layer zero. No backbone is created or
executed. Pinned tokenizer metadata resolves padding to token `151643`.

The four fixed operands are canonical offsets 7/18/75 (full lengths 192/197/214,
suffix lengths 167/172/189) and the canonical first eight requests, padded to 192
before normalization and prefix removal (B8 suffix length 167). Preserve token
IDs, case IDs, ordering, padding, activation bytes, shapes and strides. All 77
reference vectors and canonical inputs are admitted before checkpoint tensors.
These are BANKING77 training-source development cases; checkpoint exposure is
unknown.

Q/K/V have output widths 2048/1024/1024 and input width 1024. One FP32 packed
allocation is 16 MiB. Separate Q/K/V use contiguous views; packed KV uses its
contiguous tail; packed QKV uses the complete allocation. Original source tensors
are released after exact-bit checks. The temporary source-plus-packed projection
pool is 32 MiB during setup, excluding normalization, activations, output archives
and allocator storage. This is logical capacity, not RSS. Hypothetical 28-layer
QKV and KV copies would be 448/224 MiB if originals were also retained; this
primitive does not create those copies.

Every arm executes one cold call, one warm call and six measured calls, following
all six permutations of `separate`, `kv`, `qkv`: 96 total calls across four
operands. The clock contains one `project()` call, including admission, maintained
`F.linear` and output splitting. Setup, hashing, snapshots, quality comparisons
and persistence occur outside that clock. Each call's output shapes, strides,
bytes hash and raw timing are retained before its gate can reject it. Save each
distinct output and input in an excluded local Safetensors artifact. Changed
strides may affect later Q/K normalization or RoPE; neither is timed here.

Projection parity uses the original symmetric max allowance:
`abs(a-b) <= max(1e-3, 1e-3 * max(abs(a), abs(b)))`, with finite FP32 outputs.
There is no probability or full-ranking claim at this boundary. The unchanged
whole-model gates remain mandatory for any later candidate.

A candidate is eligible for a later full-model screen only if all four operands
have six-order median baseline/candidate ratio at least **1.12** and interpolated
sample p05 strictly above **1.0**. This prospective filter is not promotion. A
historical profiled projection fraction of 0.1938 motivates conservative triage
under an unchanged-cost assumption; it is not an observed ordinary fraction or
speed bound. Every ratio remains visible, including regressions. Source/assets,
process policy and thread restoration must pass before a positive completion
receipt. No new precision policy, JIT, GPU work or production changes occur.

The command below records the historical launch. Existing receipts must remain
intact; any separately authorized repeat requires a fresh output stem. No repeat
is needed or authorized by these results.

```sh
python -m benchmarks.analysis.torch.projection.run .models/Decision-2.0-Kai-0.6B \
  --output benchmarks/analysis/results/torch-qkv-primitive.json
```

The standard-library parent owns one 120-second child process group. Timeout,
failed output, missing call or incomplete receipt is unknown/rejected; do not
rerun or relax gates to hide it. Local primitive timings cannot establish a model,
server or general CPU speed gain. No production promotion is performed.

## Observed result

The repaired bounded worker completed **96 calls** across all four authentic
operands. The independent standard-library audit bound every saved output to its
actual FP32 array and recomputed the unchanged symmetric numerical gate. All
three arms have identical output hashes at every shape, with maximum absolute
and relative differences **zero**. Exact packed projection values match the
released layer-zero Q/K/V tensors after their BF16-to-FP32 expansion.

| Authentic operand | KV median | KV sample p05 | QKV median | QKV sample p05 |
| --- | ---: | ---: | ---: | ---: |
| Scalar, full length 192 | 1.01678x | 0.93591x | 1.07154x | 1.01623x |
| Scalar, full length 197 | 1.02530x | 1.01108x | 1.05531x | 1.04202x |
| Scalar, full length 214 | 1.02920x | 1.00146x | 1.05988x | 1.03029x |
| First eight, padded full length 192 | 1.04164x | 1.00155x | 1.03274x | 0.96456x |

Both variants miss the prospective **1.12x median at every operand** condition.
KV also regresses in the scalar-192 tail, and QKV regresses in the batch-eight
tail. All six ratios per operand and arm remain in the
[raw report](../../../benchmarks/analysis/results/torch-qkv-primitive-final.json)
and [independent audit](../../../benchmarks/analysis/results/torch-qkv-primitive-final-review.json).
Both `primitive_eligible` values are false; `accepted` remains false. Do not
construct or promote a full-model candidate from this screen. This rejects the
measured packing candidate under this declared filter, without ruling out
another implementation or context. The sample p05 is not a confidence bound.

The inference child exited zero and was reaped after **4.122 s** within its
120-second cap. Source/asset verification and original thread restoration pass.
The independent audit exited zero and was reaped after **3.061 s** within its
separate 60-second cap; its
[execution receipt](../../../benchmarks/analysis/results/torch-qkv-primitive-final-review-final.execution.json)
preserves exact producer text, argv, output and ownership. The audit validates
16 executed source texts, the six reviewed experiment files, unchanged
production22 and the pinned input/model assets. The raw report SHA is
`fe4d68337e4b695bdafc26c18ed261bd776175e36e6970ed03694965840afe17`;
the executed source archive SHA is
`eff85f9cc7c03a0d045499df4c984d9747e6c7c23d60efad53c841caf2b97fb6`.
The prospective protocol above remains in that immutable archive; this result
was appended afterward.

Normalized operands are authenticated by source/model/token provenance and
retained byte identities. The auditor does not independently replay Torch's
RMSNorm reduction or recompute every matrix product: separate-projection outputs
are its numerical parity oracle. Input outer strides retain the full-token
width for singleton batches, where that stride is inactive, and use the compact
suffix width for batch eight. Output split strides differ between arms and are
retained explicitly. Q/K normalization, RoPE, attention, cache updates, decision
heads, ranks and probabilities are outside this primitive. Logical allocation
counters do not establish RSS, peak memory, device aliases or whole-model speed.

## Preserved failure history

The initial real worker rejected its own supervisor-created `.wall.json` and
`.log` sidecars before input admission or checkpoint operands. It exited one
without a timeout after **1.799 s**; no main receipt or primitive sample exists.
That attempt is an execution-harness failure with unknown quality and speed.
Its [wall](../../../benchmarks/analysis/results/torch-qkv-primitive.wall.json) and
[log](../../../benchmarks/analysis/results/torch-qkv-primitive.log) remain intact.
A narrowly reviewed sidecar allowance plus collision tests preceded the fresh
final run; the repaired focused suite passed **18 tests**.

The first independent raw-audit attempt assumed compact outer input strides even
for singleton batches and stopped before output parity scans. Its
[failed execution receipt](../../../benchmarks/analysis/results/torch-qkv-primitive-final-review.execution.json)
and exact producer remain preserved. The auditor then corrected only that
layout expectation and inspected the same immutable raw arrays successfully;
no model inference was repeated for this auditor correction.
