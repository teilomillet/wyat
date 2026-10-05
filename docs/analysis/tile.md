# Output reuse trial

This declaration precedes the fixed64 real-tensor experiment. The operational
owner is the Wyat engine task. Both kernels remain experimental; frozen32 is a
live control for the previously rejected implementation. Neither is selected for
production. Keep the prior receipts, captured tensors and executable sources
unchanged so the comparison can be reproduced.

Hypothesis: changing the output tile from 32×32 to 64×64, with the same 32-element
reduction order and independently decoded BF16 weights, can reduce repeated
input and weight reads enough to improve the first-layer gate projection. The
larger tile uses 32KiB shared storage and 16 float matrix accumulators per SIMD
group, compared with 12KiB and 4. Register pressure, occupancy and physical memory
traffic are unknown. Logical read counts are predictions, not profiler results.
The source account is in [mixed-tile.md](mixed-tile.md).

Admit the exact previous main receipt `fb8bde4a…`, independent audit `08d8b0bd…`
and tooling verification `e332f2fa…`, including their complete source archives,
model identity, dataset, CPU reference, captured activation/weight file and cold
outputs. The sole admitted historical document difference is the exact audited
result append to `mixed.md`; executable bytes remain frozen. Read the actual
captured inputs at M=167, 1336, 192, 1536, K=1024 and N=3072. Do not recapture a model,
substitute data, or pool prefix and full-input shapes.

Before timing, reproduce the prior maintained and frozen32 cold output bits.
Require finite outputs within `1e-4 + 1e-4 * abs(reference)` for each candidate
element against maintained FP32. Require the new64 BF16 and new64 FP32 cold
outputs to agree bitwise, including the tiny signed-zero and arbitrary-FP32
input tests. Store all four arms' cold arrays. Numerical admission precedes
performance measurement.

Time three arms: maintained FP32, frozen32 BF16, and new64 BF16. Use all six
permutations, ten separately synchronized dispatches per arm and block, and one
warm dispatch per arm at each of the four shapes. Include dispatch, evaluation
and synchronization in each interval. Hash every warm and timed output outside
its timer against that arm's cold output. Those observations may perturb later
dispatches; this measures observed dispatches, not uninterrupted throughput.

The unchanged success gate compares new64 with maintained: the median of six
paired block ratios must be at least 1.05 and their interpolated fifth percentile
must exceed 1, at **both prefix shapes**. Report full-input controls separately.
New64 versus frozen32 is descriptive evidence about this hypothesis; improvement
over the rejected kernel does not establish an improvement over maintained
MLX. Do not relax thresholds or select a favourable shape after seeing results.

The parent remains framework-free and owns one worker with a 180-second deadline,
process-group cleanup and terminal wall receipt. Disable TF32 before MLX import,
use the GPU stream, and restore the prior allocator-cache limit after applying
512 MiB during the trial. This cache setting does not cap resident memory. Save
completed arms and preserve partial/error receipts; an interrupted arm remains
unknown. Archive exact trial sources before execution. Independent review and
tiny correctness checks precede the real run. DMBench has reserved this bounded
window; unrelated applications' activity is not established.

A projection result cannot establish complete-inference latency, held-out model
quality, or a 20× engine improvement. Any selected kernel would need a separate
complete-inference experiment under the existing logits, probabilities, full
ranking and performance gates.
