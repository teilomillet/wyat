# Packed gate/up storage feasibility

Hypothesis: one packed gate/up backing can serve both the separate-projection
control and combined-projection candidate without retaining duplicate weights.
This addresses the measured CUDA gate/up target, about 29.0% and 27.2% of
representative kernel-duration sums at batches one and eight. Those fractions
are diagnostic; they do not establish a complete-inference Amdahl bound.

A private CPU probe used the actual Kai layer-zero BF16 gate/up/down weights
restored to FP32, and the retained first-eight activation of shape 1536×1024.
The cached activation and its capture declaration were hash-pinned; consumed
model tensor payload identities were recorded before loading. Original Qwen MLP,
storage-only control and packed candidate output hashes were identical, with
finite maximum absolute error zero. Nine tests passed, including twenty tiny
Hypothesis examples and rejected owner states. Tiny random cases test mechanics,
not the deployment distribution.

The control's two contiguous row views and candidate's combined parameter share
one **25,165,824-byte (24 MiB)** backing. Counting all three parameter tensors
would double-count that storage. Replaced parameter wrappers expired; this does
not measure allocator/RSS release or prove absence of arbitrary external aliases.
The probe assumes a fresh, unhooked, gradient-disabled evaluation owner with no
optimizer, checkpoint writes or external parameter-identity contract.

This supports local CPU mechanics and one real layer's outputs. It supplies no
whole-model decision, CUDA parity, latency, speedup, memory-reservation or
production claim. An initial pytest collection failed on an unrelated protected
temporary-directory sibling before tests ran. The retry used byte-identical
source in a task-owned directory; both outcomes remain retained.

The next candidate needs explicit device/storage ownership. Independently
moving control and candidate modules to CUDA can copy their aliased views into
separate allocations. Moving the complete owned model first, packing one layer
on-device, and recreating both control views is a narrow candidate; retained and
peak allocation/reservation remain measurements. CPU storage proof does not
establish CUDA alias preservation. Storage-only control must independently pass
the existing CPU/CUDA references before comparing complete packed inference.
All77 logits, probabilities, selections, full rankings, failures and labeled
accuracy remain gated; promotion needs warmed paired whole-body measurements
and resource/lifecycle challenges. No such CUDA experiment has been released.

Evidence: [real-layer receipt](../../../benchmarks/analysis/results/cuda-packing-feasibility.json),
[provenance companion](../../../benchmarks/analysis/results/cuda-packing-provenance.json),
and [independent read-only review](../../../benchmarks/analysis/results/cuda-packing-local-review.json).
The exact executed source is retained as text, with both the successful owned wall
and initial collection-failure wall. The current full checkpoint SHA matches the
original pin; current per-key payloads match their pre-load recorded identities.
The local runtime is Torch 2.14.1, HF 5.18.0 and Python 3.12; this supplies no
numerical/device result for the remote Torch 2.12.1 runtime. The companion retains
unobserved historical environment/hash intervals as unknown rather than filling
them retroactively.
