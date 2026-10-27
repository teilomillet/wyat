# Ouro inference speed on Apple Silicon

This page records the profiling and optimization work for local
`wyat-ouro-decision-v1` inference on an Apple Silicon Mac (M4 Pro, 24 GB
unified memory, macOS 27.2, PyTorch 2.14). It covers the two-stage Ouro
decision-scoring path: shared-context prefill, per-candidate recurrent passes,
and the FP32 scalar readout.

> **Scope notice.** The numbers below were measured with a local, non-release
> profiling artifact assembled from the pinned `ByteDance/Ouro-1.4B` base
> weights and the local `last2` attention-LoRA decision checkpoint. Synthetic
> token tensors are used for the load and scaling studies. They describe this
> machine and this workload; they are not product benchmarks, accuracy claims,
> or forecasts for another host.

## Summary

| Path | P400/K6/S16 end-to-end | Speedup vs CPU default | Logit drift | Device memory |
| :--- | ---: | ---: | ---: | ---: |
| CPU FP32 (default) | 3.25 s | 1.00x | exact (bitwise) | post-run peak RSS ≈ 6.3 GiB |
| CPU FP32 + `candidate_batching` | 2.40 s | 1.36x | max 2.8e-6; rank/top-1 equal | ≈ 5.2 GiB |
| Apple GPU MPS FP32 (`mps-fp32`) | 1.55 s | 2.1x | max 4.2e-6 vs CPU | 5.4 GiB current / 6.1 GiB driver |
| Apple GPU MPS FP32 + `candidate_batching` | 1.29 s | 2.5x | max 8.8e-6 vs CPU golden; rank/top-1 equal | 5.4 GiB current / 6.1 GiB driver |

Measured matrix from the final controlled suite (median end-to-end per question,
frozen loader revision, one process per arm, 6 threads, `nice 10`; the same
machine carried unrelated background work, which is why absolute times are
higher than the coolest observations):

| Workload | CPU default | CPU `candidate_batching` | MPS FP32 | MPS + batching |
| :--- | ---: | ---: | ---: | ---: |
| P400 / K6 / S16 | 3.25 s | 2.40 s | 1.55 s | 1.29 s |
| P832 / K10 / S64 | 7.98 s | 5.04 s | 3.83 s | not run |
| P512 / K9 / heterogeneous S 8–24 | 4.40 s | not run | 2.49 s | not run |

The batching gain is real but workload- and contention-dependent:

- CPU: the candidate stage dropped from 1.32 s to 0.40 s (3.30x) at P400/K6/S16
  and from 2.99 s to 1.67 s (1.79x) at P832/K10/S64, for end-to-end gains of
  1.36x and 1.58x. An earlier run on a machine saturated by unrelated
  background work (load average ≈ 6) showed no gain, so measure on a quiet
  machine and treat absolute numbers as ±20%.
- Apple GPU: the candidate stage dropped from 0.535 s to 0.214 s (2.5x), which
  is 6.2x faster than the CPU default's candidate stage; end-to-end 1.55 → 1.29
  s because prefill dominates the MPS path.
- Batching only groups candidates with identical token length. Grouped batching
  is 1.87x on the uniform-length MPS candidate stage but 1.08x (or 0.96x,
  slower, at P512/K9) on heterogeneous lengths. A padded-batching prototype for
  heterogeneous lengths is 1.74x there (2.55x on short real decision rows) but
  is not wired into the runtime.

Load time dropped from ≈15 s to ≈2.7 s and post-run peak RSS from ≈8.4 GiB to
≈6.3 GiB (load-phase peak ≈4.2 GiB) after streaming the base checkpoint into a
meta-initialized model (see [Loading and data movement](#loading-and-data-movement)).

`P` is shared-context tokens, `K` candidates/options, `S` tokens per candidate.
`P400/K6/S16` is the canonical synthetic workload used throughout this page.

## Where the time goes

One decision runs two stages, and they are bottlenecked differently:

1. **Shared-context prefill** — four unadapted recurrent passes over the
   `P`-token context. Each pass reads every frozen backbone weight once, so the
   stage is compute/weight-bandwidth bound and scales with `P`. It dominates
   single-question latency on CPU (≈1.90 s of the ≈3.25 s end-to-end at
   `P=400`).
2. **Candidate scoring** — two adapted recurrent passes per candidate plus the
   FP32 readout. Candidate sequences are short (`S` ≈ 8–24 tokens), so the
   per-pass matrix multiplications are small and dominated by re-reading the
   frozen weights (`aten::mm` accounts for most Stage-2 CPU time). The current
   path also concatenates the retained context K/V bank with each candidate
   suffix (`aten::cat` per layer per pass); that copy is measurable
   (≈56 ms of ≈1.33 s for K=6) but secondary to the weight reads.
3. **Readout** — one FP32 LayerNorm-Linear-GELU-Linear pass over `K` hidden
   rows; negligible.

Consequences:

- Fixing data movement alone (allocation-free K/V workspace) is worth only a
  few percent on CPU, because the candidate stage is weight-read bound.
- Reusing each weight read across all candidates is the real Stage-2 lever: in
  an isolated gate a batched candidate forward was 1.84x faster over mixed K
  (2.07x at K=10, 1.85x at K=6); under a saturated machine the CPU gain
  disappears while the Apple GPU still measures 2.5x on the candidate stage.
- The Apple GPU is the largest single-question win on this machine: the same
  FP32 workload runs 2.2x faster end-to-end, and the GPU also finishes the
  candidate stage 2.7x faster than the CPU because the larger machine can hide
  the small-GEMM inefficiency.

## Enabling the fast paths

```python
import wyat

# The artifact's declared device_policy picks the device:
#   "mps-fp32"            -> Apple GPU (MPS) on Apple Silicon
#   "cpu-fp32"            -> CPU
#   "cuda-bf16-autocast"  -> CUDA when available, otherwise CPU
model = wyat.load("/path/to/ouro-artifact")

# Candidates are scored in shared-context batches by default. For bitwise
# replay against golden logits, score them one at a time:
from wyat.loading import load_ouro

scorer = load_ouro("/path/to/ouro-artifact", candidate_batching="per_candidate")
```

`load_ouro(..., device="mps")` on a `cpu-fp32` artifact is refused unless
`allow_device_policy_override=True` is passed: an artifact is qualified for the
device types its manifest declares, and executing elsewhere accepts the
artifact's declared tolerance instead of the default parity contract.

### Resource caps

The GPU and CPU paths are fast enough to compete with the rest of the desktop,
so the benchmark harness and the examples below deliberately cap them:

```sh
export PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0   # default 1.4 is aggressive on 24 GB
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2  # default 1.7 can exceed physical memory
export PYTORCH_ENABLE_MPS_FALLBACK=0         # fail loudly, never silently run CPU
nice -n 10 .venv/bin/python scripts/ouro_speed/bench.py \
  --artifact /tmp/ouro-speed/artifact-last2-mps-fp32 \
  --device mps --rss-cap-gib 10 --threads 6 --repeats 5
```

Keep one model per process: loading a second 1.4B model in the same process
exceeds a 0.6 MPS memory fraction on a 24 GB Mac. `bench.py` enforces a
resident-memory watchdog and aborts above its cap; it prints stage progress
with RSS and MPS memory so a human can watch the run.

## Loading and data movement

The loader used to materialize the BF16 checkpoint as a host dictionary and
cast it to FP32 before copying it into a model that was itself initialized with
random FP32 weights. That peak (≈8.4 GiB) and the added read pass are
unnecessary:

- The model is now constructed on the `meta` device (no random init, no
  allocation) and materialized directly on the target device with
  `to_empty`.
- The base safetensors file is read once, tensor by tensor, validating dtype,
  shape and finiteness and copying BF16 → FP32 in place. Each BF16 value maps
  exactly to one FP32 value, so this is bitwise identical to the old
  `load_state_dict` path.
- The retained context K/V banks, candidate K/V workspaces, positions, rotary
  embeddings and causal masks are allocated on the execution device and reused
  across candidates; the scoring path performs no per-candidate host round
  trips. Only the final logits vector is copied to the host.

The audit script `scripts/ouro_speed/audit_transfers.py` reports every
parameter/buffer device and dtype, the K/V bank device, and host-device transfer
counts for one scoring call. `scripts/ouro_speed/parity.py` compares logit
vectors for bitwise equality, maximum absolute/relative drift and rank changes
against a declared tolerance.

## Correctness gates

- **Default path:** bitwise identical to the pre-optimization implementation.
  The stored golden logits for `P400/K6/S16` reproduce exactly (only
  sixth-decimal JSON rounding differs), and the real-artifact golden check runs
  with `atol=2e-5` and rank equality.
- **Shared-context batching (`candidate_batching="shared_context"`, the
  default since 2026-10-24):** `score_candidates_shared_context`. Candidates are
  batched within `SHARED_SCORE_BUDGET_BYTES`, and attention reads each layer's
  context K/V once for the whole batch instead of copying it into every row.
  It is not bitwise identical to the per-candidate path. On the real 1.4B
  artifact it matches the stored CPU FP32 golden logits within `atol=2e-5`, with
  identical ranking, at P400/K6/S16 and P832/K10/S64. On MPS (the `mps-fp32`
  artifact) it lands 4.5e-6 from the CPU golden, against 4.6e-6 for the
  per-candidate path on MPS, and 3.4e-6 from that path; ranking is unchanged.
  Its speed on this machine has not been measured yet.
  `candidate_batching=True`/`False` still map to the two modes.
- **Earlier grouped batching (replaced 2026-10-24; the measurements below and
  the `candidate_batching` columns above are of this implementation):** it
  grouped equal-length candidates and copied the context K/V into every row of
  a group's workspace. On 36 synthetic
  questions with K ∈ {1,2,3,4,5,6,8,10} and varied context/candidate lengths it
  measured max |Δlogit| = 6.8e-6 (gate 2e-5), 0 rank changes, 0 top-1 changes
  and 0 pair-order flips across 442 compared pairs; 18/36 questions were
  bitwise identical. On the Apple GPU it is 1.61x on a 32-question corpus
  (1.87x uniform lengths, 1.08x heterogeneous, 0.96x on one heterogeneous
  shape), 1.76x on 32 real ARC decision rows, with batched-vs-serial drift
  ≤ 9.7e-6 and no rank/top-1/decision changes. It must not be used where
  bitwise replay is required.
- **`mps-fp32`:** FP32 math on the Apple GPU. The declared execution tolerance
  is `MPS_EXECUTION_TOLERANCE_ATOL=1e-4` / `RTOL=1e-4` and the shipping gate
  also requires identical rank and top-1 choice. Measured drift is 4.2e-6–1.1e-5
  on long-context synthetic shapes (32/32 within 2e-5, ranks identical) and up
  to 6.5e-5 on short real decision prompts, where ranks/top-1 stayed identical
  and 0 decisions flipped. BF16 on MPS is *not* shipped: it was ~9% faster in
  one probe but drift was 2.6e-2.
- A multi-candidate regression guard is required: an intermediate optimization
  that reused one K/V workspace across layers silently corrupted candidates 2+
  while candidate 0 stayed exact (max |Δ| = 0.60). Single-candidate or
  cross-path-only checks cannot see that class of bug; compare a stored golden
  vector with K ≥ 6 at the same length.

## Reproduction

```sh
# Build the local profiling artifact (profiling only; never released):
.venv/bin/python scripts/ouro_speed/build_local_artifact.py \
  --base /path/to/Ouro-1.4B-snapshot \
  --checkpoint /path/to/last2/final.pt \
  --out /tmp/ouro-speed/artifact-last2-cpu-fp32 \
  --device-policy cpu-fp32

.venv/bin/python scripts/ouro_speed/bench.py --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 \
  --device cpu --threads 6 --repeats 5 --json-out /tmp/ouro-speed/baseline.json
```

Receipts, golden logits and the transfer/parity audit are local measurement
records and are not published with the package; the benchmark harness is in
`scripts/ouro_speed/` and the MPS probe scripts and results are in
`scripts/ouro_speed/mps/`.

## What does not transfer (and what does)

These levers are runtime/precision engineering, not a change to the Ouro
architecture or its trained recipe. The recurrent-depth literature (for example
the latent-reasoning recurrent-depth model work) gains from sharing K/V across
recurrence depth; this decision path already retains only the final prefill
bank, so that multiplier is already banked and cannot be counted again.

- **Transfers to keme evaluation on GPU:** allocation-free candidate K/V
  workspaces and cached geometry reduce allocator traffic and synchronization
  points on any device; the keme cache continues to be qualified by its own
  numerical gate.
- **Opt-in only:** candidate batching. keme's training/qualification contract
  requires bitwise optimizer-state parity, and batching changes GEMM shapes, so
  it stays behind the wyat-level tolerance gate and is not applied to keme
  training.
- **Not portable:** the Apple GPU device policy and MPS watermarks are
  Apple-Silicon specific. CUDA was not measurable from this Mac.

## Context attention: parity vs throughput

A v2 artifact stores `execution.context_prefill.chunked_attention`. `lean` is
the exact MATH-equivalent path; `flash` is a fused BF16 throughput mode that is
**not** MATH-exact. On long contexts the difference is measurable: in a 20-row
long-context probe, Flash was about 2.9x faster than the exact path but 0/20
requests were bit-identical, 10/20 exceeded a |Delta p| <= 0.02 gate, and one
top choice flipped at a near-tie.

Load with `context_attention="lean"` when parity with the evaluated engine
matters; use `"flash"` only after task-level validation. The override is
load-time only and does not rewrite the artifact or its `model_id`.

```python
from wyat.loading.ouro import OuroScorer
scorer = OuroScorer.from_directory(path, context_attention="lean")
```

## Exact acceleration of the engine-parity runtime (2026-10-27)

The `wyat-2-1.4b` runtime is qualified bitwise against the gated engine on the
420-request parity set. Profiling that execution on the qualification GPU
(RTX 4070 Ti, torch 2.8.0+cu128) showed the context prefill is compute-bound
while the short shared-context candidate stage is launch-bound: about 52 ms of
a ~154 ms short-row request, with thousands of small kernel launches per call.
The following changes remove only that overhead:

- one device copy for all candidate tokens of a row;
- one combined token-range reduction per validated tensor instead of two;
- the block-diagonal mask of a candidate batch built once per batch instead of
  once per decoder-layer visit (identical De Morgan mask values);
- a vectorized fill of the padded candidate batch;
- the finite-logit check after the single host logits copy;
- **CUDA-graph replay of short candidate batches.** On CUDA with
  shared-context scoring, a candidate batch with a context of at most 1,024
  tokens, at most 32 candidates and a single score batch is captured on the
  third sighting of its exact geometry (two eager calls first, so a shape that
  never repeats pays nothing). The graph records the same kernels the eager
  batch runs on fixed buffers. Before a graph is used, its first replay must
  match the eager logits as FP32 bit patterns; a mismatch, or a capture or
  replay error, permanently disables replay for the process and leaves eager
  scoring in charge. The cache is bounded (32 geometries; one ~200 MiB static
  K/V buffer pair; each graph owns its own memory pool because replay order is
  LRU, not capture order). Capture needs `PYTORCH_CUDA_ALLOC_CONF` without
  `expandable_segments`; otherwise capture fails safely and eager scoring runs.
  Replay is bit-identical to eager scoring on the measured workload (see below)
  and never changes the artifact's recorded execution.

The artifact, its manifest and `model_id`, numerics, batching budget, chunk
sizes, the engine and the evaluation are unchanged.

Measured on the qualification GPU (RTX 4070 Ti, torch 2.8.0+cu128) on the
same artifact and request set, eager and graph-enabled in one process with
allocator churn between capture and replay:

| rows | eager end-to-end | graph end-to-end | median speedup |
| --- | ---: | ---: | ---: |
| repeated geometry (12) | 95.4 ms | 75.5 ms | 1.263x |
| singleton geometry (4) | — | — | 1.195x¹ |
| all 16 | 95.4 ms | 75.8 ms | 1.260x |

Every row's graph-enabled logits equal the eager logits as FP32 bit patterns.
The cache captured 8 geometries and replayed 88 times with no mismatch and no
error; peak allocated memory was 5,665 MiB (model-load dominated). These are
end-to-end `question_logits` calls on short rows (context prefill included and
unchanged), so the gain is candidate-stage launch overhead removed, never a
change to arithmetic. ¹ "Singleton" is classified by sorted candidate lengths;
two rows can still share a runtime key in caller order, so this column is not
an eager baseline and is not used as a claim. On the untouched 420-request engine-parity protocol (300 G0c + 120 long
public requests) the graph-enabled runtime stays bitwise vs the gated engine:
420/420 candidate logits equal, 420/420 argmax agree, max |Delta logit| = 0.0,
and the per-path counts (242 single pass / 165 chunked / 13 host cache) match
the prior qualified receipt row for row. The same receipt records 92 captures
and 892 graph replays with 0 mismatches and 0 errors, so the parity includes
the replay path; capture overhead was 14.6 s of the run. Total 420-run time was
576.3 s vs 585.8 s for the released runtime on the same protocol (-1.6%; the
baseline was unpaired and used a different allocator configuration, and the
capture overhead is included), while the within-run wyat/engine ratio improved
from 0.982 to 0.961. Full receipts: keme
`docs/ouro/WYAT2_EXACT_SPEED_20261027.md`.

Verification: `tests/test_ouro_exact_speed.py` pins the reference logits as
uint32 bit patterns for the tiny v2 fixture (40 logits, both case orders, with a
16/16 one-bit negative control). The GPU qualification is the untouched
420-request parity protocol (every candidate logit bitwise equal to the gated
engine, max |Delta| = 0.0) run through the graph-enabled runtime with the
graph-replay counters recorded in the same receipt, plus the 86-request kit
adapter comparison.
