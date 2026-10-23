# Ouro decision scoring on Apple GPU (MPS) — qualified evidence

> **PROFILING ONLY — NOT A RELEASE ARTIFACT.** Synthetic token tensors plus the
> local non-release `wyat-ouro-decision-v1` artifact (`/tmp/ouro-speed/artifact-last2-mps-fp32`,
> ByteDance/Ouro-1.4B snapshot + keme `last2` adapters). Nothing here publishes a
> release artifact; do not cite these numbers as product benchmarks.

Owner: `mps-probe` (task-3). Scope: qualify MPS decision scoring for this M4 Pro / 24 GiB
Mac, prototype candidate batching, test precision arms, document caps/failure modes.
Runtime source is owned by the Lead/`perf-core`; this directory only measures it and
describes the exact loader changes to integrate.

## TL;DR — recommendation

**Ship `device_policy="mps-fp32"` → Apple GPU FP32 as an opt-in MPS arm; keep FP32
defaults (no bf16, no autocast); expose `candidate_batching=True` as a tolerance-gated
opt-in; document the two mandatory MPS env vars + memory fraction + one-model-per-process.**

| item | recommendation | evidence |
| :--- | :--- | :--- |
| artifact policy | `device_policy="mps-fp32"`, permitted device `{mps}`; `device=None` resolves to `mps` when MPS is available; other policies rejected unless `allow_device_policy_override=True` | `wyat/loading/ouro.py:79-84, 1080-1101`; `robustness.json` |
| precision on MPS | **FP32 weights/activations/LoRA/head** (what the loader already does); bf16 and autocast are *not* ship-safe | `precision.json`, `fp32-extended.json` |
| candidate batching | opt-in `candidate_batching=True` (loader's grouped path) for MPS; 1.61x candidate-stage on varied shapes, 1.87x uniform, 1.76x on real rows; a padded variant is faster for heterogeneous lengths (1.74x vs 1.08x) but is a prototype only | `batching.json`, `scripts/ouro_speed/mps/mps_batch.py` |
| declared MPS tolerance | `atol=2e-5` on long-context synthetic shapes; shipping gate `atol=1e-4 AND rank/top-1 identical` (real short prompts reach 6.5e-5 drift while still 0 decision flips) | `fp32-extended.json` |
| expected speed | P400/K6/S16 e2e ≈ 1.6–2.0 s vs CPU 3.34 s; P832/K10/S64 e2e ≈ 3.8 s vs CPU 6.72 s; real short rows candidate stage 0.51 s → 0.15 s with padded batching | `fp32-extended.json`, `batching.json` |
| caps | `PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2` **before `import torch`**; `torch.mps.set_per_process_memory_fraction(0.6)`; one model per process; RSS watchdog 14 GiB | `robustness.json`, this file §1 |
| not safe | bf16 default, autocast default, default macOS watermarks, two models in one process, silent `cpu-fp32` artifact on MPS | §9 |

## 1. Environment and mandatory caps

- Apple M4 Pro, 12 cores (8 performance), 24 GiB unified memory, macOS (Darwin 27.x),
  Python 3.12, torch 2.14.1, transformers 5.18, venv `.venv`.
- Every heavy run enforces, in order: exclusive lock `/tmp/ouro-speed/BENCH_LOCK`
  (`mkdir` + 10 s retry + stale-owner detection, removed on exit), `os.nice(10)`,
  `torch.set_num_threads(6)`, RSS watchdog hard-abort `os._exit(42)` above 14 GiB,
  MPS fraction 0.6 of `torch.mps.recommended_max_memory()`
  (17.76 GiB → **10.656 GiB** cap), and `PYTORCH_ENABLE_MPS_FALLBACK` **unset**.
- Watermarks must be set before torch initialises the MPS allocator:
  `PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2`
  (default LOW=1.4/HIGH=1.7 is unsafe on 24 GiB; HIGH=1.0 alone trips
  `invalid low watermark ratio 1.4`).
- Timing discipline: `torch.mps.synchronize()` before/after every timed stage, one full
  warmup run, then 5 output medians (scripts in this directory print min/max/samples too).

## 2. Artifacts, frozen inputs, loader revisions

Canonical MPS artifact: `/tmp/ouro-speed/artifact-last2-mps-fp32`
(`model_id=ouro-03e7eb4e…`, `execution.device_policy="mps-fp32"`, `attention="sdpa-math"`).
CPU reference: `/tmp/ouro-speed/artifact-last2-cpu-fp32` (`device_policy="cpu-fp32"`).
Both artifacts ship **byte-identical** `model.safetensors` (`58872a72…`),
`head.safetensors` (`0b301342…`) and `adapters.safetensors` (`53a60b09…`); only the
manifest policy/model_id differs. So all MPS-vs-CPU drift below is device math, not weights.

Frozen inputs (regenerated once by `ref_cpu.py`, sha256 in the files):
`inputs/question-set.json` (`sha256=55e1c8c7…`) — headline `p400-k6-s16`,
`p832-k10-s64`, `p512-k9-hetero-8to24`; a 32-question synthetic corpus
(P 256..832, K 2..10, uniform + heterogeneous S 8..64); and the 32 real keme ARC dev
rows from `.keme/ouro/experiments/pretrained-arc-adaptation-frozen/dev-tokenized.json`
(legacy `prefix_ids`/`branches` mapped read-only to shared context + candidate
continuations; P=35..139, K=3..5, S=5..17).

Loader revisions matter: this file was produced against
`wyat/loading/ouro.py sha256=5855451c…` (git HEAD `7646d95f…`). A **transient
uncommitted revision** at 18:02 (`a35d8ed0…`) silently corrupted multi-candidate
logits — candidate 0 identical to the committed baseline, candidates 1..5 wrong
(max |Δ|=0.596) — because a K/V workspace `prepared` fast-path was reused across
candidates without re-copying the per-layer context prefix. See
`diagnostic-workspace-reuse.json` and §10. Every receipt here records loader SHA at
start and at receipt-write and flags `loader_unchanged_during_run`; the batching run
is flagged `false` only because `wyat/loading/ouro_artifact.py` was edited mid-run
(imported copy in memory unchanged; `ouro.py` identical at start/write).

## 3. Extended FP32 MPS numbers

Artifact `artifact-last2-mps-fp32`, FP32 compute, 5 reps after 1 warmup, medians.
MPS memory after load: **current 5.395 GiB / driver 6.055 GiB** (recommended 17.76 GiB,
fraction cap 10.656 GiB). Load time 3.20 s on a cool machine state, 10.92 s in the
warmer run below (vs Lead's 18.86 s before the direct-copy weight-load optimisation).

| shape | prefill | candidate-from-cache | e2e MPS | e2e CPU fp32 | speedup | max\|Δlogit\| vs CPU | rank |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| P400 / K6 / S16 | 1.332 s | 0.598 s | **2.026 s** | 3.344 s | 1.65x | 4.17e-6 | equal |
| P832 / K10 / S64 | 2.464 s | 1.392 s | **3.826 s** | 6.716 s | 1.76x | 6.08e-6 | equal |
| P512 / K9 / hetero 8..24 | 1.392 s | 0.922 s | **2.491 s** | 4.403 s | 1.77x | 1.01e-5 | equal |

Where the time goes (median split, `split_median_s` in the receipt):

| shape | prefill share | candidate share | readout head |
| :--- | ---: | ---: | ---: |
| P400/K6/S16 | 65.8% | 29.5% | 0.0003 s |
| P832/K10/S64 | 64.4% | 36.4% | 0.0003 s |

**Prefill dominates MPS e2e (~2/3)**, so candidate batching alone can only improve
end-to-end by ~1.2–1.3x on long-context shapes (but up to ~2.5x on short real
prompts, where the candidate stage dominates — see §5). MPS driver memory stays
~6.06–6.22 GiB across all three shapes (context banks for P832 add ~1.3 GiB in
`current`, absorbed in driver headroom); peak process RSS 2.6 GiB.

Run-to-run variance is real on this machine: the same probe on a cooler state 8 min
earlier measured P400 1.209/0.523/1.637 s and P832 (prefill/candidate) 2.4/1.3 s.
The table above is the conservative (warmer) run; treat absolute MPS numbers as
±20%, or re-run on an idle machine. Lead's earlier numbers (1.04/0.50/1.52 s) are
from a cooler state and remain reproducible within this variance.

`torch.profiler` top-op table for the candidate stage (P832/K10/S64, 2 runs) shows the
stage is **tiny-op launch bound**, not FLOP bound: 2.79 s CPU self-time against
~19k aten ops per run (`aten::view` ×32,690, `aten::as_strided` ×15,448,
`aten::empty_strided` ×7,770, `aten::to` ×27,062 over 2 runs), with the wall time
concentrated in MPS synchronisation points (`aten::_local_scalar_dense`). This is the
mechanism the batching fast path amortises. (torch 2.14 has no
`ProfilerActivity.MPS`, so the table is CPU-side activity only; device time is
captured by the synchronised wall-clock timings.)

## 4. Parity vs CPU FP32 and declared tolerance

- Headline shapes: max |Δlogit| 4.2e-6 … 1.0e-5, rank order identical (all 3).
- 32-question synthetic corpus: 32/32 within `atol=2e-5` (global max **1.09e-5**,
  mean per-question max 2.5e-6), rank stable 32/32, top-1 stable 32/32.
- 32 real keme ARC dev rows (short prompts): max |Δlogit| **6.51e-5** — above the
  strict 2e-5 atol — but rank 32/32 equal, top-1 32/32 equal, **0 decision flips**
  (MPS arm correct 30/32 = CPU reference correct 30/32).
- Verdict: declare `atol=2e-5` for long-context synthetic shapes, and make the
  shipping gate `atol=1e-4 AND rank/top-1 identity` so short real prompts qualify.
  A pure 2e-5-atol gate would falsely reject FP32 MPS on real short prompts.

## 5. Candidate batching on MPS — verdict

Three strategies on identical inputs, one FP32 MPS model, candidate stage only
(prefill cache prebuilt for serial/grouped/padded timing):

| strategy | what it does |
| :--- | :--- |
| `serial` | production default: one candidate at a time |
| `grouped` | loader opt-in `score_candidates_from_cache(..., candidate_batching=True)`: groups candidates by **exact** length, one batched forward per group/step |
| `padded` | this repo's prototype (`mps_batch.py`): pad all candidates to `S_max`, ONE batched forward per step, exact additive causal mask, readout at each row's last real token |

Candidate-stage speedups (32-question corpus, 21 uniform + 11 heterogeneous):

| scope | serial total | grouped | padded |
| :--- | ---: | ---: | ---: |
| all 32 questions | 23.504 s | 14.579 s (**1.61x**) | 12.884 s (**1.82x**) |
| uniform lengths (21) | 18.415 s | 9.867 s (**1.87x**) | 9.956 s (1.85x) |
| heterogeneous (11) | 5.089 s | 4.711 s (**1.08x**) | 2.928 s (**1.74x**) |
| real 32 ARC rows | 15.777 s | 8.973 s (**1.76x**) | 6.182 s (**2.55x**; median row 0.507→0.152 s) |

Headline candidate-stage medians (5 reps):

| shape | serial | grouped | padded | best speedup |
| :--- | ---: | ---: | ---: | ---: |
| P400/K6/S16 | 0.547 s | 0.267 s | 0.278 s | 2.05x (grouped) |
| P832/K10/S64 | 1.736 s | 1.467 s | 1.336 s | 1.30x (padded) |
| P512/K9 hetero 8..24 | 0.936 s | 0.978 s (slower!) | 0.613 s | 1.53x (padded) |

Drift and stability (vs CPU FP32 reference, 32-question corpus):
grouped max 1.31e-5 / padded max 1.31e-5 (< atol 2e-5), rank stable 32/32,
top-1 stable 32/32; real rows: grouped/padded max 6.89e-5 with rank/top-1 stable
and **0 decision flips** in every arm. Batched-vs-serial drift alone is ≤9.7e-6.

Verdict:
1. **Yes, the gain justifies a tolerance-gated fast path on MPS**, but gate it
   (`atol=1e-4 AND rank/top-1 identical`, or 2e-5 for long context) and keep the
   default serial path bitwise-comparable.
2. The loader's existing **`candidate_batching=True` (grouped) is the safe, already-
   implemented arm to expose on MPS**: 1.6–1.9x on uniform/varied shapes, 1.76x on the
   32 real rows, no rank/top-1 changes. Its weakness is heterogeneous lengths: it
   degenerates to K single-row passes (P512/K9: 0.96x, i.e. slightly *slower* than
   serial, because of workspace overhead).
3. The **padded prototype is strictly better for heterogeneous lengths** (1.74x on
   hetero corpus, 2.55x on real rows, 1.53x on P512/K9 vs grouped's 0.96x) and
   equal-or-better elsewhere by ≤4%; integrating it upstream would need the Lead/
   `perf-core` to accept a new loader path (evidence and exact prototype are in
   `scripts/ouro_speed/mps/mps_batch.py`; it does not touch runtime source).
4. E2E expectation, not candidate-stage: because prefill is ~64% of MPS e2e,
   batching gives ~1.2–1.3x e2e on long-context shapes; it gives ~2.5x on the
   short-context real ARC rows where the candidate stage dominates.

## 6. Precision arms

**Verdict: FP32 only. Do not ship bf16 weights or MPS autocast.**

| arm | status | drift vs CPU FP32 | speed (P400/K6/S16 e2e) | MPS driver | safe? |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **FP32 weights/activations/LoRA/head** (shipped behaviour) | measured here | max 4.2e-6 headline, 1.09e-5 corpus-32, rank/top-1 32/32 | 2.026 s (this warm run; 1.637 s cool run) | 6.06 GiB | **YES** |
| bf16 weights + fp32 LoRA/head (manual `.to(bfloat16)`) | measured by Lead's probe (`/tmp/ouro-speed/mps-probe-report.json`, cpu-fp32 artifact, same inputs) | **max 2.58e-2** on P400/K6/S16 (≈1,290x the 2e-5 gate); rank equal on that one synthetic question | 1.385 s prefill 0.938 / candidate 0.423 | 9.08 GiB (allocator holds both dtypes) | **NO** |
| `torch.autocast("mps", dtype=torch.bfloat16)` | **not executed** — final bench window was held by an external process (see §7 note); code-level: `_amp()` is CUDA-only by design, the loader disables autocast inside step-LoRA and the scalar head, and under MPS autocast the additive attention mask would be built in the pre-autocast dtype while projections autocast to bf16 (dtype-mismatch risk) | n/a | n/a | n/a | **NO** (unsupported/unmeasured) |

Why bf16 is not worth it: the drift (2.58e-2) is ~25x the lead's own 2e-5 acceptance
gate and ~400x the fp32 MPS drift (6e-6), while the measured end-to-end gain is
1.385 s vs 1.523 s (~9%) on a one-shot synthetic question, and driver memory is
*higher* (9.08 GiB) because the allocator retains both dtype copies. The same
argument applies a fortiori to autocast. Follow-up command if a future window allows
the full 32-question precision study:
`.venv/bin/python scripts/ouro_speed/mps/probe_precision.py` (takes the bench lock;
writes `precision.json`). The script is committed and ready; it was simply starved
by the external lock holder in the final window.

## 7. Failure / robustness matrix

Everything below was observed during the runs that executed (FP32 and batching
receipts) or measured by the Lead; the four explicitly-marked items were **not run**
because the final bench window was held by an external process (`keme-port-cpu-parity`,
pid 97980). No claim is made for unrun items.

| # | question | evidence | result |
| :--- | :--- | :--- | :--- |
| 1 | device-policy gating | loader source `ouro.py:79-84, 1080-1101`; current revision rejects non-permitted devices before materialisation unless `allow_device_policy_override=True` | `mps-fp32` → mps only; `cpu-fp32` → cpu only. The original silent loophole is closed. |
| 2 | MPS allocator cap under fraction 0.6 | every run: `recommended 17.76 GiB` → cap **10.656 GiB**; after load driver 6.055 GiB; peak driver across shapes **6.22 GiB** | ~4.4 GiB headroom; P832 context bank (~+1.3 GiB) stays inside it. |
| 3 | RSS watchdog | peak process RSS across all runs **2.60 GiB** (MPS allocations are driver-memory, not RSS) | the 14 GiB RSS watchdog never binds for FP32 MPS; the allocator fraction is the binding cap. |
| 4 | one model per process | Lead measurement: loading a second FP32 1.4B model in the same process OOMs under the 0.6 fraction cap | **cap rule: one model/artifact per process on MPS** (documented in §8/§9). |
| 5 | MPS fallback disabled | `PYTORCH_ENABLE_MPS_FALLBACK` unset in every recipe and receipt; the whole scoring path executed on MPS: SDPA math with additive FP32 mask, LayerNorm/GELU/SiLU, embedding, complex RoPE (`view_as_complex`/`view_as_real`), `cat`, `softmax` | no silent CPU ops; with fallback unset an unsupported op raises instead of timing a CPU op (no such raise occurred). |
| 6 | watermark env | required before `import torch`; HIGH=1.0 alone reproduces `invalid low watermark ratio 1.4` (Lead) | use LOW=1.0 / HIGH=1.2 + fraction 0.6. |
| 7 | cold start / load | MPS load 3.20 s (cool state) to 10.92 s (warm, page-cache-miss state); CPU load 2.55 s; earlier Lead run 18.86 s | load is disk-read + FP32 materialisation dominated; treat 3–11 s as the MPS load range on this host. |
| 8 | 50-call leak probe | **NOT RUN** (lock blocked at deadline) | command: see `probe_robustness.py`; earlier behaviour: driver stays 6.06–6.27 GiB across repeated runs in the FP32 probe (no growth observed across 3 shapes × 5 reps + 32 corpus + 32 real questions in one process). |
| 9 | deliberate cap approach / OOM message | **NOT RUN**; Lead's two-model-load OOM is the documented failure mode | command: `probe_robustness.py` (allocates 512 MiB steps until the fraction cap errors, then recovers and writes the receipt). |
| 10 | op-support matrix (`fft`/`svd` probes as fallback proof) | **NOT RUN** (same lock block) | command: `probe_robustness.py` |

Practical MPS failure guidance: if the allocator cap is hit (e.g. two models, or a
very long context with several candidates in flight), the process raises an MPS
allocator `RuntimeError`/aborts rather than silently falling back; run one model per
process, keep fraction ≤0.6, and reuse a prebuilt `ContextKVCache` for repeated
questions on the same context.

## 8. Recommended loader configuration and code-level changes

The current loader (sha `5855451c…`) already contains the device-policy gate the
evidence supports. Recommended final configuration for Apple GPU:

```python
# Artifact (Lead-owned writer): execution.device_policy = "mps-fp32",
# execution.attention = "sdpa-math", adapter/head dtype float32.
load_ouro("/path/to/artifact-last2-mps-fp32", device="mps")  # or device=None -> "mps"
```

1. **Keep** `_DEVICE_POLICY_DEVICES = {"mps-fp32": {"mps"}, "cpu-fp32": {"cpu"},
   "cuda-bf16-autocast": {"cuda","cpu"}}` and the early rejection without
   `allow_device_policy_override=True` (`ouro.py:79-84, 1080-1101`). This removes the
   silent loophole the first Lead probe relied on; cross-policy loads are rejected
   before any tensor materialises (this file §7).
2. **Keep FP32 materialisation on MPS**: `base_dtype = bf16 if cuda else fp32`
   (`ouro.py:1159`) is correct for MPS; add a one-line comment that MPS is FP32 by
   contract. Do not add bf16 defaults.
3. **Keep `_amp()` CUDA-only** (`ouro.py:1236-1238`): no autocast context on MPS
   (§6).
4. **Expose `candidate_batching` as an opt-in on MPS** — it is device-generic already;
   document the measured MPS numbers (§5) and the tolerance gate, default off.
5. **Do not set allocator caps inside the loader.** `torch.mps.set_per_process_memory_fraction`
   is a process-global side effect; document the two watermark env vars + 0.6
   fraction in the MPS usage docs/examples, and (optional, cheap) emit a
   `warnings.warn` if `target.type == "mps"` and
   `PYTORCH_MPS_HIGH_WATERMARK_RATIO` is unset at load time.
6. **Document "one model per process"** as an MPS cap rule (§7): two FP32 1.4B
   models exceed the 0.6 fraction cap.
7. **No source change needed for tolerance**: record `atol=2e-5` long-context /
   `atol=1e-4 + rank/top-1 gate` real short prompts in the MPS docs; parity tooling
   (`scripts/ouro_speed/mps/replay_parity.py`) enforces it.

## 9. What is NOT safe to ship

- **bf16 weights/activations as the MPS default** — see §6; drift far above the
  2e-5 gate while the speed gain is not worth it.
- **autocast on MPS as the default** — see §6.
- **Reliance on default macOS MPS watermarks** (LOW=1.4/HIGH=1.7) with a 5.4 GiB
  FP32 model on a 24 GiB machine — allocator can reserve past safe memory; always
  set LOW=1.0/HIGH=1.2 before `import torch` and fraction 0.6.
- **Two artifacts/models in one process** on MPS (§7).
- **`cpu-fp32` artifact executed on MPS without the explicit override** — the loader
  now blocks it; keep that behaviour (it was the only way the original probe loaded).
- **`candidate_batching=True` enabled by default** without a tolerance gate —
  it changes results at ~1e-5 and can be slower than serial for heterogeneous
  candidate lengths (0.96x).

## 10. Guard recommendations from the workspace-reuse diagnostic

The transient loader revision (`a35d8ed0…`, 18:02) corrupted candidates 1..K while
candidate 0 stayed exact, because a pooled K/V workspace skipped re-copying the
per-layer context prefix on candidate reuse. `diagnostic-workspace-reuse.json`
records: current revision = 4.5e-7 max |Δ| vs committed baseline, rank equal;
fresh-pool per candidate = same; buggy-revision = 0.596 max |Δ|.

Add to the loader test suite (cheap, CPU):
1. A golden CPU FP32 logits check at **P400/K6/S16 with K≥6 equal-length
   candidates** against stored logits (`atol=2e-5`, rank/top-1 identical). This
   catches candidate-2+ corruption; K=1 or serial-vs-batched equality alone does
   **not** catch it (both share the same pool).
2. A `candidate_batching=True` vs serial check at the same shape (`atol=2e-5`,
   rank/top-1 identical) — catches future batching regressions.
3. Record loader/artifact SHA in every performance receipt and flag receipts whose
   inputs were produced by a different revision.

## 11. Receipts, scripts, reproduction

Scripts (this directory; no runtime edits, no network, dependency-free beyond
torch/wyat):
`mps_common.py` (caps/env/lock/watchdog/timing/parity/fingerprint),
`ref_cpu.py` (CPU FP32 reference + frozen question set),
`probe_fp32.py`, `probe_batching.py` (+`mps_batch.py` padded prototype),
`probe_precision.py`, `probe_robustness.py`, `diag_workspace_reuse.py`,
`replay_parity.py`.

Two exact reproduction command lines (each self-caps: lock, nice 10, threads 6,
RSS watchdog, MPS fraction 0.6, watermarks exported before `import torch`; expected
numbers are the medians in §3/§5 — absolute times vary ±20% with machine warmth):

```bash
cd /Users/teilomillet/Code/wyat

# 1) MPS FP32 extended numbers + CPU parity. Expected: P400/K6/S16 e2e ~1.6-2.0 s
#    (CPU 3.34 s), P832/K10/S64 ~3.8 s, P512/K9 hetero ~2.5 s; max |dlogit| vs CPU
#    fp32 4.2e-6 / 6.1e-6 / 1.0e-5 with identical rank; driver 6.06 GiB after load.
.venv/bin/python scripts/ouro_speed/mps/ref_cpu.py && \
.venv/bin/python scripts/ouro_speed/mps/probe_fp32.py

# 2) Candidate batching on MPS. Expected candidate-stage speedups: grouped 1.61x
#    corpus / 1.87x uniform / 1.08x hetero / 1.76x real rows; padded prototype
#    1.82x corpus / 1.74x hetero / 2.55x real rows; drift <=1.31e-5, rank/top-1
#    32/32 unchanged, 0 real decision flips.
.venv/bin/python scripts/ouro_speed/mps/probe_batching.py
```

All scripts: `ref_cpu.py` (CPU reference + frozen inputs, ~3 min), `probe_fp32.py`,
`probe_batching.py`, `probe_precision.py`, `probe_robustness.py`,
`diag_workspace_reuse.py`, `replay_parity.py` (pure comparison, lock-free), plus
`mps_common.py` (caps/lock/watchdog/timing/parity) and `mps_batch.py` (padded
prototype).

## 12. Limitations of this evidence

- Absolute MPS timings vary ~±20% with machine/thermal state; the §3 table is the
  conservative (warm) run, the cool-state repeat is quoted alongside. Parity numbers
  are stable across runs.
- Precision sub-arms: only FP32 was measured in this task's window; bf16 numbers are
  the Lead's probe on the cpu-fp32 artifact (same weights/inputs); the autocast MPS
  arm and the 32-question bf16 rank/flip study were starved by an external bench-lock
  holder. Do not read a "bf16 passed" result into this file — it did not.
- Robustness items 8–10 (leak/cap-approach/op-matrix) were not executed for the same
  reason; the script is ready (`probe_robustness.py`).
- The loader was edited concurrently during this task (revisions a35d8ed0 → fdcc7630
  → 5855451c; `ouro_artifact.py` changed during the batching run). Every receipt pins
  loader SHA at start/write and flags mid-run changes; re-run after integration on the
  final revision.

Receipts in `artifacts/ouro-speed-20261021/mps/`: `fp32-extended.json`,
`batching.json`, `precision.json`, `robustness.json`, `reference-cpu-fp32.json`,
`reference-cpu-fp32-run.json`, `diagnostic-workspace-reuse.json`,
`inputs/question-set.json`, and `logits/{mps-fp32-serial,mps-batching,mps-precision}.json`.
All receipts record the profiling notice, host/torch/MPS environment, caps, artifact
fingerprints (manifest + tensor hashes + declared base checkpoint hash), and loader
SHA-256/git HEAD at start and receipt-write.
