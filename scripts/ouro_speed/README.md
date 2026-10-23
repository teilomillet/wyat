# Ouro-on-Mac speed harness (profiling only)

> **PROFILING ONLY — NOT A RELEASE ARTIFACT.** The artifact built by
> `build_local_artifact.py` is assembled from local, non-release weights
> (ByteDance/Ouro-1.4B snapshot + keme `last2` adapters). Nothing in this
> directory produces or publishes a release artifact, and the numbers below are
> synthetic-token profiling numbers, not product benchmarks.

Reproducible benchmark, parity and device-transfer harness for `OuroScorer`
(`wyat/loading/ouro.py`) on Apple silicon. Everything here is additive: these
scripts do not modify `wyat/`, `keme/`, docs or packaging.

Files:

| file | purpose |
| :--- | :--- |
| `_common.py` | caps (RSS watchdog, lock, nice/threads/MPS), provenance, deterministic inputs, tolerance comparator, receipts |
| `build_local_artifact.py` | reproducible local `cpu-fp32` profiling artifact build (refuses overwrite, hashes inputs) |
| `bench.py` | load / Stage-1 prefill / Stage-2 candidate-from-cache / end-to-end / K-scaling / context-grid benchmark, live progress, JSON+markdown receipt, `--profile` aten breakdown |
| `parity.py` | compare logits files (or run two backends sequentially) against a declared tolerance; bitwise/rank/top-1 evidence |
| `audit_transfers.py` | parameter/buffer placement, monkeypatched `Tensor.to`/`.cpu()`/`.item()`/`.tolist()` counters, `torch.cat` watcher, KV-bank device/dtype/contiguity audit, profiler cross-check |
| `make_results_table.py` | render `FINAL-RESULTS.md` (arm x workload x median e2e x speedup x drift x memory) from receipts, offline |
| `test_ouro_speed_harness.py` | tiny self-tests that never load the 1.4B model |

## Environment

- Repo: `/Users/teilomillet/Code/wyat`, interpreter `.venv/bin/python`
  (Python 3.12, torch 2.14.1, transformers 5.18).
- Apple M4 Pro: 12 cores, 24 GiB unified memory, macOS. No network needed.
- Model: `OuroScorer`, 1.4B params, 24 layers, 4 native recurrent steps,
  context prefill → 2-step candidate scoring, FP32 compute on CPU and MPS.
- Artifact inputs:
  - base snapshot
    `artifacts/ouro-study-20261018/sources/core-model-snapshots-v1/Ouro-1.4B/`
    (BF16 `model.safetensors`);
  - keme checkpoint
    `/Users/teilomillet/Code/keme/.keme/ouro/experiments/pretrained-branch-adaptation-trained-v2/last2/final.pt`.
    Its adapter keys start at `model.layers...` and are prefixed with
    `backbone.` before `write_ouro_artifact`; the scalar-head keys are prefixed
    with `head.` (the writer enforces that prefix itself).

## Mandatory resource caps (enforced in the tool, not advisory)

Every heavy run (`bench.py`, `parity.py --run-*`, `audit_transfers.py`,
`build_local_artifact.py`) enforces:

| cap | default | hard limit |
| :--- | ---: | ---: |
| exclusive lock `/tmp/ouro-speed/BENCH_LOCK` (`mkdir`, retry every 10 s, removed on exit) | always on | `--no-lock` only for orchestrated children while the parent holds it |
| `nice` | 10 | 0..20 |
| torch intra-op threads | 6 | **6** (`--threads` >6 is rejected) |
| RSS watchdog (hard abort, exit code 42) | 10 GiB | **12 GiB** (`--rss-cap-gib` >12 rejected) |
| MPS process memory fraction | 0.6 | **0.6** (`--mps-fraction` >0.6 rejected) |
| MPS watermark env (set before importing torch) | `PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.2` | `PYTORCH_ENABLE_MPS_FALLBACK` unset (a silent CPU fallback would invalidate device evidence) |

The watchdog polls RSS every 250 ms, writes the partial receipt with
`status=aborted, abort_reason=rss_cap_exceeded`, and hard-exits from its thread
(a long torch op cannot be interrupted safely from Python). **Sequential only:**
never start a second model-loading process while a lock holder runs. The MPS
memory fraction is 0.6 of `torch.mps.recommended_max_memory()` (≈10.7 GiB on
this 24 GiB machine); `driver_allocated_memory()` includes reserved pools and is
always recorded alongside `current_allocated_memory()`.

Live progress is printed for every warmup/repeat:
`[  12.34s] rss_now= 8.12GiB rss_peak= 8.44GiB | stage=prefill_stage1 iter=2/5 dt=1.961`.

## Build the profiling artifact (once)

```bash
cd /Users/teilomillet/Code/wyat
.venv/bin/python scripts/ouro_speed/build_local_artifact.py \
  --out /tmp/ouro-speed/artifact-last2-cpu-fp32-repro \
  --receipt artifacts/ouro-speed-20261021/artifact-build-receipt.json
```

- refuses to overwrite an existing `--out` (exit 2);
- records SHA-256 for every input file (base snapshot + keme checkpoint, plus the
  keme sidecar `receipt.json` when present), the derived adapter/head tensor
  mapping hashes, the manifest `execution` dict, and the output file hashes;
- by default compares the produced file hashes against the existing scratch
  artifact `/tmp/ouro-speed/artifact-last2-cpu-fp32` and records
  `reproducible_build.files_sha256_match` (byte-identical build evidence).

The standardized runs below use the canonical scratch artifact
`/tmp/ouro-speed/artifact-last2-cpu-fp32` (model id
`ouro-3e007a0a92989f6174f80b7d878ffdfae17466763f4842570a74ef5af9c3a73a`) so
numbers are directly comparable with the Lead's baseline; the repro build is the
reproducibility evidence.

## Standardized CPU protocol

Warmups ≥ 1 (default 2), ≥ 5 timed repetitions (default 5), report
median/min/max/raw samples, sequential under the exclusive lock, nice 10,
threads 6, RSS cap 10 GiB.

```bash
# Baseline A: context 400, K=6, S=16 (matches the Lead baseline)
.venv/bin/python scripts/ouro_speed/bench.py \
  --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device cpu \
  --context-tokens 400 --candidates 6 --candidate-length 16 \
  --warmups 2 --repeats 5 --seed 20261021 --profile \
  --json-out artifacts/ouro-speed-20261021/baseline-cpu-p400-k6-s16.json \
  --markdown-out artifacts/ouro-speed-20261021/baseline-cpu-p400-k6-s16.md \
  --save-logits artifacts/ouro-speed-20261021/logits-cpu-p400-k6-s16.pt

# Baseline B: context 832, K=10, S=64
.venv/bin/python scripts/ouro_speed/bench.py \
  --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device cpu \
  --context-tokens 832 --candidates 10 --candidate-length 64 \
  --warmups 2 --repeats 5 --seed 20261021 \
  --json-out artifacts/ouro-speed-20261021/baseline-cpu-p832-k10-s64.json \
  --markdown-out artifacts/ouro-speed-20261021/baseline-cpu-p832-k10-s64.md \
  --save-logits artifacts/ouro-speed-20261021/logits-cpu-p832-k10-s64.pt

# Scaling grid: K=1,2,4,6,8 x context 256/400/832 (S=16)
.venv/bin/python scripts/ouro_speed/bench.py \
  --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device cpu \
  --context-tokens 400 --candidates 8 --candidate-length 16 \
  --warmups 1 --repeats 3 --scale-contexts 256,400,832 \
  --scale-candidates 1,2,4,6,8 --scale-warmups 1 --scale-repeats 3 \
  --json-out artifacts/ouro-speed-20261021/scaling-cpu.json \
  --markdown-out artifacts/ouro-speed-20261021/scaling-cpu.md
```

Receipt fields worth reading:

- `stages.*`: `median_s` / `min_s` / `max_s` / `samples_s` / `n` for `load`,
  `prefill_stage1`, `candidate_scoring_from_cache`, `end_to_end`; `load` also
  carries `peak_rss_gib`;
- `stages.cached_vs_e2e`: bitwise check that Stage 1+2 equals end-to-end;
- `scaling.candidate_count` and `scaling.context_grid`: the tables;
- `inputs`: generator spec (`torch.Generator(device='cpu').manual_seed(seed)`),
  context/candidate lengths, and `input_sha256` over the token bytes — all
  deterministic and tokenizer-free;
- `provenance.source_start` / `source_end`: git revision + `wyat/loading/ouro.py`
  SHA-256 at start and end; `source_changed_during_run` flags a concurrent edit;
- `memory`: peak RSS, cap, watchdog peak, MPS current/driver memory;
- `logits`: values + SHA-256 (plus `saved` file hash when `--save-logits`).

**Provenance rule:** other agents edit `wyat/loading/ouro.py` during benchmark
work. Every receipt records its SHA-256; if it changes, the baselines must be
re-run. A `source_changed_during_run=true` receipt is not comparable.

## Benchmark-integrity gates (mandatory)

An intermediate `wyat/loading/ouro.py` revision (sha `a35d8ed0…`) silently
corrupted multi-candidate logits for K>=2 via a pooled K/V workspace bug
(candidate 0 exact, candidates 1+ off by up to 0.596; K=1 calls were unaffected).
The harness therefore enforces three independent gates on every run:

1. **Golden check** — `--check-baseline golden/<workload>.json` compares the
   freshly computed Stage-2 logits against a stored golden reference and fails
   (exit 5) if `max abs delta > 2e-5` **or** the rank order/top-1 changed. Golden
   files live in `artifacts/ouro-speed-20261021/golden/` and record the loader
   SHA-256 and full-precision logits. `--baseline-key dot.path` selects an entry
   from a rich JSON golden (e.g. mps-probe's `headline_logits.p400-k6-s16`).
   Create a golden with `--save-golden <path>` from a stable, gate-passing run.
2. **Multi-candidate self-consistency** (on by default; `--skip-self-consistency`
   to disable) — scores every candidate alone through the same cache and requires
   the same logits/rank as the K-candidate call. This catches the pooled-KV bug
   class without any stored reference, for every arm including
   `candidate_batching=True`.
3. **Loader stability** — every receipt records `wyat/loading/ouro.py` SHA-256 +
   git revision at start **and** end. If they differ the receipt is marked
   `provenance.unstable_loader=true` and the process exits 4; such a receipt must
   never be used as a baseline. The final suite additionally aborts mid-run if
   the loader SHA moves.

Exit codes: `0` ok, `1` unexpected error, `2` usage, `3` lock timeout, `4`
unstable loader, `5` golden-baseline mismatch, `6` self-consistency failure,
`42` RSS watchdog abort. `parity.py` exits `1` on tolerance/rank violations.

Frozen token IDs can be used verbatim instead of synthetic generation (this is
how the standardized hetero arm matches mps-probe's question set):

```bash
.venv/bin/python scripts/ouro_speed/bench.py   --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device cpu   --tokens-file artifacts/ouro-speed-20261021/mps/inputs/question-set.json   --tokens-name p512-k9-hetero-8to24   --warmups 2 --repeats 5   --json-out artifacts/ouro-speed-20261021/bench-cpu-default-p512-k9-hetero.json
```

The receipt records the question-set SHA-256, the selected entry and both
`input_sha256` (harness identity hash) and `legacy_concat_sha256`
(the Lead-compatible `sha256(cat(tokens))`, e.g. `61aa0f6d…` for the canonical
P400/K6/S16 seed-20261021 workload).

## CPU-vs-MPS runs and parity

MPS runs use the canonical `mps-fp32` artifact
`/tmp/ouro-speed/artifact-last2-mps-fp32` (whose weight files are byte-identical
to the `cpu-fp32` artifact); the loader's device policy selects MPS by default.
Cross-policy runs (e.g. a `cpu-fp32` artifact on MPS) require the loader's
explicit opt-in, exposed here as `--allow-device-override`, `--policy-override`
or `--device-policy-override` (all map to `allow_device_policy_override=True`).
`mps-probe` owns the extended batching/precision/robustness experiments under
`scripts/ouro_speed/mps/`; this harness produces the standardized receipts,
the golden-gated parity evidence, and the final results table.

```bash
.venv/bin/python scripts/ouro_speed/bench.py \
  --artifact /tmp/ouro-speed/artifact-last2-mps-fp32 --device mps \
  --context-tokens 400 --candidates 6 --candidate-length 16 \
  --warmups 2 --repeats 5 --seed 20261021 \
  --rss-cap-gib 12 \
  --json-out artifacts/ouro-speed-20261021/baseline-mps-p400-k6-s16.json \
  --markdown-out artifacts/ouro-speed-20261021/baseline-mps-p400-k6-s16.md \
  --save-logits artifacts/ouro-speed-20261021/logits-mps-p400-k6-s16.pt
# ... and the P832/K10/S64 variant as above.
```

Parity between CPU and MPS logits (declared contract: **max abs delta ≤ 2e-5**
on candidate scores; full rank order and top-1 must be identical for decision
parity; relative deltas reported but not gated):

```bash
.venv/bin/python scripts/ouro_speed/parity.py \
  --a artifacts/ouro-speed-20261021/logits-cpu-p400-k6-s16.pt \
  --b artifacts/ouro-speed-20261021/logits-mps-p400-k6-s16.pt \
  --atol 2e-5 --fail-on-rank-change \
  --label-a cpu-fp32 --label-b mps-fp32 \
  --json-out artifacts/ouro-speed-20261021/parity-cpu-vs-mps-p400-k6-s16.json
```

`parity.py --run-a ARTIFACT --run-b ARTIFACT --device-a cpu --device-b mps`
instead runs `bench.py` twice sequentially (the parent holds the lock; the
children get `--no-lock`) and compares the saved logits. Cross-backend *and*
cross-repo logits are **not** expected to be bitwise identical: CPU fp32 and MPS
fp32 accumulate in different orders, and keme's canonical CPU scorer keeps BF16
base weights while wyat's CPU loader casts to FP32. Exit code is 0 on pass, 1 on
a tolerance/rank violation, 2 on usage errors.

## Final controlled suite and results table

Run only after the loader is frozen (Lead confirms the SHA) and while no other
agent is running model loads (mps-probe's window closed). The runner aborts if
the loader SHA changes or any receipt fails its integrity gates:

```bash
EXPECTED_LOADER_SHA=<frozen-sha> bash /tmp/ouro-speed/run_final_suite.sh
```

Latest run: frozen loader sha `5855451cb1e3ede274f80111139eb99ed12ab40a04eeabde944cd48b4adda41f`
(git `7646d95f`), core matrix only. The K x context scaling grid was excluded
from that time-boxed run per the Lead (earlier grid receipt was contended and
errored; see `FINAL-RESULTS.md` notes). The runner executes, sequentially under
the exclusive lock: CPU default P400/K6/S16,
CPU default P832/K10/S64, CPU default frozen hetero P512/K9, the same three with
`--candidate-batching`, the K x context scaling grid, the CPU transfer audit,
then MPS fp32 default and batching arms (P400/K6/S16, P832/K10/S64), the MPS
transfer audit, CPU-vs-MPS parity, batching-vs-default drift, mps-probe
reference-vs-MPS collection parity, and finally writes
`artifacts/ouro-speed-20261021/final-suite-manifest.json` and
`FINAL-RESULTS.md` via:

```bash
.venv/bin/python scripts/ouro_speed/make_results_table.py   --manifest artifacts/ouro-speed-20261021/final-suite-manifest.json   --artifacts-dir artifacts/ouro-speed-20261021   --out artifacts/ouro-speed-20261021/FINAL-RESULTS.md
```

`FINAL-RESULTS.md` is the one-table summary: arm x workload x median e2e,
speedup vs CPU default, max |Δlogit| (rank/top-1 flips), memory high-water
(RSS or MPS driver) and the integrity status, plus the frozen loader SHA-256.

## Device transfer audit ("data is transiting correctly")

```bash
.venv/bin/python scripts/ouro_speed/audit_transfers.py \
  --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device cpu --include-load \
  --json-out artifacts/ouro-speed-20261021/audit-transfers-cpu.json
.venv/bin/python scripts/ouro_speed/audit_transfers.py \
  --artifact /tmp/ouro-speed/artifact-last2-cpu-fp32 --device mps --include-load \
  --rss-cap-gib 12 \
  --json-out artifacts/ouro-speed-20261021/audit-transfers-mps.json
```

It reports: device/dtype/shape/contiguity of every parameter and buffer;
Python-level `Tensor.to` / `.cpu()` / `.item()` / `.tolist()` / `.numpy()`
counts split into device-changing vs same-attribute copies, with call sites;
`torch.cat` outputs in Stage 2 (device/dtype/contiguity, transfer bytes, mixed
inputs); retained `ContextKVCache` bank placement; a `torch.profiler`
cross-check of `aten::_to_copy` / `aten::copy_` / `aten::_local_scalar_dense`
(internal copies and implicit scalar syncs); MPS current/driver memory per
phase. Hard anomalies (any tensor on the wrong device, cat outputs off-device,
mixed cat inputs) exit 1; sync points and non-contiguity are warnings.

## Tiny self-tests (no model load)

```bash
cd /Users/teilomillet/Code/wyat
.venv/bin/python -m pytest scripts/ouro_speed/test_ouro_speed_harness.py -q
```

Covers all CLIs' `--help`, argument/cap rejection, RSS watchdog raise + thread
latch, lock acquire/timeout/stale-break, deterministic inputs, tolerance
comparator (pass/fail/rank-change/bitwise/length), transfer + cat counter
bookkeeping and restoration, KV-bank inspection, parity file handling and CLI
exit codes, markdown rendering, and the build script's overwrite refusal.

## Receipts index

`artifacts/ouro-speed-20261021/`:

| file | content |
| :--- | :--- |
| `artifact-build-receipt.json` | reproducible build: input SHA-256s, derived tensor hashes, manifest execution, output hashes, reproducibility comparison |
| `baseline-cpu-p400-k6-s16.{json,md}` | CPU baseline A (+ saved logits) |
| `baseline-cpu-p832-k10-s64.{json,md}` | CPU baseline B (+ saved logits) |
| `scaling-cpu.{json,md}` | K = 1,2,4,6,8 × context 256/400/832 table |
| `baseline-mps-p400-k6-s16.{json,md}`, `baseline-mps-p832-k10-s64.{json,md}` | MPS fp32 receipts with driver/current memory |
| `parity-cpu-self-*.json` | determinism: same CPU run twice, expect bitwise equality |
| `parity-cpu-vs-mps-*.json` | cross-backend parity against atol 2e-5 + rank/top-1 |
| `golden/*.json` | canonical full-precision CPU FP32 logits + input spec + loader SHA (golden-gate references) |
| `batching-{cpu,mps}-*.json` | `candidate_batching=True` arms (same workloads, drift parity reports) |
| `bench-cpu-default-p512-k9-hetero.json` | frozen hetero K=9 lengths 8..24 workload (default and batching) |
| `parity-*-vs-*.json` | batching drift, CPU-vs-MPS, and mps-probe reference-vs-MPS collection parity |
| `audit-transfers-{cpu,mps}.json` | device/transfer audit evidence |
| `final-suite-manifest.json`, `FINAL-RESULTS.md` | final controlled suite manifest + summary table |
| `CPU-vs-MPS-placeholder.md` | placeholder if MPS receipts are pending |

All receipts embed `notice: PROFILING ONLY - NOT A RELEASE ARTIFACT`, the
command line, environment (torch/transformers/python/platform/loadavg), caps,
artifact `model_id` + manifest hashes, input `input_sha256`, and the
`wyat/loading/ouro.py` SHA-256 + git revision at start and end.

## Notes

- The harness has no dependency beyond the repo's existing torch/transformers;
  no packages are added and no network is used.
- If the lock exists because another agent is running, `bench.py` waits and prints
  `[lock] waiting ...`; a stale lock from a dead PID is broken only after
  `--lock-timeout-s`-independent age > 3600 s.
- MPS receipts must be produced while no other model-loading process runs (the
  lock enforces it): unified memory contention invalidates both timing and the
  memory high-water.
