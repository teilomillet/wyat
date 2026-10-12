# Native Wyat MLX runtime — 2026-10-12

The native trained Wyat model now has a local MLX scorer. It preserves the native task encoder, independent causal question rows, complete ordered option endpoints, unmerged FP32 q/v adapters, FP32 shared head, and Choice/Score/Noul readouts. It does not change the architecture or require retraining.

## Local use

This runtime is available from the source checkout. Native modules are not currently admitted to the public wheel's explicit inventory, so this is not a claim about `pip install wyat` availability.

```python
from wyat.loading.mlx import load_native_mlx

model = load_native_mlx(
    "artifacts/native-mlx-runtime-admission-20261012/model",
    device="gpu",
)
record = {
    "state": "The invoice is overdue.",
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "Choose the invoice status.",
            "criteria": {"paid": "Paid in full.", "overdue": "Past due."},
        }
    },
}
logits = model.logits(record)
answers = model.predict(record)
print(answers["status"].to_dict())
```

`metal=True` requests the existing qualified Metal base projections. `model.metal_enabled` reports actual admission. Unsupported MLX versions, GPU architectures or optional compiler availability use ordinary MLX. Ordinary MLX is the default; no precision, thread or cache environment is changed by the loader. Both paths keep the adapters separate and execute their original expression `base(x) + 2 * ((float32(x) @ down.T) @ up.T)`.

`responsive=True` is the default. Requests to one scorer are serialized through synchronous result extraction. GPU inference finishes each four-layer chunk and requests a 1 ms idle gap before continuing. This gives other applications submission opportunities while preserving the maintained forward order and tensor shapes. It is cooperative scheduling, not a hard responsiveness or memory guarantee: loading, one long attention kernel, other processes and multiple scorer instances remain outside that bound. `responsive=False` uses the maintained backbone directly. Neither setting changes process-wide MLX allocator, cache, wired-memory, precision or CPU-thread settings.

`device="cpu"` permits MLX CPU mechanics checks. Existing PyTorch CPU/CUDA callers continue to use `wyat.loading.native.load_native`; their inference implementation is unchanged. CUDA performance was not measured on this Mac.

No input is truncated. The artifact's context and padded-question batch budgets are checked before inference, and callers may select a smaller `max_length`. Right padding retains the reference's exact width; this implementation does not change grouping or reuse a prefix.

## Artifact identity

The original v0.2a artifact binds the release's SPDX-attributed task source, while this checkout's task source has a different byte hash. The task bodies are AST-identical. The original validator still rejects that mismatched contract; no hash exception was added.

`benchmarks/analysis/native_mlx_rebind_20261012.py` verified the bundled runtime inventory, original upstream source and unchanged executable contracts, then created an independent local copy with explicit current-runtime provenance. Its six declared model/config/tokenizer files are byte-identical to the original. Both retain the weight-derived model ID `native-68cdf47a3012971214cdf062fb3f1e023769e997bbce5761d016fa9c0e4433b2`.

- Original manifest SHA-256: `3db77b2bafdaf536a1398327df08a9b79321e67e3c5526f7f44fbd4a96fd0ffa`.
- Derived manifest SHA-256: `effbd0fc5fb18d312e20bf617593d9048d545d8b015bedf59df9a692b3ce7104`.

The frozen original release was not edited. The shared validator was extracted into a framework-free owner; historical shared-head and task identities, inventory checks and negative behavior are preserved. The native Torch execution classes/functions are AST-identical to the previous implementation.

## Verification

The existing maintained-source gate passed 622 tests, Ruff formatting/lint and strict types before the native scheduling addition. The final focused native task/artifact/MLX/resource/rotary suite passed **206 tests in 3.90 seconds** on CPU/Metal, covering seeded FP32/mixed-BF16 artifacts, separate nonzero adapters, typed outputs, reversed Noul order, independent rows, context/batch-budget rejection, malformed tensors, strict source/file identity, chunk scheduling, same-instance concurrency, lock release on exceptions and an independent Transformers rotary oracle through position 8,191. These tests establish mechanics, not trained-model quality. The earlier temporary test logs were lost during interruption; fresh benchmark receipts below persist in the repository.

Final scoped Ruff lint and formatting pass for 15 files. Strict mypy passes for 14 production/test/research files. The frozen benchmark harness is excluded from that successful type gate: including it reports seven annotation/indexing errors; it was kept byte-identical to the pre-measurement harness. `final-source-verification.json` records unchanged protected hashes, unchanged native Torch execution ASTs and the production source fingerprints. Independent review found no material production blocker. CUDA throughput itself remains unmeasured.

The trained-model benchmark protocol is recorded in `docs/analysis/native-mlx-benchmark-protocol-20261012.md`. Its 32 reused DEV records and synthetic boundary cases assess implementation parity, not held-out accuracy. Fresh measured outputs and failed gates are retained under `benchmarks/analysis/results/native-mlx-20261012/`.

## Frozen full-model measurements

The Apple M4 Pro (24 GiB) run used the original pinned release runtime as its PyTorch CPU oracle, with four intra-op threads and one inter-op thread. MLX used its production default precision environment, responsive scheduling, a standalone 6 GiB allocator guideline and 256 MiB cache guideline. Requests ran sequentially at low process priority, with 250 ms pauses outside timers. These benchmark guidelines are not library defaults or total RSS limits. No wired-memory limit was increased.

All three arms completed all 38 cases (54 question rows, 1,775 candidate logits) without exceptions, plus 18 timing observations each. Median milliseconds from three repetitions:

| Path | CPU, four threads | Ordinary MLX | Metal MLX |
| --- | ---: | ---: | ---: |
| 77-option singleton, encoded | 1,869.62 | 737.79 | 702.87 |
| Eight short rows, encoded | 2,821.87 | 1,173.33 | 1,078.82 |
| Eight mixed rows, encoded | 3,398.87 | 1,542.09 | 1,467.29 |
| 77-option singleton, public | 1,933.01 | 739.54 | 713.24 |
| Eight short rows, public | 2,775.11 | 1,149.89 | 1,083.89 |
| Eight mixed rows, public | 3,510.46 | 1,560.15 | 1,516.66 |

These initial measurements use maintained fast RoPE. They are exploratory speed measurements against the resource-restrained four-thread CPU baseline, not the fastest possible CPU configuration. **Neither initial MLX arm passed quality admission.** Both pass selected Choice answers, numerical tolerances, public readout consistency and all five negative cases, but fail the exact complete-ranking gate on the synthetic 255-option case (`single-34`). Options `k133` and `k247` exchange ranks 54/55: the CPU gap is approximately +2.15e-6 and MLX approximately -8.58e-6. The selected answer remains `k0`. The gate has not been relaxed.

The saved-state diagnostic (`rank-cpu.json`, `rank-metal.json`, `rank-head.json` and corresponding NPZ captures) localizes the disagreement to backbone hidden states. Passing both captures through the same CPU head still changes that ranking. On identical MLX hidden states, CPU and MLX heads retain all ranks and differ by at most 9.54e-7 in logits. The exact executed diagnostic source is archived as `rank-diagnostic.executed.py`; later current-script changes only add type annotations and rename an MLX local to satisfy static checks.

`precision-screen.json` then compared four source-motivated variants on one model: baseline, explicit RMSNorm, explicit RoPE, and both. Baseline and RMSNorm alone failed the same ranking gate. Explicit RoPE and both passed all 38 cases. The narrower RoPE correction was integrated: FP32 reciprocal/power frequency construction, FP32 position multiplication and separate rotation products before addition. One position table is shared across all layers and replaced on width changes; no trained tensor, head, candidate ordering or tolerance changes.

Fresh integrated ordinary MLX confirmation (`ordinary-rope.json` and `ordinary-rope-quality.json`) passes all 38 cases, all 18 timing observations, public readout consistency and all five negative checks against the original CPU oracle. Across the full cases, maximum errors are 8.44e-5 in logits, 5.62e-6 in probabilities and 1.13e-5 in typed values. Median encoded latency is 785.66 / 1,243.40 / 1,686.81 ms for singleton / short batch / mixed batch; public latency is 786.30 / 1,250.74 / 1,677.10 ms. This supports implementation parity on the frozen DEV/stress corpus, not held-out accuracy or universal numerical equivalence.

The separately loaded integrated Metal arm (`metal-rope.json`, `metal-rope-quality.json`) also passes the full unchanged quality gate. Median encoded latency is 769.82 / 1,211.26 / 1,621.24 ms and public latency 801.43 / 1,243.81 / 1,639.53 ms. These three-repeat observations do not override the earlier failed promotion gate; Metal remains explicitly optional. After the rotary correction, peak MLX allocation is 3.293 GB ordinary and 2.412 GB Metal, with the same approximately 2.398 / 1.518 GB live weights and constants. Neither a hard allocator cap nor a universal peak guarantee is claimed.

In the initial fast-RoPE experiment, ordinary MLX peak active allocation was 4.064 GB and Metal 3.183 GB; end-of-run live allocations were 2.398 GB and 1.517 GB respectively. Metal reduced those MLX allocation measures by about 21.7% and 36.7%, while its warm RSS was higher. The quality-passing corrected path has the separate lower peaks reported above; neither experiment establishes a universal memory improvement. MLX allocator bytes and process RSS must not be added. Earlier system snapshots had zero swap counters, but a later snapshot during the final expanded run showed nonzero system-wide swap and compression. Their cause is not established; no claim of swap-free execution or measured interactive responsiveness follows from these benchmarks.

## Bottleneck diagnosis

`profile-metal.json` preserves before/after outputs and a fenced per-stage diagnostic for the 77-option singleton and mixed eight-row batch. All diagnostic outputs are bit-identical to the ordinary native Metal forward. Synchronization inflated elapsed time by 1.345x and 1.307x, so these fractions are conditional diagnostics, not an exact decomposition of uninstrumented latency.

Base projections account for 50.7% and 65.4% of the instrumented work. Attention as a whole accounts for 64.2% and 52.7%; MLP work accounts for 32.9% and 44.4%. Adapter work is about 10.5% and 10.3% and overlaps attention. The head is negligible. Amdahl therefore points toward improving base projections across both major blocks. It does not support spending the main effort on tokenization or prefix reuse.

The completed initial ordinary/Metal comparison retains 15 alternating pairs on each timing shape (`paired-default.json` and `paired-default-summary.json`). All 135 three-way quality comparisons on those timing shapes pass. Metal median speedups are 1.040x / 1.044x / 1.051x, but the first two miss the 1.05 median gate and the mixed-batch fifth percentile is 0.930x. The optional Metal path is not promoted by those measurements.

The bounded base-projection screen (`tiles-default.json`) changes only the existing shared owner's direct tile, preserving split-K and weights. Larger 32x32/4-group and 32x64/4-group tiles pass baseline parity on all measured timing outputs, but median speedups range from 0.977x to 0.996x across five alternating pairs per shape/candidate. They are rejected; production Metal kernels remain unchanged. Those exploratory comparisons are against the initial Metal baseline and do not independently establish CPU parity.

The explicit rotary correction added synchronization, so a bounded fusion experiment next tested replacing its two separately rounded products and addition with one safe-math Metal kernel, with contraction disabled. The five-pair screen passed full-model quality and looked promising, but the longer confirmation rejected the candidate:

| Shape | Complete alternating pairs | Median speedup | Fifth percentile | Speed admission |
| --- | ---: | ---: | ---: | --- |
| Singleton | 15 | 1.064x | 1.009x | Pass |
| Eight short rows | 15 | 1.053x | 0.981x | Fail |
| Eight mixed rows | 15 | 1.041x | 0.895x | Fail |

Every shape had to reach a median of at least 1.05x and a fifth percentile of at least 1.0x. Full-model quality passed, but speed admission failed. The production scorer retains explicit RoPE; the unused helper and its candidate-only tests were removed. An independent read-only reviewer reconstructed the same decision from raw timings. `rope-fusion-preservation-decision.json` makes the speed rejection explicit: the original confirmation receipt's top-level `passed` describes quality only.

The numerical controls also retain contrary evidence. All eight original independent NumPy primitive checks passed. An expanded normal-to-subnormal check failed CPU equivalence identically for both existing GPU arithmetic and the candidate, with zero GPU baseline/candidate bit differences. That failed receipt remains a failed expanded CPU-equivalence test. After observing it, a separately frozen preservation plan required the original eight CPU checks, all twelve exact GPU-baseline checks including a fresh underflow perturbation, unchanged full-model CPU quality gates, and the full speed gate. GPU preservation passed; universal CPU/IEEE equivalence was never established. The abandoned first screen's JSON serialization failure and the failed underflow confirmation remain in the evidence directory with executed source snapshots.

These experiments narrow the result: the native MLX runtime supplies the measured acceleration, while larger projection tiles and rotary fusion provide no accepted further improvement under the quality and consistency constraints. Base projections remain the largest measured target. This is not evidence that the implementation is globally Pareto-optimal or that more aggressive kernels cannot help.

## Final expanded confirmation

Before generating new candidate outputs, `fresh-frozen.json` retained the original 38 cases and added 16 previously unused DEV records covering all nine family/task strata. Selection excluded the original record IDs and used a deterministic median/shortest/longest rule; no candidate output selected those records. The frozen plan SHA-256 is `bb9d056265e17af63af354d54866b1e782cce08145e2c67ba49dd6cb042a2653`. This is a fresh implementation-parity challenge within DEV, not held-out model-quality validation.

`fresh-cpu4.json`, `fresh-ordinary.json` and `fresh-metal.json` each complete **54 cases, 70 question rows and 2,338 candidate logits**, plus 18 timing observations. Both MLX comparison receipts pass every unchanged gate: exact encoding/order, selected Choice and complete rankings, logit/probability/typed-value tolerances, all five negative checks, and public predict/readout consistency. Maximum full-case absolute errors are 8.44e-5 in logits, 6.85e-6 in probabilities and 1.13e-5 in typed values for both MLX arms. No failures were dropped.

| Path | CPU, four threads (ms) | Ordinary MLX (ms) | Optional Metal MLX (ms) |
| --- | ---: | ---: | ---: |
| 77-option singleton, encoded | 1,780.06 | 800.18 | 775.68 |
| Eight short rows, encoded | 2,619.57 | 1,265.61 | 1,275.91 |
| Eight mixed rows, encoded | 4,124.94 | 1,822.07 | 1,618.25 |
| 77-option singleton, public | 1,849.78 | 821.12 | 793.53 |
| Eight short rows, public | 2,787.44 | 1,275.58 | 1,186.27 |
| Eight mixed rows, public | 3,442.15 | 1,776.96 | 1,624.08 |

These are medians of three observations per path from sequential, low-priority runs. Ordinary MLX public-call speedups are 2.25x / 2.19x / 1.94x against the restrained CPU baseline. System load and memory pressure varied, so these are observations on this Mac, not a stable throughput guarantee or evidence to promote optional Metal. Peak active MLX allocation was 3.293 GB ordinary and 2.412 GB Metal. The cache reached about 312 MB despite its 256 MiB guideline; neither setting is a hard cap.

All full-model jobs ran one at a time with 250 ms pauses outside forward timers; final regression checks used tiny fixtures after those jobs exited. The production default retains same-scorer serialization and cooperative four-layer/1 ms pacing. `resource-observation-final.json` records a late system-wide memory snapshot, including nonzero swap counters. Interactive UI latency was not measured, and other processes or scorer instances can still cause contention. The measured outcome is faster native inference with bounded benchmark concurrency and unchanged quality gates; hard resource isolation remains outside the implementation.
