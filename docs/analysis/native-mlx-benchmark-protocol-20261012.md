# Native Wyat MLX parity and timing protocol — 2026-10-12

The experiment asks whether the native Wyat model can run faster on this Mac while preserving its trained tensors, input recipe, independent question rows, ordered candidates, and Choice/Score/Noul readouts. CPU equivalence on reused DEV inputs supports implementation parity; it does not establish held-out accuracy or CUDA arithmetic equivalence. PyTorch/CUDA production execution remains the preserved reference path.

The research runner is `benchmarks/analysis/native_mlx_benchmark_20261012.py`. Its resource-paced frozen SHA-256 is `00c73e707bdb579b074a55b728725cc8ee1b97842d4b33b35f64aa7cf758246a`. Each final JSON output is created exclusively; repeated experiments must use fresh names. Existing final or partial output paths are rejected before loading a model. Each completed inference case, timing observation, and alternating pair is atomically checkpointed to `OUTPUT.partial`; finalization creates `OUTPUT` exclusively and removes its own partial file. Every observation, including exceptions, remains in the denominator. A missing runtime/compiler/device is unknown rather than a pass.

## Frozen inputs and source boundary

The current input plan is `benchmarks/analysis/results/native-mlx-20261012/frozen.json`; its SHA-256 is `6dc92635e49950369b1594d16274079790ff1aafa0dfc724daa5ad9c8c986202`. The runner refuses to measure after its own source differs from the frozen hash. Per-worker receipts record imported source paths/hashes, dependency versions, precision environment, hardware/platform, artifact manifest identity, actual Metal admission, resource policy, and raw outputs.

The interrupted earlier experiment used temporary `/private/tmp` plans and receipts, which are no longer present after the interruption. Earlier summaries reported two initial zero-case memory-observation failures caused by unavailable optional `psutil`, and an eight-thread CPU run; those raw receipts cannot currently be verified and do not qualify this new experiment. The current harness uses stdlib Mach `task_info` on macOS (Linux uses `/proc/self/statm`); its memory-only sandbox preflight passed without a model forward. Inputs, selection rules, and quality/performance gates remain fixed; the new freeze records the resource-paced harness and persistent storage location. This recovery preserves the earlier uncertainty instead of labeling missing observations a pass.

The source cohort is `artifacts/french-combined-best-checkpoint-20261009/SCREEN.jsonl`, SHA-256 `a01f65e8c46f9c2bc703b2ead322946c991e669bb93f7cf929a5ffbb35720c7a`. Selection uses only `public_record` and source family/ID; labels and supervision never enter inference. Within each family/task stratum, sort by encoded length and ID, then round-robin the 0, 1/3, 2/3, 1 length quantiles until 32 distinct DEV records are selected. This choice was frozen before model outputs were observed. All 498 source records encoded successfully; no preselection exclusions were observed.

The frozen 36 records comprise 32 DEV records, synthetic 77/151/255-option menus, and one three-question fixture spanning Choice, Score and reversed-menu Noul. The 38 execution cases comprise 35 single-row cases, one three-independent-question batch, and two eight-row batches. Encoded lengths span 89–5,436 tokens and candidate counts span 2–255. Both B8 batches obey the 8,192 padded-token budget. Right padding, full padded width, native question IDs, and complete option endpoints remain identical across backends. No length grouping, prefix cache, truncation, adapter merge, reduced precision, or alternative encoder is introduced by this experiment.

The CPU oracle uses the original complete release `artifacts/huggingface-releases/wyat-v0.2a/model` and its signed bundled runtime. The harness verifies every bundled source hash against `runtime/SOURCE.json` and checks that the imported native loader is inside that runtime. Run it from `/private/tmp` with the pinned runtime as `PYTHONPATH`, preventing current-checkout imports from shadowing the oracle. Original release files remain intact.

The candidate uses the separately derived artifact `artifacts/native-mlx-runtime-admission-20261012/model`, whose current shared validator accepts its explicit source-contract identity. The original six model/tokenizer files are hash-equal; the derived manifest is `effbd0fc5fb18d312e20bf617593d9048d545d8b015bedf59df9a692b3ce7104`. Its new provenance records the original manifest and comment-only source AST equivalence. This is an explicit new local export, not a validation exception. Both artifacts retain model ID `native-68cdf47a3012971214cdf062fb3f1e023769e997bbce5761d016fa9c0e4433b2`.

## Fixed acceptance conditions

Every requested case must produce all ordered finite logits, every candidate probability, and its typed readout. Exact encoded token IDs, candidate/query endpoints and question metadata must match the frozen plan. Selected Choice IDs and complete stable rankings must match exactly. Numeric gates are:

| Quantity | Gate |
| --- | --- |
| Every logit | absolute error ≤ `1e-3 + 1e-3 * abs(reference)` |
| Every candidate probability | absolute error ≤ `1e-4` |
| Noul probability of the actual `true` option | absolute error ≤ `1e-4` |
| Score expected ordinal value | absolute error ≤ `1e-4 * max(1, option_count - 1)` |

The Score tolerance scales with its interval count because its output is an expected rubric index. These are backend parity tolerances and do not relax the stdlib readout's independent stable-softmax contract.

All repeated timing outputs are also checked against CPU. Every alternating pair is checked three ways: CPU versus ordinary MLX, CPU versus Metal, and ordinary MLX versus Metal. No performance result can compensate for a failed quality gate. Public `predict` must match the independently recorded ordered-logit readout on the three-question fixture.

The five negative cases are private metadata, unknown task type, a one-option menu, overlength input, and an explicit padded-question batch exceeding its token budget. Each must raise `ValueError`; no negative case runs an oversized model forward. A missing case, failed load, exception, absent repeat, or warmup error prevents qualification.

For custom Metal promotion, each of the three timing shapes needs at least 15 complete alternating pairs, a median ordinary/Metal speedup of at least 1.05, and the interpolated fifth-percentile speedup of at least 1.0. All requested pairs and failures remain recorded. Ordinary MLX must actually report Metal disabled and the optimized arm must report Metal enabled. A fallback cannot be labeled an optimized-arm speedup. Failed consistency keeps Metal optional even if its median improves.

## Measurements and interpretation

The confirmatory CPU worker explicitly uses four intra-op threads and one inter-op thread (`--threads 4`). This lower CPU allocation follows the user's request to keep the Mac responsive; it differs from the interrupted earlier eight-thread run. Thread settings are isolated to that subprocess and are not claimed to be the fastest possible CPU baseline. The original release verifier used one thread; a one-thread oracle remains an optional independent follow-up if CPU scheduling changes parity.

Run only one benchmark process at a time with `nice -n 10`, leaving other CPU cores available. The harness defaults to a 250 ms pause after each inference attempt, including warmups, public `predict`, and alternating arms. These pauses and checkpoint writes are outside measured inference intervals; they reduce sustained device occupancy, not individual inference latency. MLX benchmark processes set their own allocator memory guideline to 6,144 MiB and reusable cache limit to 256 MiB before loading. These settings do not alter production library globals or PyTorch settings. The MLX allocation setting is a guideline rather than an operating-system hard cap; watch actual memory pressure, and stop or lower work if foreground responsiveness deteriorates. Paired mode keeps two models resident (approximately 4 GB before working memory); its resource receipt makes that context explicit.

Each single-arm worker records load time, live/peak process RSS and (MLX only) active/peak/cache allocation after load and after inference. `ru_maxrss` is converted to bytes according to the platform. OS file caches are uncontrolled, so these are fresh-process starts with potentially warm files, not guaranteed cold-storage starts. Repeated fresh processes are needed before drawing robust startup/memory conclusions.

Warm timing uses one warmup and three retained repeats for (a) preencoded rows and (b) native public-record encoding through logits. The singleton shape is a real 77-option DEV menu (`single-00`, 2,518 tokens). B8 short and B8 mixed use the frozen eight independent rows and their full reference width. Singleton public timing calls `model.logits`; B8 public timing composes the same encoder for each independent record then executes their fixed shared batch. Tokenization and encoded inference are separate measurements; their difference is not a causal phase attribution.

Custom-Metal timing loads both ordinary and optimized models in one process, warms each three times per shape, then alternates ordinary→Metal / Metal→ordinary for 15 pairs. The two resident models prevent reloading from dominating each pair, but also change the memory context. Use separate single-arm worker receipts for memory comparisons. GPU results synchronize before and after each timed inference. Avoid concurrent CPU/GPU benchmark work, other local model jobs, or changes to precision settings between arms.

## Reproduction

Use absolute paths. The following are single commands; choose fresh output names for additional runs.

CPU oracle, working directory `/private/tmp`:

```sh
PYTHONPATH=/Users/teilomillet/Code/wyat/artifacts/huggingface-releases/wyat-v0.2a/runtime \
nice -n 10 /Users/teilomillet/Code/wyat/.venv/bin/python \
/Users/teilomillet/Code/wyat/benchmarks/analysis/native_mlx_benchmark_20261012.py measure \
--arm cpu --threads 4 \
--artifact /Users/teilomillet/Code/wyat/artifacts/huggingface-releases/wyat-v0.2a/model \
--runtime /Users/teilomillet/Code/wyat/artifacts/huggingface-releases/wyat-v0.2a/runtime \
--frozen /Users/teilomillet/Code/wyat/benchmarks/analysis/results/native-mlx-20261012/frozen.json \
--output /Users/teilomillet/Code/wyat/benchmarks/analysis/results/native-mlx-20261012/cpu.json
```

Ordinary MLX, working directory `/Users/teilomillet/Code/wyat`:

```sh
nice -n 10 env -u MLX_ENABLE_TF32 PYTHONPATH=/Users/teilomillet/Code/wyat \
/Users/teilomillet/Code/wyat/.venv/bin/python \
benchmarks/analysis/native_mlx_benchmark_20261012.py measure \
--arm ordinary \
--artifact /Users/teilomillet/Code/wyat/artifacts/native-mlx-runtime-admission-20261012/model \
--frozen /Users/teilomillet/Code/wyat/benchmarks/analysis/results/native-mlx-20261012/frozen.json \
--output /Users/teilomillet/Code/wyat/benchmarks/analysis/results/native-mlx-20261012/ordinary.json
```

Repeat that command with `--arm metal` and a new output name for the separate optimized-arm receipt. Compare each against the CPU receipt using `compare --frozen PLAN --reference CPU --candidate ARM --output NEW_QUALITY_RECEIPT`.

The completed `ordinary-default.json` and `metal-default.json` measurements use `MLX_ENABLE_TF32` unset, matching the production environment. A separately requested strict-precision diagnostic must use a new output name and retain its environment as a distinct experiment.

Run alternating pairs with `paired --arm metal --artifact DERIVED_MODEL --frozen PLAN --pairs 15 --output NEW_PAIRS_RECEIPT`. Audit every pair with `summarize-pairs --frozen PLAN --reference CPU --candidate PAIRS --output NEW_SUMMARY_RECEIPT`.

Implementation parity on this frozen corpus, macOS speed on these shapes, native release compatibility, and preservation of the existing Torch/CUDA path are separate claims. A parity or timing result does not establish native-model held-out quality, universal numerical equivalence, training validity, or CUDA throughput. Any additional cohort, production change, or default precision setting requires fresh evidence under its own declared conditions.
