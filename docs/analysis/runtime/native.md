# Native CPU and MLX inference

On the current Apple Silicon host, the existing full-stock MLX backend is a
supported option for this local FP32 Kai artifact. Select it explicitly; default
loading remains Torch on CPU. No production code changed for this comparison.

## Use the measured configuration

Install the existing MLX/server extras, then keep `MLX_ENABLE_TF32=0` in the
process environment before importing MLX. The 512 MiB setting controls inactive
Metal allocator buffers, not total model memory or a hard RAM bound.

```sh
uv sync --extra mlx --extra server
MLX_ENABLE_TF32=0 uv run wyat serve /path/to/Decision-2.0-Kai-0.6B \
  --backend mlx --device gpu --max-length 512 --token-cache-size 0 \
  --mlx-cache-limit-mib 512 --host 127.0.0.1 --port 8000
```

For library callers, launch Python with `MLX_ENABLE_TF32=0`, configure the
process-owned allocator explicitly, and load once for repeated requests:

```python
import mlx.core as mx
import wyat

mx.set_cache_limit(512 * (1 << 20))
model = wyat.load(
    "/path/to/Decision-2.0-Kai-0.6B",
    backend="mlx",
    device="gpu",
    max_length=512,
    token_cache_size=0,
)
# model.decide(request) or model.decide_many(requests)
```

Both tested runtime backbones and heads execute FP32. The released backbone
contains mixed BF16/F32 source tensors, expanded to FP32 at load. These settings
do not establish another artifact's numerical quality. Keep
compilation disabled for this configuration. The ordinary path performs full
prefill; the experimental selected-last-block path is not promoted.

## Reproduce the comparison

The comparison also needs the existing inference extra for its Torch baseline.
Use the canonical local 77-case JSONL file and retain stdout plus diagnostics:

```sh
uv sync --extra inference --extra mlx --extra server
MLX_ENABLE_TF32=0 uv run wyat benchmark .models/Decision-2.0-Kai-0.6B \
  --backend mlx --device gpu --baseline-backend torch --baseline-device cpu \
  --batch-size 1 --baseline-batch-size 1 --max-length 512 \
  --token-cache-size 0 --torch-threads 8 --repeats 5 --warmup 1 \
  --mlx-cache-limit-mib 512 \
  < benchmarks/real/banking77-train-dev-menu6.jsonl \
  > /tmp/wyat-native-comparison.json 2> /tmp/wyat-native-comparison.log
```

The CLI exit status reports paired quality, not performance acceptance. An
external assay must also require paired median at least **1.02x**, sample p05
strictly above **1x**, aggregate cohort speedup above **1x**, and candidate cohort
p95 no worse than baseline. These four conditions are unchanged. For bounded
runs, use an owned process-group supervisor as retained in the executed source
archive; this shell example itself does not impose a wall cap.

## Observed current result

The independently audited run retained **924 outcomes**, including both warmup
cohorts and five paired measured cohorts. Every cohort has the same 77 ordered
cases and **67 correct**; no outcome failed or disappeared. All gold-reference
and cross-backend full ranks, choices, logit absolute/relative `1e-3` and
probability absolute `1e-4` gates pass. The measured paired maximum drift was
`1.54972e-5` logits and `3.04443e-6` probabilities.

| Original performance condition | Observed | Verdict |
| --- | ---: | --- |
| Paired median ≥ 1.02x | 2.83972x | Pass |
| Paired sample p05 > 1x | 2.79313x | Pass |
| Aggregate cohort speedup > 1x | 2.82992x | Pass |
| Candidate/baseline cohort p95 ≤ 1 | 0.349892 | Pass |

All five paired ratios remain visible: `2.843463`, `2.839722`, `2.878087`,
`2.795916`, `2.792436`. Median complete-cohort latency was **10,009.36 ms** for
Torch CPU and **3,527.95 ms** for MLX GPU. The native inference child exited zero,
was reaped after **87.682 s** within its 180-second cap, and passed source/asset
integrity checks. The independent standard-library audit exited zero and was
reaped after **0.609 s** within its separate 60-second cap.

Observed context: macOS 27.2 arm64, Python 3.12.11, Torch 2.14.1,
Transformers 5.18.0, MLX 0.32.3 and MLX-LM 0.31.3. The executed archive contains
28 source texts, including all 22 unchanged production Python files and the
actual supervisor; eleven model/input assets were pinned before and after.

Read the [raw report](../../../benchmarks/real/runtime-cpu-mlx77-current.json),
[independent audit](../../../benchmarks/real/runtime-cpu-mlx77-current-review.json)
and [audit execution receipt](../../../benchmarks/real/runtime-cpu-mlx77-current-review.execution.json).
The raw report SHA is
`829621d533b6a7998b3c584c7533193e0e7ec09c49a4751b7fc10381c0839d65`;
the executed source archive SHA is
`1cffd68e316c10ed7e87aedb106bbef0c4c23d9eef9573d2b11d2402d92f93aa`.

The result supports this backend choice on this host, artifact and development
cohort. The exact Apple chip model was not captured. Torch uses each scalar's
logical token width; MLX rounds physical width to eight tokens, so logical
requests match while execution shapes differ. Timed dispatch was unprofiled;
`MLX_ENABLE_TF32=0` is an observed environment setting, not proof about every
kernel's internal arithmetic. Token/reference identities do not establish
independent retokenization or native binary identity. Loading, HTTP, admission
queues and JSON serialization are excluded; cumulative RSS spans both engines.
Five pairs and their interpolated p05 are not confidence bounds. These training-
source development cases do not establish held-out generalization, another
model's quality, server throughput, or a universal backend advantage.
