# PyTorch inference (CPU, CUDA and Apple MPS)

The default model is
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m). It uses the
`wyat-option-fork-v1` format with a Qwen3.5 hybrid backbone. Load its pinned
release by name; the first call downloads and verifies the files:

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m", device="cpu")
# For a compatible CUDA device: device="cuda". For the Apple GPU: device="mps".
```

Install the inference extra in your application environment; see
[installation](install.md). Dependencies and weights are separate; follow
[model acquisition](model-acquisition.md) for the cache and local artifact
layout. On Apple Silicon, the [MLX runtime](mlx-inference.md) also runs 800M.

## 800M option-fork inference

For each question, wyat encodes one canonical prefix containing the state,
instructions and all options. It computes that prefix once, then evaluates an
isolated continuation for each option from a copy of the prefix's attention
and recurrent state. A shared FP32 scalar head scores the last non-padding
token of each branch. Returned scores follow the caller's option order.
There is no autoregressive generation loop.

Branches are evaluated in groups within the artifact's token budget. An input
that exceeds the configured length is refused rather than truncated. The
prefix is reused within a question; `token_cache_size` and `prefix_cache_size`
do not create cross-call caches for this format.

800M uses PyTorch's
[scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
and uncompiled execution. `attention="eager"` and `compile=True` are rejected.
CPU and MPS inference use FP32. CUDA runs under BF16 autocast, stores frozen backbone
linear weights in BF16, and keeps adapter calculations and the decision head in
FP32. Use a BF16-capable CUDA device and check numerical and decision parity
for the artifact and workload you deploy.

On an Apple M4 Pro (2026-10-18), MPS matched CPU logits on the released 800M
within 1.2e-5, with no change of selected option, on six synthetic questions
up to a 5,871-token prefix. It loaded in about 1 s and scored the set in 12 s,
against 17 s for one CPU thread; MLX on the GPU scored the same set in about
4 s. Measure your own requests and device.

## Runtime settings and legacy compatibility

Thread counts belong to the process running the model: changing PyTorch threads
affects other models in the same process. Choose a count for the actual machine
and workload, following
[PyTorch's threading guidance](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).
The CLI's `--torch-threads` option sets it at the process boundary; omitting the
option preserves PyTorch's default. Measure loading, first inference and
repeated inference separately.

The existing artifact families retain their distinct inference contracts:

| Artifact | Readout and cache | Attention |
| --- | --- | --- |
| `wyat-option-fork-v1` (800M) | Last token of each isolated option branch; question-local prefix reuse | SDPA, uncompiled |
| `wyat-native-full-v1` (legacy compatibility) | Candidate and final decision endpoints; cross-call prefix KV cache | SDPA, uncompiled |
| Single-choice candidate-head | Candidate and final decision endpoints; full prefill | SDPA or reference eager, uncompiled |

Native artifacts keep up to `prefix_cache_size` shared token prefixes (default
16); `prefix_cache_size=0`, or the CLI's `--prefix-cache-size 0`, restores full
prefill. They retain backbone weights in their stored FP32 or BF16 dtype on
CUDA, with FP32 adapters and decision head. Candidate-head artifacts restore
exactly representable BF16 linear weights to BF16 on CUDA; embeddings, norms,
residuals and decision head retain FP32. Both use BF16 autocast on CUDA and
FP32 on CPU. These storage rules are distinct from the 800M runtime.

Inference uses `inference_mode` and returns materialized CPU scores after device
execution. Different kernels and batch shapes may change floating-point
rounding; compare raw logits and selected IDs when changing settings. Missing
runtime dependencies require installation, and incomplete or malformed
artifacts require repairing the artifact. The loader does not substitute a
model or retry inference automatically.
