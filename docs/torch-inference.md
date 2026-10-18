# PyTorch inference (CPU and CUDA)

The upcoming v0.1.1 targets
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m). It uses the
`wyat-option-fork-v1` format with a Qwen3.5 hybrid backbone. The named download
is pending; a prepared local artifact can be loaded now:

```python
import wyat

model = wyat.load("/path/to/wyat-1-800m/model", device="cpu")
# For a compatible CUDA device: device="cuda".
```

Install the inference extra in your application environment; see
[installation](install.md). Dependencies and weights are separate; follow
[model acquisition](model-acquisition.md) for the required layout and current
availability. 800M does not support MLX.

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
CPU inference uses FP32. CUDA runs under BF16 autocast, stores frozen backbone
linear weights in BF16, and keeps adapter calculations and the decision head in
FP32. Use a BF16-capable CUDA device and check numerical and decision parity
for the artifact and workload you deploy.

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
