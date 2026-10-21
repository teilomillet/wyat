# MLX inference (Apple Silicon)

## 800M option-fork inference

The default model, [wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m),
runs on MLX with the `mlx` extra:

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m", backend="mlx")  # Apple GPU
# MLX CPU: device="cpu"
```

wyat runs the Qwen3.5 hybrid backbone with
[MLX-LM's Qwen3.5 text model](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_5.py).
Artifact validation, prompt encoding, option branches and readout are shared
with the [PyTorch runtime](torch-inference.md). Backbone weights are expanded
from BF16 storage to FP32; the unmerged adapters and scalar head stay FP32. One
deliberate difference from MLX-LM: gated-delta-net layers normalize queries and
keys with the Transformers epsilon placement used in training. MLX-LM's own
placement moved the released model's logits by up to 2e-2.

On an Apple M4 Pro (2026-10-18), MLX GPU and MLX CPU matched PyTorch CPU logits
on the released 800M within 1.8e-5, with no change of selected option, on six
synthetic questions up to a 5,871-token prefix. The GPU scored the set in about
4 s, against 17 s for one PyTorch CPU thread and 12 s for PyTorch MPS. MLX CPU
took 117 s, because MLX-LM's CPU gated-delta path steps one token at a time; use
it for parity checks, not speed. Measure your own requests and device.

The prefix is reused within a question. Option-fork artifacts also support an
opt-in cross-call cache of the rendered record-state prefix: consecutive
questions that share the same record state (for example a scenario with several
sibling questions, or retries) continue from the retained attention and
gated-delta-net state instead of re-prefilling the state block. Set its entry
count with `load(..., prefix_cache_size=N)` or the CLI's `--prefix-cache-size`;
the option-fork default is `0` (disabled) because each entry retains the hybrid
state for a full state block. `token_cache_size` does not apply, and
`compile=True` is rejected. GPU sharing
pauses work as described under [native artifacts](#native-artifacts);
`responsive=False` removes the pauses. The Metal kernel for gated-delta-net
layers needs a linear key head width divisible by 32 (128 in 800M); other
widths are refused on the GPU.

## Compatible legacy artifacts

The rest of this guide applies to compatible native full-task and single-choice
candidate-head artifacts retained for compatibility. For those artifacts, wyat
uses [MLX-LM's Qwen3 backbone](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3.py)
for Apple Silicon. The runtime reads a local wyat checkpoint directly; there is
no model conversion, downloaded Python execution, vocabulary projection or
generation loop.

```python
from wyat import Candidate, Request, load

model = load("/path/to/native-model", backend="mlx", device="gpu")
request = Request(
    query="Where should this request go?",
    context="I need to reset my password.",
    candidates=(Candidate("account", "Account access"), Candidate("sales", "Sales")),
)
decision = model.decide(request)
```

`device="gpu"` is the default; `device="cpu"` is available for runtime parity
checks. On macOS, even importing the Metal-enabled MLX wheel requires access to
the Metal device; a sandbox that hides it cannot establish a passing inference
check.

The loader shares artifact validation, file hashes and segmented tokenization
with the Torch backend. It rejects missing, unexpected, malformed and nonfinite
weights. Backbone tensors are expanded losslessly from BF16 storage to FP32,
except for the qualified candidate-head projection path described below. The
decision head stays FP32, including its exact GELU and LayerNorm with epsilon
`1e-5`. No quantization or reduced-precision approximation is enabled. On Apple
GPUs where MLX can use TF32 for FP32 matrix products, set `MLX_ENABLE_TF32=0`
before starting Python to keep FP32 products; see [installation](install.md).

Rows are right-padded and the causal mask excludes all future padding from each
selected endpoint. Candidate endpoints and the final query endpoint are gathered
before head scoring. Output rows retain the original request and candidate
order. Scoring evaluates the MLX graph and materializes host floats before
returning, so wall-clock timings include device execution.

## Native artifacts

The legacy `teilomillet/wyat-1-600m` release remains a compatible native
(`wyat-native-full-v1`) artifact. Its FP32 LoRA adapters cover every linear projection in each block
(`q`, `k`, `v`, `o`, `gate`, `up`, `down`) and load unmerged from
`native-adapters.safetensors`. Historical q/v-only artifacts also remain
supported by the MLX and Torch loaders.

- **Prefix cache.** A small least-recently-used cache keeps the K/V state of
  the token prefix before the candidates; later requests that begin with the
  same tokens reuse the longest cached match instead of recomputing it. Set its size with `load(..., prefix_cache_size=N)` (default
  16, `0` disables it) or the CLI's `--prefix-cache-size`.
- **Batches.** A batch keeps its original width and order. A padded batch above
  the artifact's token budget is rejected with `ModelInputError`
  (`reason="input-length"`) rather than split or truncated.
- **GPU sharing.** On the GPU, inference pauses for 1 ms after every fourth
  backbone layer so other applications can submit GPU work. Include this in
  latency measurements. `load(..., responsive=False)` or the CLI's
  `--no-responsive` removes the pauses; the unpaced latency gain and its
  effect on other applications are unmeasured.
- **Compilation.** Native artifacts reject `compile=True`.

## Candidate-head artifacts

Single-choice candidate-head artifacts use a different scorer with these
additional runtime paths.

With MLX 0.32.3 and the `applegpu_g16s` architecture, eager GPU inference uses
Metal TensorOps for the supported wyat projection shapes. Eligible bias-free
layers use exact BF16 checkpoint values in independent inference storage; an
FP32 checkpoint weight is compressed only when every bit survives the
conversion. Activations, accumulation and outputs remain FP32. Split reductions
preserve MLX's partition boundaries and addition order. The capability check
runs while loading; missing optional compiler support keeps ordinary MLX
projections, while allocation failures and numerical failures remain errors.
Other MLX versions, architectures, shapes, biased layers, CPU and compiled
resources keep ordinary projections. No global precision or runtime setting is
changed. Measure preparation and first-call costs separately from repeated
inference.

On the GPU, batches with widely different encoded lengths are split into
groups only when this removes at least half the rounded padded token work.
Otherwise, the original batch is used. Single requests and CPU inference keep
their original path. Each group runs a complete causal prefill with the same
FP32 arithmetic. Measure workload-specific gains, unchanged controls and output
parity on your own requests.

`compile=True` opts into [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
Weights are loaded once and captured as immutable inference parameters. Measure
cold compilation separately from warm throughput: compiled graphs specialize
to input shapes and a new batch shape can incur compilation.

Ordinary `load(..., backend="mlx")` of a candidate-head artifact uses full
prefill. For repeated requests with an exact shared token prefix, `PrefixScorer`
reuses the prefix's K/V state and computes the final block only at decision
readouts:

```python
from wyat import Model
from wyat.backends.wyat.mlx import MLXWyatScorer
from wyat.backends.wyat.prefix import PrefixScorer

base = MLXWyatScorer.from_directory(
    "/path/to/candidate-head-model", device="gpu", compile=False
)
# Choose a length shared by your encoded requests, before all candidate readouts.
# request is the representative Request above; 3 is an example, not a speed policy.
prefix = base.encode_request(request).ids[:3]
cached = PrefixScorer(base, prefix)
model = Model(cached, cached.model_id)
decision = model.decide(request)
```

Prepare the resource once and keep its weights, encoder and device fixed. Longer
shared prefixes can avoid more work; measure your actual requests including the
one-time preparation cost. Each call gets fresh concatenating caches sized to
its suffix, with no reserved decoding capacity. Prefix mismatches or a readout
inside the prefix fall back to the ordinary scorer for the complete batch.
Scaled RoPE and compiled bases are rejected. Candidate order, FP32 arithmetic,
input limits and synchronous output are preserved.

## Installation and dependencies

Install the MLX extra in the same application environment; see
[installation](install.md). MLX requires Apple Silicon macOS. Compatible native
artifacts use the same files with PyTorch and MLX; follow
[model acquisition](model-acquisition.md). The 800M artifact needs no
conversion either.

MLX-LM depends on Transformers and may initialize its optional Torch integration
when Torch happens to be installed. wyat's MLX inference uses MLX tensors and
needs no Torch installation. In a mixed environment, upstream initialization
may still import Torch.

Runtime mechanics and file integrity do not establish model quality or portable
performance. Evaluate your application's requests and device before choosing
runtime settings.
