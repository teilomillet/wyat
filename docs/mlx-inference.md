# Native MLX inference

Wyat uses [MLX-LM's Qwen3 backbone](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3.py)
for Apple Silicon. The runtime reads a local full Kai choice checkpoint directly;
there is no model conversion, downloaded Python execution, vocabulary projection,
generation loop, or persistent KV cache in the default scorer. This backend
implements the supported Kai Choice contract.

```python
from wyat import Candidate, Request, load

model = load("/path/to/Decision-2.0-Kai-0.6B", backend="mlx", device="gpu")
request = Request(
    query="Where should this request go?",
    context="I need to reset my password.",
    candidates=(Candidate("account", "Account access"), Candidate("sales", "Sales")),
)
decision = model.decide(request)
```

`device="cpu"` is available for runtime parity checks. On macOS, even importing
the Metal-enabled MLX wheel requires access to the Metal device; a sandbox that
hides it cannot establish a passing inference check.

The loader shares artifact validation, file hashes and segmented tokenization
with the Torch backend. It rejects missing, unexpected, malformed and nonfinite
weights. Backbone tensors are expanded losslessly from BF16 storage to FP32,
except for the qualified projection path described below. The decision head
stays FP32, including its exact GELU and LayerNorm with
epsilon `1e-5`. No quantization or reduced-precision approximation is enabled.

With MLX 0.32.3 and the `applegpu_g16s` architecture, eager GPU inference uses
Metal TensorOps for the supported Kai projection shapes. Eligible bias-free
layers use exact BF16 checkpoint values in independent inference storage; an
FP32 checkpoint weight is compressed only when every bit survives the conversion.
Activations, accumulation and outputs remain FP32. Split reductions preserve MLX's partition
boundaries and addition order. The capability check runs while loading; missing
optional compiler support keeps native projections, while allocation failures
and numerical failures remain errors. Other MLX versions, architectures, shapes,
biased layers, CPU and compiled resources keep native inference. No global
precision or runtime setting is changed. Preparation and first-call costs should
be measured separately from repeated inference.

On the GPU, batches with widely different encoded lengths are split into
groups only when this removes at least half the rounded padded token work.
Otherwise, the original batch is used. Single requests and CPU inference keep
their original path. Each group runs a complete causal prefill with the same
FP32 arithmetic; no prefix reuse or additional cache is required. This can
substantially accelerate mixed-length batches, rather than every workload.
Measure workload-specific gains, unchanged controls and output parity on your
own requests; detailed development evidence is retained privately.
Candidate endpoints and the final query endpoint are gathered before head
scoring. Rows are right-padded;
the causal mask excludes all future padding from each selected endpoint. Output
rows retain the original request and candidate order. `score_many` evaluates the
MLX graph and materializes host floats before returning, so wall-clock timings
include device execution.

`compile=True` opts into [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
Weights are loaded once and captured as immutable inference parameters. Measure
cold compilation separately from warm throughput: compiled graphs specialize
to input shapes and a new batch shape can incur compilation. It remains opt-in
until the selected real workload demonstrates a benefit within its parity gate.

For repeated requests with an exact shared token prefix, `PrefixScorer` reuses
the prefix's K/V state and computes the final block only at decision readouts:

```python
from wyat import Model
from wyat.backends.kai.mlx import MLXKaiScorer
from wyat.backends.kai.prefix import PrefixScorer

base = MLXKaiScorer.from_directory("/path/to/kai", device="gpu", compile=False)
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
input limits and synchronous output are preserved. This remains explicit;
ordinary `load(..., backend="mlx")` still uses full prefill.

Install the MLX extra in the same application environment; see [installation](install.md).
MLX requires Apple Silicon macOS. The artifact is the same full local download
used by PyTorch; follow [model acquisition](model-acquisition.md).

MLX-LM depends on Transformers and may initialize its optional Torch integration
when Torch happens to be installed. Wyat's MLX inference uses MLX tensors and
needs no Torch installation. In a mixed environment, upstream initialization
may still import Torch.

Runtime mechanics and file integrity do not establish model quality or portable
performance. Evaluate your application's requests and device before choosing
runtime settings.
