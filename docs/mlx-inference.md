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
weights. Backbone tensors are expanded losslessly from BF16 storage to FP32;
the decision head stays FP32, including its exact GELU and LayerNorm with
epsilon `1e-5`. No quantization or reduced-precision approximation is enabled.

For each batch, the backbone runs one causal prefill. Candidate endpoints and
the final query endpoint are gathered before head scoring. Rows are right-padded;
the causal mask excludes all future padding from each selected endpoint. Output
rows retain the original request and candidate order. `score_many` evaluates the
MLX graph and materializes host floats before returning, so wall-clock timings
include device execution.

`compile=True` opts into [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).
Weights are loaded once and captured as immutable inference parameters. Measure
cold compilation separately from warm throughput: compiled graphs specialize
to input shapes and a new batch shape can incur compilation. It remains opt-in
until the selected real workload demonstrates a benefit within its parity gate.

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
