# Native MLX inference

Wyat uses [MLX-LM's Qwen3 backbone](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3.py)
for Apple Silicon. The runtime reads a local full Kai choice checkpoint directly;
there is no model conversion, downloaded Python execution, vocabulary projection,
generation loop, or persistent KV cache in the default scorer. The eventual Wyat architecture remains
open; this backend implements the currently released Kai contract.

```python
from wyat import Candidate, Model, Request
from wyat.backends.kai.mlx import MLXKaiScorer

scorer = MLXKaiScorer.from_directory("/models/kai", device="gpu")
model = Model(scorer, scorer.model_id)
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

Applications sharing an exact fixed token prefix can explicitly construct an
eager `PrefixScorer` around an uncompiled native base. It retains immutable
prefix K/V state, uses fresh suffix cache owners and falls back whole batches on
mismatches. [Prefix usage and its measured limits](analysis/prefix.md) describe
the resource contract. Existing loading and CLI defaults retain full prefill.

MLX-LM depends on Transformers and can initialize its optional Torch integration
when Torch happens to be installed. Wyat uses MLX tensors and the maintained Transformers tokenizer policy.
An MLX-only environment needs no Torch installation. A
mixed environment may still import Torch through upstream package initialization;
this is an upstream optional import rather than an inference requirement.
The clean installation `uv pip install -e '.[mlx]'` was checked in a separate
environment on 2026-10-05: a tiny local checkpoint completed GPU scoring with
both `importlib.util.find_spec("torch") is None` and `"torch" not in sys.modules`.
That environment used MLX 0.32.3, MLX-LM 0.31.3 and Transformers 5.17.0.

The checks in `tests/backends/kai/test_mlx.py` compare native head arithmetic and complete
tiny checkpoint inference against Torch, exercise Unicode serialization,
variable-length batches, compilation and missing weights, and load a checkpoint
in a fresh process. Tiny random weights establish
mechanics only. Real-data accuracy, numerical drift and hardware performance must
be established through the recorded full-checkpoint benchmark.

The real FP32 Metal run on 2026-10-05 used 77 frozen BANKING77 development
requests with six supplied candidates, retaining all 77 Torch reference rankings
and decisions. The largest logit difference was `1.29e-5`; the largest probability
difference was `2.46e-6`. Both runtimes selected the expected label in 67 of 77
cases. This measures inference preservation on that cohort; the checkpoint's
exposure to BANKING77 is unknown.

Five paired full-cohort runs measured a median compiled/eager speedup of `0.990x`
(5th–95th paired percentiles: `0.950x`–`1.080x`). Compilation preserved the
quality gate but did not support a throughput improvement. The eager default is
retained. Raw observations and the reproducer live in
`benchmarks/real/mlx-fp32-parity77.json`, `mlx-fp32-compile77.json` and
`benchmarks/real/mlx/fp32.py`.

A separate BF16-backbone hypothesis retained all selected labels but changed
eight complete rankings. Maximum drift reached `0.108` logits and `0.0191`
probability, exceeding the declared `1e-3` logit and `1e-4` probability gates.
It was rejected; no production precision flag was added. Its raw outcomes and
reproducer remain in `mlx-bf16-screen77.json` and `benchmarks/real/mlx/precision.py`.

FP16 was tested separately because it retains more mantissa precision than
BF16. All converted parameters and outputs were finite; selected labels and
accuracy stayed unchanged. One full ranking changed, with maximum drift of
`0.0137` logits and `0.00241` probability, so it also failed the same gate.
No production dtype flag was added. Reproduce `mlx-fp16-screen77.json` with
`python benchmarks/real/mlx/precision.py --dtype float16`; the same runner handles
both precision experiments without a second inference implementation.
