# CPU and CUDA inference

Wyat performs one causal forward pass, then reads each candidate endpoint and the
final decision endpoint. It has no autoregressive decode loop. Wyat retains the
released segmented tokenizer recipe and FP32 decision head; replacing this with
last-token classification would change the trained contract.

PyTorch owns matrix kernels and attention dispatch. Its
[scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
selects an available implementation for the device and input. Wyat defaults to
SDPA and exposes eager attention as a reference for numerical checks. Different
kernels and batch shapes can produce different floating-point rounding, so
compare raw logits and selected-candidate agreement when changing runtime settings.

Wyat pads only on the right and gathers only valid candidate/query endpoints.
The causal attention rule excludes every future padding position from those
readouts, so inference needs no separate padding mask or padding to a multiple
of eight.

CPU inference uses FP32. Thread counts belong to the process running the model,
not the reusable scorer: changing PyTorch threads affects other models in the
same process. Choose thread counts for the actual machine and workload. This follows
[PyTorch's threading guidance](https://docs.pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
and [ONNX Runtime's tuning guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
Both distinguish operation parallelism from simultaneous-request parallelism.
The CLI's `--torch-threads` option sets the count at the process boundary;
omitting it preserves PyTorch's default.

CUDA inference requires a BF16-capable device. Linear parameters that were stored
as exactly representable BF16 values are restored to BF16; embeddings, norms,
the residual stream and the decision head retain FP32. Linear operations run
under BF16 autocast. Arbitrary FP32 weights are not silently rounded during
loading. CPU and CUDA therefore require explicit numerical and decision parity
checks for each released artifact.

`inference_mode` and disabled KV caching keep training graphs and unused decode
state out of the runtime. Results are materialized on the CPU before returning,
so returned scores have completed device execution.

Load a supported local artifact through the public API:

```python
import wyat

model = wyat.load("/path/to/wyat-model", device="cpu")
# For a compatible CUDA device: device="cuda".
```

Install the inference extra in the same environment as your application; see
[installation](install.md). Dependencies and weights are separate: follow the
[pinned acquisition guide](model-acquisition.md) before loading. Missing runtime
dependencies require installation; malformed or incomplete artifacts require
repairing the artifact. The loader never substitutes another model or retries
inference automatically.
