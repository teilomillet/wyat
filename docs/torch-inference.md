# CPU and CUDA inference

Kai performs one causal forward pass, then reads each candidate endpoint and the
final decision endpoint. It has no autoregressive decode loop. Wyat retains the
released segmented tokenizer recipe and FP32 decision head; replacing this with
last-token classification would change the trained contract.

PyTorch owns matrix kernels and attention dispatch. Its
[scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
selects an available implementation for the device and input. Wyat defaults to
SDPA and exposes eager attention as a reference for numerical checks. Different
kernels and batch shapes can produce different floating-point rounding, so
benchmark reports retain logit differences and selected-candidate agreement.

Wyat pads only on the right and gathers only valid candidate/query endpoints.
The causal attention rule excludes every future padding position from those
readouts, so inference needs no separate padding mask or padding to a multiple
of eight. Hypothesis tests perturb future pad tokens and compare the readouts
against an independently masked reference.

CPU inference uses FP32. Thread counts belong to the process running the model,
not the reusable scorer: changing PyTorch threads affects other models in the
same process. Sweep threads on the actual machine and workload rather than
assuming more threads are faster. This follows
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
so returned scores have completed device execution. Benchmarks also synchronize
around measured CUDA work, following
[PyTorch's CUDA timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution).

The architecture has similarities to
[vLLM pooling models](https://docs.vllm.ai/en/latest/models/pooling_models/), but
Kai's candidate endpoints and shared bilinear/MLP head require a custom pooler.
Wyat does not introduce a second serving engine before that integration has a
measured benefit and passes the same artifact, tokenization and quality gates.
Compilation, weight quantization and KV prefix reuse also remain experiments
until they improve a stated real workload without unacceptable decision drift.

The real Kai release was checked on 77 fixed BANKING77 development cases with
six supplied candidates per case. Kai's exposure to the source data is unknown;
these cases check engine parity, not generalization. The simpler causal path
preserved every complete candidate ranking and selected candidate against the
previous explicit-mask path, with maximum logit difference `9.54e-7`.
Both paths classified 67 of the 77 cases correctly.
[The FP32 reference](../benchmarks/real/torch-fp32-reference77.json) retains token
IDs, endpoint positions and both sets of logits.

Seven alternating timing pairs on the fixed first 16 cases found roughly a
`1.02x` median change for causal-only attention. Singleton timing varied across
pairs, so this change supports a simpler implementation without a meaningful
CPU speedup claim. The
[singleton](../benchmarks/real/torch-causal-confirm-batch1.json) and
[batch-eight](../benchmarks/real/torch-causal-confirm-batch8.json) receipts retain
every measured output. Thread and batch choices are machine/workload settings;
the screen does not establish a portable thread default.
The [screen reproducer](../benchmarks/real/torch/causal/screen.py) and
[confirmation reproducer](../benchmarks/real/torch/causal/confirm.py) preserve
the original rounded, explicitly masked implementation as their reference.
Their default output directory keeps newly measured receipts separate from the
historical evidence.

Dynamic INT8 QNNPACK quantization was rejected. On the same 16 cases it was
approximately three times slower, changed 13 of the 16 complete rankings, and
introduced maximum logit/probability differences of `1.808`/`0.219`. It failed
the predeclared quality gate, so the 77-case follow-up was skipped and no
quantization runtime option was added. The
[reproducer](../benchmarks/real/torch/quantization.py) and
[failure-inclusive receipt](../benchmarks/real/torch-qnnpack-16.json) preserve the
experiment. This result applies to the tested QNNPACK dynamic Linear recipe on
the M4 Pro, not every quantization method or CPU.

CPU Inductor compilation was also rejected for this workload. The dedicated
subprocess completed within its 120-second budget, spending `30.62 s` on JIT
and its first 16-case cohort. Five paired warmed measurements preserved all
rankings and decisions, but compiled inference was slower in every pair:
median speedup `0.919x`, versus the unchanged FP32 engine. The
[bounded reproducer](../benchmarks/real/torch/compile.py),
[paired receipt](../benchmarks/real/torch-compile-16.json) and
[wall-time receipt](../benchmarks/real/torch-compile-wall.json) retain the result.
No compilation option was added to the runtime.
