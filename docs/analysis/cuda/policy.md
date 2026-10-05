# CUDA precision policy ownership

The exact Torch v2.12.1 tagged source supports **process-wide** ownership of
SDPA selection and FP32 precision. This replaces the earlier inference from
local Torch 2.14 source. The installed CUDA binary and a concurrent execution
race have not been independently inspected or exercised.

`globalContext()` owns one function-static `Context`. SDPA setters mutate its
ordinary fields; the FP32 getter resolves inherited settings before returning
an effective value. Python bindings invoke that owner.
[Context implementation](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/Context.cpp),
[fields](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/Context.h),
[Python bindings](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/csrc/Module.cpp).

The public SDPA context snapshots backends, changes global flags and restores
its snapshot in `finally`, without a lock or task-local owner. Consequently,
overlapping threads or an asynchronous yield inside a scope can affect another
inference or restore stale settings. This consequence is inferred from source;
it is not a measured race.
[Attention context](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/attention/__init__.py).

Saving an effective precision getter and restoring it through a leaf setter can
turn an inherited setting into an explicit override. Effective read-back equality
does not establish restoration of the hidden inheritance tree. Existing serial
research workers own their process; cleanup claims retain this distinction.
[Precision properties](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/backends/__init__.py),
[inheritance implementation](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/Context.cpp).

The smallest coherent deployment candidate is one dedicated process with a
fixed precision/backend policy configured before admitting work. Incompatible
policies need separate processes. A per-application lock cannot control unrelated
Torch callers. This proposes an ownership boundary; it releases no serving
change. Numerical, causal, shape, cancellation and resource gates still apply.
IEEE matmul remains distinct from attention's internal arithmetic.
[Public controls](https://docs.pytorch.org/docs/2.12/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

`cuda-policy-ownership-review.json` retains URLs, line references, observed and
inferred claims, and limits. Canonical source bytes/hashes were unavailable to
the web reader; hashes remain unknown. No GPU job, private payload transfer or
production edit accompanied this investigation.
