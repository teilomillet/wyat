# CUDA numerical admission

Prospective declaration, 2026-10-05. The Wyat engine task owns this bounded
experiment. Production CUDA precision and default loading remain unchanged.
The earlier [feasibility audit](cuda.md) distinguishes the released BF16 CUDA
recipe from a strict FP32 control.

Fresh read-only observations found the RTX 4070 Ti at zero utilization, about
128–131 MiB allocated, and no compute-process rows. These observations are not
a reservation. Require an uncontended pre-allocation observation and reject
foreign compute ownership before each batch and after the cohort. Do not signal
or alter other jobs.
Retain contention, unsupported settings, allocation errors and timeouts.

The private experiment directory uses an existing read-only Torch
`2.12.1+cu130` installation, Python 3.12 and a separate Transformers `5.18.0`
dependency layer. Existing environments are unchanged; no Torch download is
needed. Record full package/build identities and exact source bytes, including
the maintained Qwen and SDPA integration. The nine canonical files, including
attribution, total about 1.519 GB; their bytes, dataset and CPU reference
must match the manifest before device allocation. No arbitrary remote code is
loaded. The model identity remains `kai:1f5525e01de43a724c71824579ef5195595fba916ce1e02be7bbbbea50f25cee`.

The first arm is one complete batch-one cohort of all 77 original BANKING77
training-source six-choice cases. Load through the CPU FP32 path, validate all
token IDs/endpoints/query positions against the frozen CPU reference, then move
all floating parameters and buffers to CUDA FP32. Preserve encoder, candidate
order, endpoint gathering, causal padding and the FP32 head. Use modern IEEE
global and CUDA matmul settings, disabled autocast and forced SDPA MATH.
Do not call production CUDA `score_many` for this control because it explicitly
enables BF16 autocast. Verify observed projection/head dtypes and autocast state;
recording those fields without enforcing the declared precision is insufficient.

Use the unchanged full-ranking and selected-choice gates, logit absolute/relative
tolerance `1e-3`, probability tolerance `1e-4`, no failures and no accuracy drop
against the frozen CPU reference. The original dataset/reference/input hashes
are respectively `0020d9a3…`, `b8a90b31…` and `fd5dd942…`. Retain every output or
failure in the denominator. A canonical runner that has not returned its entire
cohort after a timeout supplies no completed partial-quality claim.

Only a reviewed, immutable successful FP32 batch-one receipt permits either a
batch-eight FP32 control or a separate ordinary released-CUDA arm. Keep identities
and gates fixed. Batch-eight additionally requires the declared memory margin.
Released CUDA uses its existing lossless BF16 Linear storage and BF16 backbone
autocast, with FP32 residuals, norms and head. A failed strict gate rejects that
arm; the upstream recipe does not waive numerical admission.

Run each arm in one owned fresh worker with a 180-second whole-process cap and
process-group cleanup. Restore precision state, remove observers and preserve
late-write/cleanup failures so completed-quality flags cannot survive a failed
required boundary. Store raw receipts and exact source archives; independently
review the actual result before using it as a downstream control.

This is numerical feasibility, not a latency benchmark or production promotion.
Observer hooks and a single cold cohort cannot support a speed comparison.
Checkpoint exposure is unknown; these are 77 unique training-source development
examples, not held-out Wyat quality. Repetitions would not add independent labels.
