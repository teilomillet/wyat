# Constant fragment loops

Prospective declaration, 2026-10-05. The Wyat engine task owns this experimental
increment. The previous fixed32 and fixed64 sources and receipts remain frozen
controls; production is unchanged.

Hypothesis: explicit full unrolling of the fixed64 kernel's constant fragment
loops can improve its real gate-projection dispatches. Apple's
[Metal compute guidance](https://developer.apple.com/videos/play/tech-talks/10580/)
describes possible register spills from dynamically indexed arrays. Maintained
Steel uses a Clang full-unroll pragma. These sources motivate the perturbation;
they do not establish that our compiled kernel spills or that unrolling helps.

Change only constant fragment initialization, K8 iteration, fragment loading,
row/column matrix accumulation and fragment stores. Preserve the outer K32
traversal, 64×64 output tile, 128 threads, 32 KiB shared storage, shared layouts,
safe math, operand types, BF16 local decoding, reduction order and tail bounds.
Do not combine this with transposed shared loads or direct output stores.

Admit the previous fixed64 main receipt `22dcca46…`, independent audit
`9099425d…`, tooling verification `966ba6a3…`, their exact source archives and the
original fixed32 evidence. Reuse the exact captured real X/W file `70f0245d…`.
Require the model, data and CPU-reference identities before importing the device
framework. No model recapture or substitute activations are allowed.

Before timing, reproduce maintained and frozen64 cold output bits, at all four
shapes: M=167, 1336, 192, 1536; K=1024; N=3072. Require every finite candidate
output within `1e-4 + 1e-4 * abs(reference)` against maintained FP32. Require the
unrolled kernel's FP32/BF16 cold outputs to agree bitwise. Tiny tests separately
cover arbitrary FP32 activations, low bits, signed zeros, tails and layout
rejection. Save all cold outputs, including the FP32 storage control.

Time three arms: maintained FP32, frozen64 BF16 and unrolled64 BF16. Use all six
permutations, ten separately synchronized dispatches per arm/block, and one warm
dispatch per arm/shape. Include dispatch, evaluation and synchronization inside
each interval. Hash every warm/timed output outside its timer against its own
cold output. This observation may perturb later dispatches; it does not measure
uninterrupted throughput.

Success still requires maintained/unrolled paired median at least 1.05 and
interpolated fifth percentile greater than 1 at **both prefix shapes**. Keep
full-input controls separate. Frozen64/unrolled is descriptive evidence of the
perturbation, not a promotion gate or a substitute for beating maintained MLX.
Retain all orders, failures and variation; do not revise thresholds after data.

Use a framework-free parent and one owned worker capped at 180 seconds. Disable
TF32 before MLX import, use the explicit GPU stream, restore the parent environment
and the allocator-cache limit after applying 512 MiB, and reap only owned
processes. The cache setting is not a total-memory bound. Preserve completed
arms, partial/error receipts and late-write failure status. Archive exact
sources before dispatches; independent source review and native tiny/fake checks
precede the real trial.

Any GPU capture is a separate diagnostic artifact with a separate bounded
owner. Profiled/replayed timing cannot replace this uninstrumented comparison.
Compiler spill counts, occupancy and physical memory traffic remain unknown
until directly observed. A projection win would still require a complete-engine
quality and latency confirmation before production selection.

Pre-run diagnostic limit: the documented `gpudebug` replay attempt timed out
without producing compiler or performance fields. Its owned client was killed
and reaped; no matching replay process was visible afterward, while hidden XPC
session state remains unknown. Capture workers and native correctness tests were
terminal before timing. The result retains this uncertainty and does not claim
physically exclusive compute or a demonstrated register-spill mechanism.
