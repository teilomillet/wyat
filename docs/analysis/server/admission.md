# Server admission before threadpool dispatch

A valid concurrent decision request must receive 503 while this application's
model slot is occupied, including when AnyIO's shared pool is saturated.
Request-body limits, validation order, response schema, sanitized backend errors
and off-event-loop scoring retain their existing behavior. A started inference
continues owning its slot after HTTP cancellation until scoring actually exits.

## Counterexample and change

The previous server acquired its nonblocking model lock inside the worker.
Installed Starlette 1.7.0 delegates to AnyIO 4.15.1, which waits for capacity before
invoking that worker. A controlled HTTPX ASGI reproduction used one thread token
and an event-blocked scorer. Before release, a second request was queued with one
thread token borrowed and one waiter. After release, both requests returned 200
and the scorer ran twice, contradicting the documented concurrent-503 contract.
This is observed scheduling behavior; it needs neither a trained model nor a
network socket or timing-based throughput inference.

The handler now reserves the existing application gate before awaiting the pool.
One small per-request thread lock protects three states and their gate effects:

| Event | Prior state | Result |
| --- | --- | --- |
| Worker claims admission | queued | running; worker owns the slot |
| Handler cancellation or dispatch failure | queued | released; free the slot |
| Handler exits while worker owns admission | running | retain worker ownership |
| Worker exits, including failure | running | released; free the slot |
| Late callback after withdrawal | released | reject before scoring; release nothing |

Every transition and release shares the same handoff critical section. The
handoff lock is never held through an await or scoring call. The app gate stays
reserved until its owner releases it; the short handoff lock protects that
ownership.
No scheduler, executor, global thread budget or model-arithmetic policy was added.

## Verification and limits

Five focused tests passed with actual capacity-one admission, cancellation before
worker start, late callbacks before and during another worker, and release after
submission/scorer failures. Twelve Hypothesis examples perturb cancellation and
callback counts and the two imposed handoff orders. They support these bounded
histories, not exhaustive scheduling proof. Existing started-worker cancellation,
validation, body limit and error tests remain unchanged; all fourteen pass.

The complete native CPU/Metal and actual DMBench regression passed **894 tests**,
zero skips and one existing Starlette/httpx deprecation warning. The owned pytest
child exited and was reaped in 67.712 seconds under its 120-second cap. Strict
mypy passed on 138 maintained files; Ruff lint and formatting passed. Sources
remained stable across checks. Only `wyat/server.py` changed among twenty runtime
files; all other 136 prior maintained sources remain unchanged.

Evidence: [typed counterexample](../../../benchmarks/analysis/results/server-admission-counterexample.json),
[owned reproduction](../../../benchmarks/analysis/results/server-admission-counterexample-wall.json),
[independent patch review](../../../benchmarks/analysis/results/server-admission-source-review.json),
[focused tests](../../../benchmarks/analysis/results/server-admission-focused-wall.json),
and [complete regression](../../../benchmarks/real/engine-server-admission-verification.json).
Initial reproduction/source-check rough edges and exact executed text are retained
separately, rather than overwritten. The patch supports admission correctness;
it establishes no HTTP capacity, online latency, model-quality or kernel speedup.

## Frozen inference evidence

Original CUDA declarations, raw runs, audits and source archives remain immutable.
The pending eight-file L192/L197 transfer does not include the changed server;
the remote twenty-file runtime remains its original frozen snapshot. Its source
identity must be checked against those exact archives, rather than requiring the
current checkout to stay byte-identical forever. The former 889-test receipt is
historical; the new 894-test receipt describes the current server. No new private
payload transfer or CUDA job has occurred while authorization remains pending.
