# Python API

Load a model once, describe a decision with `wyat.Request`, and call
`model.decide(request)`. Import public values from `wyat`; `wyat.wire` handles
JSON parsing for the CLI and server and is not the Python model interface.

## Load, describe, decide

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m")
request = wyat.Request(
    query="Choose the requested action",
    context="Cancel my booking",
    candidates=[
        wyat.Candidate(id="cancel", text="Cancel the booking"),
        wyat.Candidate(id="create", text="Create a booking"),
    ],
)
decision = model.decide(request)
print(decision.selected)
```

The candidate ID identifies a choice in your application; the text describes
that choice to the model. wyat renders both IDs and descriptions in its input
(Score renders rubric positions, not custom IDs).
Use the selected ID to dispatch an action. A decision chooses among supplied
candidates; it does not execute that action.

`query` is the instruction or question. `context` is the supporting message or
evidence, as text or as a JSON object or array. Python constructors store a JSON
context as the same canonical text that `predict` renders for a record `state`
(sorted keys, compact separators), so do not call `json.dumps` first. The HTTP
`context` field remains text. Candidate IDs and text preserve their exact
Unicode spelling in Python values, rendering and JSON transport. The configured tokenizer may normalize
text internally; exact application IDs do not establish learned language quality.

The default for `wyat.load()` is
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m).
The first load downloads its immutable release and verifies every file against
the pins in the installed package; later loads verify and reuse the cache.
Pass a local model directory to use an existing copy instead. Other strings
are treated as local paths unless they name a supported pinned release.

800M uses the `wyat-option-fork-v1` format and PyTorch on CPU by default;
`device="cuda"` or `device="mps"` selects a compatible CUDA or Apple GPU, and
`backend="mlx"` selects MLX on Apple Silicon. Native full-task and single-choice
candidate-head artifacts retain their existing PyTorch and MLX paths. See the
[application installation guide](install.md) and
[artifact requirements](model-acquisition.md).
Keep the loaded model outside your per-request handler. Loading and inference
have separate lifetimes.

## Public values and operations

| Name | Purpose |
| --- | --- |
| `wyat.Candidate(id, text)` | Immutable application ID and model-readable description. |
| `wyat.Candidate.from_schema(id, schema)` | Build a candidate from a tool or JSON schema mapping (`description`, `parameters`). |
| `wyat.Request(query, candidates, context="", task="choice")` | Immutable input to one decision; also `Request.choice(...)`, `Request.score(...)`, and `Request.noul(...)`. |
| `wyat.Decision` | Model identity, raw scores, probabilities, ranked IDs, uncertainty metrics (`score_gap`, `probability_margin`, `entropy`, `normalized_entropy`), and task properties (`probability_true`, `expected_ordinal_value`). |
| `wyat.load(path=..., ...)` | Load a local option-fork, native full-task or single-choice candidate-head artifact, or a pinned release, and return a `wyat.Model`; the default is the pinned 800M release. |
| `model.decide(request, *, temperature=None)` | Return one `Decision`. |
| `model.decide_many(requests, *, temperature=None, errors="raise")` | Return decisions in request order, using one scorer batch call (800M scores its requests one after another within it); with `errors="return"`, a refused request gets its `ModelInputError` in its slot. |
| `wyat.PublicRecord` | Input to `predict`: `{"state": text or JSON, "questions": {id: {"type", "instructions"?, "criteria"}}}`. Choice and Noul `criteria` map IDs to text; Score `criteria` is an ordered list of levels. |
| `model.predict(record, *, temperature=None)` | Interpret a public multi-question record (`choice`, `score`, `noul`) into typed answers (`ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`) with the same uncertainty properties as `Decision`, the model ID and an input digest. Returns `wyat.Answers`. |
| `wyat.Answers` | The `dict` of typed answers keyed by question ID that `predict` and `predict_many` return. `answers.choice(id)`, `answers.score(id)` and `answers.noul(id)` return that question's answer with its type narrowed, and raise `TypeError` if the question has another type. |
| `model.predict_many(records, *, temperature=None)` | Typed answers for several records, in record order. Native models encode every record before scoring any, so one refused record fails the call before compute; each record is then scored in its own budgeted groups. |
| `model.logits(record)` | Return one raw option logit list per question in a public multi-question record. |
| `wyat.public_questions(record)` | Validate a record into its canonical state and `wyat.Question` values. |
| `wyat.interpret(question, logits)` | Interpret raw logits for a `wyat.Question` as a typed answer, without temperature. |
| `wyat.Scorer` | Typed boundary for custom models returning raw logits. |
| `wyat.Model(scorer, model_id, *, temperature=1.0)` | Wrap a custom scorer with the same decision interpretation. |
| `wyat.ModelOutputError` | A scorer returned output that violates the logit contract. |
| `wyat.ModelInputError` | A model rejected input with an allowlisted recovery reason; also a `ValueError`. |

Construct requests with named fields or task constructors (`Request.choice`,
`Request.score`, `Request.noul`). `candidates` accepts a nonempty ordered
sequence of `Candidate` values, such as a list or tuple, with unique, nonempty
IDs. The request stores an immutable tuple snapshot; later changes to the
original list do not change the request. Unordered collections and generators
are not input forms. The query and candidate text must also be nonempty. The
single-choice candidate-head loader requires at least two candidates,
`task="choice"`, and enforces its artifact's candidate and token limits; the native full-task loader
(`wyat-native-full-v1`) and option-fork loader (`wyat-option-fork-v1`) support
`choice`, `score`, and `noul`. Both full-task loaders support cross-call prefix
caching (`prefix_cache_size`): native artifacts retain candidate-prefix K/V,
while option-fork artifacts retain the rendered record-state block plus the
hybrid backbone's attention and gated-delta-net state. Omit the argument for the
family defaults (native 16 entries, option-fork 0), or pass `0` to disable
caching. A retained prefix is only reused when its token IDs exactly prefix the
new request; reuse changes floating-point grouping, so cached results follow the
documented per-family numerical tolerance rather than bitwise equality. Loaders
reject unsupported tasks and inputs beyond the configured limits; they never
truncate a request silently.
Native `predict` and `logits` score a record's questions in contiguous groups
that fit the artifact's padded batch budget, so a record with many questions
gives the same answers as one batch within numeric tolerance. Their input-size
and encoding refusals raise `wyat.ModelInputError`, as `decide` does; a
malformed record still raises a plain `ValueError`.

Reuse a loaded model and the same request values for batching:

```python
decisions = model.decide_many((request, request))
assert len(decisions) == 2
```

An empty batch returns an empty tuple without invoking the scorer. Input errors
raise `ValueError`. Invalid output shape, cardinality, or numeric logits raise
`wyat.ModelOutputError`, a subclass of `ValueError`: catch it first when the
distinction matters. Missing runtime dependencies, unreadable artifacts, and exceptions raised while executing
the scorer retain their own exceptions. A failed inference produces no successful
`Decision` for that call.

```python
try:
    decision = model.decide(request)
except wyat.ModelOutputError:
    raise  # Investigate the scorer's output contract.
except wyat.ModelInputError as error:
    print(error.reason)  # Safe category: task, candidate count or input size.
    raise  # Correct the stated cause before explicitly submitting again.
except ValueError:
    raise  # Inspect the input or the backend's stated rejection.
```

By default one refused request fails the whole `decide_many` call. Pass
`errors="return"` to keep the batch going: each request the model refuses gets
its `wyat.ModelInputError` in its result slot, and the others are decided. A
batch-level refusal, such as a padded token budget, is split in halves until
every request that fits is decided. Treat the two kinds of failure differently:

- `ModelInputError` is about one input. Retrying it unchanged gives the same
  refusal; change the input or route it elsewhere.
- `ModelOutputError` and other exceptions from the scorer are about the
  backend. They still raise with `errors="return"`; investigate them, and retry
  only when the backend failure is transient.

```python
results = model.decide_many(requests, errors="return")
for request, result in zip(requests, results):
    if isinstance(result, wyat.ModelInputError):
        print("refused:", result.reason)
    else:
        print(result.selected)
```

The [bulk-job example](../examples/use_cases/expert/09_batch_throughput.py)
uses this to label a backlog in which one pasted log is too long for the model.

The wrapper does not relabel exceptions thrown by the scorer itself or while
advancing a custom output iterator. In particular, a custom scorer can raise its
own `ValueError`; inspect its documented failure contract rather than assuming every such exception originates in request
validation.

`ModelInputError.reason` is one of `unsupported-task`, `candidate-count`,
`input-length` or `input-segment`. wyat uses these reasons at its task, menu and
encoding boundaries. The local message can include a configured numerical limit;
HTTP exposes only the allowlisted reason, never arbitrary backend text. The
library never repairs or truncates an input, and never retries a failed call.
Some paths split work without changing any input: native `predict` scores a
long record's questions in budgeted groups, and both
`decide_many(..., errors="return")` and the HTTP server's batcher halve a
refused batch to isolate the refused requests.

## Read a result

`decision.selected` is the highest-ranked candidate ID. `decision.ranking`
contains every candidate ID in descending score order; exact ties preserve input
order. `decision.scores` and `decision.probabilities` follow the original
candidate order, **not** ranking order.

```python
print(decision.scores_by_id[decision.selected])
print(decision.probabilities_by_id[decision.selected])
print(decision.score_gap, decision.probability_margin, decision.normalized_entropy)

for candidate_id, score in decision.scores_by_id.items():
    print(candidate_id, score)
```

`decision.candidate_ids` gives IDs in original input order. The keyed properties
return independent dictionary snapshots in that order. They need no retained
request, and changing a returned dictionary does not change the decision.
They also work on a `Decision` built directly from its four fields
(`model_id`, `scores`, `probabilities`, `ranking`). These properties reject
invalid structure and non-finite numbers, and preserve supplied values. They do
not verify probability bounds, normalization, or calibration.

For abstention and fallback routing, `Decision` also provides four read-only
uncertainty properties:

- `decision.score_gap`: raw logit difference between the top-ranked and
  second-ranked candidates (requires at least two candidates).
- `decision.probability_margin`: probability difference $p_{(1)} - p_{(2)}$
  between the top two candidates under the decision's temperature (requires a
  valid probability distribution over at least two candidates).
- `decision.entropy`: Shannon entropy $-\sum_i p_i \ln p_i$ in nats over the
  candidate probabilities.
- `decision.normalized_entropy`: Shannon entropy divided by $\ln K$, bounded in
  $[0, 1]$ across different menu sizes $K \ge 2$.

For decisions returned by `Model`, probabilities are the softmax of the
scorer's candidate logits scaled by `temperature`. They are relative to that
candidate set and require post-hoc temperature calibration before being treated
as calibrated confidence. A directly constructed `Decision` preserves supplied
probabilities; establish its producer before claiming this relationship.
`decision.to_dict()` provides the JSON-compatible result used by the CLI and
HTTP server. It contains `model_id`, `selected`, `ranking`, `scores`, and
`probabilities`.

Typed answers from `model.predict` and `model.predict_many` have the same four
uncertainty properties, with the same definitions. They are computed from the
answer's FP32 logit snapshots. Each answer also records its source:
- `model_id` is the model that produced it.
- `input_sha256` is a digest of that question's public input: the canonical
  state, question ID, type, instructions and options. It is
  `wyat.tasks.question_input_sha256(state, question)`.
Log the digest to match an answer to its exact input without storing the
input. A JSON instruction and its canonical text give the same digest.
`answer.to_dict()` includes `model_id` and `input_sha256` when they are set.
Answers built directly with `wyat.interpret` leave them empty.

`predict` returns a `wyat.Answers` dict. `answers.choice(id)`,
`answers.score(id)` and `answers.noul(id)` give an answer with its type
narrowed for type checkers, and raise `TypeError` for another question type.

```python
record = {
    "state": "Cancel my booking",
    "questions": {
        "route": {
            "type": "choice",
            "criteria": {"cancel": "Cancel the booking", "create": "Create a booking"},
        },
    },
}
answers = model.predict(record)
route = answers.choice("route")
if route.probability_margin < 0.2:
    print("send to a person", route.model_id, route.input_sha256)
```

Raw candidate scores are logits, not a rubric rating. Single-choice candidate-head
artifacts support Choice, while native full-task (`wyat-native-full-v1`) and
option-fork (`wyat-option-fork-v1`) artifacts support Choice, ordinal Score, and
yes/no Noul. See [decision task meanings](decision-tasks.md).
The [use-case examples](../examples/use_cases/README.md) show each task in a
short application, from support routing to reranking.

## Evaluate labeled cases in Python

Use the evaluation module with the same loaded model and request:

```python
from wyat.evaluation import EvaluationCase, evaluate

cases = [EvaluationCase(id="booking-1", request=request, expected=("cancel",))]
report = evaluate(model, cases)
print("Accuracy:", report.accuracy)
print("Uncalibrated ECE:", report.expected_calibration_error)

# Fit temperature on validation cases to optimize log-loss without retraining:
best_temperature, calibrated = report.calibrate()
print(f"Optimal temperature: {best_temperature:.3f}")
print("Calibrated ECE:", calibrated.expected_calibration_error)
```

`expected` contains accepted application IDs. Cases have distinct IDs, and the
whole cohort is validated before inference. Failed model calls remain rows in
the report and remain in the accuracy denominator; coverage reports successful
calls separately. Calibration metrics (`expected_calibration_error`,
`brier_score`, `log_loss`, `mean_confidence`) measure probability alignment.
`brier_score` sums over all options, so for a `Request.noul` case it is twice
the usual binary Brier score of `probability_true`.
`report.selective(min_margin)` returns what a policy that automates only
decisions with `probability_margin >= min_margin` would do: `accuracy` on the
automated cases (`None` if there are none) and `automation_rate`, the share of
all cases automated. Failed calls stay in the total and are never automated; a
decision with a single candidate always is. Choose the margin on cases other
than those you report it on.
`report.rescale(temperature)` recomputes probabilities for any temperature
without re-running model inference, and `report.calibrate()` fits optimal
scaling post-hoc. The [runnable evaluation example](../examples/evaluation/main.py)
shows one correct prediction, one wrong prediction, and one failed call using
an explicitly fixed scorer without weights. The
[calibration example](../examples/use_cases/expert/05_calibrate_threshold.py)
goes on to choose an abstention threshold from labelled cases and check it on
held-back ones.

From the command line, `wyat evaluate MODEL < cases.jsonl` reads one
`{"id", "request", "expected": [ids]}` object per line, writes one report, and
exits with status 1 if any call failed.

## Use the same input over HTTP

The JSON request has the same fields as the Python value; JSON arrays replace
Python tuples:

```json
{
  "query": "Choose the requested action",
  "context": "Cancel my booking",
  "candidates": [
    {"id": "cancel", "text": "Cancel the booking"},
    {"id": "create", "text": "Create a booking"}
  ]
}
```

Send this object to `POST /v1/decide`, or one object per line to `wyat decide`.
The server validates JSON, constructs a `Request`, and calls the same
`model.decide` operation. See [serving](install.md#serve-the-same-model) for
startup and admission limits, and
[recovery](#recover-from-transport-failures) for error responses. To embed the server in your application,
use `from wyat.server import create_app`, then `app = create_app(model)` with an
already-loaded model.

The server publishes its OpenAPI 3.1 contract at `/openapi.json`, with interactive
Swagger UI at `/docs` and ReDoc at `/redoc`. `POST /v1/decide` declares a required
JSON body, a five-field decision response, examples for Choice, Score and Noul,
and typed `{"detail": "..."}` errors for 400, 413, 422, 500 and 503. `GET /health`
and `GET /readyz` declare typed `HealthResponse` and `ReadinessResponse` bodies.
Every documented response advertises the `X-Request-ID` header, and `/v1/decide`
also declares the optional `X-wyat-Rejection` header for the model-owned rejection
categories.

Unknown request and candidate fields are rejected. Query, task, candidate IDs
and descriptions must contain non-whitespace text; all text must be valid UTF-8.
Candidate IDs must be unique. Score additionally accepts text levels, assigning
their zero-based positions as IDs. Noul supplies the default false/true pair
when candidates are omitted or empty. Custom task names remain available to
scorers that support them; the schema does not imply model support. The loaded
model owns candidate-count and input-size limits.

The schema is generated from Pydantic models, but requests are decoded and
validated by the core parser after a bounded body read: oversized bodies,
duplicate JSON keys and malformed JSON are rejected there. Cross-field rules
such as unique candidate IDs and equal response-vector lengths are described in
the contract and enforced by the core, not by JSON Schema alone.

## Health, readiness and request IDs

`GET /health` is a cheap liveness probe for the process and returns the typed
`HealthResponse`, `{"status": "ok", "model_id": "..."}`. `GET /readyz` reports
readiness together with the loaded artifact's identity and the server's admission
limits as a typed `ReadinessResponse` (values shown for `wyat serve` defaults
with a native artifact):

```json
{
  "status": "ready",
  "model_id": "native-…",
  "model_type": "NativeModel",
  "temperature": 1.0,
  "max_body_bytes": 1048576,
  "max_queue_size": 16,
  "queue_timeout": 30.0
}
```

For native artifacts `model_id` is a content digest, so the same release is
identified by the same string across hosts and restarts. Readiness carries no
request data and is safe to expose to a load balancer.

Every HTTP response carries an `X-Request-ID`. A supplied value is echoed when
it matches `[A-Za-z0-9._-]{1,64}`; otherwise, or when the header is absent, the
server generates a 32-character hex identifier. The response echoes the safe
value it used. When the access log is enabled the app also emits one INFO
record per request on the `wyat.server.access` logger with the method, path,
status, request ID and wall-clock latency. It never logs request text, headers,
bodies or client addresses. `wyat serve` prints these records to stderr in place
of Uvicorn's access line unless logging is already configured; an embedded app
configures its own handler. Disable them with
`create_app(model, log_access=False)` or `wyat serve --no-access-log`.

## Recover from transport failures

CLI `decide` reports the physical input line or line range for a failing inference
batch, including blank lines in that numbering. Already emitted decisions remain
emitted; no decisions from the failed batch are emitted. A range identifies the
whole batch, not a known failing member. Inference is not retried or split to find
one. Keep the input and diagnostic together and correct the cause before a new run.

The CLI configures real standard text streams as strict UTF-8 when needed. It
keeps injected custom text streams unchanged. Applications embedding
`wyat.cli.main()` must supply fresh standard input or input already configured
as strict UTF-8: Python cannot change a text input's encoding after it has been
read. Use the Python API for ordinary in-process integration.

HTTP uses the status codes and JSON bodies below. A model-owned rejection
additionally carries `X-wyat-Rejection`; generic backend `ValueError` has no such
header. Do not infer an input-policy reason from an arbitrary backend exception.

| Response | Permitted next step |
| --- | --- |
| 400 invalid JSON | Correct UTF-8, duplicate keys or JSON syntax before resubmission. |
| 413 body limit | Reduce the body deliberately; do not silently discard evidence. |
| 422 with `unsupported-task` | Use a task the loaded artifact supports: native artifacts accept `choice`, `score` and `noul`; candidate-head artifacts accept only `choice`. |
| 422 with `candidate-count` | Supply a count the artifact accepts: 2–255 for native Choice and Score, exactly 2 for Noul, or 2 through `max_options` for candidate-head artifacts. |
| 422 with `input-length` or `input-segment` | Inspect the complete rendered input and configured token/segment limits; shorten deliberately or use an independently admitted larger-limit model. |
| 422 without a reason header | Inspect request validation and the documented backend contract; no successful decision exists. |
| 500 | Investigate the producer/runtime failure; keep it separate from semantic review. |
| 503 | Wait for the model's admission slot; the application owns any bounded retry. |

`X-wyat-Rejection` adds a response header only: the 422 body remains
`{"detail":"request is unsupported by this model"}`. Consumers may ignore the
header. Proxies need to preserve it if the application uses it for recovery.
The header contains a fixed category, never submitted text, a file path or
arbitrary exception content. Browser clients need their deployment's usual CORS
header exposure policy to read custom headers.

## Why retain `Request`?

`Request` names the input shared by single decisions, batches, JSONL, evaluation,
and HTTP. Its immutable candidates preserve the ID-to-score relationship across
those paths. Keeping `model.decide(request)` also keeps input validation separate
from the loaded model's state and effects.

A method accepting either a request or optional query/candidate arguments would
introduce conflicting combinations and two input forms to explain. We retain
one form and make the examples explicit. This is a design judgment supported by
the current call sites, not a measured claim that every user prefers it.

Comparable libraries also separate model loading from a named task operation:
[Sentence Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
loads a `CrossEncoder` and calls `rank(query, documents)`;
[MLX LM](https://github.com/ml-explore/mlx-lm/blob/main/README.md) loads a model
and tokenizer before calling `generate`. These examples support the separation
of lifetimes, but do not prescribe wyat's input shape. wyat retains application
candidate IDs and one request shape across local and HTTP decisions.

The design follows these primary-source principles:

- [Google's Go guide](https://google.github.io/styleguide/go/guide.html) puts
  clarity and simplicity before concision, and asks that values and decisions
  remain easy to trace.
- [Google's Python guide](https://google.github.io/styleguide/pyguide.html#219-power-features)
  favors straightforward code and supports ordinary dataclasses;
  its [type guidance](https://google.github.io/styleguide/pyguide.html#221-type-annotated-code)
  recommends checking annotations when public APIs change.
- [Joshua Bloch's API-design essay](https://www.infoq.com/articles/API-Design-Joshua-Bloch/)
  recommends designing around concrete use cases, retaining those examples as
  tests, limiting mutability, and using overloads cautiously.

For an example without model weights or inference dependencies, run the
[custom scorer](../examples/scorer/main.py). Custom scorers return finite raw
logits in request and candidate order; embeddings and other score semantics need
their own model contract.

The [ergonomics standard](ergonomics.md) describes the public API design principles.


### macOS resource headroom

`wyat.load(..., max_resource_fraction=0.85)` leaves a default 15% memory
reserve on macOS. Set the fraction explicitly to change it, for example
`wyat.load(path, device="mps", max_resource_fraction=0.75)`.

Before loading and each inference call, Wyat checks system-wide available RAM
(free plus inactive VM pages), including other applications. Compressed and
wired memory are not added to the available amount. Loading uses a conservative weight-size estimate;
Ouro additionally estimates context K/V, attention and candidate workspaces before
scoring and checks pressure between layers/chunks. Oversized work or unavailable
telemetry raises `wyat.ResourceLimitError` (a `MemoryError`), including with
`errors="return"`; it is never retried by the input-error isolation mechanism.
Close other applications or reduce the input before retrying.

CPU thread counts are reduced to at most the selected fraction of logical cores
(with a minimum of one). Apple GPU allocators receive a memory limit; MLX treats
its limit as a guideline. These runtime settings apply to the entire process.
Guarded calls in one process run sequentially. Non-macOS execution is unchanged.
Direct `load_ouro`, `OuroScorer.from_directory` and `OuroScorer` accept the same
`max_resource_fraction` option. Other families receive the guard through
`wyat.load`; constructing their backend classes directly bypasses it.

This is a cooperative memory safeguard, **not a hard 85% CPU/GPU utilization
cap or an operating-system reservation**. Estimates are conservative, memory
pressure is sampled, and other apps or already queued kernels can still consume
resources between checks. Other model families have call-level admission checks,
not Ouro's token-aware peak estimates or layer checkpoints. A 32K model context
limit therefore does not promise that 32K fits the current Mac and budget.


### Bounded-memory Ouro inference

For an Ouro v2 artifact, `wyat.load(path, device="mps", memory_mode="streaming")`
(or `device="cpu"`) stages one original base-weight module in FP32 at a time.
The original BF16 checkpoint values promote exactly; trained FP32 adapters and
scoring head retain their values. This performs no quantization or retraining.
Resident mode remains the default.

Long chunked contexts store their exact FP32 K/V bank in an anonymous temporary
file. Candidate scoring reads one layer into reusable CPU/Apple GPU buffers.
The file is uncached on macOS and reclaimed when the request finishes, fails,
or its process exits. At32K, the bank needs approximately12GiB temporary disk
space. The same resource fraction bounds RAM admission and disk capacity;
it defaults to85%. Streaming trades additional disk reads and transfers for
lower resident memory. It requires shared-context candidate batching and no
prefix cache. It preserves the artifact's context chunking and attention mode.

The working-set estimate, live checkpoints, CPU-worker and GPU-allocation
bounds remain active. A busy Mac or insufficient disk headroom can still cause
`ResourceLimitError`; streaming does not reserve capacity against other apps.
Qualified on 2026-10-27 on a 24 GiB Apple Silicon Mac: exact 32,768-token
Choice, Noul and Score requests completed on both CPU and Apple GPU under the
default 85% ceiling, and the 32,769-token boundary was refused. Two Apple GPU
32K Choice runs were bitwise identical, and CPU versus Apple GPU answers agreed
with the same argmax (maximum probability difference 1.9e-5). Those are
measurements of this streaming implementation on one machine, not a claim about
any published release: the shipped release still uses the resident loader.
