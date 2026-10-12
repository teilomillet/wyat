# Python API

Load a model once, describe a decision with `wyat.Request`, and call
`model.decide(request)`. Import public values from `wyat`; `wyat.wire` handles
JSON parsing for the CLI and server and is not the Python model interface.

## Load, describe, decide

```python
import wyat

model = wyat.load("/path/to/Decision-2.0-Kai-0.6B")
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
that choice to the model. Kai renders both IDs and descriptions in its input.
Use the selected ID to dispatch an action. A decision chooses among supplied
candidates; it does not execute that action.

`query` is the instruction or question. `context` is the supporting message or
evidence. Candidate IDs and text preserve their exact Unicode spelling in Python
values, rendering and JSON transport. The configured tokenizer may normalize
text internally; exact application IDs do not establish learned language quality.

`wyat.load` reads a supported local artifact. It does not download weights.
The default is PyTorch on CPU. On Apple Silicon, use
`wyat.load(path, backend="mlx")` for native MLX. See the
[application installation guide](install.md) and
[artifact requirements](model-acquisition.md).
Keep the loaded model outside your per-request handler. Loading and inference
have separate lifetimes.

## Public values and operations

| Name | Purpose |
| --- | --- |
| `wyat.Candidate(id, text)` | Immutable application ID and model-readable description. |
| `wyat.Request(query, candidates, context="", task="choice")` | Immutable input to one decision. |
| `wyat.Decision` | Model identity, raw scores, probabilities, and ranked candidate IDs. |
| `wyat.load(path, ...)` | Load a supported local model and return a `wyat.Model`. |
| `model.decide(request)` | Return one `Decision`. |
| `model.decide_many(requests)` | Return decisions in request order, using one scorer batch call. |
| `wyat.Scorer` | Typed boundary for custom models returning raw logits. |
| `wyat.Model(scorer, model_id)` | Wrap a custom scorer with the same decision interpretation. |
| `wyat.ModelOutputError` | A scorer returned output that violates the logit contract. |
| `wyat.ModelInputError` | A model rejected input with an allowlisted recovery reason; also a `ValueError`. |

Construct requests with named fields. `candidates` accepts a nonempty ordered
sequence of `Candidate` values, such as a list or tuple, with unique, nonempty
IDs. The request stores an immutable tuple snapshot; later changes to the
original list do not change the request. Unordered collections and generators
are not input forms. The query and candidate text must
also be nonempty. The current Kai loader requires at least two candidates,
`task="choice"`, and enforces its artifact's candidate and token limits.
The loader rejects unsupported tasks and inputs beyond the configured limits;
it never truncates a request silently.

Reuse a loaded model and the same request values for batching:

```python
decisions = model.decide_many((request, request))
assert len(decisions) == 2
```

An empty batch returns an empty tuple without invoking the scorer. Input errors
raise `ValueError`. Invalid output shape, cardinality, or numeric logits raise
`wyat.ModelOutputError`, a subclass of `ValueError`: catch it first when the
distinction matters. Missing
runtime dependencies, unreadable artifacts, and exceptions raised while executing
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

The wrapper does not relabel exceptions thrown by the scorer itself or while
advancing a custom output iterator. In
particular, a custom scorer can raise its own `ValueError`; inspect its documented
failure contract rather than assuming every such exception originates in request
validation.

`ModelInputError.reason` is one of `unsupported-task`, `candidate-count`,
`input-length` or `input-segment`. Kai uses these reasons at its task, menu and
encoding boundaries. The local message can include a configured numerical limit;
HTTP exposes only the allowlisted reason, never arbitrary backend text. The
library performs no automatic repair, truncation, splitting or retry.

## Read a result

`decision.selected` is the highest-ranked candidate ID. `decision.ranking`
contains every candidate ID in descending score order; exact ties preserve input
order. `decision.scores` and `decision.probabilities` follow the original
candidate order, **not** ranking order.

```python
print(decision.scores_by_id[decision.selected])
print(decision.probabilities_by_id[decision.selected])

for candidate_id, score in decision.scores_by_id.items():
    print(candidate_id, score)
```

`decision.candidate_ids` gives IDs in original input order. The keyed properties
return independent dictionary snapshots in that order. They need no retained
request, and changing a returned dictionary does not change the decision.
They also work on coherent values created with the existing four-argument
`Decision` constructor. These properties reject invalid structure and non-finite
numbers, and preserve supplied values. They do not verify probability bounds,
normalization, or calibration. The constructor itself remains unchanged.

For decisions returned by `Model`, probabilities are the softmax of the
scorer's candidate logits. They are relative to that candidate set and are not
calibrated confidence. A directly constructed `Decision` preserves supplied
probabilities; establish its producer before claiming this relationship.
`decision.to_dict()` provides the JSON-compatible result used by the CLI and
HTTP server. It contains `model_id`, `selected`, `ranking`, `scores`, and
`probabilities`.

Raw candidate scores are logits, not a rubric rating. The current local Kai
loader supports Choice; native rubric Score and yes/no Noul require separate
task contracts. See [decision task meanings](decision-tasks.md).

## Evaluate labeled cases in Python

Use the evaluation module with the same loaded model and request:

```python
from wyat.evaluation import EvaluationCase, evaluate

cases = [EvaluationCase(id="booking-1", request=request, expected=("cancel",))]
report = evaluate(model, cases)
print(report.accuracy)
print(report.to_dict())
```

`expected` contains accepted application IDs. Cases have distinct IDs, and the
whole cohort is validated before inference. Failed model calls remain rows in
the report and remain in the accuracy denominator; coverage reports successful
calls separately. This does not turn the single illustrative booking case into
a quality benchmark. The [runnable evaluation example](../examples/evaluation/main.py)
shows one correct prediction, one wrong prediction, and one failed call using
an explicitly fixed scorer without weights.

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
`model.decide` operation. See [serving](../README.md#serve) for startup,
admission limits, and error responses. To embed the server in your application,
use `from wyat.server import create_app`, then `app = create_app(model)` with an
already-loaded model.

## Recover from transport failures

CLI `decide` reports the physical input line or line range for a failing inference
batch, including blank lines in that numbering. Already emitted decisions remain
emitted; no decisions from the failed batch are emitted. A range identifies the
whole batch, not a known failing member. Inference is not retried or split to find
one. Keep the input and diagnostic together and correct the cause before a new run.

HTTP keeps its existing status codes and JSON bodies. A model-owned rejection
additionally carries `X-Wyat-Rejection`; generic backend `ValueError` has no such
header. Do not infer an input-policy reason from an arbitrary backend exception.

| Response | Permitted next step |
| --- | --- |
| 400 invalid JSON | Correct UTF-8, duplicate keys or JSON syntax before resubmission. |
| 413 body limit | Reduce the body deliberately; do not silently discard evidence. |
| 422 with `unsupported-task` | Use the admitted Choice path; changing a task string cannot admit native Score/Noul. |
| 422 with `candidate-count` | Supply 2 through the artifact's declared `max_options` candidates. |
| 422 with `input-length` or `input-segment` | Inspect the complete rendered input and configured token/segment limits; shorten deliberately or use an independently admitted larger-limit model. |
| 422 without a reason header | Inspect request validation and the documented backend contract; no successful decision exists. |
| 500 | Investigate the producer/runtime failure; keep it separate from semantic review. |
| 503 | Wait for the model's admission slot; the application owns any bounded retry. |

`X-Wyat-Rejection` adds a response header only: the 422 body remains
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
of lifetimes, but do not prescribe Wyat's input shape. Wyat retains application
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

The [result example](../examples/results/main.py) exercises a winner in the second
position and multilingual IDs through keyed score and probability access.

The [ergonomics standard](ergonomics.md) describes the public API design principles.


The CLI configures real standard text streams as strict UTF-8 when needed. It
keeps injected custom text streams unchanged. Applications embedding
`wyat.cli.main()` must supply fresh standard input or input already configured
as strict UTF-8: Python cannot change a text input's encoding after it has been
read. Use the Python API for ordinary in-process integration.
