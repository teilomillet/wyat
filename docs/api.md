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
    candidates=(
        wyat.Candidate(id="cancel", text="Cancel the booking"),
        wyat.Candidate(id="create", text="Create a booking"),
    ),
)
decision = model.decide(request)
print(decision.selected)
```

The candidate ID identifies a choice in your application; the text describes
that choice to the model. Kai renders both IDs and descriptions in its input.
Use the selected ID to dispatch an action. A decision chooses among supplied
candidates; it does not execute that action.

`wyat.load` reads a supported local artifact. It does not download weights.
The default is PyTorch on CPU. On Apple Silicon, use
`wyat.load(path, backend="mlx")` for native MLX. See the
[application installation guide](install.md) and
[artifact requirements](training-contracts.md).
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

Construct requests with named fields. `candidates` is a nonempty tuple of
`Candidate` values with unique, nonempty IDs. The query and candidate text must
also be nonempty. The current Kai loader requires at least two candidates,
`task="choice"`, and enforces its artifact's candidate and token limits.
These model limits are described in the [training contracts](training-contracts.md).

Reuse a loaded model and the same request values for batching:

```python
decisions = model.decide_many((request, request))
assert len(decisions) == 2
```

An empty batch returns an empty tuple without invoking the scorer. Input errors
raise `ValueError`; missing runtime dependencies, unreadable artifacts, and
backend failures can raise their own exceptions. A failed inference produces no
successful `Decision` for that call.

## Read a result

`decision.selected` is the highest-ranked candidate ID. `decision.ranking`
contains every candidate ID in descending score order; exact ties preserve input
order. `decision.scores` and `decision.probabilities` follow the original
candidate order, **not** ranking order.

```python
for candidate, score in zip(request.candidates, decision.scores, strict=True):
    print(candidate.id, score)
```

Probabilities are the softmax of the supplied candidate logits. They are
relative to that candidate set and are not calibrated confidence.
`decision.to_dict()` provides the JSON-compatible result used by the CLI and
HTTP server. It contains `model_id`, `selected`, `ranking`, `scores`, and
`probabilities`.

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
