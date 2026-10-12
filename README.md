# Wyat

Choose between application-owned candidates with a local decision model. Use the
same request and result in Python, JSONL evaluation, and an HTTP server.

Wyat has a dependency-free Python core and optional PyTorch, Apple Silicon MLX,
and HTTP runtimes. Model weights are acquired separately. The current loader
supports the full **Kai Decision 2.0 Qwen3 candidate-head artifact**, for Choice.

## Install

Use Python **3.12 or newer**. Version **0.0.9 is a release candidate in this
source tree, not a published release**. The existing PyPI `0.0.1` package differs
from this API. These are the target commands after a matching release:

```sh
python -m pip install wyat
# For trained inference on PyTorch CPU:
python -m pip install 'wyat[inference]'
# For Apple Silicon macOS instead:
python -m pip install 'wyat[mlx]'
# Add HTTP serving to your chosen runtime:
python -m pip install 'wyat[inference,server]'
```

For now, install a checked candidate wheel:

```sh
python -m pip install /path/to/wyat-0.0.9-py3-none-any.whl
```

Select extras on that wheel when needed, for example
`'/path/to/wyat-0.0.9-py3-none-any.whl[inference,server]'`.
See the [installation guide](docs/install.md) for environments and local builds.

## Try the API without weights

Copy this into `demo.py` and run `python demo.py` after installing the core:

```python
from collections.abc import Sequence

import wyat


class FixedScorer:
    def score_many(
        self, requests: Sequence[wyat.Request]
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((0.0, 2.0) for _ in requests)


model = wyat.Model(FixedScorer(), model_id="demo-fixed-logits")
request = wyat.Request(
    query="Choose the requested action",
    context="Cancel my booking",
    candidates=[
        wyat.Candidate(id="create", text="Create a booking"),
        wyat.Candidate(id="cancel", text="Cancel the booking"),
    ],
)
decision = model.decide(request)
print(decision.selected)  # cancel
print(decision.scores_by_id[decision.selected])  # 2.0
print(decision.to_dict())
```

This fixed scorer demonstrates integration; it has no learned decision quality.
Selecting an ID executes no application action. A custom scorer returns finite
raw logits in request and candidate order and owns its runtime and resource limits.

## Use a trained local model

Follow the [pinned acquisition and integrity guide](docs/model-acquisition.md)
to download and verify `vllm-sr/Decision-2.0-Kai-0.6B`. The supported artifact
needs local tokenizer files, configuration, full backbone Safetensors weights,
and its decision head. Loading is local: Wyat downloads no weights and executes
no custom model Python. Adapter-only and unsupported artifacts are rejected.

With the `inference` extra installed, replace the demo model with:

```python
model = wyat.load("/path/to/Decision-2.0-Kai-0.6B")  # PyTorch CPU
# Reuse the request above and keep the loaded model across calls.
decision = model.decide(request)
decisions = model.decide_many((request, request))
```

On Apple Silicon macOS with the `mlx` extra, use
`wyat.load(path, backend="mlx")`. It defaults to the Apple GPU. See the
[MLX](docs/mlx-inference.md) and [PyTorch](docs/torch-inference.md) runtime guides
for device, precision, and resource settings.

Kai supports `task="choice"`, requires at least two candidates, and enforces the
artifact's declared maximum candidate count. Candidate IDs must be unique.
Inputs beyond the configured token limit fail without truncation; the loader
default is 2048 tokens. Raw scores are logits. Softmax probabilities are relative
to the supplied candidates and are not calibrated confidence.

Scores and probabilities follow input order; rankings contain candidate IDs.
Use `scores_by_id` and `probabilities_by_id` for keyed access. Exact score ties
preserve input order, and Unicode application IDs remain exact.

## JSONL and HTTP

Save one request object per line in `requests.jsonl`:

```json
{"query":"Choose the requested action","context":"Cancel my booking","candidates":[{"id":"create","text":"Create a booking"},{"id":"cancel","text":"Cancel the booking"}]}
```

```sh
wyat decide /path/to/kai < requests.jsonl > decisions.jsonl
wyat serve /path/to/kai --host 127.0.0.1 --port 8000
```

Serving also needs the `server` extra. Add `--backend mlx` to select MLX.
`GET /health` reports model identity and status; `POST /v1/decide` accepts the same
JSON request. One inference is admitted at a time, and overlapping requests
receive 503. Localhost is the default; applications choose authentication and TLS.

For evaluation, `wyat evaluate /path/to/kai` reads labeled JSONL cases from stdin.
Failures remain in the denominator. The [API guide](docs/api.md) documents case
format, output ordering, errors, and Python evaluation.

## Learn more

- [Installation](docs/install.md) and [Python API](docs/api.md)
- [Model acquisition](docs/model-acquisition.md)
- [Application workflows](docs/agent-workflows.md) and [explaining decisions](docs/explaining-decisions.md)
- [Contributing](CONTRIBUTING.md) and [release preparation](docs/releasing.md)

Wyat's code is licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE) for
attribution. Model artifacts have their own licenses.
