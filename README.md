# Wyat

Choose between application-owned candidates with a local decision model. Use the
same request and result in Python, JSONL evaluation, and an HTTP server.

Wyat has a dependency-free Python core and optional PyTorch, Apple Silicon MLX,
and HTTP runtimes. Model weights are acquired separately. The current loader
supports the full **Kai Decision 2.0 Qwen3 candidate-head artifact**, for Choice.

## Install

In your application's [uv](https://docs.astral.sh/uv/getting-started/installation/)
project with Python **3.12 or newer**, the matching-release install command is:

```sh
uv add wyat
```

Version **0.0.9 is a release candidate in this source tree, not a published
release**. The existing PyPI `0.0.1` package differs from this API. Until a
matching release is published, install a checked candidate wheel instead:

```sh
uv add /absolute/path/to/wyat-0.0.9-py3-none-any.whl
```

If you need a new application project, create it before adding Wyat:

```sh
uv init --bare --python 3.12 wyat-demo
cd wyat-demo
```

Choose optional features after a matching release:

```sh
# For trained inference on PyTorch CPU:
uv add wyat --extra inference
# For Apple Silicon macOS instead:
uv add wyat --extra mlx
# Add HTTP serving to your chosen runtime:
uv add wyat --extra inference --extra server
```

Select extras on that wheel with `--extra inference --extra server`, for example.
uv manages your environment and lockfile; use `uv run` to execute commands.
See the [installation guide](docs/install.md) for local builds and the pip alternative.

## Your first decision

The API is **load → describe the choices → decide → use the selected ID**.
With the `inference` extra installed and a
[verified local Kai model](docs/model-acquisition.md):

```python
import wyat

model = wyat.load("/path/to/Decision-2.0-Kai-0.6B")
request = wyat.Request(
    query="Choose the requested booking action.",
    context="Please cancel my booking.",
    candidates=[
        wyat.Candidate(id="create", text="Create a booking"),
        wyat.Candidate(id="cancel", text="Cancel the booking"),
    ],
)
decision = model.decide(request)
print(decision.selected)  # One of your candidate IDs: create or cancel.
```

`query` is the instruction; `context` is the message to consider. Each candidate
pairs your application's `id` with model-readable `text`. Keep the loaded model
across calls. Selecting an ID executes no application action.

For a complete walkthrough, follow [your first Wyat decision](docs/quickstart.md).
To try the API with only the core installed, copy the
[demo](examples/quickstart/main.py) as `demo.py` and run:

```sh
uv run python demo.py --demo
# Selected: cancel
```

Demo mode uses fixed scores and needs no weights. Run
`uv run python demo.py --model /path/to/Decision-2.0-Kai-0.6B` to use a trained model
with the same request and result API.

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
uv run wyat decide /path/to/kai < requests.jsonl > decisions.jsonl
uv run wyat serve /path/to/kai --host 127.0.0.1 --port 8000
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
