<p align="center">
  <img src="assets/banner.jpg" alt="Wyat" width="100%">
</p>

<h1 align="center">Wyat</h1>

<p align="center">
  Decisions, made locally.
</p>

---

Wyat chooses between options your application defines. You provide a message
and a set of candidates. A small local model returns the ID of one of them.
It is built to perform well in French while staying strong in English.
Nothing leaves your machine, and nothing runs until you act on the result.

## Try it

No model required. The demo uses fixed scores to show the interface.

```sh
git clone https://github.com/teilomillet/wyat && cd wyat
uv run python examples/quickstart/main.py --demo
# Selected: cancel
```

## Use it

```python
import wyat

model = wyat.load("/path/to/wyat-model")

decision = model.decide(wyat.Request(
    query="Choose the requested booking action.",
    context="Please cancel my booking.",
    candidates=[
        wyat.Candidate(id="create", text="Create a booking"),
        wyat.Candidate(id="cancel", text="Cancel the booking"),
    ],
))

decision.selected  # "cancel"
```

Load once. Decide many times. The same request and result work in Python,
JSONL, and HTTP:

```sh
uv run wyat decide /path/to/model < requests.jsonl > decisions.jsonl
uv run wyat serve  /path/to/model
```

## Install

Python 3.12 or newer, with [uv](https://docs.astral.sh/uv/).

```sh
uv add wyat                                    # Core, no dependencies
uv add wyat --extra inference                  # PyTorch
uv add wyat --extra mlx                        # Apple Silicon
uv add wyat --extra inference --extra server   # HTTP
```

> [!NOTE]
> 0.0.9 is a release candidate and is not yet on PyPI. The published 0.0.1 has
> a different API. Until release, install a checked wheel as described in the
> [installation guide](docs/install.md).

Model weights are acquired separately and verified locally. See
[model acquisition](docs/model-acquisition.md).

## Documentation

| | |
| --- | --- |
| [Quickstart](docs/quickstart.md) | Your first decision, step by step |
| [Embedded demo](docs/embedded.md) | A coding agent acting on Wyat's choice |
| [Python API](docs/api.md) | Requests, results, batching, serving, evaluation |
| [Workflows](docs/agent-workflows.md) | Routing, prioritization, review, recovery |
| [Explaining decisions](docs/explaining-decisions.md) | Reading scores and owning policy |
| [All documentation](docs/README.md) | |

More examples and short films are coming.

## License

Code is licensed under [Apache 2.0](LICENSE); see [NOTICE](NOTICE).
Model artifacts carry their own licenses. Contributions are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md).
