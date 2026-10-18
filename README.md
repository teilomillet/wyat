<p align="center">
  <img src="https://raw.githubusercontent.com/teilomillet/wyat/main/assets/banner.gif" alt="wyat" width="100%">
</p>

<h1 align="center">wyat</h1>

<p align="center">
  Decisions, made locally.
</p>

<p align="center">
  <a href="https://wyat.io">wyat.io</a> ·
  <a href="https://huggingface.co/teilomillet/wyat-1-800m">wyat-1-800m</a> ·
  <a href="https://github.com/teilomillet/wyat/blob/main/docs/README.md">Documentation</a>
</p>

---

wyat chooses between options your application defines. You provide a message
and a set of candidates. A small local model returns the ID of one of them.
The model was trained on French and English decisions.
Nothing leaves your machine, and nothing runs until you act on the result.

The model is [wyat-1 800M](https://huggingface.co/teilomillet/wyat-1-800m).
It runs with PyTorch on CPU, CUDA or Apple MPS, or with MLX on Apple Silicon.
Version **0.1.1** downloads its pinned
release and verifies every file before loading.

## Try it

No model required. The demo uses fixed scores to show the interface.

```sh
git clone https://github.com/teilomillet/wyat && cd wyat
uv run python examples/quickstart/main.py --demo
# Selected: cancel
```

## Install

Python 3.12 or newer, with [uv](https://docs.astral.sh/uv/).

```sh
uv add wyat --extra inference   # PyTorch: CPU, CUDA or Apple MPS
uv add wyat --extra mlx         # MLX on Apple Silicon
```

Add `--extra server` to serve over HTTP. The core alone (`uv add wyat`) has
no dependencies. Model weights are separate from the package. `wyat.load()`
downloads and verifies wyat-1 800M on first use, then reuses the verified cache.
Run `uv run wyat download` to fetch it ahead of time. Inference itself never
uses the network.

On Apple Silicon, use the GPU: `wyat.load(backend="mlx")` with the `mlx` extra,
or `wyat.load(device="mps")` with PyTorch. Both give the same scores as PyTorch
CPU, which is several times slower there.

## Use it

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m")  # also the default for wyat.load()

decision = model.decide(wyat.Request(
    query="Choose the requested booking action.",
    context="Please cancel my booking.",
    candidates=[
        wyat.Candidate(id="create", text="Create a booking"),
        wyat.Candidate(id="cancel", text="Cancel the booking"),
    ],
))

decision.selected  # one of your IDs, such as "cancel"
```

Load once. Decide many times. The same request and result work in Python,
JSONL, and HTTP:

```sh
uv run wyat decide teilomillet/wyat-1-800m < requests.jsonl > decisions.jsonl
uv run wyat serve  teilomillet/wyat-1-800m
```

Besides choosing an option, wyat rates on an ordered scale (`Request.score`)
and answers yes or no with a probability (`Request.noul`).

## Documentation

| | |
| --- | --- |
| [Quickstart](https://github.com/teilomillet/wyat/blob/main/docs/quickstart.md) | Your first decision, step by step |
| [Embedded demo](https://github.com/teilomillet/wyat/blob/main/docs/embedded.md) | A coding agent acting on wyat's choice |
| [Python API](https://github.com/teilomillet/wyat/blob/main/docs/api.md) | Requests, results, batching, serving, evaluation |
| [Workflows](https://github.com/teilomillet/wyat/blob/main/docs/agent-workflows.md) | Routing, review, recovery |
| [Explaining decisions](https://github.com/teilomillet/wyat/blob/main/docs/explaining-decisions.md) | Reading scores and owning policy |
| [Use cases](https://github.com/teilomillet/wyat/blob/main/examples/use_cases/README.md) | Choice, Score and Noul in short scripts |
| [Expert use cases](https://github.com/teilomillet/wyat/blob/main/examples/use_cases/expert/README.md) | Triage, agents, RAG gates, calibration |
| [Integrations](https://github.com/teilomillet/wyat/blob/main/examples/use_cases/integrations/README.md) | LangChain, LangGraph, LlamaIndex, MCP, pandas |
| [All documentation](https://github.com/teilomillet/wyat/blob/main/docs/README.md) | |

Coding agents can start from [llms.txt](https://github.com/teilomillet/wyat/blob/main/llms.txt).

## License

Code is licensed under [Apache 2.0](https://github.com/teilomillet/wyat/blob/main/LICENSE);
see [NOTICE](https://github.com/teilomillet/wyat/blob/main/NOTICE). Model
artifacts carry their own licenses. The 800M trained adapters and decision head
are released for non-commercial use; its backbone and runtime use Apache 2.0.
Contributions are welcome; see
[CONTRIBUTING.md](https://github.com/teilomillet/wyat/blob/main/CONTRIBUTING.md).
