<p align="center">
  <img src="https://wyat.io/banners/banner.jpg" alt="Wyat" width="100%">
</p>

<h1 align="center">Wyat</h1>

<p align="center">
  Decisions, made locally.
</p>

<p align="center">
  <a href="https://wyat.io">wyat.io</a> ·
  <a href="https://huggingface.co/teilomillet/wyat-1-600m">wyat-1-600m</a> ·
  <a href="https://github.com/teilomillet/wyat/blob/main/docs/README.md">Documentation</a>
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

## Install

Python 3.12 or newer, with [uv](https://docs.astral.sh/uv/).

```sh
uv add wyat --extra inference   # PyTorch, CPU or CUDA
```

Use `--extra mlx` instead on Apple Silicon, and add `--extra server` to serve
over HTTP. The core alone (`uv add wyat`) has no dependencies. The model,
[Wyat-1 600M](https://huggingface.co/teilomillet/wyat-1-600m), downloads on
first use (about 1.5 GB) and every file is checked against pinned SHA-256
digests; `uv run wyat download` fetches it ahead of time. Inference itself
never uses the network.

## Use it

```python
import wyat

model = wyat.load("teilomillet/wyat-1-600m")

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
uv run wyat decide teilomillet/wyat-1-600m < requests.jsonl > decisions.jsonl
uv run wyat serve  teilomillet/wyat-1-600m
```

Besides choosing an option, Wyat rates on an ordered scale (`Request.score`)
and answers yes or no with a probability (`Request.noul`).

## Documentation

| | |
| --- | --- |
| [Quickstart](https://github.com/teilomillet/wyat/blob/main/docs/quickstart.md) | Your first decision, step by step |
| [Embedded demo](https://github.com/teilomillet/wyat/blob/main/docs/embedded.md) | A coding agent acting on Wyat's choice |
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
artifacts carry their own licenses. Contributions are welcome; see
[CONTRIBUTING.md](https://github.com/teilomillet/wyat/blob/main/CONTRIBUTING.md).
