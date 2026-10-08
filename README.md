# Wyat

Load a local decision model, choose between supplied candidates, evaluate labeled
requests, and serve the same model over HTTP. Python values and JSONL are the
entry points.

The current loader supports **Kai Decision 2.0's full Qwen3 candidate-head
artifact**, for the `choice` task. Wyat's eventual architecture remains open;
Kayak and CLM Lab research checkpoints need an explicit export contract before
this loader can deploy them. See [supported model contracts](docs/training-contracts.md).

## Install in your application

In your own Python **3.12 or newer** uv project, add this checkout as a dependency:

```sh
uv add /path/to/wyat --extra inference
uv run wyat --help
```

On **Apple Silicon macOS**, use `--extra mlx` instead. Add `--extra server`
when you need HTTP serving.
The package and Python import are `wyat`; the public input value is `wyat.Request`.
uv supports [local path dependencies and dependency extras](https://docs.astral.sh/uv/concepts/projects/dependencies/).
The [PyPI 0.0.1 release](https://pypi.org/project/wyat/) differs from this
implementation, so use the local path until a matching release is available.
The installed-package guide also shows how to build and use a transferable wheel,
so an application need not retain an editable checkout.
The [installed-package guide](docs/install.md) has copyable load and server
examples; the [Python API guide](docs/api.md) explains the public contract.

## Start without model weights

From this checkout, with Python **3.12 or newer** and `uv`:

```sh
uv sync
uv run python examples/scorer/main.py
uv run wyat --help
```

The example uses a small custom scorer to exercise the public API. It does not
load trained weights or establish model quality. The Python core has no runtime
dependencies.

The [result](examples/results/main.py) and [evaluation](examples/evaluation/main.py)
examples also run without weights. Use `uv run python` with either path.

## Install an inference backend

Choose the runtime for your machine, then activate the same environment:

```sh
# PyTorch CPU, plus the HTTP server
uv sync --extra inference --extra server
source .venv/bin/activate
```

On **Apple Silicon macOS**, native MLX is an alternative:

```sh
uv sync --extra mlx --extra server
source .venv/bin/activate
export MLX_ENABLE_TF32=0
```

MLX requires no PyTorch installation. Select `backend="mlx"` in Python or
`--backend mlx` on the CLI; it defaults to the Apple GPU and uses FP32 inference.
The default backend is PyTorch CPU, also FP32. These commands install from the
checkout; they do not assume a published package or download model weights.
Set `MLX_ENABLE_TF32=0` before starting Python or the CLI to use the measured MLX
configuration below; changing it after importing MLX does not establish that
configuration.

## Load and decide

Follow the [pinned acquisition and integrity guide](docs/model-acquisition.md)
to obtain the supported local Kai artifact. It needs
`decision_config.json`, local tokenizer files, `backbone/config.json`, full
Safetensors backbone weights, and `decision_head.safetensors`. Loading does not
download files or execute custom model Python. Adapter-only and unsupported
architectures are rejected.

```python
import wyat

model = wyat.load("/path/to/Decision-2.0-Kai-0.6B")  # PyTorch CPU
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
print(decision.scores_by_id[decision.selected])
print(decision.probabilities_by_id[decision.selected])
print(decision.to_dict())

# Reuse the loaded model for a batch.
batch = model.decide_many((request, request))
```

From the checkout, run the [model example](examples/model/main.py) with
`uv run python examples/model/main.py /path/to/kai`.

For Apple GPU inference, change the load call to
`wyat.load(path, backend="mlx")`; no conversion or reduced-precision flag is
needed. Keep the model loaded across requests.

Scores and softmax probabilities follow input candidate order. Rankings contain
candidate IDs, with input order preserved for exact ties. Probabilities are
relative to the supplied candidates and are not calibrated confidence. The model
ID fingerprints the local artifact contents.

Use `scores_by_id` and `probabilities_by_id` for ID-based lookups; do not zip
ranked IDs with input-ordered values. Candidate lists are accepted and stored as
immutable tuple snapshots. `query` is the instruction; `context` is the evidence.
Unicode IDs remain exact. Raw candidate scores are not rubric ratings; the
[task guide](docs/decision-tasks.md) distinguishes Choice, Score, and Noul.

Kai requires **at least 2 candidates** and enforces the artifact's declared
`max_options` (up to 255), unique candidate IDs, and `task="choice"`.
Inputs exceeding `max_length` fail without truncation; the
loader default is 2048 tokens. Token caching and compilation are opt-in and are
not required for first use.

The [complete workflows](docs/agent-workflows.md) connect preparation, result
interpretation, application policy, action dispatch and failure handling. Their
routing, prioritization, human-review and policy-evaluation examples use fixed
logits, run without weights and state their limits explicitly.

## Decide and evaluate from JSONL

The checked-in [request](examples/booking/requests.jsonl) contains one JSON object
per line, using the same fields as `Request`. The [evaluation case](examples/booking/cases.jsonl)
adds an ID and accepted labels outside the request:

```json
{"id":"booking-1","request":{"query":"Choose the requested action","context":"Cancel my booking","candidates":[{"id":"cancel","text":"Cancel the booking"},{"id":"create","text":"Create a booking"}]},"expected":["cancel"]}
```

```sh
MODEL=/path/to/Decision-2.0-Kai-0.6B
wyat decide "$MODEL" < examples/booking/requests.jsonl > decisions.jsonl
wyat evaluate "$MODEL" < examples/booking/cases.jsonl > report.json
```

Add `--backend mlx` for Apple GPU inference. `decide --batch-size 8` batches up to
eight requests. Output goes to stdout and diagnostics to stderr; the model loads
once per command. Invalid input or inference failure returns status 1. Previously
emitted streaming decisions remain emitted.

Evaluation validates the complete cohort before inference and retains failures
in the denominator. Its report includes predictions, errors, accuracy, coverage,
input identity and p50/p95 elapsed latency. Loading is excluded; rendering,
inference and decision interpretation are included. Evaluation makes one model
call per case. Multiple accepted labels
are supported. Independent labels and a frozen cohort are required to assess
model quality. [DMBench adapters](docs/dmbench-integration.md) reuse a loaded model
for scalar or batch evaluation.

## Serve

```sh
wyat serve "$MODEL" --host 127.0.0.1 --port 8000
```

For integration into a Python application, see the
[server example](examples/server/main.py) and [installed-package guide](docs/install.md).
The checkout's server example is an ASGI factory:

```sh
WYAT_MODEL=/path/to/kai uv run uvicorn examples.server.main:app --factory --workers 1 --host 127.0.0.1 --port 8000
```

Use `--backend mlx` for Apple GPU inference. `GET /health` reports model identity
and status. `POST /v1/decide` accepts the same request JSON:

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  --data-binary @examples/booking/requests.jsonl
```

The default body limit is 1 MiB. Malformed JSON returns 400, unsupported input
422, oversized input 413, and failed inference 500. One inference is admitted at
a time; an overlapping request receives 503. A worker thread runs inference so
health and admission remain available. Localhost is the default; deployments
choose their own authentication and TLS boundary.

## Measure on your workload

Install both runtimes to compare CPU and Apple GPU:

```sh
uv sync --extra inference --extra mlx --extra server
MLX_ENABLE_TF32=0 wyat benchmark "$MODEL" --backend mlx --device gpu --batch-size 1 \
  --baseline-backend torch --baseline-device cpu --baseline-batch-size 1 \
  --torch-threads 8 --max-length 512 --token-cache-size 0 \
  --mlx-cache-limit-mib 512 --warmup 1 --repeats 5 \
  < cases.jsonl > measurement.json
```

Use your labeled `cases.jsonl`, not the single illustrative booking case, for a
performance conclusion. The benchmark retains five alternating pairs after
warmup, all predictions and failures, and numerical, full-ranking and accuracy
gates. Status 1 indicates inference or quality failure; a positive median alone
is not a performance guarantee. `--torch-threads` explicitly changes process-wide
Torch threads. Loading, HTTP and admission queues are outside the timing.

On the recorded Apple Silicon host, current full-stock FP32 MLX measured a **2.84×**
median gain over eight-thread full-stock CPU inference, preserving all strict
numerical gates and full rankings across 924 warmup/measured outputs. This applies
to 77 six-choice BANKING77 training-source development requests and the pinned
Kai artifact; checkpoint exposure is unknown. It is a backend comparison, not a
held-out quality score, server result, or universal speed claim. The
[measured setup and limits](docs/analysis/runtime/native.md) and
[real-data protocol](docs/real-benchmark.md) retain the evidence.

## Reference and development

- [Documentation map](docs/README.md): runtime options, contracts and measurements.
- [MLX runtime](docs/mlx-inference.md) and [CPU/CUDA runtime](docs/torch-inference.md):
  precision and device policies. CUDA uses a separate BF16 policy and needs its
  own checkpoint parity evidence; the CPU/MLX result does not establish it.
- [Opt-in prefix reuse](docs/analysis/prefix.md) for MLX and
  [CPU prefix reuse](docs/analysis/cpu/prefix.md): explicit resource lifetimes;
  default loading remains full prefill.
- [Contribution guide](CONTRIBUTING.md): structure, dependency groups and checks.
- [Release guide](docs/releasing.md): CI, checked artifacts and deliberate publication.
- [Implementation evidence](docs/implementation-evidence.md),
  [benchmark receipts](benchmarks/README.md), [Semantic Router investigation](docs/semantic-router-inference.md),
  and [open questions](open-question.md).

For another trained logit model, implement `wyat.Scorer.score_many(requests)` and
construct `wyat.Model(scorer, model_id="your-artifact-identity")`. The scorer owns
its rendering, limits, resource lifetime and concurrency. It returns finite raw
logits in request and candidate order. Embeddings, sigmoid validity scores and
calibrated probabilities require their own interpretation; see the
[dependency-free example](examples/scorer/main.py).

The [decision-task guide](docs/decision-tasks.md) also documents the pure native
Choice/Score/Noul training contract. Try its no-weight readout example with
`uv run python examples/tasks/main.py`. This does not enable native research
adapters in the ordinary Kai loader.
