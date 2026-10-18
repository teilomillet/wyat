# Install wyat in your application

Use Python **3.12 or newer**. The package and import are `wyat`.
Install the core, describe a decision with `wyat.Request`, and call a loaded
model's `decide` method. Inference libraries and model weights are separate.
The recommended workflow uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage an application project, its environment and its lockfile.

## Create your application project

```sh
uv init --bare --python 3.12 wyat-demo
cd wyat-demo
```

If you already have a uv project, run the commands below from its directory.
`uv run` uses that project's environment; no shell activation is needed.

## Add wyat

```sh
uv add wyat --extra inference
uv run wyat --help
```

These uv commands also work in PowerShell. The Python API is the same; native
Windows inference requires its own runtime verification. Environment-variable
examples below use POSIX syntax unless labeled otherwise.

Choose only the extras you need:

| Extra | Purpose |
| --- | --- |
| *(none)* | The core: requests, results, evaluation and `wyat download`; no dependencies |
| `inference` | PyTorch inference, using CPU unless a device is explicit |
| `mlx` | MLX inference for compatible native/candidate-head artifacts on Apple Silicon; excludes 800M |
| `server` | FastAPI and Uvicorn for HTTP serving alongside an inference backend |

For 800M serving, use `uv add wyat --extra inference --extra server`.
Compatible native/candidate-head artifacts can instead use
`uv add wyat --extra mlx --extra server` on Apple Silicon. Dependency installation
provides no trained weights. The self-contained [quickstart demo](quickstart.md)
works with the core alone. Installed wheels do not include the source checkout's
`examples/` files.

## Acquire and load a trained model

The upcoming v0.1.1 targets
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m), which is also the
library default. Its named download is not pinned or available yet. Until it
is, obtain a prepared 800M artifact and set `WYAT_MODEL` to its local model
directory, as below. `wyat.load()` and `wyat download` report the pending release
rather than downloading a different model.

800M uses the `wyat-option-fork-v1` format with PyTorch, on CPU by default or a
compatible CUDA device with `device="cuda"`. MLX does not support this model.
The [acquisition guide](model-acquisition.md) covers artifact layout, validation
and explicit legacy compatibility. Loading local weights needs no network or
downloaded model code.

Copy this into `decide.py` in your application:

```python
import os

import wyat

model = wyat.load(os.environ["WYAT_MODEL"])  # PyTorch CPU
request = wyat.Request(
    query="Choose the requested action",
    context="Cancel my booking",
    candidates=[
        wyat.Candidate(id="cancel", text="Cancel the booking"),
        wyat.Candidate(id="create", text="Create a booking"),
    ],
)
decision = model.decide(request)
actions = {"cancel": "cancel_booking", "create": "create_booking"}
print(actions[decision.selected])  # ID lookup; no booking changed.
print(decision.scores_by_id[decision.selected])
print(decision.to_dict())
```

```sh
WYAT_MODEL=/path/to/wyat-1-800m/model uv run python decide.py
```

For a compatible native/candidate-head artifact, the `mlx` extra and
`wyat.load(local_path, backend="mlx")` select MLX on Apple Silicon. MLX defaults
to the Apple GPU; `device="cpu"` selects MLX CPU explicitly. Set
`MLX_ENABLE_TF32=0` before starting Python for the FP32 configuration in the
[MLX guide](mlx-inference.md). These settings do not apply to 800M.

Keep the loaded model across requests. Call `model.decide_many((request,
request))` for a batch. wyat requires 2 to 255 candidates per request and
rejects other counts with `ModelInputError` instead of truncating. See the [API guide](api.md) and runtime guides for
[PyTorch](torch-inference.md) and [MLX](mlx-inference.md).

For Windows PowerShell, set the model name before running the same Python script:

```powershell
$env:WYAT_MODEL = "C:\path\to\wyat-1-800m\model"
uv run python decide.py
```

## Serve the same model

Install the `server` extra with your chosen inference backend. The CLI loads
one model and serves it on localhost:

```sh
uv run wyat serve /path/to/wyat-1-800m/model --host 127.0.0.1 --port 8000
```

For a compatible native/candidate-head artifact, add `--backend mlx` for MLX.
To embed serving in your application, copy this into
`app.py`:

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import wyat

if TYPE_CHECKING:
    from fastapi import FastAPI


def app() -> FastAPI:
    from wyat.server import create_app

    return create_app(wyat.load(os.environ["WYAT_MODEL"]))
```

Change the load call to `wyat.load(os.environ["WYAT_MODEL"], backend="mlx")`
for an MLX-compatible artifact. Start the factory with one worker:

```sh
WYAT_MODEL=/path/to/wyat-1-800m/model uv run uvicorn app:app --factory --workers 1 --host 127.0.0.1 --port 8000
```

Importing `app.py` loads no weights. The factory loads one model per worker;
additional workers each need their own model memory and admission gate.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/readyz
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: startup-check-1' \
  --data '{"query":"Choose the requested action","context":"Cancel my booking","candidates":[{"id":"cancel","text":"Cancel the booking"},{"id":"create","text":"Create a booking"}]}'
```

`GET /health` is a cheap liveness check. `GET /readyz` reports the loaded
artifact's identity (its `model_id` is a content digest for native artifacts) and
the admission limits, so a load balancer or release check can confirm the exact
model behind a deployment. Every response carries an `X-Request-ID`: a supplied
value is echoed when it matches `[A-Za-z0-9._-]{1,64}` and the server generates
one otherwise. `wyat serve` prints one access line per request to stderr
(method, path, status, request ID and latency, never request text or client
addresses) in place of Uvicorn's own access line; `--no-access-log` turns it
off. In an embedded app, the same records go to the `wyat.server.access` logger
at INFO level; configure a handler to record them.

By default, `create_app(model)` admits one active inference and returns 503 for
overlapping requests; pass `max_queue_size` and `queue_timeout` (enabled by
default in `wyat serve`) to bound concurrent queueing and micro-batching. Pass
`log_access=False` to disable the access records. Applications choose authentication and TLS at their deployment boundary.
Relative probabilities do not certify correctness, and a selection performs no
action. The [application workflows](agent-workflows.md) connect ID lookup,
dispatch, review and recovery.

## Alternative: pip

If you prefer pip, copy `demo.py` from the [quickstart](quickstart.md), then
create and activate a Python 3.12+ virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install 'wyat[inference]'
python demo.py --model /path/to/wyat-1-800m/model
```

On PowerShell, activate
with `.venv/Scripts/Activate.ps1`. Use `python -m pip install wyat` for the
core-only demo. Once the environment is active, run Python scripts and the
`wyat` CLI directly instead of prefixing them with `uv run`.
