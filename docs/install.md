# Install Wyat in your application

Use Python **3.12 or newer**. The package and import are `wyat`.
Install the core, describe a decision with `wyat.Request`, and call a loaded
model's `decide` method. Inference libraries and model weights are separate.
The recommended workflow uses [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage an application project, its environment and its lockfile.

From your application's uv project, the matching-release install command is:

```sh
uv add wyat
```

Version **0.0.9 is currently unpublished**. Until a matching release is
available, use the [checked candidate wheel](#install-a-checked-candidate-wheel-now).

## Create your application project

```sh
uv init --bare --python 3.12 wyat-demo
cd wyat-demo
```

If you already have a uv project, run the commands below from its directory.
`uv run` uses that project's environment; no shell activation is needed.

## Registry installation target

Version **0.0.9 is the candidate in this source tree and has not been published**.
The existing PyPI `0.0.1` distribution differs from the API documented here.
Use a checked local wheel until a matching release is available.

After a matching release, add the needed features to your application:

```sh
uv add wyat
uv add wyat --extra inference
uv run wyat --help
```

These uv commands also work in PowerShell. The Python API is the same; native
Windows inference requires its own runtime verification. Environment-variable
examples below use POSIX syntax unless labeled otherwise.

Choose only the extras you need:

| Extra | Purpose |
| --- | --- |
| `inference` | PyTorch inference, using CPU unless a device is explicit |
| `mlx` | Native MLX inference on Apple Silicon macOS |
| `server` | FastAPI and Uvicorn for HTTP serving alongside an inference backend |

For example, the matching-release targets are
`uv add wyat --extra inference --extra server` or, on Apple Silicon macOS,
`uv add wyat --extra mlx --extra server`. The core itself has no runtime
dependencies. Dependency installation provides no trained weights.

## Install a checked candidate wheel now

In your application environment, install the exact wheel supplied by the
maintainer:

```sh
uv add /absolute/path/to/wyat-0.0.9-py3-none-any.whl --extra inference --extra server
uv run wyat --help
uv run python -c 'import sys, wyat; print(sys.executable); print(wyat.__file__)'
```

Use `--extra mlx --extra server` instead on Apple Silicon macOS, or omit extras
for the core. `uv add` records the wheel source and requested extras in your project.
Keep the wheel's hash with your application dependency records. Run the import
check from your application directory so a checkout cannot shadow the installed
wheel. The imported file should belong to your application environment.

Maintainers build the candidate from an approved public source export; see
[release preparation](releasing.md). To build from that exported directory:

```sh
uv sync
uv build --no-build-isolation
uv run --no-sync python scripts/package/check.py dist
```

The self-contained [quickstart demo](quickstart.md)
works with the core alone. Installed wheels do not include the source checkout's
`examples/` files.

## Acquire and load a trained model

Follow the [pinned acquisition and integrity guide](model-acquisition.md) to
obtain the supported full Kai Choice artifact. Copy its standard-library
verifier from the public source example and retain the verified artifact path.
The loader accepts a local artifact directory, not an arbitrary Hub model name.
It downloads nothing and executes no custom model Python.

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
WYAT_MODEL=/absolute/path/to/kai uv run python decide.py
```

With the `mlx` extra, change the load call to
`wyat.load(os.environ["WYAT_MODEL"], backend="mlx")`. MLX defaults to the Apple
GPU; `device="cpu"` selects MLX CPU explicitly. For the documented FP32 MLX
configuration, set `MLX_ENABLE_TF32=0` before starting Python:

```sh
MLX_ENABLE_TF32=0 WYAT_MODEL=/absolute/path/to/kai uv run python decide.py
```

Keep the loaded model across requests. Call `model.decide_many((request,
request))` for a batch. Kai requires Choice requests with two through the
artifact's maximum number of candidates and rejects over-limit inputs without
truncation. See the [API guide](api.md) and runtime guides for
[PyTorch](torch-inference.md) and [MLX](mlx-inference.md).

For Windows PowerShell, set the path before running the same Python script:

```powershell
$env:WYAT_MODEL = "C:/Models/kai"
uv run python decide.py
```

## Serve the same model

Install the `server` extra with your chosen inference backend. The CLI loads
one model and serves it on localhost:

```sh
uv run wyat serve /absolute/path/to/kai --host 127.0.0.1 --port 8000
```

Add `--backend mlx` for MLX. To embed serving in your application, copy this into
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
for MLX. Start the factory with one worker:

```sh
WYAT_MODEL=/absolute/path/to/kai uv run uvicorn app:app --factory --workers 1 --host 127.0.0.1 --port 8000
```

Importing `app.py` loads no weights. The factory loads one model per worker;
additional workers each need their own model memory and admission gate.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  --data '{"query":"Choose the requested action","context":"Cancel my booking","candidates":[{"id":"cancel","text":"Cancel the booking"},{"id":"create","text":"Create a booking"}]}'
```

Each app admits one active inference; an overlapping request receives 503.
Applications choose authentication and TLS at their deployment boundary.
Relative probabilities do not certify correctness, and a selection performs no
action. The [application workflows](agent-workflows.md) connect ID lookup,
dispatch, review and recovery.

## Alternative: pip

If you prefer pip, create and activate a Python 3.12+ virtual environment and
install the checked candidate wheel:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '/absolute/path/to/wyat-0.0.9-py3-none-any.whl[inference]'
python demo.py --model /absolute/path/to/kai
```

Copy `demo.py` from the [quickstart](quickstart.md) first. On PowerShell, activate
with `.venv/Scripts/Activate.ps1`. Omit `[inference]` for the core-only demo.
After a matching PyPI release, use `python -m pip install wyat` or
`python -m pip install 'wyat[inference]'`. Once the environment is active, run
Python scripts and the `wyat` CLI directly instead of prefixing them with `uv run`.
