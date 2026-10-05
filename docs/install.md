# Use Wyat in your application

The package and Python import are `wyat`. Create a `wyat.Request`, load a model
once with `wyat.load`, and call `model.decide(request)`. `wyat.wire` implements
internal JSON handling; it is not the request API.

## Install the current implementation

Use Python **3.12 or newer**. From an existing uv application project, add your
local Wyat checkout with the features you need:

```sh
# PyTorch CPU inference and the HTTP server
uv add /absolute/path/to/wyat --extra inference --extra server
uv run wyat --help
```

uv builds and installs a local path dependency into your application's environment;
`--extra` selects the dependency's optional features. Use `uv run` to execute in
that environment. [uv dependency guidance](https://docs.astral.sh/uv/concepts/projects/dependencies/#path),
[dependency extras](https://docs.astral.sh/uv/reference/cli/#uv-add--extra),
[running commands](https://docs.astral.sh/uv/concepts/projects/run/).

For **Apple Silicon macOS**, select MLX instead:

```sh
uv add /absolute/path/to/wyat --extra mlx --extra server
export MLX_ENABLE_TF32=0
uv run wyat --help
```

MLX defaults to the Apple GPU and needs no Torch installation. Select
`backend="mlx"` when loading. Set `MLX_ENABLE_TF32=0` before starting Python to
use the [measured MLX configuration](analysis/runtime/native.md).

The extras have distinct purposes:

| Extra | Needed for |
| --- | --- |
| `inference` | The default PyTorch loader, using CPU unless a device is explicit |
| `mlx` | Native MLX inference on Apple Silicon macOS |
| `server` | FastAPI and Uvicorn for HTTP serving, together with an inference backend |

For a custom scorer using only the Python core, omit the extras. Model weights
are acquired separately; dependency installation does not provide them. The
current loader accepts the [full local Kai artifact contract](training-contracts.md),
not arbitrary checkpoints or model names from a registry.

## Load and decide

Copy this into `decide.py` in your application:

```python
import os

import wyat

model = wyat.load(os.environ["WYAT_MODEL"])
request = wyat.Request(
    query="Choose the requested action",
    context="Cancel my booking",
    candidates=(
        wyat.Candidate(id="cancel", text="Cancel the booking"),
        wyat.Candidate(id="create", text="Create a booking"),
    ),
)
print(model.decide(request).to_dict())
```

Run it with the path to your supported local artifact:

```sh
WYAT_MODEL=/absolute/path/to/kai uv run python decide.py
```

For MLX, change the load call to
`wyat.load(os.environ["WYAT_MODEL"], backend="mlx")`. Keep this model loaded for
subsequent requests or call `model.decide_many((request, request))` for a batch.
See the [API guide](api.md) for inputs, output ordering and failure contracts.

## Serve the same model

The shortest CLI path needs no Python application file:

```sh
uv run wyat serve /absolute/path/to/kai --host 127.0.0.1 --port 8000
```

Add `--backend mlx` for Apple GPU inference. To integrate with an ASGI application,
copy this into `app.py`:

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

For MLX, pass `backend="mlx"` to `wyat.load` here as in the decision example.
Start Uvicorn with the application factory import string:

```sh
WYAT_MODEL=/absolute/path/to/kai uv run uvicorn app:app --factory --workers 1 --host 127.0.0.1 --port 8000
```

`app:app` imports the `app` function from your `app.py`; `--factory` calls it
once per worker. Importing the module does not load weights or optional server
libraries. This follows [Uvicorn's factory launch](https://uvicorn.dev/deployment/)
for a [FastAPI ASGI application](https://fastapi.tiangolo.com/deployment/manually/#run-the-server-program).
The command explicitly uses one worker and loads one model. Additional workers each load
their own model and have their own admission gate; assess their resource use
separately. Avoid development reload in a deployed process.

Send a request without needing files from the Wyat checkout:

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/decide \
  -H 'Content-Type: application/json' \
  --data '{"query":"Choose the requested action","context":"Cancel my booking","candidates":[{"id":"cancel","text":"Cancel the booking"},{"id":"create","text":"Create a booking"}]}'
```

Each app admits one active inference; an overlapping request receives 503.
Localhost is the default. Choose authentication and TLS at your deployment
boundary. Neither this example nor starting a server establishes model quality
or HTTP throughput.

## Registry installation after a matching release

As checked on **2026-10-05**, [PyPI's Wyat metadata](https://pypi.org/pypi/wyat/json)
lists version `0.0.1`, uploaded on September 28, with no inference/server/MLX
extras and a different distribution from this checkout. The API described here
must use the local dependency above until a matching release is available.

After such a release, the corresponding registry workflow is:

```sh
# Future workflow: requires a published release containing this API and extras.
uv add wyat --extra inference --extra server
```

For that release's MLX backend, select `--extra mlx --extra server` on Apple
Silicon macOS. Verify the release before replacing the local source dependency.

Installed wheels contain the runtime, not the checkout's `examples/` files.
The snippets above work from your own application directory. The source checkout
also has [model](../examples/model/main.py) and [server](../examples/server/main.py)
examples; [contributor setup](../CONTRIBUTING.md) applies when working on Wyat itself.
