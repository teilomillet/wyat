# Contributing to Wyat

Use Python 3.12 or newer. From the public source checkout, install the development
environment and run the core example:

```sh
uv sync
uv run python examples/scorer/main.py
uv run wyat --help
```

The example uses fixed logits and requires no model weights. Install optional
runtimes for the connected checks:

```sh
uv sync --extra inference --extra server --group examples
uv run --no-sync python scripts/checks/main.py
```

MLX checks require `--extra mlx` on Apple Silicon macOS. Pass `--require-mlx` when
MLX must be available, and `--require-metal` when Apple GPU execution is required.
Read the reported skips; an unavailable capability is not a passing check. Tests use small local
fixtures and do not download model weights.

## Code map

| Location | Responsibility |
| --- | --- |
| `wyat/core.py` | Request validation, scorer contract, ranking and softmax |
| `wyat/loading/` | Explicit local backend selection |
| `wyat/backends/wyat/` | Artifact admission, rendering and Wyat runtimes |
| `wyat/wire.py`, `wyat/cli.py` | JSON decoding and command orchestration |
| `wyat/evaluation.py` | Failure-inclusive labeled evaluation |
| `wyat/server.py` | HTTP admission, body limits and worker lifetime |
| `examples/` | Runnable public-interface examples |
| `tests/` | Public contract and regression checks |
| `scripts/checks/`, `scripts/package/` | Quality, export and distribution checks |
| `docs/` | User guides and maintainer instructions |

Keep optional libraries at their runtime boundary so importing `wyat` remains
lightweight. Preserve application IDs, score ordering, explicit loading, and
failure outcomes. For a public API change, start with the caller's complete task
and follow the [ergonomics standard](docs/ergonomics.md).

## Making a change

Identify the requested behavior and preserved contracts before editing. Keep
changes focused, run proportionate checks, and retain failures or unavailable
checks in the result. Do not infer trained-model quality or performance from
fixture tests. New dependencies and public interfaces need a present purpose.

The shared check command runs types, formatting and public regressions. It writes
ignored JUnit and coverage outputs under `artifacts/`. Coverage exposes untested
paths; it does not measure model quality.

Build and inspect both distributions from the public source checkout:

```sh
uv build --no-build-isolation
uv run --no-sync python scripts/package/check.py dist
```

The checker validates metadata and contents, rebuilds the source distribution,
and exercises the installed core and CLI without optional inference runtimes.
Building does not publish a release. The wheel contains the approved runtime,
type marker and attribution. The source distribution also contains approved
public tests, examples, documentation and check tooling.

## Maintaining the public boundary

The maintainer workspace can contain private research, datasets, model artifacts
and execution history. Public release content is selected by an explicit file
allowlist; those private files are excluded from the wheel, source distribution
and public source snapshot. Preserve private work in its private location.

To prepare a public source snapshot from the maintainer workspace, use a fresh
destination:

```sh
.venv/bin/python scripts/package/export.py artifacts/public-release/source
```

The export must contain only approved files and must not inherit the workspace's
`.git` directory or history. Build and check the exported source before choosing
an audience or publishing anything. See the [release guide](docs/releasing.md).
