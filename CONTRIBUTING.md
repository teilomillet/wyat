# Working on Wyat

Use Python 3.12 or newer and run commands from a source checkout. `uv sync`
installs the development group by default. The core example needs no model
weights or optional inference libraries:

```sh
uv sync
uv run python examples/scorer/main.py
uv run wyat --help
```

The example returns fixed logits to show the public interface; it has no learned
decision quality. For backend and HTTP checks, install the existing extras:

```sh
uv sync --extra inference --extra server
uv run --no-sync python scripts/checks/main.py
```

Tests use small seeded local fixtures rather than downloading released weights.
MLX checks require `--extra mlx` on Apple Silicon; GPU checks also require usable
Metal access. DMBench checks skip when that optional project is absent. Read the
reported skips: unavailable checks establish no backend or integration result.
The mypy flags isolate optional third-party imports; maintained Wyat code keeps
strict checking from `pyproject.toml`.
Examples and scripts are checked separately with directory-based module names
because they and the checkout entry point use `main.py`. The main check retains the tests'
existing module mapping.

The same command runs in CI. It stops on the first failed check and gives each
check a three-minute cap, terminating only its owned subprocesses. JUnit and
branch-coverage reports go to ignored `artifacts/`. Coverage makes untested paths
visible; it is not a model-quality or performance gate. For native Apple Silicon
checks, install `--extra mlx` and pass `--require-mlx`; add `--require-metal` when
GPU execution is required. An unavailable backend remains an explicit skip in
the test reports.

To check DMBench against its actual source, set both paths explicitly:

```sh
PYTHONPATH=/path/to/dmbench/src uv run pytest -q tests/test_dmbench.py tests/interop/dmbench
MYPYPATH=/path/to/dmbench/src uv run mypy --ignore-missing-imports \
  --follow-imports=silent wyat/interop tests/test_dmbench.py tests/interop/dmbench
```

## Code map

| Location | Responsibility |
| --- | --- |
| `wyat/core.py` | Request validation, scorer contract, ranking and softmax |
| `wyat/loading/` | Explicit local backend selection |
| `wyat/backends/kai/` | Artifact admission, rendering and Kai runtime implementations |
| `wyat/wire.py`, `wyat/cli.py` | JSON decoding and command orchestration |
| `wyat/evaluation.py`, `wyat/benchmark.py` | Quality reports and paired engine measurements |
| `wyat/server.py` | HTTP admission, body limits and worker lifetime |
| `wyat/interop/` | Optional DMBench projections |
| `examples/` | Small runnable public-interface examples |
| `scripts/checks/`, `scripts/package/` | Shared quality and distribution checks |
| `tests/` | Contract, regression and Hypothesis checks |
| `benchmarks/` | Research runners, fixtures and immutable measurement history |
| `docs/` | User guides, contracts and experiment explanations |

Keep domain transformations separate from I/O when that makes ownership clearer.
Use precise types, explicit failures and straightforward control flow. New files
belong in folders named for their role; avoid speculative abstractions. Optional
libraries load at their runtime boundary so importing `wyat` remains lightweight.

## Making a change

Identify the observable behavior and preserved contracts before editing. Run
focused checks first, then connected regressions. Runtime optimizations need
real-model measurements as well as fixture tests; preserve numerical and complete
ranking gates. Follow the [measurement protocol](docs/real-benchmark.md) and record
the baseline, data, device, settings, raw repetitions and rejected hypotheses.

Write new experiment outputs to a fresh location, such as a new directory under
`/tmp`. Existing result files and source archives describe past executions and
must remain unchanged. Their embedded source text is historical evidence, not a
second maintained implementation. Model weights and raw capture tensors stay
outside distributable code.

Build with the locked development backend, then verify both distributions:

```sh
uv build --no-build-isolation
uv run --no-sync python scripts/package/check.py dist
```

The checker validates metadata and contents, rebuilds the source distribution,
and exercises the built wheel without optional runtimes. Building does not
publish a release. See [release preparation](docs/releasing.md) and the
[documentation index](docs/README.md) for the current supported backends and
research boundaries.

The wheel contains the runtime package, type marker and attribution. The source
distribution also retains research tooling, tests and their required evidence;
it is intentionally larger. Those files are not imported by the installed runtime.
