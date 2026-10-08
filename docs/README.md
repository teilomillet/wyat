# Documentation

Start with the [project quickstart](../README.md) to install Wyat, load a local
model, make a decision, evaluate examples, or start the HTTP server.
Use the [contributor guide](../CONTRIBUTING.md) for the code map and local checks.
Use the [release guide](releasing.md) to prepare and inspect a distribution.

## Using Wyat

| Need | Guide |
| --- | --- |
| Install in an application | [Application installation](install.md) |
| Acquire and verify the supported Kai artifact | [Pinned model acquisition](model-acquisition.md) |
| Run routing, prioritization, human-review and policy-evaluation workflows | [Complete workflows](agent-workflows.md) |
| Understand the Python input and output | [Python API](api.md) |
| Distinguish raw scores, Choice, rubric Score, and yes/no Noul | [Decision task meanings](decision-tasks.md) |
| Explain a result, its calculation and the team's policy | [Explaining decisions](explaining-decisions.md) |
| Understand the ergonomic objective and API design standards | [Ergonomics standard](ergonomics.md) |
| Inspect timed agent integration evidence and limitations | [Agent study](ergonomics-agent-study.md) |
| Repeat interpretation checks and plan independent language validation | [Agent and international protocol](agent-validation.md) |
| Run the measured FP32 configuration on Apple Silicon | [Native CPU and MLX](analysis/runtime/native.md) |
| Understand Torch devices and precision | [CPU and CUDA inference](torch-inference.md) |
| Understand the MLX loader and optional runtime | [MLX inference](mlx-inference.md) |
| Evaluate through DMBench | [DMBench integration](dmbench-integration.md) |
| Compare two engines with parity gates | [Real-data measurement protocol](real-benchmark.md) |
| Export a checkpoint from training | [Training contracts](training-contracts.md) |

## Understanding the implementation

The public path is local artifact → scorer → validated logits → decision.
The CLI, evaluator and HTTP server consume the same `Model` interface.
`wyat/core.py` contains immutable request and decision values; `wyat/loading/`
selects a backend; `wyat/backends/kai/` implements the current Kai contract.

[Semantic Router inference](semantic-router-inference.md) explains the reference
contract. [Implementation evidence](implementation-evidence.md) records bounded
verification results. [Open questions](../open-question.md) identify the remaining
training and deployment decisions.

## Engine research

[The investigation index](analysis/plan.md) records hypotheses, measurements,
rejected candidates and promotion decisions. [Benchmark documentation](../benchmarks/README.md)
separates tiny mechanics fixtures from real-model assays. Experiment code lives
under `benchmarks/`; historical results and executed-source snapshots remain
there for reproducibility. A research candidate is supported for production only
when its promotion is explicit in the backend documentation.
