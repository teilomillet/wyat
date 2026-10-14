# Documentation

Start with [your first decision](quickstart.md), then use the
[installation guide](install.md) for a checked local wheel or a matching registry
release. Model weights are acquired separately.

| Need | Guide |
| --- | --- |
| Install and make your first decision | [Quickstart demo](quickstart.md) |
| Watch Wyat choose and a coding agent act | [Step-by-step embedded demo](embedded.md) |
| Install Wyat and serve it from your application | [Installation](install.md) |
| Download and verify a supported Wyat model artifact | [Model acquisition](model-acquisition.md) |
| Understand requests, results, batches, evaluation and errors | [Python API](api.md) |
| Integrate routing, prioritization, review and recovery | [Application workflows](agent-workflows.md) |
| Learn Choice, Score and Noul through short use cases | [Use cases](../examples/use_cases/README.md) |
| Build triage, agent, RAG and calibration systems | [Expert use cases](../examples/use_cases/expert/README.md) |
| Plug Wyat into LangChain, LangGraph, LlamaIndex, MCP or pandas | [Integrations](../examples/use_cases/integrations/README.md) |
| Explain a result and application-owned policy | [Explaining decisions](explaining-decisions.md) |
| Select PyTorch devices and precision | [PyTorch runtime](torch-inference.md) |
| Use native MLX on Apple Silicon | [MLX runtime](mlx-inference.md) |
| Understand public API design principles | [Ergonomics](ergonomics.md) |
| Contribute code and run public checks | [Contributing](../CONTRIBUTING.md) |
| Prepare and inspect a public release | [Releasing](releasing.md) |

The runtime path is local artifact → scorer → validated logits → decision.
Python, JSONL evaluation and HTTP use the same `Model` interface. Selecting a
candidate ID does not execute an application action. Fixture examples demonstrate
integration and do not establish trained-model quality.
