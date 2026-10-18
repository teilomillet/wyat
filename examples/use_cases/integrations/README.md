# Integrations

wyat is a plain Python object: build a request, get a scored decision. So it
plugs into the tools teams already use, usually through the extension point
those tools already have. Nothing here needs a special adapter package.

| # | With | wyat plays the role of | Extension point |
| --- | --- | --- | --- |
| 01 | [LangChain](01_langchain_router.py) | Router of an LCEL chain | `RunnableLambda` + `RunnableBranch` |
| 02 | [LangChain](02_langchain_rerank.py) | Reranker in a RAG chain | `BaseDocumentCompressor` |
| 03 | [LangGraph](03_langgraph_workflow.py) | The edges of a workflow graph | `add_conditional_edges` |
| 04 | [LlamaIndex](04_llamaindex_rerank.py) | Reranker for any query engine | `BaseNodePostprocessor` |
| 05 | [BM25, no framework](05_bm25_rag.py) | Filter between search and the LLM | a function |
| 06 | [MCP](06_mcp_server.py) | A decision tool any agent can call | MCP server tools |
| 07 | [pandas](07_pandas_labels.py) | Labeller for a whole table | new columns |
| 08 | [Pydantic](08_pydantic_schema.py) | Structured labels from a schema | field types |

For an HTTP service, see the [server example](../../server/main.py).

## Run

wyat installs none of these libraries. With `--with`, each one is installed
only for the run; nothing is added to your project:

```sh
cd examples/use_cases/integrations
uv run --with 'langchain-core>=1.6' python 01_langchain_router.py --demo
uv run --with 'langgraph>=1.2' python 03_langgraph_workflow.py --demo
uv run --with 'mcp>=2.3' python 06_mcp_server.py --model teilomillet/wyat-1-800m --serve
```

In a wyat checkout, `uv sync --group examples` installs all of them. A plain
`uv sync` removes them again; `uv sync --inexact` and `uv run` keep them. An
example whose library is missing or too old stops and says what to install.

The 800M named download is pending; pass its prepared local `model/` directory
until it is pinned. Use the PyTorch inference extra for that artifact.

`--demo` uses a word-overlap stand-in instead of a model (not a model; it
only makes the numbers move). The fake LLMs in 01 and 02 stand in for any
chat model; replace them with yours.

## What stays the same everywhere

- **The framework orchestrates, wyat decides.** Chains, graphs and agents call
  wyat at the points where a choice is made: which branch, which documents,
  which tool, which label.
- **Fall back when unsure.** Low-margin routes go to a default or a human
  (01, 03, 07).
- **Batch where the framework gives you a list.** Rerankers and table labelling
  score everything in one call (02, 04, 05, 07).
- **Keep stdout clean for protocols.** An MCP stdio server owns stdout, so log
  to stderr (06).
