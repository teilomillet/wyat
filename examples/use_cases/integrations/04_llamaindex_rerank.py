# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex: Wyat as a node postprocessor (reranker and filter).

``WyatRerank`` is a standard ``BaseNodePostprocessor``. Pass it to any query
engine to rerank what the retriever found before the LLM sees it:

    index.as_query_engine(node_postprocessors=[WyatRerank(model=model)])

Here the nodes are built by hand so the example runs without embeddings.

    uv run --with 'llama-index-core>=0.14' python 04_llamaindex_rerank.py --demo
"""

from typing import Any

from common import load_model, missing

import wyat

try:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as error:
    missing("llama-index-core>=0.14", error)

GRADES = [
    "Unrelated to the question.",
    "Related background only.",
    "Partly answers the question.",
    "Directly answers the question.",
]


class WyatRerank(BaseNodePostprocessor):
    model: Any  # wyat.Model
    top_n: int = 2

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if query_bundle is None or not nodes:
            return nodes
        decisions = self.model.decide_many(
            [
                wyat.Request.score(
                    "How well does this passage answer the question?",
                    GRADES,
                    context=f"Question: {query_bundle.query_str}\n"
                    f"Passage: {node.node.get_content()}",
                )
                for node in nodes
            ]
        )
        # Replace the retriever's similarity with Wyat's expected grade.
        graded = [
            NodeWithScore(node=n.node, score=d.expected_ordinal_value)
            for n, d in zip(nodes, decisions, strict=True)
        ]
        return sorted(graded, key=lambda n: n.score or 0.0, reverse=True)[: self.top_n]


model = load_model(__doc__)
retrieved = [
    NodeWithScore(node=TextNode(text=text), score=0.8)
    for text in [
        "The warranty covers manufacturing defects for two years.",
        "Our offices are closed on public holidays.",
        "Warranty claims need the receipt and a photo of the defect.",
    ]
]
query = "What does the warranty cover and how do I claim?"
for node in WyatRerank(model=model).postprocess_nodes(retrieved, query_str=query):
    print(f"{node.score:.2f}  {node.node.get_content()}")
