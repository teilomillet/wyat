# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""LangChain: Wyat as a document reranker inside a RAG chain.

``WyatReranker`` is a standard ``BaseDocumentCompressor``, so it drops into
anything that takes one (for example ``ContextualCompressionRetriever``). It
grades every retrieved document in one batched call and keeps the best
``top_n`` above a minimum grade. The chain: retrieve -> rerank -> prompt -> LLM.

    uv run --with 'langchain-core>=1.6' python 02_langchain_rerank.py --demo
"""

from collections.abc import Sequence
from typing import Any

from common import load_model, missing

import wyat

try:
    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import BaseDocumentCompressor, Document
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
except ImportError as error:
    missing("langchain-core>=1.6", error)

GRADES = [
    "Unrelated to the question.",
    "Related background only.",
    "Partly answers the question.",
    "Directly answers the question.",
]


class WyatReranker(BaseDocumentCompressor):
    model: Any  # wyat.Model; Any keeps pydantic from validating it
    top_n: int = 2
    min_grade: float = 1.0

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        decisions = self.model.decide_many(
            [
                wyat.Request.score(
                    "How well does this passage answer the question?",
                    GRADES,
                    context=f"Question: {query}\nPassage: {doc.page_content}",
                )
                for doc in documents
            ]
        )
        graded = [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "grade": d.expected_ordinal_value},
            )
            for doc, d in zip(documents, decisions, strict=True)
        ]
        graded.sort(key=lambda doc: doc.metadata["grade"], reverse=True)
        return [doc for doc in graded if doc.metadata["grade"] >= self.min_grade][
            : self.top_n
        ]


DOCS = [
    Document(page_content="Refunds are accepted within 30 days with a receipt."),
    Document(page_content="Our company was founded in 1998 in Lyon."),
    Document(page_content="Refunds are paid to the original card within 5 days."),
    Document(page_content="Shipping takes 3 to 5 working days."),
]


def retrieve(question: str) -> list[Document]:
    """Stands in for your vector store or search: returns everything here."""
    return DOCS


model = load_model(__doc__)
reranker = WyatReranker(model=model)


def context(question: str) -> str:
    kept = reranker.compress_documents(retrieve(question), question)
    for doc in kept:
        print(f"  kept grade={doc.metadata['grade']:.2f}: {doc.page_content}")
    return "\n".join(doc.page_content for doc in kept)


prompt = PromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
llm = FakeListLLM(responses=["You can get a refund within 30 days, paid in 5 days."])
chain = (
    {"context": RunnableLambda(context), "question": RunnablePassthrough()}
    | prompt
    | llm
)

question = "How do refunds work?"
print(f"Q: {question}")
print(f"A: {chain.invoke(question)}")
