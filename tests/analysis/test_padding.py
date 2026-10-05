"""Challenge scheduler order independently of the transformer implementation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given
from hypothesis import strategies as st

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from benchmarks.analysis.mlx.padding import Microbatches  # noqa: E402
from wyat import Candidate, Request  # noqa: E402
from wyat.backends.kai.encoding import EncodedRequest  # noqa: E402


class Oracle:
    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def encode_request(self, request: Request) -> EncodedRequest:
        return EncodedRequest(tuple(range(len(request.query))), (0, 1), 1)

    def score_many(self, requests: Sequence[Request]) -> list[list[float]]:
        self.batches.append(tuple(request.query for request in requests))
        return [[float(sum(map(ord, request.query))), 0.0] for request in requests]


@given(
    queries=st.lists(st.text(alphabet="abc", min_size=1, max_size=12), max_size=20),
    size=st.integers(1, 8),
    sort=st.booleans(),
)
def test_scheduler_preserves_duplicates_cardinality_and_original_order(
    queries: list[str], size: int, sort: bool
) -> None:
    requests = [
        Request(query=query, candidates=(Candidate("a", "yes"), Candidate("b", "no")))
        for query in queries
    ]
    oracle = Oracle()
    actual = Microbatches(oracle, size, sort=sort).score_many(requests)
    assert actual == [[float(sum(map(ord, query))), 0.0] for query in queries]
    observed = [query for batch in oracle.batches for query in batch]
    assert observed == (sorted(queries, key=len) if sort else queries)
    assert all(0 < len(batch) <= size for batch in oracle.batches)
    assert not queries or len(oracle.batches) == (len(queries) + size - 1) // size


@pytest.mark.parametrize("size", [0, -1, True])
def test_scheduler_rejects_invalid_batch_size(size: int) -> None:
    with pytest.raises(ValueError):
        Microbatches(Oracle(), size, sort=True)
