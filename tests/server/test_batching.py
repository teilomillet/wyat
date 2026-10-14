# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import httpx
import pytest
from starlette.concurrency import run_in_threadpool

from wyat import server
from wyat.core import Model, ModelInputError, ModelOutputError, Request, Scorer

BODY = {
    "query": "Question",
    "candidates": [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}],
}


async def accumulated_batch(
    scorer: Scorer, queries: Sequence[str], monkeypatch: pytest.MonkeyPatch
) -> list[httpx.Response]:
    """Hold the leader's dispatch until every follower has entered the queue."""
    dispatched = asyncio.Event()
    accumulated = asyncio.Event()
    follower_count = 0

    class ObservedItem(server._QueuedDecision):
        def __init__(
            self,
            request: Request,
            loop: asyncio.AbstractEventLoop,
            *,
            create_event: bool = True,
        ) -> None:
            nonlocal follower_count
            super().__init__(request, loop, create_event=create_event)
            if create_event:
                follower_count += 1
                if follower_count == len(queries) - 1:
                    accumulated.set()

    async def dispatch(infer: Callable[[], dict[str, object]]) -> dict[str, object]:
        dispatched.set()
        await asyncio.wait_for(accumulated.wait(), 3)
        return await run_in_threadpool(infer)

    monkeypatch.setattr(server, "_QueuedDecision", ObservedItem)
    monkeypatch.setattr(server, "run_in_threadpool", dispatch)
    app = server.create_app(
        Model(scorer, model_id="fixture"), max_queue_size=len(queries) - 1
    )
    tasks: list[asyncio.Task[httpx.Response]] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://fixture"
    ) as client:
        try:
            tasks.append(
                asyncio.create_task(
                    client.post("/v1/decide", json={**BODY, "query": queries[0]})
                )
            )
            await asyncio.wait_for(dispatched.wait(), 3)
            for query in queries[1:]:
                tasks.append(
                    asyncio.create_task(
                        client.post("/v1/decide", json={**BODY, "query": query})
                    )
                )
            return await asyncio.wait_for(asyncio.gather(*tasks), 5)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def test_token_budget_rejection_recovers_subbatches_without_singleton_storm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    successful: list[tuple[str, ...]] = []

    class BudgetScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            queries = tuple(request.query for request in requests)
            calls.append(queries)
            if len(requests) > 4:
                raise ModelInputError(
                    "padded batch exceeds budget", reason="input-length"
                )
            successful.append(queries)
            return [(1.0, 2.0) for _ in requests]

    queries = tuple(str(index) for index in range(17))
    responses = asyncio.run(accumulated_batch(BudgetScorer(), queries, monkeypatch))
    assert all(response.status_code == 200 for response in responses)
    assert calls[0] == queries
    assert [len(batch) for batch in successful] == [4, 4, 4, 2, 3]
    assert tuple(query for batch in successful for query in batch) == queries
    assert all(len(batch) > 1 for batch in calls)
    assert len(calls) == 9


def test_subdivision_isolates_unsupported_input_and_preserves_rejection_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    class SelectiveScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            queries = tuple(request.query for request in requests)
            calls.append(queries)
            if "unsupported" in queries:
                raise ModelInputError(
                    "private request content", reason="candidate-count"
                )
            return [(1.0, 2.0) for _ in requests]

    responses = asyncio.run(
        accumulated_batch(
            SelectiveScorer(),
            ("leader", "unsupported", "good-1", "good-2"),
            monkeypatch,
        )
    )
    assert [response.status_code for response in responses] == [200, 422, 200, 200]
    assert responses[1].headers["X-Wyat-Rejection"] == "candidate-count"
    assert "private" not in responses[1].text
    assert calls == [
        ("leader", "unsupported", "good-1", "good-2"),
        ("leader", "unsupported"),
        ("leader",),
        ("unsupported",),
        ("good-1", "good-2"),
    ]


def test_generic_value_error_keeps_singleton_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    class LegacyScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            queries = tuple(request.query for request in requests)
            calls.append(queries)
            if len(requests) > 1 or "unsupported" in queries:
                raise ValueError("private request content")
            return [(1.0, 2.0) for _ in requests]

    responses = asyncio.run(
        accumulated_batch(
            LegacyScorer(), ("leader", "unsupported", "good"), monkeypatch
        )
    )
    assert [response.status_code for response in responses] == [200, 422, 200]
    assert "X-Wyat-Rejection" not in responses[1].headers
    assert "private" not in responses[1].text
    assert calls == [
        ("leader", "unsupported", "good"),
        ("leader",),
        ("unsupported",),
        ("good",),
    ]


@pytest.mark.parametrize("error_type", [RuntimeError, ModelOutputError])
def test_batch_execution_and_output_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    calls: list[tuple[str, ...]] = []

    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            calls.append(tuple(request.query for request in requests))
            raise error_type("private model path")

    queries = ("leader", "follower-1", "follower-2")
    responses = asyncio.run(accumulated_batch(BrokenScorer(), queries, monkeypatch))
    assert calls == [queries]
    assert all(response.status_code == 500 for response in responses)
    assert all("private" not in response.text for response in responses)
