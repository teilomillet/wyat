"""Event-controlled HTTP admission and cancellation ownership regressions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from threading import Event, Lock

import anyio
import anyio.to_thread
import httpx
import pytest
from fastapi import HTTPException
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.concurrency import run_in_threadpool

import wyat.server as server
from wyat import Model, Request

BODY = {
    "query": "Question",
    "candidates": [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}],
}


class ControlledScorer:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.calls = 0
        self.lock = Lock()
        self.entered = Event()
        self.release = Event()
        self.finished = Event()

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        with self.lock:
            self.calls += 1
            block = self.block
        if block:
            self.entered.set()
            try:
                if not self.release.wait(5):
                    raise RuntimeError("Test inference release timed out")
            finally:
                self.finished.set()
        return tuple((0.0, 1.0) for _ in requests)


async def entered(event: Event) -> None:
    # This executor is independent of the deliberately saturated AnyIO pool.
    assert await asyncio.to_thread(event.wait, 3), "Controlled worker did not enter"


async def queued(
    limiter: anyio.CapacityLimiter, task: asyncio.Task[httpx.Response]
) -> None:
    async def observe() -> None:
        while limiter.statistics().tasks_waiting != 1:
            assert not task.done(), "Request ended before reserving a pool slot"
            await asyncio.sleep(0)

    await asyncio.wait_for(observe(), timeout=3)


async def settle(
    tasks: Sequence[asyncio.Task[httpx.Response] | asyncio.Task[None]],
) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=3
        )


def client(scorer: ControlledScorer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.create_app(Model(scorer, "fixture"))),
        base_url="http://asgi.invalid",
    )


def test_saturated_pool_rejects_concurrent_request_before_worker_finishes() -> None:
    async def exercise() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        scorer = ControlledScorer(block=True)
        tasks: list[asyncio.Task[httpx.Response] | asyncio.Task[None]] = []
        limiter.total_tokens = 1
        try:
            async with client(scorer) as session:
                active = asyncio.create_task(session.post("/v1/decide", json=BODY))
                tasks.append(active)
                await entered(scorer.entered)
                response = await asyncio.wait_for(
                    session.post("/v1/decide", json=BODY), timeout=3
                )
                assert response.status_code == 503
                assert scorer.calls == 1 and not scorer.finished.is_set()
                assert limiter.statistics().tasks_waiting == 0
                scorer.release.set()
                assert (await asyncio.wait_for(active, 3)).status_code == 200
                scorer.block = False
                assert (await session.post("/v1/decide", json=BODY)).status_code == 200
        finally:
            scorer.release.set()
            try:
                await settle(tasks)
                if scorer.entered.is_set():
                    await entered(scorer.finished)
            finally:
                limiter.total_tokens = original_tokens

    asyncio.run(exercise())


def test_cancelled_pool_wait_releases_reservation_before_next_request() -> None:
    async def exercise() -> None:
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        blocker_entered, blocker_release, blocker_finished = Event(), Event(), Event()
        scorer = ControlledScorer()
        tasks: list[asyncio.Task[httpx.Response] | asyncio.Task[None]] = []

        def block_pool() -> None:
            blocker_entered.set()
            try:
                if not blocker_release.wait(5):
                    raise RuntimeError("Test unrelated pool release timed out")
            finally:
                blocker_finished.set()

        limiter.total_tokens = 1
        try:
            blocker = asyncio.create_task(anyio.to_thread.run_sync(block_pool))
            tasks.append(blocker)
            await entered(blocker_entered)
            async with client(scorer) as session:
                withdrawn = asyncio.create_task(session.post("/v1/decide", json=BODY))
                tasks.append(withdrawn)
                await queued(limiter, withdrawn)
                assert scorer.calls == 0
                # The accepted reservation itself must reject another request,
                # even though no model worker has started yet.
                busy = await asyncio.wait_for(
                    session.post("/v1/decide", json=BODY), timeout=3
                )
                assert busy.status_code == 503
                withdrawn.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await withdrawn
                following = asyncio.create_task(session.post("/v1/decide", json=BODY))
                tasks.append(following)
                await queued(limiter, following)
                assert scorer.calls == 0
                blocker_release.set()
                assert (await asyncio.wait_for(following, 3)).status_code == 200
                assert scorer.calls == 1
        finally:
            blocker_release.set()
            try:
                await settle(tasks)
                if blocker_entered.is_set():
                    await entered(blocker_finished)
            finally:
                limiter.total_tokens = original_tokens

    asyncio.run(exercise())


@settings(max_examples=12, deadline=None, derandomize=True)
@given(
    invoke_before=st.booleans(),
    cancellations=st.integers(1, 3),
    invocations=st.integers(1, 3),
)
def test_late_withdrawn_callback_never_scores_or_releases_next_worker(
    invoke_before: bool, cancellations: int, invocations: int
) -> None:
    original_dispatch = run_in_threadpool
    callbacks: list[Callable[[], dict[str, object]]] = []
    captured = asyncio.Event()
    scorer = ControlledScorer()

    async def dispatch(infer: Callable[[], dict[str, object]]) -> dict[str, object]:
        if not callbacks:
            callbacks.append(infer)
            captured.set()
            await asyncio.Event().wait()
            raise AssertionError("Withdrawn dispatch unexpectedly resumed")
        return await original_dispatch(infer)

    async def exercise() -> None:
        tasks: list[asyncio.Task[httpx.Response] | asyncio.Task[None]] = []
        try:
            async with client(scorer) as session:
                withdrawn = asyncio.create_task(session.post("/v1/decide", json=BODY))
                tasks.append(withdrawn)
                await asyncio.wait_for(captured.wait(), 3)
                for _ in range(cancellations):
                    withdrawn.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await withdrawn
                # Force callbacks the pool would normally discard, including
                # when the model gate is free and no pool check can protect us.
                if invoke_before:
                    for _ in range(invocations):
                        with pytest.raises(HTTPException) as stale:
                            callbacks[0]()
                        assert stale.value.status_code == 503
                assert scorer.calls == 0
                scorer.block = True
                active = asyncio.create_task(session.post("/v1/decide", json=BODY))
                tasks.append(active)
                await entered(scorer.entered)
                for _ in range(invocations):
                    with pytest.raises(HTTPException) as repeated:
                        callbacks[0]()
                    assert repeated.value.status_code == 503
                busy = await asyncio.wait_for(
                    session.post("/v1/decide", json=BODY), timeout=3
                )
                assert busy.status_code == 503 and scorer.calls == 1
                assert not scorer.finished.is_set()
                scorer.release.set()
                assert (await asyncio.wait_for(active, 3)).status_code == 200
                scorer.block = False
                assert (await session.post("/v1/decide", json=BODY)).status_code == 200
                assert scorer.calls == 2
        finally:
            scorer.release.set()
            await settle(tasks)
            if scorer.entered.is_set():
                await entered(scorer.finished)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(server, "run_in_threadpool", dispatch)
        asyncio.run(exercise())


def test_submission_failure_releases_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dispatch = run_in_threadpool
    submitted = 0
    scorer = ControlledScorer()

    async def dispatch(infer: Callable[[], dict[str, object]]) -> dict[str, object]:
        nonlocal submitted
        submitted += 1
        if submitted == 1:
            raise RuntimeError("private submission failure")
        return await original_dispatch(infer)

    monkeypatch.setattr(server, "run_in_threadpool", dispatch)

    async def exercise() -> None:
        async with client(scorer) as session:
            response = await session.post("/v1/decide", json=BODY)
            assert response.status_code == 500 and "private" not in response.text
            assert scorer.calls == 0
            assert (await session.post("/v1/decide", json=BODY)).status_code == 200
            assert scorer.calls == 1

    asyncio.run(exercise())


def test_scorer_failure_releases_worker_owned_reservation() -> None:
    class FailingOnce(ControlledScorer):
        def score_many(
            self, requests: Sequence[Request]
        ) -> tuple[tuple[float, ...], ...]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("private scorer failure")
            return tuple((0.0, 1.0) for _ in requests)

    scorer = FailingOnce()

    async def exercise() -> None:
        async with client(scorer) as session:
            response = await session.post("/v1/decide", json=BODY)
            assert response.status_code == 500 and "private" not in response.text
            assert (await session.post("/v1/decide", json=BODY)).status_code == 200
            assert scorer.calls == 2

    asyncio.run(exercise())
