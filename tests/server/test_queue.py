"""Event-controlled queue handoff and worker ownership regressions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from threading import Event, Lock

import httpx
import pytest
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

import wyat.server as server
from wyat import Model, Request

BODY = {
    "query": "Question",
    "candidates": [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}],
}


class WaveScorer:
    def __init__(self, waves: int = 3) -> None:
        self.entered = [Event() for _ in range(waves)]
        self.release = [Event() for _ in range(waves)]
        self.batches: list[tuple[str, ...]] = []
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def score_many(self, requests: Sequence[Request]) -> tuple[tuple[float, ...], ...]:
        with self.lock:
            index = len(self.batches)
            self.batches.append(tuple(request.query for request in requests))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if index < len(self.entered):
                self.entered[index].set()
                if not self.release[index].wait(5):
                    raise RuntimeError("Controlled wave did not receive release")
            return tuple((0.0, 1.0) for _ in requests)
        finally:
            with self.lock:
                self.active -= 1

    def release_all(self) -> None:
        for event in self.release:
            event.set()


async def entered(event: Event) -> None:
    assert await asyncio.to_thread(event.wait, 3), "Controlled worker did not enter"


async def observe(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), 3)


async def settle(tasks: Sequence[asyncio.Task[httpx.Response]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)


def track_queue(monkeypatch: pytest.MonkeyPatch) -> list[server._QueuedDecision]:
    original = server._QueuedDecision.__init__
    items: list[server._QueuedDecision] = []

    def initialize(
        self: server._QueuedDecision,
        request: Request,
        loop: asyncio.AbstractEventLoop,
        **kwargs: bool,
    ) -> None:
        original(self, request, loop, **kwargs)
        if self.event is not None:
            items.append(self)

    monkeypatch.setattr(server._QueuedDecision, "__init__", initialize)
    return items


def client(scorer: WaveScorer, capacity: int = 2) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=server.create_app(
                Model(scorer, "fixture"), max_queue_size=capacity, queue_timeout=3
            )
        ),
        base_url="http://asgi.invalid",
    )


def submit(session: httpx.AsyncClient, query: str) -> asyncio.Task[httpx.Response]:
    return asyncio.create_task(
        session.post("/v1/decide", json={**BODY, "query": query})
    )


def test_each_leader_returns_without_waiting_for_following_wave(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = track_queue(monkeypatch)
    scorer = WaveScorer()

    async def exercise() -> None:
        tasks: list[asyncio.Task[httpx.Response]] = []
        try:
            async with client(scorer) as session:
                first = submit(session, "first")
                tasks.append(first)
                await entered(scorer.entered[0])
                tasks.extend((submit(session, "second"), submit(session, "third")))
                await observe(lambda: len(items) == 2)
                scorer.release[0].set()
                await entered(scorer.entered[1])
                assert (await asyncio.wait_for(first, 3)).status_code == 200
                assert not scorer.release[1].is_set()
                tasks.extend((submit(session, "fourth"), submit(session, "fifth")))
                await observe(lambda: len(items) == 4)
                scorer.release[1].set()
                await entered(scorer.entered[2])
                for task in tasks[1:3]:
                    assert (await asyncio.wait_for(task, 3)).status_code == 200
                assert not scorer.release[2].is_set()
                scorer.release[2].set()
                assert all(
                    response.status_code == 200
                    for response in await asyncio.gather(*tasks)
                )
                assert scorer.batches == [
                    ("first",),
                    ("second", "third"),
                    ("fourth", "fifth"),
                ]
                assert scorer.max_active == 1
        finally:
            scorer.release_all()
            await settle(tasks)

    asyncio.run(exercise())


def test_waiting_cancellation_restores_queue_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = track_queue(monkeypatch)
    scorer = WaveScorer(1)

    async def exercise() -> None:
        tasks: list[asyncio.Task[httpx.Response]] = []
        try:
            async with client(scorer, capacity=1) as session:
                first = submit(session, "first")
                tasks.append(first)
                await entered(scorer.entered[0])
                withdrawn = submit(session, "withdrawn")
                tasks.append(withdrawn)
                await observe(lambda: len(items) == 1)
                assert (await session.post("/v1/decide", json=BODY)).status_code == 503
                withdrawn.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await withdrawn
                following = submit(session, "following")
                tasks.append(following)
                await observe(lambda: len(items) == 2)
                assert items[0].state == "cancelled" and items[1].state == "waiting"
                scorer.release_all()
                assert (await first).status_code == 200
                assert (await following).status_code == 200
                assert scorer.batches == [("first",), ("following",)]
        finally:
            scorer.release_all()
            await settle(tasks)

    asyncio.run(exercise())


def test_batched_cancellation_does_not_release_running_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = track_queue(monkeypatch)
    scorer = WaveScorer(2)

    async def exercise() -> None:
        tasks: list[asyncio.Task[httpx.Response]] = []
        try:
            async with client(scorer) as session:
                first = submit(session, "first")
                tasks.append(first)
                await entered(scorer.entered[0])
                promoted = submit(session, "promoted")
                batched = submit(session, "batched")
                tasks.extend((promoted, batched))
                await observe(lambda: len(items) == 2)
                scorer.release[0].set()
                await entered(scorer.entered[1])
                assert items[1].state == "batched"
                batched.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await batched
                following = submit(session, "following")
                tasks.append(following)
                await observe(lambda: len(items) == 3)
                assert items[2].state == "waiting"
                assert scorer.batches == [("first",), ("promoted", "batched")]
                assert scorer.active == 1
                scorer.release[1].set()
                for task in (first, promoted, following):
                    assert (await asyncio.wait_for(task, 3)).status_code == 200
                assert scorer.max_active == 1
                assert scorer.batches[-1] == ("following",)
        finally:
            scorer.release_all()
            await settle(tasks)

    asyncio.run(exercise())


def test_cancelled_promoted_dispatch_cannot_score_or_release_next_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = track_queue(monkeypatch)
    scorer = WaveScorer(2)
    callbacks: list[Callable[[], dict[str, object]]] = []
    captured = asyncio.Event()
    dispatches = 0

    async def dispatch(infer: Callable[[], dict[str, object]]) -> dict[str, object]:
        nonlocal dispatches
        dispatches += 1
        if dispatches == 2:
            callbacks.append(infer)
            captured.set()
            await asyncio.Event().wait()
            raise AssertionError("Withdrawn dispatch unexpectedly resumed")
        return await run_in_threadpool(infer)

    monkeypatch.setattr(server, "run_in_threadpool", dispatch)

    async def exercise() -> None:
        tasks: list[asyncio.Task[httpx.Response]] = []
        try:
            async with client(scorer) as session:
                first = submit(session, "first")
                tasks.append(first)
                await entered(scorer.entered[0])
                withdrawn = submit(session, "withdrawn")
                following = submit(session, "following")
                tasks.extend((withdrawn, following))
                await observe(lambda: len(items) == 2)
                scorer.release[0].set()
                await asyncio.wait_for(captured.wait(), 3)
                withdrawn.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await withdrawn
                await entered(scorer.entered[1])
                for _ in range(2):
                    with pytest.raises(HTTPException) as stale:
                        callbacks[0]()
                    assert stale.value.status_code == 503
                assert scorer.batches == [("first",), ("following",)]
                additional = submit(session, "additional")
                tasks.append(additional)
                await observe(lambda: len(items) == 3)
                assert items[-1].state == "waiting" and scorer.active == 1
                scorer.release[1].set()
                for task in (first, following, additional):
                    assert (await asyncio.wait_for(task, 3)).status_code == 200
                assert scorer.max_active == 1
        finally:
            scorer.release_all()
            await settle(tasks)

    asyncio.run(exercise())
