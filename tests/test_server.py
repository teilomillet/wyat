from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from wyat.core import Model, Request
from wyat.server import create_app

BODY = {
    "query": "Question",
    "candidates": [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}],
}


class Scorer:
    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        return [(1.0, 2.0) for _ in requests]


def test_health_and_decision_round_trip() -> None:
    with TestClient(create_app(Model(Scorer(), model_id="fixture"))) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == 200
    assert response.json()["selected"] == "b"
    assert response.json()["scores"] == [1.0, 2.0]
    assert response.json()["model_id"] == "fixture"


def test_readyz_reports_identity_and_is_documented() -> None:
    with TestClient(create_app(Model(Scorer(), model_id="fixture"))) as client:
        health = client.get("/health").json()
        readyz = client.get("/readyz")
        document = client.get("/openapi.json").json()
    assert readyz.status_code == 200
    body = readyz.json()
    assert body["status"] == "ready"
    assert body["model_id"] == health["model_id"] == "fixture"
    assert body["max_queue_size"] == 0 and body["temperature"] == 1.0
    assert {"/health", "/readyz", "/v1/decide"} <= set(document["paths"])
    schema = document["paths"]["/readyz"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    Draft202012Validator({**schema, "components": document["components"]}).validate(
        body
    )


def test_openapi_contract_and_examples() -> None:
    class ExampleScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [list(map(float, range(len(item.candidates)))) for item in requests]

    with TestClient(create_app(Model(ExampleScorer(), model_id="fixture"))) as client:
        document = client.get("/openapi.json").json()
        assert client.get("/docs").status_code == 200
        assert client.get("/redoc").status_code == 200
        operation = document["paths"]["/v1/decide"]["post"]
        assert operation["requestBody"]["required"] is True
        content = operation["requestBody"]["content"]["application/json"]
        schema = {**content["schema"], "components": document["components"]}
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for example in content["examples"].values():
            validator.validate(example["value"])
            response = client.post("/v1/decide", json=example["value"])
            assert response.status_code == 200
            response_schema = operation["responses"]["200"]["content"][
                "application/json"
            ]["schema"]
            Draft202012Validator(
                {**response_schema, "components": document["components"]}
            ).validate(response.json())
        assert set(operation["responses"]) == {"200", "400", "413", "422", "500", "503"}
        for status in ("400", "413", "422", "500", "503"):
            error = operation["responses"][status]["content"]["application/json"]
            Draft202012Validator(
                {**error["schema"], "components": document["components"]}
            ).validate(error["example"])
        header = operation["responses"]["422"]["headers"]["X-Wyat-Rejection"]
        assert set(header["schema"]["enum"]) == {
            "unsupported-task",
            "candidate-count",
            "input-length",
            "input-segment",
        }


@pytest.mark.parametrize(
    "body,valid",
    [
        (BODY, True),
        ({**BODY, "task": "custom"}, True),
        ({"query": "Q", "task": "noul"}, True),
        ({"query": "Q", "task": "noul", "candidates": []}, True),
        ({"query": "Q", "task": "score", "candidates": ["Level"]}, True),
        (
            {
                "query": "Q",
                "task": "score",
                "candidates": ["Level", {"id": "other", "text": "Other"}],
            },
            True,
        ),
        ({"query": "Q"}, False),
        ({"query": "Q", "candidates": ["Level"]}, False),
        ({"query": "Q", "task": "custom", "candidates": ["Level"]}, False),
        ({**BODY, "candidates": []}, False),
        ({**BODY, "candidates": None, "task": "noul"}, False),
        ({**BODY, "query": 1}, False),
        ({**BODY, "query": "  "}, False),
        ({**BODY, "expected": "a"}, False),
        ({**BODY, "candidates": [{"id": "a", "text": "First", "extra": 1}]}, False),
    ],
)
def test_openapi_request_constraints_match_core(body: object, valid: bool) -> None:
    from wyat.core import request_from_dict

    document = create_app(Model(Scorer(), model_id="fixture")).openapi()
    schema = {
        "$ref": "#/components/schemas/DecisionBody",
        "components": document["components"],
    }
    assert Draft202012Validator(schema).is_valid(body) is valid
    if valid:
        request_from_dict(body)
    else:
        with pytest.raises(ValueError):
            request_from_dict(body)


@pytest.mark.parametrize(
    "body",
    [{**BODY, "expected": "b"}, {**BODY, "query": 10}, {**BODY, "candidates": []}],
)
def test_invalid_request_returns_422(body: object) -> None:
    with TestClient(create_app(Model(Scorer(), model_id="fixture"))) as client:
        assert client.post("/v1/decide", json=body).status_code == 422


def test_malformed_json_returns_400() -> None:
    with TestClient(create_app(Model(Scorer(), model_id="fixture"))) as client:
        assert (
            client.post(
                "/v1/decide", content="{", headers={"content-type": "application/json"}
            ).status_code
            == 400
        )


def test_request_body_limit_returns_413() -> None:
    with TestClient(
        create_app(Model(Scorer(), model_id="fixture"), max_body_bytes=20)
    ) as client:
        assert client.post("/v1/decide", json=BODY).status_code == 413


def test_backend_failure_is_sanitized(caplog: pytest.LogCaptureFixture) -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            raise RuntimeError("private checkpoint path /secret/model")

    with TestClient(create_app(Model(BrokenScorer(), model_id="fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == 500
    assert "secret" not in response.text
    records = [record for record in caplog.records if record.name == "wyat.server"]
    assert len(records) == 1
    assert records[0].getMessage() == "model inference failed"
    assert records[0].exc_info is None
    assert "/secret/model" not in caplog.text


def test_invalid_backend_output_is_a_server_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(float("nan"), 1.0) for _ in requests]

    with TestClient(create_app(Model(BrokenScorer(), model_id="fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == 500
    records = [record for record in caplog.records if record.name == "wyat.server"]
    assert len(records) == 1
    assert records[0].getMessage() == "model output contract failed"
    assert records[0].exc_info is None


def test_concurrent_request_returns_busy_until_active_inference_finishes() -> None:
    entered, release = Event(), Event()

    class BlockingScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test synchronization timed out")
            return [(1.0, 2.0) for _ in requests]

    app = create_app(Model(BlockingScorer(), model_id="fixture"))
    with TestClient(app) as first, TestClient(app) as second:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(first.post, "/v1/decide", json=BODY)
            try:
                assert entered.wait(timeout=3), "inference did not start"
                assert second.post("/v1/decide", json=BODY).status_code == 503
            finally:
                release.set()
            assert active.result(timeout=3).status_code == 200
        assert second.post("/v1/decide", json=BODY).status_code == 200


@pytest.mark.parametrize(
    "content",
    ['{"query":NaN}', '{"query":1e999}', '{"query":"a","query":"b"}', b"\xff"],
)
def test_nonstandard_or_ambiguous_json_returns_400(content: str | bytes) -> None:
    with TestClient(create_app(Model(Scorer(), model_id="fixture"))) as client:
        response = client.post(
            "/v1/decide", content=content, headers={"content-type": "application/json"}
        )
    assert response.status_code == 400


def test_cancelled_http_task_keeps_inference_busy_until_worker_finishes() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    entered, release, finished = Event(), Event(), Event()

    class BlockingScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            entered.set()
            try:
                if not release.wait(timeout=5):
                    raise RuntimeError("test synchronization timed out")
                return [(1.0, 2.0) for _ in requests]
            finally:
                finished.set()

    async def exercise() -> None:
        app = create_app(Model(BlockingScorer(), model_id="fixture"))
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            active = asyncio.create_task(client.post("/v1/decide", json=BODY))
            try:
                assert await asyncio.to_thread(entered.wait, 3), (
                    "inference did not start"
                )
                active.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await active
                busy = await asyncio.wait_for(
                    client.post("/v1/decide", json=BODY), timeout=1
                )
                assert busy.status_code == 503
                assert not finished.is_set(), "active worker finished before busy check"
            finally:
                release.set()
                active.cancel()
                assert await asyncio.to_thread(finished.wait, 3), (
                    "worker did not finish"
                )

    asyncio.run(exercise())


def test_bounded_queue_microbatches_followers_and_rejects_overflow() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    first_entered, release_first = Event(), Event()
    batch_sizes: list[int] = []

    class BatchRecordingScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            batch_sizes.append(len(requests))
            if len(batch_sizes) == 1:
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise RuntimeError("first inference timed out")
            return [(1.0, 2.0) for _ in requests]

    async def exercise() -> None:
        app = create_app(
            Model(BatchRecordingScorer(), model_id="fixture"),
            max_queue_size=2,
            queue_timeout=5.0,
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            req1 = asyncio.create_task(client.post("/v1/decide", json=BODY))
            try:
                assert await asyncio.to_thread(first_entered.wait, 3)
                req2 = asyncio.create_task(client.post("/v1/decide", json=BODY))
                req3 = asyncio.create_task(client.post("/v1/decide", json=BODY))
                await asyncio.sleep(0.05)
                # Queue has 2 items (max_queue_size=2), so 4th request gets 503
                overflow = await client.post("/v1/decide", json=BODY)
                assert overflow.status_code == 503
                assert overflow.json()["detail"] == "model is busy; retry later"
            finally:
                release_first.set()
            r1, r2, r3 = await asyncio.gather(req1, req2, req3)
            assert (r1.status_code, r2.status_code, r3.status_code) == (200, 200, 200)
            # First call scored 1 item; the 2 queued followers were micro-batched
            assert batch_sizes == [1, 2]

    asyncio.run(exercise())


def test_queued_request_times_out_with_503_when_queue_timeout_expires() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    entered, release = Event(), Event()

    class BlockingScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("inference timed out")
            return [(1.0, 2.0) for _ in requests]

    async def exercise() -> None:
        app = create_app(
            Model(BlockingScorer(), model_id="fixture"),
            max_queue_size=1,
            queue_timeout=0.05,
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            req1 = asyncio.create_task(client.post("/v1/decide", json=BODY))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                timed_out = await client.post("/v1/decide", json=BODY)
                assert timed_out.status_code == 503
                assert timed_out.json()["detail"] == "model is busy; retry later"
            finally:
                release.set()
            assert (await req1).status_code == 200

    asyncio.run(exercise())


def test_microbatch_isolates_per_request_input_error() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    from wyat.core import ModelInputError

    first_entered, release_first = Event(), Event()

    class SelectiveScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            if any(req.query == "block" for req in requests):
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise RuntimeError("timed out")
            for req in requests:
                if req.query == "too-long":
                    raise ModelInputError(
                        "input exceeds max_length", reason="input-length"
                    )
            return [(1.0, 2.0) for _ in requests]

    async def exercise() -> None:
        app = create_app(
            Model(SelectiveScorer(), model_id="fixture"),
            max_queue_size=4,
            queue_timeout=5.0,
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            req1 = asyncio.create_task(
                client.post("/v1/decide", json={**BODY, "query": "block"})
            )
            try:
                assert await asyncio.to_thread(first_entered.wait, 3)
                bad_req = asyncio.create_task(
                    client.post("/v1/decide", json={**BODY, "query": "too-long"})
                )
                good_req = asyncio.create_task(
                    client.post("/v1/decide", json={**BODY, "query": "ok"})
                )
                await asyncio.sleep(0.05)
            finally:
                release_first.set()
            r1, r_bad, r_good = await asyncio.gather(req1, bad_req, good_req)
            assert r1.status_code == 200
            assert r_bad.status_code == 422
            assert r_good.status_code == 200

    asyncio.run(exercise())


def test_create_app_validates_queue_parameters() -> None:
    model = Model(Scorer(), model_id="fixture")
    for bad_size in (-1, True, "4"):
        with pytest.raises(ValueError, match="max_queue_size"):
            create_app(model, max_queue_size=bad_size)  # type: ignore[arg-type]
    for bad_timeout in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="queue_timeout"):
            create_app(model, queue_timeout=bad_timeout)


def test_queued_leader_returns_before_subsequent_wave_finishes() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    wave1_entered, wave1_release = Event(), Event()
    wave2_entered, wave2_release = Event(), Event()
    batches: list[tuple[str, ...]] = []

    class WaveScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            queries = tuple(req.query for req in requests)
            batches.append(queries)
            if len(batches) == 1:
                wave1_entered.set()
                if not wave1_release.wait(timeout=5):
                    raise RuntimeError("wave 1 timed out")
            elif len(batches) == 2:
                wave2_entered.set()
                if not wave2_release.wait(timeout=5):
                    raise RuntimeError("wave 2 timed out")
            return [(1.0, 2.0) for _ in requests]

    async def exercise() -> None:
        app = create_app(
            Model(WaveScorer(), model_id="fixture"),
            max_queue_size=4,
            queue_timeout=5.0,
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            req1 = asyncio.create_task(
                client.post("/v1/decide", json={**BODY, "query": "q1"})
            )
            try:
                assert await asyncio.to_thread(wave1_entered.wait, 3)
                req2 = asyncio.create_task(
                    client.post("/v1/decide", json={**BODY, "query": "q2"})
                )
                req3 = asyncio.create_task(
                    client.post("/v1/decide", json={**BODY, "query": "q3"})
                )
                await asyncio.sleep(0.05)
                wave1_release.set()
                # Leader req1 must complete with HTTP 200 while wave 2 is blocked.
                r1 = await asyncio.wait_for(req1, timeout=3)
                assert r1.status_code == 200
                assert await asyncio.to_thread(wave2_entered.wait, 3)
                assert not req2.done() and not req3.done()
            finally:
                wave1_release.set()
                wave2_release.set()
            r2, r3 = await asyncio.gather(req2, req3)
            assert (r2.status_code, r3.status_code) == (200, 200)
            assert batches == [("q1",), ("q2", "q3")]

    asyncio.run(exercise())


def test_microbatch_subdivides_on_batch_budget_error_before_singletons() -> None:
    import asyncio

    from httpx import ASGITransport, AsyncClient

    from wyat.core import ModelInputError

    first_entered, release_first = Event(), Event()
    batch_sizes: list[int] = []

    class BudgetScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            batch_sizes.append(len(requests))
            if len(batch_sizes) == 1:
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise RuntimeError("timed out")
            if len(requests) > 2:
                raise ModelInputError(
                    "complete padded question batch exceeds explicit budget",
                    reason="input-length",
                )
            return [(1.0, 2.0) for _ in requests]

    async def exercise() -> None:
        app = create_app(
            Model(BudgetScorer(), model_id="fixture"),
            max_queue_size=8,
            queue_timeout=5.0,
        )
        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://fixture"
        ) as client:
            req1 = asyncio.create_task(client.post("/v1/decide", json=BODY))
            try:
                assert await asyncio.to_thread(first_entered.wait, 3)
                followers = [
                    asyncio.create_task(client.post("/v1/decide", json=BODY))
                    for _ in range(4)
                ]
                await asyncio.sleep(0.05)
            finally:
                release_first.set()
            responses = await asyncio.gather(req1, *followers)
            assert all(resp.status_code == 200 for resp in responses)
            # Wave 2 attempts [4] and subdivides into [2] and [2] (not 4 singletons).
            assert batch_sizes == [1, 4, 2, 2]

    asyncio.run(exercise())
