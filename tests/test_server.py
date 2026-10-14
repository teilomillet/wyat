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
        ({"query": "Q", "task": "noul"}, False),
        ({"query": "Q", "task": "noul", "candidates": []}, False),
        ({"query": "Q", "task": "score", "candidates": ["Level"]}, False),
        (
            {
                "query": "Q",
                "task": "score",
                "candidates": ["Level", {"id": "other", "text": "Other"}],
            },
            False,
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


def test_backend_failure_is_sanitized() -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            raise RuntimeError("private checkpoint path /secret/model")

    with TestClient(create_app(Model(BrokenScorer(), model_id="fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == 500
    assert "secret" not in response.text


def test_invalid_backend_output_is_a_server_failure() -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            return [(float("nan"), 1.0) for _ in requests]

    with TestClient(create_app(Model(BrokenScorer(), model_id="fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == 500


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
