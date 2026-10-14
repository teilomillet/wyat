# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0

"""Recovery provenance and safe reasons across Python, CLI and HTTP."""

import io
import json
import sys
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from wyat import Candidate, Model, ModelInputError, ModelOutputError, Request, cli
from wyat.backends.wyat.encoding import WyatEncoder
from wyat.server import create_app


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode("utf-8"))


class EncodingScorer:
    def __init__(self) -> None:
        self.encoder = WyatEncoder(CharacterTokenizer(), max_length=256, max_options=3)

    def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
        for request in requests:
            self.encoder.encode_request(request)
        return [(0.0, 1.0) for _ in requests]


BODY = {
    "query": "Choose",
    "candidates": [{"id": "a", "text": "A"}, {"id": "b", "text": "B"}],
}


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"task": "score"}, "unsupported-task"),
        ({"candidates": [{"id": "a", "text": "A"}]}, "candidate-count"),
        ({"query": "x" * 500}, "input-segment"),
        ({"query": "x" * 100}, "input-length"),
    ],
)
def test_owned_model_rejections_keep_v1_body_and_supply_safe_reason(
    changes: dict[str, object], reason: str
) -> None:
    with TestClient(create_app(Model(EncodingScorer(), "fixture"))) as client:
        response = client.post("/v1/decide", json={**BODY, **changes})
        assert response.status_code == 422
        assert response.json() == {"detail": "request is unsupported by this model"}
        assert response.headers["X-Wyat-Rejection"] == reason
        # Rejection releases admission; the next independent request can run.
        assert client.post("/v1/decide", json=BODY).status_code == 200


@pytest.mark.parametrize(
    "error,status,reason",
    [
        (ValueError("sensitive-request-sentinel"), 422, None),
        (RuntimeError("sensitive-request-sentinel"), 500, None),
        (ModelOutputError("sensitive-request-sentinel"), 500, None),
        (
            ModelInputError("sensitive-request-sentinel", reason="candidate-count"),
            422,
            "candidate-count",
        ),
    ],
)
def test_http_never_copies_arbitrary_backend_messages_to_recovery_header(
    error: Exception, status: int, reason: str | None
) -> None:
    class BrokenScorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            raise error

    with TestClient(create_app(Model(BrokenScorer(), "fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
    assert response.status_code == status
    assert response.headers.get("X-Wyat-Rejection") == reason
    assert "sensitive-request-sentinel" not in response.text
    assert "sensitive-request-sentinel" not in str(response.headers)


@pytest.mark.parametrize("tail,location", [(False, "lines 4..5"), (True, "line 4")])
def test_cli_retains_failing_batch_line_provenance_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tail: bool,
    location: str,
) -> None:
    calls: list[tuple[str, ...]] = []

    class Scorer:
        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            calls.append(tuple(request.query for request in requests))
            if any(request.query == "failure" for request in requests):
                raise RuntimeError("fixture failure")
            return [(0.0, 1.0) for _ in requests]

    monkeypatch.setattr(cli, "load", lambda *args, **kwargs: Model(Scorer(), "fixture"))
    rows = [
        json.dumps(BODY),
        "",
        json.dumps(BODY),
        json.dumps({**BODY, "query": "failure"}),
    ]
    if not tail:
        rows.append(json.dumps(BODY))
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(rows)))
    assert cli.main(["decide", "fixture", "--batch-size", "2"]) == 1
    output = capsys.readouterr()
    assert len(output.out.splitlines()) == 2
    assert all(json.loads(line)["selected"] == "b" for line in output.out.splitlines())
    assert f"input {location}" in output.err
    assert "no decisions emitted for this batch" in output.err
    assert "fixture failure" in output.err
    assert len(calls) == 2  # One successful batch and one failure; no splitting/retry.


def test_model_input_reasons_are_allowlisted() -> None:
    with pytest.raises(ValueError, match="unknown model input rejection"):
        ModelInputError("private", reason="private-request-content")  # type: ignore[arg-type]


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("reason", ["private-request-sentinel", None, ["input-length"]])
def test_http_revalidates_mutated_or_overridden_reasons(
    override: bool, reason: object
) -> None:
    class OverriddenError(ModelInputError):
        @property
        def reason(self) -> str:
            return reason  # type: ignore[return-value]

    error = (
        OverriddenError("private-request-sentinel", reason="input-length")
        if override
        else ModelInputError("private-request-sentinel", reason="input-length")
    )
    if not override:
        error._reason = reason  # type: ignore[assignment]

    class Scorer:
        calls = 0

        def score_many(self, requests: Sequence[Request]) -> Sequence[Sequence[float]]:
            self.calls += 1
            if self.calls == 1:
                raise error
            return [(0.0, 1.0) for _ in requests]

    with TestClient(create_app(Model(Scorer(), "fixture"))) as client:
        response = client.post("/v1/decide", json=BODY)
        assert response.status_code == 422
        assert response.json() == {"detail": "request is unsupported by this model"}
        assert "X-Wyat-Rejection" not in response.headers
        assert "private-request-sentinel" not in str(response.headers)
        assert client.post("/v1/decide", json=BODY).status_code == 200


def test_model_input_reason_preserves_value_error_and_request_identity() -> None:
    request = Request("Choose", [Candidate("a", "A")])
    with pytest.raises(ModelInputError) as error:
        Model(EncodingScorer(), "fixture").decide(request)
    assert isinstance(error.value, ValueError)
    assert error.value.reason == "candidate-count"
    assert request.candidates[0].id == "a"
