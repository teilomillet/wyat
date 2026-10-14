"""One loaded model, one active inference, bounded HTTP request bodies and queue."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import uuid
from collections import deque
from numbers import Real
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .core import (
    _MODEL_INPUT_REASONS,
    Model,
    ModelInputError,
    ModelOutputError,
    Request,
    request_from_dict,
)
from .wire import decode_json

logger = logging.getLogger(__name__)
access_logger = logging.getLogger(__name__ + ".access")

NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]

# Accepted request IDs are echoed verbatim; unsafe values are replaced so the
# header, and therefore every access-log line, stays free of arbitrary text.
_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._-]{1,64}\Z")
_REQUEST_ID_HEADER = {
    "X-Request-ID": {
        "description": "Identifier for this request. A supplied value is echoed when "
        "it matches [A-Za-z0-9._-]{1,64}; otherwise the server generates one. The "
        "same value appears in the access log, which never contains request text.",
        "schema": {"type": "string"},
    }
}


class CandidateBody(BaseModel):
    """Application-owned ID and the complete description read by the model."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: NonemptyText
    text: NonemptyText


class DecisionBody(BaseModel):
    """HTTP input schema; the shared core parser owns semantic validation."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"task": {"const": "noul"}},
                        "required": ["task"],
                    },
                    "else": {
                        "required": ["candidates"],
                        "properties": {"candidates": {"minItems": 1}},
                    },
                },
                {
                    "if": {
                        "properties": {"task": {"const": "score"}},
                        "required": ["task"],
                    },
                    "else": {
                        "properties": {
                            "candidates": {"items": CandidateBody.model_json_schema()}
                        }
                    },
                },
            ],
        },
    )
    query: NonemptyText = Field(description="Instruction for this decision.")
    context: str = Field(default="", description="Supporting information.")
    task: NonemptyText = Field(
        default="choice",
        description="choice, score, noul, or a model-supported custom task. "
        "The loaded model determines which tasks are supported.",
    )
    candidates: list[CandidateBody | NonemptyText] = Field(
        default_factory=list,
        description="Ordered candidates with unique IDs. Only score accepts text "
        "levels, whose IDs are their zero-based positions. For noul, omitted or "
        "empty candidates default to false/False. and true/True. "
        "Candidate count and input-size limits are model-specific.",
    )


class DecisionResponse(BaseModel):
    """One decision, with scores and probabilities in submitted candidate order."""

    model_config = ConfigDict(extra="forbid")
    model_id: NonemptyText
    selected: NonemptyText = Field(description="Winning candidate ID.")
    ranking: list[NonemptyText] = Field(
        min_length=1,
        description="All candidate IDs in descending score order; ties retain "
        "submitted order.",
        json_schema_extra={"uniqueItems": True},
    )
    scores: list[Annotated[float, Field(allow_inf_nan=False)]] = Field(
        min_length=1, description="Finite raw logits, one per candidate."
    )
    probabilities: list[Annotated[float, Field(ge=0, le=1)]] = Field(
        min_length=1,
        description="Softmax probabilities, one per candidate, summing to one. "
        "These are not a calibration guarantee.",
    )


class HTTPError(BaseModel):
    """Stable v1 error envelope; validation errors also use a string detail."""

    detail: str


class HealthResponse(BaseModel):
    """Liveness: the process is up and one model is loaded."""

    model_config = ConfigDict(extra="forbid")
    status: NonemptyText
    model_id: NonemptyText


class ReadinessResponse(BaseModel):
    """Readiness identity for the single loaded model; carries no request data."""

    model_config = ConfigDict(extra="forbid")
    status: NonemptyText = Field(description="Always 'ready' when the model is loaded.")
    model_id: NonemptyText = Field(
        description="Byte identity of the loaded artifact. For native artifacts this "
        "is a content digest, so it is stable across hosts and restarts."
    )
    model_type: NonemptyText = Field(
        description="Identifier of the loaded model implementation."
    )
    temperature: Annotated[float, Field(gt=0, allow_inf_nan=False)] = Field(
        description="Softmax temperature applied by this server."
    )
    max_body_bytes: int = Field(
        gt=0, description="Maximum accepted HTTP request body size in bytes."
    )
    max_queue_size: int = Field(
        ge=0,
        description="Maximum queued requests while inference is active; "
        "0 disables queueing and rejects overlap with 503.",
    )
    queue_timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)] = Field(
        description="Maximum seconds a queued request waits before a 503."
    )


_REQUEST_EXAMPLES = {
    "choice": {
        "value": {
            "query": "Choose the requested action",
            "context": "Cancel my booking",
            "candidates": [
                {"id": "cancel", "text": "Cancel the booking"},
                {"id": "create", "text": "Create a booking"},
            ],
        }
    },
    "score": {
        "value": {
            "query": "Rate the answer",
            "task": "score",
            "candidates": ["Incorrect", "Partially correct", "Correct"],
        }
    },
    "noul": {
        "value": {
            "query": "Is the claim supported?",
            "task": "noul",
            "context": "Evidence for the claim",
        }
    },
}


class _QueuedDecision:
    __slots__ = ("request", "loop", "event", "state", "result", "error")

    def __init__(
        self,
        request: Request,
        loop: asyncio.AbstractEventLoop,
        *,
        create_event: bool = True,
    ) -> None:
        self.request = request
        self.loop = loop
        self.event: asyncio.Event | None = asyncio.Event() if create_event else None
        self.state: Literal["waiting", "batched", "promoted", "cancelled"] = "waiting"
        self.result: dict[str, object] | None = None
        self.error: Exception | None = None


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ModelOutputError):
        # Backend exceptions may contain submitted text or local paths.
        logger.error("model output contract failed")
        return HTTPException(500, "model inference failed")
    if isinstance(exc, ModelInputError):
        # Keep the v1 status/body; only the validated reason crosses HTTP.
        reason = exc.reason
        headers = (
            {"X-Wyat-Rejection": reason}
            if type(reason) is str and reason in _MODEL_INPUT_REASONS
            else None
        )
        return HTTPException(
            422,
            "request is unsupported by this model",
            headers=headers,
        )
    if isinstance(exc, ValueError):
        # Backend input-policy errors must not echo submitted prompt text.
        return HTTPException(422, "request is unsupported by this model")
    logger.error("model inference failed")
    return HTTPException(500, "model inference failed")


class _AccessLogMiddleware:
    """Attach a safe request ID and emit one metadata-only access log per request.

    Implemented as a pure ASGI middleware so response streaming and the existing
    task-cancellation semantics of ``/v1/decide`` are unchanged. The log line and
    the echoed ``X-Request-ID`` never contain request text: only the method, path,
    status and a validated or generated identifier.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = MutableHeaders(scope=scope)
        supplied = headers.get("x-request-id")
        request_id = supplied if supplied and _REQUEST_ID.match(supplied) else None
        if request_id is None:
            new_id = uuid.uuid4().hex
            headers["x-request-id"] = new_id
            request_id = new_id
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            status = status_holder["status"]
            access_logger.info(
                "request method=%s path=%s status=%d id=%s duration_ms=%.3f",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status,
                request_id,
                (time.perf_counter() - started) * 1000.0,
            )


def create_app(
    model: Model,
    *,
    max_body_bytes: int = 1_048_576,
    max_queue_size: int = 0,
    queue_timeout: float = 30.0,
    log_access: bool = True,
) -> FastAPI:
    if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int):
        raise ValueError("max_body_bytes must be a positive integer")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be a positive integer")
    if (
        isinstance(max_queue_size, bool)
        or not isinstance(max_queue_size, int)
        or max_queue_size < 0
    ):
        raise ValueError("max_queue_size must be a nonnegative integer")
    if (
        isinstance(queue_timeout, bool)
        or not isinstance(queue_timeout, Real)
        or not math.isfinite(queue_timeout)
        or queue_timeout <= 0
    ):
        raise ValueError("queue_timeout must be a positive finite number")
    if not isinstance(log_access, bool):
        raise ValueError("log_access must be a boolean")
    timeout = float(queue_timeout)
    app = FastAPI(title="Wyat", version="0.0.9")
    if log_access:
        app.add_middleware(_AccessLogMiddleware)

    def openapi() -> dict[str, object]:
        if app.openapi_schema is None:
            schema = get_openapi(
                title=app.title, version=app.version, routes=app.routes
            )
            body = DecisionBody.model_json_schema(
                ref_template="#/components/schemas/{model}"
            )
            definitions = body.pop("$defs", {})
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            components.update(definitions)
            components["DecisionBody"] = body
            app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = openapi
    gate = Lock()
    state_lock = Lock()
    busy = False
    queue: deque[_QueuedDecision] = deque()

    def _release_or_promote() -> None:
        nonlocal busy
        while queue:
            candidate = queue.popleft()
            if candidate.state == "waiting":
                candidate.state = "promoted"
                assert candidate.event is not None
                candidate.loop.call_soon_threadsafe(candidate.event.set)
                return
        busy = False

    def _score_batch(items: list[_QueuedDecision]) -> None:
        try:
            decisions = model.decide_many([item.request for item in items])
        except ModelOutputError as batch_exc:
            for item in items:
                item.error = batch_exc
        except ModelInputError as batch_exc:
            if len(items) == 1:
                items[0].error = batch_exc
            else:
                mid = len(items) // 2
                _score_batch(items[:mid])
                _score_batch(items[mid:])
        except ValueError as batch_exc:
            if len(items) == 1:
                items[0].error = batch_exc
            else:
                for item in items:
                    try:
                        item.result = model.decide(item.request).to_dict()
                    except Exception as single_exc:
                        item.error = single_exc
        except Exception as batch_exc:
            for item in items:
                item.error = batch_exc
        else:
            for item, decision in zip(items, decisions, strict=True):
                item.result = decision.to_dict()

    @app.get(
        "/health",
        summary="Liveness",
        description="Reports that the process is up with one model loaded. Cheap and "
        "safe for liveness probes; use /readyz for artifact identity and limits.",
        response_model=HealthResponse,
        responses={"200": {"headers": _REQUEST_ID_HEADER}},
    )
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_id=model.model_id)

    @app.get(
        "/readyz",
        summary="Readiness and artifact identity",
        description="Reports the loaded model identity and the server's admission "
        "limits so a release can be identified by digest. Carries no request data.",
        response_model=ReadinessResponse,
        responses={"200": {"headers": _REQUEST_ID_HEADER}},
    )
    async def readyz() -> ReadinessResponse:
        return ReadinessResponse(
            status="ready",
            model_id=model.model_id,
            model_type=type(model._scorer).__name__,
            temperature=model.temperature,
            max_body_bytes=max_body_bytes,
            max_queue_size=max_queue_size,
            queue_timeout=timeout,
        )

    @app.post(
        "/v1/decide",
        summary="Decide among ordered candidates",
        description="Returns a decision from the loaded model. Candidate IDs must "
        "be unique and all text must be valid UTF-8. Scores and probabilities "
        "have the same length as the resolved candidates. Model-owned task, "
        "candidate-count and input-size limits may reject valid requests. "
        f"The request body is limited to {max_body_bytes} bytes.",
        response_model=DecisionResponse,
        responses={
            200: {
                "headers": _REQUEST_ID_HEADER,
                "content": {
                    "application/json": {
                        "example": {
                            "model_id": "example-model",
                            "selected": "cancel",
                            "ranking": ["cancel", "create"],
                            "scores": [2.0, 1.0],
                            "probabilities": [0.7310585786300049, 0.2689414213699951],
                        }
                    }
                },
            },
            **{
                status: {
                    "model": HTTPError,
                    "description": description,
                    "headers": _REQUEST_ID_HEADER,
                    "content": {"application/json": {"example": {"detail": detail}}},
                }
                for status, description, detail in (
                    (
                        400,
                        "Invalid UTF-8, malformed or ambiguous JSON.",
                        "invalid JSON body",
                    ),
                    (
                        413,
                        "Request body exceeds the configured byte limit.",
                        "request body exceeds limit",
                    ),
                    (
                        422,
                        "Request validation or model input-policy rejection.",
                        "request is unsupported by this model",
                    ),
                    (
                        500,
                        "Model inference or output-contract failure.",
                        "model inference failed",
                    ),
                    (
                        503,
                        "Admission unavailable: model busy, queue full or timed out.",
                        "model is busy; retry later",
                    ),
                )
            },
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DecisionBody"},
                        "examples": _REQUEST_EXAMPLES,
                    }
                },
            },
            "responses": {
                "422": {
                    "headers": {
                        "X-Wyat-Rejection": {
                            "description": "Optional model-owned rejection reason; "
                            "absent for request validation and generic backend errors.",
                            "schema": {
                                "type": "string",
                                "enum": sorted(_MODEL_INPUT_REASONS),
                            },
                        },
                        **_REQUEST_ID_HEADER,
                    }
                }
            },
        },
    )
    async def decide(request: HttpRequest) -> JSONResponse:
        nonlocal busy
        data = bytearray()
        try:
            async for chunk in request.stream():
                if len(data) + len(chunk) > max_body_bytes:
                    raise HTTPException(413, "request body exceeds limit")
                data.extend(chunk)
            value = decode_json(bytes(data))
        except (ValueError, UnicodeError, RecursionError, ClientDisconnect) as exc:
            raise HTTPException(400, "invalid JSON body") from exc
        try:
            decision_request = request_from_dict(value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

        if max_queue_size == 0:
            if not gate.acquire(blocking=False):
                raise HTTPException(503, "model is busy; retry later")
            handoff = Lock()
            owner: Literal["queued", "running", "released"] = "queued"

            def infer() -> dict[str, object]:
                nonlocal owner
                with handoff:
                    if owner != "queued":
                        raise HTTPException(503, "inference admission was withdrawn")
                    owner = "running"
                try:
                    return model.decide(decision_request).to_dict()
                finally:
                    # A cancelled handler cannot release a running worker's slot.
                    with handoff:
                        owner = "released"
                        gate.release()

            try:
                # Blocking inference runs off the event loop. The gate also protects
                # the scorer's tokenizer and tensor state across HTTP requests.
                result = await run_in_threadpool(infer)
                return JSONResponse(status_code=200, content=result)
            except Exception as exc:
                raise _translate_error(exc) from exc
            finally:
                # Cancellation or dispatch failure may happen before a worker starts.
                # Withdraw atomically so a late callback cannot enter inference.
                with handoff:
                    if owner == "queued":
                        owner = "released"
                        gate.release()

        loop = asyncio.get_running_loop()
        with state_lock:
            if not busy and not queue:
                busy = True
                queued_item: _QueuedDecision | None = None
            else:
                if len(queue) >= max_queue_size:
                    raise HTTPException(503, "model is busy; retry later")
                queued_item = _QueuedDecision(decision_request, loop)
                queue.append(queued_item)

        if queued_item is not None:
            assert queued_item.event is not None
            try:
                await asyncio.wait_for(queued_item.event.wait(), timeout=timeout)
            except TimeoutError as exc:
                with state_lock:
                    if queued_item.state == "waiting":
                        queued_item.state = "cancelled"
                        queue.remove(queued_item)
                    elif queued_item.state == "promoted":
                        queued_item.state = "cancelled"
                        _release_or_promote()
                raise HTTPException(503, "model is busy; retry later") from exc
            except BaseException:
                with state_lock:
                    if queued_item.state == "waiting":
                        queued_item.state = "cancelled"
                        queue.remove(queued_item)
                    elif queued_item.state == "promoted":
                        queued_item.state = "cancelled"
                        _release_or_promote()
                raise

            if queued_item.state == "batched":
                if queued_item.error is not None:
                    raise _translate_error(queued_item.error) from queued_item.error
                assert queued_item.result is not None
                return JSONResponse(status_code=200, content=queued_item.result)

        leader_owner: Literal["queued", "running", "released"] = "queued"

        def infer_queued() -> dict[str, object]:
            nonlocal leader_owner
            with state_lock:
                if leader_owner != "queued":
                    raise HTTPException(503, "inference admission was withdrawn")
                leader_owner = "running"
                initial_followers: list[_QueuedDecision] = []
                while queue:
                    follower = queue.popleft()
                    if follower.state == "waiting":
                        follower.state = "batched"
                        initial_followers.append(follower)
            try:
                leader_item = _QueuedDecision(
                    decision_request, loop, create_event=False
                )
                try:
                    _score_batch([leader_item, *initial_followers])
                finally:
                    for follower in initial_followers:
                        assert follower.event is not None
                        follower.loop.call_soon_threadsafe(follower.event.set)
                if leader_item.error is not None:
                    raise leader_item.error
                assert leader_item.result is not None
                return leader_item.result
            finally:
                with state_lock:
                    leader_owner = "released"
                    _release_or_promote()

        try:
            result = await run_in_threadpool(infer_queued)
            return JSONResponse(status_code=200, content=result)
        except Exception as exc:
            raise _translate_error(exc) from exc
        finally:
            with state_lock:
                if leader_owner == "queued":
                    leader_owner = "released"
                    _release_or_promote()

    return app
