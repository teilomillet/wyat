"""One loaded model, one active inference, bounded HTTP request bodies."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .core import (
    _MODEL_INPUT_REASONS,
    Model,
    ModelInputError,
    ModelOutputError,
    request_from_dict,
)
from .wire import decode_json

logger = logging.getLogger(__name__)

NonemptyText = Annotated[str, Field(min_length=1, pattern=r"\S")]


class CandidateBody(BaseModel):
    """Application-owned ID and the complete description read by the model."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: NonemptyText
    text: NonemptyText


class DecisionBody(BaseModel):
    """HTTP input schema; the shared core parser owns semantic validation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    query: NonemptyText = Field(description="Instruction for this decision.")
    context: str = Field(default="", description="Supporting information.")
    task: NonemptyText = Field(
        default="choice",
        description="choice, score, noul, or a model-supported custom task. "
        "The loaded model determines which tasks are supported.",
    )
    candidates: list[CandidateBody] = Field(
        min_length=1,
        description="Ordered candidates with unique application-owned IDs. "
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
    "custom-task": {
        "value": {
            "query": "Rate the answer",
            "task": "score",
            "candidates": [
                {"id": "0", "text": "Incorrect"},
                {"id": "1", "text": "Correct"},
            ],
        }
    },
}


def create_app(model: Model, *, max_body_bytes: int = 1_048_576) -> FastAPI:
    if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int):
        raise ValueError("max_body_bytes must be a positive integer")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be a positive integer")
    app = FastAPI(title="Wyat", version="0.0.1")

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

    app.openapi = openapi  # type: ignore[method-assign]
    gate = Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_id": model.model_id}

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
                }
            },
            **{
                status: {
                    "model": HTTPError,
                    "description": description,
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
                        "Model busy or inference admission withdrawn.",
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
                        }
                    }
                }
            },
        },
    )
    async def decide(request: HttpRequest) -> dict[str, object]:
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
            return await run_in_threadpool(infer)
        except HTTPException:
            raise
        except ModelOutputError as exc:
            logger.exception("model output contract failed")
            raise HTTPException(500, "model inference failed") from exc
        except ModelInputError as exc:
            # Keep the v1 status/body; only the validated reason crosses HTTP.
            reason = exc.reason
            headers = (
                {"X-Wyat-Rejection": reason}
                if type(reason) is str and reason in _MODEL_INPUT_REASONS
                else None
            )
            raise HTTPException(
                422,
                "request is unsupported by this model",
                headers=headers,
            ) from exc
        except ValueError as exc:
            # Backend input-policy errors must not echo submitted prompt text.
            raise HTTPException(422, "request is unsupported by this model") from exc
        except Exception as exc:
            logger.exception("model inference failed")
            raise HTTPException(500, "model inference failed") from exc
        finally:
            # Cancellation or dispatch failure may happen before a worker starts.
            # Withdraw atomically so a late callback cannot enter inference.
            with handoff:
                if owner == "queued":
                    owner = "released"
                    gate.release()

    return app
