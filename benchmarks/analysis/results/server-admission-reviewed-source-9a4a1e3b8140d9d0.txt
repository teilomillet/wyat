"""One loaded model, one active inference, bounded HTTP request bodies."""

from __future__ import annotations

import logging
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request as HttpRequest
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from .core import Model, ModelOutputError, request_from_dict
from .wire import decode_json

logger = logging.getLogger(__name__)


def create_app(model: Model, *, max_body_bytes: int = 1_048_576) -> FastAPI:
    if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int):
        raise ValueError("max_body_bytes must be a positive integer")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be a positive integer")
    app = FastAPI(title="Wyat", version="0.0.1")
    gate = Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model_id": model.model_id}

    @app.post("/v1/decide")
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
