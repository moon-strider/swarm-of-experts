"""A text/tool Chat Completions subset with bounded request ownership."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .backend import Backend, SwarmError
from .config import Settings, load_settings
from .models import ChatRequest
from .runtime import Runtime


def error_body(error: SwarmError) -> dict:
    return {
        "error": {
            "message": error.message,
            "type": "invalid_request_error" if error.status < 500 else "server_error",
            "param": None,
            "code": error.code,
        }
    }


class RequestBoundary:
    """Authenticate before reading a bounded body, including chunked requests."""

    def __init__(self, app, settings: Settings):
        self.app, self.settings = app, settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        secret = self.settings.api_key
        if (
            scope["path"] != "/health"
            and secret
            and not hmac.compare_digest(
                headers.get(b"authorization", b""), ("Bearer " + secret.get_secret_value()).encode()
            )
        ):
            return await JSONResponse(
                error_body(SwarmError("unauthorized", "A valid bearer token is required", 401)), 401
            )(scope, receive, send)
        data = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            data.extend(event.get("body", b""))
            if len(data) > self.settings.limits.request_bytes:
                return await JSONResponse(
                    error_body(
                        SwarmError("request_too_large", "Request body exceeds byte limit", 413)
                    ),
                    413,
                )(scope, receive, send)
            if not event.get("more_body", False):
                break
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(data), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class OwnedStream(StreamingResponse):
    def __init__(self, content, lease):
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self.lease = lease
        self.owned_iterator = content

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.owned_iterator.aclose()
            await self.lease.__aexit__(None, None, None)


async def until_disconnect(request: Request, operation):
    async def disconnected():
        while (await request.receive())["type"] != "http.disconnect":
            pass

    work = asyncio.create_task(operation)
    monitor = asyncio.create_task(disconnected())
    try:
        await asyncio.wait([work, monitor], return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            return await work
        raise SwarmError("client_disconnected", "Client disconnected", 499)
    finally:
        work.cancel()
        monitor.cancel()
        await asyncio.gather(work, monitor, return_exceptions=True)


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    config = settings or load_settings()
    runtime = Runtime(config, backend)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await runtime.backend.close()

    app = FastAPI(title="Swarm of Experts", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime
    app.add_middleware(RequestBoundary, settings=config)

    @app.exception_handler(SwarmError)
    async def failed(request, error):
        return JSONResponse(error_body(error), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, error):
        fields = [".".join(map(str, e["loc"])) for e in error.errors()[:5]]
        return JSONResponse(
            error_body(
                SwarmError(
                    "validation_error", "Invalid or unsupported fields: " + ", ".join(fields), 400
                )
            ),
            400,
        )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "version": __version__}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "swarm-of-experts"}
                for name in config.swarms
            ],
        }

    @app.get("/v1/sessions/stats")
    async def stats():
        return {
            "active_sessions": 0,
            "active_requests": runtime.active,
            "persistent_sessions": False,
        }

    @app.post("/v1/sessions/cleanup")
    async def cleanup():
        return {"cleaned_sessions": 0, "persistent_sessions": False}

    @app.post("/v1/chat/completions")
    async def chat(payload: ChatRequest, request: Request):
        runtime.validate(payload)
        identity = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "created": int(time.time()),
            "model": payload.model,
        }
        lease = runtime.admission()
        await lease.__aenter__()
        if payload.stream:

            async def stream():
                try:
                    async with asyncio.timeout(config.limits.timeout_seconds):
                        async with aclosing(runtime.stream(payload)) as events:
                            async for event in events:
                                if "trace" in event:
                                    item = {
                                        **identity,
                                        "object": "chat.completion.chunk",
                                        "choices": [],
                                        "swarm": event["trace"],
                                    }
                                    if (
                                        payload.stream_options
                                        and payload.stream_options.include_usage
                                    ):
                                        item["usage"] = event["usage"]
                                else:
                                    item = {
                                        **identity,
                                        "object": "chat.completion.chunk",
                                        "choices": [{"index": 0, **event}],
                                    }
                                yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
                    yield "data: [DONE]\n\n"
                except (SwarmError, TimeoutError) as exc:
                    error = (
                        exc
                        if isinstance(exc, SwarmError)
                        else SwarmError("timeout", "Request deadline exceeded", 504)
                    )
                    yield "data: " + json.dumps(error_body(error)) + "\n\n"
                    # No success terminator after a broken stream.

            return OwnedStream(stream(), lease)
        try:
            async with asyncio.timeout(config.limits.timeout_seconds):
                result, trace = await until_disconnect(request, runtime.complete(payload))
            return {
                **identity,
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": result.message, "finish_reason": result.finish_reason}
                ],
                "usage": trace.usage,
                "swarm": trace.report(),
            }
        except TimeoutError as exc:
            raise SwarmError("timeout", "Request deadline exceeded", 504) from exc
        finally:
            await lease.__aexit__(None, None, None)

    return app
