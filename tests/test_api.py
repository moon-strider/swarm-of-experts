import asyncio
import json

import httpx
from openai import AsyncOpenAI

from swarm_of_experts.api import create_app
from swarm_of_experts.config import Settings

from .conftest import response
from .test_runtime import Events, sse


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_official_client_nonstream_and_stream(make_backend, settings):
    async def handler(req):
        return (
            httpx.Response(200, stream=Events(sse()))
            if json.loads(req.content)["stream"]
            else response()
        )

    app = create_app(settings, make_backend(handler))
    async with AsyncOpenAI(
        api_key="unused", base_url="http://test/v1", http_client=client_for(app), max_retries=0
    ) as client:
        result = await client.chat.completions.create(
            model="basic", messages=[{"role": "user", "content": "hi"}]
        )
        assert result.choices[0].message.content == "answer"
        assert result.usage.total_tokens == 5
        stream = await client.chat.completions.create(
            model="basic",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk async for chunk in stream]
        assert chunks[0].choices[0].delta.content == "hello"
        assert chunks[-1].usage.total_tokens == 5
    assert app.state.runtime.active == 0


async def test_auth_and_errors_are_openai_shaped(make_backend, settings):
    config = settings.model_copy(
        update={"api_key": __import__("pydantic").SecretStr("local-secret")}
    )
    app = create_app(config, make_backend(lambda req: response(), config))
    async with client_for(app) as client:
        assert (await client.get("/health")).status_code == 200
        denied = await client.get("/v1/models")
        assert denied.status_code == 401 and "error" in denied.json()
        client.headers["Authorization"] = "Bearer local-secret"
        assert (await client.get("/v1/models")).status_code == 200
        bad = await client.post(
            "/v1/chat/completions",
            json={"model": "missing", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert bad.status_code == 404 and bad.json()["error"]["code"] == "model_not_found"
        invalid = await client.post("/v1/chat/completions", json={"secret_field": "do-not-echo"})
        assert invalid.status_code == 400 and "do-not-echo" not in invalid.text


async def test_oversized_chunked_input_is_rejected_before_upstream(make_backend, settings):
    data = settings.model_dump()
    data["limits"]["request_bytes"] = 1024
    config = Settings.model_validate(data)

    async def forbidden(req):
        raise AssertionError("upstream must not be called")

    async def chunks():
        for _ in range(4):
            yield b"x" * 400

    app = create_app(config, make_backend(forbidden, config))
    async with client_for(app) as client:
        response = await client.post("/v1/chat/completions", content=chunks())
        assert response.status_code == 413


async def test_request_capacity_returns_http_error_and_releases(make_backend, settings):
    data = settings.model_dump()
    data["limits"]["requests"] = 1
    config = Settings.model_validate(data)
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(req):
        started.set()
        await release.wait()
        return response()

    app = create_app(config, make_backend(handler, config))
    payload = {"model": "basic", "messages": [{"role": "user", "content": "hi"}]}
    async with client_for(app) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json=payload))
        await started.wait()
        second = await client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert second.status_code == 429
        release.set()
        assert (await first).status_code == 200
    assert app.state.runtime.active == 0


async def test_error_after_stream_start_has_no_success_terminator(make_backend, settings):
    app = create_app(
        settings, make_backend(lambda req: httpx.Response(200, stream=Events(sse(done=False))))
    )
    async with client_for(app) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "basic",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert "hello" in result.text and "incomplete_stream" in result.text
    assert "data: [DONE]\n\n" not in result.text
    assert app.state.runtime.active == 0


async def test_multimodel_tools_rejected(make_backend, settings):
    app = create_app(settings, make_backend(lambda req: response()))
    async with client_for(app) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "pair",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "Read", "parameters": {}}}],
            },
        )
        assert result.status_code == 400 and result.json()["error"]["code"] == "unsupported_tools"


async def test_lifespan_does_not_cancel_unrelated_tasks(make_backend, settings):
    app = create_app(settings, make_backend(lambda req: response()))
    unrelated = asyncio.create_task(asyncio.Event().wait())
    async with app.router.lifespan_context(app):
        pass
    assert not unrelated.done()
    unrelated.cancel()
    await asyncio.gather(unrelated, return_exceptions=True)


async def test_request_deadline_releases_capacity(make_backend, settings):
    data = settings.model_dump()
    data["limits"]["timeout_seconds"] = 0.03
    config = Settings.model_validate(data)
    stopped = asyncio.Event()

    async def handler(req):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    app = create_app(config, make_backend(handler, config))
    async with client_for(app) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "basic", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert result.status_code == 504
    assert stopped.is_set() and app.state.runtime.active == 0
