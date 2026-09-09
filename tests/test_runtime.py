import asyncio
import json
from contextlib import aclosing

import httpx
import pytest

from swarm_of_experts.backend import SwarmError
from swarm_of_experts.config import Settings
from swarm_of_experts.models import ChatRequest
from swarm_of_experts.runtime import Runtime

from .conftest import response


def request(model="basic", **options):
    return ChatRequest(model=model, messages=[{"role": "user", "content": "once"}], **options)


class Events(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def sse(text="hello", done=True):
    first = {"choices": [{"delta": {"role": "assistant", "content": text}, "finish_reason": None}]}
    chunks = [
        f"data: {json.dumps(first)}\n\n".encode(),
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: {"choices":[],"usage":'
        b'{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
    ]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


async def test_request_history_is_not_duplicated_or_shared(make_backend, settings):
    seen = []

    async def handler(req):
        seen.append(json.loads(req.content))
        await asyncio.sleep(0.01)
        return response()

    runtime = Runtime(settings, make_backend(handler))
    await asyncio.gather(runtime.complete(request()), runtime.complete(request()))
    assert [r["messages"] for r in seen] == [[{"role": "user", "content": "once"}]] * 2
    assert seen[0]["model"] == "test-model"


async def test_ensemble_usage_sums_all_calls_with_out_of_order_completions(make_backend, settings):
    counter = 0

    async def handler(req):
        nonlocal counter
        counter += 1
        current = counter
        await asyncio.sleep(0.02 if current == 1 else 0)
        return response(str(current), tokens=current)

    runtime = Runtime(settings, make_backend(handler))
    result, trace = await runtime.complete(request("pair"))
    assert result.text == "3"
    assert result.usage["prompt_tokens"] == 3
    assert trace.usage == {"prompt_tokens": 6, "completion_tokens": 6, "total_tokens": 12}
    assert [c["usage"]["prompt_tokens"] for c in trace.calls] == [1, 2, 3]


async def test_options_reach_every_provider_call(make_backend, settings):
    seen = []

    async def handler(req):
        seen.append(json.loads(req.content))
        return response()

    await Runtime(settings, make_backend(handler)).complete(
        request("pair", max_completion_tokens=12, temperature=0, top_p=0.8, stop=["end"], seed=8)
    )
    assert len(seen) == 3
    assert all(
        r["max_completion_tokens"] == 12
        and "max_tokens" not in r
        and r["temperature"] == 0
        and r["stop"] == ["end"]
        for r in seen
    )


async def test_partial_failures_and_minimum_success(make_backend, settings):
    count = 0

    async def handler(req):
        nonlocal count
        count += 1
        return response("survivor") if count % 2 else httpx.Response(503)

    result, trace = await Runtime(settings, make_backend(handler)).complete(request("pair"))
    assert result.text == "survivor" and trace.degraded
    assert trace.usage is None
    data = settings.model_dump()
    data["swarms"]["pair"]["min_success"] = 2
    strict = Settings.model_validate(data)
    with pytest.raises(SwarmError, match="Too few"):
        await Runtime(strict, make_backend(handler, strict)).complete(request("pair"))


async def test_cancelled_ensemble_closes_all_owned_calls(make_backend, settings):
    started, settled = [], []
    ready = asyncio.Event()

    async def handler(req):
        index = len(started)
        started.append(index)
        if len(started) == 2:
            ready.set()
        try:
            await asyncio.Event().wait()
        finally:
            settled.append(index)

    task = asyncio.create_task(Runtime(settings, make_backend(handler)).complete(request("pair")))
    await ready.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(settled) == [0, 1]


async def test_provider_parallelism_limit(make_backend, settings):
    data = settings.model_dump()
    data["limits"]["provider_calls"] = 1
    config = Settings.model_validate(data)
    active = maximum = 0

    async def handler(req):
        nonlocal active, maximum
        active += 1
        maximum = max(active, maximum)
        await asyncio.sleep(0.01)
        active -= 1
        return response()

    await Runtime(config, make_backend(handler, config)).complete(request("pair"))
    assert maximum == 1


async def test_stream_is_incremental_and_utf8_safe(make_backend, settings):
    wire = b"".join(sse("привет"))
    stream = Events([wire[i : i + 1] for i in range(len(wire))])
    runtime = Runtime(settings, make_backend(lambda req: httpx.Response(200, stream=stream)))
    events = [e async for e in runtime.stream(request())]
    assert events[0]["delta"]["content"] == "привет"
    assert events[-1]["usage"]["total_tokens"] == 5
    assert stream.closed


async def test_broken_stream_never_appends_fallback(make_backend, settings):
    data = settings.model_dump()
    data["swarms"]["pair"]["merger_failure"] = "first"
    config = Settings.model_validate(data)
    broken = Events(sse("partial:", done=False))

    async def handler(req):
        return (
            httpx.Response(200, stream=broken)
            if json.loads(req.content)["stream"]
            else response("fallback")
        )

    seen = []
    with pytest.raises(SwarmError, match="without"):
        async for event in Runtime(config, make_backend(handler, config)).stream(request("pair")):
            seen.append(event)
    assert "".join(e.get("delta", {}).get("content", "") for e in seen) == "partial:"
    assert broken.closed


@pytest.mark.parametrize(
    "upstream, code",
    [
        (lambda: httpx.Response(401, text="SECRET"), "upstream_http_error"),
        (lambda: httpx.Response(200, text="bad"), "invalid_response"),
        (lambda: httpx.Response(200, json={"choices": []}), "invalid_response"),
        (lambda: httpx.Response(200, content=b"x" * 1_048_577), "response_too_large"),
    ],
)
async def test_backend_errors_are_bounded_and_sanitized(make_backend, settings, upstream, code):
    with pytest.raises(SwarmError) as caught:
        await Runtime(settings, make_backend(lambda req: upstream())).complete(request())
    assert caught.value.code == code
    assert "SECRET" not in str(caught.value)


async def test_unknown_usage_is_not_invented(make_backend, settings):
    body = response().json()
    body.pop("usage")
    result, trace = await Runtime(
        settings, make_backend(lambda req: httpx.Response(200, json=body))
    ).complete(request())
    assert result.usage is None and trace.usage is None


async def test_provider_timeout_and_connection_error(make_backend, settings):
    for error, code in [
        (httpx.ReadTimeout("secret"), "timeout"),
        (httpx.ConnectError("secret"), "connection_error"),
    ]:

        async def handler(req, error=error):
            raise error

        with pytest.raises(SwarmError) as caught:
            await Runtime(settings, make_backend(handler)).complete(request())
        assert caught.value.code == code


async def test_closing_stream_early_closes_provider(make_backend, settings):
    stream = Events(sse())
    runtime = Runtime(settings, make_backend(lambda req: httpx.Response(200, stream=stream)))
    async with aclosing(runtime.stream(request())) as output:
        await anext(output)
    assert stream.closed


async def test_completion_tool_call_is_preserved(make_backend, settings):
    body = response().json()
    body["choices"][0] = {
        "message": {
            "role": "assistant",
            "content": None,
            "refusal": None,
            "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        },
        "finish_reason": "tool_calls",
    }
    result, _ = await Runtime(
        settings, make_backend(lambda req: httpx.Response(200, json=body))
    ).complete(request())
    assert result.message["tool_calls"][0]["id"] == "call"
    assert result.finish_reason == "tool_calls"


async def test_taskmaster_uses_dynamic_count_and_keeps_system_context(make_backend, settings):
    data = settings.model_dump()
    data["swarms"]["pair"]["taskmaster"] = data["swarms"]["pair"]["merger"]
    config = Settings.model_validate(data)
    seen = []

    async def handler(req):
        seen.append(json.loads(req.content))
        return response('{"prompts":["first","second"]}') if len(seen) == 1 else response()

    task = ChatRequest(
        model="pair",
        messages=[{"role": "system", "content": "keep"}, {"role": "user", "content": "question"}],
    )
    await Runtime(config, make_backend(handler, config)).complete(task)
    assert "exactly 2" in seen[0]["messages"][-1]["content"]
    assert seen[1]["messages"] == [
        {"role": "system", "content": "keep"},
        {"role": "user", "content": "first"},
    ]
