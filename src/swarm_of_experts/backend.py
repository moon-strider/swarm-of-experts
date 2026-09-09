"""Bounded OpenAI-compatible HTTP transport, shared by all configured endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Generator, Settings
from .models import Message


class SwarmError(Exception):
    def __init__(self, code: str, message: str, status: int = 502):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def usage_counts(value: Any) -> dict[str, int] | None:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(value, dict) or any(
        type(value.get(k)) is not int or value[k] < 0 for k in keys
    ):
        return None
    if value["total_tokens"] != value["prompt_tokens"] + value["completion_tokens"]:
        return None
    return {k: value[k] for k in keys}


@dataclass
class Completion:
    message: dict[str, Any]
    finish_reason: str
    usage: dict[str, int] | None

    @property
    def text(self) -> str:
        return self.message.get("content") or ""


class Trace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.degraded = False

    @property
    def usage(self) -> dict[str, int] | None:
        if not self.calls or any(c["usage"] is None for c in self.calls):
            return None
        return {
            k: sum(c["usage"][k] for c in self.calls)
            for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        }

    def report(self) -> dict[str, Any]:
        return {
            "calls": list(self.calls),
            "degraded": self.degraded,
            "usage_complete": self.usage is not None,
        }


class Backend:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self._owns_client = client is None
        self._slots = asyncio.Semaphore(settings.limits.provider_calls)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def payload(self, gen: Generator, messages: list[dict], options: dict, stream: bool) -> dict:
        payload = {
            "model": gen.model,
            "messages": messages,
            "temperature": gen.temperature,
            "max_tokens": gen.max_tokens,
            **options,
            "stream": stream,
        }
        if "max_completion_tokens" in options:
            payload.pop("max_tokens", None)
        limit = payload.get("max_completion_tokens", payload.get("max_tokens"))
        if type(limit) is not int or limit > self.settings.limits.output_tokens:
            raise SwarmError("token_limit", "Output token limit exceeds server configuration", 400)
        if stream and self.settings.providers[gen.provider].stream_usage:
            payload["stream_options"] = {"include_usage": True}
        if len(json.dumps(payload).encode()) > self.settings.limits.request_bytes:
            raise SwarmError(
                "request_too_large", "Expanded provider request exceeds byte limit", 413
            )
        return payload

    async def _bytes(self, response: httpx.Response) -> AsyncIterator[bytes]:
        count = 0
        async for chunk in response.aiter_bytes():
            count += len(chunk)
            if count > self.settings.limits.response_bytes:
                raise SwarmError("response_too_large", "Provider response exceeds byte limit")
            yield chunk

    async def _events(self, response: httpx.Response) -> AsyncIterator[dict]:
        pending = b""
        data: list[bytes] = []
        async for chunk in self._bytes(response):
            pending += chunk
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line = line.rstrip(b"\r")
                if line.startswith(b"data:"):
                    data.append(line[5:].lstrip(b" "))
                elif not line and data:
                    raw = b"\n".join(data)
                    data.clear()
                    if raw == b"[DONE]":
                        return
                    try:
                        event = json.loads(raw)
                    except (ValueError, RecursionError) as exc:
                        raise SwarmError(
                            "invalid_stream", "Provider emitted malformed SSE JSON"
                        ) from exc
                    if not isinstance(event, dict) or "error" in event:
                        raise SwarmError("upstream_error", "Provider reported a streaming error")
                    yield event
        raise SwarmError("incomplete_stream", "Provider stream ended without [DONE]")

    @staticmethod
    def _status(response: httpx.Response) -> None:
        if not response.is_success:
            status = 429 if response.status_code == 429 else 502
            raise SwarmError(
                "upstream_http_error", f"Provider returned HTTP {response.status_code}", status
            )

    async def events(
        self,
        gen: Generator,
        messages: list[dict],
        options: dict,
        trace: Trace,
        stage: str,
        *,
        stream: bool,
    ) -> AsyncGenerator[dict, None]:
        started = time.monotonic()
        record: dict[str, Any] = {
            "stage": stage,
            "provider": gen.provider,
            "model": gen.model,
            "status": "running",
            "usage": None,
        }
        trace.calls.append(record)
        try:
            provider = self.settings.providers[gen.provider]
            try:
                key = provider.key()
            except ValueError as exc:
                raise SwarmError("provider_not_configured", str(exc), 503) from exc
            payload = self.payload(gen, messages, options, stream)
            async with asyncio.timeout(self.settings.limits.timeout_seconds):
                async with self._slots:
                    async with self.client.stream(
                        "POST",
                        provider.base_url + "/chat/completions",
                        json=payload,
                        headers={"Authorization": "Bearer " + key},
                        timeout=self.settings.limits.timeout_seconds,
                    ) as response:
                        self._status(response)
                        if stream:
                            finished = False
                            async for event in self._events(response):
                                if counts := usage_counts(event.get("usage")):
                                    record["usage"] = counts
                                choices = event.get("choices")
                                if not isinstance(choices, list) or len(choices) > 1:
                                    raise SwarmError(
                                        "invalid_stream", "Expected one streaming choice"
                                    )
                                if choices:
                                    choice = choices[0]
                                    if not isinstance(choice, dict) or not isinstance(
                                        choice.get("delta", {}), dict
                                    ):
                                        raise SwarmError(
                                            "invalid_stream", "Invalid streaming choice"
                                        )
                                    delta = choice.get("delta", {})
                                    if delta.get("content") is not None and not isinstance(
                                        delta["content"], str
                                    ):
                                        raise SwarmError("invalid_stream", "Expected text delta")
                                    reason = choice.get("finish_reason")
                                    if reason is not None and (
                                        not isinstance(reason, str) or finished
                                    ):
                                        raise SwarmError(
                                            "invalid_stream", "Invalid repeated finish reason"
                                        )
                                    if finished and any(v for v in delta.values()):
                                        raise SwarmError(
                                            "invalid_stream", "Content followed terminal choice"
                                        )
                                    finished |= reason is not None
                                    yield {"delta": delta, "finish_reason": reason}
                            if not finished:
                                raise SwarmError(
                                    "incomplete_stream", "Provider omitted finish reason"
                                )
                        else:
                            raw = b"".join([chunk async for chunk in self._bytes(response)])
                            try:
                                body = json.loads(raw)
                                choices = body["choices"]
                                if len(choices) != 1:
                                    raise ValueError("choice count")
                                choice = choices[0]
                                if not isinstance(choice, dict) or not isinstance(
                                    choice.get("message"), dict
                                ):
                                    raise ValueError("invalid message")
                                message = Message.model_validate(
                                    {
                                        k: v
                                        for k, v in choice["message"].items()
                                        if k in Message.model_fields
                                    }
                                ).wire()
                                reason = choice["finish_reason"]
                                if message["role"] != "assistant" or not isinstance(reason, str):
                                    raise ValueError("invalid result")
                            except (ValueError, KeyError, TypeError, RecursionError) as exc:
                                raise SwarmError(
                                    "invalid_response", "Provider returned an invalid completion"
                                ) from exc
                            record["usage"] = usage_counts(body.get("usage"))
                            yield {
                                "message": message,
                                "finish_reason": reason,
                                "usage": record["usage"],
                            }
            record["status"] = "completed"
        except (httpx.TimeoutException, TimeoutError) as exc:
            record["status"] = "timeout"
            raise SwarmError("timeout", "Provider call deadline exceeded", 504) from exc
        except httpx.HTTPError as exc:
            record["status"] = "failed"
            raise SwarmError("connection_error", "Provider connection failed") from exc
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            raise
        except Exception:
            record["status"] = "failed"
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 4)

    async def complete(
        self, gen: Generator, messages: list[dict], options: dict, trace: Trace, stage: str
    ) -> Completion:
        events = [e async for e in self.events(gen, messages, options, trace, stage, stream=False)]
        e = events[0]
        return Completion(e["message"], e["finish_reason"], e["usage"])
