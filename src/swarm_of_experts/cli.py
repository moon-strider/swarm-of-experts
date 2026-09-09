"""Console commands and a small interactive client with explicit local history."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import __version__
from .backend import SwarmError
from .config import load_settings
from .models import ChatRequest
from .runtime import Runtime


async def chat(args, settings) -> None:
    runtime = Runtime(settings)
    history: list[dict[str, str]] = []
    try:
        while True:
            prompt = (
                args.prompt if args.prompt is not None else await asyncio.to_thread(input, "you> ")
            )
            if prompt in {"exit", "quit"}:
                return
            if prompt == "/clear":
                history.clear()
                continue
            messages = [*history, {"role": "user", "content": prompt}]
            request = ChatRequest.model_validate(
                {
                    "model": args.model or settings.default_swarm,
                    "messages": messages,
                    "stream": args.stream,
                }
            )
            async with asyncio.timeout(settings.limits.timeout_seconds):
                if args.stream:
                    chunks = []
                    async for event in runtime.stream(request):
                        text = event.get("delta", {}).get("content") or ""
                        chunks.append(text)
                        print(text, end="", flush=True)
                    print()
                    answer = "".join(chunks)
                else:
                    result, trace = await runtime.complete(request)
                    answer = result.text
                    print(
                        json.dumps(
                            {"content": answer, "usage": trace.usage, "swarm": trace.report()},
                            indent=2,
                        )
                        if args.json
                        else answer
                    )
            history = [*messages, {"role": "assistant", "content": answer}][-32:]
            if args.prompt is not None:
                return
    finally:
        await runtime.backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded model ensembles")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check", action="store_true")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    client = sub.add_parser("chat")
    client.add_argument("--model")
    client.add_argument("--prompt")
    client.add_argument("--stream", action="store_true")
    client.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.check:
            available = {}
            for name, provider in settings.providers.items():
                try:
                    provider.key()
                    available[name] = True
                except ValueError:
                    available[name] = False
            print(
                json.dumps(
                    {
                        "version": __version__,
                        "providers_configured": available,
                        "swarms": list(settings.swarms),
                        "default_swarm": settings.default_swarm,
                        "limits": settings.limits.model_dump(),
                        "authentication": bool(settings.api_key),
                    },
                    indent=2,
                )
            )
        elif args.command == "serve":
            if args.host not in {"127.0.0.1", "localhost", "::1"} and not settings.api_key:
                raise ValueError("Set SWARM_API_KEY before binding a non-loopback interface")
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="warning")
        elif args.command == "chat":
            asyncio.run(chat(args, settings))
        else:
            parser.print_help()
    except (SwarmError, ValueError, OSError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(130) from None
