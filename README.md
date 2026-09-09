# swarm-of-experts

Run one model or a small ensemble behind an OpenAI-compatible Chat Completions API.

Swarm sends independent requests to configured generators, optionally decomposes a prompt with a taskmaster, and combines successful answers with a merger. Every response reports the calls it actually made. It also works as a local API between llama.cpp and OpenClaw.

## Quick start

Requires Python 3.11 or later.

~~~bash
git clone https://github.com/moon-strider/swarm-of-experts
cd swarm-of-experts
uv sync --frozen
export OPENAI_API_KEY='your-key'
uv run swarm-of-experts chat --model basic --prompt 'Explain optimistic concurrency'
~~~

For a running local OpenAI-compatible server:

~~~bash
export LLM_BASE_URL=http://127.0.0.1:8080/v1
export LLM_MODEL=local-model
uv run swarm-of-experts --check
uv run swarm-of-experts serve --port 8000
~~~

~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-single","messages":[{"role":"user","content":"Hello"}]}'
~~~

Use `local-single` for tool-using clients, including OpenClaw. Use `local-swarm` for three text generators followed by a merger. Configure your own models in [examples/local.json](examples/local.json).

For an installed CLI without an active environment, run `uv tool install .`. The source-tree `python main.py` entrypoint remains available.

## What it handles

- Independent request contexts; callers supply their own message history.
- Text Chat Completions, real SSE streaming, and function tool calls in single-model configurations.
- Optional taskmaster decomposition, partial generator failure, and an explicit minimum-success threshold.
- Shared provider concurrency limits, request admission, deadlines, bounded bodies and cancellation.
- Actual aggregate usage when every upstream call reports it; otherwise `usage: null`.
- Optional bearer authentication. The CLI binds to loopback by default.

The API is a supported subset of Chat Completions. It does not implement Responses, images, embeddings, server-side conversations, or tool execution. Ensemble configurations reject tool requests because competing tool actions need a separate execution policy.

An ensemble can repeat or amplify the same mistake. Its output is an answer to evaluate, not a correctness guarantee.

## Local models and long tasks

[openclaw-long-tasks](https://github.com/moon-strider/openclaw-long-tasks) owns durable jobs, checkpoints and the MAKER-inspired Hanoi experiment. It can use this server as its model endpoint. Swarm remains usable independently.

[Local inference and OpenClaw](docs/local-inference.md) explains the setup. [The experiment report](https://github.com/moon-strider/openclaw-long-tasks/blob/main/docs/experiments.md) contains the measured outcomes and failed attempts.

## Configuration and API

| Topic | Guide |
| --- | --- |
| Routes, accepted fields, errors, streaming and limits | [API reference](docs/api.md) |
| CPU inference, OpenClaw and MAKER integration | [Local inference](docs/local-inference.md) |
| Changes from the original prototype | [Migration notes](docs/migration.md) |

~~~bash
uv run swarm-of-experts --config examples/local.json --check
uv run swarm-of-experts --config examples/local.json serve
uv run swarm-of-experts chat --model local-single --prompt 'Hello' --stream
~~~

Configuration is explicit JSON. Credentials are selected by environment-variable name, never embedded in configuration files. The package does not load `.env` files automatically.

## Development

~~~bash
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run python -m pytest --cov --cov-report=term-missing
uv build
~~~

CI tests Python 3.11–3.14, checks the built wheel, and audits the locked production dependencies. Network model experiments are separate from deterministic tests; ordinary tests need no provider account or model download.

MIT licensed. Issues and small, tested pull requests are welcome.
