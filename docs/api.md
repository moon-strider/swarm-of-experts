# API and configuration

## Routes

| Route | Purpose |
| --- | --- |
| GET /health | Minimal public liveness response |
| GET /v1/models | Configured swarm names, not upstream discovery |
| POST /v1/chat/completions | Text completion or SSE stream |
| GET /sessions/stats | Compatibility response describing the stateless server |
| POST /sessions/cleanup | Compatibility no-op; no conversations are retained |

Except for health, routes require `Authorization: Bearer ...` when `SWARM_API_KEY` is set. Non-loopback CLI binding requires that key. When embedding the ASGI app behind your own server, configure authentication and TLS at that boundary.

## Request subset

Required: `model` and `messages`. Messages accept system, developer, user, assistant and tool roles, with strings or text-only content parts. Tool messages require `tool_call_id`.

Optional generation fields: `temperature`, `top_p`, `max_tokens` or `max_completion_tokens` (choose one), `frequency_penalty`, `presence_penalty`, `stop`, `seed`, and `n: 1`. `user` is accepted for client compatibility but is not forwarded or retained.

Single-model routes also forward `tools`, `tool_choice`, `parallel_tool_calls` and tool-call history. Tools must use the function schema. The client executes the selected tool and sends its result in a subsequent request.

`response_format` is forwarded; the upstream provider determines which structured-output formats it supports. Unknown request fields fail validation instead of being silently ignored. In particular, Responses API and provider-specific reasoning extensions are not accepted.

## Streaming and errors

Set `stream: true`. To request aggregate usage, also set `stream_options: {"include_usage":true}`. SSE chunks retain one completion id and the requested model name. A successful stream ends with `data: [DONE]`.

Generator work finishes before merger text streams. If an upstream stream fails after sending text, the server emits a structured error and ends without a success marker. It never appends a fallback answer to an already-started answer. Clients must treat that output as incomplete.

Errors use an OpenAI-style `error` object. Typical HTTP status codes are 400 for unsupported or invalid input, 401 for authentication, 404 for unknown model names, 413 for size limits, 429 for admission or provider capacity, and 502 for upstream failures. Errors that occur after streaming headers arrive travel in SSE.

## Topology

A single configuration contains one generator and no merger or taskmaster. A text ensemble contains 2–16 generators and a merger. A taskmaster, when configured, must return exactly one prompt for each generator; the server preserves earlier conversation context.

A generator result is eligible for merging only when it has nonempty text, no tool call, and a normal stop. The default `min_success` is one. A single surviving answer is returned with degradation metadata. Set a higher threshold when partial results are unsuitable.

`merger_failure: "error"` is the default. Opting into `"first"` returns the first successful generator if the merger fails before producing text. This fallback is always marked as degraded.

Request generation options override model defaults for each generator and merger. Taskmaster generation uses its configured defaults.

## Limits and accounting

Default limits are 8 admitted requests, 8 concurrent provider calls, a 300-second request deadline, one MiB input and upstream-response bodies, and 4096 output tokens per call. These are independently configurable in the `limits` object.

Provider concurrency is shared across requests. Expanded merger prompts are checked again against the request byte limit. Disconnects cancel work owned by that request; shutdown closes the owned HTTP client without cancelling unrelated event-loop tasks.

The final response includes a `swarm` extension with per-call stage, provider, model, status, elapsed time and reported usage. Top-level usage sums taskmaster, generators and merger only when all calls report valid counts. Failed calls can make total usage unknown even if they consumed tokens. No character-based token estimates or cost estimates are invented.

## Providers and credentials

Presets use OpenAI-compatible HTTP endpoints for OpenAI, Anthropic, Google, Groq and DeepSeek. The Anthropic route uses its [limited compatibility layer](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk); it does not expose native Claude features. Google documents its [compatibility endpoint](https://ai.google.dev/gemini-api/docs/openai). Cloud inference is not part of the local test results.

Set `api_key_env` to the name of an environment variable. HTTPS is required except for literal loopback hosts. URLs cannot include embedded credentials, queries or fragments. Custom endpoints are operator-controlled configuration.

Built-in names are retained for migration, but hosted model availability changes. Replace model ids explicitly when a provider retires one.
