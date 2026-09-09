# Migrating from the prototype

Version 0.2 changes the package and its execution model.

| Earlier behavior | Current behavior |
| --- | --- |
| Generic `src` Python package | Import `swarm_of_experts` |
| Shared history derived from request content | Every API request is independent |
| Implicit environment-file loading | Explicit environment and JSON configuration |
| Separate LangChain provider adapters | A bounded OpenAI-compatible HTTP backend |
| Partial streaming followed by fallback text | Explicit stream failure; no appended fallback |
| Character-based token estimates | Upstream usage, or unknown |
| Cancellation of all event-loop tasks | Cancellation of request-owned work |
| In-memory conversation cleanup | Stateless compatibility endpoints |
| Python-only source entrypoint | Installable console script and wheel |

The console script is `swarm-of-experts`. Source-tree `python main.py` delegates to the same CLI. The programmatic ASGI entrypoint is `swarm_of_experts.api.create_app(settings)`; configuration is not read and network clients are not created merely by importing the package.

Local clients can retain conversation history themselves. Multi-generator configurations handle text only; select a single-generator route for tools. Review strict request validation if a client sends provider-specific fields.

The new lockfile removes the old provider SDK and LangChain dependency tree. Production dependencies and their transitive versions are audited separately from test tooling.
