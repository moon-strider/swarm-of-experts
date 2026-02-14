# Swarm of Experts

A multi-LLM orchestration system that queries multiple AI providers in parallel and merges their responses into a single, high-quality answer. Expose it as an OpenAI-compatible API server for seamless integration with any tool that speaks the OpenAI protocol, or use the built-in interactive CLI.

## How It Works

```
User Query
    │
    ├──► Generator 1 (e.g. DeepSeek R1 70B, temp 0.7) ──┐
    ├──► Generator 2 (e.g. DeepSeek R1 70B, temp 0.5) ──┼──► Merger Model ──► Final Response
    └──► Generator 3 (e.g. DeepSeek R1 70B, temp 0.3) ──┘
```

Each generator produces an independent response. A dedicated merger model then analyzes all responses for accuracy, completeness, and clarity, synthesizing them into a single output that combines the best elements from each.

Single-model configurations skip the merge step entirely and work like a standard LLM proxy.

## Features

- **Parallel multi-model inference** with configurable temperature variation
- **OpenAI-compatible REST API** — drop-in replacement for `/v1/chat/completions`
- **Streaming support** via Server-Sent Events (SSE)
- **Task decomposition** — Taskmaster mode breaks complex queries into sub-prompts
- **5 LLM providers** — OpenAI, Anthropic, Google, Groq, DeepSeek
- **Session management** with automatic cleanup and reuse
- **Interactive CLI** with real-time streaming and spinner animations
- **Graceful degradation** — partial failures don't crash the system

## Quick Start

### Prerequisites

- Python 3.11+
- At least one API key from a supported provider

### Installation

```bash
git clone https://github.com/moon-strider/swarm-of-experts.git
cd swarm-of-experts
pip install -e .
```

### Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
DEEPSEEK_API_KEY=sk-...
```

You only need the keys for providers you plan to use. For example, if you only want to use `groq-swarm`, a single `GROQ_API_KEY` is enough.

### Optional Settings

```env
DEFAULT_MODEL=gpt-4.1-mini
TEMPERATURE=0.7
MAX_TOKENS=
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1
STREAM=true
```

## Usage

### CLI Mode

Launch the interactive chat interface:

```bash
swarm-of-experts
```

Or run directly:

```bash
python main.py
```

On startup the CLI validates your API keys, lists available swarm configurations, and prompts you to select one. Then it drops you into an interactive session with streaming responses.

**Commands inside the CLI:**

| Command | Description |
|---|---|
| `exit` / `quit` | End the conversation |
| `/clear` | Clear screen and conversation history |
| `Ctrl+C` | Interrupt a streaming response |

### API Server Mode

Start the OpenAI-compatible API server:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

For development with auto-reload:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Once running, any OpenAI-compatible client can talk to it. Swarm configuration names act as "model" identifiers.

### Using the API

**With curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "groq-swarm",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": false
  }'
```

**With the OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused"
)

response = client.chat.completions.create(
    model="groq-swarm",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**With any OpenAI-compatible tool** — just point its `base_url` to `http://localhost:8000/v1` and set any string as the API key.

## Swarm Configurations

These appear as "models" in the API and CLI:

| Config | Generators | Merger | Taskmaster | Description |
|---|---|---|---|---|
| `basic` | 1× OpenAI gpt-4.1 | — | — | Single model, no merging |
| `swarm-lite` | 3× OpenAI gpt-4.1-mini | gpt-4.1-mini | — | Parallel OpenAI with merge |
| `groq-swarm` | 3× Groq DeepSeek R1 70B | Kimi K2 | — | Parallel Groq with merge |
| `groq-single` | 1× Groq DeepSeek R1 70B | — | — | Single Groq model |
| `groq-taskmaster` | 3× Groq DeepSeek R1 70B | Kimi K2 | Kimi K2 | Task decomposition + merge |

Multi-generator configs use temperature variation (0.3 / 0.5 / 0.7) across generators to produce diverse responses before merging.

## API Reference

### Endpoints

#### `GET /health`

Health check with session statistics.

```json
{"status": "healthy", "timestamp": 1234567890, "session_stats": {"active_sessions": 2, "total_access_count": 15, "session_timeout_minutes": 30}}
```

#### `GET /v1/models`

List all available swarm configurations (OpenAI-compatible format).

#### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Supports both streaming (`"stream": true`) and non-streaming modes.

**Request body:**

```json
{
  "model": "groq-swarm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": null,
  "stream": false,
  "stream_options": {"include_usage": true}
}
```

#### `GET /v1/sessions/stats`

Active session count and access statistics.

#### `POST /v1/sessions/cleanup`

Force cleanup of expired sessions.

## Supported Providers and Models

| Provider | Models |
|---|---|
| **OpenAI** | gpt-4o, gpt-4.1-nano, gpt-4.1-mini, gpt-4.1, o4-mini, o3, o3-pro |
| **Anthropic** | claude-opus-4-20250514, claude-sonnet-4-20250514, claude-3-5-haiku-20241022 |
| **Groq** | deepseek-r1-distill-llama-70b, moonshotai/kimi-k2-instruct, gemma2-9b-it |
| **Google** | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite-preview-06-17 |
| **DeepSeek** | deepseek-chat, deepseek-reasoner |

## Project Structure

```
swarm-of-experts/
├── src/
│   ├── api/
│   │   ├── server.py            # FastAPI server, OpenAI-compatible endpoints
│   │   └── schemas.py           # Pydantic request/response models
│   ├── cli/
│   │   ├── app.py               # Interactive CLI application
│   │   ├── ui.py                # Terminal UI and theming
│   │   └── animations.py        # Spinner animations
│   ├── config/
│   │   ├── settings.py          # Environment-based configuration
│   │   └── swarm_configs.py     # Swarm configuration definitions
│   ├── core/
│   │   ├── chat.py              # ChatSession orchestration
│   │   ├── executor.py          # Parallel generator execution
│   │   ├── merger.py            # XML-based response merging
│   │   └── messages.py          # Conversation history management
│   ├── providers/
│   │   ├── base.py              # LLMProvider abstract base
│   │   ├── factory.py           # Provider instantiation
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── groq.py
│   │   ├── google.py
│   │   └── deepseek.py
│   └── utils/
│       ├── logging.py
│       └── terminal.py
├── tests/
│   ├── conftest.py              # Shared fixtures and mocks
│   ├── test_config.py           # Settings and swarm config tests
│   ├── test_providers.py        # Provider factory and base tests
│   ├── test_core.py             # Executor, merger, history tests
│   └── test_api.py              # Schema validation and endpoint tests
├── main.py                      # CLI entry point
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
└── LICENSE
```

## Development

### Running Tests

Install test dependencies and run the suite:

```bash
pip install -e ".[test]"
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing -v
```

The test suite (92 tests) runs entirely without API keys using mocks and fixtures.

### Linting

```bash
ruff check src/ tests/
```

### CI/CD

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs linting and tests on every push and pull request against `main`. Tests run on Python 3.11 and 3.12.

## Extending

### Adding a New Provider

1. Create `src/providers/your_provider.py` inheriting from `LLMProvider`
2. Implement `generate()`, `stream()`, `validate_model()`, and the `MODELS` class attribute
3. Register it in `ProviderFactory._providers` in `src/providers/factory.py`
4. Add the corresponding `YOUR_PROVIDER_API_KEY` to `src/config/settings.py`
5. Export it from `src/providers/__init__.py`

### Adding a New Swarm Configuration

Add an entry to `SWARM_CONFIGS` in `src/config/swarm_configs.py`:

```python
"my-swarm": SwarmConfig(
    name="my-swarm",
    generators=[
        GeneratorConfig(provider="openai", model="gpt-4.1", temperature=0.7),
        GeneratorConfig(provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.5),
    ],
    merger=GeneratorConfig(provider="openai", model="gpt-4.1", temperature=0.3),
)
```

The new config automatically appears in `GET /v1/models` and the CLI selection menu.

## License

[MIT](LICENSE)
