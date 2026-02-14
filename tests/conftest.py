import pytest
from unittest.mock import MagicMock, AsyncMock

from src.config.swarm_configs import SwarmConfig, GeneratorConfig


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY", "GROQ_API_KEY", "DEFAULT_MODEL",
        "TEMPERATURE", "MAX_TOKENS", "SERVER_HOST", "SERVER_PORT",
        "SERVER_WORKERS",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-groq-key")


@pytest.fixture
def single_generator_config():
    return SwarmConfig(
        name="test-single",
        generators=[GeneratorConfig(provider="openai", model="gpt-4.1-mini")],
        merger=None,
    )


@pytest.fixture
def multi_generator_config():
    return SwarmConfig(
        name="test-multi",
        generators=[
            GeneratorConfig(provider="openai", model="gpt-4.1-mini", temperature=0.7),
            GeneratorConfig(provider="openai", model="gpt-4.1-mini", temperature=0.3),
            GeneratorConfig(provider="openai", model="gpt-4.1-mini", temperature=0.5),
        ],
        merger=GeneratorConfig(provider="openai", model="gpt-4.1-mini", temperature=0.3),
    )


@pytest.fixture
def taskmaster_config():
    return SwarmConfig(
        name="test-taskmaster",
        generators=[
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.7),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.5),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.3),
        ],
        taskmaster=GeneratorConfig(provider="groq", model="moonshotai/kimi-k2-instruct", temperature=0.3),
        merger=GeneratorConfig(provider="groq", model="moonshotai/kimi-k2-instruct", temperature=0.3),
    )


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.generate = AsyncMock(return_value="Mocked response content")
    provider.stream = AsyncMock()

    async def mock_stream(messages):
        for chunk in ["Hello", " ", "world"]:
            yield chunk

    provider.stream = mock_stream
    return provider
