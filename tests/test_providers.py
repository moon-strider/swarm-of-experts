import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.providers.factory import ProviderFactory
from src.providers.base import LLMProvider
from src.providers.openai import OpenAIProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.google import GoogleProvider
from src.providers.groq import GroqProvider


class TestProviderFactory:
    @pytest.mark.parametrize("provider_name,expected_class", [
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
        ("deepseek", DeepSeekProvider),
        ("google", GoogleProvider),
        ("groq", GroqProvider),
    ])
    def test_create_all_providers(self, provider_name, expected_class):
        with patch.object(expected_class, '__init__', return_value=None):
            provider = ProviderFactory.create(
                provider_name, api_key="test-key", model="test-model"
            )
            assert isinstance(provider, expected_class)

    def test_create_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create("nonexistent", api_key="key", model="model")

    def test_case_insensitive_provider_name(self):
        with patch.object(OpenAIProvider, '__init__', return_value=None):
            provider = ProviderFactory.create("OpenAI", api_key="key", model="m")
            assert isinstance(provider, OpenAIProvider)

    def test_all_providers_registered(self):
        expected = {"openai", "anthropic", "deepseek", "google", "groq"}
        assert set(ProviderFactory._providers.keys()) == expected


class TestLLMProviderBase:
    def test_init_defaults(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(api_key="key", model="gpt-4.1")
            assert provider.model == "gpt-4.1"
            assert provider.temperature == 0.7
            assert provider.max_tokens is None
            assert provider.streaming_enabled is True

    def test_init_custom_kwargs(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(
                api_key="key", model="gpt-4.1",
                temperature=0.3, max_tokens=1000, stream=False,
            )
            assert provider.temperature == 0.3
            assert provider.max_tokens == 1000
            assert provider.streaming_enabled is False

    @pytest.mark.asyncio
    async def test_generate_calls_ainvoke(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(api_key="key", model="m")
            mock_response = MagicMock()
            mock_response.content = "Test response"
            provider._client = MagicMock()
            provider._client.ainvoke = AsyncMock(return_value=mock_response)

            from langchain_core.messages import HumanMessage
            result = await provider.generate([HumanMessage(content="Hello")])
            assert result == "Test response"
            provider._client.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_wraps_errors(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(api_key="key", model="m")
            provider._client = MagicMock()
            provider._client.ainvoke = AsyncMock(side_effect=Exception("API error"))

            from langchain_core.messages import HumanMessage
            with pytest.raises(RuntimeError, match="Failed to generate"):
                await provider.generate([HumanMessage(content="Hello")])

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(api_key="key", model="m")

            chunk1 = MagicMock()
            chunk1.content = "Hello"
            chunk2 = MagicMock()
            chunk2.content = " world"
            chunk3 = MagicMock()
            chunk3.content = ""

            async def mock_astream(messages):
                for c in [chunk1, chunk2, chunk3]:
                    yield c

            provider._client = MagicMock()
            provider._client.astream = mock_astream

            from langchain_core.messages import HumanMessage
            chunks = []
            async for chunk in provider.stream([HumanMessage(content="Hi")]):
                chunks.append(chunk)

            assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_wraps_errors(self):
        with patch.object(OpenAIProvider, '__init__', lambda self, **kw: LLMProvider.__init__(self, **kw)):
            provider = OpenAIProvider(api_key="key", model="m")

            async def mock_astream_error(messages):
                raise Exception("Stream error")
                yield

            provider._client = MagicMock()
            provider._client.astream = mock_astream_error

            from langchain_core.messages import HumanMessage
            with pytest.raises(RuntimeError, match="Failed to stream"):
                async for _ in provider.stream([HumanMessage(content="Hi")]):
                    pass


class TestProviderModels:
    @pytest.mark.parametrize("provider_class,expected_models", [
        (OpenAIProvider, ["gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]),
        (AnthropicProvider, ["claude-opus-4-20250514", "claude-sonnet-4-20250514"]),
        (GroqProvider, ["deepseek-r1-distill-llama-70b"]),
    ])
    def test_models_list_not_empty(self, provider_class, expected_models):
        models = provider_class.MODELS
        assert isinstance(models, list)
        assert len(models) > 0
        for model in expected_models:
            assert model in models
