import pytest
import os
from src.providers.factory import ProviderFactory
from src.providers.openai import OpenAIProvider
from src.providers.anthropic import AnthropicProvider
from src.providers.google import GoogleProvider
from src.providers.deepseek import DeepSeekProvider
from src.providers.base import LLMProvider


class TestProviderFactory:
    @pytest.fixture
    def factory(self):
        return ProviderFactory()
    
    @pytest.fixture
    def openai_key(self):
        return os.getenv("OPENAI_API_KEY", "test-key")
    
    def test_create_openai_provider(self, factory, openai_key):
        provider = factory.create("openai", api_key=openai_key, model="gpt-4.1-mini")
        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4.1-mini"
        assert provider.api_key == openai_key
    
    def test_create_anthropic_provider(self, factory):
        api_key = os.getenv("ANTHROPIC_API_KEY", "test-key")
        provider = factory.create("anthropic", api_key=api_key, model="claude-3-5-haiku-20241022")
        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-3-5-haiku-20241022"
    
    def test_create_google_provider(self, factory):
        api_key = os.getenv("GOOGLE_API_KEY", "test-key")
        provider = factory.create("google", api_key=api_key, model="gemini-2.5-flash-lite-exp")
        assert isinstance(provider, GoogleProvider)
        assert provider.model == "gemini-2.5-flash-lite-exp"
    
    def test_create_deepseek_provider(self, factory):
        api_key = os.getenv("DEEPSEEK_API_KEY", "test-key")
        provider = factory.create("deepseek", api_key=api_key, model="deepseek-chat")
        assert isinstance(provider, DeepSeekProvider)
        assert provider.model == "deepseek-chat"
    
    def test_create_invalid_provider(self, factory):
        with pytest.raises(ValueError) as exc_info:
            factory.create("invalid_provider", api_key="test", model="test-model")
        assert "Unknown provider" in str(exc_info.value)
    
    def test_register_custom_provider(self, factory):
        class CustomProvider(LLMProvider):
            def generate(self, messages): return "custom"
            def stream(self, messages): yield "custom"
            def validate_model(self): return True
            @property
            def available_models(self): return ["custom-model"]
        
        factory.register("custom", CustomProvider)
        assert "custom" in factory.get_available_providers()
        
        provider = factory.create("custom", api_key="test", model="custom-model")
        assert isinstance(provider, CustomProvider)
    
    @pytest.mark.parametrize("model,expected_provider", [
        ("gpt-4.1-mini", "openai"),
        ("gpt-4o", "openai"),
        ("o3", "openai"),
        ("o4-mini", "openai"),
        ("claude-3-5-haiku-20241022", "anthropic"),
        ("claude-opus-4-20250514", "anthropic"),
        ("gemini-2.5-flash-exp", "google"),
        ("gemini-2.5-pro-exp", "google"),
        ("deepseek-chat", "deepseek"),
        ("deepseek-reasoner", "deepseek")
    ])
    def test_get_provider_from_model(self, factory, model, expected_provider):
        provider_name = factory.get_provider_from_model(model)
        assert provider_name == expected_provider
    
    def test_get_provider_from_unknown_model(self, factory):
        provider_name = factory.get_provider_from_model("unknown-model-xyz")
        assert provider_name is None
    
    def test_create_with_custom_parameters(self, factory, openai_key):
        provider = factory.create(
            "openai",
            api_key=openai_key,
            model="gpt-4.1-mini",
            temperature=0.5,
            max_tokens=100,
            stream=False
        )
        assert provider.temperature == 0.5
        assert provider.max_tokens == 100
        assert provider.streaming_enabled is False