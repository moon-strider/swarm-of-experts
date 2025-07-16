import pytest
from src.providers.groq import GroqProvider


class TestGroqProvider:
    def test_model_list(self):
        provider = GroqProvider(api_key="test", model="deepseek-r1-distill-llama-70b")
        models = provider.available_models
        assert "deepseek-r1-distill-llama-70b" in models
        assert "kimi-k2-instruct" in models
        assert len(models) > 5

    def test_validate_model(self):
        provider = GroqProvider(api_key="test", model="deepseek-r1-distill-llama-70b")
        assert provider.validate_model() is True
        
        provider = GroqProvider(api_key="test", model="invalid-model")
        assert provider.validate_model() is False

    def test_provider_creation(self):
        provider = GroqProvider(
            api_key="test-key",
            model="kimi-k2-instruct",
            temperature=0.5
        )
        assert provider.model == "kimi-k2-instruct"
        assert provider.temperature == 0.5
        assert provider.api_key == "test-key"