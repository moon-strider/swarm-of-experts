import pytest
import os
from src.providers.anthropic import AnthropicProvider
from src.providers.base import Message


class TestAnthropicProvider:
    @pytest.fixture
    def api_key(self):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not found in environment")
        return key
    
    def test_provider_initialization(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        assert provider.max_tokens == 4096
        assert provider._client is not None
    
    
    def test_validate_model_valid(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        assert provider.validate_model() is True
    
    def test_validate_model_invalid(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-invalid")
        assert provider.validate_model() is False
    
    @pytest.mark.parametrize("model", [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022"
    ])
    def test_all_models_validation(self, api_key, model):
        provider = AnthropicProvider(api_key=api_key, model=model)
        assert provider.validate_model() is True
    
    def test_generate_simple_response(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022", max_tokens=10)
        messages = [Message(role="user", content="Say 'test'")]
        
        response = provider.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_stream_simple_response(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022", max_tokens=10, stream=True)
        messages = [Message(role="user", content="Say 'test'")]
        
        chunks = list(provider.stream(messages))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    
    def test_max_tokens_default(self, api_key):
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022")
        assert provider.max_tokens == 4096
        
        provider = AnthropicProvider(api_key=api_key, model="claude-3-5-haiku-20241022", max_tokens=100)
        assert provider.max_tokens == 100
    
