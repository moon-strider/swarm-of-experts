import pytest
import os
from src.providers.openai import OpenAIProvider
from src.providers.base import Message


class TestOpenAIProvider:
    @pytest.fixture
    def api_key(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            pytest.skip("OPENAI_API_KEY not found in environment")
        return key
    
    def test_provider_initialization(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="gpt-4.1-mini")
        assert provider._client is not None
    
    
    def test_validate_model_valid(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="gpt-4.1-mini")
        assert provider.validate_model() is True
    
    def test_validate_model_invalid(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="invalid-model")
        assert provider.validate_model() is False
    
    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "o4-mini", "o3", "o3-pro"])
    def test_all_models_validation(self, api_key, model):
        provider = OpenAIProvider(api_key=api_key, model=model)
        assert provider.validate_model() is True
    
    def test_generate_simple_response(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="gpt-4.1-mini", max_tokens=10)
        messages = [Message(role="user", content="Say 'test'")]
        
        response = provider.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_stream_simple_response(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="gpt-4.1-mini", max_tokens=10, stream=True)
        messages = [Message(role="user", content="Say 'test'")]
        
        chunks = list(provider.stream(messages))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    
    
    def test_max_tokens_limit(self, api_key):
        provider = OpenAIProvider(api_key=api_key, model="gpt-4.1-mini", max_tokens=1)
        messages = [Message(role="user", content="Write a long story")]
        
        response = provider.generate(messages)
        assert len(response.split()) <= 3