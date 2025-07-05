import pytest
import os
from src.providers.google import GoogleProvider
from src.providers.base import Message


class TestGoogleProvider:
    @pytest.fixture
    def api_key(self):
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GOOGLE_API_KEY not found in environment")
        return key
    
    def test_provider_initialization(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-2.5-flash-lite-exp")
        assert provider._client is not None
    
    
    def test_validate_model_valid(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-2.5-flash-lite-exp")
        assert provider.validate_model() is True
    
    def test_validate_model_invalid(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-invalid")
        assert provider.validate_model() is False
    
    @pytest.mark.parametrize("model", [
        "gemini-2.5-pro-exp",
        "gemini-2.5-flash-exp",
        "gemini-2.5-flash-lite-exp"
    ])
    def test_all_models_validation(self, api_key, model):
        provider = GoogleProvider(api_key=api_key, model=model)
        assert provider.validate_model() is True
    
    def test_generate_simple_response(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-2.5-flash-lite-exp", max_tokens=10)
        messages = [Message(role="user", content="Say 'test'")]
        
        response = provider.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_stream_simple_response(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-2.5-flash-lite-exp", max_tokens=10, stream=True)
        messages = [Message(role="user", content="Say 'test'")]
        
        chunks = list(provider.stream(messages))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    
    def test_max_output_tokens_parameter(self, api_key):
        provider = GoogleProvider(api_key=api_key, model="gemini-2.5-flash-lite-exp", max_tokens=100)
        assert provider.max_tokens == 100
        assert hasattr(provider.client, 'max_output_tokens')
    
