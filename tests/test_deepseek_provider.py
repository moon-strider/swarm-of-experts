import pytest
import os
from src.providers.deepseek import DeepSeekProvider
from src.providers.base import Message


class TestDeepSeekProvider:
    @pytest.fixture
    def api_key(self):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            pytest.skip("DEEPSEEK_API_KEY not found in environment")
        return key
    
    def test_provider_initialization(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-chat")
        assert provider._client is not None
    
    
    def test_validate_model_valid(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-chat")
        assert provider.validate_model() is True
    
    def test_validate_model_invalid(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-invalid")
        assert provider.validate_model() is False
    
    @pytest.mark.parametrize("model", ["deepseek-chat", "deepseek-reasoner"])
    def test_all_models_validation(self, api_key, model):
        provider = DeepSeekProvider(api_key=api_key, model=model)
        assert provider.validate_model() is True
    
    def test_generate_simple_response(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-chat", max_tokens=10)
        messages = [Message(role="user", content="Say 'test'")]
        
        response = provider.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_stream_simple_response(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-chat", max_tokens=10, stream=True)
        messages = [Message(role="user", content="Say 'test'")]
        
        chunks = list(provider.stream(messages))
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        full_response = "".join(chunks)
        assert len(full_response) > 0
    
    
    
    def test_custom_base_url(self, api_key):
        provider = DeepSeekProvider(api_key=api_key, model="deepseek-chat")
        assert provider._client.openai_api_base == "https://api.deepseek.com"