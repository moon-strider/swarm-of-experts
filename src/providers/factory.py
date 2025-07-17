from typing import Dict, Type
from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .deepseek import DeepSeekProvider
from .google import GoogleProvider
from .groq import GroqProvider


class ProviderFactory:
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "deepseek": DeepSeekProvider,
        "google": GoogleProvider,
        "groq": GroqProvider,
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def create(cls, provider_name: str, api_key: str, model: str, temperature: float = 0.7, **kwargs) -> LLMProvider:
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls._providers.keys())}")
            
        return provider_class(api_key=api_key, model=model, temperature=temperature, **kwargs)
        
