from typing import Optional, Dict, Type
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
    def create(cls, provider_name: str, api_key: str, model: str, **kwargs) -> LLMProvider:
        provider_class = cls._providers.get(provider_name.lower())
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}. Available: {list(cls._providers.keys())}")
            
        return provider_class(api_key=api_key, model=model, **kwargs)
        
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        cls._providers[name.lower()] = provider_class
        
    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(cls._providers.keys())
        
    @classmethod
    def get_provider_from_model(cls, model: str) -> Optional[str]:
        model_lower = model.lower()
        if model_lower.startswith(("gpt", "o3", "o4")):
            return "openai"
        elif model_lower.startswith("claude"):
            return "anthropic"
        elif model_lower.startswith("gemini"):
            return "google"
        elif model_lower.startswith("deepseek"):
            return "deepseek"
        elif model_lower.startswith(("deepseek-r1", "kimi", "llama", "mixtral", "gemma")):
            return "groq"
        return None