from langchain_groq import ChatGroq

from .base import LLMProvider


class GroqProvider(LLMProvider):
    MODELS = [
        "deepseek-r1-distill-llama-70b",
        "moonshotai/kimi-k2-instruct",
        "gemma2-9b-it",
    ]
    
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b", **kwargs):
        super().__init__(api_key, model, **kwargs)
        
        client_kwargs = {
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming_enabled
        }
        
        self._client = ChatGroq(**client_kwargs)
