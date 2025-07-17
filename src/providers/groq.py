from typing import Iterator, List
from langchain_groq import ChatGroq

from .base import LLMProvider
from src.utils.message_converter import convert_messages


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
        
        
    def generate(self, messages: List) -> str:
        try:
            langchain_messages = convert_messages(messages)
            response = self._client.invoke(langchain_messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from Groq: {str(e)}")
        
    def stream(self, messages: List) -> Iterator[str]:
        try:
            langchain_messages = convert_messages(messages)
            for chunk in self._client.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Failed to stream response from Groq: {str(e)}")