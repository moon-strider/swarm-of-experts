from typing import Iterator, List
from langchain_openai import ChatOpenAI

from .base import LLMProvider, Message
from src.utils.message_converter import convert_messages


class OpenAIProvider(LLMProvider):
    MODELS = [
        "gpt-4o",
        "gpt-4.1-nano", 
        "gpt-4.1-mini",
        "gpt-4.1",
        "o4-mini",
        "o3",
        "o3-pro"
    ]
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming_enabled
        )
        
        
    def generate(self, messages: List[Message]) -> str:
        langchain_messages = convert_messages(messages)
        response = self._client.invoke(langchain_messages)
        return response.content
        
    def stream(self, messages: List[Message]) -> Iterator[str]:
        langchain_messages = convert_messages(messages)
        for chunk in self._client.stream(langchain_messages):
            if chunk.content:
                yield chunk.content