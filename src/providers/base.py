from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[datetime] = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class LLMProvider(ABC):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.streaming_enabled = kwargs.get('stream', True)
        
    def generate(self, messages: List[Message]) -> str:
        try:
            from ..utils.message_converter import convert_messages
            langchain_messages = convert_messages(messages)
            response = self._client.invoke(langchain_messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from {self.__class__.__name__}: {str(e)}")
        
    def stream(self, messages: List[Message]) -> Iterator[str]:
        try:
            from ..utils.message_converter import convert_messages
            langchain_messages = convert_messages(messages)
            for chunk in self._client.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Failed to stream response from {self.__class__.__name__}: {str(e)}")