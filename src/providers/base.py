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
        
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        pass
        
    @abstractmethod
    def stream(self, messages: List[Message]) -> Iterator[str]:
        pass