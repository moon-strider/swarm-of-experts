from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage


class LLMProvider(ABC):
    def __init__(self, api_key: str, model: str, **kwargs):
        self.api_key = api_key
        self.model = model
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.streaming_enabled = kwargs.get('stream', True)
        
    async def generate(self, messages: List[BaseMessage]) -> str:
        try:
            response = await self._client.ainvoke(messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from {self.__class__.__name__}: {str(e)}")
        
    async def stream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        try:
            async for chunk in self._client.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Failed to stream response from {self.__class__.__name__}: {str(e)}")