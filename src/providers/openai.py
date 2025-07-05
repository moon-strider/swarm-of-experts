from typing import Iterator, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    MODELS = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo"
    ]
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming_enabled
        )
        
    def _convert_messages(self, messages: List[Message]):
        converted = []
        for msg in messages:
            if msg.role == "user":
                converted.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                converted.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                converted.append(SystemMessage(content=msg.content))
        return converted
        
    def generate(self, messages: List[Message]) -> str:
        langchain_messages = self._convert_messages(messages)
        response = self._client.invoke(langchain_messages)
        return response.content
        
    def stream(self, messages: List[Message]) -> Iterator[str]:
        langchain_messages = self._convert_messages(messages)
        for chunk in self._client.stream(langchain_messages):
            if chunk.content:
                yield chunk.content
                
    def validate_model(self) -> bool:
        return self.model in self.MODELS
        
    @property
    def available_models(self) -> List[str]:
        return self.MODELS