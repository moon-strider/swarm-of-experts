from typing import Iterator, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base import LLMProvider, Message


class GoogleProvider(LLMProvider):
    MODELS = [
        "gemini-2.5-pro-exp",
        "gemini-2.5-flash-exp",
        "gemini-2.5-flash-lite-exp",
    ]
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite-exp", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
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