from typing import Iterator, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .base import LLMProvider, Message


class GroqProvider(LLMProvider):
    MODELS = [
        "deepseek-r1-distill-llama-70b",
        "moonshotai/kimi-k2-instruct",
        "gemma2-9b-it",
    ]
    
    def __init__(self, api_key: str, model: str = "deepseek-r1-distill-llama-70b", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatGroq(
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
        
    def generate(self, messages: List) -> str:
        try:
            langchain_messages = self._convert_messages(messages)
            response = self._client.invoke(langchain_messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from Groq: {str(e)}")
        
    def stream(self, messages: List) -> Iterator[str]:
        try:
            langchain_messages = self._convert_messages(messages)
            for chunk in self._client.stream(langchain_messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Failed to stream response from Groq: {str(e)}")
                
    def validate_model(self) -> bool:
        return self.model in self.MODELS
        
    @property
    def available_models(self) -> List[str]:
        return self.MODELS