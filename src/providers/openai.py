from langchain_openai import ChatOpenAI

from .base import LLMProvider


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