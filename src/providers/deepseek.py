from langchain_openai import ChatOpenAI

from .base import LLMProvider


class DeepSeekProvider(LLMProvider):
    MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=self.streaming_enabled
        )