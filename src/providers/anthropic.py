from langchain_anthropic import ChatAnthropic

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    MODELS = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
    ]
    
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatAnthropic(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens if self.max_tokens else 4096,
            streaming=self.streaming_enabled
        )