from langchain_google_genai import ChatGoogleGenerativeAI

from .base import LLMProvider


class GoogleProvider(LLMProvider):
    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite-preview-06-17",
    ]
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self._client = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )