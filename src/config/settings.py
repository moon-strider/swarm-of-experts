import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    default_model: str = "gpt-4.1-mini"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = True
    
    def __post_init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        
        if env_model := os.getenv('DEFAULT_MODEL'):
            self.default_model = env_model
        if env_temp := os.getenv('TEMPERATURE'):
            self.temperature = float(env_temp)
        if env_tokens := os.getenv('MAX_TOKENS'):
            self.max_tokens = int(env_tokens)
            
    def validate(self) -> tuple[bool, str]:
        if not any([self.openai_api_key, self.anthropic_api_key, self.google_api_key, self.deepseek_api_key]):
            return False, "No API keys found. Please set at least one provider API key."
        return True, ""


settings = Settings()