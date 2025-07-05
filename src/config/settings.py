import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from .swarm_configs import get_swarm_config, SwarmConfig

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
    swarm_config_name: str = "basic"
    
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
    
    def get_swarm_config(self) -> SwarmConfig:
        return get_swarm_config(self.swarm_config_name)
    
    def get_api_key_for_provider(self, provider_name: str) -> Optional[str]:
        provider_lower = provider_name.lower()
        if provider_lower == "openai":
            return self.openai_api_key
        elif provider_lower == "anthropic":
            return self.anthropic_api_key
        elif provider_lower == "google":
            return self.google_api_key
        elif provider_lower == "deepseek":
            return self.deepseek_api_key
        else:
            raise ValueError(f"Unknown provider: {provider_name}")


settings = Settings()