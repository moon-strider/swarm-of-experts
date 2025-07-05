from typing import Iterator, Optional
from ..providers.base import LLMProvider
from .messages import MessageHistory


class ChatSession:
    def __init__(self, provider: LLMProvider, max_history: Optional[int] = None):
        self.provider = provider
        self.history = MessageHistory(max_messages=max_history)
        
    def send_message(self, message: str) -> str:
        self.history.add_message("user", message)
        messages = self.history.get_messages()
        
        response = self.provider.generate(messages)
        self.history.add_message("assistant", response)
        
        return response
        
    def stream_message(self, message: str) -> Iterator[str]:
        self.history.add_message("user", message)
        messages = self.history.get_messages()
        
        full_response = []
        for chunk in self.provider.stream(messages):
            full_response.append(chunk)
            yield chunk
            
        self.history.add_message("assistant", "".join(full_response))
        
    def clear_history(self) -> None:
        self.history.clear()
        
    def get_history_length(self) -> int:
        return len(self.history)