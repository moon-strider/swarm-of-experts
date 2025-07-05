from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from ..providers.base import Message


@dataclass
class ChatMessage(Message):
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    
class MessageHistory:
    def __init__(self, max_messages: Optional[int] = None):
        self._messages: List[ChatMessage] = []
        self.max_messages = max_messages
        
    def add_message(self, role: str, content: str, **metadata) -> None:
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self._messages.append(message)
        
        if self.max_messages and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
            
    def get_messages(self) -> List[Message]:
        return [Message(role=msg.role, content=msg.content) for msg in self._messages]
        
    def clear(self) -> None:
        self._messages.clear()
        
    def __len__(self) -> int:
        return len(self._messages)
        
    def __iter__(self):
        return iter(self._messages)