from typing import List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


class MessageHistory:
    def __init__(self, max_messages: Optional[int] = None):
        self._messages: List[BaseMessage] = []
        self.max_messages = max_messages
        
    def add_message(self, role: str, content: str, **metadata) -> None:
        if role == "user":
            message = HumanMessage(content=content, additional_kwargs=metadata)
        elif role == "assistant":
            message = AIMessage(content=content, additional_kwargs=metadata)
        elif role == "system":
            message = SystemMessage(content=content, additional_kwargs=metadata)
        else:
            raise ValueError(f"Unknown role: {role}")
        self._messages.append(message)
        
        if self.max_messages and len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]
            
    def get_messages(self) -> List[BaseMessage]:
        return self._messages[:]
        
    def clear(self) -> None:
        self._messages.clear()