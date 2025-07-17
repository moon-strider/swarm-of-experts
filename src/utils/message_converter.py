from typing import List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.providers.base import Message


def convert_messages(messages: List[Message]):
    converted = []
    for msg in messages:
        if msg.role == "user":
            converted.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            converted.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            converted.append(SystemMessage(content=msg.content))
    return converted