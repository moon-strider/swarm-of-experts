import pytest
from src.providers.base import Message


class TestMessage:
    def test_message_to_dict(self):
        msg = Message(role="assistant", content="Hi there")
        result = msg.to_dict()
        assert result == {"role": "assistant", "content": "Hi there"}