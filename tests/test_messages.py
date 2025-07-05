import pytest
from datetime import datetime
from src.core.messages import ChatMessage, MessageHistory
from src.providers.base import Message


class TestChatMessage:
    def test_chat_message_with_metadata(self):
        metadata = {"source": "test", "id": 123}
        msg = ChatMessage(role="assistant", content="Hi", metadata=metadata)
        assert msg.metadata == metadata
    
    def test_chat_message_timestamp(self):
        before = datetime.now()
        msg = ChatMessage(role="user", content="Test")
        after = datetime.now()
        
        assert before <= msg.timestamp <= after


class TestMessageHistory:
    def test_history_with_max_messages(self):
        history = MessageHistory(max_messages=3)
        assert history.max_messages == 3
    
    def test_add_message(self):
        history = MessageHistory()
        history.add_message("user", "Hello")
        
        assert len(history) == 1
        assert history._messages[0].role == "user"
        assert history._messages[0].content == "Hello"
    
    def test_add_multiple_messages(self):
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        history.add_message("user", "How are you?")
        
        assert len(history) == 3
        assert history._messages[0].content == "Hello"
        assert history._messages[1].content == "Hi"
        assert history._messages[2].content == "How are you?"
    
    def test_max_messages_pruning(self):
        history = MessageHistory(max_messages=2)
        history.add_message("user", "First")
        history.add_message("assistant", "Second")
        history.add_message("user", "Third")
        
        assert len(history) == 2
        assert history._messages[0].content == "Second"
        assert history._messages[1].content == "Third"
    
    def test_clear_history(self):
        history = MessageHistory()
        history.add_message("user", "Test")
        history.add_message("assistant", "Response")
        
        assert len(history) == 2
        history.clear()
        assert len(history) == 0
    
    def test_history_iterator(self):
        history = MessageHistory()
        history.add_message("user", "First")
        history.add_message("assistant", "Second")
        
        iterated_messages = list(history)
        assert len(iterated_messages) == 2
        assert iterated_messages[0].content == "First"
        assert iterated_messages[1].content == "Second"