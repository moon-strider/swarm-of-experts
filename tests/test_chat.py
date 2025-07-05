import pytest
import os
from unittest.mock import Mock, MagicMock
from src.core.chat import ChatSession
from src.core.messages import ChatMessage
from src.providers.base import Message, LLMProvider


class MockProvider(LLMProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generate_calls = []
        self.stream_calls = []
    
    def generate(self, messages):
        self.generate_calls.append(messages)
        return f"Response to: {messages[-1].content}"
    
    def stream(self, messages):
        self.stream_calls.append(messages)
        response = f"Response to: {messages[-1].content}"
        for char in response:
            yield char
    
    def validate_model(self):
        return True
    
    @property
    def available_models(self):
        return ["mock-model"]


class TestChatSession:
    @pytest.fixture
    def mock_provider(self):
        return MockProvider(api_key="test", model="mock-model")
    
    @pytest.fixture
    def chat_session(self, mock_provider):
        return ChatSession(mock_provider)
    
    def test_send_message(self, chat_session):
        response = chat_session.send_message("Hello")
        
        assert response == "Response to: Hello"
        assert len(chat_session.history) == 2
        assert chat_session.history._messages[0].role == "user"
        assert chat_session.history._messages[0].content == "Hello"
        assert chat_session.history._messages[1].role == "assistant"
        assert chat_session.history._messages[1].content == response
    
    def test_stream_message(self, chat_session):
        chunks = list(chat_session.stream_message("Hello"))
        
        expected_response = "Response to: Hello"
        assert "".join(chunks) == expected_response
        assert len(chat_session.history) == 2
        assert chat_session.history._messages[0].content == "Hello"
        assert chat_session.history._messages[1].content == expected_response
    
    def test_multiple_messages(self, chat_session):
        chat_session.send_message("First message")
        chat_session.send_message("Second message")
        
        assert len(chat_session.history) == 4
        assert chat_session.history._messages[0].content == "First message"
        assert chat_session.history._messages[2].content == "Second message"
    
    def test_clear_history(self, chat_session):
        chat_session.send_message("Test message")
        assert len(chat_session.history) == 2
        
        chat_session.clear_history()
        assert len(chat_session.history) == 0
    
    def test_provider_receives_full_history(self, chat_session, mock_provider):
        chat_session.send_message("First")
        chat_session.send_message("Second")
        
        assert len(mock_provider.generate_calls) == 2
        
        first_call = mock_provider.generate_calls[0]
        assert len(first_call) == 1
        assert first_call[0].content == "First"
        
        second_call = mock_provider.generate_calls[1]
        assert len(second_call) == 3
        assert second_call[0].content == "First"
        assert second_call[1].content == "Response to: First"
        assert second_call[2].content == "Second"
    
    def test_streaming_chunks_accumulation(self, chat_session):
        chunks = []
        for chunk in chat_session.stream_message("Test"):
            chunks.append(chunk)
        
        assert len(chunks) > 1
        assert "".join(chunks) == "Response to: Test"
    
    def test_history_with_max_messages(self):
        provider = MockProvider(api_key="test", model="mock-model")
        session = ChatSession(provider, max_history=4)
        
        session.send_message("One")
        session.send_message("Two")
        session.send_message("Three")
        
        assert session.get_history_length() == 4
        assert session.history._messages[0].content == "Two"
        assert session.history._messages[1].content == "Response to: Two"
    
    def test_messages_converted_correctly(self, chat_session, mock_provider):
        chat_session.send_message("Hello")
        
        provider_messages = mock_provider.generate_calls[0]
        assert all(isinstance(msg, Message) for msg in provider_messages)
        assert provider_messages[0].role == "user"
        assert provider_messages[0].content == "Hello"