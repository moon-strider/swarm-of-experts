import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.api.schemas import (
    ChatCompletionRequest, ChatMessage, Role,
    ErrorType, ErrorCode, ErrorResponse, ErrorDetail,
    ChatCompletionResponse, ChatCompletionChoice,
    Usage, Model, ModelsResponse,
)


class TestSchemas:
    def test_chat_message_valid(self):
        msg = ChatMessage(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_chat_message_empty_content_raises(self):
        with pytest.raises(Exception):
            ChatMessage(role=Role.USER, content="")

    def test_chat_message_whitespace_only_raises(self):
        with pytest.raises(Exception):
            ChatMessage(role=Role.USER, content="   ")

    def test_chat_message_with_name(self):
        msg = ChatMessage(role=Role.USER, content="Hi", name="test_user")
        assert msg.name == "test_user"

    def test_chat_message_invalid_name_pattern(self):
        with pytest.raises(Exception):
            ChatMessage(role=Role.USER, content="Hi", name="invalid name!")

    def test_request_valid(self):
        req = ChatCompletionRequest(
            model="basic",
            messages=[ChatMessage(role=Role.USER, content="Hello")],
        )
        assert req.model == "basic"
        assert req.temperature == 0.7

    def test_request_empty_model_raises(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="",
                messages=[ChatMessage(role=Role.USER, content="Hi")],
            )

    def test_request_temperature_bounds(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="basic",
                messages=[ChatMessage(role=Role.USER, content="Hi")],
                temperature=3.0,
            )

    def test_request_last_message_must_be_user_or_system(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="basic",
                messages=[
                    ChatMessage(role=Role.USER, content="Hi"),
                    ChatMessage(role=Role.ASSISTANT, content="Hello"),
                ],
            )

    def test_request_stop_sequences_max_4(self):
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="basic",
                messages=[ChatMessage(role=Role.USER, content="Hi")],
                stop=["a", "b", "c", "d", "e"],
            )

    def test_error_response_structure(self):
        error = ErrorResponse(
            error=ErrorDetail(
                message="Something failed",
                type=ErrorType.INTERNAL_ERROR.value,
                code=ErrorCode.INTERNAL_ERROR.value,
            )
        )
        assert error.error.message == "Something failed"
        assert error.object == "error"

    def test_model_schema(self):
        model = Model(id="basic", created=1000, owned_by="test")
        assert model.object == "model"

    def test_models_response(self):
        resp = ModelsResponse(data=[
            Model(id="basic", created=1000, owned_by="test"),
        ])
        assert resp.object == "list"
        assert len(resp.data) == 1

    def test_response_schema(self):
        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role=Role.ASSISTANT, content="Hi"),
            finish_reason="stop",
        )
        resp = ChatCompletionResponse(
            id="test-id", created=1000, model="basic",
            choices=[choice],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert resp.object == "chat.completion"
        assert resp.usage.total_tokens == 15


class TestTokenEstimator:
    def test_empty_string(self):
        from src.api.server import TokenEstimator
        assert TokenEstimator.estimate_tokens("") == 0

    def test_normal_text(self):
        from src.api.server import TokenEstimator
        tokens = TokenEstimator.estimate_tokens("Hello world")
        assert tokens > 0

    def test_messages_estimation(self):
        from src.api.server import TokenEstimator
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there, how can I help?"),
        ]
        tokens = TokenEstimator.estimate_messages_tokens(messages)
        assert tokens > 0

    def test_longer_text_more_tokens(self):
        from src.api.server import TokenEstimator
        short = TokenEstimator.estimate_tokens("Hi")
        long = TokenEstimator.estimate_tokens("This is a much longer sentence with many words")
        assert long > short


class TestAPIValidator:
    @pytest.fixture(autouse=True)
    def setup_env(self, fake_env):
        pass

    def test_validate_empty_model_raises(self):
        from src.api.server import APIValidator, APIValidationError
        with pytest.raises(APIValidationError):
            APIValidator.validate_model("")

    def test_validate_unknown_model_raises(self):
        from src.api.server import APIValidator, APIValidationError
        with pytest.raises(APIValidationError, match="not found"):
            APIValidator.validate_model("nonexistent-model")

    def test_validate_valid_model(self):
        from src.api.server import APIValidator
        APIValidator.validate_model("basic")

    def test_validate_request_empty_messages(self):
        from src.api.server import APIValidator, APIValidationError
        req = MagicMock()
        req.model = "basic"
        req.messages = []
        req.stream = False
        req.stream_options = None
        req.n = 1
        with pytest.raises(APIValidationError, match="empty"):
            APIValidator.validate_request(req)

    def test_validate_request_n_greater_than_1(self):
        from src.api.server import APIValidator, APIValidationError
        req = MagicMock()
        req.model = "basic"
        req.messages = [MagicMock()]
        req.stream = False
        req.stream_options = None
        req.n = 3
        with pytest.raises(APIValidationError, match="n > 1"):
            APIValidator.validate_request(req)


class TestErrorResponses:
    def test_create_error_response(self):
        from src.api.server import create_error_response, APIValidationError
        err = APIValidationError(
            "Test error", ErrorType.INVALID_REQUEST_ERROR,
            ErrorCode.INVALID_REQUEST, "model",
        )
        resp = create_error_response(err)
        assert resp.error.message == "Test error"
        assert resp.error.param == "model"

    def test_create_generic_error_response(self):
        from src.api.server import create_generic_error_response
        resp = create_generic_error_response("Something broke")
        assert resp.error.message == "Something broke"
        assert resp.error.type == ErrorType.INTERNAL_ERROR.value


class TestAPIEndpoints:
    @pytest.fixture
    def client(self, fake_env):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api import server as server_module

        test_app = FastAPI()
        for route in server_module.app.routes:
            test_app.routes.append(route)
        for handler_tuple in server_module.app.exception_handlers.items():
            test_app.add_exception_handler(handler_tuple[0], handler_tuple[1])
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "session_stats" in data

    def test_list_models(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        model_ids = [m["id"] for m in data["data"]]
        assert "basic" in model_ids
        assert "groq-swarm" in model_ids

    def test_session_stats(self, client):
        response = client.get("/v1/sessions/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_sessions" in data

    def test_chat_completions_invalid_model(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert response.status_code in (400, 422)

    def test_chat_completions_empty_messages(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "basic",
            "messages": [],
        })
        assert response.status_code in (400, 422)

    def test_chat_completions_missing_content(self, client):
        response = client.post("/v1/chat/completions", json={
            "model": "basic",
            "messages": [{"role": "user"}],
        })
        assert response.status_code == 400

    @patch("src.api.server.server_instance")
    def test_chat_completions_non_streaming(self, mock_server, client):
        mock_session = MagicMock()
        mock_session.history = MagicMock()
        mock_session.history.get_messages.return_value = []
        mock_session.history.add_message = MagicMock()
        mock_session.send_message = AsyncMock(return_value="Test response")

        mock_server.session_manager = MagicMock()
        mock_server.session_manager.get_or_create_session.return_value = mock_session

        response = client.post("/v1/chat/completions", json={
            "model": "basic",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    @patch("src.api.server.server_instance")
    def test_chat_completions_streaming(self, mock_server, client):
        mock_session = MagicMock()
        mock_session.history = MagicMock()
        mock_session.history.get_messages.return_value = []
        mock_session.history.add_message = MagicMock()

        async def mock_stream(msg):
            for chunk in ["Hello", " world"]:
                yield chunk

        mock_session.stream_message = mock_stream

        mock_server.session_manager = MagicMock()
        mock_server.session_manager.get_or_create_session.return_value = mock_session

        response = client.post("/v1/chat/completions", json={
            "model": "basic",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        assert len(data_lines) >= 2
