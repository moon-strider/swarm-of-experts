import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.core.messages import MessageHistory
from src.core.executor import ParallelExecutor, GeneratorResponse
from src.core.merger import ResponseMerger
from src.config.swarm_configs import SwarmConfig, GeneratorConfig


class TestMessageHistory:
    def test_add_and_get(self):
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        messages = history.get_messages()
        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "Hello"

    def test_system_message(self):
        history = MessageHistory()
        history.add_message("system", "You are helpful")
        messages = history.get_messages()
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)

    def test_unknown_role_raises(self):
        history = MessageHistory()
        with pytest.raises(ValueError, match="Unknown role"):
            history.add_message("invalid_role", "content")

    def test_max_messages_truncation(self):
        history = MessageHistory(max_messages=3)
        for i in range(5):
            history.add_message("user", f"Message {i}")
        messages = history.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
        assert messages[2].content == "Message 4"

    def test_clear(self):
        history = MessageHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi")
        history.clear()
        assert len(history.get_messages()) == 0

    def test_get_messages_returns_copy(self):
        history = MessageHistory()
        history.add_message("user", "Hello")
        msgs = history.get_messages()
        msgs.append(HumanMessage(content="extra"))
        assert len(history.get_messages()) == 1


class TestGeneratorResponse:
    def test_successful_response(self):
        config = GeneratorConfig(provider="openai", model="gpt-4.1")
        resp = GeneratorResponse(
            generator_config=config, content="Hello", elapsed_time=1.5
        )
        assert resp.error is None
        assert resp.content == "Hello"

    def test_error_response(self):
        config = GeneratorConfig(provider="openai", model="gpt-4.1")
        resp = GeneratorResponse(
            generator_config=config, content="", elapsed_time=0.1, error="API failed"
        )
        assert resp.error == "API failed"


class TestResponseMerger:
    def _make_response(self, content, provider="openai", model="gpt-4.1", error=None):
        return GeneratorResponse(
            generator_config=GeneratorConfig(provider=provider, model=model),
            content=content,
            elapsed_time=1.0,
            error=error,
        )

    @pytest.mark.asyncio
    async def test_single_valid_response_passthrough(self, mock_provider, multi_generator_config):
        merger = ResponseMerger(mock_provider, multi_generator_config)
        responses = [
            self._make_response("Only valid", error=None),
            self._make_response("", error="failed"),
        ]
        result = await merger.merge_responses("query", responses)
        assert result == "Only valid"

    @pytest.mark.asyncio
    async def test_no_valid_responses_raises(self, mock_provider, multi_generator_config):
        merger = ResponseMerger(mock_provider, multi_generator_config)
        responses = [
            self._make_response("", error="fail1"),
            self._make_response("", error="fail2"),
        ]
        with pytest.raises(ValueError, match="All generators failed"):
            await merger.merge_responses("query", responses)

    @pytest.mark.asyncio
    async def test_multiple_responses_calls_provider(self, multi_generator_config):
        provider = MagicMock()
        provider.generate = AsyncMock(return_value="Merged result")
        merger = ResponseMerger(provider, multi_generator_config)

        responses = [
            self._make_response("Response A"),
            self._make_response("Response B"),
        ]
        result = await merger.merge_responses("test query", responses)
        assert result == "Merged result"
        provider.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_merge_with_history(self, multi_generator_config):
        provider = MagicMock()
        provider.generate = AsyncMock(return_value="Merged")
        merger = ResponseMerger(provider, multi_generator_config)

        history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
            HumanMessage(content="New question"),
        ]
        responses = [
            self._make_response("A"),
            self._make_response("B"),
        ]
        result = await merger.merge_responses("New question", responses, history=history)
        assert result == "Merged"
        call_messages = provider.generate.call_args[0][0]
        assert len(call_messages) == 3

    @pytest.mark.asyncio
    async def test_fallback_on_merge_failure(self, multi_generator_config):
        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=Exception("Merge failed"))
        merger = ResponseMerger(provider, multi_generator_config)

        responses = [
            self._make_response("Short"),
            self._make_response("This is a longer response"),
        ]
        result = await merger.merge_responses("query", responses)
        assert result == "This is a longer response"

    def test_format_responses_xml(self, mock_provider, multi_generator_config):
        merger = ResponseMerger(mock_provider, multi_generator_config)
        responses = [
            self._make_response("Content A", provider="openai", model="gpt-4.1"),
            self._make_response("Content B", provider="groq", model="llama-70b"),
        ]
        xml = merger._format_responses_xml(responses)
        assert "<response1" in xml
        assert "<response2" in xml
        assert 'provider="openai/gpt-4.1"' in xml
        assert 'provider="groq/llama-70b"' in xml
        assert "Content A" in xml
        assert "Content B" in xml

    def test_select_best_fallback(self, mock_provider, multi_generator_config):
        merger = ResponseMerger(mock_provider, multi_generator_config)
        responses = [
            self._make_response("Short"),
            self._make_response("Much longer response content here"),
            self._make_response("Medium length"),
        ]
        result = merger._select_best_fallback(responses)
        assert result == "Much longer response content here"

    @pytest.mark.asyncio
    async def test_stream_single_response(self, mock_provider, multi_generator_config):
        merger = ResponseMerger(mock_provider, multi_generator_config)
        responses = [
            self._make_response("Only one"),
            self._make_response("", error="failed"),
        ]
        chunks = []
        async for chunk in merger.stream_merge_responses("query", responses):
            chunks.append(chunk)
        assert chunks == ["Only one"]


class TestParallelExecutor:
    @pytest.fixture(autouse=True)
    def mock_settings(self):
        mock_s = MagicMock()
        mock_s.get_api_key_for_provider.return_value = "fake-api-key"
        with patch("src.core.executor.settings", mock_s):
            yield mock_s

    @pytest.mark.asyncio
    async def test_execute_single_generator(self, single_generator_config):
        mock_factory = MagicMock()
        mock_prov = MagicMock()
        mock_prov.generate = AsyncMock(return_value="Generated content")
        mock_factory.create.return_value = mock_prov

        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Hello")]
        results = await executor.execute_parallel(
            single_generator_config, messages, stream=False
        )
        assert len(results) == 1
        assert results[0].content == "Generated content"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_execute_handles_generator_failure(self, single_generator_config):
        mock_factory = MagicMock()
        mock_prov = MagicMock()
        mock_prov.generate = AsyncMock(side_effect=Exception("API down"))
        mock_factory.create.return_value = mock_prov

        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Hello")]
        results = await executor.execute_parallel(
            single_generator_config, messages, stream=False
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "API down" in results[0].error

    @pytest.mark.asyncio
    async def test_execute_multiple_generators(self, multi_generator_config):
        mock_factory = MagicMock()
        call_count = 0

        def create_mock(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            prov = MagicMock()
            prov.generate = AsyncMock(return_value=f"Response {call_count}")
            return prov

        mock_factory.create = create_mock
        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Test")]
        results = await executor.execute_parallel(
            multi_generator_config, messages, stream=False
        )
        assert len(results) == 3
        successful = [r for r in results if not r.error]
        assert len(successful) == 3

    @pytest.mark.asyncio
    async def test_execute_with_sub_prompts(self, multi_generator_config):
        mock_factory = MagicMock()
        captured_messages = []

        def create_mock(*args, **kwargs):
            prov = MagicMock()
            async def fake_generate(msgs):
                captured_messages.append(msgs[0].content)
                return "Done"
            prov.generate = fake_generate
            return prov

        mock_factory.create = create_mock
        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Original")]
        sub_prompts = {
            "sub_prompt_1": "First task",
            "sub_prompt_2": "Second task",
            "sub_prompt_3": "Third task",
        }
        results = await executor.execute_parallel(
            multi_generator_config, messages, stream=False, sub_prompts=sub_prompts
        )
        assert len(results) == 3
        assert "First task" in captured_messages
        assert "Second task" in captured_messages
        assert "Third task" in captured_messages

    @pytest.mark.asyncio
    async def test_sub_prompt_count_mismatch_raises(self, multi_generator_config):
        mock_factory = MagicMock()
        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Test")]
        sub_prompts = {"sub_prompt_1": "Only one"}
        with pytest.raises(ValueError, match="Mismatch"):
            await executor.execute_parallel(
                multi_generator_config, messages, stream=False, sub_prompts=sub_prompts
            )

    @pytest.mark.asyncio
    async def test_stream_parallel(self, single_generator_config):
        mock_factory = MagicMock()

        async def mock_stream(messages):
            for chunk in ["Hello", " ", "world"]:
                yield chunk

        mock_prov = MagicMock()
        mock_prov.stream = mock_stream
        mock_factory.create.return_value = mock_prov

        executor = ParallelExecutor(mock_factory)
        messages = [HumanMessage(content="Hi")]
        chunks = []
        async for chunk in executor.stream_parallel(single_generator_config, messages):
            chunks.append(chunk)
        assert chunks == ["Hello", " ", "world"]
