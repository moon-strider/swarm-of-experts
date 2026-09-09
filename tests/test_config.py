import json

import pytest
from pydantic import ValidationError

from swarm_of_experts.config import Generator, Limits, Provider, Settings, Swarm, load_settings
from swarm_of_experts.models import ChatRequest


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",
        "ftp://127.0.0.1/v1",
        "https://u:p@example.com",
        "https://example.com/?key=secret",
        "https://example.com/#fragment",
        "relative",
        "http://localhost.evil/v1",
        "http://127.0.0.1:wrong/v1",
    ],
)
def test_endpoint_rejects_insecure_or_ambiguous_urls(url):
    with pytest.raises(ValueError):
        Provider(base_url=url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/v1", "http://[::1]/v1", "https://example.com/api/v1/"]
)
def test_endpoint_accepts_supported_urls(url):
    assert Provider(base_url=url).base_url == url.rstrip("/")


def test_keys_are_explicit_and_not_in_configuration_repr():
    provider = Provider(base_url="https://example.com", api_key_env="TEST_KEY")
    with pytest.raises(ValueError):
        provider.key({})
    assert provider.key({"TEST_KEY": "private-key"}) == "private-key"
    assert Provider(base_url="http://localhost/v1").key({}) == "local"
    config = load_settings(
        env={
            "LLM_BASE_URL": "http://127.0.0.1:8080/v1",
            "LLM_MODEL": "small",
            "SWARM_API_KEY": "private-key",
        }
    )
    assert "private-key" not in repr(config)
    assert "private-key" not in config.model_dump_json()


def test_local_defaults_and_no_automatic_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SWARM_API_KEY=not-loaded\n")
    config = load_settings(env={"LLM_BASE_URL": "http://127.0.0.1:8080/v1", "LLM_MODEL": "small"})
    assert config.default_swarm == "local-single"
    assert set(config.providers) == {"local"}
    assert set(config.swarms) == {"local-single", "local-swarm"}
    assert len(config.swarms["local-swarm"].generators) == 3
    assert config.api_key is None
    with pytest.raises(ValueError):
        load_settings(env={"LLM_BASE_URL": "http://127.0.0.1"})


@pytest.mark.parametrize("env", [{}, {"LLM_MODEL": "small"}])
def test_endpoint_and_model_must_be_explicit(env):
    with pytest.raises(ValueError, match="LLM_BASE_URL and LLM_MODEL"):
        load_settings(env=env)


def test_file_configuration_is_validated(tmp_path, settings):
    path = tmp_path / "config.json"
    path.write_text(settings.model_dump_json())
    assert load_settings(path, {}) == settings
    data = settings.model_dump()
    data["default_swarm"] = "missing"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_settings(path, {})
    path.write_text('{"api_key":"secret"}')
    with pytest.raises(ValueError, match="SWARM_API_KEY"):
        load_settings(path, {})
    path.write_text("[]")
    with pytest.raises(ValueError):
        load_settings(path, {})
    path.write_text(" " * 1_048_577)
    with pytest.raises(ValueError):
        load_settings(path, {})


def test_topology_and_references(settings):
    gen = Generator(provider="local", model="x")
    with pytest.raises(ValueError, match="merger"):
        Swarm(generators=(gen, gen))
    with pytest.raises(ValueError, match="min_success"):
        Swarm(generators=(gen,), min_success=2)
    data = settings.model_dump()
    data["swarms"]["basic"]["generators"][0]["provider"] = "missing"
    with pytest.raises(ValueError, match="Unknown provider"):
        Settings.model_validate(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests": True},
        {"provider_calls": 0},
        {"timeout_seconds": float("nan")},
        {"response_bytes": 999},
        {"output_tokens": 50000},
    ],
)
def test_limits_are_strict(kwargs):
    with pytest.raises(ValidationError):
        Limits(**kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"n": 2},
        {"temperature": float("nan")},
        {"max_tokens": True},
        {"unexpected": 1},
        {"user": "unused"},
        {"messages": []},
        {"max_tokens": 5, "max_completion_tokens": 7},
        {"stop": ["a"] * 5},
        {"stop": ""},
        {"messages": [{"role": "tool", "content": "result"}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}]},
    ],
)
def test_request_rejects_unsupported_options(changes):
    with pytest.raises(ValueError):
        ChatRequest.model_validate(
            {"model": "basic", "messages": [{"role": "user", "content": "hi"}], **changes}
        )


def test_tool_conversation_and_text_parts():
    request = ChatRequest.model_validate(
        {
            "model": "basic",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "read"}]},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call", "function": {"name": "Read", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call", "content": "result"},
            ],
        }
    )
    assert request.messages[0].wire()["content"] == "read"
    assert request.messages[-1].wire()["tool_call_id"] == "call"
