import json
import sys

import httpx
import pytest

from swarm_of_experts import cli
from swarm_of_experts.runtime import Runtime

from .conftest import response
from .test_runtime import Events, sse


@pytest.fixture
def invoke(monkeypatch, settings, make_backend):
    monkeypatch.setattr(cli, "load_settings", lambda path: settings)

    async def handler(req):
        return (
            httpx.Response(200, stream=Events(sse()))
            if json.loads(req.content)["stream"]
            else response()
        )

    monkeypatch.setattr(cli, "Runtime", lambda config: Runtime(config, make_backend(handler)))

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["swarm-of-experts", *args])
        cli.main()

    return run


def test_check_help_and_version(invoke, capsys):
    invoke("--check")
    assert json.loads(capsys.readouterr().out)["providers_configured"]["local"]
    invoke()
    assert "serve" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        invoke("--version")
    assert result.value.code == 0
    assert "0.2.0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "options, expected", [((), "answer"), (("--json",), '"usage"'), (("--stream",), "hello")]
)
def test_one_shot_chat(invoke, capsys, options, expected):
    invoke("chat", "--prompt", "hello", *options)
    assert expected in capsys.readouterr().out


def test_interactive_history_clear_and_exit(invoke, monkeypatch, capsys):
    prompts = iter(["first", "/clear", "second", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))
    invoke("chat")
    assert capsys.readouterr().out.count("answer") == 2


def test_nonloopback_requires_auth(invoke, capsys):
    with pytest.raises(SystemExit) as result:
        invoke("serve", "--host", "0.0.0.0")
    assert result.value.code == 1
    assert "SWARM_API_KEY" in capsys.readouterr().err


def test_serve_uses_configured_app(invoke, monkeypatch):
    captured = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: captured.append(kwargs))
    invoke("serve", "--port", "8765")
    assert captured[0]["host"] == "127.0.0.1" and captured[0]["port"] == 8765


def test_invalid_configuration_and_interruption(invoke, monkeypatch, capsys):
    def invalid(path):
        raise ValueError("invalid setup")

    monkeypatch.setattr(cli, "load_settings", invalid)
    invoke()
    assert "serve" in capsys.readouterr().out
    with pytest.raises(SystemExit) as result:
        invoke("--check")
    assert result.value.code == 1
    assert "invalid setup" in capsys.readouterr().err
    monkeypatch.setattr(
        cli, "load_settings", lambda path: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(SystemExit) as result:
        invoke("--check")
    assert result.value.code == 130
