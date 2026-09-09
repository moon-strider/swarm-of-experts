import httpx
import pytest

from swarm_of_experts.backend import Backend
from swarm_of_experts.config import Settings


@pytest.fixture
def settings():
    gen = {"provider": "local", "model": "test-model", "max_tokens": 64}
    return Settings.model_validate(
        {
            "providers": {"local": {"base_url": "http://127.0.0.1:9000/v1"}},
            "swarms": {
                "basic": {"generators": [gen]},
                "pair": {"generators": [gen, gen], "merger": gen},
            },
            "limits": {"timeout_seconds": 2},
        }
    )


def response(text="answer", tokens=3, finish="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": finish}
            ],
            "usage": {"prompt_tokens": tokens, "completion_tokens": 2, "total_tokens": tokens + 2},
        },
    )


@pytest.fixture
def make_backend(settings):
    clients = []

    def make(handler, config=None):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return Backend(config or settings, client)

    return make
