"""Installed OpenClaw + real Swarm HTTP; deterministic upstream, no LLM claims."""

import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import uvicorn

from swarm_of_experts.api import create_app
from swarm_of_experts.config import load_settings


@pytest.mark.skipif(
    not os.environ.get("OPENCLAW_BIN"), reason="set OPENCLAW_BIN for installed CLI integration"
)
@pytest.mark.parametrize("mode", ["text", "read"])
def test_installed_openclaw_through_real_swarm(tmp_path, mode):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            history = [m for m in body["messages"] if m["role"] == "tool"]
            if body.get("tools") and not history:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_read",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"sentinel.txt"}'},
                        }
                    ],
                }
                finish = "tool_calls"
            else:
                found = re.search(r"read-[0-9a-f]{32}", json.dumps(history))
                delta = {"role": "assistant", "content": found.group(0) if found else "OPENCLAW_OK"}
                finish = "stop"
            base = {
                "id": "fixture",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "fixture",
            }
            chunks = [
                {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                },
            ]
            data = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    upstream_thread = threading.Thread(target=upstream.serve_forever)
    upstream_thread.start()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    settings = load_settings(
        env={"LLM_BASE_URL": f"http://127.0.0.1:{upstream.server_port}/v1", "LLM_MODEL": "fixture"}
    )
    server = uvicorn.Server(uvicorn.Config(create_app(settings), log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]))
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        expected = "read-" + uuid.uuid4().hex if mode == "read" else "OPENCLAW_OK"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "sentinel.txt").write_text(expected + "\n")
        model = {
            "id": "local-single",
            "name": "Fixture",
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 16384,
            "maxTokens": 512,
        }
        config = {
            "agents": {
                "defaults": {"model": {"primary": "test/local-single"}, "skipBootstrap": True}
            },
            "models": {
                "providers": {
                    "test": {
                        "baseUrl": f"http://127.0.0.1:{sock.getsockname()[1]}/v1",
                        "apiKey": "local",
                        "api": "openai-completions",
                        "models": [model],
                    }
                }
            },
            "tools": {"allow": ["read"], "fs": {"workspaceOnly": True}}
            if mode == "read"
            else {"deny": ["*"]},
            "plugins": {"enabled": False},
        }
        path = tmp_path / "openclaw.json"
        path.write_text(json.dumps(config))
        prompt = (
            "Read sentinel.txt using the read tool; reply only its exact contents."
            if mode == "read"
            else "Reply exactly OPENCLAW_OK."
        )
        proc = subprocess.run(
            [
                os.environ["OPENCLAW_BIN"],
                "agent",
                "exec",
                "--config",
                str(path),
                "--cwd",
                str(workspace),
                "--message-file",
                "-",
                "--thinking",
                "off",
                "--code-mode",
                "direct",
                "--local-model-lean",
                "--timeout",
                "60",
                "--json",
            ],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        envelope = json.loads(proc.stdout)
        assert envelope["ok"] and envelope["final"] == expected
        if mode == "read":
            assert envelope["toolSummary"] == {"calls": 1, "tools": ["read"], "failures": 0}
            assert len(requests) == 2
            assert expected not in json.dumps(requests[0])
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join()
        assert not thread.is_alive()
