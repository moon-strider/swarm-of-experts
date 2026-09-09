"""Explicit, validated configuration; no environment loading at import time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Provider(StrictModel):
    base_url: str
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    stream_usage: bool = True

    @field_validator("base_url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        u = urlsplit(value)
        if not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError("Use an endpoint URL without credentials, query or fragment")
        if u.scheme != "https" and not (
            u.scheme == "http" and u.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("HTTPS is required except for explicit loopback endpoints")
        _ = u.port
        return value.rstrip("/")

    def key(self, env: dict[str, str] | None = None) -> str:
        source = os.environ if env is None else env
        key = source.get(self.api_key_env, "") if self.api_key_env else ""
        if key:
            return key
        if urlsplit(self.base_url).hostname in {"127.0.0.1", "localhost", "::1"}:
            return "local"
        raise ValueError(f"Missing provider credential in {self.api_key_env or 'api_key_env'}")


class Generator(StrictModel):
    provider: Name
    model: str = Field(min_length=1, max_length=256)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=32768, strict=True)


class Swarm(StrictModel):
    generators: tuple[Generator, ...] = Field(min_length=1, max_length=16)
    merger: Generator | None = None
    taskmaster: Generator | None = None
    min_success: int = Field(default=1, ge=1, le=16, strict=True)
    merger_failure: Literal["error", "first"] = "error"

    @model_validator(mode="after")
    def topology(self) -> Self:
        if self.min_success > len(self.generators):
            raise ValueError("min_success exceeds generator count")
        if len(self.generators) > 1 and self.merger is None:
            raise ValueError("Multiple generators require a merger")
        return self

    @property
    def single(self) -> bool:
        return len(self.generators) == 1 and self.taskmaster is None and self.merger is None


class Limits(StrictModel):
    requests: int = Field(default=8, ge=1, le=128, strict=True)
    provider_calls: int = Field(default=8, ge=1, le=128, strict=True)
    timeout_seconds: float = Field(default=300, gt=0, le=3600)
    request_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216, strict=True)
    response_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216, strict=True)
    output_tokens: int = Field(default=4096, ge=1, le=32768, strict=True)


class Settings(StrictModel):
    providers: dict[Name, Provider] = Field(min_length=1, max_length=64)
    swarms: dict[Name, Swarm] = Field(min_length=1, max_length=128)
    default_swarm: Name = "basic"
    limits: Limits = Field(default_factory=Limits)
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def references(self) -> Self:
        if self.default_swarm not in self.swarms:
            raise ValueError("default_swarm is not configured")
        for swarm in self.swarms.values():
            for gen in (*swarm.generators, swarm.merger, swarm.taskmaster):
                if gen is not None and gen.provider not in self.providers:
                    raise ValueError(f"Unknown provider: {gen.provider}")
                if gen is not None and gen.max_tokens > self.limits.output_tokens:
                    raise ValueError("Generator max_tokens exceeds limits.output_tokens")
        return self


def defaults(env: dict[str, str]) -> dict:
    endpoints = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta/openai",
        "groq": "https://api.groq.com/openai/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }
    providers = {
        name: {"base_url": url, "api_key_env": f"{name.upper()}_API_KEY"}
        for name, url in endpoints.items()
    }

    def gen(provider: str, model: str, temp: float = 0.7) -> dict:
        return {"provider": provider, "model": model, "temperature": temp}

    basic = gen("openai", env.get("DEFAULT_MODEL", "gpt-4.1"))
    mini = gen("openai", "gpt-4.1-mini")
    groq = gen("groq", "deepseek-r1-distill-llama-70b")
    merger = gen("groq", "moonshotai/kimi-k2-instruct", 0.3)
    swarms = {
        "basic": {"generators": [basic]},
        "swarm-lite": {
            "generators": [dict(mini, temperature=t) for t in [0.3, 0.5, 0.7]],
            "merger": mini,
        },
        "groq-single": {"generators": [groq]},
        "groq-swarm": {
            "generators": [dict(groq, temperature=t) for t in [0.3, 0.5, 0.7]],
            "merger": merger,
        },
        "groq-taskmaster": {"generators": [groq] * 3, "merger": merger, "taskmaster": merger},
    }
    default_swarm = "basic"
    if url := env.get("LLM_BASE_URL"):
        if not env.get("LLM_MODEL"):
            raise ValueError("LLM_MODEL is required with LLM_BASE_URL")
        providers["local"] = {"base_url": url, "api_key_env": "LLM_API_KEY"}
        local = gen("local", env["LLM_MODEL"])
        swarms["local-single"] = {"generators": [local]}
        swarms["local-swarm"] = {
            "generators": [dict(local, temperature=t) for t in [0.3, 0.5, 0.7]],
            "merger": local,
        }
        default_swarm = "local-single"
    return {
        "providers": providers,
        "swarms": swarms,
        "default_swarm": env.get("DEFAULT_SWARM", default_swarm),
    }


def load_settings(path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    source = dict(os.environ) if env is None else env
    if path is None and source.get("SWARM_CONFIG"):
        path = Path(source["SWARM_CONFIG"])
    if path is None:
        data = defaults(source)
    else:
        if path.stat().st_size > 1_048_576:
            raise ValueError("Configuration exceeds one MiB")
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")
    if "api_key" in data:
        raise ValueError("Use SWARM_API_KEY in the environment, not inline credentials")
    return Settings.model_validate({**data, "api_key": source.get("SWARM_API_KEY") or None})
