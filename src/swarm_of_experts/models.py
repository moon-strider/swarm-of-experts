"""The supported Chat Completions contract. Unsupported fields fail explicitly."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .config import StrictModel


class FunctionCall(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    arguments: str = Field(max_length=131072)


class ToolCall(StrictModel):
    id: str = Field(min_length=1, max_length=256)
    type: Literal["function"] = "function"
    function: FunctionCall


class TextPart(StrictModel):
    type: Literal["text"]
    text: str = Field(max_length=262144)


class Message(StrictModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    name: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    tool_call_id: str | None = Field(default=None, max_length=256)
    tool_calls: list[ToolCall] | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool results require tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool calls")
        if self.tool_call_id and self.role != "tool":
            raise ValueError("tool_call_id requires role=tool")
        if self.content is None and not self.tool_calls:
            raise ValueError("Message requires content or tool_calls")
        if isinstance(self.content, str) and len(self.content) > 262144:
            raise ValueError("Message is too long")
        return self

    def wire(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        if isinstance(self.content, list):
            data["content"] = "\n".join(p.text for p in self.content)
        return data


class FunctionSpec(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=32768)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    strict: bool | None = None


class ToolSpec(StrictModel):
    type: Literal["function"] = "function"
    function: FunctionSpec


class StreamOptions(StrictModel):
    include_usage: bool = False


class ChatRequest(StrictModel):
    model: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    messages: list[Message] = Field(min_length=1, max_length=256)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32768, strict=True)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768, strict=True)
    top_p: float | None = Field(default=None, gt=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop: str | list[str] | None = None
    seed: int | None = Field(default=None, strict=True)
    n: Literal[1] = 1
    tools: list[ToolSpec] | None = Field(default=None, max_length=64)
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def options_valid(self) -> Self:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Choose one output token limit")
        if self.stop is not None:
            values = [self.stop] if isinstance(self.stop, str) else self.stop
            if not 1 <= len(values) <= 4 or any(not v or len(v) > 256 for v in values):
                raise ValueError("Use up to four nonempty stop strings of at most 256 characters")
        return self

    def options(self) -> dict[str, Any]:
        return self.model_dump(
            exclude_none=True,
            exclude={"model", "messages", "stream", "stream_options", "n"},
        )
