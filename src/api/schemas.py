from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ErrorType(str, Enum):
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_MODEL = "invalid_model"
    INVALID_PARAMETER = "invalid_parameter"
    MISSING_PARAMETER = "missing_parameter"
    INTERNAL_ERROR = "internal_error"
    MODEL_NOT_FOUND = "model_not_found"
    UNSUPPORTED_MODEL = "unsupported_model"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1, max_length=1000000)
    name: Optional[str] = Field(None, pattern=r'^[a-zA-Z0-9_-]+$', max_length=64)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError("Name cannot be empty if provided")
            if len(v) > 64:
                raise ValueError("Name cannot exceed 64 characters")
        return v


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(min_length=1, max_length=1000)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=131072)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    n: Optional[int] = Field(default=1, ge=1, le=5)
    user: Optional[str] = Field(None, max_length=256)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = Field(None, ge=-2147483648, le=2147483647)
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        
        if len(v) > 1:
            for i in range(len(v) - 1):
                if v[i].role == v[i + 1].role and v[i].role != Role.SYSTEM:
                    pass
        
        if v[-1].role not in [Role.USER, Role.SYSTEM]:
            raise ValueError("Last message must be from user or system")
        
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if not v or not v.strip():
            raise ValueError("Model cannot be empty")
        return v.strip()
    
    @field_validator('stop')
    @classmethod
    def validate_stop(cls, v):
        if v is not None:
            if isinstance(v, list):
                if len(v) > 4:
                    raise ValueError("Stop sequences cannot exceed 4 entries")
                for item in v:
                    if not isinstance(item, str):
                        raise ValueError("All stop sequences must be strings")
            elif not isinstance(v, str):
                raise ValueError("Stop must be a string or list of strings")
        return v


class ChatCompletionChoiceDelta(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[ChatCompletionChoiceDelta] = None
    finish_reason: Optional[str] = None


class PromptTokensDetails(BaseModel):
    cached_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: Optional[int] = None
    audio_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[PromptTokensDetails] = None
    completion_tokens_details: Optional[CompletionTokensDetails] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Any] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
    object: str = "error"


