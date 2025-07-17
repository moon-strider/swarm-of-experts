import time
import uuid
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from contextlib import asynccontextmanager
import json
import logging
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

from ..config.settings import Settings
from ..config.swarm_configs import SwarmConfig, get_swarm_config, get_all_swarm_configs
from ..core.chat import ChatSession
from ..providers.base import Message as CoreMessage
from ..providers.factory import ProviderFactory
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceDelta,
    ChatMessage,
    ModelsResponse,
    Model,
    ErrorResponse,
    ErrorDetail,
    ErrorType,
    ErrorCode,
    ValidationErrorDetail,
    Role,
    StreamOptions,
    Usage,
    PromptTokensDetails,
    CompletionTokensDetails
)

logger = logging.getLogger(__name__)


class APIValidationError(Exception):
    def __init__(self, message: str, error_type: ErrorType, error_code: ErrorCode, param: str = None):
        self.message = message
        self.error_type = error_type
        self.error_code = error_code
        self.param = param
        super().__init__(message)


class APIValidator:
    @staticmethod
    def validate_model(model: str) -> None:
        if not model or not model.strip():
            raise APIValidationError(
                "Model cannot be empty",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.MISSING_PARAMETER,
                "model"
            )
        
        try:
            swarm_config = get_swarm_config(model)
        except ValueError as e:
            raise APIValidationError(
                f"Model '{model}' not found. Available models: {list(get_available_models())}",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.MODEL_NOT_FOUND,
                "model"
            )
        
        if not swarm_config.generators:
            raise APIValidationError(
                f"Model '{model}' has no generators configured",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.INVALID_MODEL,
                "model"
            )
    
    
    @staticmethod
    def validate_providers(model: str) -> None:
        try:
            swarm_config = get_swarm_config(model)
        except ValueError as e:
            raise APIValidationError(
                f"Model '{model}' not found",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.MODEL_NOT_FOUND,
                "model"
            )
        
        settings = Settings()
        
        for generator in swarm_config.generators:
            api_key = settings.get_api_key_for_provider(generator.provider)
            if not api_key:
                raise APIValidationError(
                    f"API key not configured for provider '{generator.provider}'",
                    ErrorType.AUTHENTICATION_ERROR,
                    ErrorCode.INVALID_REQUEST,
                    "model"
                )
        
        if swarm_config.merger:
            api_key = settings.get_api_key_for_provider(swarm_config.merger.provider)
            if not api_key:
                raise APIValidationError(
                    f"API key not configured for merger provider '{swarm_config.merger.provider}'",
                    ErrorType.AUTHENTICATION_ERROR,
                    ErrorCode.INVALID_REQUEST,
                    "model"
                )
        
        if swarm_config.has_taskmaster:
            api_key = settings.get_api_key_for_provider(swarm_config.taskmaster.provider)
            if not api_key:
                raise APIValidationError(
                    f"API key not configured for taskmaster provider '{swarm_config.taskmaster.provider}'",
                    ErrorType.AUTHENTICATION_ERROR,
                    ErrorCode.INVALID_REQUEST,
                    "model"
                )
    
    @staticmethod
    def validate_request(request: ChatCompletionRequest) -> None:
        APIValidator.validate_model(request.model)
        
        APIValidator.validate_providers(request.model)
        if not request.messages:
            raise APIValidationError(
                "Messages cannot be empty",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.MISSING_PARAMETER,
                "messages"
            )
        
        if len(request.messages) > 1000:
            raise APIValidationError(
                "Too many messages. Maximum allowed: 1000",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.INVALID_PARAMETER,
                "messages"
            )
        
        if request.stream and request.stream_options:
            if request.stream_options.include_usage and request.n > 1:
                raise APIValidationError(
                    "Usage tracking is not supported with n > 1 in streaming mode",
                    ErrorType.INVALID_REQUEST_ERROR,
                    ErrorCode.INVALID_PARAMETER,
                    "stream_options"
                )
        
        if request.n > 1:
            raise APIValidationError(
                "Multiple completions (n > 1) are not supported in swarm mode",
                ErrorType.INVALID_REQUEST_ERROR,
                ErrorCode.INVALID_PARAMETER,
                "n"
            )


def get_available_models() -> List[str]:
    from ..config.swarm_configs import SWARM_CONFIGS
    return list(SWARM_CONFIGS.keys())


def create_error_response(error: APIValidationError) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            message=error.message,
            type=error.error_type.value,
            code=error.error_code.value,
            param=error.param
        )
    )


def create_generic_error_response(message: str, error_type: ErrorType = ErrorType.INTERNAL_ERROR, 
                                 error_code: ErrorCode = ErrorCode.INTERNAL_ERROR) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type.value,
            code=error_code.value
        )
    )


@dataclass
class SessionInfo:
    session_id: str
    chat_session: 'ChatSession'
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    def touch(self):
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return datetime.now() - self.last_accessed > timedelta(minutes=timeout_minutes)


class TokenEstimator:
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        
        words = text.split()
        token_estimate = len(words) * 1.3
        punct_chars = sum(1 for char in text if char in ".,!?;:()[]{}\"'-")
        token_estimate += punct_chars * 0.3
        whitespace_chars = sum(1 for char in text if char in " \t\n\r")
        token_estimate += whitespace_chars * 0.1
        
        return int(token_estimate)
    
    @staticmethod
    def estimate_messages_tokens(messages: List[CoreMessage]) -> int:
        total_tokens = 0
        
        for message in messages:
            content_tokens = TokenEstimator.estimate_tokens(message.content)
            role_tokens = TokenEstimator.estimate_tokens(message.role)
            structure_overhead = 3
            total_tokens += content_tokens + role_tokens + structure_overhead
        
        chat_overhead = 3
        return total_tokens + chat_overhead


class SessionManager:
    
    def __init__(self, session_timeout_minutes: int = 30, cleanup_interval_minutes: int = 5):
        self.sessions: Dict[str, SessionInfo] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes
        self._lock = threading.RLock()
        self._cleanup_task = None
        self._shutdown_event = threading.Event()
        self.settings = Settings()
        self.provider_factory = ProviderFactory()
        
    def _generate_session_key(self, messages: List[Dict[str, Any]], model: str) -> str:
        key_data = {
            'model': model,
            'conversation': json.dumps(messages, sort_keys=True)
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def get_or_create_session(self, messages: List[Dict[str, Any]], model: str) -> 'ChatSession':
        session_key = self._generate_session_key(messages, model)
        
        with self._lock:
            if session_key in self.sessions:
                session_info = self.sessions[session_key]
                if not session_info.is_expired(self.session_timeout_minutes):
                    session_info.touch()
                    logger.debug(f"Reusing existing session {session_key}, access count: {session_info.access_count}")
                    return session_info.chat_session
                else:
                    logger.debug(f"Session {session_key} expired, cleaning up")
                    self._cleanup_session(session_key)
            
            from ..config.swarm_configs import get_swarm_config
            try:
                swarm_config = get_swarm_config(model)
            except ValueError as e:
                logger.error(f"Failed to get swarm config for model {model}: {e}")
                raise APIValidationError(
                    f"Unknown swarm config: {model}",
                    ErrorType.INVALID_REQUEST_ERROR,
                    ErrorCode.MODEL_NOT_FOUND,
                    "model"
                )
            
            try:
                chat_session = ChatSession(
                    swarm_config=swarm_config
                )
            except Exception as e:
                logger.error(f"Failed to create chat session for model {model}: {e}")
                raise APIValidationError(
                    f"Failed to create session for model {model}: {str(e)}",
                    ErrorType.INTERNAL_ERROR,
                    ErrorCode.INTERNAL_ERROR,
                    "model"
                )
            
            session_info = SessionInfo(
                session_id=session_key,
                chat_session=chat_session,
                created_at=datetime.now(),
                last_accessed=datetime.now()
            )
            
            self.sessions[session_key] = session_info
            logger.info(f"Created new session {session_key} for model {model}")
            return chat_session
    
    def _cleanup_session(self, session_key: str):
        if session_key in self.sessions:
            session_info = self.sessions[session_key]
            try:
                session_info.chat_session.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up session {session_key}: {e}")
            finally:
                del self.sessions[session_key]
                logger.debug(f"Cleaned up session {session_key}")
    
    def _cleanup_expired_sessions(self):
        with self._lock:
            expired_keys = []
            for session_key, session_info in self.sessions.items():
                if session_info.is_expired(self.session_timeout_minutes):
                    expired_keys.append(session_key)
            
            for session_key in expired_keys:
                self._cleanup_session(session_key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired sessions")
    
    def _background_cleanup(self):
        while not self._shutdown_event.is_set():
            try:
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error during background cleanup: {e}")
            
            self._shutdown_event.wait(timeout=self.cleanup_interval_minutes * 300)
    
    def start_cleanup_task(self):
        if self._cleanup_task is None or not self._cleanup_task.is_alive():
            self._cleanup_task = threading.Thread(target=self._background_cleanup, daemon=True)
            self._cleanup_task.start()
            logger.info("Started background session cleanup task")
    
    def shutdown(self):
        logger.info("Shutting down session manager")
        
        self._shutdown_event.set()
        
        if self._cleanup_task and self._cleanup_task.is_alive():
            self._cleanup_task.join(timeout=5)
        with self._lock:
            session_keys = list(self.sessions.keys())
            for session_key in session_keys:
                self._cleanup_session(session_key)
        
        logger.info(f"Session manager shutdown complete")
    
    def get_session_stats(self) -> Dict[str, Any]:
        with self._lock:
            active_sessions = len(self.sessions)
            total_access_count = sum(info.access_count for info in self.sessions.values())
            
            return {
                'active_sessions': active_sessions,
                'total_access_count': total_access_count,
                'session_timeout_minutes': self.session_timeout_minutes
            }


class OpenAICompatibleServer:
    def __init__(self):
        self.settings = Settings()
        self.provider_factory = ProviderFactory()
        self.session_manager = SessionManager(session_timeout_minutes=30, cleanup_interval_minutes=5)
        self._loop = None
        
    async def startup(self):
        logger.info("Starting OpenAI-compatible API server")
        
        try:
            self._loop = asyncio.get_running_loop()
            logger.debug(f"Using event loop: {self._loop}")
        except RuntimeError:
            logger.warning("No running event loop found, creating new one")
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        self.session_manager.start_cleanup_task()
        logger.info("Server startup completed successfully")
        
    async def shutdown(self):
        logger.info("Shutting down OpenAI-compatible API server")
        
        try:
            self.session_manager.shutdown()
            
            if self._loop and not self._loop.is_closed():
                tasks = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
                
                if tasks:
                    logger.info(f"Cancelling {len(tasks)} pending tasks")
                    for task in tasks:
                        task.cancel()
                    
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks, return_exceptions=True),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Some tasks did not complete within timeout")
        
        except Exception as e:
            logger.error(f"Error during server shutdown: {e}", exc_info=True)
        
        logger.info("Server shutdown completed")


server_instance = OpenAICompatibleServer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await server_instance.startup()
    yield
    await server_instance.shutdown()


app = FastAPI(
    title="Swarm of Experts API",
    description="OpenAI-compatible API server for parallel LLM inference",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    safe_headers = {k: v for k, v in request.headers.items() 
                   if k.lower() not in ['authorization', 'x-api-key', 'cookie']}
    logger.debug(f"Request headers: {safe_headers}")
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url} - {response.status_code} in {process_time:.3f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url} - Error: {e} in {process_time:.3f}s")
        raise


@app.get("/health")
async def health_check():
    logger.debug("Processing health check request")
    
    try:
        stats = server_instance.session_manager.get_session_stats()
        response = {
            "status": "healthy", 
            "timestamp": int(time.time()),
            "session_stats": stats
        }
        logger.debug(f"Health check successful: {response}")
        return response
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": int(time.time()),
            "error": str(e)
        }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    logger.info("Processing request to list available models")
    
    try:
        models = []
        current_time = int(time.time())
        
        available_models = get_all_swarm_configs()
        logger.debug(f"Found {len(available_models)} available swarm configurations")
        
        for config_name in available_models:
            models.append(Model(
                id=config_name,
                created=current_time,
                owned_by="swarm-of-experts",
                root=config_name,
                parent=None
            ))
        
        logger.info(f"Successfully listed {len(models)} models")
        return ModelsResponse(data=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        error_response = create_generic_error_response(
            "Failed to list models",
            ErrorType.INTERNAL_ERROR,
            ErrorCode.INTERNAL_ERROR
        )
        raise HTTPException(status_code=500, detail=error_response.error.model_dump())


@app.get("/v1/sessions/stats")
async def get_session_stats():
    logger.info("Processing request for session statistics")
    
    try:
        stats = server_instance.session_manager.get_session_stats()
        logger.debug(f"Retrieved session stats: {stats}")
        logger.info(f"Successfully retrieved session statistics: {stats['active_sessions']} active sessions")
        return stats
    except Exception as e:
        logger.error(f"Error getting session stats: {e}", exc_info=True)
        error_response = create_generic_error_response(
            "Failed to get session statistics",
            ErrorType.INTERNAL_ERROR,
            ErrorCode.INTERNAL_ERROR
        )
        raise HTTPException(status_code=500, detail=error_response.error.model_dump())


@app.post("/v1/sessions/cleanup")
async def force_session_cleanup():
    logger.info("Processing request for forced session cleanup")
    
    try:
        initial_count = server_instance.session_manager.get_session_stats()['active_sessions']
        logger.debug(f"Initial active sessions count: {initial_count}")
        
        server_instance.session_manager._cleanup_expired_sessions()
        
        final_count = server_instance.session_manager.get_session_stats()['active_sessions']
        sessions_cleaned = initial_count - final_count
        
        logger.info(f"Session cleanup completed: {sessions_cleaned} sessions cleaned ({initial_count} -> {final_count})")
        
        return {
            "message": "Session cleanup completed",
            "sessions_before": initial_count,
            "sessions_after": final_count,
            "sessions_cleaned": sessions_cleaned
        }
    except Exception as e:
        logger.error(f"Error during forced session cleanup: {e}", exc_info=True)
        error_response = create_generic_error_response(
            "Failed to cleanup sessions",
            ErrorType.INTERNAL_ERROR,
            ErrorCode.INTERNAL_ERROR
        )
        raise HTTPException(status_code=500, detail=error_response.error.model_dump())


def _convert_messages(messages: List[ChatMessage]) -> List[CoreMessage]:
    converted = []
    for msg in messages:
        converted.append(CoreMessage(
            role=msg.role.value,
            content=msg.content
        ))
    return converted


def _validate_input(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text


async def _stream_response(
    chat_session: ChatSession,
    messages: List[CoreMessage],
    model: str,
    request_id: str,
    include_usage: bool = False
) -> AsyncGenerator[str, None]:
    accumulated_content = ""
    chunk_index = 0
    
    try:
        logger.info(f"Starting streaming response for request {request_id} with model {model}")
        
        sanitized_messages = []
        for msg in messages:
            sanitized_content = _validate_input(msg.content)
            sanitized_messages.append(CoreMessage(
                role=msg.role,
                content=sanitized_content,
                timestamp=msg.timestamp
            ))
        
        current_history = chat_session.history.get_messages()
        if len(current_history) == 0 or current_history[-1].content != sanitized_messages[-1].content:
            logger.debug(f"Adding {len(sanitized_messages)} messages to session history")
            for message in sanitized_messages:
                chat_session.history.add_message(message.role, message.content)
        else:
            logger.debug("Reusing existing session history")
        
        logger.debug(f"Starting message streaming for: {sanitized_messages[-1].content[:100]}...")
        
        try:
            async for chunk in chat_session.stream_message(sanitized_messages[-1].content):
                if chunk:
                    accumulated_content += chunk
                    
                    delta = ChatCompletionChoiceDelta(content=chunk)
                    choice = ChatCompletionChoice(
                        index=0,
                        delta=delta,
                        finish_reason=None
                    )
                    
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        created=int(time.time()),
                        model=model,
                        choices=[choice]
                    )
                    
                    yield f"data: {response.model_dump_json()}\n\n"
                    chunk_index += 1
                    
                    if chunk_index % 50 == 0:
                        logger.debug(f"Streamed {chunk_index} chunks for request {request_id}")
        
        except ValueError as ve:
            if "Task decomposition failed" in str(ve):
                logger.error(f"Task decomposition error during streaming for request {request_id}: {ve}")
                error_detail = ErrorDetail(
                    message=str(ve),
                    type=ErrorType.INVALID_REQUEST_ERROR.value,
                    code=ErrorCode.INVALID_REQUEST.value,
                    param="messages"
                )
            else:
                logger.error(f"Validation error during streaming for request {request_id}: {ve}")
                error_detail = ErrorDetail(
                    message=str(ve),
                    type=ErrorType.INVALID_REQUEST_ERROR.value,
                    code=ErrorCode.INVALID_REQUEST.value
                )
            error_response = ErrorResponse(error=error_detail)
            yield f"data: {error_response.model_dump_json()}\n\n"
            return
        except Exception as stream_error:
            logger.error(f"Error during streaming for request {request_id}: {stream_error}")
            error_detail = ErrorDetail(
                message=f"Streaming error: {str(stream_error)}",
                type=ErrorType.INTERNAL_ERROR.value,
                code=ErrorCode.INTERNAL_ERROR.value
            )
            error_response = ErrorResponse(error=error_detail)
            yield f"data: {error_response.model_dump_json()}\n\n"
            return
        
        final_choice = ChatCompletionChoice(
            index=0,
            delta=ChatCompletionChoiceDelta(),
            finish_reason="stop"
        )
        
        usage = None
        if include_usage:
            prompt_tokens = TokenEstimator.estimate_messages_tokens(sanitized_messages)
            completion_tokens = TokenEstimator.estimate_tokens(accumulated_content)
            total_tokens = prompt_tokens + completion_tokens
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails()
            )
            
            logger.debug(f"Usage calculated for request {request_id}: {usage}")
        
        final_response = ChatCompletionStreamResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[final_choice],
            usage=usage
        )
        
        yield f"data: {final_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info(f"Completed streaming response for request {request_id} with {chunk_index} chunks")
        
    except Exception as e:
        logger.error(f"Critical error in streaming response for request {request_id}: {e}", exc_info=True)
        error_detail = ErrorDetail(
            message=f"Internal streaming error: {str(e)}",
            type=ErrorType.INTERNAL_ERROR.value,
            code=ErrorCode.INTERNAL_ERROR.value
        )
        error_response = ErrorResponse(error=error_detail)
        yield f"data: {error_response.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"




@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    try:
        logger.info(f"Processing chat completion request {request_id} for model {request.model}")
        
        APIValidator.validate_request(request)
        logger.debug(f"Request validation passed for {request_id}")
        
        sanitized_messages = []
        for msg in request.messages:
            sanitized_content = _validate_input(msg.content)
            sanitized_messages.append(ChatMessage(
                role=msg.role,
                content=sanitized_content
            ))
        
        core_messages = _convert_messages(sanitized_messages)
        logger.debug(f"Converted {len(core_messages)} messages for request {request_id}")
        
        message_dicts = [{'role': msg.role.value, 'content': msg.content} for msg in sanitized_messages]
        try:
            chat_session = server_instance.session_manager.get_or_create_session(
                message_dicts, request.model
            )
            logger.debug(f"Session obtained for request {request_id}")
        except Exception as session_error:
            logger.error(f"Failed to get session for request {request_id}: {session_error}")
            raise APIValidationError(
                f"Failed to create session: {str(session_error)}",
                ErrorType.INTERNAL_ERROR,
                ErrorCode.INTERNAL_ERROR
            )
        
        if request.stream:
            logger.debug(f"Starting streaming response for request {request_id}")
            include_usage = request.stream_options and request.stream_options.include_usage
            return StreamingResponse(
                _stream_response(chat_session, core_messages, request.model, request_id, include_usage),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Methods": "*"
                }
            )
        
        else:
            logger.debug(f"Starting non-streaming response for request {request_id}")
            
            current_history = chat_session.history.get_messages()
            if len(current_history) == 0 or current_history[-1].content != core_messages[-1].content:
                logger.debug(f"Adding {len(core_messages)} messages to session history")
                for message in core_messages:
                    chat_session.history.add_message(message.role, message.content)
            else:
                logger.debug("Reusing existing session history")
            
            try:
                response_content = await chat_session.send_message(core_messages[-1].content)
                logger.debug(f"Generated response content ({len(response_content)} chars) for request {request_id}")
            except ValueError as ve:
                logger.error(f"Value error in chat completions for request {request_id}: {ve}")
                if "Task decomposition failed" in str(ve):
                    error_response = create_generic_error_response(
                        str(ve),
                        ErrorType.INTERNAL_ERROR,
                        ErrorCode.INTERNAL_ERROR
                    )
                    raise HTTPException(status_code=500, detail=error_response.error.model_dump())
                else:
                    error_response = create_generic_error_response(
                        str(ve),
                        ErrorType.INVALID_REQUEST_ERROR,
                        ErrorCode.VALIDATION_ERROR
                    )
                    raise HTTPException(status_code=400, detail=error_response.error.model_dump())
            except Exception as generation_error:
                logger.error(f"Error generating response for request {request_id}: {generation_error}")
                raise APIValidationError(
                    f"Failed to generate response: {str(generation_error)}",
                    ErrorType.INTERNAL_ERROR,
                    ErrorCode.INTERNAL_ERROR
                )
            
            choice = ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role=Role.ASSISTANT,
                    content=response_content
                ),
                finish_reason="stop"
            )
            
            prompt_tokens = TokenEstimator.estimate_messages_tokens(core_messages)
            completion_tokens = TokenEstimator.estimate_tokens(response_content)
            total_tokens = prompt_tokens + completion_tokens
            
            logger.debug(f"Token usage for request {request_id}: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            
            response = ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    prompt_tokens_details=PromptTokensDetails(),
                    completion_tokens_details=CompletionTokensDetails()
                ),
                system_fingerprint="swarm-of-experts-v1.0.0"
            )
            
            logger.info(f"Successfully completed chat completion request {request_id}")
            return response
            
    except APIValidationError as e:
        logger.warning(f"Validation error in chat completions for request {request_id}: {e}")
        error_response = create_error_response(e)
        status_code = 400 if e.error_type == ErrorType.INVALID_REQUEST_ERROR else 500
        raise HTTPException(status_code=status_code, detail=error_response.error.model_dump())
    except ValueError as e:
        logger.error(f"Value error in chat completions for request {request_id}: {e}")
        error_response = create_generic_error_response(
            str(e),
            ErrorType.INVALID_REQUEST_ERROR,
            ErrorCode.VALIDATION_ERROR
        )
        raise HTTPException(status_code=400, detail=error_response.error.model_dump())
    except Exception as e:
        logger.error(f"Unexpected error in chat completions for request {request_id}: {e}", exc_info=True)
        error_response = create_generic_error_response(
            "Internal server error",
            ErrorType.INTERNAL_ERROR,
            ErrorCode.INTERNAL_ERROR
        )
        raise HTTPException(status_code=500, detail=error_response.error.model_dump())


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url}: {exc}", exc_info=True)
    
    error_response = create_generic_error_response(
        "Internal server error",
        ErrorType.INTERNAL_ERROR,
        ErrorCode.INTERNAL_ERROR
    )
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content=error_response.error.model_dump()
    )


@app.exception_handler(APIValidationError)
async def validation_exception_handler(request: Request, exc: APIValidationError):
    logger.warning(f"API validation error on {request.method} {request.url}: {exc}")
    
    error_response = create_error_response(exc)
    status_code = 400
    if exc.error_type == ErrorType.AUTHENTICATION_ERROR:
        status_code = 401
    elif exc.error_type == ErrorType.NOT_FOUND_ERROR:
        status_code = 404
    elif exc.error_type == ErrorType.INTERNAL_ERROR:
        status_code = 500
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content=error_response.error.model_dump()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning(f"Value error on {request.method} {request.url}: {exc}")
    
    error_response = create_generic_error_response(
        str(exc),
        ErrorType.INVALID_REQUEST_ERROR,
        ErrorCode.VALIDATION_ERROR
    )
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400,
        content=error_response.error.model_dump()
    )


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Request validation error on {request.method} {request.url}: {exc}")
    first_error = exc.errors()[0] if exc.errors() else None
    
    if first_error:
        field_path = " -> ".join(str(loc) for loc in first_error['loc'])
        message = f"Validation error in field '{field_path}': {first_error['msg']}"
        param = field_path
    else:
        message = "Invalid request format"
        param = None
    
    error_response = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=ErrorType.INVALID_REQUEST_ERROR.value,
            code=ErrorCode.VALIDATION_ERROR.value,
            param=param
        )
    )
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400,
        content=error_response.error.model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)