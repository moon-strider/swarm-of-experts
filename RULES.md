# Swarm of Experts - Development Rules & Best Practices

## Table of Contents
- [Architecture Rules](#architecture-rules)
- [Code Quality Rules](#code-quality-rules)
- [Performance Rules](#performance-rules)
- [Security Rules](#security-rules)
- [Error Handling Rules](#error-handling-rules)
- [Configuration Rules](#configuration-rules)
- [CLI/UX Rules](#cli-ux-rules)
- [Extension Rules](#extension-rules)
- [Monitoring Rules](#monitoring-rules)

## Architecture Rules

### 1. Provider Architecture
- **Abstract Base Class**: All providers MUST extend `LLMProvider` from `src/providers/base.py:15`
- **Factory Pattern**: ALWAYS use `ProviderFactory` for provider instantiation, NEVER instantiate directly
- **Model Validation**: Each provider MUST implement `validate_model()` and `available_models` property
- **LangChain Integration**: Use appropriate LangChain chat models for each provider
- **Thread Safety**: Provider instances MUST be thread-safe or use thread-local storage
- **Connection Pooling**: Implement connection pooling for HTTP clients to reduce latency

### 2. Message Handling
- **Standard Format**: Use `Message` dataclass from `src/providers/base.py:6` for internal representation
- **Conversion**: Always convert to provider-specific format using `_convert_messages()`
- **History Management**: Use `MessageHistory` from `src/core/messages.py:13` with proper limits
- **Memory Bounds**: Implement both message count AND memory-based limits for history
- **Cleanup**: Implement proper cleanup in `ChatSession.cleanup()` for resource management

### 3. Swarm Mode Implementation
- **Configuration**: Use `SwarmConfig` from `src/config/swarm_configs.py:44`
- **Parallel Execution**: Always use `ParallelExecutor` from `src/core/executor.py:36`
- **Token Distribution**: Implement intelligent token distribution based on model capabilities
- **Error Resilience**: Continue execution even if some generators fail
- **Fallback Strategy**: Implement sophisticated fallback selection in merger
- **Timeout Management**: Use proper timeout handling with task cancellation

### 4. Threading and Concurrency
- **Event Loop Management**: Use `asyncio.run()` instead of `asyncio.get_event_loop()`
- **Thread Pool Sizing**: Dynamically size thread pools based on generator count
- **Resource Management**: Implement context managers for executors and providers
- **Animation Threading**: Use daemon threads for animations with proper synchronization
- **Thread-Local Storage**: Use thread-local storage for provider instances

## Code Quality Rules

### 1. Type Annotations
- **Required**: All function parameters and return types MUST be annotated
- **Optional**: Use `Optional[T]` instead of `Union[T, None]`
- **Generics**: Use proper generic type hints for collections
- **Dataclasses**: Use `@dataclass` for configuration objects with proper field types

### 2. Import Organization
```python
# Standard library imports
import os
import sys
from typing import List, Optional, Dict, Any

# Third-party imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Local imports
from .base import LLMProvider, Message
from ..config.settings import settings
```

### 3. Naming Conventions
- **Classes**: PascalCase (e.g., `ChatSession`, `ResponseMerger`)
- **Functions**: snake_case (e.g., `send_message`, `validate_model`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MODELS`, `MERGER_PROMPT_TEMPLATE`)
- **Private Methods**: Prefix with underscore (e.g., `_convert_messages`, `_format_responses_xml`)
- **Configuration**: Use descriptive names for swarm configurations

### 4. Documentation Standards
- **Docstrings**: Triple quotes for all public methods and classes
- **Type Information**: Include parameter and return types in docstrings
- **Examples**: Provide usage examples for complex methods
- **Error Cases**: Document expected exceptions and error conditions

## Performance Rules

### 1. Memory Management
- **Bounded Collections**: Use `collections.deque` with `maxlen` for circular buffers
- **Streaming**: Implement true streaming without buffering full responses
- **Memory Monitoring**: Track memory usage and implement cleanup strategies
- **Token Counting**: Use proper tokenization libraries, not character-based estimation

### 2. Connection Optimization
- **Connection Pooling**: Use `requests.Session` with connection pooling
- **Provider Reuse**: Cache provider instances per thread to avoid recreation
- **HTTP Keep-Alive**: Enable HTTP keep-alive for persistent connections
- **Timeout Configuration**: Set appropriate timeouts for all HTTP operations

### 3. Caching Strategies
- **Response Caching**: Implement LRU cache for repeated queries
- **Provider Caching**: Cache provider instances based on configuration
- **Semantic Caching**: Use semantic similarity for partial cache hits
- **Cache Invalidation**: Implement proper cache invalidation strategies

### 4. Concurrency Optimization
- **Dynamic Thread Pools**: Adjust thread pool size based on workload
- **Backpressure Handling**: Implement flow control for streaming operations
- **Resource Pooling**: Pool expensive resources like tokenizers
- **Batch Processing**: Group similar operations for efficiency

## Security Rules

### 1. API Key Management
- **Environment Variables**: NEVER hardcode API keys in source code
- **Secure Storage**: Use system keyring or encrypted storage for production
- **Key Rotation**: Implement API key rotation mechanisms
- **Access Control**: Limit API key access to necessary components only
- **Logging**: NEVER log API keys or sensitive information

### 2. Input Validation
- **User Input**: Validate and sanitize all user inputs
- **Configuration**: Validate all configuration parameters before use
- **Model Names**: Validate model names against known lists
- **Token Limits**: Enforce token limits to prevent abuse

### 3. Error Information
- **Sensitive Data**: Don't expose sensitive information in error messages
- **Stack Traces**: Log detailed stack traces but show user-friendly messages
- **API Errors**: Sanitize API error messages before showing to users

## Error Handling Rules

### 1. Exception Hierarchy
- **Custom Exceptions**: Define specific exception types for different error conditions
- **Base Exception**: Create base `SwarmError` class for all application errors
- **Context Preservation**: Maintain original error context when re-raising
- **Error Codes**: Use consistent error codes for different failure types

### 2. Resilience Patterns
- **Retry Logic**: Implement exponential backoff for transient failures
- **Circuit Breaker**: Implement circuit breaker pattern for external API calls
- **Timeout Handling**: Use appropriate timeouts with proper cleanup
- **Graceful Degradation**: Continue operation with reduced functionality when possible

### 3. Error Propagation
- **Layered Handling**: Handle errors at appropriate abstraction levels
- **Context Addition**: Add context when propagating errors up the stack
- **User Feedback**: Provide actionable error messages to users
- **Logging**: Log errors with appropriate severity levels

### 4. Recovery Mechanisms
- **Automatic Recovery**: Implement automatic recovery for transient failures
- **State Consistency**: Maintain consistent state during error conditions
- **Resource Cleanup**: Ensure proper cleanup of resources on errors
- **Partial Success**: Handle partial success scenarios in swarm mode

## Configuration Rules

### 1. Configuration Management
- **Schema Validation**: Use Pydantic or similar for configuration validation
- **Environment Specific**: Support different configurations for different environments
- **Default Values**: Provide sensible defaults for all optional settings
- **Configuration Files**: Support YAML/JSON configuration files

### 2. Settings Architecture
- **Immutability**: Treat configuration as immutable after initialization
- **Validation**: Validate configuration at startup and runtime
- **Hot Reload**: Support hot reloading of configurations where appropriate
- **Migration**: Implement configuration migration for version changes

### 3. Provider Configuration
- **Model Validation**: Validate model availability at startup
- **Capability Detection**: Detect and validate provider capabilities
- **Configuration Templates**: Provide configuration templates for common scenarios
- **Documentation**: Document all configuration options with examples

## CLI/UX Rules

### 1. User Experience
- **Consistency**: Use consistent command patterns and responses
- **Feedback**: Provide immediate feedback for all user actions
- **Progress Indicators**: Show progress for long-running operations
- **Error Messages**: Provide clear, actionable error messages

### 2. Command Design
- **Discoverability**: Make commands easily discoverable with help
- **Validation**: Validate all command inputs before execution
- **Confirmation**: Require confirmation for destructive operations
- **Undo/Redo**: Provide undo capabilities where appropriate

### 3. Output Formatting
- **Consistent Styling**: Use consistent colors and formatting
- **Responsive Design**: Adapt to different terminal sizes
- **Accessibility**: Support high-contrast modes and screen readers
- **Internationalization**: Design for future internationalization

## Extension Rules

### 1. Adding New Providers
1. Create provider file in `src/providers/`
2. Extend `LLMProvider` base class
3. Implement all required abstract methods
4. Add proper error handling and validation
5. Register in `ProviderFactory._providers`
6. Add API key support in `Settings` class
7. Update model validation and available models
8. Update documentation

### 2. Adding New Swarm Configurations
1. Define configuration in `swarm_configs.py`
2. Validate all generator and merger configurations
3. Verify with actual models and API keys
4. Implement proper token distribution
5. Add configuration validation
6. Update documentation with examples

### 3. Adding New CLI Features
1. Design consistent command interface
2. Implement proper input validation
3. Add help text and documentation
4. Handle errors gracefully
5. Verify with various configurations
6. Update user documentation
7. Consider backward compatibility

## Monitoring Rules

### 1. Logging
- **Structured Logging**: Use structured logging with consistent formats
- **Log Levels**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **Context**: Include relevant context in all log messages
- **Performance**: Log performance metrics for monitoring
- **Rotation**: Implement log rotation for long-running processes

### 2. Metrics
- **Key Metrics**: Track request latency, success rates, and error rates
- **Provider Metrics**: Monitor individual provider performance
- **Resource Usage**: Track memory and CPU usage
- **User Metrics**: Track user engagement and feature usage

### 3. Health Checks
- **Provider Health**: Implement health checks for all providers
- **Configuration Health**: Validate configuration health
- **Resource Health**: Monitor resource availability
- **Dependency Health**: Check external dependency health

## Development Workflow Rules

### 1. Code Review
- **Peer Review**: All code changes must be peer reviewed
- **Automated Checks**: Run linting and type checking
- **Security Review**: Review for security implications
- **Performance Review**: Consider performance impact of changes

### 2. Version Control
- **Atomic Commits**: Make atomic commits with clear messages
- **Branch Strategy**: Use feature branches for development
- **Merge Strategy**: Use merge commits for feature integration
- **Tagging**: Tag releases with semantic versioning

### 3. Deployment
- **Staging**: Verify all changes in staging environment
- **Rollback**: Maintain ability to rollback changes
- **Monitoring**: Monitor deployments for issues
- **Documentation**: Update documentation with deployments

## Anti-Patterns to Avoid

### 1. Common Anti-Patterns
- **Bare Exception Handling**: Don't use bare `except:` clauses
- **Generic Error Messages**: Don't show raw exceptions to users
- **Hardcoded Configuration**: Don't hardcode configuration values
- **Blocking Operations**: Don't block the main thread unnecessarily
- **Resource Leaks**: Don't forget to clean up resources

### 2. Performance Anti-Patterns
- **Premature Optimization**: Don't optimize before measuring
- **Inefficient Algorithms**: Don't use O(n²) algorithms where O(n) exists
- **Memory Leaks**: Don't hold references to unused objects
- **Excessive Logging**: Don't log excessively in hot paths

### 3. Security Anti-Patterns
- **Logging Secrets**: Don't log sensitive information
- **Weak Validation**: Don't trust user input without validation
- **Plain Text Storage**: Don't store secrets in plain text
- **Overprivileged Access**: Don't grant excessive permissions

## Critical Implementation Issues

### 1. Provider System Fixes
- **Model Detection Conflict**: Fix overlapping patterns in `factory.py:46-47`
- **Error Handling Consistency**: Standardize error handling across all providers
- **Streaming Configuration**: Fix Google provider missing streaming parameter
- **Provider Registration**: Ensure all providers are properly exported

### 2. Performance Optimizations
- **Memory-Bounded History**: Replace simple message count with memory-based limits
- **Token Estimation**: Replace character-based estimation with proper tokenization
- **Connection Pooling**: Implement HTTP connection pooling for all providers
- **Streaming Fixes**: Remove buffering in `executor.py:119-124`

### 3. Security Enhancements
- **API Key Storage**: Implement secure storage for API keys
- **Input Validation**: Add comprehensive input validation
- **Error Message Sanitization**: Sanitize error messages before showing to users
- **Rate Limiting**: Implement rate limiting for API calls

This comprehensive rules document provides the foundation for maintaining code quality, performance, and security in the swarm-of-experts project. All developers should follow these guidelines to ensure consistent, maintainable, and scalable code.