from typing import Iterator, Optional, AsyncIterator
import asyncio
from ..providers.base import LLMProvider
from .messages import MessageHistory
from ..config.swarm_configs import SwarmConfig
from .executor import ParallelExecutor
from .merger import ResponseMerger
from ..providers.factory import ProviderFactory
from ..config.settings import settings


class ChatSession:
    def __init__(self, max_history: Optional[int] = None, swarm_config: Optional[SwarmConfig] = None):
        self.swarm_config = swarm_config
        self.history = MessageHistory(max_messages=max_history)
        self.factory = ProviderFactory()
        self.executor = ParallelExecutor(self.factory) if swarm_config else None
        
    async def send_message(self, message: str) -> str:
        self.history.add_message("user", message)
        
        try:
            if not self.swarm_config:
                raise ValueError("Swarm config is required")
                
            responses = await self.executor.execute_parallel(
                self.swarm_config,
                self.history.get_messages(),
                stream=False
            )
            
            if self.swarm_config.has_merger:
                merger_api_key = settings.get_api_key_for_provider(self.swarm_config.merger.provider)
                merger_provider = self.factory.create(
                    self.swarm_config.merger.provider,
                    api_key=merger_api_key,
                    model=self.swarm_config.merger.model,
                    temperature=self.swarm_config.merger.temperature
                )
                merger = ResponseMerger(merger_provider, self.swarm_config)
                response = merger.merge_responses(message, responses)
            else:
                response = responses[0].content if responses else ""
            
            self.history.add_message("assistant", response)
            return response
            
        except Exception as e:
            raise Exception(f"Failed to get response: {str(e)}")
        
    async def stream_message(self, message: str) -> AsyncIterator[str]:
        self.history.add_message("user", message)
        
        full_response = []
        
        try:
            if not self.swarm_config:
                raise ValueError("Swarm config is required")
                
            if self.swarm_config.has_merger:
                responses = await self.executor.execute_parallel(
                    self.swarm_config,
                    self.history.get_messages(),
                    stream=False
                )
                
                merger_api_key = settings.get_api_key_for_provider(self.swarm_config.merger.provider)
                merger_provider = self.factory.create(
                    self.swarm_config.merger.provider,
                    api_key=merger_api_key,
                    model=self.swarm_config.merger.model,
                    temperature=self.swarm_config.merger.temperature
                )
                merger = ResponseMerger(merger_provider, self.swarm_config)
                
                for chunk in merger.stream_merge_responses(message, responses):
                    full_response.append(chunk)
                    yield chunk
            else:
                async for chunk in self.executor.stream_parallel(
                    self.swarm_config,
                    self.history.get_messages()
                ):
                    full_response.append(chunk)
                    yield chunk
            
            self.history.add_message("assistant", "".join(full_response))
            
        except Exception as e:
            raise Exception(f"Failed to stream response: {str(e)}")
        
    def clear_history(self) -> None:
        self.history.clear()
        
    def get_history_length(self) -> int:
        return len(self.history)
    
    def cleanup(self):
        if self.executor:
            self.executor.close()