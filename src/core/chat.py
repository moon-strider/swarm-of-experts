from typing import Iterator, Optional, AsyncIterator, Dict, List
import asyncio
import json
import re
from xml.sax.saxutils import escape
from ..providers.base import LLMProvider, Message
from .messages import MessageHistory
from ..config.swarm_configs import SwarmConfig, TASKMASTER_PROMPT_TEMPLATE
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
    
    async def _decompose_task(self, messages: List[Message]) -> Dict[str, str]:
        taskmaster_prompt = TASKMASTER_PROMPT_TEMPLATE.format(user_query=escape(messages[-1].content))
        taskmaster_messages = messages[:-1] + [Message("user", taskmaster_prompt)]
        
        taskmaster_api_key = settings.get_api_key_for_provider(self.swarm_config.taskmaster.provider)
        if not taskmaster_api_key:
            raise ValueError(f"API key not found for taskmaster provider: {self.swarm_config.taskmaster.provider}")
        
        taskmaster_provider = self.factory.create(
            self.swarm_config.taskmaster.provider,
            api_key=taskmaster_api_key,
            model=self.swarm_config.taskmaster.model,
            temperature=self.swarm_config.taskmaster.temperature
        )
        
        try:
            decomposition_response = taskmaster_provider.generate(taskmaster_messages)
        except Exception as e:
            raise RuntimeError(f"Taskmaster decomposition failed: {str(e)}")
        
        sub_prompts_dict = {}
        for i in range(1, 4):
            pattern = f'<sub_prompt_{i}>(.*?)</sub_prompt_{i}>'
            match = re.search(pattern, decomposition_response, re.DOTALL)
            if not match:
                raise ValueError(f"Task decomposition failed: Missing <sub_prompt_{i}> tag in response")
            sub_prompts_dict[f"sub_prompt_{i}"] = match.group(1).strip()
            if not sub_prompts_dict[f"sub_prompt_{i}"]:
                raise ValueError(f"Task decomposition failed: Empty sub_prompt_{i}")
        
        return sub_prompts_dict
        
    async def send_message(self, message: str) -> str:
        self.history.add_message("user", message)
        
        try:
            if not self.swarm_config:
                raise ValueError("Swarm config is required")
            
            sub_prompts = None
            if self.swarm_config.has_taskmaster:
                sub_prompts = await self._decompose_task(self.history.get_messages())
                
            responses = await self.executor.execute_parallel(
                self.swarm_config,
                self.history.get_messages(),
                stream=False,
                sub_prompts=sub_prompts
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
                response = merger.merge_responses(message, responses, history=self.history.get_messages())
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
            
            sub_prompts = None
            if self.swarm_config.has_taskmaster:
                sub_prompts = await self._decompose_task(self.history.get_messages())
                
            if self.swarm_config.has_merger:
                responses = await self.executor.execute_parallel(
                    self.swarm_config,
                    self.history.get_messages(),
                    stream=False,
                    sub_prompts=sub_prompts
                )
                
                merger_api_key = settings.get_api_key_for_provider(self.swarm_config.merger.provider)
                merger_provider = self.factory.create(
                    self.swarm_config.merger.provider,
                    api_key=merger_api_key,
                    model=self.swarm_config.merger.model,
                    temperature=self.swarm_config.merger.temperature
                )
                merger = ResponseMerger(merger_provider, self.swarm_config)
                
                for chunk in merger.stream_merge_responses(message, responses, history=self.history.get_messages()):
                    full_response.append(chunk)
                    yield chunk
            else:
                async for chunk in self.executor.stream_parallel(
                    self.swarm_config,
                    self.history.get_messages(),
                    sub_prompts=sub_prompts
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