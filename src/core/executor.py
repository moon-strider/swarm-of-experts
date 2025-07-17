import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import time

from ..providers.base import LLMProvider, Message
from ..providers.factory import ProviderFactory
from ..config.swarm_configs import GeneratorConfig, SwarmConfig
from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class GeneratorResponse:
    generator_config: GeneratorConfig
    content: str
    elapsed_time: float
    error: Optional[str] = None


class ParallelExecutor:
    
    def __init__(self, factory: ProviderFactory, timeout: int = 300):
        self.factory = factory
        self.timeout = timeout

    async def execute_parallel(
        self, 
        swarm_config: SwarmConfig,
        messages: List[Message],
        stream: bool = False,
        sub_prompts: Optional[Dict[str, str]] = None
    ) -> List[GeneratorResponse]:
        
        logger.info(f"Starting execution with {len(swarm_config.generators)} generators")
        
        if sub_prompts:
            if len(sub_prompts) != len(swarm_config.generators):
                raise ValueError(f"Mismatch between sub-prompts ({len(sub_prompts)}) and generators ({len(swarm_config.generators)})")
            sub_prompt_values = [sub_prompts[f"sub_prompt_{i+1}"] for i in range(len(swarm_config.generators))]
            for i, sub_prompt in enumerate(sub_prompt_values):
                if not sub_prompt.strip():
                    raise ValueError(f"Empty sub-prompt for generator {i+1}")

        tasks = []
        for i, gen_config in enumerate(swarm_config.generators):
            gen_messages = messages
            if sub_prompts:
                gen_messages = [Message("user", sub_prompt_values[i])]
            
            task = asyncio.create_task(
                self._execute_single_generator(
                    gen_config,
                    gen_messages,
                    stream
                )
            )
            tasks.append(task)

        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Execution timed out after {self.timeout}s")
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Generator {i} failed: {response}")
                results.append(GeneratorResponse(
                    generator_config=swarm_config.generators[i],
                    content="",
                    elapsed_time=0,
                    error=str(response)
                ))
            else:
                results.append(response)

        return results

    async def _execute_single_generator(
        self,
        gen_config: GeneratorConfig,
        messages: List[Message],
        stream: bool
    ) -> GeneratorResponse:
        
        start_time = time.time()
        provider = None

        try:
            api_key = settings.get_api_key_for_provider(gen_config.provider)
            provider = self.factory.create(
                gen_config.provider,
                api_key=api_key,
                model=gen_config.model,
                temperature=gen_config.temperature
            )

            logger.info(f"Executing {gen_config.provider}/{gen_config.model}")

            if stream:
                chunks = []
                for chunk in provider.stream(messages):
                    chunks.append(chunk)
                content = "".join(chunks)
            else:
                content = provider.generate(messages)

            elapsed = time.time() - start_time
            logger.info(f"{gen_config.provider}/{gen_config.model} completed in {elapsed:.2f}s")

            return GeneratorResponse(
                generator_config=gen_config,
                content=content,
                elapsed_time=elapsed
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Generator {gen_config.provider}/{gen_config.model} failed: {e}")
            return GeneratorResponse(
                generator_config=gen_config,
                content="",
                elapsed_time=elapsed,
                error=str(e)
            )

    async def stream_parallel(self, swarm_config: SwarmConfig, messages: List[Message], sub_prompts: Optional[Dict[str, str]] = None):
        """Stream responses from the first generator (for single-generator configs)"""
        if not swarm_config.generators:
            return
        
        if sub_prompts:
            if len(sub_prompts) != len(swarm_config.generators):
                raise ValueError(f"Mismatch between sub-prompts ({len(sub_prompts)}) and generators ({len(swarm_config.generators)})")
            
        gen_config = swarm_config.generators[0]
        gen_messages = messages
        if sub_prompts:
            sub_prompt_value = sub_prompts["sub_prompt_1"]
            if not sub_prompt_value.strip():
                raise ValueError("Empty sub-prompt for generator 1")
            gen_messages = [Message("user", sub_prompt_value)]
        
        try:
            api_key = settings.get_api_key_for_provider(gen_config.provider)
            provider = self.factory.create(
                gen_config.provider,
                api_key=api_key,
                model=gen_config.model,
                temperature=gen_config.temperature
            )
            
            logger.info(f"Streaming from {gen_config.provider}/{gen_config.model}")
            
            for chunk in provider.stream(gen_messages):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming from {gen_config.provider}/{gen_config.model} failed: {e}")
            raise

    def close(self):
        pass