import asyncio
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

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

class TokenLimiter:
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    @staticmethod
    def truncate_to_token_limit(text: str, max_tokens: int) -> str:
        estimated_chars = max_tokens * 4
        if len(text) > estimated_chars:
            logger.warning(f"Truncating response from ~{TokenLimiter.estimate_tokens(text)} to {max_tokens} tokens")
            return text[:estimated_chars] + "..."
        return text

class ParallelExecutor:
    
    def __init__(self, factory: ProviderFactory, timeout: int = 60):
        self.factory = factory
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def execute_parallel(
        self, 
        swarm_config: SwarmConfig,
        messages: List[Message],
        stream: bool = False
    ) -> List[GeneratorResponse]:
        
        if not swarm_config.is_parallel:
            raise ValueError("SwarmConfig must have multiple generators for parallel execution")

        logger.info(f"Starting parallel execution with {len(swarm_config.generators)} generators")

        tasks = []
        for gen_config in swarm_config.generators:
            task = asyncio.create_task(
                self._execute_single_generator(
                    gen_config,
                    messages,
                    swarm_config.per_generator_token_limit,
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
            logger.error(f"Parallel execution timed out after {self.timeout}s")
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
        token_limit: Optional[int],
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

            loop = asyncio.get_event_loop()

            if stream:
                chunks = []
                stream_gen = await loop.run_in_executor(
                    self.executor,
                    lambda: list(provider.stream(messages))
                )
                content = "".join(stream_gen)
            else:
                content = await loop.run_in_executor(
                    self.executor,
                    provider.generate,
                    messages
                )

            if token_limit:
                content = TokenLimiter.truncate_to_token_limit(content, token_limit)

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

    def close(self):
        self.executor.shutdown(wait=True)