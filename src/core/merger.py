import logging
from typing import List, Optional, AsyncIterator
from xml.sax.saxutils import escape

from langchain_core.messages import BaseMessage, HumanMessage
from ..providers.base import LLMProvider
from ..config.swarm_configs import SwarmConfig
from .executor import GeneratorResponse

logger = logging.getLogger(__name__)

class ResponseMerger:
    def __init__(self, provider: LLMProvider, swarm_config: SwarmConfig):
        self.provider = provider
        self.swarm_config = swarm_config

    async def merge_responses(
        self,
        user_query: str,
        responses: List[GeneratorResponse],
        history: Optional[List[BaseMessage]] = None
    ) -> str:
        valid_responses = [r for r in responses if not r.error and r.content]

        if not valid_responses:
            logger.error("No valid responses to merge")
            raise ValueError("All generators failed to produce responses")

        if len(valid_responses) == 1:
            logger.warning("Only one valid response, returning it directly")
            return valid_responses[0].content

        formatted_responses = self._format_responses_xml(valid_responses)

        merger_prompt = self.swarm_config.merger_prompt_template.format(
            user_query=escape(user_query),
            responses=formatted_responses
        )

        logger.info(f"Merging {len(valid_responses)} responses")

        if history:
            messages = history[:-1] + [HumanMessage(content=merger_prompt)]
        else:
            messages = [HumanMessage(content=merger_prompt)]

        try:
            merged_response = await self.provider.generate(messages)
            logger.info("Successfully merged responses")
            return merged_response
        except Exception as e:
            logger.error(f"Merger failed: {e}")
            return self._select_best_fallback(valid_responses)

    async def stream_merge_responses(
        self,
        user_query: str,
        responses: List[GeneratorResponse],
        history: Optional[List[BaseMessage]] = None
    ) -> AsyncIterator[str]:
        valid_responses = [r for r in responses if not r.error and r.content]

        if not valid_responses:
            raise ValueError("All generators failed to produce responses")

        if len(valid_responses) == 1:
            yield valid_responses[0].content
            return

        formatted_responses = self._format_responses_xml(valid_responses)
        merger_prompt = self.swarm_config.merger_prompt_template.format(
            user_query=escape(user_query),
            responses=formatted_responses
        )

        if history:
            messages = history[:-1] + [HumanMessage(content=merger_prompt)]
        else:
            messages = [HumanMessage(content=merger_prompt)]

        try:
            async for chunk in self.provider.stream(messages):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming merger failed: {e}")
            yield self._select_best_fallback(valid_responses)

    def _format_responses_xml(self, responses: List[GeneratorResponse]) -> str:
        formatted = []
        for i, response in enumerate(responses, 1):
            provider_info = f"{response.generator_config.provider}/{response.generator_config.model}"
            formatted.append(
                f"<response{i} provider=\"{escape(provider_info)}\" "
                f"time=\"{response.elapsed_time:.2f}s\">\n"
                f"{escape(response.content)}\n"
                f"</response{i}>"
            )

        return "\n\n".join(formatted)

    def _select_best_fallback(self, responses: List[GeneratorResponse]) -> str:
        sorted_responses = sorted(
            responses,
            key=lambda r: (r.error is None, len(r.content)),
            reverse=True
        )

        best = sorted_responses[0]
        logger.info(f"Fallback: selected response from {best.generator_config.provider}")
        return best.content