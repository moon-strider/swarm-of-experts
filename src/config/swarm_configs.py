from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

MERGER_PROMPT_TEMPLATE = """You are an expert AI response synthesizer. Your task is to analyze multiple AI-generated responses to the same query and create the best possible answer.

## Original User Query:
{user_query}

## Responses to Analyze:
{responses}

## Your Task:
1. Carefully analyze each response for:
   - Accuracy and correctness
   - Completeness of information
   - Clarity and coherence
   - Relevance to the user's question

2. Identify the strongest elements from each response

3. Synthesize a final answer that:
   - Combines the best insights from all responses
   - Corrects any inaccuracies found
   - Provides the most comprehensive and helpful answer
   - Maintains a natural, conversational tone

## Important Rules:
- Do NOT mention that you are analyzing multiple responses
- Do NOT reference "other answers" or "responses"
- Write as if you are directly answering the user's question
- Ensure your answer is self-contained and complete

Provide your synthesized response below:"""

@dataclass
class GeneratorConfig:
    provider: str
    model: str
    temperature: float = 0.7

@dataclass
class SwarmConfig:
    name: str
    generators: List[GeneratorConfig] = field(default_factory=list)
    merger: Optional[GeneratorConfig] = None
    merger_prompt_template: str = MERGER_PROMPT_TEMPLATE
    max_context_tokens: int = 128000

    @property
    def per_generator_token_limit(self) -> Optional[int]:
        if self.merger and self.generators:
            available_tokens = int(self.max_context_tokens * 0.9)
            return available_tokens // len(self.generators)
        return None

    @property
    def is_parallel(self) -> bool:
        return self.merger is not None and len(self.generators) > 1

SWARM_CONFIGS = {
    "basic": SwarmConfig(
        name="basic",
        generators=[GeneratorConfig(provider="openai", model="gpt-4.1-mini")],
        merger=None
    ),
    "swarm-lite": SwarmConfig(
        name="swarm-lite",
        generators=[
            GeneratorConfig(provider="openai", model="gpt-4.1-mini", temperature=0.7),
            GeneratorConfig(provider="google", model="gemini-2.5-flash", temperature=0.3),
            GeneratorConfig(provider="google", model="gemini-2.5-flash", temperature=0.7),
        ],
        merger=GeneratorConfig(provider="openai", model="gpt-4o", temperature=0.3)
    )
}

def get_swarm_config(name: str) -> SwarmConfig:
    if name not in SWARM_CONFIGS:
        raise ValueError(f"Unknown swarm config: {name}. Available: {list(SWARM_CONFIGS.keys())}")
    return SWARM_CONFIGS[name]