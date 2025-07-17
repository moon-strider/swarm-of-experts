from dataclasses import dataclass, field
from typing import List, Optional

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
   - Fully executes upon users' task/question

## Important Rules:
- Do NOT mention that you are analyzing multiple responses
- Do NOT reference "other answers" or "responses"
- Write as if you are directly answering the user's question
- Ensure your answer is self-contained and complete
- You may augment the final answer with something that is absent from all of the answers provided to you, if you feel that it is important to include in the final answer for it to be complete and helpful

Provide your synthesized response below:"""

TASKMASTER_PROMPT_TEMPLATE = """You are a task decomposition expert. Break the query into exactly 3 self-contained sub-prompts covering all aspects.

Original Query: {user_query}

Rules:
- Each sub-prompt must be independent and complementary
- Output your response using XML tags
- Use exactly these tags: <sub_prompt_1>, <sub_prompt_2>, <sub_prompt_3>

Example format:
<sub_prompt_1>First focused sub-task here</sub_prompt_1>
<sub_prompt_2>Second focused sub-task here</sub_prompt_2>
<sub_prompt_3>Third focused sub-task here</sub_prompt_3>"""

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
    taskmaster: Optional[GeneratorConfig] = None
    merger_prompt_template: str = MERGER_PROMPT_TEMPLATE
    max_context_tokens: int = 128000

    @property
    def has_merger(self) -> bool:
        return self.merger is not None
    
    @property
    def has_taskmaster(self) -> bool:
        return self.taskmaster is not None

SWARM_CONFIGS = {
    "basic": SwarmConfig(
        name="basic",
        generators=[GeneratorConfig(provider="openai", model="o3-pro")],
        merger=None
    ),
    "swarm-lite": SwarmConfig(
        name="swarm-lite",
        generators=[
            GeneratorConfig(provider="google", model="gemini-2.5-pro", temperature=0.7),
            GeneratorConfig(provider="google", model="gemini-2.5-pro", temperature=0.3),
            GeneratorConfig(provider="google", model="gemini-2.5-pro", temperature=0.7),
        ],
        merger=GeneratorConfig(provider="google", model="gemini-2.5-pro", temperature=0.3)
    ),
    "groq-swarm": SwarmConfig(
        name="groq-swarm",
        generators=[
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.7),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.5),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.3),
        ],
        merger=GeneratorConfig(provider="groq", model="moonshotai/kimi-k2-instruct", temperature=0.3)
    ),
    "groq-single": SwarmConfig(
        name="groq-single",
        generators=[GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b")],
        merger=None
    ),
    "groq-taskmaster": SwarmConfig(
        name="groq-taskmaster",
        generators=[
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.7),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.5),
            GeneratorConfig(provider="groq", model="deepseek-r1-distill-llama-70b", temperature=0.3),
        ],
        taskmaster=GeneratorConfig(provider="groq", model="moonshotai/kimi-k2-instruct", temperature=0.3),
        merger=GeneratorConfig(provider="groq", model="moonshotai/kimi-k2-instruct", temperature=0.3)
    )
}

def get_swarm_config(name: str) -> SwarmConfig:
    if name not in SWARM_CONFIGS:
        raise ValueError(f"Unknown swarm config: {name}. Available: {list(SWARM_CONFIGS.keys())}")
    return SWARM_CONFIGS[name]


def get_all_swarm_configs() -> dict:
    return SWARM_CONFIGS