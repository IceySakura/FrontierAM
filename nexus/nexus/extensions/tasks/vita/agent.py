"""VitaBench agent entry point for nexus framework.

Implements the three-party orchestration loop (Agent <-> User <-> Environment)
for VitaBench life-service benchmarking.
"""

import logging

from typing import Any

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
)
from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.orchestrator import AsyncOrchestrator, VitaLLMAgent
from nexus.extensions.tasks.vita.user_simulator import UserSimulator


logger = logging.getLogger(__name__)


class VitaAgentInput(AgentRunInput, WithUserPrompt):
    """Input for the VitaBench agent."""

    max_steps: int = Field(default=300, description='Maximum orchestration steps')
    max_errors: int = Field(default=10, description='Maximum tool call errors before termination')
    enable_think: bool = Field(default=False, description='Enable thinking/reasoning mode for agent LLM')
    seed: int | None = Field(default=300, description='Random seed for reproducibility')
    user_llm_kwargs: dict = Field(default_factory=dict, description='Extra kwargs for user simulator LLM')
    agent_llm_kwargs: dict = Field(default_factory=dict, description='Extra kwargs for agent LLM')


class VitaAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Output from the VitaBench agent."""

    simulation: dict | None = Field(default=None, description='Serialized SimulationRun')
    termination_reason: str = ''
    agent_cost: float | None = None
    user_cost: float | None = None


def _convert_trajectory(messages: list) -> list[dict[str, Any]]:
    """Convert VitaBench messages to nexus trajectory format."""
    trajectory = []
    for msg in messages:
        entry: dict[str, Any] = {'role': getattr(msg, 'role', 'unknown')}
        content = getattr(msg, 'content', None)
        if content is not None:
            entry['content'] = content
        tool_calls = getattr(msg, 'tool_calls', None)
        if tool_calls:
            entry['tool_calls'] = [{'id': tc.id, 'name': tc.name, 'arguments': tc.arguments} for tc in tool_calls]
        if hasattr(msg, 'name') and msg.name:
            entry['name'] = msg.name
        if hasattr(msg, 'raw_data') and msg.raw_data:
            reasoning = msg.raw_data.get('message', {}).get('reasoning_content')
            if reasoning:
                entry['reasoning_content'] = reasoning
        trajectory.append(entry)
    return trajectory


@agent_registry.register('vita_agent')
async def run_vita_agent(agent_input: VitaAgentInput) -> VitaAgentOutput:
    """VitaBench agent entry point: runs the three-party orchestration loop."""
    ctx = require_context()
    task_config = ctx.task_config

    task_data = task_config.get('task', {})
    domain = task_config.get('domain', 'ota')
    language = task_config.get('language', 'chinese')

    task = Task.model_validate(task_data)

    environment = ctx.tool_state.get('vita_env')
    if environment is None:
        raise RuntimeError('vita_env not found in tool_state -- must be set by the task')

    system_prompt = task_config.get('system_prompt', '')

    agent_client = ctx.get_openai_client('main')
    vita_agent = VitaLLMAgent(
        openai_client=agent_client,
        tools=environment.get_tools(),
        system_prompt=system_prompt,
        enable_think=agent_input.enable_think,
        llm_kwargs=agent_input.agent_llm_kwargs,
    )

    user_client_name = 'user' if 'user' in (ctx.openai_clients or {}) else 'main'
    user_client = ctx.get_openai_client(user_client_name)
    user_sim = UserSimulator(
        openai_client=user_client,
        persona=str(task.user_scenario.user_profile) if task.user_scenario else '',
        instructions=str(task.instructions) if task.instructions else '',
        language=language,
        llm_kwargs=agent_input.user_llm_kwargs,
    )

    orchestrator = AsyncOrchestrator(
        domain=domain,
        agent=vita_agent,
        user=user_sim,
        environment=environment,
        task=task,
        max_steps=agent_input.max_steps,
        max_errors=agent_input.max_errors,
        seed=agent_input.seed,
        language=language,
    )

    simulation = await orchestrator.run()

    sim_dict = simulation.model_dump()
    main_trajectory = _convert_trajectory(simulation.messages)

    return VitaAgentOutput(
        simulation=sim_dict,
        main_trajectory=main_trajectory,
        termination_reason=simulation.termination_reason or '',
        agent_cost=simulation.agent_cost,
        user_cost=simulation.user_cost,
    )
