import logging

from nexus.core import ToolExecutor, agent_registry, tool_registry
from nexus.extensions.agents.base.schema import (
    ToolCallAgentInput,
    ToolCallAgentOutput,
)
from nexus.extensions.agents.base.toolcall import ToolCallAgent


logger = logging.getLogger(__name__)


def _resolve_tools(tools: list[str] | None = None) -> list:
    # the order matters -- it decide the order of tool schema in prompts
    tools = tools or [
        'edu_one_agent/web_search',
        'edu_one_agent/edu_search',
    ]
    return [tool_registry.get(name) for name in tools]


class EduOneAgent(ToolCallAgent):
    name: str = 'edu_one_agent'
    description: str = 'An one agent for Edu'


@agent_registry.register('edu_one_agent')
async def run_edu_one_agent(agent_input: ToolCallAgentInput) -> ToolCallAgentOutput:
    executor = ToolExecutor(_resolve_tools(agent_input.tools))
    agent = EduOneAgent.create(agent_input, executor=executor)
    await agent.run(agent_input.user_prompt)
    return agent.get_output()
