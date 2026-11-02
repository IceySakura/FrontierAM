"""Code MT agent — unified agent supporting tool_call and text_parse modes."""

from nexus.core import agent_registry
from nexus.extensions.projects.code_mt.agent.models import CodeMTAgentInput, CodeMTAgentOutput


async def _run_code_mt_agent(agent_input: CodeMTAgentInput) -> CodeMTAgentOutput:
    # Env prep runs once, on the first agent turn (no-op on MT retries and when
    # no setup_strategy is configured). Owned by the agent so the task stays
    # agent-agnostic.
    from nexus.extensions.projects.code_mt.agent.setup import run_turn0_setup

    await run_turn0_setup(agent_input)

    if agent_input.call_mode == 'text_parse':
        from nexus.extensions.projects.code_mt.agent.text_parse_loop import text_parse_loop

        return await text_parse_loop(agent_input)
    from nexus.extensions.projects.code_mt.agent.tool_call_loop import tool_call_loop

    return await tool_call_loop(agent_input)


@agent_registry.register(
    'code_mt',
    meta={'gateway': {'scaffold_policy': 'generic'}},
)
async def code_mt_agent(agent_input: CodeMTAgentInput) -> CodeMTAgentOutput:
    """Run the standard Code-MT agent."""
    return await _run_code_mt_agent(agent_input)


@agent_registry.register(
    'ch_code_mt',
    meta={'gateway': {'scaffold_policy': 'generic'}},
)
async def ch_code_mt_agent(agent_input: CodeMTAgentInput) -> CodeMTAgentOutput:
    """Run Code-MT under the Chinese-response policy profile."""
    agent_input = agent_input.model_copy(update={'tool_description_language': 'zh-CN'})
    return await _run_code_mt_agent(agent_input)
