"""Demo agent showing how to use ToolExecutor with ExecutionContext.

This agent demonstrates:
- Accessing runtime dependencies via require_context()
- Using ToolExecutor for unified tool calling
- Handling tool calls in a conversation loop
"""

import json
import logging

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.base.schema import ToolCallAgentInput, ToolCallAgentOutput
from nexus.extensions.agents.base.toolcall import ToolCallAgent


logger = logging.getLogger(__name__)


Registry.import_module('nexus.extensions.agents.demo.tools')


class DemoAgentInput(AgentRunInput, WithUserPrompt):
    """Demo agent input - only business parameters."""

    max_iterations: int = 10


class DemoAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Demo agent output - the conversation trajectory."""

    pass


@agent_registry.register('demo_agent')
async def run_demo_agent(agent_input: DemoAgentInput) -> DemoAgentOutput:
    """Demo agent that uses ToolExecutor for tool calling.

    Shows:
    - ctx.openai_client: OpenAI-compatible client from context
    - ToolExecutor: Unified interface for LOCAL and RUNTIME tools
    - Tool schemas are auto-generated from Pydantic input models
    """
    ctx = require_context()
    client = ctx.openai_client

    # Create tool executor with demo tools
    executor = ToolExecutor([
        tool_registry.get('demo/think'),
        tool_registry.get('demo/bash'),
    ])

    system_prompt = {
        'role': 'system',
        'content': (
            'You are a helpful assistant with access to tools.\n'
            'Use the think tool to reason through problems.\n'
            'Use the bash tool to execute commands when needed.'
        ),
    }
    history = [system_prompt, {'role': 'user', 'content': agent_input.user_prompt}]

    # Get tool schemas for LLM
    tools = executor.get_tool_schemas()

    for _ in range(agent_input.max_iterations):
        response = await client.create_completion(messages=history, tools=tools or None)
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.debug(f'Model output: {message.content}')

        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            logger.debug(f'Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            tool_response = await executor.call(tool_name, tool_args)

            logger.debug(f'Tool output: {tool_response.output}')

            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': tool_response.output,
            })

    return DemoAgentOutput(main_trajectory=history)


@agent_registry.register('toolcall_agent')
async def run_toolcall_agent(agent_input: ToolCallAgentInput) -> ToolCallAgentOutput:
    """Run a ToolCallAgent with the given input."""
    executor = ToolExecutor([
        tool_registry.get('demo/think'),
        tool_registry.get('demo/bash'),
    ])
    agent = ToolCallAgent.create(agent_input, executor)
    await agent.run(agent_input.user_prompt)
    return agent.get_output()
