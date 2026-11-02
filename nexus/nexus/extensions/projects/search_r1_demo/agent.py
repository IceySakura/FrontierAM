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


logger = logging.getLogger(__name__)


Registry.import_module('nexus.extensions.projects.search_r1_demo.tools')


class SearchR1AgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = 10
    system_prompt: str | None = None


class SearchR1AgentOutput(AgentRunOutput, WithMainTrajectory): ...


@agent_registry.register('search_r1_agent')
async def run_search_r1_agent(agent_input: SearchR1AgentInput) -> SearchR1AgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([
        tool_registry.get('search_r1/search'),
    ])

    history = [
        {'role': 'system', 'content': agent_input.system_prompt or 'You are a helpful assistant.'},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    for _ in range(agent_input.max_iterations):
        # Generate response
        response = await client.create_completion(messages=history, tools=executor.get_tool_schemas() or None)
        if not response.choices:
            logger.error('No choices returned in the response')
            break
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.debug(f'Assistant: {message.content}')

        # No tool calls - conversation complete
        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                error_msg = f'Failed to parse tool arguments: {tool_call.function.arguments}'
                logger.warning(error_msg)
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': f'Error: {error_msg}',
                })
                continue

            logger.debug(f'Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            try:
                tool_response = await executor.call(tool_name, tool_args)
                output = tool_response.output
                if isinstance(output, list):
                    output = ''.join(
                        item.get('text', '') for item in output if isinstance(item, dict) and item.get('type') == 'text'
                    )
            except Exception as e:
                error_msg = f'Error executing tool {tool_name}: {e}'
                logger.warning(error_msg)
                output = f'Error: {error_msg}'

            logger.debug(f'Tool output: {output}')

            history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': output,
            })

    return SearchR1AgentOutput(main_trajectory=history)
