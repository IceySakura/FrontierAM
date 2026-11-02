import json
import logging

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)


logger = logging.getLogger(__name__)


Registry.import_module('nexus.extensions.projects.calculator.tools')


class CalculatorAgentInput(AgentRunInput, WithUserPrompt):
    max_iterations: int = 10
    system_prompt: str | None = None


class CalculatorAgentOutput(AgentRunOutput, WithMainTrajectory): ...


@agent_registry.register('calculator_agent', meta={'gateway': {'scaffold_policy': 'nexus_main_agent'}})
async def run_calculator_agent(agent_input: CalculatorAgentInput) -> CalculatorAgentOutput:
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([
        tool_registry.get('calculator/calculate'),
    ])

    history = [
        {'role': 'system', 'content': agent_input.system_prompt or 'You are a helpful assistant.'},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]

    for _ in range(agent_input.max_iterations):
        # Generate response
        try:
            response = await client.create_completion(messages=history, tools=executor.get_tool_schemas() or None)
        except ContextLengthExceededError:
            # NOTE(petersheng): stop like main_agent so the task still scores the run; the Gateway
            # records the refusal and labels the trajectory TRUNCATION.
            logger.warning('Context length exceeded, stopping calculator agent loop')
            break
        if not response.choices:
            logger.error('No choices returned in the response')
            break
        choice = response.choices[0]
        message = choice.message
        history.append(message.model_dump(exclude_none=True))
        if choice.finish_reason == 'length':
            # NOTE(petersheng): the response budget is spent; another turn would only be refused.
            # The Gateway already records this call as length-limited and labels the run TRUNCATION.
            logger.warning('Response hit max_tokens, stopping calculator agent loop')
            break

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

    return CalculatorAgentOutput(
        main_trajectory=history,
    )
