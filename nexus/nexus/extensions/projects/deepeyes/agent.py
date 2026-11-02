"""DeepEyes multi-turn visual tool-calling agent.

This agent implements the DeepEyes interaction loop:
1. Receives an initial prompt (including an image embedded in the messages)
2. Calls the LLM, which may request image_zoom_in_tool calls
3. Executes tool calls and appends results to the conversation
4. Repeats until the LLM produces a final answer (no tool calls) or max_iterations is reached
"""

import json
import logging

from typing import Any

from openai.types.chat import ChatCompletionContentPartParam

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.trace import get_current_span_id, update_span


logger = logging.getLogger(__name__)

Registry.import_module('nexus.extensions.projects.deepeyes.tools')


class DeepEyesAgentInput(AgentRunInput):
    """DeepEyes agent input.

    Supports both the legacy ``prompt`` message list and the current
    ``system_prompt`` + ``user_prompt`` task contract.
    """

    prompt: list[dict[str, Any]] | None = None
    system_prompt: str | list[ChatCompletionContentPartParam] = 'You are a helpful assistant.'
    user_prompt: str | list[ChatCompletionContentPartParam] | None = None
    tools: list[str] | None = None
    max_iterations: int = 10


class DeepEyesAgentOutput(AgentRunOutput, WithMainTrajectory):
    """DeepEyes agent output containing the full conversation trajectory."""

    pass


def _build_history(agent_input: DeepEyesAgentInput) -> list[dict[str, Any]]:
    if agent_input.prompt is not None:
        return list(agent_input.prompt)
    if agent_input.user_prompt is None:
        raise ValueError('DeepEyes agent requires either prompt or user_prompt')
    return [
        {'role': 'system', 'content': agent_input.system_prompt},
        {'role': 'user', 'content': agent_input.user_prompt},
    ]


@agent_registry.register('deepeyes_agent')
async def run_deepeyes_agent(agent_input: DeepEyesAgentInput) -> DeepEyesAgentOutput:
    """DeepEyes multi-turn visual reasoning agent.

    Runs a ReAct-style loop where the LLM can call image_zoom_in_tool
    to zoom into image regions before producing its final answer.

    The source image must be provided in ExecutionContext.tool_state['image']
    (as a PIL Image) so the LOCAL tool can access it.
    """
    ctx = require_context()
    client = ctx.openai_client

    tool_names = agent_input.tools or ['deepeyes/image_zoom_in_tool']
    executor = ToolExecutor([tool_registry.get(tool_name) for tool_name in tool_names])

    tools = executor.get_tool_schemas()
    history = _build_history(agent_input)
    termination_reason = 'ENV_DONE'

    for iteration in range(agent_input.max_iterations):
        logger.debug(f'DeepEyes agent: iteration {iteration} history types={[m.get("role") for m in history]}')
        response = await client.create_completion(messages=history, tools=tools or None)
        message = response.choices[0].message
        history.append(message.model_dump(exclude_none=True))

        if message.content:
            logger.debug(f'DeepEyes agent iteration {iteration}: model output length={len(message.content)}')

        if not message.tool_calls:
            # No tool calls → model produced final answer
            logger.info(f'DeepEyes agent finished after {iteration + 1} iterations (no tool call)')
            break

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                logger.warning(f'DeepEyes agent: failed to parse tool args for {tool_name}: {exc}')
                tool_args = {}

            logger.debug(f'DeepEyes agent: calling tool {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            tool_response = await executor.call(tool_name, tool_args)

            # Build the tool result content — include any image parts returned by the tool
            result_content = tool_response.output
            if len(tool_response.output) == 1 and tool_response.output[0].get('type') == 'text':
                result_content = tool_response.output[0].get('text', '')

            logger.debug(f'DeepEyes agent: tool result content: {result_content[:-1]}')
            tool_msg: dict = {
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': result_content,
            }
            if tool_response.meta:
                tool_msg['meta'] = tool_response.meta
            history.append(tool_msg)
    else:
        logger.warning(f'DeepEyes agent: reached max_iterations={agent_input.max_iterations}')
        termination_reason = 'MAX_STEPS'

    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'_hytuner_termination_reason': termination_reason})

    return DeepEyesAgentOutput(main_trajectory=history)
