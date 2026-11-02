"""VideoCut multi-turn video tool-calling agent.

This agent implements the VideoCut interaction loop:
1. Receives an initial prompt (including video frames embedded in the messages)
2. Calls the LLM with standard OpenAI tool schemas
3. Executes any requested video_cut_tool calls and appends tool results
4. Repeats until the LLM produces a final answer (no tool calls) or max_iterations is reached
"""

import json
import logging

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

Registry.import_module('nexus.extensions.projects.videocut.tools')


async def _log_raw_message_debug(client, history: list[dict], tools: list[dict], tool_choice: str) -> None:
    """Log the raw SDK response for empty assistant messages without changing the framework layer."""
    raw_client = getattr(client, 'client', None)
    config = getattr(client, 'config', None)
    if raw_client is None or config is None:
        logger.debug('\033[32m[VideoCut raw debug] unavailable: client has no raw SDK handle\033[0m')
        return

    try:
        request_kwargs = {
            'messages': history,
            **config.request_args,
            'tools': tools or None,
            'tool_choice': tool_choice,
        }
        request_kwargs = client._apply_enforced_request_args(request_kwargs)
        raw_response = await raw_client.chat.completions.create(**request_kwargs)
        if hasattr(raw_response, 'model_dump'):
            logger.debug(
                '\033[32m[VideoCut raw completion payload] %s\033[0m',
                raw_response.model_dump(exclude_none=False),
            )
        elif raw_response.choices:
            raw_message = raw_response.choices[0].message
            if hasattr(raw_message, 'model_dump'):
                logger.debug(
                    '\033[32m[VideoCut raw message payload] %s\033[0m',
                    raw_message.model_dump(exclude_none=False),
                )
            else:
                logger.debug('\033[32m[VideoCut raw message payload] %s\033[0m', raw_message)
        else:
            logger.debug('\033[32m[VideoCut raw debug returned no choices] %s\033[0m', raw_response)
    except Exception as exc:
        logger.warning(f'\033[32m[VideoCut raw debug request failed] {exc}\033[0m')


class VideoCutAgentInput(AgentRunInput):
    """VideoCut agent input.

    prompt: full message list in OpenAI chat format (may include video content parts)
    max_iterations: maximum number of LLM + tool-call rounds
    """

    prompt: list[dict]
    max_iterations: int = 10


class VideoCutAgentOutput(AgentRunOutput, WithMainTrajectory):
    """VideoCut agent output containing the full conversation trajectory."""

    pass


@agent_registry.register('videocut_agent')
async def run_videocut_agent(agent_input: VideoCutAgentInput) -> VideoCutAgentOutput:
    """VideoCut multi-turn video temporal reasoning agent.

    Runs a ReAct-style loop where the LLM can call video_cut_tool to zoom into
    specific temporal segments of the video before producing its final answer.

    The source video must be provided in ExecutionContext.tool_state as:
        - 'video_tensor': torch.Tensor of shape (T, C, H, W)
        - 'video_fps': float, the fps of the video tensor
    so the LOCAL tool can access it.
    """
    ctx = require_context()
    client = ctx.openai_client

    executor = ToolExecutor([
        tool_registry.get('videocut/video_cut_tool'),
    ])
    tools = executor.get_tool_schemas()

    history = list(agent_input.prompt)
    termination_reason = 'ENV_DONE'

    for iteration in range(agent_input.max_iterations):
        tool_choice = 'required' if iteration == 0 else 'auto'
        logger.debug(
            f'VideoCut agent: iteration {iteration} '
            f'history types={[m.get("role") for m in history]} '
            f'VideoCut agent: tool_choice {tool_choice} '
        )
        response = await client.create_completion(messages=history, tools=tools or None, tool_choice=tool_choice)
        choice = response.choices[0]
        message = choice.message
        history.append(message.model_dump(exclude_none=True))

        if message.reasoning_content:
            logger.debug('\033[35m[VideoCut assistant reasoning] %s\033[0m', message.reasoning_content)
        if message.content:
            logger.debug('\033[35m[VideoCut assistant answer] %s\033[0m', message.content)

        tool_calls = list(message.tool_calls or [])

        if tool_calls:
            for tool_call in tool_calls:
                logger.debug(
                    '\033[35m[VideoCut tool_call] %s(%s)\033[0m',
                    tool_call.function.name,
                    tool_call.function.arguments,
                )

        if not message.content and not tool_calls:
            logger.debug('\033[35m[VideoCut assistant] <empty content>\033[0m')
            logger.debug('VideoCut empty assistant payload: %s', message.model_dump(exclude_none=True))
            await _log_raw_message_debug(client, history[:-1], tools, tool_choice)

        logger.debug(
            'VideoCut agent: finish_reason=%s, has_content=%s, n_tool_calls=%d',
            choice.finish_reason,
            bool(message.content),
            len(tool_calls),
        )

        if not tool_calls:
            logger.info(f'VideoCut agent finished after {iteration + 1} iterations (no tool call)')
            break

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_call_id = tool_call.id

            try:
                tool_args = json.loads(tool_args_raw)
            except json.JSONDecodeError as exc:
                logger.warning(f'VideoCut agent: failed to parse tool args for {tool_name}: {exc}')
                tool_args = {}

            logger.debug(f'VideoCut agent: calling tool {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

            tool_response = await executor.call(tool_name, tool_args)

            logger.debug(f'VideoCut agent: tool result meta={tool_response.meta}')

            result_content = tool_response.output
            if len(tool_response.output) == 1 and tool_response.output[0].get('type') == 'text':
                result_content = tool_response.output[0].get('text', '')

            tool_msg: dict = {
                'role': 'tool',
                'tool_call_id': tool_call_id,
                'content': result_content,
            }
            if tool_response.meta:
                tool_msg['meta'] = tool_response.meta
            history.append(tool_msg)
    else:
        logger.warning(f'VideoCut agent: reached max_iterations={agent_input.max_iterations}')
        termination_reason = 'MAX_STEPS'

    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'_hytuner_termination_reason': termination_reason})

    return VideoCutAgentOutput(main_trajectory=history)
