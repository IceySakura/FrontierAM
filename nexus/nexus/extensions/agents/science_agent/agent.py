"""
Science agent with stateful search tools.
Based on info_seek agent but using stateful search tools.
Supports configurable code interpreter and search tools.
"""

import json
import logging

from typing import Any, Literal

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithMainTrajectory,
    WithUserPrompt,
    agent_registry,
    get_tool_meta,
    require_context,
    tool_registry,
)
from nexus.trace import get_current_span_id, update_span
from nexus.utils.tool_schema_overrides import load_tool_schema_overrides


logger = logging.getLogger(__name__)

# Import required tools
Registry.import_module('nexus.extensions.tools.code_interpreter')
Registry.import_module('nexus.extensions.agents.base.tools')


class ScienceAgentInput(AgentRunInput, WithUserPrompt):
    """Science agent input.

    Attributes:
        max_iterations: Maximum number of agent iterations
        code_interpreter: Code interpreter type - 'none', 'python', or 'jupyter'
        enable_search: Whether to enable search tools
    """

    max_iterations: int = 10
    # Code interpreter: none=disabled, python=python_code_interpreter, jupyter=jupyter_code_interpreter
    code_interpreter: Literal['none', 'python', 'jupyter'] = 'jupyter'
    # Whether to enable search tools
    enable_search: bool = True
    # Override default tool schemas with external OpenAI-format schemas.
    # Supports either a list of OpenAI-format schemas or a path to a YAML/JSON file.
    tool_schema_overrides: list[dict[str, Any]] | str | None = None
    # LLM tool name -> executor tool name mapping (e.g. {"python": "jupyter_code_interpreter"})
    tool_name_map: dict[str, str] | None = None


class ScienceAgentOutput(AgentRunOutput, WithMainTrajectory):
    """Science agent output with trajectory."""

    final_answer: str
    rendered_trajectory: str = ''


def build_system_prompt(tool_schemas: str) -> str:
    """Build system prompt for science agent."""
    # Use f-string with explicit placeholder to avoid issues with JSON braces in tool_schemas
    return f"""\
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schemas}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
"""


def _append_tool_message(history: list[dict], tool_call_id: str, content: str) -> None:
    """Append a tool role message to history."""
    history.append({
        'role': 'tool',
        'tool_call_id': tool_call_id,
        'content': content,
    })


def _extract_tool_output_text(output: list[Any] | None) -> str:
    """Extract and concatenate text items from a tool response payload."""
    if not output:
        return ''

    text_parts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            item_type = item.get('type')
            item_text = item.get('text')
        else:
            item_type = getattr(item, 'type', None)
            item_text = getattr(item, 'text', None)

        if item_type == 'text' and isinstance(item_text, str):
            text_parts.append(item_text)

    return ''.join(text_parts)


def _render_trajectory(history: list[dict]) -> str:
    """Render trajectory in HY3.0-style text format for comparison with HyTuner output.

    Format:
        user:      <|user|>\n{content}\n
        assistant: <|assistant|>\n<think>{reasoning}</think>{content}{tool_calls}\n
        tool:      <|tool|>\n{content}\n
    """
    parts: list[str] = []
    for msg in history:
        role = msg.get('role', '')
        content = msg.get('content') or ''

        if role == 'user':
            parts.append(f'<|user|>\n{content}\n')

        elif role == 'assistant':
            reasoning_content = msg.get('reasoning_content') or ''
            raw_response = msg.get('raw_response') or ''
            tool_calls = msg.get('tool_calls') or []

            if raw_response:
                # raw_response already contains <think> and tool call tokens
                parts.append(f'<|assistant|>\n{raw_response}\n')
            else:
                text = ''
                if reasoning_content:
                    text += f'<think>{reasoning_content}</think>'
                text += content

                # Format tool calls in HY3.0 style
                if tool_calls:
                    tc_parts: list[str] = []
                    for tc in tool_calls:
                        func = tc.get('function', {}) if isinstance(tc, dict) else {}
                        name = func.get('name', '')
                        args_str = func.get('arguments', '{}')
                        try:
                            args = json.loads(args_str)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        arg_text = ''.join(f'<arg_key>{k}<arg_value>{v}' for k, v in args.items())
                        tc_parts.append(f'<tool_call><tool_sep>{name}{arg_text}</tool_call>')
                    text += '<tool_calls>' + ''.join(tc_parts) + '</tool_calls>'

                parts.append(f'<|assistant|>\n{text}\n')

        elif role == 'tool':
            parts.append(f'<|tool|>\n{content}\n')

    return ''.join(parts)


async def _release_runtime(reason: str = 'unknown') -> None:
    """Release all runtimes so the next tool call provisions a fresh sandbox."""
    ctx = require_context()
    if ctx.runtime_provider is not None:
        logger.warning(f'Releasing runtime: {reason}')
        await ctx.runtime_provider.release_all()
        ctx._runtimes.clear()


@agent_registry.register('science_agent')
async def run_science_agent(
    agent_input: ScienceAgentInput,
) -> ScienceAgentOutput:
    """Run science agent with configurable tools.

    Args:
        agent_input: Agent configuration including max_iterations, code_interpreter, enable_search, etc.

    Returns:
        Agent output with trajectory and final answer
    """
    ctx = require_context()
    client = ctx.openai_client

    # Set origin question in tool_state for leak detection
    if ctx.tool_state.get('origin_question') is None:
        ctx.tool_state['origin_question'] = agent_input.user_prompt

    # Build tool list based on config
    tools = []

    # Search tools
    if agent_input.enable_search:
        tools.extend([
            tool_registry.get('search_stateful/search'),
            tool_registry.get('search_stateful/open'),
            tool_registry.get('search_stateful/find'),
        ])

    # Code interpreter
    if agent_input.code_interpreter == 'jupyter':
        tools.append(tool_registry.get('jupyter_code_interpreter'))
        logger.debug('Using jupyter_code_interpreter (stateful)')

        # Default mapping for python -> jupyter_code_interpreter
        if agent_input.tool_name_map is None:
            agent_input.tool_name_map = {}
        agent_input.tool_name_map.setdefault('python', 'jupyter_code_interpreter')
    elif agent_input.code_interpreter == 'python':
        tools.append(tool_registry.get('python_code_interpreter'))
        logger.debug('Using python_code_interpreter (stateless with timeout)')
    else:
        logger.debug('No code interpreter enabled')

    logger.debug(f'Configured tools: {[get_tool_meta(t).name for t in tools]}')

    executor = ToolExecutor(tools)

    # Get tool schemas for LLM
    tool_schemas = executor.get_tool_schemas()

    # Override schemas if provided
    resolved_tool_schema_overrides = load_tool_schema_overrides(agent_input.tool_schema_overrides)
    if resolved_tool_schema_overrides:
        tool_schemas = resolved_tool_schema_overrides
        logger.debug(f'Using overridden tool schemas: {[s.get("function", {}).get("name", "?") for s in tool_schemas]}')

    history = [
        {'role': 'user', 'content': f'{agent_input.user_prompt}'},
    ]

    # Main agent loop
    final_answer = ''
    termination_reason = None

    for iteration in range(agent_input.max_iterations):
        try:
            response = await client.create_completion(messages=history, tools=tool_schemas or None)
            message = response.choices[0].message
            history.append(message.model_dump(exclude_none=True))

            if message.content:
                logger.debug(f'Model output: {message.content}')
                final_answer = message.content

            if not message.tool_calls:
                # Stop if no tool calls (agent provided final answer)
                logger.info('No tool calls in response, stopping agent loop')
                termination_reason = 'ENV_DONE'
                break

            # Process tool calls
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Map tool name if mapping exists (e.g., "python" -> "jupyter_code_interpreter")
                dispatch_name = tool_name
                if agent_input.tool_name_map and tool_name in agent_input.tool_name_map:
                    dispatch_name = agent_input.tool_name_map[tool_name]

                logger.debug(f'Tool call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})')

                try:
                    tool_response = await executor.call(dispatch_name, tool_args)
                except Exception as e:
                    raw_output = getattr(message, 'raw', None) or 'N/A'

                    logger.error(
                        f'\n{"=" * 80}\n'
                        f'TOOL EXECUTION ERROR DETAILS\n'
                        f'{"=" * 80}\n'
                        f'Tool Name: {tool_name}\n'
                        f'Dispatch Name: {dispatch_name}\n'
                        f'Function Name: {tool_call.function.name}\n'
                        f'Raw Arguments: {tool_call.function.arguments}\n'
                        f'Parsed Arguments: {tool_args}\n'
                        f'Error Type: {type(e).__name__}\n'
                        f'Error Message: {str(e)}\n'
                        f'Model Raw Output: {raw_output}\n'
                        f'{"=" * 80}\n'
                    )

                    error_msg = str(e)
                    tool_error_text = f'Error executing tool: {e}'
                    if 'OOMKilled' in error_msg:
                        await _release_runtime(reason='OOMKilled during tool execution')
                        tool_error_text += '\nThe sandbox has been destroyed due to OOM. A fresh sandbox will be created for the next tool call. All previous state (variables, files, processes) has been lost.'

                    _append_tool_message(history, tool_call.id, tool_error_text)
                    continue

                logger.debug(f'Tool output: {tool_response.output}')
                tool_output_text = _extract_tool_output_text(tool_response.output)
                _append_tool_message(history, tool_call.id, tool_output_text)
        except ContextLengthExceededError as e:
            # Stop gracefully when context length is exceeded
            logger.warning(f'Context length exceeded at iteration {iteration + 1}/{agent_input.max_iterations}: {e}')
            logger.info('Stopping agent loop due to context length limit')
            termination_reason = 'TRUNCATION'
            # Use the last final_answer if available, otherwise indicate context limit reached
            if not final_answer:
                final_answer = (
                    'Agent stopped due to context length limit. '
                    'Please try with a smaller max_iterations or a model with larger context window.'
                )
            break

    if termination_reason is None:
        termination_reason = 'MAX_STEPS'
        logger.warning(f'Agent reached max iterations ({agent_input.max_iterations})')

    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                '_hytuner_termination_reason': termination_reason,
            },
        )

    return ScienceAgentOutput(
        main_trajectory=history,
        final_answer=final_answer or 'No answer provided.',
        rendered_trajectory=_render_trajectory(history),
    )
