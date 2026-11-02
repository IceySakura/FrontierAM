"""Terminus2 agent implementation for nexus framework.

This agent implements a prompt-based terminal interaction loop without tools.
It supports both JSON and XML formatted responses for sending keystrokes to
a terminal session.
"""

import asyncio
import json
import logging
import time

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from pydantic import Field
from tenacity import retry, retry_if_exception_type, retry_if_not_exception_type, stop_after_attempt

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    WithMainTmuxSessionId,
    WithUserPrompt,
    agent_registry,
    require_context,
)
from nexus.core.openai_client import ContextLengthExceededError
from nexus.extensions.agents.terminus2.chat import (
    Chat,
    LLMResponse,
    OutputLengthExceededError,
    count_total_tokens,
    lite_llm_call,
)
from nexus.extensions.agents.terminus2.parsing import TerminusJSONParser
from nexus.extensions.agents.terminus2.prompts import (
    SWE_GENERIC_INSTANCE_TEMPLATE,
    SWE_PYTHON_INSTANCE_TEMPLATE,
    get_response_type_label,
    get_system_template,
)
from nexus.extensions.agents.terminus2.summarization import get_model_context_limit, get_model_output_limit, summarize
from nexus.extensions.agents.terminus2.terminal_output import (
    Terminus2State,
    get_completion_confirmation_message,
    get_incremental_output,
    limit_output_length,
)
from nexus.extensions.agents.terminus2.xml_parsing import TerminusXMLParser
from nexus.trace import get_current_span_id, span_context, start_span, update_span
from nexus.trace.types import Status
from nexus.utils.timeout import with_timeout


# ============ Data Models ============


class ExitStatus(str, Enum):
    """Exit status for Terminus2 agent."""

    COMPLETED = 'completed'
    """Normal exit - agent marked task as complete"""

    MAX_ITERATIONS = 'max_iterations'
    """Agent reached maximum iterations limit"""

    TRUNCATION = 'truncation'
    """Agent hit context length limit and all summarization fallbacks failed"""

    ERROR = 'error'
    """Abnormal exit - agent encountered an error"""

    TIMEOUT = 'timeout'
    """Abnormal exit - agent encountered a timeout error"""


EXIT_STATUS_TO_TERMINATION_REASON = {
    ExitStatus.COMPLETED: 'ENV_DONE',
    ExitStatus.MAX_ITERATIONS: 'MAX_STEPS',
    ExitStatus.TRUNCATION: 'TRUNCATION',
    ExitStatus.ERROR: 'UNHANDLED_ERROR',
    ExitStatus.TIMEOUT: 'TASK_TIMEOUT',
}


class Terminus2Input(AgentRunInput, WithUserPrompt):
    """Input configuration for Terminus2 agent."""

    max_iterations: int = Field(default=200, description='Maximum number of agent steps')
    tmux_pane_width: int = Field(default=160, description='Terminal pane width')
    tmux_pane_height: int = Field(default=40, description='Terminal pane height')
    parser_name: Literal['json', 'xml'] = Field(default='json', description='Response format parser: "json" or "xml"')
    enable_summarize: bool = Field(
        default=False, description='Enable context summarization when approaching token limit'
    )
    proactive_summarization_threshold: int = Field(
        default=8000, description='Number of free tokens below which to trigger proactive summarization'
    )
    model_name: str = Field(default='', description='Model name for token counting and context limit lookup')
    context_limit: int | None = Field(
        default=None,
        description='Model context limit in tokens. If None, will try to auto-detect from litellm or use fallback (200000)',
    )


class Terminus2Output(AgentRunOutput, WithMainTmuxSessionId):
    """Output from Terminus2 agent run."""

    exit_status: ExitStatus = Field(description='How the agent exited')
    n_iterations: int = Field(default=0, description='Number of iterations completed')


@dataclass
class Command:
    """A command to execute in the terminal."""

    keystrokes: str
    duration_sec: float


# ============ Agent Logic ============


logger = logging.getLogger(__name__)


def _get_parser(parser_name: Literal['json', 'xml'] = 'json') -> TerminusJSONParser | TerminusXMLParser:
    """Return the appropriate parser instance for the given format."""
    if parser_name == 'xml':
        return TerminusXMLParser()
    return TerminusJSONParser()


async def execute_commands(commands: list[Command], session_id: str, state: Terminus2State) -> tuple[bool, str]:
    """Execute a batch of commands in the terminal.

    Args:
        commands: List of commands to execute
        session_id: Terminal session ID
        state: State object for tracking terminal buffer

    Returns:
        Tuple of (timeout_occurred, terminal_output)
    """
    runtime = await require_context().get_runtime()

    for command in commands:
        await runtime.send_keys_to_terminal_session(session_id=session_id, keys=command.keystrokes)
        await asyncio.sleep(command.duration_sec)

    return False, limit_output_length(await get_incremental_output(session_id, state))


async def check_proactive_summarization(
    chat: Chat,
    original_instruction: str,
    model_name: str,
    session_id: str,
    threshold: int = 8000,
    context_limit: int | None = None,
) -> tuple[str, list] | None:
    """Check if we should proactively summarize due to token usage."""

    context_limit = context_limit or get_model_context_limit(model_name)
    current_tokens = count_total_tokens(chat, model_name)
    free_tokens = context_limit - current_tokens

    if free_tokens < threshold:
        logger.debug(f'Proactively summarizing. Free tokens: approximately {free_tokens}')
        try:
            summary_prompt, subagent_trajectory_refs = await summarize(
                chat, original_instruction, session_id=session_id
            )
            return (summary_prompt, subagent_trajectory_refs)
        except Exception as e:
            logger.exception(f'Error in proactively summarizing: {e}')

    return None


def _unwind_messages_to_free_tokens(
    chat: Chat,
    target_free_tokens: int = 4000,
    model_name: str = '',
    context_limit: int | None = None,
) -> None:
    """Remove recent messages until we have enough free tokens."""
    context_limit = context_limit or get_model_context_limit(model_name)

    while len(chat.messages) > 1:  # Keep at least the first message
        current_tokens = count_total_tokens(chat, model_name)
        free_tokens = context_limit - current_tokens

        if free_tokens >= target_free_tokens:
            break

        # Remove the most recent pair of messages (user + assistant)
        if len(chat.messages) >= 2:
            chat._messages = chat.messages[:-2]
        else:
            break

    free_tokens = context_limit - count_total_tokens(chat, model_name)
    logger.debug(
        f'Unwound messages. Remaining messages: {len(chat.messages)}, Free tokens: approximately {free_tokens}'
    )


@retry(
    stop=stop_after_attempt(3),
    retry=(
        retry_if_not_exception_type((ContextLengthExceededError, OutputLengthExceededError))
        & retry_if_exception_type(Exception)
    ),
    reraise=True,
)
async def _query_llm(
    chat: Chat,
    prompt: str,
    original_instruction: str = '',
    model_name: str = '',
    enable_summarize: bool = True,
    session_id: str = '',
    parser_name: Literal['json', 'xml'] = 'json',
    _output_length_retry: int = 0,
    context_limit: int | None = None,
):
    try:
        return await chat.chat(prompt)

    except ContextLengthExceededError:
        if not enable_summarize:
            logger.debug('Context length exceeded and summarization is OFF.')
            raise
        logger.debug('Context length exceeded. Using fallback summarization.')

        _unwind_messages_to_free_tokens(
            chat, target_free_tokens=4000, model_name=model_name, context_limit=context_limit
        )

        summary_prompt = None

        # Fallback 1: Try full summary (three-step summarization process)
        try:
            logger.debug('SUMMARIZATION: Attempting full summary')
            summary_prompt, _ = await summarize(chat, original_instruction, session_id)
            logger.debug('SUMMARIZATION: Full summary succeeded')
        except Exception as e:
            logger.debug(f'SUMMARIZATION: Full summary failed: {e}')

        # Fallback 2: Try short summary
        if summary_prompt is None:
            try:
                logger.debug('SUMMARIZATION: Attempting short summary')
                runtime = await require_context().get_runtime()
                current_screen = (
                    await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)
                ).content
                limited_screen = current_screen[-1000:] if current_screen else ''

                short_prompt = f'Briefly continue this task: {original_instruction}\n\nCurrent state: {limited_screen}\n\nNext steps (2-3 sentences):'

                short_llm_response = await lite_llm_call(prompt=short_prompt)

                summary_prompt = f'{original_instruction}\n\nSummary: {short_llm_response.content}'
                logger.debug('SUMMARIZATION: Short summary succeeded')
            except Exception as e:
                logger.debug(f'SUMMARIZATION: Short summary failed: {e}')

        # Fallback 3: Ultimate fallback (no LLM)
        if summary_prompt is None:
            logger.debug('SUMMARIZATION: Using ultimate fallback')
            runtime = await require_context().get_runtime()
            current_screen = (await runtime.capture_terminal_pane(session_id=session_id, capture_entire=False)).content
            limited_screen = current_screen[-1000:] if current_screen else ''
            summary_prompt = f'{original_instruction}\n\nCurrent state: {limited_screen}'

        try:
            llm_response = await chat.chat(summary_prompt)
        except Exception as e:
            logger.exception(f'Even fallback LLM call failed: {e}')
            llm_response = LLMResponse(content='Technical difficulties. Please continue with the task.')

        return llm_response

    except OutputLengthExceededError as e:
        logger.debug(f'Output length exceeded: {e}')

        if _output_length_retry >= 3:
            logger.warning('Output length exceeded retry limit, raising.')
            raise

        truncated_response = getattr(e, 'truncated_response', '[TRUNCATED RESPONSE NOT AVAILABLE]')
        truncated_message = getattr(e, 'raw_message', None)
        if not isinstance(truncated_message, dict):
            truncated_message = {'role': 'assistant', 'content': truncated_response}

        # Try to salvage a valid response from truncated output (XML parser only)
        parser = _get_parser(parser_name)
        if isinstance(parser, TerminusXMLParser):
            salvaged_response, _has_multiple_blocks = parser.salvage_truncated_response(truncated_response)
            if salvaged_response:
                logger.debug('Output exceeded length but found valid response via salvage.')
                chat.messages.append({'role': 'user', 'content': prompt})
                chat.messages.append(truncated_message)
                return LLMResponse(
                    content=salvaged_response,
                    raw_message=truncated_message,
                )

        warnings_text = ''
        try:
            parse_result = parser.parse_response(truncated_response)
            if parse_result.warning:
                warnings_text = f'\n\nParser warnings from your truncated response:\n{parse_result.warning}'
        except Exception as parse_error:
            logger.debug(f'Failed to parse truncated response: {parse_error}')

        output_limit = get_model_output_limit(model_name)
        limit_str = f'{output_limit} tokens' if output_limit is not None else 'the maximum output length'

        error_msg = (
            'ERROR!! NONE of the actions you just requested were performed '
            f'because you exceeded {limit_str}. '
            f'Your outputs must be less than {limit_str}. Re-issue this request, '
            f'breaking it into chunks each of which is less than {limit_str}.'
        )

        if warnings_text:
            error_msg += warnings_text

        chat.messages.append({'role': 'user', 'content': prompt})
        chat.messages.append(truncated_message)

        return await _query_llm(
            chat,
            error_msg,
            original_instruction,
            model_name,
            enable_summarize,
            session_id,
            parser_name=parser_name,
            _output_length_retry=_output_length_retry + 1,
            context_limit=context_limit,
        )

    except Exception as e:
        logger.exception(f'Unknown Error in LLM interaction: {e}')
        raise


async def _handle_llm_interaction(
    chat: Chat,
    prompt: str,
    original_instruction: str = '',
    model_name: str = '',
    enable_summarize: bool = True,
    session_id: str = '',
    parser_name: Literal['json', 'xml'] = 'json',
    context_limit: int | None = None,
) -> tuple[list[Command], bool, str, str, str, LLMResponse]:
    llm_response = await _query_llm(
        chat,
        prompt,
        original_instruction,
        model_name,
        enable_summarize,
        session_id,
        parser_name=parser_name,
        context_limit=context_limit,
    )

    parser = _get_parser(parser_name)
    result = parser.parse_response(llm_response.content)

    # Build feedback from errors and warnings
    feedback = ''
    if result.error:
        feedback += f'ERROR: {result.error}'
        if result.warning:
            feedback += f'\nWARNINGS: {result.warning}'
    elif result.warning:
        feedback += f'WARNINGS: {result.warning}'

    if result.warning:
        logger.warning(f'Parser warnings: {result.warning}')

    # Convert parsed commands to Command objects
    commands: list[Command] = []
    for parsed_cmd in result.commands:
        commands.append(
            Command(
                keystrokes=parsed_cmd.keystrokes,
                duration_sec=min(parsed_cmd.duration, 60),
            )
        )

    return (
        commands,
        result.is_task_complete,
        feedback,
        result.analysis,
        result.plan,
        llm_response,
    )


def _finish_iter_span(span) -> None:
    """Finish an iteration span with OK status and return None for reassignment."""
    if span:
        span.update(status=Status(code='OK'))
        span.finish()


async def _run_agent_loop(
    initial_prompt: str,
    agent_input: Terminus2Input,
    terminal_state_tracker: Terminus2State | None = None,
    session_id: str = '',
    observation_time_timeout: float | None = None,
    per_observation_timeout: float | None = None,
    timeline_collect_tool_args: bool = False,
    timeline_tool_args_min_duration: float = 0,
) -> tuple[int, ExitStatus, float]:
    enable_summarize = agent_input.enable_summarize
    original_instruction = (
        agent_input.user_prompt if isinstance(agent_input.user_prompt, str) else str(agent_input.user_prompt)
    )
    prompt = initial_prompt
    model_name = agent_input.model_name
    context_limit = agent_input.context_limit
    parser_name = agent_input.parser_name
    response_type_label = get_response_type_label(parser_name)
    terminal_state_tracker = terminal_state_tracker or Terminus2State()

    resolved_context_limit = context_limit or get_model_context_limit(model_name)
    logger.debug(
        f'Terminus2 config: model_name={model_name!r}, context_limit={context_limit}, '
        f'resolved_context_limit={resolved_context_limit}, enable_summarize={enable_summarize}, '
        f'proactive_threshold={agent_input.proactive_summarization_threshold}'
    )

    chat = Chat()

    _pending_completion = False

    n_episode = -1
    total_observation_time = 0.0
    start_time = None
    iter_span = None
    try:
        for n_episode in range(1, agent_input.max_iterations + 1):
            iter_span = start_span(
                'agent.iteration',
                attributes={
                    'type': 'agent_iteration',
                    'iteration': n_episode,
                },
            )
            logger.debug(f'=== Terminus2 STEP {n_episode} ===')

            if original_instruction and enable_summarize:
                proactive_summary_result = await check_proactive_summarization(
                    chat,
                    original_instruction,
                    model_name,
                    session_id,
                    threshold=agent_input.proactive_summarization_threshold,
                    context_limit=context_limit,
                )
                if proactive_summary_result:
                    prompt, _ = proactive_summary_result

            with span_context(
                'agent.inference',
                attributes={
                    'type': 'agent_inference',
                    'iteration': n_episode,
                },
            ):
                (
                    commands,
                    is_task_complete,
                    feedback,
                    _analysis,
                    _plan,
                    _llm_response,
                ) = await _handle_llm_interaction(
                    chat,
                    prompt,
                    original_instruction,
                    model_name,
                    enable_summarize,
                    session_id,
                    parser_name=parser_name,
                    context_limit=context_limit,
                )

            if feedback and 'ERROR:' in feedback:
                prompt = (
                    f'Previous response had parsing errors:\n{feedback}\n\n'
                    f'Please fix these issues and provide a proper '
                    f'{response_type_label}.'
                )
                _finish_iter_span(iter_span)
                iter_span = None
                continue

            start_time = time.monotonic()
            observation_span_attrs: dict = {
                'type': 'agent_observation',
                'iteration': n_episode,
            }
            # 如果阈值为0，立即记录；否则延迟到执行完成后根据耗时判断
            if timeline_collect_tool_args and timeline_tool_args_min_duration <= 0:
                observation_span_attrs['hytuner.tool_args'] = json.dumps(
                    [{'keystrokes': cmd.keystrokes, 'duration_sec': cmd.duration_sec} for cmd in commands],
                    ensure_ascii=False,
                )
            observation_span = start_span(
                'agent.observation',
                observation_span_attrs,
            )
            try:
                async with with_timeout(per_observation_timeout):
                    _timeout_occurred, terminal_output = await execute_commands(
                        commands, session_id, terminal_state_tracker
                    )
                observation_span.update(status=Status(code='OK'))
            except Exception as e:
                observation_span.update(status=Status(code='ERROR', message=str(e)))
                raise
            finally:
                # 如果设置了阈值且执行时间超过阈值，补充记录 tool_args
                if timeline_collect_tool_args and timeline_tool_args_min_duration > 0:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeline_tool_args_min_duration:
                        observation_span.update(
                            attributes={
                                'hytuner.tool_args': json.dumps(
                                    [
                                        {'keystrokes': cmd.keystrokes, 'duration_sec': cmd.duration_sec}
                                        for cmd in commands
                                    ],
                                    ensure_ascii=False,
                                ),
                            }
                        )
                observation_span.finish()

            total_observation_time += time.monotonic() - start_time
            if observation_time_timeout and total_observation_time > observation_time_timeout:
                _finish_iter_span(iter_span)
                iter_span = None
                return n_episode, ExitStatus.TIMEOUT, total_observation_time

            was_pending_completion = _pending_completion

            if is_task_complete:
                if was_pending_completion:
                    observation = terminal_output
                else:
                    _pending_completion = True
                    observation = get_completion_confirmation_message(terminal_output, parser_name)
            else:
                _pending_completion = False
                if feedback and 'WARNINGS:' in feedback:
                    observation = (
                        f'Previous response had warnings:\n{feedback}\n\n{limit_output_length(terminal_output)}'
                    )
                else:
                    observation = limit_output_length(terminal_output)

            if is_task_complete:
                if was_pending_completion:
                    _finish_iter_span(iter_span)
                    iter_span = None
                    return n_episode, ExitStatus.COMPLETED, total_observation_time
                prompt = observation
                _finish_iter_span(iter_span)
                iter_span = None
                continue

            prompt = observation

            _finish_iter_span(iter_span)
            iter_span = None

        return n_episode, ExitStatus.MAX_ITERATIONS, total_observation_time
    except ContextLengthExceededError as e:
        logger.error(f'Context length exceeded after all summarization fallbacks: {e}')
        return n_episode, ExitStatus.TRUNCATION, total_observation_time
    except (TimeoutError, asyncio.TimeoutError) as e:
        if start_time:
            total_observation_time += time.monotonic() - start_time
        logger.warning(f'Observation timeout reached: {e}')
        return n_episode, ExitStatus.TIMEOUT, total_observation_time
    except Exception as e:
        logger.exception(f'Unknown Error in Agent Loop: {e}')
        return n_episode, ExitStatus.ERROR, total_observation_time
    finally:
        if iter_span:
            iter_span.finish()


def _format_initial_terminal_state(terminal_state: str) -> str:
    import re

    # Strip all initialization lines (nexus_capture, Capture session started, export PS1=)
    # between "Current Terminal Screen:\n" and the line after "export PS1=".
    # The number of lines varies depending on shell echo behavior, so use
    # non-greedy match to consume everything up through the export PS1 line.
    pattern = r'(Current Terminal Screen:\n)(?:.*\n)*?.*export PS1=.*\n'
    return re.sub(pattern, r'\1\n', terminal_state)


@agent_registry.register('terminus2')
async def run_terminus2_agent(agent_input: Terminus2Input) -> Terminus2Output:
    ctx = require_context()
    runtime = await ctx.get_runtime()

    create_response = await runtime.create_terminal_session(
        height=agent_input.tmux_pane_height,
        width=agent_input.tmux_pane_width,
    )
    session_id = create_response.session_id

    terminal_state_tracker = Terminus2State()

    await runtime.send_keys_to_terminal_session(session_id=session_id, keys=[r"export PS1='\u@\h:\w\$ '", 'Enter'])
    await asyncio.sleep(1)

    terminal_state = limit_output_length(await get_incremental_output(session_id, terminal_state_tracker))
    terminal_state = _format_initial_terminal_state(terminal_state)

    system_template = get_system_template(agent_input.parser_name)

    initial_prompt = system_template.format(
        instruction=agent_input.user_prompt,
        terminal_state=terminal_state,
    )

    # swe task
    if ctx.task_name and 'swe' in ctx.task_name:
        working_dir = ctx.task_config.get('working_dir')
        if not working_dir:
            raise ValueError('working_dir is required in task_config for swe_task')
        problem_statement = ctx.task_config.get('problem_statement')
        if not problem_statement:
            raise ValueError('problem_statement is required in task_config for swe_task')

        language = ctx.task_config.get('language', 'python')
        template = SWE_PYTHON_INSTANCE_TEMPLATE if language == 'python' else SWE_GENERIC_INSTANCE_TEMPLATE
        initial_prompt = system_template.format(
            instruction=template.format(
                working_dir=working_dir,
                problem_statement=problem_statement,
            ),
            terminal_state=terminal_state,
        )
    observation_time_timeout = ctx.task_config.get('observation_time_timeout', None)
    per_observation_timeout = ctx.task_config.get('per_observation_timeout', None)
    timeline_collect_tool_args = bool(ctx.task_config.get('timeline_collect_tool_args', False))
    timeline_tool_args_min_duration = float(ctx.task_config.get('timeline_tool_args_min_duration', 0))
    if per_observation_timeout is None:
        per_observation_timeout = observation_time_timeout

    actual_episodes, exit_status, total_observation_time = await _run_agent_loop(
        initial_prompt=initial_prompt,
        agent_input=agent_input,
        terminal_state_tracker=terminal_state_tracker,
        session_id=session_id,
        observation_time_timeout=observation_time_timeout,
        per_observation_timeout=per_observation_timeout,
        timeline_collect_tool_args=timeline_collect_tool_args,
        timeline_tool_args_min_duration=timeline_tool_args_min_duration,
    )

    logger.info(f'Terminus2 agent finished with status: {exit_status.value}, iterations: {actual_episodes}')

    termination_reason = EXIT_STATUS_TO_TERMINATION_REASON.get(exit_status, 'UNKNOWN')
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': termination_reason,
                'hytuner.observation_time': total_observation_time,
            },
        )

    return Terminus2Output(exit_status=exit_status, n_iterations=actual_episodes, main_tmux_session_id=session_id)
