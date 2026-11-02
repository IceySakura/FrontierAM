"""R2 STEM mini-SWE-agent style bash-only loop for Nexus."""

import asyncio
import json
import logging
import time
import traceback

from enum import Enum
from typing import Any

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolArgumentValidationError,
    ToolExecutor,
    ToolNameResolutionError,
    WithUserPrompt,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.trace import get_current_span_id, span_context, start_span, update_span
from nexus.trace.types import Status


logger = logging.getLogger(__name__)

Registry.import_module('nexus.extensions.projects.r2_stem.tools')

DEFAULT_SYSTEM_TEMPLATE = 'You are a helpful assistant that can interact with a computer to solve tasks.'
MAX_ERROR_TRACEBACK_CHARS = 20_000
# Floor for a command whose timeout was clipped to the wall-clock budget left. A command is
# only started when the budget is still open, but the model call before it can consume the
# rest of it, and a non-positive timeout would fail the call instead of running it briefly.
MIN_CLIPPED_COMMAND_TIMEOUT = 5.0

EXIT_STATUS_TO_TERMINATION_REASON = {
    'submitted': 'ENV_DONE',
    'max_iterations': 'MAX_STEPS',
    'format_error': 'FORMAT_ERROR',
    'tool_error': 'FORMAT_ERROR',
    'error': 'UNHANDLED_ERROR',
    'truncation': 'TRUNCATION',
    'output_length_exceeded': 'OUTPUT_LENGTH_EXCEEDED',
    'timeout': 'TASK_TIMEOUT',
}


class MiniSWEAgentStemExitStatus(str, Enum):
    SUBMITTED = 'submitted'
    MAX_ITERATIONS = 'max_iterations'
    FORMAT_ERROR = 'format_error'
    TOOL_ERROR = 'tool_error'
    ERROR = 'error'
    TRUNCATION = 'truncation'
    """The prompt outgrew the model window mid-run, so no sampling call was made."""

    OUTPUT_LENGTH_EXCEEDED = 'output_length_exceeded'
    """A single sampling call hit max_tokens. Kept separate from TRUNCATION because the
    two need different operator responses: this one says max_tokens is too small for the
    turn, TRUNCATION says the trajectory budget or context window is."""

    TIMEOUT = 'timeout'


class MiniSWEAgentStemInput(AgentRunInput, WithUserPrompt):
    """Configuration for the R2 STEM mini-SWE-agent style loop."""

    working_dir: str = Field(default='/workspace', description='Working directory inside the runtime container.')
    max_iterations: int = Field(default=50, description='Maximum model turns.')
    command_timeout: float = Field(default=300.0, description='Default timeout in seconds for each bash command.')
    wall_time_limit_seconds: float = Field(
        default=0.0,
        description='Stop the loop after this many wall-clock seconds (0 disables the limit).',
    )
    max_consecutive_format_errors: int = Field(
        default=3,
        description='Stop after this many consecutive missing/invalid bash tool calls. 0 disables the limit.',
    )
    system_template: str = Field(default=DEFAULT_SYSTEM_TEMPLATE, description='System prompt for the agent.')


class MiniSWEAgentStemOutput(AgentRunOutput):
    """Result from the R2 STEM mini-SWE-agent style loop."""

    exit_status: MiniSWEAgentStemExitStatus = Field(description='How the agent exited.')
    iterations: int = Field(default=0, description='Number of model turns completed.')
    raw_terminal_output: str | None = Field(default=None, description='Concatenated raw bash outputs.')
    submission: str = Field(default='', description='Final assistant content when the agent submitted, if any.')
    main_trajectory: list[dict[str, Any]] = Field(default_factory=list, description='Conversation trajectory.')
    error: str | None = Field(default=None, description='Human-readable agent failure summary, if any.')
    error_type: str | None = Field(default=None, description='Exception or failure category for non-success exits.')
    error_traceback: str | None = Field(default=None, description='Truncated traceback for unexpected agent failures.')


def _coerce_prompt(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False)


def _truncate_error_text(text: str) -> str:
    if len(text) <= MAX_ERROR_TRACEBACK_CHARS:
        return text
    omitted_chars = len(text) - MAX_ERROR_TRACEBACK_CHARS
    return f'{text[:MAX_ERROR_TRACEBACK_CHARS]}\n...<truncated {omitted_chars} chars>'


def _exception_summary(exc: BaseException) -> str:
    message = str(exc)
    return f'{type(exc).__name__}: {message}' if message else type(exc).__name__


def _exception_traceback(exc: BaseException) -> str:
    return _truncate_error_text(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _set_error(error: str, error_type: str, error_traceback: str | None = None) -> tuple[str, str, str | None]:
    return error, error_type, error_traceback


def _build_user_prompt(task: str, working_dir: str) -> str:
    return f"""Please solve the follow issue: {task}

    You can execute bash commands and inspect, create, or edit files when doing so helps solve the task. The task may require scientific reasoning, mathematical derivation, data analysis, simulation, code inspection, implementation, or a combination of these.

    Your primary objective is to produce a scientifically defensible answer. Code and file changes are supporting artifacts, not necessarily the final objective.

    ## Recommended Workflow

    Adapt this workflow to the task rather than following every step mechanically.

    1. Parse the task carefully.
    2. Inspect the available resources.
    3. Choose an appropriate scientific approach.
    4. Execute the analysis or implementation.
    5. Validate the result scientifically.
    6. Produce your final answer with normal content, do not use the tool command.

    ## Command Execution Rules

    You are working at `{working_dir}` inside the runtime container. You are operating in an environment where

    1. You issue a bash tool call when you need command output.
    2. The system executes the command(s) in a subshell.
    3. You see the result(s).
    4. You decide the next action.

    While working, each response should include:

    1. **Reasoning text** where you explain your analysis and plan
    2. At least one useful tool call with your command

    When you have the final answer, respond with the final answer in normal assistant content and do not call any tool.

    **CRITICAL REQUIREMENTS:**

    - Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
    - However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files
    - Do not submit your final answer by using the tool command."""


def _format_error_message(error: str, tool_call_id: str) -> dict[str, str]:
    content = (
        'Tool call error:\n\n'
        f'<error>\n{error}\n</error>\n\n'
        'Here is general guidance on how to submit correct toolcalls:\n\n'
        "Use the 'bash' tool when you need to execute commands.\n\n"
        'Call the bash tool with your command as the argument:\n'
        '- Tool: bash\n'
        '- Arguments: {"command": "your_command_here"}\n\n'
        'If you want to end the task, provide the final answer in assistant content without calling any tool.'
    )
    return {'role': 'tool', 'tool_call_id': tool_call_id, 'content': content}


def _parse_tool_calls(
    message: Any,
    default_timeout: float,
    executor: ToolExecutor,
) -> tuple[list[tuple[Any, dict[str, Any]]], list[tuple[Any, str]]]:
    tool_calls = getattr(message, 'tool_calls', None) or []
    if not tool_calls:
        return [], []

    parsed: list[tuple[Any, dict[str, Any]]] = []
    failed: list[tuple[Any, str]] = []
    for tool_call in tool_calls:
        function = getattr(tool_call, 'function', None)
        name = getattr(function, 'name', '')
        try:
            args = json.loads(getattr(function, 'arguments', '') or '{}')
        except json.JSONDecodeError as exc:
            failed.append((tool_call, f'Error parsing bash tool arguments: {exc}.'))
            continue
        if not isinstance(args, dict):
            failed.append((tool_call, 'Bash tool arguments must be a JSON object.'))
            continue
        command = args.get('command')
        if not isinstance(command, str) or not command.strip():
            failed.append((tool_call, "Missing non-empty 'command' argument in bash tool call."))
            continue
        args.setdefault('timeout', default_timeout)
        try:
            executor.validate_tool_call(name, args)
        except (ToolNameResolutionError, ToolArgumentValidationError) as exc:
            failed.append((tool_call, exc.message_for_model))
            continue
        parsed.append((tool_call, args))
    return parsed, failed


def _tool_response_text(response: Any) -> str:
    pieces = []
    for item in response.output or []:
        if isinstance(item, dict) and item.get('type') == 'text':
            pieces.append(str(item.get('text', '')))
        else:
            pieces.append(json.dumps(item, ensure_ascii=False))
    return '\n'.join(piece for piece in pieces if piece)


def _raw_output_from_response(response: Any) -> str:
    meta = getattr(response, 'meta', {}) or {}
    if isinstance(meta.get('raw_output'), str):
        return meta['raw_output']
    parts = []
    if isinstance(meta.get('stdout'), str):
        parts.append(meta['stdout'])
    if isinstance(meta.get('stderr'), str) and meta['stderr']:
        parts.append('[stderr]\n' + meta['stderr'])
    return '\n'.join(parts) or _tool_response_text(response)


@agent_registry.register('mini_swe_agent_stem')
async def run_mini_swe_agent_stem(agent_input: MiniSWEAgentStemInput) -> MiniSWEAgentStemOutput:
    """Run an R2 STEM mini-SWE-agent style bash-only loop inside a Nexus runtime."""
    ctx = require_context()
    client = ctx.openai_client
    ctx.tool_state.update({'working_dir': agent_input.working_dir, 'command_timeout': agent_input.command_timeout})

    executor = ToolExecutor([tool_registry.get('mini_swe_agent_stem/bash')])
    tools = executor.get_tool_schemas()
    history: list[dict[str, Any]] = [
        {'role': 'system', 'content': agent_input.system_template},
        {
            'role': 'user',
            'content': _build_user_prompt(_coerce_prompt(agent_input.user_prompt), agent_input.working_dir),
        },
    ]
    raw_outputs: list[str] = []
    exit_status: MiniSWEAgentStemExitStatus | None = None
    iterations = 0
    consecutive_format_errors = 0
    total_observation_time = 0.0
    error: str | None = None
    error_type: str | None = None
    error_traceback: str | None = None
    submission = ''

    # Deadline this loop enforces on itself. Callers bound the agent with an outer
    # ``asyncio.wait_for``, which cancels this coroutine and takes the trajectory, the
    # submission and the termination reason published below down with it, leaving the
    # rollout unscorable. Stopping here instead returns all three and exits as ``timeout``.
    deadline: float | None = None
    if agent_input.wall_time_limit_seconds > 0:
        deadline = time.monotonic() + agent_input.wall_time_limit_seconds

    try:
        while iterations < agent_input.max_iterations:
            if deadline is not None and time.monotonic() >= deadline:
                exit_status = MiniSWEAgentStemExitStatus.TIMEOUT
                error, error_type, error_traceback = _set_error(
                    f'Reached wall_time_limit_seconds={agent_input.wall_time_limit_seconds:g} without submission.',
                    'WallTimeLimit',
                )
                break

            iter_span = start_span(
                'agent.iteration',
                attributes={'type': 'agent_iteration', 'iteration': iterations + 1},
            )
            try:
                with span_context(
                    'agent.inference',
                    attributes={'type': 'agent_inference', 'iteration': iterations + 1},
                ):
                    completion = client.create_completion(messages=history, tools=tools or None)
                    if deadline is None:
                        response = await completion
                    else:
                        # A single sampling call has been observed to run for hours, so the
                        # between-turn check above cannot on its own keep the loop inside its
                        # budget. The check just passed, so what is left of it is positive.
                        try:
                            response = await asyncio.wait_for(completion, timeout=deadline - time.monotonic())
                        except asyncio.TimeoutError:
                            exit_status = MiniSWEAgentStemExitStatus.TIMEOUT
                            error, error_type, error_traceback = _set_error(
                                'Sampling call exceeded the remaining '
                                f'wall_time_limit_seconds={agent_input.wall_time_limit_seconds:g} budget.',
                                'WallTimeLimit',
                            )
                            iter_span.update(status=Status(code='ERROR', message=error))
                            break
                message = response.choices[0].message
                finish_reason = getattr(response.choices[0], 'finish_reason', None)
                message_dict = message.model_dump(exclude_none=True)

                if finish_reason == 'length':
                    history.append(message_dict)
                    iterations += 1
                    exit_status = MiniSWEAgentStemExitStatus.OUTPUT_LENGTH_EXCEEDED
                    error, error_type, error_traceback = _set_error(
                        'Model response was truncated before a complete final answer was produced.',
                        'OutputLengthExceeded',
                    )
                    iter_span.update(status=Status(code='ERROR', message=error))
                    break

                parsed_tool_calls, failed_tool_calls = _parse_tool_calls(message, agent_input.command_timeout, executor)
                parse_error = '; '.join(
                    f'{tool_call.id}: {failure_reason}' for tool_call, failure_reason in failed_tool_calls
                )

                if failed_tool_calls:
                    history.append(message_dict)
                    for tool_call, failure_reason in failed_tool_calls:
                        history.append(_format_error_message(failure_reason, tool_call.id))
                    consecutive_format_errors += 1
                    if not parsed_tool_calls:
                        iterations += 1
                        iter_span.update(status=Status(code='ERROR', message=parse_error))
                        if (
                            agent_input.max_consecutive_format_errors > 0
                            and consecutive_format_errors >= agent_input.max_consecutive_format_errors
                        ):
                            exit_status = MiniSWEAgentStemExitStatus.FORMAT_ERROR
                            error, error_type, error_traceback = _set_error(parse_error, 'FormatError')
                            break
                        continue
                else:
                    consecutive_format_errors = 0
                    history.append(message_dict)

                if not parsed_tool_calls:
                    iterations += 1
                    if finish_reason != 'stop':
                        exit_status = MiniSWEAgentStemExitStatus.ERROR
                        error, error_type, error_traceback = _set_error(
                            f'Model stopped without a valid final answer (finish_reason={finish_reason!r}).',
                            'FinishReasonError',
                        )
                        iter_span.update(status=Status(code='ERROR', message=error))
                        break
                    if not isinstance(message.content, str) or not message.content.strip():
                        exit_status = MiniSWEAgentStemExitStatus.FORMAT_ERROR
                        error, error_type, error_traceback = _set_error(
                            'Model stopped without a non-empty final assistant response.',
                            'EmptySubmissionError',
                        )
                        iter_span.update(status=Status(code='ERROR', message=error))
                        break

                    submission = message.content.strip()
                    exit_status = MiniSWEAgentStemExitStatus.SUBMITTED
                    iter_span.update(status=Status(code='OK'))
                    break

                for tool_call, tool_args in parsed_tool_calls:
                    start_time = time.time()
                    command_timeout = float(tool_args.get('timeout') or agent_input.command_timeout)
                    if deadline is not None:
                        # The deadline is only read between turns, so a command left free to run
                        # past it would still expire the caller's outer timeout. Clip it to the
                        # budget left and let the check above end the loop on the next pass.
                        command_timeout = max(
                            min(command_timeout, deadline - time.monotonic()),
                            MIN_CLIPPED_COMMAND_TIMEOUT,
                        )
                        tool_args['timeout'] = command_timeout
                    tool_span = start_span(
                        'agent.observation',
                        attributes={
                            'type': 'agent_observation',
                            'iteration': iterations + 1,
                            'tool_name': 'bash',
                        },
                    )
                    try:
                        tool_response = await executor.call(
                            tool_call.function.name,
                            tool_args,
                            timeout=command_timeout + 5,
                        )
                    except TimeoutError as exc:
                        timeout_message = (
                            f'Tool execution timed out after {command_timeout + 5}s '
                            f'(command timeout was {command_timeout}s): {_exception_summary(exc)}'
                        )
                        timeout_observation = json.dumps(
                            {'returncode': -1, 'output': '', 'exception_info': timeout_message},
                            ensure_ascii=False,
                        )
                        raw_outputs.append(timeout_message)
                        history.append({'role': 'tool', 'tool_call_id': tool_call.id, 'content': timeout_observation})
                        tool_span.update(status=Status(code='ERROR', message=str(exc)))
                        continue
                    except Exception as exc:
                        error_message = f'Error executing tool: {_exception_summary(exc)}'
                        history.append({'role': 'tool', 'tool_call_id': tool_call.id, 'content': error_message})
                        exit_status = MiniSWEAgentStemExitStatus.TOOL_ERROR
                        error, error_type, error_traceback = _set_error(
                            error_message,
                            type(exc).__name__,
                            _exception_traceback(exc),
                        )
                        tool_span.update(status=Status(code='ERROR', message=str(exc)))
                        break
                    finally:
                        total_observation_time += time.time() - start_time
                        tool_span.finish()

                    raw_outputs.append(_raw_output_from_response(tool_response))
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': _tool_response_text(tool_response),
                    })

                iterations += 1
                if exit_status is not None:
                    break
                if failed_tool_calls:
                    iter_span.update(status=Status(code='ERROR', message=parse_error))
                    if (
                        agent_input.max_consecutive_format_errors > 0
                        and consecutive_format_errors >= agent_input.max_consecutive_format_errors
                    ):
                        exit_status = MiniSWEAgentStemExitStatus.FORMAT_ERROR
                        error, error_type, error_traceback = _set_error(parse_error, 'FormatError')
                        break
                else:
                    iter_span.update(status=Status(code='OK'))
            finally:
                iter_span.finish()
    except ContextLengthExceededError as exc:
        logger.error('model context length exceeded: %s', exc)
        exit_status = MiniSWEAgentStemExitStatus.TRUNCATION
        error, error_type, error_traceback = _set_error(
            _exception_summary(exc),
            type(exc).__name__,
            _exception_traceback(exc),
        )
    except Exception as exc:
        logger.exception('mini_swe_agent_stem failed: %s', exc)
        exit_status = MiniSWEAgentStemExitStatus.ERROR
        error, error_type, error_traceback = _set_error(
            _exception_summary(exc),
            type(exc).__name__,
            _exception_traceback(exc),
        )

    if exit_status is None:
        exit_status = MiniSWEAgentStemExitStatus.MAX_ITERATIONS
        error, error_type, error_traceback = _set_error(
            f'Reached max_iterations={agent_input.max_iterations} without submission.',
            'MaxIterations',
        )

    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': EXIT_STATUS_TO_TERMINATION_REASON.get(exit_status.value, 'UNKNOWN'),
                'hytuner.observation_time': total_observation_time,
            },
        )

    return MiniSWEAgentStemOutput(
        exit_status=exit_status,
        iterations=iterations,
        raw_terminal_output='\n'.join(raw_outputs) if raw_outputs else None,
        submission=submission,
        main_trajectory=history,
        error=error,
        error_type=error_type,
        error_traceback=error_traceback,
    )


__all__ = [
    'MiniSWEAgentStemExitStatus',
    'MiniSWEAgentStemInput',
    'MiniSWEAgentStemOutput',
    'run_mini_swe_agent_stem',
]
