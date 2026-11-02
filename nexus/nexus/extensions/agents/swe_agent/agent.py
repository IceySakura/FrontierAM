"""SWE-agent implementation for nexus framework.

This agent implements the SWE-agent loop for solving software engineering tasks.
It registers two agents over one loop: `swe_agent` and `ch_swe_agent`, which
differ only in the language of their prompts and tool schema.
"""

import asyncio
import json
import logging
import sys
import time

from enum import Enum
from typing import Literal

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    AgentRunOutput,
    ContextLengthExceededError,
    Registry,
    ToolExecutor,
    WithUserPrompt,
    WithWorkingDir,
    agent_registry,
    require_context,
    tool_registry,
)
from nexus.extensions.agents.swe_agent.platform_status import (
    EXIT_CODES,
    PlatformExitStatus,
    exception_details,
    runtime_failure_status,
    sandbox_failure_status,
)
from nexus.extensions.agents.swe_agent.prompts import CHINESE, ENGLISH, PromptSet
from nexus.extensions.agents.swe_agent.tools import ToolInfrastructureError, resolve_bash_backend
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.runtime.providers.gongfeng import SandboxApiError
from nexus.trace import get_current_span_id, span_context, start_span, update_span
from nexus.trace.types import Status
from nexus.utils.timeout import InferenceTimeoutError, TimeBudget, async_timeout, await_completion_with_time_budget


logger = logging.getLogger(__name__)


def _is_loaded_hytuner_client(client: object) -> bool:
    """Check the client type without importing the optional HyTuner module."""
    module = sys.modules.get('nexus.extensions.openai_clients.hytuner')
    if module is None:
        return False
    client_type = getattr(module, 'HyTunerClient', None)
    return isinstance(client_type, type) and isinstance(client, client_type)


# Import tools to register them
Registry.import_module('nexus.extensions.agents.swe_agent.tools')


# ============== Observation Truncation ==============
# Copied from openhands-sdk/openhands/sdk/utils/truncate.py

# Default truncation limits
DEFAULT_TEXT_CONTENT_LIMIT = 50_000

# Default truncation notice
DEFAULT_TRUNCATE_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full '
    'response has been shown to you.</NOTE>'
)  # 113 chars


def maybe_truncate(
    content: str,
    truncate_after: int | None = None,
    truncate_notice: str = DEFAULT_TRUNCATE_NOTICE,
) -> str:
    """Truncate the middle of content if it exceeds the specified length.

    Keeps the head and tail of the content to preserve context at both ends.

    Args:
        content: The text content to potentially truncate
        truncate_after: Maximum length before truncation. If None, no truncation occurs
        truncate_notice: Notice to insert in the middle when content is truncated

    Returns:
        Original content if under limit, or truncated content with head and tail preserved
    """
    # 1) Early exits: no truncation requested, or content already within limit
    if not truncate_after or len(content) <= truncate_after or truncate_after < 0:
        return content

    # 2) If even the base notice doesn't fit, return a slice of it
    if len(truncate_notice) >= truncate_after:
        return truncate_notice[:truncate_after]

    # 3) Calculate head and tail sizes
    available_chars = truncate_after - len(truncate_notice)
    # Prefer giving the "extra" char to head (ceil split)
    head_chars = available_chars // 2 + (available_chars % 2)
    tail_chars = available_chars - head_chars

    return content[:head_chars] + truncate_notice + (content[-tail_chars:] if tail_chars > 0 else '')


# ============== Exit Status ==============


class ExitStatus(str, Enum):
    """Exit status for SWE-agent."""

    SUBMITTED = 'submitted'
    """Normal exit - agent submitted a solution"""

    FINISHED = 'finished'
    """Normal exit - agent finished without explicit submit"""

    MAX_ITERATIONS = 'max_iterations'
    """Agent reached maximum iterations limit"""

    TOOL_ERROR = 'tool_error'
    """Abnormal exit - tool call failed (e.g., invalid JSON arguments)"""

    ERROR = 'error'
    """Abnormal exit - agent encountered an error"""

    TRUNCATION = 'truncation'
    """Abnormal exit - agent encountered a truncation error"""

    TIMEOUT = 'timeout'
    """Abnormal exit - agent encountered a timeout error"""


EXIT_STATUS_TO_TERMINATION_REASON = {
    ExitStatus.SUBMITTED: 'ENV_DONE',
    ExitStatus.FINISHED: 'ENV_DONE',
    ExitStatus.MAX_ITERATIONS: 'MAX_STEPS',
    ExitStatus.TOOL_ERROR: 'FORMAT_ERROR',
    ExitStatus.ERROR: 'UNHANDLED_ERROR',
    ExitStatus.TRUNCATION: 'TRUNCATION',
    ExitStatus.TIMEOUT: 'TASK_TIMEOUT',
}


# ============== Input/Output Models ==============


class SWEAgentInput(AgentRunInput, WithUserPrompt, WithWorkingDir):
    """Input configuration for SWE-agent."""

    platform_exit_status: bool = Field(default=False, description='Use the AgentBench status contract')
    bash_backend: Literal['terminal', 'stateless'] | None = Field(
        default=None, description='Shell backend; when omitted, use SWE_AGENT_BASH_BACKEND (default: terminal)'
    )
    continue_on_tool_timeout: bool = Field(
        default=False, description='Return an owned per-tool timeout to the agent instead of ending the run'
    )
    timeout: float | None = Field(default=None, gt=0, description='Overall agent deadline in seconds')
    max_iterations: int = Field(default=50, description='Maximum number of agent steps')
    completion_retry_on_tool_error: int = Field(
        default=0,
        description='Number of retries when model output is invalid (no tool calls or invalid JSON). '
        '0 means no retry, fail immediately on first error.',
    )
    early_stop: bool = Field(
        default=False,
        description='If True, stop the agent immediately when tool execution error occurs. '
        'Environment will return empty string and the interaction ends.',
    )


class SWEAgentOutput(AgentRunOutput):
    """Output from SWE-agent run."""

    exit_status: ExitStatus | PlatformExitStatus = Field(description='How the agent exited')
    exit_code: int | None = None
    error: str | None = None
    native_exit_status: ExitStatus | None = None
    failure_stage: str | None = None
    failure_kind: Literal['agent', 'infrastructure'] | None = None
    sandbox: dict | None = None
    iterations: int = Field(default=0, description='Number of iterations completed')


# ============== Agent Implementation ==============


@agent_registry.register('swe_agent')
async def run_swe_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
    """Run SWE-agent on a problem statement."""
    return await _run_with_platform_status(agent_input, agent_name='swe_agent', prompts=ENGLISH)


@agent_registry.register('ch_swe_agent')
async def run_ch_swe_agent(agent_input: SWEAgentInput) -> SWEAgentOutput:
    """Run SWE-agent with Chinese prompts and a Chinese tool schema.

    Same loop and same tool implementations as `swe_agent`; only the language the
    model is addressed in differs.
    """
    return await _run_with_platform_status(agent_input, agent_name='ch_swe_agent', prompts=CHINESE)


def _agent_output(
    agent_input: SWEAgentInput,
    native_status: ExitStatus,
    iterations: int,
    platform_status: PlatformExitStatus | None = None,
    error: str | None = None,
    stage: str | None = None,
    sandbox: dict | None = None,
) -> SWEAgentOutput:
    if not agent_input.platform_exit_status:
        return SWEAgentOutput(exit_status=native_status, iterations=iterations)
    if platform_status is None:
        platform_status = {
            ExitStatus.SUBMITTED: PlatformExitStatus.FINISHED,
            ExitStatus.FINISHED: PlatformExitStatus.FINISHED,
            ExitStatus.MAX_ITERATIONS: PlatformExitStatus.MAX_ITERATIONS,
        }.get(native_status, PlatformExitStatus.UNKNOWN_ERROR)
    return SWEAgentOutput(
        exit_status=platform_status,
        exit_code=EXIT_CODES[platform_status],
        error=None if platform_status == PlatformExitStatus.FINISHED else error,
        native_exit_status=native_status,
        failure_stage=stage,
        failure_kind=(
            'agent'
            if platform_status == PlatformExitStatus.RUNTIME_ERROR_BY_AGENT
            else 'infrastructure'
            if platform_status in (PlatformExitStatus.RUNTIME_ERROR_BY_ENV, PlatformExitStatus.ENVIRONMENT_ERROR)
            else None
        ),
        sandbox=sandbox,
        iterations=iterations,
    )


async def _run_with_platform_status(
    agent_input: SWEAgentInput, *, agent_name: str, prompts: PromptSet
) -> SWEAgentOutput:
    if not agent_input.platform_exit_status:
        return await _run_agent_loop(agent_input, agent_name=agent_name, prompts=prompts)
    deadline = None
    progress = {'iterations': 0}
    try:
        async with async_timeout(agent_input.timeout) as deadline:
            return await _run_agent_loop(agent_input, agent_name=agent_name, prompts=prompts, progress=progress)
    except TimeoutError as exc:
        expired = deadline is not None and deadline.expired()
        sandbox = exception_details(exc).get('sandbox')
        status = (
            PlatformExitStatus.AGENT_TIMEOUT
            if expired
            else sandbox_failure_status(sandbox)
            if sandbox is not None
            else PlatformExitStatus.ENVIRONMENT_ERROR
        )
        message = f'Agent exceeded timeout={agent_input.timeout}s' if expired else f'{type(exc).__name__}: {exc}'
        return _agent_output(
            agent_input,
            ExitStatus.TIMEOUT,
            progress['iterations'],
            status,
            message,
            'agent' if expired else 'setup',
            sandbox,
        )
    except Exception as exc:
        sandbox = exception_details(exc).get('sandbox')
        return _agent_output(
            agent_input,
            ExitStatus.ERROR,
            0,
            sandbox_failure_status(sandbox) if sandbox is not None else PlatformExitStatus.ENVIRONMENT_ERROR,
            f'{type(exc).__name__}: {exc}',
            'setup',
            sandbox,
        )


async def _run_agent_loop(
    agent_input: SWEAgentInput,
    *,
    agent_name: str,
    prompts: PromptSet,
    progress: dict[str, int] | None = None,
) -> SWEAgentOutput:
    """Run the SWE-agent loop for one registered agent.

    Simple agent loop:
    1. Initialize with system prompt and problem statement
    2. Query model for next action
    3. Execute tool and get observation
    4. Check for exit conditions (submit, max iterations)
    5. Add observation to history and repeat

    Args:
        agent_input: Agent configuration.
        agent_name: The registered agent name, which is also the namespace its tools
            are registered under.
        prompts: The language-specific prompts this agent speaks.
    """
    ctx = require_context()
    client = ctx.openai_client
    is_hytuner_client = _is_loaded_hytuner_client(client)

    # Override from task_config if running as swe_task
    if ctx.task_name in {'swe_task', 'superswe_task'}:
        working_dir = agent_input.working_dir or ctx.task_config.get('working_dir')
        if not working_dir:
            raise ValueError('working_dir is required for swe_task')
        agent_input.working_dir = working_dir
        problem_statement = ctx.task_config.get('problem_statement')
        if not problem_statement:
            raise ValueError('problem_statement is required in task_config for swe_task')

        language = (ctx.task_config.get('language') or 'python').lower()
        template = prompts.instance_template(language)

        agent_input.user_prompt = template.format(
            working_dir=working_dir,
            problem_statement=problem_statement,
        )

    ctx.tool_state['bash_backend'] = resolve_bash_backend(agent_input.bash_backend)

    # Create tool executor with this agent's tools
    executor = ToolExecutor([
        tool_registry.get(f'{agent_name}/bash'),
        tool_registry.get(f'{agent_name}/str_replace_editor'),
        tool_registry.get(f'{agent_name}/review_submit'),
    ])

    # Use the task-provided problem statement, including any configured anti-hacking notice.

    # Build prompts
    system_text = prompts.system
    if ctx.tool_state['bash_backend'] == 'stateless':
        system_text += (
            '\nEach bash call starts a fresh shell in the task working directory. '
            'Shell variables and directory changes do not persist between calls; '
            'include any required setup in each command.'
        )
    system_prompt = {'role': 'system', 'content': system_text}
    history = [system_prompt, {'role': 'user', 'content': agent_input.user_prompt}]

    # Get tool schemas for LLM
    tools = executor.get_tool_schemas()

    # Helper function to parse tool calls
    def parse_tool_calls(message) -> tuple[list, Exception | None]:
        """Parse tool call arguments, return (parsed_args, error)."""
        parsed_args = []
        for tool_call in message.tool_calls:
            try:
                parsed_args.append(json.loads(tool_call.function.arguments))
            except json.JSONDecodeError as e:
                return [], e
        return parsed_args, None

    # None means "still running", set to a value when we want to exit
    exit_status: ExitStatus | None = None
    platform_status: PlatformExitStatus | None = None
    exit_error: str | None = None
    failure_stage: str | None = None
    sandbox: dict | None = None
    current_stage = 'setup'
    iteration = 0
    total_observation_time = 0.0
    observation_time_timeout = ctx.task_config.get('observation_time_timeout', None)
    if agent_input.platform_exit_status and observation_time_timeout is not None:
        raise ValueError(
            'AgentBench status mode does not support observation_time_timeout; '
            'use agent_config.timeout for the overall deadline and '
            'task_config.per_observation_timeout for individual tool calls'
        )
    per_observation_timeout = ctx.task_config.get('per_observation_timeout', None)
    inference_time_timeout = ctx.task_config.get('inference_time_timeout', None)
    inference_budget = TimeBudget(inference_time_timeout)
    timeline_collect_tool_args = bool(ctx.task_config.get('timeline_collect_tool_args', False))
    timeline_tool_args_min_duration = float(ctx.task_config.get('timeline_tool_args_min_duration', 0))
    if per_observation_timeout is None:
        per_observation_timeout = observation_time_timeout

    iter_span = None
    try:
        while iteration < agent_input.max_iterations:
            if progress is not None:
                progress['iterations'] = iteration + 1
            iter_span = start_span(
                'agent.iteration',
                attributes={
                    'type': 'agent_iteration',
                    'iteration': iteration + 1,
                },
            )
            logger.debug(f'=== {agent_name} Step {iteration + 1} ===')

            # Retry loop for handling model output errors (no tool calls or invalid JSON)
            max_retries = agent_input.completion_retry_on_tool_error
            for retry_attempt in range(max_retries + 1):
                completion_kwargs = {}
                if is_hytuner_client and max_retries > 0 and retry_attempt < max_retries:
                    completion_kwargs['allow_retry_for_tool_call'] = True
                try:
                    current_stage = 'inference'
                    with span_context(
                        'agent.inference',
                        attributes={
                            'type': 'agent_inference',
                            'iteration': iteration + 1,
                            'retry': retry_attempt,
                        },
                    ):
                        response = await await_completion_with_time_budget(
                            inference_budget,
                            lambda completion_kwargs=completion_kwargs: client.create_completion(
                                messages=history,
                                tools=tools or None,
                                **completion_kwargs,
                            ),
                        )
                except InferenceTimeoutError as exc:
                    platform_status = PlatformExitStatus.INFERENCE_ERROR
                    exit_error = f'{type(exc).__name__}: {exc}'
                    failure_stage = 'inference_budget'
                    logger.info(
                        'Inference timeout reached: %s seconds, used %.1f seconds',
                        inference_time_timeout,
                        inference_budget.elapsed,
                    )
                    exit_status = ExitStatus.TIMEOUT
                    break

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # The API's finish reason is authoritative even when a truncated
                # response happens to contain parseable tool arguments.
                if agent_input.platform_exit_status and finish_reason in ('length', 'max_tokens'):
                    exit_status = ExitStatus.TRUNCATION
                    platform_status = PlatformExitStatus.OUTPUT_LENGTH_EXCEEDED
                    exit_error = f'Model generation ended with finish_reason={finish_reason}'
                    failure_stage = 'inference'
                    break

                # Check for tool error: no tool calls
                if not message.tool_calls:
                    if agent_input.platform_exit_status and message.content and finish_reason == 'stop':
                        exit_status = ExitStatus.FINISHED
                        break
                    if finish_reason == 'length':
                        # The turn is malformed because generation was cut at the response budget,
                        # not because the model cannot call a tool. Retrying is pointless (the
                        # budget is spent) and reporting tool_error would bury a length problem
                        # under a format one.
                        logger.warning(f'{agent_name}: generation cut at the response budget -> truncation')
                        exit_status = ExitStatus.TRUNCATION
                        platform_status = PlatformExitStatus.OUTPUT_LENGTH_EXCEEDED
                        exit_error = 'Model generation ended with finish_reason=length'
                        failure_stage = 'inference'
                        break
                    if retry_attempt < max_retries:
                        logger.warning(f'No tool calls - model output error, retry {retry_attempt + 1}/{max_retries}')
                        continue  # Retry without adding to history
                    logger.error('No tool calls - model did not call any tool, all retries exhausted')
                    exit_status = ExitStatus.TOOL_ERROR
                    if message.content and finish_reason == 'stop':
                        platform_status = PlatformExitStatus.FINISHED
                    else:
                        platform_status = PlatformExitStatus.INFERENCE_ERROR
                        exit_error = 'Model response contained neither a usable tool call nor a final answer'
                        failure_stage = 'inference'
                    break

                # Parse tool call arguments
                parsed_tool_args, json_parse_error = parse_tool_calls(message)
                if json_parse_error is not None:
                    if finish_reason == 'length':
                        # Arguments are unparseable because the generation was cut mid-JSON;
                        # see the no-tool-calls branch above.
                        logger.warning(f'{agent_name}: tool arguments cut at the response budget -> truncation')
                        exit_status = ExitStatus.TRUNCATION
                        platform_status = PlatformExitStatus.OUTPUT_LENGTH_EXCEEDED
                        exit_error = 'Model generation ended with finish_reason=length'
                        failure_stage = 'inference'
                        break
                    if retry_attempt < max_retries:
                        logger.warning(
                            f'Failed to parse tool arguments: {json_parse_error}, '
                            f'retry {retry_attempt + 1}/{max_retries}'
                        )
                        continue  # Retry without adding to history
                    logger.error(f'Failed to parse tool arguments: {json_parse_error}, all retries exhausted')
                    exit_status = ExitStatus.TOOL_ERROR
                    platform_status = PlatformExitStatus.UNKNOWN_ERROR
                    exit_error = f'Invalid tool arguments: {json_parse_error}'
                    failure_stage = 'tool_arguments'
                    break

                # Success: valid response with parseable tool calls
                break

            # If exit_status is set from retry loop, break main loop
            if exit_status is not None:
                break

            # Now we have a valid response, add to history
            history.append(message.model_dump(exclude_none=True))

            # Execute all tool calls
            for idx, tool_call in enumerate(message.tool_calls):
                tool_name = tool_call.function.name
                tool_args = parsed_tool_args[idx]

                logger.debug(f'Tool call: {tool_name}')
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'Tool args: {json.dumps(tool_args, ensure_ascii=False)[:500]}')

                # Execute tool with error handling (start_span + finally: tool errors are handled without re-raise)
                start_time = time.time()
                tool_span_attrs: dict = {
                    'type': 'agent_observation',
                    'iteration': iteration + 1,
                    'tool_name': tool_name,
                }
                if timeline_collect_tool_args and timeline_tool_args_min_duration <= 0:
                    tool_span_attrs['hytuner.tool_args'] = json.dumps(tool_args, ensure_ascii=False)
                tool_span = start_span(
                    'agent.observation',
                    tool_span_attrs,
                )
                observation_deadline = None
                current_stage = 'tool'
                try:
                    if agent_input.platform_exit_status or agent_input.continue_on_tool_timeout:
                        async with async_timeout(per_observation_timeout) as observation_deadline:
                            tool_response = await executor.call(tool_name, tool_args)
                    else:
                        tool_response = await executor.call(tool_name, tool_args, timeout=per_observation_timeout)
                except (RuntimeUnhealthyError, SandboxApiError, ToolInfrastructureError) as e:
                    exit_status = ExitStatus.ERROR
                    sandbox = exception_details(e).get('sandbox')
                    platform_status = (
                        sandbox_failure_status(sandbox, during_tool=True)
                        if sandbox is not None
                        else PlatformExitStatus.RUNTIME_ERROR_BY_ENV
                    )
                    exit_error = f'{type(e).__name__}: {e}'
                    failure_stage = 'tool'
                    logger.exception(f'Runtime unhealthy error occurred, stopping agent, error: {e}')
                    total_observation_time += time.time() - start_time
                    tool_span.update(status=Status(code='ERROR', message=str(e)))
                    # Append error observation so history keeps growing (required by HyTuner)
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error executing tool: {e}',
                    })
                    break  # Break inner tool-call loop; outer loop will exit via exit_status check
                except (TimeoutError, asyncio.TimeoutError) as e:
                    sandbox = exception_details(e).get('sandbox')
                    if (
                        agent_input.continue_on_tool_timeout
                        and sandbox is None
                        and observation_deadline is not None
                        and observation_deadline.expired()
                    ):
                        # An owned tool deadline is an observation, not the overall agent deadline.
                        elapsed = time.time() - start_time
                        total_observation_time += elapsed
                        diagnostic = f'Tool {tool_name} exceeded per_observation_timeout={per_observation_timeout}s'
                        logger.warning('[SINGLE-OBSERVATION TIMEOUT] %s; continuing agent', diagnostic)
                        tool_span.update(status=Status(code='ERROR', message=diagnostic))
                        history.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': diagnostic,
                        })
                        if observation_time_timeout and total_observation_time >= observation_time_timeout:
                            exit_status = ExitStatus.TIMEOUT
                            exit_error = f'Observation budget exhausted: {observation_time_timeout}s'
                            failure_stage = 'observation_budget'
                            logger.info('[CUMULATIVE OBSERVATION TIMEOUT] %s', exit_error)
                            break
                        continue
                    exit_status = ExitStatus.TIMEOUT
                    platform_status = (
                        sandbox_failure_status(sandbox, during_tool=True)
                        if sandbox is not None
                        else PlatformExitStatus.RUNTIME_ERROR_BY_ENV
                    )
                    exit_error = f'{type(e).__name__}: {e}'
                    failure_stage = 'tool'
                    total_observation_time += time.time() - start_time
                    # Distinguish a single tool-call timeout from the cumulative tool-time budget.
                    logger.warning(
                        f'[SINGLE-OBSERVATION TIMEOUT] one tool call exceeded its deadline '
                        f'(limit={per_observation_timeout}s, accumulated={total_observation_time:.1f}s), '
                        f'stopping agent: {type(e).__name__}: {e}'
                    )
                    tool_span.update(status=Status(code='ERROR', message=str(e)))
                    # Append error observation so history keeps growing (required by HyTuner)
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error executing tool: {e}',
                    })
                    break  # Break inner tool-call loop; outer loop will exit via exit_status check
                except Exception as e:
                    end_time = time.time()
                    tool_span.update(status=Status(code='ERROR', message=str(e)))
                    sandbox = exception_details(e).get('sandbox')

                    # Stop on sandbox or fatal terminal failures.
                    # Recognize terminal errors by name or the fatal flag for compatibility with runtime-local classes.
                    if (
                        sandbox is not None
                        or type(e).__name__ == 'TerminalInfraError'
                        or getattr(e, 'fatal', False) is True
                    ):
                        exit_status = ExitStatus.ERROR
                        platform_status = (
                            sandbox_failure_status(sandbox, during_tool=True)
                            if sandbox is not None
                            else PlatformExitStatus.RUNTIME_ERROR_BY_ENV
                        )
                        exit_error = f'{type(e).__name__}: {e}'
                        failure_stage = 'tool'
                        logger.exception(
                            f'Runtime failure in tool {tool_name}, stopping agent '
                            f'({platform_status.value}): {type(e).__name__}: {e}'
                        )
                        total_observation_time += end_time - start_time
                        # Append observation so history keeps growing (required by HyTuner),
                        # matching the RuntimeUnhealthyError branch above.
                        history.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': f'Error executing tool: {e}',
                        })
                        break  # Break inner tool-call loop; outer loop exits via exit_status check

                    if agent_input.platform_exit_status and isinstance(e, ImportError):
                        exit_status = ExitStatus.ERROR
                        platform_status = PlatformExitStatus.ENVIRONMENT_ERROR
                        exit_error = f'{type(e).__name__}: {e}'
                        failure_stage = 'tool'
                        total_observation_time += end_time - start_time
                        history.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': f'Error executing tool: {e}',
                        })
                        logger.exception('Required tool dependency is unavailable; stopping the agent')
                        break
                    # Get raw model output if available
                    raw_output = getattr(message, 'raw', None) or 'N/A'

                    # Enhanced error logging with detailed tool call information
                    logger.error(
                        f'\n{"=" * 80}\n'
                        f'TOOL EXECUTION ERROR DETAILS\n'
                        f'{"=" * 80}\n'
                        f'Tool Name: {tool_name}\n'
                        f'Function Name: {tool_call.function.name}\n'
                        f'Raw Arguments: {tool_call.function.arguments}\n'
                        f'Parsed Arguments: {tool_args}\n'
                        f'Error Type: {type(e).__name__}\n'
                        f'Error Message: {str(e)}\n'
                        f'Model Raw Output: {raw_output}\n'
                        f'{"=" * 80}\n'
                    )

                    # Early stop: if enabled, exit immediately on tool error
                    if agent_input.early_stop:
                        logger.warning('Early stop enabled: tool execution error occurred, stopping agent immediately')
                        exit_status = ExitStatus.TOOL_ERROR
                        platform_status = runtime_failure_status(e, during_tool=True)
                        exit_error = f'{type(e).__name__}: {e}'
                        failure_stage = 'tool'
                        # Add empty tool response to history and break
                        history.append({
                            'role': 'tool',
                            'tool_call_id': tool_call.id,
                            'content': f'Error executing tool: {e}',  # Empty content as requested
                        })
                        break  # Break from tool call loop

                    # Add placeholder message to history to ensure history grows for HyTuner
                    # HyTuner requires history length to increase on each call
                    # Use empty content to avoid polluting context while ensuring history growth
                    history.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': f'Error executing tool: {e}',  # Error content to avoid context pollution
                    })
                    total_observation_time += end_time - start_time
                    if observation_time_timeout and total_observation_time >= observation_time_timeout:
                        logger.info(
                            f'[CUMULATIVE OBSERVATION TIMEOUT] budget exhausted: {observation_time_timeout} seconds, used {total_observation_time} seconds'
                        )
                        exit_status = ExitStatus.TIMEOUT
                        exit_error = f'Observation budget exhausted: {observation_time_timeout}s'
                        failure_stage = 'observation_budget'
                        break
                    continue
                finally:
                    if timeline_collect_tool_args and timeline_tool_args_min_duration > 0:
                        elapsed = time.time() - start_time
                        if elapsed >= timeline_tool_args_min_duration:
                            tool_span.update(
                                attributes={
                                    'hytuner.tool_args': json.dumps(tool_args, ensure_ascii=False),
                                }
                            )
                    tool_span.finish()

                end_time = time.time()
                total_observation_time += end_time - start_time
                # Get output text and format with template
                raw_output = tool_response.output[0].get('text', '') if tool_response.output else ''
                if raw_output:
                    # Truncate long observation to avoid context overflow
                    truncated_output = maybe_truncate(raw_output, truncate_after=DEFAULT_TEXT_CONTENT_LIMIT)
                    if len(truncated_output) < len(raw_output):
                        logger.debug(f'Observation truncated: {len(raw_output)} -> {len(truncated_output)} chars')
                    output_text = prompts.next_step.format(observation=truncated_output)
                else:
                    output_text = prompts.next_step_no_output

                logger.debug(f'Tool output: {output_text[:200]}...')

                # Add tool response to history to ensure history grows for HyTuner
                # HyTuner requires history length to increase on each call
                history.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': output_text,
                })

                # Check for exit condition
                if ctx.tool_state.get('exit_status') == 'submitted':
                    exit_status = ExitStatus.SUBMITTED
                    break

                # Set the observation timeout to limit tool call
                if observation_time_timeout and total_observation_time >= observation_time_timeout:
                    logger.info(
                        f'[CUMULATIVE OBSERVATION TIMEOUT] budget exhausted: {observation_time_timeout} seconds, used {total_observation_time} seconds'
                    )
                    exit_status = ExitStatus.TIMEOUT
                    exit_error = f'Observation budget exhausted: {observation_time_timeout}s'
                    failure_stage = 'observation_budget'
                    break

            # Break outer loop if exit status is set
            if exit_status is not None:
                break

            iter_span.update(status=Status(code='OK'))
            iter_span.finish()
            iter_span = None
            iteration += 1
    except ContextLengthExceededError as e:
        logger.error(f'model context length exceeded: {e}, exiting.')
        exit_status = ExitStatus.TRUNCATION
        platform_status = PlatformExitStatus.CONTEXT_LENGTH_EXCEEDED
        exit_error = f'{type(e).__name__}: {e}'
        failure_stage = 'inference'
    except Exception as e:
        logger.exception(f'Agent encountered error: {e}')
        exit_status = ExitStatus.ERROR
        sandbox = exception_details(e).get('sandbox')
        platform_status = (
            sandbox_failure_status(sandbox, during_tool=current_stage == 'tool')
            if sandbox is not None
            else PlatformExitStatus.INFERENCE_ERROR
            if current_stage == 'inference'
            else runtime_failure_status(e, during_tool=current_stage == 'tool')
        )
        exit_error = f'{type(e).__name__}: {e}'
        failure_stage = current_stage
    finally:
        if iter_span:
            iter_span.finish()

    # If loop finished without setting exit_status, we hit max iterations
    if exit_status is None:
        exit_status = ExitStatus.MAX_ITERATIONS
        exit_error = f'Agent reached max_iterations={agent_input.max_iterations}'
        failure_stage = 'iteration_budget'
        logger.warning(f'Agent reached max iterations ({agent_input.max_iterations})')

    logger.debug(f'{agent_name} finished with status: {exit_status.value}, iterations: {iteration + 1}')

    # Set termination_reason for HyTuner training
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

    return _agent_output(
        agent_input,
        exit_status,
        min(iteration + 1, max(0, agent_input.max_iterations)),
        platform_status,
        exit_error,
        failure_stage,
        sandbox,
    )
