"""White-box mini-swe-agent for the nexus framework.

This is a transparent, Python-level reimplementation of the upstream
mini-swe-agent agent loop (``minisweagent.agents.default.DefaultAgent`` +
``LitellmModel`` + ``LocalEnvironment``), specialized for coding tasks
(SWE-bench). It is designed to be behaviorally equivalent to upstream
mini-swe-agent: the same prompts, tool schema, message shapes, control flow,
and per-command execution semantics, so the resolve rate matches.

The whole agent loop is visible here and runs host-side on nexus interfaces:

- The LLM is driven through ``ctx.openai_client`` (host).
- Bash commands run through the RUNTIME ``mini_swe_agent/bash`` tool, which
  executes a fresh ``bash -c`` subshell in the sandbox (mini's non-persistent
  ``LocalEnvironment`` semantics).

Patch extraction is done by ``swe_task`` via ``git diff`` (the agent output is
informational), so the loop only needs to drive the model to edit files and
reach the submit sentinel.
"""

import asyncio
import logging
import time

from typing import Any

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
from nexus.extensions.agents.installed_common import coerce_user_prompt
from nexus.extensions.agents.mini_swe_agent.constants import DEFAULT_ENV_TIMEOUT
from nexus.extensions.agents.mini_swe_agent.mini_core import (
    BASH_TOOL,
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    SYSTEM_TEMPLATE,
    TOOL_NAME,
    FormatError,
    check_submitted,
    format_observation,
    parse_toolcall_actions,
    render,
)
from nexus.runtime.client.base import RuntimeUnhealthyError
from nexus.trace import get_current_span_id, span_context, start_span, update_span
from nexus.trace.types import Status
from nexus.utils.timeout import response_inference_time


logger = logging.getLogger(__name__)


# Import the bash tool so it is registered when this agent module is imported.
Registry.import_module('nexus.extensions.agents.mini_swe_agent.tools')


# ============== Exit status vocabulary (aligned with swe_agent) ==============
# mini reports its stop reason using the same string vocabulary as
# ``swe_agent.ExitStatus`` (kept as plain strings to avoid a cross-agent import),
# so downstream config/parsing can treat mini and swe_agent uniformly.
EXIT_STATUS_SUBMITTED = 'submitted'
EXIT_STATUS_MAX_ITERATIONS = 'max_iterations'  # step/cost limit reached
EXIT_STATUS_TIMEOUT = 'timeout'
EXIT_STATUS_TOOL_ERROR = 'tool_error'  # repeated format / parse error
EXIT_STATUS_TRUNCATION = 'truncation'
EXIT_STATUS_ERROR = 'error'

# ============== Exit status -> termination reason (for HyTuner) ==============
EXIT_STATUS_TO_TERMINATION_REASON: dict[str, str] = {
    EXIT_STATUS_SUBMITTED: 'ENV_DONE',
    EXIT_STATUS_MAX_ITERATIONS: 'MAX_STEPS',
    EXIT_STATUS_TIMEOUT: 'TASK_TIMEOUT',
    EXIT_STATUS_TOOL_ERROR: 'FORMAT_ERROR',
    EXIT_STATUS_TRUNCATION: 'TRUNCATION',
    EXIT_STATUS_ERROR: 'UNHANDLED_ERROR',
}
DEFAULT_TERMINATION_REASON = 'UNHANDLED_ERROR'


# ============== Input / Output models ==============


class MiniSweAgentInput(AgentRunInput, WithUserPrompt, WithWorkingDir):
    """Input for the white-box mini-swe-agent.

    Defaults reproduce upstream mini-swe-agent's SWE-bench config. Extra keys
    passed via ``agent_config`` (e.g. ``model_id``, ``timeout``) are ignored.
    """

    max_iterations: int = Field(
        default=250,
        description='Max number of model calls (0 = unlimited). Aligned with swe_agent.max_iterations '
        '(this is upstream mini agent.step_limit).',
    )
    cost_limit: float = Field(
        default=0.0, description='Stop after exceeding this cost (0 = disabled). mini agent.cost_limit.'
    )
    wall_time_limit_seconds: int = Field(
        default=0,
        description='Stop after this many wall-clock seconds (0 = disabled). mini agent.wall_time_limit_seconds.',
    )
    max_consecutive_format_errors: int = Field(
        default=1,
        description='Exit after this many format errors in a row (0 = no limit). '
        'Set to 1 to disallow any retry. mini agent.max_consecutive_format_errors.',
    )
    env_timeout: int = Field(
        default=DEFAULT_ENV_TIMEOUT, description='Per-command timeout in seconds. mini environment.timeout.'
    )
    parallel_tool_calls: bool = Field(
        default=False, description='Value forwarded to the completion call (matches upstream model_kwargs).'
    )
    extra_env: dict[str, str] = Field(
        default_factory=dict, description='Extra env vars merged into every bash command.'
    )
    system_template: str = Field(default=SYSTEM_TEMPLATE, description='System prompt template (Jinja2).')
    instance_template: str = Field(
        default=INSTANCE_TEMPLATE, description='Task prompt template (Jinja2), uses {{task}} and {{cwd}}.'
    )
    observation_template: str = Field(
        default=OBSERVATION_TEMPLATE, description='Observation template (Jinja2), uses {{output}}.'
    )
    format_error_template: str = Field(
        default=FORMAT_ERROR_TEMPLATE, description='Format-error template (Jinja2), uses {{error}}.'
    )
    extra_completion_kwargs: dict[str, Any] = Field(
        default_factory=dict, description='Extra kwargs forwarded to create_completion.'
    )


class MiniSweAgentOutput(AgentRunOutput):
    """Output from the white-box mini-swe-agent run.

    ``exit_status`` and ``iterations`` mirror ``swe_agent``'s output vocabulary so
    both agents can be parsed uniformly; ``submission`` is a mini-specific extra.
    """

    exit_status: str = Field(
        description="How the agent exited (swe_agent vocabulary), e.g. 'submitted', 'max_iterations', 'tool_error'."
    )
    iterations: int = Field(default=0, description='Number of model calls (iterations) completed.')
    submission: str = Field(
        default='', description='The submitted patch text (informational; swe_task re-derives via git diff).'
    )


# ============== Agent implementation ==============


@agent_registry.register('mini_swe_agent', meta={'gateway': {'scaffold_policy': 'generic'}})
async def run_mini_swe_agent(agent_input: MiniSweAgentInput) -> MiniSweAgentOutput:
    """Run the white-box mini-swe-agent loop.

    Faithful port of ``DefaultAgent.run``:
    1. Seed messages with rendered system + instance templates.
    2. Loop: query the model (checking step/cost/wall limits first), parse tool
       calls into actions, execute each action as a bash command, render the
       observation, and check for the submit sentinel.
    3. A format error (no/invalid tool call) keeps the malformed assistant turn
       (upstream drops it) and appends a user error message, so the context stays
       consistent with the recorded training tokens;
       ``max_consecutive_format_errors`` in a row exits with ``RepeatedFormatError``.
    """
    ctx = require_context()
    client = ctx.openai_client

    task = coerce_user_prompt(agent_input.user_prompt)
    cwd = agent_input.working_dir or ctx.task_config.get('working_dir')
    if not cwd:
        raise ValueError('working_dir is required for mini_swe_agent')

    # Seed tool_state so the RUNTIME bash tool (sandbox side) picks up the
    # working dir, per-command timeout, and extra env vars.
    ctx.tool_state['working_dir'] = cwd
    ctx.tool_state['env_timeout'] = agent_input.env_timeout
    ctx.tool_state['bash_extra_env'] = dict(agent_input.extra_env)

    executor = ToolExecutor([tool_registry.get('mini_swe_agent/bash')])
    observation_timeout = agent_input.env_timeout + 120

    # Seed conversation with system + instance messages (rendered like mini).
    messages: list[dict] = [
        {'role': 'system', 'content': render(agent_input.system_template)},
        {'role': 'user', 'content': render(agent_input.instance_template, task=task, cwd=cwd)},
    ]

    n_calls = 0
    n_consecutive_format_errors = 0
    cost = 0.0
    start_time = time.time()
    total_inference_time = 0.0
    total_observation_time = 0.0

    exit_status: str | None = None
    submission = ''

    try:
        while True:
            # --- query(): limit checks come first, before consuming a call ---
            if (0 < agent_input.max_iterations <= n_calls) or (0 < agent_input.cost_limit <= cost):
                exit_status = EXIT_STATUS_MAX_ITERATIONS
                logger.info('mini_swe_agent: max_iterations/cost limit reached (n_calls=%d, cost=%.4f)', n_calls, cost)
                break
            if 0 < agent_input.wall_time_limit_seconds <= int(time.time() - start_time):
                exit_status = EXIT_STATUS_TIMEOUT
                logger.info('mini_swe_agent: timeout after %ds', int(time.time() - start_time))
                break

            iter_span = start_span(
                'agent.iteration',
                attributes={'type': 'agent_iteration', 'iteration': n_calls + 1},
            )
            try:
                n_calls += 1

                # --- model.query() ---
                completion_kwargs: dict[str, Any] = {
                    'tools': [BASH_TOOL],
                    'parallel_tool_calls': agent_input.parallel_tool_calls,
                    **agent_input.extra_completion_kwargs,
                }
                hook_request_meta = completion_kwargs.pop('request_meta', None)
                request_meta = dict(hook_request_meta or {})
                request_meta.update(iteration=n_calls, retry=0)
                inference_start = time.time()
                with span_context(
                    'agent.inference',
                    attributes={'type': 'agent_inference', 'iteration': n_calls},
                ):
                    response = await client.create_completion(
                        messages=messages,
                        request_meta=request_meta,
                        **completion_kwargs,
                    )
                total_inference_time += response_inference_time(response, time.time() - inference_start)

                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # --- parse tool calls into actions (raises FormatError) ---
                try:
                    actions = parse_toolcall_actions(
                        message.tool_calls or [],
                        format_error_template=agent_input.format_error_template,
                        finish_reason=finish_reason,
                    )
                except FormatError as e:
                    # Deviation from upstream (which drops the malformed turn):
                    # keep the assistant message, then append the error feedback, so
                    # the context matches the tokens the HyTuner client already
                    # cached. The next call then appends instead of retokenizing,
                    # keeping the training tokens consistent with the inference context.
                    messages.append(message.model_dump(exclude_none=True))
                    messages.extend(e.messages)
                    if finish_reason == 'length':
                        # The turn is malformed because generation was cut at the response
                        # budget, not because the model cannot format a tool call. Retrying
                        # is pointless (the budget is spent) and reporting tool_error would
                        # bury a length problem under a format one.
                        exit_status = EXIT_STATUS_TRUNCATION
                        logger.warning('mini_swe_agent: generation cut at the response budget -> truncation')
                        break
                    n_consecutive_format_errors += 1
                    if 0 < agent_input.max_consecutive_format_errors <= n_consecutive_format_errors:
                        exit_status = EXIT_STATUS_TOOL_ERROR
                        logger.warning(
                            'mini_swe_agent: repeated format error (%d in a row) -> tool_error',
                            n_consecutive_format_errors,
                        )
                        break
                    iter_span.update(status=Status(code='OK'))
                    continue

                # --- success: keep the assistant message ---
                messages.append(message.model_dump(exclude_none=True))

                # --- execute_actions() ---
                submitted = False
                for action in actions:
                    obs_start = time.time()
                    tool_span = start_span(
                        'agent.observation',
                        attributes={'type': 'agent_observation', 'iteration': n_calls, 'tool_name': TOOL_NAME},
                    )
                    try:
                        tool_response = await executor.call(
                            TOOL_NAME, {'command': action['command']}, timeout=observation_timeout
                        )
                    except (RuntimeUnhealthyError, TimeoutError, asyncio.TimeoutError) as e:
                        total_observation_time += time.time() - obs_start
                        tool_span.update(status=Status(code='ERROR', message=str(e)))
                        tool_span.finish()
                        # Keep assistant/tool pairing valid, then stop the run.
                        messages.append({
                            'role': 'tool',
                            'tool_call_id': action['tool_call_id'],
                            'content': f'Error executing tool: {e}',
                        })
                        exit_status = (
                            EXIT_STATUS_TIMEOUT
                            if isinstance(e, (TimeoutError, asyncio.TimeoutError))
                            else EXIT_STATUS_ERROR
                        )
                        logger.exception('mini_swe_agent: tool execution failed, stopping run')
                        submitted = False
                        break

                    total_observation_time += time.time() - obs_start
                    tool_span.update(status=Status(code='OK'))
                    tool_span.finish()

                    raw = tool_response.output[0].get('text', '') if tool_response.output else ''
                    returncode = int(tool_response.meta.get('returncode', -1))
                    exception_info = tool_response.meta.get('exception_info', '') or ''
                    output_dict = {'output': raw, 'returncode': returncode, 'exception_info': exception_info}

                    messages.append(
                        format_observation(action, output_dict, observation_template=agent_input.observation_template)
                    )

                    sub = check_submitted(raw, returncode)
                    if sub is not None:
                        submission = sub
                        submitted = True
                        break

                if exit_status is not None:
                    # Set by the tool-error handler above.
                    break
                if submitted:
                    exit_status = EXIT_STATUS_SUBMITTED
                    logger.debug('mini_swe_agent: submitted (submission_len=%d)', len(submission))
                    break

                # Clean step: reset the consecutive format error counter.
                n_consecutive_format_errors = 0
                iter_span.update(status=Status(code='OK'))
            finally:
                iter_span.finish()
    except ContextLengthExceededError as e:
        logger.error('mini_swe_agent: context length exceeded: %s', e)
        exit_status = EXIT_STATUS_TRUNCATION
    except Exception as e:  # noqa: BLE001 - never crash the task; let swe_task still extract the patch
        logger.exception('mini_swe_agent: unhandled error: %s', e)
        exit_status = EXIT_STATUS_ERROR

    if exit_status is None:
        exit_status = EXIT_STATUS_ERROR

    # Record the termination reason on the current span (for HyTuner training).
    termination_reason = EXIT_STATUS_TO_TERMINATION_REASON.get(exit_status, DEFAULT_TERMINATION_REASON)
    span_id = get_current_span_id()
    if span_id:
        update_span(
            span_id,
            attributes={
                'hytuner.termination_reason': termination_reason,
                'hytuner.observation_time': total_observation_time,
                'hytuner.inference_time': total_inference_time,
            },
        )

    logger.info('mini_swe_agent finished: exit_status=%s, iterations=%d', exit_status, n_calls)
    # ``n_calls`` (model calls) == swe_agent's ``iterations`` (one call per loop turn).
    return MiniSweAgentOutput(exit_status=exit_status, iterations=n_calls, submission=submission)


__all__ = ['MiniSweAgentInput', 'MiniSweAgentOutput', 'run_mini_swe_agent']
