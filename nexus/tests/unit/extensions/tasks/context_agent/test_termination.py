"""ContextAgent reports outcomes to unchanged ordinary and Gateway runners."""

import asyncio

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core.task import validate_task
from nexus.extensions.tasks.context_agent import termination
from nexus.extensions.tasks.context_agent.enterprisebench_native.errors import (
    EnterpriseAgentTimeoutError,
    EnterpriseJudgeError,
    EnterpriseJudgeLengthError,
    EnterpriseJudgeTimeoutError,
    EnterpriseRuntimeRPCError,
    EnterpriseRuntimeStartupTimeoutError,
)
from nexus.extensions.tasks.context_agent.enterprisebench_native.task import (
    EnterpriseBenchNativeTaskInput,
    EnvironmentInitError,
    enterprisebench_native_task,
)
from nexus.extensions.tasks.context_agent.general_task import (
    ContextAgentGeneralTaskInput,
    ContextAgentGeneralTaskOutput,
    run_context_agent_general_task,
)
from nexus.extensions.tasks.context_agent.output import agent_termination_reason
from nexus.gateway.nexus_run_tracker import NexusRunTracker
from nexus.runner import single
from nexus.runner.hytuner import _resolve_termination_reason
from nexus.trace import processor_context, span_context
from nexus.trace.processors.hytuner import HytunerProcessor
from nexus.utils.timeout import async_timeout


@pytest.mark.parametrize(
    ('exit_status', 'expected'),
    [
        ('SUCCESS', 'ENV_DONE'),
        ('MAX_ITERATIONS', 'MAX_STEPS'),
        ('TRUNCATION', 'TRUNCATION'),
        ('OUTPUT_LENGTH_EXCEEDED', 'TRUNCATION'),
        ('FORMAT_ERROR', 'FORMAT_ERROR'),
        ('INITIALIZATION_ERROR', 'INITIALIZATION_ERROR'),
        ('ENV_ERROR', 'ENV_ERROR'),
        ('ENV_TIMEOUT', 'ENV_TIMEOUT'),
        ('TIMEOUT', 'TASK_TIMEOUT'),
        ('PROVIDER_ERROR', 'UNHANDLED_ERROR'),
        ('REWARD_ERROR', 'REWARD_ERROR'),
    ],
)
async def test_task_trace_overrides_agent_reason_in_both_processors(exit_status: str, expected: str) -> None:
    @termination.report_task_termination
    async def task(task_input: ContextAgentGeneralTaskInput) -> ContextAgentGeneralTaskOutput:
        with span_context('agent', attributes={'agent.termination_reason': exit_status}):
            pass
        return ContextAgentGeneralTaskOutput(
            score=0.0, termination_reason=agent_termination_reason(SimpleNamespace(exit_status=exit_status))
        )

    registered = validate_task(task, 'test_context_outcome', {}).wrapped_obj
    assert registered is not None, 'Task registration must produce a traced callable'
    ordinary = HytunerProcessor(file_path='', trajectory_id='test_context')
    gateway = NexusRunTracker()
    with processor_context([ordinary, gateway]):
        output = await registered(ContextAgentGeneralTaskInput(context_question='test', run_agent=AsyncMock()))

    for reported_reason in (ordinary.termination_reason, gateway.termination_reason):
        assert reported_reason == expected, 'Task span must override the earlier Agent span'
        result = single.RunOutput(success=True, task_output=output)
        assert _resolve_termination_reason(reported_reason, result, gateway_truncated=True) == expected


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (EnvironmentInitError('boot failed'), 'INITIALIZATION_ERROR'),
        (EnterpriseRuntimeStartupTimeoutError('boot', 'timeout', elapsed=1), 'ENV_TIMEOUT'),
        (EnterpriseRuntimeRPCError('runtime', 'timeout', elapsed=1), 'ENV_TIMEOUT'),
        (EnterpriseAgentTimeoutError('agent', 'timeout', elapsed=1), 'TASK_TIMEOUT'),
        (EnterpriseJudgeError('judge', 'failed', elapsed=1), 'REWARD_ERROR'),
        (EnterpriseJudgeTimeoutError('judge', 'timeout', elapsed=1), 'REWARD_ERROR'),
        (EnterpriseJudgeLengthError('judge', 'length', elapsed=1), 'REWARD_ERROR'),
        (TimeoutError('timeout'), 'TASK_TIMEOUT'),
        (RuntimeError('failed'), 'UNHANDLED_ERROR'),
    ],
)
async def test_task_failure_overrides_stale_truncation_without_changing_exception(
    error: Exception, expected: str
) -> None:
    @termination.report_task_termination
    async def task() -> ContextAgentGeneralTaskOutput:
        with span_context('agent', attributes={'agent.termination_reason': 'TRUNCATION'}):
            pass
        raise error

    ordinary = HytunerProcessor(file_path='', trajectory_id='test_context')
    gateway = NexusRunTracker()
    with processor_context([ordinary, gateway]):
        with pytest.raises(type(error)) as caught:
            with span_context('task', attributes={'type': 'task'}):
                await task()

    assert caught.value is error, 'Failure reporting must preserve the original exception'
    for reported_reason in (ordinary.termination_reason, gateway.termination_reason):
        assert reported_reason == expected, 'Failure must replace the earlier truncation'
        result = single.RunOutput(success=False, error=single.ErrorInfo(type=type(error).__name__, message=str(error)))
        assert _resolve_termination_reason(reported_reason, result, gateway_truncated=True) == expected


@pytest.mark.parametrize('deadline_expired', [False, True])
async def test_cancellation_preserves_exception_and_distinguishes_runner_deadline(
    monkeypatch: pytest.MonkeyPatch, deadline_expired: bool
) -> None:
    context = SimpleNamespace(timeout_handle=None)
    monkeypatch.setattr(termination, 'require_context', lambda: context)

    @termination.report_task_termination
    async def task() -> ContextAgentGeneralTaskOutput:
        with span_context('agent', attributes={'agent.termination_reason': 'TRUNCATION'}):
            pass
        if deadline_expired:
            await asyncio.Event().wait()
        raise asyncio.CancelledError()

    ordinary = HytunerProcessor(file_path='', trajectory_id='test_context')
    gateway = NexusRunTracker()

    async def run_with_deadline() -> None:
        async with async_timeout(0.01 if deadline_expired else None) as timeout:
            context.timeout_handle = timeout
            with span_context('task', attributes={'type': 'task'}):
                await task()

    with processor_context([ordinary, gateway]):
        with pytest.raises(TimeoutError if deadline_expired else asyncio.CancelledError):
            await run_with_deadline()

    expected = 'TASK_TIMEOUT' if deadline_expired else 'UNHANDLED_ERROR'
    assert ordinary.termination_reason == gateway.termination_reason == expected


@pytest.mark.parametrize(
    ('task', 'input_type'),
    [
        (run_context_agent_general_task, ContextAgentGeneralTaskInput),
        (enterprisebench_native_task, EnterpriseBenchNativeTaskInput),
    ],
)
def test_registered_context_task_preserves_input_discovery(task, input_type) -> None:
    assert single._get_input_type(task) is input_type, 'Runner must discover the original task input type'
