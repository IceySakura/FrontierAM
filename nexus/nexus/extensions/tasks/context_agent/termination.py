"""Report ContextAgent task outcomes through the existing tracing contract."""

from __future__ import annotations

import asyncio

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar, get_type_hints

from nexus.core.context import require_context
from nexus.trace import get_current_span_id, update_span


class _TaskOutcome(Protocol):
    termination_reason: str


TaskParameters = ParamSpec('TaskParameters')
TaskOutput = TypeVar('TaskOutput', bound=_TaskOutcome)


def report_task_termination(
    task: Callable[TaskParameters, Awaitable[TaskOutput]],
) -> Callable[TaskParameters, Awaitable[TaskOutput]]:
    """Publish the final task outcome without changing its result or exception.

    Apply inside task registration so the task span closes after this update.
    Both HyTuner trace consumers read this attribute before finalizing the run.
    """

    @wraps(task)
    async def wrapped(*args: TaskParameters.args, **kwargs: TaskParameters.kwargs) -> TaskOutput:
        span_id = get_current_span_id()
        reason = 'UNHANDLED_ERROR'
        try:
            output = await task(*args, **kwargs)
            reason = output.termination_reason
            return output
        except asyncio.CancelledError:
            if span_id:
                timeout = require_context().timeout_handle
                if timeout is not None and timeout.expired():
                    reason = 'TASK_TIMEOUT'
            raise
        except Exception as error:
            reason = {
                'TimeoutError': 'TASK_TIMEOUT',
                'EnvironmentInitError': 'INITIALIZATION_ERROR',
                'EnterpriseRuntimeStartupTimeoutError': 'ENV_TIMEOUT',
                'EnterpriseAgentTimeoutError': 'TASK_TIMEOUT',
                'EnterpriseRuntimeRPCError': 'ENV_TIMEOUT',
                'EnterpriseJudgeError': 'REWARD_ERROR',
                'EnterpriseJudgeTimeoutError': 'REWARD_ERROR',
                'EnterpriseJudgeLengthError': 'REWARD_ERROR',
            }.get(type(error).__name__, 'UNHANDLED_ERROR')
            raise
        finally:
            if span_id:
                update_span(span_id, attributes={'agent.termination_reason': reason})

    # NOTE(yufeihuang): Registry input discovery resolves annotations in the wrapper's module.
    wrapped.__annotations__ = get_type_hints(task)
    return wrapped
