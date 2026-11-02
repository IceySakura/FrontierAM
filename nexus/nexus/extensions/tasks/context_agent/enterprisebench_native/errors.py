"""Task-local failure attribution; intentionally independent of runner timeouts."""

import asyncio
import time

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from nexus.utils.timeout import async_timeout


class EnterpriseStageError(RuntimeError):
    """A failed EnterpriseBench stage, not an expired outer runner deadline."""

    def __init__(self, stage: str, detail: str, *, elapsed: float, limit: float | None = None):
        self.stage = stage
        self.elapsed = elapsed
        self.limit = limit
        super().__init__(f'stage={stage} elapsed={elapsed:.3f}s limit={limit}s: {detail}')


class EnterpriseJudgeError(EnterpriseStageError):
    """The verifier did not produce a usable evaluation."""


class EnterpriseJudgeTimeoutError(EnterpriseJudgeError):
    """The total Judge deadline expired, including preparation and retries."""


class EnterpriseAgentTimeoutError(EnterpriseStageError):
    """The Agent rollout deadline expired."""


class EnterpriseRuntimeStartupTimeoutError(EnterpriseStageError):
    """The environment startup deadline expired."""


class EnterpriseRuntimeRPCError(EnterpriseStageError):
    """An inner operation timed out before its enclosing stage deadline."""


@asynccontextmanager
async def enterprise_stage(
    stage: str,
    limit: float | None = None,
    timeout_error: type[EnterpriseStageError] = EnterpriseRuntimeRPCError,
) -> AsyncIterator[None]:
    """Convert only local timeouts; outer cancellation must reach the runner."""
    started = time.monotonic()
    deadline = async_timeout(limit)
    try:
        async with deadline:
            yield
    except (TimeoutError, asyncio.TimeoutError) as exc:
        error_type = timeout_error if deadline.expired() else EnterpriseRuntimeRPCError
        raise error_type(
            stage,
            f'{type(exc).__name__}: {exc}' if not deadline.expired() else 'stage deadline exceeded',
            elapsed=time.monotonic() - started,
            limit=limit,
        ) from exc


class EnterpriseJudgeLengthError(EnterpriseJudgeError):
    """Judge exceeded context/output budget; never retry or compact this rollout."""


def judge_length_reason(payload: Any) -> str | None:
    """Recognize structured provider length errors without scanning prompt text."""
    if not isinstance(payload, dict):
        return None
    details = payload.get('incomplete_details') or {}
    if (
        payload.get('status') == 'incomplete'
        and isinstance(details, dict)
        and details.get('reason') in {'max_output_tokens', 'max_tokens', 'context_length_exceeded'}
    ):
        return str(details['reason'])
    error = payload.get('error', payload)
    if isinstance(error, dict):
        code = error.get('code') or error.get('type')
        if code in {'context_length_exceeded', 'max_output_tokens', 'max_tokens', 'model_context_window_exceeded'}:
            return str(code)
        message = str(error.get('message', '')).lower()
        if any(
            term in message
            for term in (
                'maximum context length',
                'context length exceeded',
                'context window exceeded',
                'exceeds the context window',
                'input is too long',
            )
        ):
            return 'context_length_exceeded'
    return None
