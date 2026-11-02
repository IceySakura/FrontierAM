"""Async timeout utilities built on the standard library's asyncio.timeout."""

from __future__ import annotations

import asyncio
import logging
import math
import time

from asyncio import timeout as async_timeout
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass
from typing import Any, Protocol


logger = logging.getLogger(__name__)


@dataclass
class _TimeoutScope:
    """Retain one inference scope's deadline and overlapping pause leases."""

    handle: Any = None
    deadline: float | None = None
    pause_started_at: float | None = None
    paused_time: float = 0.0
    holders: int = 0
    active: bool = True

    def pause(self) -> bool:
        if not self.active:
            return False
        if self.holders == 0:
            self.deadline = None if self.handle is None else self.handle.when()
            if self.deadline is not None:
                if self.deadline <= asyncio.get_running_loop().time():
                    return False
                try:
                    self.handle.reschedule(None)
                except RuntimeError as error:
                    logger.warning('inference timeout pause failed: %s', error)
                    return False
            self.pause_started_at = time.perf_counter()
        self.holders += 1
        return True

    def resume(self) -> None:
        if not self.active:
            return
        if self.holders <= 0:
            raise RuntimeError('Inference timeout pause lease underflow')
        self.holders -= 1
        if self.holders == 0:
            assert self.pause_started_at is not None
            duration = time.perf_counter() - self.pause_started_at
            self.paused_time += duration
            self.pause_started_at = None
            if self.deadline is not None:
                try:
                    self.handle.reschedule(self.deadline + duration)
                except RuntimeError as error:
                    logger.warning('inference timeout resume failed: %s', error)

    def get_paused_time(self) -> float:
        if self.pause_started_at is None:
            return self.paused_time
        return self.paused_time + time.perf_counter() - self.pause_started_at


_ACTIVE_TIMEOUT_SCOPES: ContextVar[tuple[_TimeoutScope, ...]] = ContextVar('nexus_active_timeout_scopes', default=())
RUNNER_TIMEOUT_PAUSE_GROUP_KEY = 'nexus.hytuner.abort_timeout_pause_group'


class SharedExecutionTimeoutBudget(Protocol):
    """Expose caller-owned shared deadline leases to an execution controller."""

    async def acquire(self, request_id: str) -> None: ...

    async def release(self, request_id: str) -> None: ...


class ExecutionTimeoutController:
    """Control captured inference scopes and an explicitly supplied shared budget."""

    def __init__(
        self,
        scopes: tuple[_TimeoutScope, ...],
        shared_budget: SharedExecutionTimeoutBudget | None,
        request_id: str,
    ) -> None:
        self._context = copy_context()
        self._scopes = scopes
        self._shared_budget = shared_budget
        self._request_id = request_id
        self._lock = asyncio.Lock()
        self._holders = 0
        self._paused_scopes: list[_TimeoutScope] = []

    async def pause(self) -> None:
        """Acquire a lease, rolling it back before propagating cancellation."""
        cancellation = await _settle_timeout_update(self._pause(), self._context)
        if cancellation is not None:
            await _settle_timeout_update(self._resume(), self._context)
            raise cancellation

    async def resume(self) -> None:
        """Release one lease, completing restoration before propagating cancellation."""
        cancellation = await _settle_timeout_update(self._resume(), self._context)
        if cancellation is not None:
            raise cancellation

    async def _pause(self) -> None:
        async with self._lock:
            if self._holders > 0:
                self._holders += 1
                return
            try:
                for scope in self._scopes:
                    if scope.pause():
                        self._paused_scopes.append(scope)
                if self._shared_budget is not None:
                    await self._shared_budget.acquire(self._request_id)
            except BaseException:
                self._restore_scopes()
                raise
            self._holders = 1

    async def _resume(self) -> None:
        async with self._lock:
            if self._holders <= 0:
                raise RuntimeError(f'execution timeout pause lease underflow; request_id={self._request_id}')
            self._holders -= 1
            if self._holders == 0:
                try:
                    if self._shared_budget is not None:
                        await self._shared_budget.release(self._request_id)
                finally:
                    self._restore_scopes()

    def _restore_scopes(self) -> None:
        try:
            for scope in reversed(self._paused_scopes):
                scope.resume()
        finally:
            self._paused_scopes.clear()


def capture_execution_timeouts(
    request_id: str,
    *,
    include_inference: bool = True,
    shared_budget: SharedExecutionTimeoutBudget | None = None,
) -> ExecutionTimeoutController | None:
    """Capture active inference deadlines and an optional caller-owned shared budget."""
    scopes = _ACTIVE_TIMEOUT_SCOPES.get() if include_inference else ()
    if not scopes and shared_budget is None:
        return None
    return ExecutionTimeoutController(scopes, shared_budget, request_id)


async def _settle_timeout_update(
    operation: Coroutine[Any, Any, None], context: Context
) -> asyncio.CancelledError | None:
    """Finish a control transition in its captured context before propagating cancellation."""
    update_task = context.run(asyncio.create_task, operation)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(update_task)
            return cancellation
        except asyncio.CancelledError as error:
            if update_task.cancelled():
                raise
            if cancellation is None:
                cancellation = error


class InferenceTimeoutError(TimeoutError):
    """Raised when an agent's remaining inference-time budget is exhausted."""


@dataclass
class TimeBudget:
    """Tracks elapsed time against an optional total timeout budget."""

    total_timeout: float | None
    elapsed: float = 0.0

    def remaining(self) -> float | None:
        if self.total_timeout is None:
            return None
        return max(0.0, self.total_timeout - self.elapsed)

    def consume(self, duration: float) -> None:
        self.elapsed += duration

    def exhausted(self) -> bool:
        return self.total_timeout is not None and self.elapsed >= self.total_timeout


def reported_inference_time(response: Any) -> float | None:
    """Return provider-reported inference seconds from a completion response."""
    provider_fields = getattr(response, 'provider_specific_fields', None)
    if not isinstance(provider_fields, dict):
        model_extra = getattr(response, 'model_extra', None)
        if isinstance(model_extra, dict):
            provider_fields = model_extra.get('provider_specific_fields')
            if not isinstance(provider_fields, dict) and 'inference_time' in model_extra:
                provider_fields = model_extra
    provider_fields = provider_fields or {}
    raw_value = provider_fields.get('inference_time') if isinstance(provider_fields, dict) else None
    if raw_value is None:
        return None
    try:
        inference_time = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(inference_time) or inference_time < 0.0:
        return None
    return inference_time


def response_inference_time(response: Any, fallback: float) -> float:
    """Return provider-reported inference seconds, falling back to local wall time."""
    inference_time = reported_inference_time(response)
    return max(0.0, fallback) if inference_time is None else inference_time


def credit_runner_timeout_for_response(response: Any, wall_time: float, *, started_at: float | None = None) -> float:
    """Exclude completed non-inference time from the enclosing runner timeout.

    This correction is post-hoc. If the runner deadline fires before the
    completion returns, no response metadata is available to apply.
    """
    inference_time = reported_inference_time(response)
    if inference_time is None:
        return max(0.0, wall_time)

    provider_fields = getattr(response, 'provider_specific_fields', None) or {}
    raw_already_credited = (
        provider_fields.get('_runner_timeout_credited') if isinstance(provider_fields, dict) else None
    )
    if raw_already_credited is None:
        already_credited = 0.0
    else:
        try:
            already_credited = float(raw_already_credited)
        except (TypeError, ValueError):
            already_credited = 0.0
    if not math.isfinite(already_credited) or already_credited < 0.0:
        already_credited = 0.0

    excluded_time = max(0.0, wall_time - inference_time - already_credited)
    if excluded_time == 0.0:
        return inference_time

    try:
        # Import lazily to avoid a module cycle: ExecutionContext imports the
        # OpenAI client base, which imports this module.
        from nexus.core.context import require_context

        context = require_context()
        timeout_handle = context.timeout_handle
    except RuntimeError:
        return inference_time

    if timeout_handle is None:
        return inference_time
    pause_group = None
    if started_at is not None:
        runtime_shared = getattr(context, 'runtime_shared', None)
        group = None if runtime_shared is None else runtime_shared.get_optional(RUNNER_TIMEOUT_PAUSE_GROUP_KEY)
        if group is not None and getattr(group, 'track_shared_pause', False):
            pause_group = group
            already_credited = max(already_credited, group.get_paused_time(started_at))
            excluded_time = max(0.0, wall_time - inference_time - already_credited)
            if isinstance(provider_fields, dict):
                provider_fields['_runner_timeout_credited'] = already_credited
            if excluded_time == 0.0:
                return inference_time
    try:
        deadline = timeout_handle.when()
        if deadline is not None:
            timeout_handle.reschedule(deadline + excluded_time)
        elif pause_group is not None:
            pause_group.credit_timeout(excluded_time)
    except Exception as error:
        logger.warning('runner timeout inference-time adjustment failed: %s', error)

    return inference_time


@asynccontextmanager
async def with_timeout(timeout_sec: float | None):
    """Apply timeout when configured; no-op otherwise."""
    if timeout_sec is not None and timeout_sec <= 0:
        raise TimeoutError('Timeout budget exhausted')

    scope = _TimeoutScope()
    token = _ACTIVE_TIMEOUT_SCOPES.set((*_ACTIVE_TIMEOUT_SCOPES.get(), scope))
    try:
        if timeout_sec is None:
            yield
            return
        async with async_timeout(timeout_sec) as handle:
            scope.handle = handle
            yield
    finally:
        scope.paused_time = scope.get_paused_time()
        scope.pause_started_at = None
        scope.active = False
        _ACTIVE_TIMEOUT_SCOPES.reset(token)


def _active_wall_time(started: float, scope: _TimeoutScope | None) -> float:
    elapsed = time.monotonic() - started
    paused_time = 0.0 if scope is None else scope.get_paused_time()
    return elapsed if paused_time == 0.0 else max(0.0, elapsed - paused_time)


async def await_completion_with_time_budget(
    budget: TimeBudget,
    operation: Callable[[], Awaitable[Any]],
) -> Any:
    """Run one completion under the remaining budget and reconcile it post-hoc."""
    started = time.monotonic()
    remaining = budget.remaining()
    timeout_scope = None
    if remaining is None:
        try:
            async with with_timeout(None):
                scopes = _ACTIVE_TIMEOUT_SCOPES.get()
                timeout_scope = scopes[-1] if scopes else None
                response = await operation()
        except Exception:
            budget.consume(_active_wall_time(started, timeout_scope))
            raise
    else:
        provider_timeout: TimeoutError | None = None
        try:
            async with with_timeout(remaining):
                scopes = _ACTIVE_TIMEOUT_SCOPES.get()
                timeout_scope = scopes[-1] if scopes else None
                try:
                    response = await operation()
                except TimeoutError as error:
                    # Suppress the provider timeout until after the budget
                    # context exits so its own TimeoutError remains distinct.
                    provider_timeout = error
        except TimeoutError as error:
            budget.consume(_active_wall_time(started, timeout_scope))
            raise InferenceTimeoutError('Inference time budget exhausted') from error
        except Exception:
            budget.consume(_active_wall_time(started, timeout_scope))
            raise
        if provider_timeout is not None:
            budget.consume(_active_wall_time(started, timeout_scope))
            raise provider_timeout
    budget.consume(response_inference_time(response, _active_wall_time(started, timeout_scope)))
    if budget.exhausted():
        raise InferenceTimeoutError('Inference time budget exhausted')
    return response


__all__ = [
    'ExecutionTimeoutController',
    'InferenceTimeoutError',
    'TimeBudget',
    'SharedExecutionTimeoutBudget',
    'async_timeout',
    'await_completion_with_time_budget',
    'capture_execution_timeouts',
    'credit_runner_timeout_for_response',
    'reported_inference_time',
    'response_inference_time',
    'with_timeout',
]
