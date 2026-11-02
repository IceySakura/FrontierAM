"""Task-local Judge progress and failure limits; never inspect or log credentials."""

import asyncio
import time

from typing import Any

from nexus.extensions.tasks.context_agent.enterprisebench_native.errors import (
    EnterpriseJudgeError,
    EnterpriseJudgeLengthError,
)


class JudgeMonitor:
    def __init__(self, *, max_failures: int = 3, max_rate_limits: int = 3, idle_timeout: float = 600):
        self.started = time.monotonic()
        self.last_progress = self.started
        self.max_failures = max_failures
        self.max_rate_limits = max_rate_limits
        self.idle_timeout = idle_timeout
        self.failures = 0
        self.rate_limits = 0
        self.requests = 0
        self.pending = 0
        self.first_request_seconds: float | None = None
        self.events: list[dict[str, Any]] = []
        self.runtime_stage: dict[str, Any] | None = None
        self.failure: EnterpriseJudgeError | None = None
        self.stopped = asyncio.Event()

    def record(self, event: str, **fields: Any) -> None:
        self.events.append({'event': event, 'elapsed_seconds': time.monotonic() - self.started, **fields})

    def progress(self, event: str) -> None:
        self.last_progress = time.monotonic()
        self.record(event)

    def start_request(self) -> tuple[int, float]:
        if self.failure is not None:
            raise self.failure
        self.requests += 1
        self.pending += 1
        if self.first_request_seconds is None:
            self.first_request_seconds = time.monotonic() - self.started
        self.record('model_request_start', request=self.requests)
        return self.requests, time.monotonic()

    def finish_request(self, request: int, started: float, error: BaseException | None) -> None:
        self.pending -= 1
        status = getattr(error, 'status', None)
        if not isinstance(status, int):
            status = getattr(error, 'status_code', None)
        if status is None:
            status = getattr(getattr(error, 'response', None), 'status_code', None)
        self.record(
            'model_request_end',
            request=request,
            duration_seconds=time.monotonic() - started,
            outcome='success' if error is None else type(error).__name__,
            status_code=status,
        )
        if error is None:
            self.failures = self.rate_limits = 0
            self.progress('model_response_received')
            return
        if isinstance(error, asyncio.CancelledError):
            return
        self.failures += 1
        self.rate_limits = self.rate_limits + 1 if status == 429 else 0
        if self.max_rate_limits and self.rate_limits >= self.max_rate_limits:
            self.trip('judge.rate_limit', f'{self.rate_limits} consecutive logical requests failed with HTTP 429')
        elif self.max_failures and self.failures >= self.max_failures:
            self.trip('judge.model_failures', f'{self.failures} consecutive logical model requests failed')

    def trip(self, stage: str, detail: str) -> None:
        if self.failure is None:
            self.failure = EnterpriseJudgeError(stage, detail, elapsed=time.monotonic() - self.started)
            self.record('aborted', stage=stage, detail=detail)
            self.stopped.set()

    def length_failure(self, reason: str) -> EnterpriseJudgeError:
        if self.failure is None:
            self.failure = EnterpriseJudgeLengthError(
                'judge.length',
                reason,
                elapsed=time.monotonic() - self.started,
            )
            self.record('aborted', stage='judge.length', detail=reason)
            self.stopped.set()
        return self.failure

    def check_idle(self) -> None:
        idle = time.monotonic() - self.last_progress
        if self.idle_timeout and idle >= self.idle_timeout:
            stage = 'judge.first_request' if self.first_request_seconds is None else 'judge.no_progress'
            current = self.runtime_stage or {}
            self.trip(
                stage,
                f'No completed preparation step, model response or tool completion for {idle:.1f}s; '
                f'last_runtime_stage={current.get("stage")} event={current.get("event")}',
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            'first_request_seconds': self.first_request_seconds,
            'last_runtime_stage': self.runtime_stage,
            'model_requests': self.requests,
            'pending_requests': self.pending,
            'consecutive_failures': self.failures,
            'consecutive_rate_limits': self.rate_limits,
            'idle_seconds': time.monotonic() - self.last_progress,
            'events': list(self.events),
        }
