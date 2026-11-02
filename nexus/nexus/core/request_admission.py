"""Lightweight client request admission protocol and pacing primitive.

This module intentionally does not import ``nexus.distributed`` or any ORM models so SDK clients can
depend on it in both local and cloud execution modes.
"""

from __future__ import annotations

import math

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic
from typing import Protocol


class RequestPermitExpiredError(RuntimeError):
    """Raised when a permit's local lease deadline has already elapsed before send."""


@dataclass(frozen=True, slots=True)
class RequestAttempt:
    """Identity of one actual request attempt (initial send or retry)."""

    execution_id: str
    attempt_id: str
    client_id: str


@dataclass(frozen=True, slots=True)
class RequestPermit:
    """One-time send permit bound to a request attempt and Manager session."""

    execution_id: str
    attempt_id: str
    session_id: str
    lease_id: str | None = None
    deadline_at: float | None = None

    def is_valid(self, *, now: float | None = None) -> bool:
        if self.deadline_at is None:
            return True
        current = monotonic() if now is None else now
        return current < self.deadline_at

    def ensure_valid(self, *, now: float | None = None) -> None:
        if not self.is_valid(now=now):
            raise RequestPermitExpiredError('request_permit_expired')


class RequestAdmission(Protocol):
    """Protocol implemented by Manager RequestBudgetControl and local rate adapters."""

    async def acquire(self, request: RequestAttempt) -> RequestPermit: ...


@dataclass(frozen=True, slots=True)
class BoundRequestAdmission:
    """Trusted execution-scoped binding injected by the Driver."""

    admission: RequestAdmission
    execution_id: str
    attempt_id: str

    async def acquire(self, *, client_id: str) -> RequestPermit:
        return await self.admission.acquire(
            RequestAttempt(
                execution_id=self.execution_id,
                attempt_id=self.attempt_id,
                client_id=client_id,
            )
        )


_bound_request_admission: ContextVar[BoundRequestAdmission | None] = ContextVar(
    'nexus_bound_request_admission', default=None
)


def current_request_admission() -> BoundRequestAdmission | None:
    return _bound_request_admission.get()


@contextmanager
def bind_request_admission(
    admission: RequestAdmission | None,
    *,
    execution_id: str,
    attempt_id: str,
    cancel_on_exit: bool = True,
    on_exit: Callable[[], None] | None = None,
) -> Iterator[BoundRequestAdmission | None]:
    """Bind the trusted execution identity and admission controller for the running task."""
    if admission is None:
        yield None
        return
    clear_fn = getattr(admission, 'clear_execution_cancellation', None)
    if callable(clear_fn):
        clear_fn(execution_id, attempt_id=attempt_id)
    bound = BoundRequestAdmission(
        admission=admission,
        execution_id=execution_id,
        attempt_id=attempt_id,
    )
    token = _bound_request_admission.set(bound)
    try:
        yield bound
    finally:
        _bound_request_admission.reset(token)
        if on_exit is not None:
            on_exit()
        elif cancel_on_exit:
            cancel_fn = getattr(admission, 'cancel_execution', None)
            if callable(cancel_fn):
                cancel_fn(execution_id, attempt_id=attempt_id)


class SmoothRatePacer:
    """Smooth QPM pacer with zero initial free permits and at most one request of idle burst.

    Preserving ``last_issued_at`` across continuous renewals of the same lease allows low-QPM grants
    (for example < 6 QPM across 10-second leases) to accumulate progress without starvation, while
    resetting ``last_issued_at`` on a new lease or after an expired gap prevents back-calculating
    permits for an unpermitted window.
    """

    __slots__ = ('_last_issued_at',)

    def __init__(self, *, now: float | None = None) -> None:
        self._last_issued_at: float | None = now

    @property
    def last_issued_at(self) -> float | None:
        return self._last_issued_at

    def reset(self, now: float) -> None:
        """Start pacing from ``now`` with zero free permits and no carried-over burst."""
        self._last_issued_at = now

    def try_acquire(self, *, qpm: float, now: float) -> float:
        """Return 0.0 and advance pacing if a slot is due at ``now``, else the positive wait seconds."""
        if isinstance(qpm, bool) or not math.isfinite(qpm) or qpm <= 0:
            return math.inf
        if self._last_issued_at is None:
            self._last_issued_at = now
        delay = self._last_issued_at + 60.0 / qpm - now
        if delay <= 0:
            self._last_issued_at = now
            return 0.0
        return delay


__all__ = [
    'BoundRequestAdmission',
    'RequestAdmission',
    'RequestAttempt',
    'RequestPermit',
    'RequestPermitExpiredError',
    'SmoothRatePacer',
    'bind_request_admission',
    'current_request_admission',
]
