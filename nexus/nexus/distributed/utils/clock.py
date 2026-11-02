from __future__ import annotations

import time

from datetime import datetime, timedelta, timezone


DEFAULT_STOP_TIMEOUT_SECONDS = 60


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def deadline_after(started_at: datetime | None, timeout_seconds: int | None) -> datetime | None:
    if started_at is None or timeout_seconds is None:
        return None
    return started_at + timedelta(seconds=timeout_seconds)


def monotonic_deadline_from_iso8601(value: str) -> float:
    deadline_at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if deadline_at.tzinfo is not None:
        deadline_at = deadline_at.astimezone(timezone.utc).replace(tzinfo=None)
    return time.monotonic() + max(0.0, (deadline_at - utcnow()).total_seconds())


def stop_deadline_after(started_at: datetime, timeout_seconds: int | None) -> datetime:
    timeout = DEFAULT_STOP_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    return started_at + timedelta(seconds=timeout)
