from __future__ import annotations

import asyncio
import contextvars

from collections.abc import Coroutine
from typing import Any, TypeVar


_T = TypeVar('_T')


def get_current_task_cancellation_count() -> int | None:
    """Return the current task cancellation count when the runtime exposes it."""
    task = asyncio.current_task()
    cancelling = getattr(task, 'cancelling', None)
    return cancelling() if callable(cancelling) else None


def get_current_task_name() -> str | None:
    """Return the current task name when the runtime exposes it."""
    task = asyncio.current_task()
    get_name = getattr(task, 'get_name', None)
    return get_name() if callable(get_name) else None


def create_detached_task(coro: Coroutine[Any, Any, _T], *, name: str | None = None) -> asyncio.Task[_T]:
    """Create a long-lived infrastructure task without inheriting request context."""

    def create_task() -> asyncio.Task[_T]:
        return asyncio.create_task(coro, name=name)

    return contextvars.Context().run(create_task)
