from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_current_context: ContextVar[dict[str, Any] | None] = ContextVar('distributed_execution_context', default=None)


def current_distributed_execution_context() -> dict[str, Any]:
    context = _current_context.get()
    if context is None:
        return {}
    return dict(context)


@contextmanager
def distributed_execution_context(**fields: Any):
    current = current_distributed_execution_context()
    update = {key: value for key, value in fields.items() if value is not None}
    token = _current_context.set({**current, **update})
    try:
        yield
    finally:
        _current_context.reset(token)
