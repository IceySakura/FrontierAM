from __future__ import annotations

from typing import Any, Callable

import pytest

from nexus.core.context import ExecutionContext


class _NoEventRuntime:
    """Runtime stub whose event API is unsupported (like LocalRuntime).

    ``on``/``emit`` exist as attributes (so ``hasattr`` is True), but ``on``
    raises ``NotImplementedError`` exactly as the base ``Runtime`` default does.
    """

    def on(self, event: str, listener: Callable[..., Any]) -> Callable[[], None]:
        raise NotImplementedError('runtime does not support event subscription')

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - not called
        raise NotImplementedError


def test_attach_runtime_shared_skips_when_events_unsupported() -> None:
    context = ExecutionContext()
    runtime = _NoEventRuntime()

    # Should not raise, and should not register any unsub handles.
    context._attach_runtime_shared(runtime)  # type: ignore[arg-type]

    assert id(runtime) not in context._runtime_shared_unsubs


def test_attach_task_files_runtime_skips_when_events_unsupported() -> None:
    context = ExecutionContext()
    context.task_files = object()  # type: ignore[assignment]
    runtime = _NoEventRuntime()

    # Should not raise, and should not register any unsub handles.
    context._attach_task_files_runtime(runtime)  # type: ignore[arg-type]

    assert id(runtime) not in context._task_files_runtime_unsubs


def test_attach_runtime_shared_registers_when_events_supported() -> None:
    unsub_calls: list[str] = []

    class _EventRuntime:
        def on(self, event: str, listener: Callable[..., Any]) -> Callable[[], None]:
            unsub_calls.append(event)
            return lambda: None

        def emit(self, event: str, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            pass

    context = ExecutionContext()
    runtime = _EventRuntime()

    context._attach_runtime_shared(runtime)  # type: ignore[arg-type]

    assert id(runtime) in context._runtime_shared_unsubs
    assert unsub_calls == ['shared-value:request']
