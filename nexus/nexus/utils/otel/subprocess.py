from __future__ import annotations

import asyncio
import os
import subprocess as _subprocess
import threading

from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from nexus.utils.otel.auto import is_enabled
from nexus.utils.otel.environment import is_managed_otel_env_key
from nexus.utils.otel.propagation import inject_carrier


_POPEN_ENV_POSITION = 10
_preserve_managed_environment: ContextVar[bool] = ContextVar(
    'nexus_otel_preserve_managed_subprocess_environment',
    default=False,
)


@contextmanager
def preserve_managed_otel_environment() -> Iterator[None]:
    token = _preserve_managed_environment.set(True)
    try:
        yield
    finally:
        _preserve_managed_environment.reset(token)


def inject_current_context_env(env: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(os.environ if env is None else env)
    result.pop('TRACEPARENT', None)
    result.pop('TRACESTATE', None)
    if not is_enabled():
        return result
    if not _preserve_managed_environment.get():
        result = {key: value for key, value in result.items() if not is_managed_otel_env_key(key)}

    carrier = inject_carrier()
    traceparent = carrier.get('traceparent')
    tracestate = carrier.get('tracestate')
    if traceparent:
        result['TRACEPARENT'] = traceparent
    if tracestate:
        result['TRACESTATE'] = tracestate
    return result


class NexusSubprocessInstrumentor(BaseInstrumentor):
    _state_lock = threading.RLock()

    def __init__(self) -> None:
        with self._state_lock:
            if not hasattr(self, '_original_popen_init'):
                self._original_popen_init: Any | None = None
                self._original_create_exec: Any | None = None
                self._original_create_shell: Any | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return ()

    def instrument(self, **kwargs: Any) -> Any:
        if not is_enabled():
            return None
        with self._state_lock:
            return super().instrument(**kwargs)

    def uninstrument(self, **kwargs: Any) -> Any:
        with self._state_lock:
            return super().uninstrument(**kwargs)

    def _instrument(self, **kwargs: Any) -> None:
        del kwargs
        original_popen_init = _subprocess.Popen.__init__
        original_create_exec = asyncio.create_subprocess_exec
        original_create_shell = asyncio.create_subprocess_shell
        self._original_popen_init = original_popen_init
        self._original_create_exec = original_create_exec
        self._original_create_shell = original_create_shell

        def popen_init(instance: Any, *args: Any, **call_kwargs: Any) -> None:
            try:
                if len(args) > _POPEN_ENV_POSITION:
                    updated_args = list(args)
                    updated_args[_POPEN_ENV_POSITION] = inject_current_context_env(updated_args[_POPEN_ENV_POSITION])
                    updated_kwargs = call_kwargs
                else:
                    updated_args = args
                    updated_kwargs = dict(call_kwargs)
                    updated_kwargs['env'] = inject_current_context_env(updated_kwargs.get('env'))
            except Exception:
                updated_args = args
                updated_kwargs = call_kwargs
            original_popen_init(instance, *updated_args, **updated_kwargs)

        async def create_exec(program: str, *args: str, **call_kwargs: Any) -> Any:
            try:
                updated_kwargs = dict(call_kwargs)
                updated_kwargs['env'] = inject_current_context_env(updated_kwargs.get('env'))
            except Exception:
                return await original_create_exec(program, *args, **call_kwargs)
            return await original_create_exec(program, *args, **updated_kwargs)

        async def create_shell(cmd: str, *args: Any, **call_kwargs: Any) -> Any:
            try:
                updated_kwargs = dict(call_kwargs)
                updated_kwargs['env'] = inject_current_context_env(updated_kwargs.get('env'))
            except Exception:
                return await original_create_shell(cmd, *args, **call_kwargs)
            return await original_create_shell(cmd, *args, **updated_kwargs)

        _subprocess.Popen.__init__ = popen_init  # ty: ignore[invalid-assignment]
        asyncio.create_subprocess_exec = create_exec  # ty: ignore[invalid-assignment]
        asyncio.create_subprocess_shell = create_shell  # ty: ignore[invalid-assignment]

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        if self._original_popen_init is not None:
            _subprocess.Popen.__init__ = self._original_popen_init
        if self._original_create_exec is not None:
            asyncio.create_subprocess_exec = self._original_create_exec
        if self._original_create_shell is not None:
            asyncio.create_subprocess_shell = self._original_create_shell
        self._original_popen_init = None
        self._original_create_exec = None
        self._original_create_shell = None
