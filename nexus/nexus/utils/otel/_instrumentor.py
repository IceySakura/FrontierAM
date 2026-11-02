from __future__ import annotations

import threading

from abc import abstractmethod
from collections.abc import Callable, Collection
from types import ModuleType
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt.importer import register_post_import_hook

from nexus.utils.otel.auto import is_enabled


class NexusModuleInstrumentor(BaseInstrumentor):
    """Patch Nexus modules after import, without pulling optional dependencies into startup."""

    _state_lock = threading.RLock()
    _module_name: str

    def __init__(self) -> None:
        with self._state_lock:
            if not hasattr(self, '_patches'):
                self._patches: list[tuple[Any, str, Any, Any]] = []
                self._generation: object | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return ()

    def instrument(self, **kwargs: Any) -> Any:
        if not is_enabled():
            return None
        with self._state_lock:
            if self.is_instrumented_by_opentelemetry:
                return None
            return super().instrument(**kwargs)

    def uninstrument(self, **kwargs: Any) -> Any:
        with self._state_lock:
            if not self.is_instrumented_by_opentelemetry:
                return None
            return super().uninstrument(**kwargs)

    def _instrument(self, **kwargs: Any) -> None:
        generation = self._generation = object()

        def patch(module: ModuleType) -> None:
            with self._state_lock:
                # Pending import callbacks from a previous installation must remain inert.
                if self._generation is generation:
                    self._patch_module(module)

        register_post_import_hook(patch, self._module_name)

    def _wrap(self, target: Any, name: str, decorate: Callable[[Any], Any]) -> None:
        original = getattr(target, name)
        wrapped = decorate(original)
        setattr(target, name, wrapped)
        self._patches.append((target, name, original, wrapped))

    @abstractmethod
    def _patch_module(self, module: ModuleType) -> None:
        raise NotImplementedError

    def _uninstrument(self, **kwargs: Any) -> None:
        self._generation = None
        for target, name, original, wrapped in reversed(self._patches):
            if getattr(target, name) is wrapped:
                setattr(target, name, original)
        self._patches.clear()
