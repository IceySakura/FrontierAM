"""Decorator-based registries for OSWorld evaluator getters and metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar


_F = TypeVar('_F', bound=Callable[..., Any])


class _Registry:
    """A minimal name-to-callable registry with a decorator interface."""

    def __init__(self, name: str, *, strip_prefix: str = '') -> None:
        self.name = name
        self._strip_prefix = strip_prefix
        self._entries: dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __getitem__(self, key: str) -> Callable[..., Any]:
        return self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def get(self, key: str) -> Callable[..., Any] | None:
        return self._entries.get(key)

    def items(self):
        return self._entries.items()

    def set(self, name: str, fn: _F) -> _F:
        """Register or replace a callable under an explicit name."""
        self._entries[name] = fn
        return fn

    def register(self, name: str | None = None) -> Callable[[_F], _F]:
        """Register a callable under an explicit or derived name."""

        def decorator(fn: _F) -> _F:
            key = name if name is not None else fn.__name__
            if self._strip_prefix and key.startswith(self._strip_prefix):
                key = key[len(self._strip_prefix) :]
            if key in self._entries:
                return fn
            self._entries[key] = fn
            return fn

        return decorator

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f'<{self.name} registry: {len(self._entries)} entries>'


getter_registry = _Registry('getter', strip_prefix='get_')
metric_registry = _Registry('metric')
