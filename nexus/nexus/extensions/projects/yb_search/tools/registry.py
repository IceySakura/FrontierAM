"""Lightweight registry for :class:`YbTool` subclasses.

Lookup key is ``(name, version)``; each key maps to a singleton instance of
the tool class (tool objects are stateless).
"""

from __future__ import annotations

import logging

from typing import TypeVar

from nexus.extensions.projects.yb_search.tools.base import YbTool


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=type[YbTool])


def _key(name: str, version: str) -> str:
    return f'{name}@{version}'


class YbToolRegistry:
    """In-memory mapping of ``(name, version) -> YbTool``."""

    def __init__(self) -> None:
        self._tools: dict[str, YbTool] = {}

    def register(self, tool_cls: T) -> T:
        """Register a tool class. Returns the class for decorator usage."""
        if not issubclass(tool_cls, YbTool):
            raise TypeError(f'{tool_cls!r} is not a subclass of YbTool')

        name = getattr(tool_cls, 'name', None)
        version = getattr(tool_cls, 'version', None)
        if not name or not version:
            raise ValueError(f'{tool_cls.__name__} must declare class-level name and version')

        key = _key(name, version)
        if key in self._tools:
            logger.warning(
                'duplicate registration for %s; overriding %s -> %s',
                key,
                type(self._tools[key]).__name__,
                tool_cls.__name__,
            )
        self._tools[key] = tool_cls()  # instantiate once
        return tool_cls

    def get(self, name: str, version: str) -> YbTool | None:
        return self._tools.get(_key(name, version))

    def all_tools(self) -> list[YbTool]:
        return list(self._tools.values())


# Module-level singleton, imported by the agent at runtime.
yb_tool_registry = YbToolRegistry()
