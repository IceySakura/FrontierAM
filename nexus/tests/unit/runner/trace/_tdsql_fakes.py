"""Shared async database stand-ins for the TDSQL trace tests.

Underscore-prefixed so pytest does not collect it. Every scriptable connection in these
tests needs the same async context-manager and cursor plumbing while scripting different
answers; only the scripting belongs to an individual test.
"""

from __future__ import annotations

from typing import Any


class AsyncCursorMixin:
    """``async with`` support for a stand-in that is its own cursor."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class ConnectionMixin:
    """The connection side of the protocol, for a stand-in that serves both roles."""

    def cursor(self, *_args: Any) -> Any:
        return self

    def close(self) -> None:
        return None


def install_connections(monkeypatch, *connections: Any) -> None:
    """Hand out ``connections`` in order, one per ``connect()`` call."""
    import nexus.runner.trace.tdsql._driver as driver

    pending = list(connections)

    async def connect(**_kwargs: Any) -> Any:
        return pending.pop(0)

    monkeypatch.setattr(driver, 'connect', connect)


__all__ = ['AsyncCursorMixin', 'ConnectionMixin', 'install_connections']
