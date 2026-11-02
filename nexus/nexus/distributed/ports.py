from __future__ import annotations

from typing import Protocol


class AsyncCloseable(Protocol):
    async def close(self) -> None: ...


class ReconcileController(AsyncCloseable, Protocol):
    async def reconcile_once(self) -> int: ...


__all__ = ['AsyncCloseable', 'ReconcileController']
