from __future__ import annotations

import asyncio

from concurrent.futures import Future
from typing import TypeVar


_T = TypeVar('_T')


async def await_threadsafe(future: Future[_T], *, timeout: float) -> _T:
    """Await a cross-thread future without allowing an unbounded bridge."""
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout=max(0.0, timeout))
    except (TimeoutError, asyncio.CancelledError):
        future.cancel()
        raise


__all__ = ['await_threadsafe']
