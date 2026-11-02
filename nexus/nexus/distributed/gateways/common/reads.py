"""Cancel expensive optional reads when their HTTP caller disconnects."""

from __future__ import annotations

import asyncio


async def read_while_connected(request, read):
    task = asyncio.create_task(read)
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.1)
            if not task.done() and (request.transport is None or request.transport.is_closing()):
                raise asyncio.CancelledError
        return await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
