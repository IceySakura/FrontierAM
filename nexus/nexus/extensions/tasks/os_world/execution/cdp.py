"""Provider-neutral helpers for connecting to the task Chrome DevTools endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


async def chromium_cdp_connection(
    runtime: Any,
    *,
    ensure_target: bool = True,
) -> tuple[str, dict[str, str] | None] | None:
    # Import at use time: getters register Chrome readers that themselves use
    # this CDP module. Module-level import makes setup depend on import order.
    from nexus.extensions.tasks.os_world.getters.runtime import runtime_headers, runtime_port_url

    connection_fn = getattr(runtime, 'get_chromium_cdp_connection', None)
    if callable(connection_fn):
        try:
            endpoint, headers = await connection_fn(ensure_target=ensure_target)
        except TypeError:
            endpoint, headers = await connection_fn()
        return str(endpoint).rstrip('/'), dict(headers) if headers else None

    value = getattr(runtime, 'chromium_url', None) or runtime_port_url(
        runtime, int(getattr(runtime, 'chromium_port', 9222))
    )
    return (str(value).rstrip('/'), runtime_headers(runtime)) if value else None


@asynccontextmanager
async def connect_chromium(runtime: Any) -> AsyncIterator[Any]:
    """Connect to the task Chrome via CDP.

    Usage:
        async with connect_chromium(runtime) as browser:
            page = await browser.new_page()
            ...
    """
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.connect_over_cdp(
            await runtime.chromium_cdp_url(), headers=runtime.handle.headers, timeout=15000
        )
    except Exception:
        await pw.stop()
        raise
    try:
        yield browser
    finally:
        await browser.close()
        await pw.stop()
