"""Managed stdio MCP fixture for the unified OTel trace E2E."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import mcp.server.stdio

from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

import nexus  # noqa: F401 - initializes the configured OTel pipeline

from nexus.utils.otel.lifecycle import force_flush, shutdown
from nexus.utils.otel.mcp import mcp_server_span
from nexus.utils.otel.subprocess import preserve_managed_otel_environment


SERVER_NAME = 'otel-e2e'
TOOL_NAME = 'spawn_child'
SUBPROCESS_NAME = 'managed.subprocess'

server = Server(SERVER_NAME)

_CHILD_SCRIPT = f"""
import os

import nexus

from nexus.utils.otel.lifecycle import force_flush, shutdown
from nexus.utils.otel.propagation import use_remote_context
from nexus.utils.otel.transport import transport_span

carrier = {{key.lower(): os.environ[key] for key in ('TRACEPARENT', 'TRACESTATE') if key in os.environ}}
with use_remote_context(carrier, fallback_only=False):
    with transport_span({SUBPROCESS_NAME!r}, kind='server'):
        pass
assert force_flush(10_000)
shutdown()
"""


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name=TOOL_NAME,
            description='Spawn one managed traced subprocess.',
            inputSchema={'type': 'object', 'properties': {}},
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    del arguments
    if name != TOOL_NAME:
        raise ValueError(f'unknown tool: {name}')

    with mcp_server_span(server.request_context.meta, SERVER_NAME, TOOL_NAME):
        with preserve_managed_otel_environment():
            completed = subprocess.run(
                [sys.executable, '-c', _CHILD_SCRIPT],
                env=dict(os.environ),
                capture_output=True,
                check=False,
                text=True,
                timeout=30,
            )
        if completed.returncode != 0:
            raise RuntimeError(f'managed child failed: {completed.stderr}')

    if not force_flush(10_000):
        raise RuntimeError('MCP server OTel flush timed out')
    return [TextContent(type='text', text='OK')]


async def main() -> None:
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        force_flush(10_000)
        shutdown()


if __name__ == '__main__':
    asyncio.run(main())
