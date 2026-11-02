"""Loopback Streamable HTTP MCP server that exposes Host-declared tools to one session.

The Host owns tool names, descriptions, and input schemas; this server only validates
arguments against those schemas and forwards each call through a Host RPC callable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import socket

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from nexus.extensions.agents.installed_common.host_rpc import HostToolError


logger = logging.getLogger(__name__)

HostToolCall = Callable[..., Awaitable[Any]]
_STARTUP_TIMEOUT = 5.0
_SHUTDOWN_TIMEOUT = 5.0
_LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '[::1]')


def _positive_timeout(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number of seconds, got {value!r}')
    return float(value)


class HostToolMCPServer:
    """Serve Host-declared MCP tools on a loopback port for the lifetime of one session.

    Args:
        server_name: MCP server name advertised to the agent.
        tools: MCP ``Tool`` objects as dicts (``name``, ``description``, ``inputSchema``, ...).
        call_host: Coroutine taking keyword-only ``tool_name``, ``arguments``, ``timeout``.
        request_timeout: Default per-call Host budget in seconds.
        tool_timeouts: Per-tool budgets that override ``request_timeout``.
        listen_host: Loopback address to bind.
    """

    def __init__(
        self,
        *,
        server_name: str,
        tools: Sequence[dict[str, Any]],
        call_host: HostToolCall,
        request_timeout: float = 60.0,
        tool_timeouts: dict[str, float] | None = None,
        listen_host: str = '127.0.0.1',
    ) -> None:
        from mcp import types
        from pydantic import ValidationError

        if not isinstance(server_name, str) or not server_name:
            raise ValueError('server_name must be a non-empty string')
        if not tools:
            raise ValueError('HostToolMCPServer needs at least one tool')
        self._tools: dict[str, types.Tool] = {}
        for spec in tools:
            try:
                tool = types.Tool.model_validate(spec)
            except ValidationError as error:
                raise ValueError(f'Invalid host tool specification {spec!r}: {error}') from error
            if tool.outputSchema is not None:
                raise ValueError(f'Host tool {tool.name!r} must not declare outputSchema; results are text content')
            if tool.name in self._tools:
                raise ValueError(f'Duplicate host tool name {tool.name!r}')
            self._tools[tool.name] = tool
        self._request_timeout = _positive_timeout(request_timeout, name='request_timeout')
        unknown = set(tool_timeouts or {}) - set(self._tools)
        if unknown:
            raise ValueError(f'tool_timeouts names unknown tools: {sorted(unknown)}')
        self._tool_timeouts = {
            name: _positive_timeout(value, name=f'tool_timeouts[{name!r}]')
            for name, value in (tool_timeouts or {}).items()
        }
        self.server_name = server_name
        self._call_host = call_host
        self._listen_host = listen_host
        self._started = False
        self._server: Any = None
        self._server_task: asyncio.Task[None] | None = None
        self._listener: socket.socket | None = None

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        from mcp import types

        if name not in self._tools:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type='text', text=f'Unknown tool {name!r}')]
            )
        try:
            result = await self._call_host(
                tool_name=name, arguments=arguments, timeout=self._tool_timeouts.get(name, self._request_timeout)
            )
        except HostToolError as error:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type='text', text=f'{error.code}: {error}')]
            )
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, indent=2)
        return [types.TextContent(type='text', text=text)]

    def _create_app(self) -> Any:
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from mcp.server.transport_security import TransportSecuritySettings
        from starlette.applications import Starlette
        from starlette.routing import Route

        mcp_server: Server[Any, Any] = Server(self.server_name)

        async def list_tools() -> list[Any]:
            return list(self._tools.values())

        mcp_server.list_tools()(list_tools)
        mcp_server.call_tool()(self._call_tool)
        session_manager = StreamableHTTPSessionManager(
            app=mcp_server,
            json_response=True,
            stateless=True,
            security_settings=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=[f'{host}:*' for host in _LOOPBACK_HOSTS],
                allowed_origins=[f'http://{host}:*' for host in _LOOPBACK_HOSTS],
            ),
        )

        class _StreamableHTTPEndpoint:
            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                await session_manager.handle_request(scope, receive, send)

        return Starlette(
            routes=[Route('/mcp', endpoint=_StreamableHTTPEndpoint())],
            lifespan=lambda _app: session_manager.run(),
        )

    async def start(self) -> dict[str, Any]:
        """Start serving and return the ACP ``session/new`` MCP server specification."""
        import uvicorn

        if self._started:
            raise RuntimeError('HostToolMCPServer can only be started once')
        self._started = True
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((self._listen_host, 0))
        self._listener = listener
        port = listener.getsockname()[1]
        # NOTE(wallyshi): The ACP peer holds this server only through its URL, so it serves from its own task.
        self._server = uvicorn.Server(uvicorn.Config(self._create_app(), log_level='warning', lifespan='on'))
        self._server_task = asyncio.create_task(
            self._server.serve(sockets=[listener]), name=f'host-tool-mcp:{self.server_name}'
        )
        try:
            deadline = asyncio.get_running_loop().time() + _STARTUP_TIMEOUT
            while not self._server.started:
                if self._server_task.done():
                    await self._server_task
                    raise RuntimeError(f'Host tool MCP server {self.server_name!r} exited during startup')
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f'Host tool MCP server {self.server_name!r} did not start')
                await asyncio.sleep(0.01)
        except BaseException:
            try:
                await self.close()
            except Exception:
                logger.exception('Failed to stop host tool MCP server %s after startup failure', self.server_name)
            raise
        return {
            'type': 'http',
            'name': self.server_name,
            'url': f'http://{self._listen_host}:{port}/mcp',
            'headers': [],
        }

    async def close(self) -> None:
        """Stop serving; repeated calls are no-ops."""
        server, server_task, listener = self._server, self._server_task, self._listener
        self._server = self._server_task = self._listener = None
        try:
            if server is not None:
                server.should_exit = True
            if server_task is not None and not server_task.done():
                done, _ = await asyncio.wait((server_task,), timeout=_SHUTDOWN_TIMEOUT)
                if not done:
                    logger.warning('Host tool MCP server %s ignored graceful shutdown; cancelling', self.server_name)
                    server_task.cancel()
                    await asyncio.gather(server_task, return_exceptions=True)
            if server_task is not None and server_task.done() and not server_task.cancelled():
                error = server_task.exception()
                if error is not None:
                    raise error
        finally:
            if listener is not None:
                listener.close()


__all__ = ['HostToolCall', 'HostToolMCPServer']
