from __future__ import annotations

import functools
import logging

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import socketio

from fastapi import FastAPI

from nexus.runtime.server.services.anthropic_proxy import AnthropicProxy, LLMBridge
from nexus.runtime.server.services.responses_proxy import ResponsesProxy


if TYPE_CHECKING:
    # Real type for the shim below; imported only for type checking so the native
    # bridge bundle (which does not vendor nexus.runtime.context) stays importable.
    from nexus.runtime.context import SIOBridge


logger = logging.getLogger(__name__)


class _ServerSio:
    """Minimal ``SIOBridge``-shaped adapter over a raw ``AsyncServer``.

    ``LLMBridge`` expects sid-stripped ``on(event)`` handlers and a broadcast
    ``emit``. This mirrors ``nexus.runtime.context.SIOBridge`` server semantics
    but is dependency-free, so the native bridge bundle (which vendors only
    ``nexus/runtime/bridge`` + ``anthropic_proxy``) needn't ship that module.
    """

    def __init__(self, sio: socketio.AsyncServer) -> None:
        self._sio = sio

    def on(self, event: str) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        def decorator(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
            @functools.wraps(fn)
            async def wrapper(sid: str, *args: Any) -> Any:  # noqa: ARG001 - drop server-side sid
                return await fn(*args)

            self._sio.on(event, wrapper)
            return fn

        return decorator

    async def emit(self, event: str, data: Any = None) -> None:
        await self._sio.emit(event, data)


def create_bridge_fastapi_app(
    *,
    api_protocol: str = 'anthropic',
    request_timeout: float = 300,
    conversion_log_level: str = 'INFO',
    thinking_config: str | dict[str, Any] | None = 'hunyuan',
    launch_id: str = '',
    responses_converter: Callable[..., dict[str, Any]] | None = None,
) -> tuple[FastAPI, LLMBridge]:
    sio = socketio.AsyncServer(
        async_mode='asgi',
        cors_allowed_origins='*',
        logger=False,
        engineio_logger=False,
        ping_interval=15.0,
        ping_timeout=120.0,
        # Large LLM responses flow host -> bridge; match the client's 200 MiB frame limit
        # so a big completion isn't dropped at the default 1 MiB and killed as a transport error.
        max_http_buffer_size=200 * 1024 * 1024,
    )
    connected_hosts: set[str] = set()

    @sio.on('connect')
    async def on_connect(sid: str, environ: dict | None = None, auth: dict | None = None) -> None:
        connected_hosts.add(str(sid))
        logger.debug('LLM bridge host connected: sid=%s hosts=%d', sid, len(connected_hosts))

    @sio.on('disconnect')
    async def on_disconnect(sid: str) -> None:
        connected_hosts.discard(str(sid))
        logger.info('LLM bridge host disconnected: sid=%s hosts=%d', sid, len(connected_hosts))

    # _ServerSio is a dependency-free SIOBridge stand-in; the proxies only use on()/emit().
    bridge_sio = cast('SIOBridge', _ServerSio(sio))
    if api_protocol == 'anthropic':
        proxy = AnthropicProxy(
            sio=bridge_sio,
            request_timeout=request_timeout,
            conversion_log_level=conversion_log_level,
            thinking_config=thinking_config,
        )
    elif api_protocol == 'responses':
        proxy_options: dict[str, Any] = {}
        if responses_converter is not None:
            proxy_options['responses_to_openai_converter'] = responses_converter
        proxy = ResponsesProxy(
            sio=bridge_sio,
            request_timeout=request_timeout,
            **proxy_options,
        )
    else:
        raise ValueError(f'Unsupported bridge API protocol: {api_protocol}')
    app = proxy.create_app()
    bridge = proxy.bridge
    assert bridge is not None

    @app.get('/health')
    async def health() -> dict[str, str | int]:
        return {
            'status': 'ok',
            'bridge': 'nexus-llm-bridge',
            'launch_id': launch_id,
            'connected_hosts': len(connected_hosts),
            # RSIOLLMBridge tracks in-flight calls on ._pending; diagnostic only.
            'pending_requests': len(getattr(bridge, '_pending', {})),
        }

    app.state.sio = sio
    app.state.bridge = bridge
    return app, bridge


def create_bridge_asgi_app(
    *,
    api_protocol: str = 'anthropic',
    request_timeout: float = 300,
    conversion_log_level: str = 'INFO',
    thinking_config: str | dict[str, Any] | None = 'hunyuan',
    launch_id: str = '',
    responses_converter: Callable[..., dict[str, Any]] | None = None,
) -> socketio.ASGIApp:
    fastapi_app, _bridge = create_bridge_fastapi_app(
        api_protocol=api_protocol,
        request_timeout=request_timeout,
        conversion_log_level=conversion_log_level,
        thinking_config=thinking_config,
        launch_id=launch_id,
        responses_converter=responses_converter,
    )
    return socketio.ASGIApp(fastapi_app.state.sio, other_asgi_app=fastapi_app)
