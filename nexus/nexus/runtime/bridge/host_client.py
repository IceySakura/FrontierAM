from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import time

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

import aiohttp
import socketio

from yarl import URL


if TYPE_CHECKING:
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler


logger = logging.getLogger(__name__)

# Match the bridge server's websocket frame limit; aiohttp's client default is only 4 MiB.
_WS_MAX_MSG_SIZE = 200 * 1024 * 1024


class _HTTPProxyWebSocketRequest(aiohttp.ClientRequest):
    """Send the WebSocket upgrade in HTTP form understood by forward proxies."""

    def __init__(self, method: str, url: URL, **kwargs: Any) -> None:
        # aiohttp otherwise writes `GET ws://host/... HTTP/1.1` to an HTTP
        # proxy. Some proxies reject that scheme before forwarding the upgrade.
        # Keep Upgrade headers and TLS semantics; change only the URI scheme.
        if kwargs.get('proxy') is not None and url.scheme in {'ws', 'wss'}:
            url = url.with_scheme('https' if url.scheme == 'wss' else 'http')
        super().__init__(method, url, **kwargs)


class BridgeHostClient:
    """Socket.IO client shim used by host-side LLMRequestHandler."""

    def __init__(
        self,
        *,
        base_url: str,
        headers: dict[str, str] | None = None,
        request_timeout: float = 60.0,
        reconnection: bool = True,
        keepalive_interval: float | None = 25.0,
        proxy_url: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
        self.proxy_url = proxy_url
        self._proxy_session: aiohttp.ClientSession | None = None
        self.request_timeout = request_timeout
        self.keepalive_interval = keepalive_interval
        self._ws_connected = False
        self._event_listeners: dict[str, list[Callable[..., Awaitable[None]]]] = {}
        self._keepalive_task: asyncio.Task[None] | None = None
        self._sio = socketio.AsyncClient(
            reconnection=reconnection,
            reconnection_attempts=60,
            reconnection_delay=1.0,
            reconnection_delay_max=10.0,
            request_timeout=request_timeout,
            logger=False,
            engineio_logger=False,
            # aiohttp's ws_connect defaults to a 4 MiB max_msg_size and tears the
            # connection down (as a transport error → reconnect loop) on any single
            # frame past that. The bridge server accepts 200 MiB, so match it here
            # or large context requests from the sandbox would trigger the reconnect loop.
            websocket_extra_options={
                'max_msg_size': _WS_MAX_MSG_SIZE,
                **({'proxy': proxy_url} if proxy_url else {}),
            },
        )

        @self._sio.event
        async def connect() -> None:
            self._ws_connected = True
            self._start_keepalive()
            logger.debug('LLM bridge host connected: %s', self.base_url)

        @self._sio.event
        async def disconnect() -> None:
            self._ws_connected = False
            await self._stop_keepalive()
            logger.info('LLM bridge host disconnected: %s', self.base_url)

    def _start_keepalive(self) -> None:
        if self.keepalive_interval is None or self.keepalive_interval <= 0:
            return
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _stop_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _keepalive_loop(self) -> None:
        try:
            while self._ws_connected and self._sio.connected:
                await asyncio.sleep(self.keepalive_interval or 0)
                if not self._ws_connected or not self._sio.connected:
                    break
                await self._sio.emit('bridge_keepalive', {'time': time.time()})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug('LLM bridge keepalive stopped after emit failure', exc_info=True)

    async def ws_connect(self) -> None:
        # Explicit forward proxies need WebSocket from the first handshake (no polling).
        if self.proxy_url and (self._proxy_session is None or self._proxy_session.closed):
            self._proxy_session = aiohttp.ClientSession(request_class=_HTTPProxyWebSocketRequest)
            self._sio.eio.http = self._proxy_session
            self._sio.eio.external_http = True
        try:
            await self._sio.connect(
                self.base_url,
                headers=self.headers or None,
                socketio_path=urlsplit(self.base_url).path.rstrip('/') + '/socket.io',
                **({'transports': ['websocket']} if self.proxy_url else {}),
            )
        except BaseException:
            if self.proxy_url:
                await self.ws_disconnect()
            raise
        self._ws_connected = True
        self._start_keepalive()

    async def ws_disconnect(self) -> None:
        try:
            if self._sio.connected or self._proxy_session is not None:
                await self._sio.disconnect()
        finally:
            self._ws_connected = False
            await self._stop_keepalive()
            if self._proxy_session is not None:
                await self._proxy_session.close()

    @property
    def ws_connected(self) -> bool:
        return self._ws_connected

    def on(self, event: str, listener: Callable[..., None] | Callable[..., Awaitable[None]]) -> Callable[[], None]:
        if not asyncio.iscoroutinefunction(listener) and not inspect.iscoroutinefunction(listener):
            original_listener = listener

            async def wrapper(*args: Any, **kwargs: Any) -> None:
                original_listener(*args, **kwargs)

        else:
            wrapper = cast(Callable[..., Awaitable[None]], listener)

        self._event_listeners.setdefault(event, []).append(wrapper)
        self._sio.on(event, self._make_event_dispatcher(event))

        def unsubscribe() -> None:
            self.off(event, wrapper)

        return unsubscribe

    def _make_event_dispatcher(self, event: str) -> Callable[..., Awaitable[None]]:
        async def dispatch(*args: Any, **kwargs: Any) -> None:
            for listener in list(self._event_listeners.get(event, ())):
                await listener(*args, **kwargs)

        return dispatch

    def off(
        self,
        event: str,
        listener: Callable[..., None] | Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        if listener is None:
            self._event_listeners.pop(event, None)
            return
        listeners = self._event_listeners.get(event)
        if listeners and listener in listeners:
            listeners.remove(cast(Callable[..., Awaitable[None]], listener))
        if listeners == []:
            self._event_listeners.pop(event, None)

    async def emit(self, event: str, *args: Any, ack_callback: Callable[..., None] | None = None) -> None:
        if not self._ws_connected:
            raise RuntimeError('Socket.IO is not connected. Call ws_connect() first.')
        data = args[0] if len(args) == 1 else args if args else None
        if ack_callback is not None:
            await self._sio.emit(event, data, callback=ack_callback)
        else:
            await self._sio.emit(event, data)


async def connect_bridge_llm_handler(
    bridge_client: Any,
    openai_client: Any,
    *,
    no_thinking_request_args: dict[str, Any] | None = None,
    haiku_model: str | None = None,
    preserve_thinking: bool = True,
) -> LLMRequestHandler:
    """Connect *bridge_client* and register an LLM request handler backed by *openai_client*."""
    # Imported lazily so this module stays bundle-safe (llm_handler is not vendored).
    from nexus.runtime.server.services.llm_handler import LLMRequestHandler

    await bridge_client.ws_connect()
    handler = LLMRequestHandler(
        runtime=bridge_client,
        openai_client=openai_client,
        no_thinking_request_args=no_thinking_request_args,
        haiku_model=haiku_model,
        preserve_thinking=preserve_thinking,
    )
    handler.register()
    return handler
