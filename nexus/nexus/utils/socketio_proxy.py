"""Proxy-aware Socket.IO transport with lazy, Engine.IO-owned sessions."""

from __future__ import annotations

from collections.abc import Callable

import aiohttp

from socketio.async_client import AsyncClient

from nexus.utils.host_network import host_trust_env, websocket_proxy


class _ProxyAsyncClient(AsyncClient):
    def __init__(self, *, http_session_factory=None, **kwargs):
        self._http_session_factory = http_session_factory
        self._explicit_ws_proxy = 'proxy' in (kwargs.get('websocket_extra_options') or {})
        super().__init__(**kwargs)

    async def connect(self, url, *args, **kwargs):
        # Engine.IO creates sessions without trust_env, including after reconnect.
        # Keep ownership internal so disconnect/reset closes the session as usual.
        if self.eio.http is None or self.eio.http.closed:
            self.eio.http = (
                self._http_session_factory() if self._http_session_factory else aiohttp.ClientSession(trust_env=True)
            )
            self.eio.external_http = False
        if not self._explicit_ws_proxy and self._http_session_factory is None:
            self.eio.websocket_extra_options['proxy'] = websocket_proxy(url)
        try:
            return await super().connect(url, *args, **kwargs)
        except BaseException:
            if not self.connected and not self.eio.external_http:
                await self.eio.http.close()
            raise


def create_socketio_client(
    *, trust_env: bool | None = None, http_session_factory: Callable[[], aiohttp.ClientSession] | None = None, **kwargs
):
    use_proxy = host_trust_env() if trust_env is None else trust_env
    if use_proxy or http_session_factory is not None:
        return _ProxyAsyncClient(http_session_factory=http_session_factory, **kwargs)
    return AsyncClient(**kwargs)
