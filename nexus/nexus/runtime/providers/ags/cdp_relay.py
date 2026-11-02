"""Runtime-owned loopback relay for CDP connections through environment proxies."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import secrets

from collections.abc import Callable

import aiohttp

from aiohttp import web
from aiohttp.helpers import get_env_proxy_for_url
from yarl import URL

from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


class _HandshakeError(Exception):
    def __init__(self, status: int, diagnostic_id: str, stage: str):
        self.status = status
        self.diagnostic_id = diagnostic_id
        self.stage = stage
        super().__init__(f'CDP upstream unavailable; stage={stage}; diagnostic_id={diagnostic_id}')


class CDPRelay:
    def __init__(
        self,
        upstream_url: str,
        headers: dict[str, str],
        *,
        heartbeat: float | None = None,
        connect_timeout: float = 30,
        connect_attempts: int = 3,
        diagnostic_callback: Callable[[dict], None] | None = None,
    ) -> None:
        if not math.isfinite(connect_timeout) or connect_timeout <= 0:
            raise ValueError('CDP connect_timeout must be finite and positive')
        if not 1 <= connect_attempts <= 3:
            raise ValueError('CDP connect_attempts must be between 1 and 3')
        self._upstream_url = URL(upstream_url)
        self._headers = headers
        self._heartbeat = heartbeat
        self._connect_timeout = connect_timeout
        self._connect_attempts = connect_attempts
        self._diagnostic_callback = diagnostic_callback
        self._lock = asyncio.Lock()
        self._runner: web.AppRunner | None = None
        self._session: aiohttp.ClientSession | None = None
        self._connections: set[web.WebSocketResponse] = set()
        self._handshakes: set[asyncio.Task] = set()
        self._url: str | None = None
        self._closed = False

    async def start(self) -> str:
        async with self._lock:
            if self._closed:
                raise RuntimeError('CDP relay is closed')
            if self._url is not None:
                return self._url
            app = web.Application()
            path = f'/cdp/{secrets.token_urlsafe(24)}'
            app.router.add_get(path, self._connect)
            app.on_shutdown.append(self._shutdown)
            runner = web.AppRunner(app, access_log=None, shutdown_timeout=5)
            self._session = aiohttp.ClientSession(
                trust_env=True, timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)
            )
            try:
                await runner.setup()
                site = web.TCPSite(runner, '127.0.0.1', 0)
                await site.start()
            except BaseException:
                await runner.cleanup()
                await self._session.close()
                self._session = None
                raise
            self._runner = runner
            self._url = f'ws://127.0.0.1:{runner.addresses[0][1]}{path}'
            return self._url

    def _record(self, diagnostic_id: str, **fields) -> None:
        # No URLs, query strings, response bodies, proxy credentials or raw
        # exception messages: endpoint identity is sufficient for reconnection.
        record = {'diagnostic_id': diagnostic_id, **fields}
        if fields.get('event') in {'failure', 'exhausted', 'forwarding_error'}:
            logger.warning('CDP relay: %s', record)
        else:
            logger.debug('CDP relay: %s', record)
        if self._diagnostic_callback is not None:
            try:
                self._diagnostic_callback(record)
            except Exception:
                logger.warning('CDP relay diagnostic sink failed')

    async def _establish(self, diagnostic_id: str) -> aiohttp.ClientWebSocketResponse:
        """Readiness is a real upstream handshake, not a separate racy probe.

        No CDP command or downstream upgrade occurs until this succeeds.
        Every retry rediscovers the browser ID; established sessions are never
        transparently reconnected or replayed.
        """
        assert self._session is not None
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self._connect_timeout
        previous_endpoint = None
        last_status = 502
        stage = 'discovery'
        try:
            proxy, proxy_auth = get_env_proxy_for_url(self._upstream_url)
        except LookupError:
            proxy, proxy_auth = None, None
        for attempt in range(1, self._connect_attempts + 1):
            if self._closed:
                raise _HandshakeError(503, diagnostic_id, stage)
            remaining = deadline - loop.time()
            if remaining <= 0:
                last_status = 504
                break
            stage = 'discovery'
            attempt_started = loop.time()
            try:
                # Cap individual attempts so a hung first request cannot
                # consume all opportunities to recover.
                async with async_timeout(min(10, remaining)):
                    async with self._session.get(
                        self._upstream_url / 'json/version',
                        headers=self._headers,
                        timeout=aiohttp.ClientTimeout(total=min(10, remaining)),
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()
                    if not isinstance(data, dict) or not isinstance(data.get('webSocketDebuggerUrl'), str):
                        raise ValueError('Missing CDP endpoint')
                    endpoint = URL(data['webSocketDebuggerUrl'])
                    if endpoint.scheme not in {'ws', 'wss'} or not endpoint.host or not endpoint.path.strip('/'):
                        raise ValueError('Invalid CDP endpoint')
                    endpoint_id = hashlib.sha256(str(endpoint).encode()).hexdigest()[:16]
                    self._record(
                        diagnostic_id,
                        event='discovered',
                        stage=stage,
                        attempt=attempt,
                        endpoint_id=endpoint_id,
                        endpoint_changed=previous_endpoint is not None and endpoint_id != previous_endpoint,
                        downstream_prepared=False,
                    )
                    previous_endpoint = endpoint_id
                    # Only use the advertised path/query, never its authority,
                    # credentials or localhost address behind the AGS gateway.
                    host = self._upstream_url.host
                    assert host is not None
                    endpoint = endpoint.with_user(None).with_fragment(None)
                    endpoint = endpoint.with_scheme('wss' if self._upstream_url.scheme == 'https' else 'ws')
                    endpoint = endpoint.with_host(host).with_port(self._upstream_url.port)
                    stage = 'upstream_ws'
                    upstream = await self._session.ws_connect(
                        endpoint,
                        headers=self._headers,
                        proxy=proxy,
                        proxy_auth=proxy_auth,
                        max_msg_size=0,
                        heartbeat=self._heartbeat,
                    )
                self._record(
                    diagnostic_id,
                    event='connected',
                    stage=stage,
                    attempt=attempt,
                    elapsed_s=round(loop.time() - started, 3),
                    downstream_prepared=False,
                )
                return upstream
            except (aiohttp.ClientError, OSError, TimeoutError, ValueError, KeyError) as exc:
                upstream_status = getattr(exc, 'status', None)
                retry = (
                    upstream_status in RETRY_STATUSES
                    or (stage == 'upstream_ws' and upstream_status == 404)
                    or isinstance(exc, (aiohttp.ClientConnectionError, OSError, TimeoutError))
                )
                if isinstance(exc, (aiohttp.ClientSSLError, aiohttp.InvalidURL)):
                    retry = False
                last_status = 504 if isinstance(exc, TimeoutError) else 502
                retry = retry and attempt < self._connect_attempts and deadline - loop.time() > 1
                self._record(
                    diagnostic_id,
                    event='failure',
                    stage=stage,
                    attempt=attempt,
                    upstream_status=upstream_status,
                    error_type=type(exc).__name__,
                    attempt_elapsed_s=round(loop.time() - attempt_started, 3),
                    elapsed_s=round(loop.time() - started, 3),
                    retry=retry,
                    downstream_prepared=False,
                )
                if not retry:
                    break
                await asyncio.sleep(min(2 ** (attempt - 1), max(0, deadline - loop.time())))
        self._record(
            diagnostic_id,
            event='exhausted',
            stage=stage,
            status=last_status,
            elapsed_s=round(loop.time() - started, 3),
            downstream_prepared=False,
        )
        raise _HandshakeError(last_status, diagnostic_id, stage)

    async def _connect(self, request: web.Request) -> web.StreamResponse:
        diagnostic_id = secrets.token_hex(8)
        handshake = asyncio.create_task(self._establish(diagnostic_id))
        self._handshakes.add(handshake)

        async def disconnected():
            while request.transport is not None and not request.transport.is_closing():
                await asyncio.sleep(0.1)

        watcher = asyncio.create_task(disconnected())
        upstream = None
        try:
            await asyncio.wait((handshake, watcher), return_when=asyncio.FIRST_COMPLETED)
            if watcher.done():
                raise asyncio.CancelledError
            upstream = await handshake
        except _HandshakeError as exc:
            return web.json_response(
                {'error': 'cdp_upstream_unavailable', 'stage': exc.stage, 'diagnostic_id': exc.diagnostic_id},
                status=exc.status,
            )
        except asyncio.CancelledError:
            self._record(diagnostic_id, event='cancelled', stage='handshake', downstream_prepared=False)
            raise
        finally:
            watcher.cancel()
            if not handshake.done():
                handshake.cancel()
            await asyncio.gather(watcher, handshake, return_exceptions=True)
            self._handshakes.discard(handshake)
            # Cancellation may race a completed upstream connection.
            if upstream is None and not handshake.cancelled() and handshake.exception() is None:
                await handshake.result().close()
        try:
            downstream = web.WebSocketResponse(max_msg_size=0)
            await downstream.prepare(request)
            self._connections.add(downstream)
            tasks = [
                asyncio.create_task(self._forward(downstream, upstream)),
                asyncio.create_task(self._forward(upstream, downstream)),
            ]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except (aiohttp.ClientError, OSError, RuntimeError) as exc:
                self._record(
                    diagnostic_id,
                    event='forwarding_error',
                    stage='forwarding',
                    error_type=type(exc).__name__,
                    downstream_prepared=True,
                )
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self._connections.discard(downstream)
                await downstream.close()
            return downstream
        finally:
            await upstream.close()

    @staticmethod
    async def _forward(source, target) -> None:
        async for message in source:
            if message.type == aiohttp.WSMsgType.TEXT:
                await target.send_str(message.data)
            elif message.type == aiohttp.WSMsgType.BINARY:
                await target.send_bytes(message.data)
            elif message.type == aiohttp.WSMsgType.ERROR:
                raise source.exception() or RuntimeError('CDP WebSocket failed')

    async def _shutdown(self, app: web.Application) -> None:
        for task in tuple(self._handshakes):
            task.cancel()
        await asyncio.gather(*tuple(self._handshakes), return_exceptions=True)
        await asyncio.gather(*(ws.close() for ws in tuple(self._connections)))

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            try:
                if self._runner is not None:
                    await self._runner.cleanup()
            finally:
                if self._session is not None:
                    await self._session.close()
                self._runner = None
                self._session = None
                self._url = None
