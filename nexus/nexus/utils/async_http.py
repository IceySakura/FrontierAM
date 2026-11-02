import asyncio
import atexit

from typing import Any

import aiohttp

from aiohttp import RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL


class AsyncHTTP:
    """Async HTTP client with a requests-like simple interface"""

    # Global instance pool, multiple configured instances per event loop
    _instances: dict[tuple, 'AsyncHTTP'] = {}
    _sessions_to_close = []

    def __init__(self, *, keepalive_timeout: float | None = None, **session_kwargs: Any):
        """
        Create AsyncHTTP instance

        Args:
            **session_kwargs: Parameters passed to aiohttp.ClientSession
                such as timeout, headers, connector, etc.
        """
        self._session_kwargs = session_kwargs
        self._keepalive_timeout = keepalive_timeout
        self._session: aiohttp.ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create session for current event loop"""
        loop = asyncio.get_running_loop()

        # If event loop changed or session not created yet
        if self._loop != loop or self._session is None or self._session.closed:
            # Use custom connector if provided, otherwise create one with unlimited connections
            # limit=0 means no limit, each request gets its own connection if needed
            if 'connector' not in self._session_kwargs:
                if self._keepalive_timeout is None:
                    connector = aiohttp.TCPConnector(limit=0)
                else:
                    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=self._keepalive_timeout)
                self._session = aiohttp.ClientSession(connector=connector, **self._session_kwargs)
            else:
                self._session = aiohttp.ClientSession(**self._session_kwargs)
            self._loop = loop
            # Register to cleanup list
            if self._session not in self._sessions_to_close:
                self._sessions_to_close.append(self._session)

        return self._session

    async def get(self, url: str, **kwargs) -> 'HTTPResponse':
        session = self._get_session()
        async with session.get(url, **kwargs) as response:
            return await HTTPResponse.from_response(response)

    async def post(self, url: str, **kwargs) -> 'HTTPResponse':
        session = self._get_session()
        async with session.post(url, **kwargs) as response:
            return await HTTPResponse.from_response(response)

    async def put(self, url: str, **kwargs) -> 'HTTPResponse':
        session = self._get_session()
        async with session.put(url, **kwargs) as response:
            return await HTTPResponse.from_response(response)

    async def delete(self, url: str, **kwargs) -> 'HTTPResponse':
        session = self._get_session()
        async with session.delete(url, **kwargs) as response:
            return await HTTPResponse.from_response(response)

    async def request(self, method: str, url: str, **kwargs) -> 'HTTPResponse':
        session = self._get_session()
        async with session.request(method, url, **kwargs) as response:
            return await HTTPResponse.from_response(response)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            if self._session in self._sessions_to_close:
                self._sessions_to_close.remove(self._session)

    @classmethod
    async def close_all(cls):
        sessions = cls._sessions_to_close.copy()
        cls._sessions_to_close.clear()

        for session in sessions:
            if not session.closed:
                await session.close()

    @classmethod
    def _cleanup_sync(cls):
        """Synchronous cleanup function called on program exit"""
        sessions = cls._sessions_to_close.copy()
        cls._sessions_to_close.clear()

        for session in sessions:
            if not session.closed:
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        loop.run_until_complete(session.close())
                except Exception:
                    pass


class HTTPResponse:
    """HTTP response object, similar to requests.Response"""

    def __init__(self, status: int, headers: dict, text: str, content: bytes, url: str, encoding: str):
        self.status = status
        self.status_code = status
        self.headers = headers
        self._text = text
        self._content = content
        self.url = url
        self.encoding = encoding
        self.ok = 200 <= status < 300

    @classmethod
    async def from_response(cls, response: aiohttp.ClientResponse) -> 'HTTPResponse':
        """Create from aiohttp.ClientResponse"""
        content = await response.read()
        encoding = response.get_encoding()
        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = content.decode('utf-8', errors='replace')

        return cls(
            status=response.status,
            headers=dict(response.headers),
            text=text,
            content=content,
            url=str(response.url),
            encoding=encoding,
        )

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._content

    def json(self, **kwargs) -> Any:
        import json

        return json.loads(self._text, **kwargs)

    def raise_for_status(self):
        """Check status code, raise exception on failure"""
        if not self.ok:
            # Create a minimal RequestInfo for the error
            request_info = RequestInfo(
                url=URL(self.url),
                method='GET',
                headers=CIMultiDictProxy(CIMultiDict()),
                real_url=URL(self.url),
            )
            raise aiohttp.ClientResponseError(
                request_info=request_info, history=(), status=self.status, message=f'HTTP {self.status}'
            )


atexit.register(AsyncHTTP._cleanup_sync)

http_client = AsyncHTTP(timeout=aiohttp.ClientTimeout(total=None))

# Opt-in client for callers that need environment proxies and NO_PROXY.
http_client_with_trust_env = AsyncHTTP(timeout=aiohttp.ClientTimeout(total=None), trust_env=True)
