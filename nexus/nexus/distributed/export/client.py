"""Cloud export client."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, BinaryIO
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import aiohttp

from aiohttp.http_exceptions import TransferEncodingError

from nexus.distributed.export.types import ExportError


try:
    import orjson
except ImportError:  # pragma: no cover - orjson is an optional accelerator
    orjson = None  # type: ignore[assignment]


_API_PAGE_SIZE = 500


_API_ATTEMPTS = 3


# Result downloads must not live under a wall-clock cap: a many-GB body plus a
# legacy (non-streaming) server's silent assembly phase cannot fit any fixed
# total — the previous session-wide total=300s aborted exactly those transfers
# and then re-downloaded everything three times. Bound each idle gap instead
# (sock_read also covers waiting for the response headers); the default exceeds
# the observed legacy assembly silence (~10 minutes). Slow environments can
# raise it via NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS.
_DEFAULT_RESULT_SOCK_READ_SECONDS = 900.0


def _result_stream_timeout() -> aiohttp.ClientTimeout:
    raw = os.environ.get('NEXUS_EXPORT_RESULT_SOCK_READ_SECONDS')
    try:
        sock_read = float(raw) if raw else _DEFAULT_RESULT_SOCK_READ_SECONDS
    except ValueError:
        sock_read = _DEFAULT_RESULT_SOCK_READ_SECONDS
    return aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=max(sock_read, 1.0))


class _TransientApiError(RuntimeError):
    pass


class CloudApi:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        control_url: str,
        application_url: str | None = None,
        token: str,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not control_url or not token:
            raise ExportError('Control URL and Taihu token are required')
        self.session = session
        self.control_url = control_url.rstrip('/')
        self.application_url = application_url.rstrip('/') if application_url else None
        self.headers = {'Authorization': f'Bearer {token}'}
        self.sleep = sleep

    async def get_execution(self, execution_id: str) -> dict[str, Any]:
        value = await self._get_json(f'{self.control_url}/v1/executions/{quote(execution_id, safe="")}')
        assert value is not None
        return value

    async def get_batch_summary(self, execution_id: str) -> dict[str, Any]:
        path = f'/v1/executions/{quote(execution_id, safe="")}/batch?include_status_counts=1'
        value = await self._get_json(f'{self.control_url}{path}')
        assert value is not None
        return value

    async def get_batch_items(self, execution_id: str, cursor: str | None) -> dict[str, Any]:
        query: dict[str, Any] = {'limit': _API_PAGE_SIZE}
        if cursor is not None:
            query['cursor'] = cursor
        path = f'/v1/executions/{quote(execution_id, safe="")}/batch/items?{urlencode(query)}'
        value = await self._get_json(f'{self.control_url}{path}')
        assert value is not None
        return value

    async def get_result(self, execution_id: str) -> dict[str, Any] | None:
        if self.application_url is None:
            raise ExportError('Application URL is required for results')
        path = f'/v1/executions/{quote(execution_id, safe="")}/result'
        value = await self._get_json(
            f'{self.application_url}{path}', allow_missing=True, timeout=_result_stream_timeout()
        )
        if value is None or not value:
            return None
        if not isinstance(value.get('success'), bool):
            raise ExportError(f'invalid RunOutput returned for {execution_id}')
        return value

    async def get_result_to_file(
        self,
        execution_id: str,
        fileobj: BinaryIO,
        *,
        chunk_size: int = 1 << 20,
        scan: Any | None = None,
    ) -> int | None:
        """Stream the RunOutput body into a binary fileobj; return its byte length.

        Disk-prefetch companion of get_result: the body is written chunk by
        chunk so a prefetched record never holds the payload in RAM. Semantics
        mirror get_result/_get_json: 429/5xx/connection failures retry with the
        same backoff (the file is truncated before each retry), other 4xx raise
        ExportError, and 404 returns None (a missing result is a valid state).

        ``scan`` (optional) receives every downloaded chunk via ``feed`` and is
        reset at the start of each retry attempt — used to validate and parse
        the body during download (方案 D) instead of re-reading the spill file.
        """
        if self.application_url is None:
            raise ExportError('Application URL is required for results')
        url = f'{self.application_url}/v1/executions/{quote(execution_id, safe="")}/result'
        last_error: BaseException | None = None
        for attempt in range(_API_ATTEMPTS):
            retry_after = 0.0
            try:
                fileobj.seek(0)
                fileobj.truncate()
                if scan is not None:
                    scan.reset()
                async with self.session.get(url, headers=self.headers, timeout=_result_stream_timeout()) as response:
                    if response.status == 404:
                        return None
                    if response.status == 429 or response.status >= 500:
                        retry_after = _retry_after_seconds(response.headers.get('Retry-After'))
                        raise _TransientApiError(f'HTTP {response.status}')
                    if response.status >= 400:
                        raise ExportError(f'API request failed with HTTP {response.status}: {_safe_url(url)}')
                    written = 0
                    head = b''
                    while True:
                        chunk = await response.content.read(chunk_size)
                        if not chunk:
                            # Mirror get_result's falsy guard: a 200 body that
                            # parses to an empty JSON value ({}/null/[]/0/false/"")
                            # is a lost item's "no result" — the in-RAM path
                            # returns None for it, so keep the same missing-row
                            # semantics here instead of failing schema checks
                            # on an empty body.
                            if written <= 1024 and head:
                                try:
                                    falsy = not _loads_json_bytes(head[:1024])
                                except Exception:
                                    falsy = False
                                if falsy:
                                    return None
                            return written
                        if not head:
                            head = chunk
                        fileobj.write(chunk)
                        if scan is not None:
                            scan.feed(chunk)
                        written += len(chunk)
            except ExportError:
                raise
            except (
                _TransientApiError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,  # mid-body connection resets
                TransferEncodingError,  # broken chunked stream (server-side abort)
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                last_error = exc
                if attempt + 1 == _API_ATTEMPTS:
                    break
                await self.sleep(max(retry_after, (0.25 * (2**attempt)) + random.uniform(0, 0.1)))
        raise ExportError(f'API request failed after {_API_ATTEMPTS} attempts: {_safe_url(url)}') from last_error

    async def get_control_events(self, execution_id: str) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        cursor = 0
        while True:
            query = urlencode({'cursor': cursor, 'limit': _API_PAGE_SIZE})
            path = f'/v1/executions/{quote(execution_id, safe="")}/control-events?{query}'
            payload = await self._get_json(f'{self.control_url}{path}')
            assert payload is not None
            raw_items = payload.get('items')
            if not isinstance(raw_items, list):
                raise ExportError(f'invalid control-events response for {execution_id}')
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    raise ExportError(f'invalid control event for {execution_id}')
                sequence = raw_item.get('seq')
                if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= cursor:
                    raise ExportError(f'control event cursor did not advance for {execution_id}')
                cursor = sequence
                events.append(raw_item)
            if len(raw_items) < _API_PAGE_SIZE:
                return events

    async def _get_json(
        self,
        url: str,
        *,
        allow_missing: bool = False,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any] | None:
        last_error: BaseException | None = None
        for attempt in range(_API_ATTEMPTS):
            retry_after = 0.0
            try:
                async with self.session.get(url, headers=self.headers, timeout=timeout) as response:
                    if allow_missing and response.status == 404:
                        return None
                    if response.status == 429 or response.status >= 500:
                        retry_after = _retry_after_seconds(response.headers.get('Retry-After'))
                        raise _TransientApiError(f'HTTP {response.status}')
                    if response.status >= 400:
                        raise ExportError(f'API request failed with HTTP {response.status}: {_safe_url(url)}')
                    try:
                        payload = _loads_json_bytes(await response.read())
                    except ValueError as exc:
                        raise ExportError(f'API returned invalid JSON: {_safe_url(url)}') from exc
                    if not isinstance(payload, dict):
                        raise ExportError(f'API returned a non-object response: {_safe_url(url)}')
                    return payload
            except ExportError:
                raise
            except (
                _TransientApiError,
                aiohttp.ClientConnectionError,
                aiohttp.ClientPayloadError,  # mid-body connection resets
                TransferEncodingError,  # broken chunked stream (server-side abort)
                aiohttp.ServerTimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                last_error = exc
                if attempt + 1 == _API_ATTEMPTS:
                    break
                await self.sleep(max(retry_after, (0.25 * (2**attempt)) + random.uniform(0, 0.1)))
        raise ExportError(f'API request failed after {_API_ATTEMPTS} attempts: {_safe_url(url)}') from last_error


def _loads_json_bytes(body: bytes | bytearray | memoryview) -> Any:
    """Parse an API JSON body, preferring orjson when available.

    orjson parses UTF-8 bytes directly, so a large result blob never becomes a
    full Python str first (a 2-4x wider copy for non-ASCII data is the largest
    single component of export peak memory). stdlib stays as the fallback: it
    tolerates lone surrogate escapes (\\udXXX) and non-UTF-8 encodings that
    orjson rejects, matching the lenient behaviour of aiohttp's response.json().
    """
    if orjson is not None:
        try:
            return orjson.loads(body)
        except orjson.JSONDecodeError:
            pass  # lone surrogates, non-UTF-8, or >64-bit ints: retry leniently
    if isinstance(body, memoryview):
        body = body.tobytes()
    return json.loads(body)


def _retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, delay) if math.isfinite(delay) else 0.0


def _safe_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    if ':' in host:
        host = f'[{host}]'
    if parsed.port is not None:
        host += f':{parsed.port}'
    return urlunsplit((parsed.scheme, host, parsed.path, '', ''))
