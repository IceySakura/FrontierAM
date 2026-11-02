"""Shared infrastructure for the internal Yuanbao search tools."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import threading
import time

from datetime import datetime
from os import PathLike
from typing import Any, BinaryIO

import httpx


logger = logging.getLogger(__name__)

_WEEKDAY_CN = ('星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日')

_consumer_api: Any | None = None
_consumer_api_lock = threading.Lock()


def _get_consumer_api() -> Any:
    """Create the Polaris consumer on first use, keeping the SDK optional."""
    global _consumer_api
    if _consumer_api is None:
        with _consumer_api_lock:
            if _consumer_api is None:
                try:
                    from polaris.api.consumer import create_consumer_by_config
                except ImportError as exc:  # pragma: no cover - only present in the internal runtime
                    raise RuntimeError(
                        'yb search tools require the optional polaris-cpp-py package at runtime'
                    ) from exc
                _consumer_api = create_consumer_by_config('')
    return _consumer_api


class ServiceDiscovery:
    """Resolve a Polaris service and cache the selected instance for a TTL."""

    _REFRESH_FAILURE_BACKOFF = 1.0

    def __init__(self, namespace: str, service: str, ttl: float = 60.0):
        if ttl < 0:
            raise ValueError(f'service discovery ttl must be >= 0, got {ttl}')
        self.namespace = namespace
        self.service = service
        self.ttl = ttl
        self._ip: str | None = None
        self._port: int | None = None
        self._last_refresh = 0.0
        self._next_refresh_attempt = 0.0
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        try:
            from polaris.pkg.model.service import GetOneInstanceRequest
        except ImportError as exc:  # pragma: no cover - only present in the internal runtime
            raise RuntimeError('yb search tools require the optional polaris-cpp-py package at runtime') from exc

        try:
            request = GetOneInstanceRequest(namespace=self.namespace, service=self.service)
            response = _get_consumer_api().get_one_instance_resp(request)
            instance = response[0]
            ip = instance.get_host()
            port = int(instance.get_port())
            if not ip or port <= 0:
                raise RuntimeError(f'Polaris returned an invalid instance for {self.namespace}/{self.service}')
        except Exception:
            if self._ip is None:
                raise
            logger.warning(
                'Failed to refresh Polaris service %s/%s; keeping the cached instance', self.namespace, self.service
            )
            # Don't touch _last_refresh — that would silently undo invalidate() (which sets it to -inf so
            # the next call must refresh) and re-stick a dead instance for a full TTL. Use a short backoff
            # window instead so we retry Polaris soon, without hammering it on every request.
            self._next_refresh_attempt = time.monotonic() + self._REFRESH_FAILURE_BACKOFF
            return

        self._ip = str(ip)
        self._port = port
        self._last_refresh = time.monotonic()
        self._next_refresh_attempt = 0.0
        logger.debug('Resolved Polaris service %s/%s to %s:%s', self.namespace, self.service, self._ip, self._port)

    def _maybe_refresh(self) -> None:
        now = time.monotonic()
        if self._ip is not None and now - self._last_refresh <= self.ttl:
            return
        if self._ip is not None and now < self._next_refresh_attempt:
            return
        with self._lock:
            now = time.monotonic()
            if self._ip is None or (now - self._last_refresh > self.ttl and now >= self._next_refresh_attempt):
                self._refresh()

    def invalidate(self) -> None:
        """Force the next URL lookup to refresh the selected instance."""
        with self._lock:
            self._last_refresh = float('-inf')
            self._next_refresh_attempt = 0.0

    def get_url(self, path: str) -> str:
        self._maybe_refresh()
        normalized_path = path if path.startswith('/') else f'/{path}'
        return f'http://{self._ip}:{self._port}{normalized_path}'


_discoveries: dict[tuple[str, str, float], ServiceDiscovery] = {}
_discovery_lock = threading.Lock()


def get_service_discovery(namespace: str, service: str, ttl: float = 60.0) -> ServiceDiscovery:
    """Get a discovery cache scoped to a concrete service configuration."""
    key = (namespace, service, ttl)
    with _discovery_lock:
        if key not in _discoveries:
            _discoveries[key] = ServiceDiscovery(namespace, service, ttl)
        return _discoveries[key]


def make_request_timeout(total: float = 60.0, connect: float = 10.0) -> httpx.Timeout:
    return httpx.Timeout(total, connect=connect)


class _SharedHttpxRegistry:
    """Keep one async HTTP connection pool per event loop and configuration."""

    def __init__(self):
        self._clients: dict[tuple[Any, ...], tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = {}
        self._lock = threading.Lock()

    def get(
        self,
        name: str,
        *,
        timeout: httpx.Timeout,
        max_connections: int,
        max_keepalive_connections: int,
        proxy: str | None = None,
        verify: bool | str = True,
    ) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        key = (
            id(loop),
            name,
            proxy,
            verify,
            max_connections,
            max_keepalive_connections,
            timeout.connect,
            timeout.read,
            timeout.write,
            timeout.pool,
        )
        existing = self._clients.get(key)
        if existing is not None and existing[0] is loop and not existing[1].is_closed:
            return existing[1]

        with self._lock:
            existing = self._clients.get(key)
            if existing is not None and existing[0] is loop and not existing[1].is_closed:
                return existing[1]
            client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive_connections,
                ),
                proxy=proxy,
                verify=verify,
            )
            self._clients[key] = (loop, client)
            return client


_shared_httpx_registry = _SharedHttpxRegistry()


def get_shared_httpx_client(
    name: str,
    *,
    timeout: httpx.Timeout,
    max_connections: int = 128,
    max_keepalive_connections: int = 64,
    proxy: str | None = None,
    verify: bool | str = True,
) -> httpx.AsyncClient:
    return _shared_httpx_registry.get(
        name,
        timeout=timeout,
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
        proxy=proxy,
        verify=verify,
    )


async def download_image(
    url: str,
    *,
    timeout: float = 10.0,
    proxy: str | None = None,
    verify: bool | str = True,
    max_bytes: int = 20 * 1024 * 1024,
) -> bytes:
    """Download a bounded image using the shared async client."""
    if max_bytes <= 0:
        raise ValueError(f'max_bytes must be > 0, got {max_bytes}')
    client = get_shared_httpx_client(
        'yb_image_download',
        timeout=make_request_timeout(timeout, min(timeout, 10.0)),
        max_connections=32,
        max_keepalive_connections=16,
        proxy=proxy,
        verify=verify,
    )
    chunks: list[bytes] = []
    size = 0
    async with client.stream('GET', url, follow_redirects=True) as response:
        response.raise_for_status()
        content_length = response.headers.get('Content-Length')
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                logger.debug('Ignoring invalid image Content-Length header')
            else:
                if declared_size > max_bytes:
                    raise ValueError(f'image download exceeds {max_bytes} bytes')
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f'image download exceeds {max_bytes} bytes')
            chunks.append(chunk)
    return b''.join(chunks)


def get_current_time_str(now: datetime | None = None) -> str:
    """Return the timestamp format used by the oneagent tool-response contract."""
    now = now or datetime.now()
    return now.strftime('%Y年%m月%d日（{}） %H时%M分%S秒').format(_WEEKDAY_CN[now.weekday()])


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _resize_smart(
    width: int,
    height: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int,
) -> tuple[int, int]:
    if min(width, height) <= 0:
        raise ValueError('image dimensions must be positive')
    if max(width, height) / min(width, height) >= max_ratio:
        scale = math.sqrt(max_pixels / (width * height))
        return (
            max(factor, _floor_by_factor(width * scale, factor)),
            max(factor, _floor_by_factor(height * scale, factor)),
        )

    resized_width = max(factor, _round_by_factor(width, factor))
    resized_height = max(factor, _round_by_factor(height, factor))
    if resized_width * resized_height > max_pixels:
        scale = math.sqrt((width * height) / max_pixels)
        resized_width = _floor_by_factor(width / scale, factor)
        resized_height = _floor_by_factor(height / scale, factor)
    elif resized_width * resized_height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        resized_width = _ceil_by_factor(width * scale, factor)
        resized_height = _ceil_by_factor(height * scale, factor)
    return resized_width, resized_height


def image_bytes_to_data_uri(
    content: bytes,
    *,
    max_pixels: int = 1024 * 1024,
    min_pixels: int = 256 * 256,
    factor: int = 32,
    max_ratio: int = 200,
) -> tuple[str, int, int]:
    """Resize downloaded image bytes with the oneagent policy and encode a JPEG data URI."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the runtime extra
        raise RuntimeError('yb_vision_search requires Pillow at runtime') from exc

    with Image.open(io.BytesIO(content)) as opened:
        image = opened.convert('RGB')
        width, height = image.size
        resized_width, resized_height = _resize_smart(
            width,
            height,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_ratio=max_ratio,
        )
        if (resized_width, resized_height) != image.size:
            image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/jpeg;base64,{encoded}', resized_width, resized_height


async def download_image_as_data_uri(
    url: str,
    *,
    timeout: float = 30.0,
    proxy: str | None = None,
    verify: bool | str = True,
    max_bytes: int = 20 * 1024 * 1024,
    max_pixels: int = 1024 * 1024,
    min_pixels: int = 256 * 256,
    factor: int = 32,
    max_ratio: int = 200,
) -> tuple[str, int, int]:
    content = await download_image(url, timeout=timeout, proxy=proxy, verify=verify, max_bytes=max_bytes)
    return await asyncio.to_thread(
        image_bytes_to_data_uri,
        content,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        factor=factor,
        max_ratio=max_ratio,
    )


def compress_image(
    image_source: str | PathLike[str] | bytes | bytearray | BinaryIO,
    *,
    max_size: int = 1024 * 1024,
    quality: int = 95,
) -> tuple[bool | None, int, Any | None, bytes | None]:
    """Load an image and reduce JPEG quality until it fits ``max_size``.

    Returns ``(was_recompressed, final_quality, pil_image, jpeg_bytes)`` so callers can keep the decoded
    PIL image and, when appropriate for their transport contract, reuse the JPEG bytes produced by the
    size-check encode pass.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the runtime extra
        raise RuntimeError('yb_vision_search requires Pillow at runtime') from exc
    if max_size <= 0:
        raise ValueError(f'max_size must be > 0, got {max_size}')
    if isinstance(image_source, (bytes, bytearray)):
        source: Any = io.BytesIO(image_source)
    else:
        source = image_source

    try:
        with Image.open(source) as opened:
            image = opened.convert('RGB') if opened.mode != 'RGB' else opened.copy()
    except Exception as exc:
        logger.warning('Failed to load image for vision search: %s', exc)
        return None, -1, None, None

    original_quality = min(max(int(quality), 5), 100)
    current_quality = original_quality
    while current_quality >= 5:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=current_quality)
        if buffer.tell() <= max_size:
            return current_quality != original_quality, current_quality, image, buffer.getvalue()
        current_quality -= 5
    return None, -1, None, None


def pil_image_to_base64(image: Any, quality: int = 95) -> str:
    """Encode a PIL image as base64 JPEG for the internal RPC payload."""
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('ascii')
