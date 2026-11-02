"""Async client for Taiji Nexus dataset IO service."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import stat

from pathlib import Path
from typing import Any

import aiohttp

from nexus.dataset.auth import (
    DatasetAuthenticator,
    TaiPatAuthenticator,
    build_default_authenticator,
    format_dataset_auth_error,
    prefer_tai_pat,
)
from nexus.dataset.exceptions import DatasetAuthenticationError, DatasetClientError, DatasetConfigurationError
from nexus.utils.cos import (
    DEFAULT_MULTIPART_PART_SIZE,
    CosClientError,
    select_multipart_part_size,
    upload_presigned_file,
    upload_presigned_multipart_file,
)


ENV_SERVER_URL = 'TAIJI_DATASET_SERVER_URL'
ENV_AUTH_TOKEN = 'TAIJI_DATASET_AUTH_TOKEN'
# Legacy TAuth-backed IO endpoint (direct service, no Taihu gateway).
DEFAULT_SERVER_URL = 'http://taiji-nexus-dataset-io-dev.woa.com:8080'
# Taihu MCP gateway endpoint, used when authenticating with a Taihu PAT.
DEFAULT_TAIJI_SERVER_URL = 'https://taiji-nexus-dataset-io-dev.mcp.woa.com'
DEFAULT_TIMEOUT_SECONDS = 30
DATASET_MULTIPART_THRESHOLD = 1024 * 1024 * 1024
DEFAULT_UPLOAD_CONCURRENCY = 5
DATASET_UPLOAD_READ_TIMEOUT_SECONDS = 600

# --------------------------------------------------------------------------- retry
#
# 只对**幂等只读**端点启用有限重试（GET 类 + sign_files），且只重试瞬时性
# 状态码（429+ 5xx 家族）。写路径（sign_uploads / sign_multipart_uploads /
# commit / publish）保持一次即抛，避免在服务端已经部分执行的情况下重复动作。
#
# 节奏与 SDK DatasetClient 对称：3次重试上限、
# 指数退避（base * 2^(attempt-1)）+ 抖动 —— 抖动是关键，多线程/多协程在同一
# 秒失败时无抖动重试会同时再撞回后端，反而加剧风暴。
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 0.5  # 秒；第 n 次重试前 sleep = base * 2^(n-1) + jitter
DEFAULT_RETRY_BACKOFF_JITTER = 0.3  # 秒；random.random() * jitter
DEFAULT_RETRY_BACKOFF_MAX = 8.0  # 秒；单次 sleep 上限，避免指数退避打得过久
RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

logger = logging.getLogger(__name__)


class DatasetIoClient:
    """Client for IO service dataset version, item, and file-sign endpoints."""

    def __init__(
        self,
        *,
        server_url: str | None = None,
        authenticator: DatasetAuthenticator | None = None,
        auth_token: str | None = None,
        session: aiohttp.ClientSession | None = None,
        allow_anonymous: bool = False,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
        retry_backoff_jitter: float = DEFAULT_RETRY_BACKOFF_JITTER,
        retry_backoff_max: float = DEFAULT_RETRY_BACKOFF_MAX,
    ) -> None:
        resolved_auth_token = auth_token if auth_token is not None else os.getenv(ENV_AUTH_TOKEN)
        self._prefer_tai_pat = prefer_tai_pat(auth_token=resolved_auth_token, authenticator=authenticator)
        # Priority: explicit arg > env override > auth-mode default. Resolving the
        # env var before the auth-mode default keeps a user-set
        # TAIJI_DATASET_SERVER_URL from being silently overridden in PAT mode.
        resolved_server_url = server_url if server_url is not None else os.getenv(ENV_SERVER_URL)
        if resolved_server_url is None:
            resolved_server_url = DEFAULT_TAIJI_SERVER_URL if self._prefer_tai_pat else DEFAULT_SERVER_URL
        self.server_url = resolved_server_url.rstrip('/')
        self.auth_token = resolved_auth_token
        self.authenticator = None if self.auth_token is not None else authenticator
        self.session = session
        self.allow_anonymous = allow_anonymous
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_base = max(0.0, float(retry_backoff_base))
        self._retry_backoff_jitter = max(0.0, float(retry_backoff_jitter))
        self._retry_backoff_max = max(0.0, float(retry_backoff_max))

    def _uses_tai_pat(self) -> bool:
        """Return ``True`` when the active authentication mode is Taihu PAT."""

        if self.auth_token is not None:
            return False
        if self.authenticator is not None:
            return isinstance(self.authenticator, TaiPatAuthenticator)
        return self._prefer_tai_pat

    async def get_dataset(self, namespace: str, dataset: str) -> dict[str, Any]:
        return await self._request(
            'GET',
            f'/api/datasets/{namespace}/{dataset}',
            allow_anonymous=self.allow_anonymous,
            retriable=True,
        )

    async def get_version(self, namespace: str, dataset: str, version: str) -> dict[str, Any]:
        return await self._request(
            'GET',
            f'/api/datasets/{namespace}/{dataset}/versions/{version}',
            allow_anonymous=self.allow_anonymous,
            retriable=True,
        )

    async def list_items(
        self,
        namespace: str,
        dataset: str,
        version: str,
        *,
        split: str | None = None,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        params = _drop_none({'split': split, 'cursor': cursor, 'limit': limit})
        return await self._request(
            'GET',
            f'/api/datasets/{namespace}/{dataset}/versions/{version}/items',
            params=params,
            allow_anonymous=self.allow_anonymous,
            retriable=True,
        )

    async def get_item(
        self,
        namespace: str,
        dataset: str,
        version: str,
        item_id: str,
        *,
        split: str | None = None,
    ) -> dict[str, Any]:
        params = _drop_none({'split': split})
        return await self._request(
            'GET',
            f'/api/datasets/{namespace}/{dataset}/versions/{version}/items/{item_id}',
            params=params,
            allow_anonymous=self.allow_anonymous,
            retriable=True,
        )

    async def sign_files(
        self,
        namespace: str,
        dataset: str,
        version: str,
        files: list[str],
        *,
        split: str | None = None,
        expires: int | None = None,
    ) -> dict[str, Any]:
        params = _drop_none({'split': split})
        body = _drop_none({'files': files, 'expires': expires})
        return await self._request(
            'POST',
            f'/api/datasets/{namespace}/{dataset}/versions/{version}/files:sign',
            params=params,
            json_body=body,
            allow_anonymous=self.allow_anonymous,
            retriable=True,
        )

    async def sign_uploads(
        self,
        namespace: str,
        dataset: str,
        target_version: str,
        files: list[str],
        *,
        expires: int | None = None,
    ) -> dict[str, Any]:
        body = _drop_none({'target_version': target_version, 'files': files, 'expires': expires})
        return await self._request('POST', f'/api/datasets/{namespace}/{dataset}/uploads:sign', json_body=body)

    async def sign_multipart_uploads(
        self,
        namespace: str,
        dataset: str,
        target_version: str,
        key: str,
        total_size: int,
        *,
        part_size: int | None = None,
        expires: int | None = None,
    ) -> dict[str, Any]:
        body = _drop_none({
            'target_version': target_version,
            'key': key,
            'total_size': total_size,
            'part_size': part_size,
            'expires': expires,
        })
        return await self._request(
            'POST',
            f'/api/datasets/{namespace}/{dataset}/uploads:sign-multipart',
            json_body=body,
        )

    async def commit(self, namespace: str, dataset: str, snapshot: str) -> dict[str, Any]:
        return await self._request(
            'POST',
            f'/api/datasets/{namespace}/{dataset}:commit',
            json_body={'snapshot': snapshot},
        )

    async def publish(
        self,
        namespace: str,
        dataset: str,
        *,
        source_snapshot: str,
        release_version: str,
    ) -> dict[str, Any]:
        return await self._request(
            'POST',
            f'/api/datasets/{namespace}/{dataset}:publish',
            json_body={'source_snapshot': source_snapshot, 'release_version': release_version},
        )

    async def upload_file(
        self,
        signed_url: str,
        data: bytes | Path,
        *,
        content_type: str | None = None,
    ) -> None:
        session = self._session()
        headers = {'Expect': ''}
        if content_type:
            headers['Content-Type'] = content_type
        if isinstance(data, Path):
            with open(data, 'rb') as f:
                async with session.put(signed_url, data=f, headers=headers) as response:
                    if response.status < 200 or response.status >= 300:
                        text = await response.text()
                        raise DatasetClientError(f'Upload failed (HTTP {response.status}): {text[:300]}')
            return
        async with session.put(signed_url, data=data, headers=headers) as response:
            if response.status < 200 or response.status >= 300:
                text = await response.text()
                raise DatasetClientError(f'Upload failed (HTTP {response.status}): {text[:300]}')

    async def upload_local_file(
        self,
        namespace: str,
        dataset: str,
        target_version: str,
        local_path: str | Path,
        *,
        key: str | None = None,
        content_type: str | None = None,
        part_size: int | None = None,
        expires: int | None = None,
        max_concurrency: int = DEFAULT_UPLOAD_CONCURRENCY,
    ) -> dict[str, Any]:
        """Upload one local file, selecting a regular or multipart COS transfer."""

        path = Path(local_path)
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise DatasetClientError(f'Failed to inspect local file {path}: {exc}') from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f'Upload path is not a regular file: {path}')
        if max_concurrency < 1:
            raise ValueError('max_concurrency must be at least 1')

        object_key = key or path.name
        if not object_key:
            raise ValueError('Upload file name must not be empty')

        try:
            if metadata.st_size < DATASET_MULTIPART_THRESHOLD:
                result = await self.sign_uploads(
                    namespace,
                    dataset,
                    target_version,
                    [object_key],
                    expires=expires,
                )
                signed_url = _extract_upload_url(result, object_key)
                await self.upload_signed_local_file(
                    signed_url,
                    path,
                    key=object_key,
                    content_type=content_type,
                    expected_stat=metadata,
                )
                return result

            selected_part_size = select_multipart_part_size(
                metadata.st_size,
                DEFAULT_MULTIPART_PART_SIZE if part_size is None else part_size,
            )
            result = await self.sign_multipart_uploads(
                namespace,
                dataset,
                target_version,
                object_key,
                metadata.st_size,
                part_size=selected_part_size,
                expires=expires,
            )
            await upload_presigned_multipart_file(
                self._session(),
                path,
                result,
                key=object_key,
                max_concurrency=max_concurrency,
                timeout=DATASET_UPLOAD_READ_TIMEOUT_SECONDS,
                expected_stat=metadata,
            )
            return result
        except CosClientError as exc:
            raise DatasetClientError(str(exc)) from exc

    async def upload_signed_local_file(
        self,
        signed_url: str,
        local_path: str | Path,
        *,
        key: str,
        content_type: str | None = None,
        expected_stat: os.stat_result | None = None,
    ) -> None:
        """Upload a local file through an existing single-PUT URL."""

        try:
            await upload_presigned_file(
                self._session(),
                local_path,
                signed_url,
                key=key,
                content_type=content_type,
                timeout=DATASET_UPLOAD_READ_TIMEOUT_SECONDS,
                expected_stat=expected_stat,
            )
        except CosClientError as exc:
            raise DatasetClientError(str(exc)) from exc

    async def download(self, signed_url: str) -> bytes:
        session = self._session()
        async with session.get(signed_url) as response:
            if response.status < 200 or response.status >= 300:
                text = await response.text()
                raise DatasetClientError(f'Download failed (HTTP {response.status}): {text[:300]}')
            return await response.read()

    async def download_file(self, signed_url: str, target: Path) -> None:
        session = self._session()
        target.parent.mkdir(parents=True, exist_ok=True)
        async with session.get(signed_url) as response:
            if response.status < 200 or response.status >= 300:
                text = await response.text()
                raise DatasetClientError(f'Download failed (HTTP {response.status}): {text[:300]}')
            with open(target, 'wb') as f:
                async for chunk in response.content.iter_chunked(1024 * 1024):
                    f.write(chunk)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        allow_anonymous: bool = False,
        retriable: bool = False,
    ) -> dict[str, Any]:
        """Send a single API request, optionally with bounded retry on transient failures.

        ``retriable`` MUST only be set to True by callers that are **idempotent**
        and **safe to replay** (GET endpoints, plus ``sign_files`` which just
        re-signs download URLs). Write / commit / publish paths must leave it
        False so a 5xx response isn't blindly replayed against a server that
        may already have partially applied the change.
        """
        url = f'{self.server_url}{path}'
        max_attempts = self._max_retries + 1 if retriable else 1
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            session = self._session()
            headers = {'Content-Type': 'application/json'}
            headers.update(await self._auth_headers(session, allow_anonymous=allow_anonymous))
            try:
                async with session.request(method, url, params=params, json=json_body, headers=headers) as response:
                    if retriable and attempt < max_attempts and response.status in RETRIABLE_STATUS_CODES:
                        await self._sleep_backoff(attempt, reason=f'HTTP {response.status} on {method} {path}')
                        continue
                    data = await _parse_api_result(response, taihu=self._uses_tai_pat())
                    if attempt > 1:
                        logger.info(
                            'dataset io request recovered after retry: %s %s attempt=%d/%d',
                            method,
                            path,
                            attempt,
                            max_attempts,
                        )
                    return data
            except aiohttp.ClientError as e:
                last_exc = e
                if not retriable or attempt >= max_attempts:
                    raise DatasetClientError(f'Request failed: {e}') from e
                await self._sleep_backoff(attempt, reason=f'network error: {e}')
                continue

        # Unreachable in practice — the loop either returns or raises above.
        # Kept as a defensive fallback so type checkers see a definite exit.
        assert last_exc is not None
        raise DatasetClientError(f'Request failed after {max_attempts} attempts: {last_exc}')

    async def _sleep_backoff(self, attempt: int, reason: str) -> None:
        """Sleep between retry attempts with exponential backoff + jitter.

        Extracted so tests can shrink both backoff and jitter to 0 without
        monkeypatching asyncio.sleep.
        """
        sleep_s = min(
            self._retry_backoff_base * (2 ** (attempt - 1)) + random.random() * self._retry_backoff_jitter,
            self._retry_backoff_max,
        )
        logger.info('dataset io request attempt %d failed (%s); retrying in %.3fs', attempt, reason, sleep_s)
        if sleep_s > 0:
            await asyncio.sleep(sleep_s)

    async def _auth_headers(self, session: aiohttp.ClientSession, *, allow_anonymous: bool = False) -> dict[str, str]:
        if self.auth_token is not None:
            return {'secure-authentication': self.auth_token}
        if self.authenticator is None:
            try:
                self.authenticator = build_default_authenticator()
            except DatasetConfigurationError:
                if allow_anonymous:
                    return {'secure-authentication': ''}
                raise
        try:
            return await self.authenticator.get_headers(session)
        except (DatasetAuthenticationError, DatasetConfigurationError):
            if allow_anonymous:
                return {'secure-authentication': ''}
            raise

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError('DatasetIoClient requires an aiohttp session')
        return self.session


async def _parse_api_result(response: aiohttp.ClientResponse, *, taihu: bool = False) -> dict[str, Any]:
    try:
        body = await response.json(content_type=None)
    except Exception as exc:
        text = await response.text()
        if response.status == 401:
            raise DatasetClientError(
                format_dataset_auth_error(f'IO HTTP 401: {text[:300]}', taihu=taihu),
                http_status=response.status,
            ) from exc
        if response.status < 200 or response.status >= 300:
            raise DatasetClientError(f'HTTP {response.status}: {text[:300]}') from exc
        raise DatasetClientError(f'Non-JSON response (HTTP {response.status})') from exc

    trace_id = body.get('trace_id') if isinstance(body, dict) else None
    if response.status == 401:
        message = body.get('message') if isinstance(body, dict) else None
        raise DatasetClientError(
            format_dataset_auth_error(f'IO HTTP 401: {message or "Unauthorized"}', taihu=taihu),
            trace_id=trace_id,
            http_status=response.status,
        )
    if response.status < 200 or response.status >= 300:
        message = body.get('message') if isinstance(body, dict) else str(body)
        raise DatasetClientError(f'HTTP {response.status}: {message}', trace_id=trace_id)
    if isinstance(body, dict) and body.get('code', 0) != 0:
        raise DatasetClientError(
            f'API error (code={body.get("code")}): {body.get("message")}',
            trace_id=trace_id,
        )
    data = body.get('data', {}) if isinstance(body, dict) else {}
    return data if isinstance(data, dict) else {'value': data}


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _extract_upload_url(data: dict[str, Any], key: str) -> str:
    urls = data.get('urls')
    if isinstance(urls, dict):
        value = urls.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(urls, list):
        for entry in urls:
            if not isinstance(entry, dict):
                continue
            entry_key = entry.get('file') or entry.get('path') or entry.get('key')
            value = entry.get('url') or entry.get('signed_url') or entry.get('signedUrl')
            if entry_key == key and isinstance(value, str) and value:
                return value
    raise DatasetClientError(f'uploads:sign response did not include a URL for {key!r}')
