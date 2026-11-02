"""Small asynchronous Tencent COS client built on aiohttp."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import mimetypes
import os
import re
import stat
import threading
import time
import uuid

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from ssl import SSLContext
from typing import Any, TypeVar
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree

import aiofiles
import aiohttp

from pydantic import BaseModel, ConfigDict, SecretStr

from nexus.utils.paths import COS_AUTH_KEY_FILE, resolve_workdir_path


_SIGNABLE_HEADERS = {
    'cache-control',
    'content-disposition',
    'content-encoding',
    'content-length',
    'content-md5',
    'content-type',
    'expect',
    'expires',
    'host',
    'if-match',
    'if-modified-since',
    'if-none-match',
    'if-unmodified-since',
    'origin',
    'pic-operations',
    'range',
    'transfer-encoding',
}
_AUTH_QUERY_KEYS = {
    'q-sign-algorithm',
    'q-ak',
    'q-sign-time',
    'q-key-time',
    'q-header-list',
    'q-url-param-list',
    'q-signature',
}
_T = TypeVar('_T')
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
DEFAULT_MULTIPART_PART_SIZE = 256 * _MIB
_MIN_MULTIPART_PART_SIZE = _MIB
_MAX_MULTIPART_PART_SIZE = 5 * _GIB
_MAX_MULTIPART_PARTS = 10_000
_STREAM_CHUNK_SIZE = _MIB
_CONTENT_RANGE_PATTERN = re.compile(r'^bytes (\d+)-(\d+)/(\d+)$')
# COS returns HTTP 400/403 (not 408/429/5xx) for some transient conditions that
# are still safe to retry. The most common under load are:
#   * ``RequestTimeout``    -- COS aborts when the client upload stalls or the
#                              socket idles past the server's write window (400).
#   * ``UserNetworkTooSlow``-- COS aborts when the observed upload throughput
#                              stays below its minimum-rate threshold. Under very
#                              high fan-out (tens of thousands of simultaneous
#                              20MB PUTs) the client egress bandwidth is shared
#                              thin, a single PUT dips under the floor and COS
#                              kills it. This is transient/contention-driven:
#                              retrying (with exponential backoff, off the
#                              congestion peak) succeeds. Without this code the
#                              deep retry budget below never triggers -- the
#                              first failure raises immediately (observed as
#                              attempt=0 task failures during the 20x2500 run).
#   * ``SlowDown``          -- COS request-rate throttling.
# Retrying these is safe because every COS write here is an idempotent
# whole-object / whole-part overwrite. Matched on the parsed ``<Code>``
# regardless of the accompanying HTTP status.
_RETRYABLE_COS_ERROR_CODES = frozenset({
    'RequestTimeout',
    'RequestTimeTooSkewed',
    'InternalError',
    'ServiceUnavailable',
    'SlowDown',
    'UserNetworkTooSlow',
})
COS_SHARED_VALUE_NAME = 'cos_credentials'
ENV_COS_SECRET_ID = 'COS_SECRET_ID'
ENV_COS_SECRET_KEY = 'COS_SECRET_KEY'
ENV_COS_TOKEN = 'COS_TOKEN'
ENV_COS_KEY_FILE = 'COS_KEY_FILE'
ENV_COS_BUCKET = 'COS_BUCKET'
ENV_COS_REGION = 'COS_REGION'
ENV_COS_ENDPOINT = 'COS_ENDPOINT'
ENV_COS_UPLOAD_PREFIX = 'COS_UPLOAD_PREFIX'
DEFAULT_COS_UPLOAD_PREFIX = 'nexus-upload'
DEFAULT_COS_LINK_EXPIRES = 7 * 24 * 3600
# ``{bucket}-{appid}.{cos|cos-internal}.{region}.{myqcloud.com|tencentcos.cn}``. The service label and
# the DNS suffix vary independently: ``cos.<region>.tencentcos.cn`` is a real combination in use.
_COS_HOST_PATTERN = re.compile(
    r'^(?P<bucket>[a-z0-9][a-z0-9.-]*?)\.'
    r'(?P<service>cos-internal|cos)\.'
    r'(?P<region>[a-z0-9-]+)\.'
    r'(?P<suffix>myqcloud\.com|tencentcos\.cn)$'
)
COS_URL_SCHEME = 'cos'


class CosClientError(RuntimeError):
    """Base error raised by the COS client."""


class _CosSourceChangedError(CosClientError):
    """A local source file no longer matches the state captured at upload start."""


class CosConfigurationError(CosClientError):
    """COS configuration or credentials are missing or invalid."""


class CosServiceError(CosClientError):
    """An error response returned by COS."""

    def __init__(
        self,
        *,
        method: str,
        key: str,
        status: int,
        code: str | None = None,
        message: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.method = method
        self.key = key
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id
        detail = code or message or f'HTTP {status}'
        suffix = f' (request_id={request_id})' if request_id else ''
        super().__init__(f'COS {method} {key!r} failed: {detail}{suffix}')


class CosCredentials(BaseModel):
    """COS credentials kept in memory and redacted from representations."""

    model_config = ConfigDict(frozen=True)

    secret_id: SecretStr
    secret_key: SecretStr
    token: SecretStr | None = None

    def to_shared_payload(self) -> dict[str, str]:
        payload = {
            'secret_id': self.secret_id.get_secret_value(),
            'secret_key': self.secret_key.get_secret_value(),
        }
        if self.token is not None:
            payload['token'] = self.token.get_secret_value()
        return payload

    @classmethod
    def from_shared_payload(cls, payload: Mapping[str, Any]) -> CosCredentials:
        secret_id = payload.get('secret_id') or payload.get('SecretId') or payload.get('secretId')
        secret_key = payload.get('secret_key') or payload.get('SecretKey') or payload.get('secretKey')
        token = payload.get('token') or payload.get('Token') or payload.get('session_token')
        if not isinstance(secret_id, str) or not secret_id.strip():
            raise CosConfigurationError('COS credentials are missing secret_id')
        if not isinstance(secret_key, str) or not secret_key.strip():
            raise CosConfigurationError('COS credentials are missing secret_key')
        if token is not None and not isinstance(token, str):
            raise CosConfigurationError('COS credential token must be a string')
        return cls(secret_id=secret_id.strip(), secret_key=secret_key.strip(), token=token.strip() if token else None)


_CACHED_COS_CREDENTIALS: CosCredentials | None = None
_CACHED_COS_CREDENTIALS_LOCK = threading.Lock()


def load_cos_credentials(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    token: str | None = None,
    key_file: str | Path | None = None,
) -> CosCredentials:
    """Load COS credentials from environment variables or ``~/.nexus/auth/cos/key``."""

    explicit = secret_id is not None or secret_key is not None or token is not None or key_file is not None
    if explicit:
        return _load_cos_credentials_uncached(
            secret_id=secret_id,
            secret_key=secret_key,
            token=token,
            key_file=key_file,
        )

    global _CACHED_COS_CREDENTIALS
    if _CACHED_COS_CREDENTIALS is not None:
        return _CACHED_COS_CREDENTIALS
    with _CACHED_COS_CREDENTIALS_LOCK:
        if _CACHED_COS_CREDENTIALS is None:
            _CACHED_COS_CREDENTIALS = _load_cos_credentials_uncached()
        return _CACHED_COS_CREDENTIALS


def clear_cos_credentials_cache() -> None:
    """Clear the process-local implicit COS credential cache."""

    global _CACHED_COS_CREDENTIALS
    with _CACHED_COS_CREDENTIALS_LOCK:
        _CACHED_COS_CREDENTIALS = None


def _load_cos_credentials_uncached(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    token: str | None = None,
    key_file: str | Path | None = None,
) -> CosCredentials:
    env_secret_id = secret_id if secret_id is not None else os.getenv(ENV_COS_SECRET_ID)
    env_secret_key = secret_key if secret_key is not None else os.getenv(ENV_COS_SECRET_KEY)
    env_token = token if token is not None else os.getenv(ENV_COS_TOKEN)
    if env_secret_id or env_secret_key:
        if not env_secret_id or not env_secret_key:
            raise CosConfigurationError(f'{ENV_COS_SECRET_ID} and {ENV_COS_SECRET_KEY} must be set together')
        return CosCredentials.from_shared_payload({
            'secret_id': env_secret_id,
            'secret_key': env_secret_key,
            'token': env_token,
        })

    path_value = key_file if key_file is not None else os.getenv(ENV_COS_KEY_FILE)
    path = Path(path_value).expanduser() if path_value else COS_AUTH_KEY_FILE
    if not path.is_file():
        raise CosConfigurationError(
            f'COS credentials are missing: set {ENV_COS_SECRET_ID} and {ENV_COS_SECRET_KEY}, '
            f'or create {COS_AUTH_KEY_FILE}'
        )
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except OSError as exc:
        raise CosConfigurationError(f'Failed to read COS credential file {path}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise CosConfigurationError(f'Invalid COS credential JSON in {path}') from exc
    if not isinstance(payload, dict):
        raise CosConfigurationError(f'Invalid COS credential JSON in {path}: expected an object')
    if env_token:
        payload['token'] = env_token
    return CosCredentials.from_shared_payload(payload)


async def resolve_cos_credentials(
    *,
    secret_id: str | None = None,
    secret_key: str | None = None,
    token: str | None = None,
    key_file: str | Path | None = None,
) -> CosCredentials:
    """Load COS credentials from arguments, the Runtime bridge, or the local environment.

    Works both inside a sandbox and on the Host without the caller knowing which it is: when a
    Socket.IO bridge is attached the credentials are requested from the Host, otherwise they are
    read locally. A :class:`CosConfigurationError` raised from inside a sandbox therefore means the
    *Host* has no COS credentials configured.
    """

    if secret_id is not None or secret_key is not None or token is not None or key_file is not None:
        return load_cos_credentials(secret_id=secret_id, secret_key=secret_key, token=token, key_file=key_file)

    from nexus.runtime.context import get_runtime_control

    control = get_runtime_control()
    if control is not None and getattr(control, 'sio', None) is not None:
        # Never require_runtime_control(): it raises on the Host. A bridge failure falls through to
        # the local loader, which reproduces the same problem with an actionable message.
        with suppress(Exception):
            return await control.get_shared_value(
                COS_SHARED_VALUE_NAME,
                CosCredentials,
                decode=CosCredentials.from_shared_payload,
            )
    return load_cos_credentials()


@dataclass(frozen=True)
class CosConfig:
    """Connection configuration for one COS bucket."""

    bucket: str
    region: str | None = None
    endpoint: str | None = None
    use_internal_endpoint: bool = True
    scheme: str = 'http'
    timeout: float | None = 300  # Maximum idle time between response chunks, not a total transfer limit.
    connect_timeout: float = 30
    max_retries: int = 3
    retry_backoff: float = 0.25
    verify_ssl: bool | SSLContext = True
    large_file_threshold: int = DEFAULT_MULTIPART_PART_SIZE
    part_size: int = DEFAULT_MULTIPART_PART_SIZE
    max_concurrency: int = 5

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError('COS bucket must be non-empty')
        if self.endpoint is None and not self.region:
            raise ValueError('COS region is required when endpoint is not set')
        if self.scheme not in {'http', 'https'}:
            raise ValueError('COS scheme must be http or https')
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError('COS timeout must be greater than zero')
        if self.connect_timeout <= 0:
            raise ValueError('COS connect_timeout must be greater than zero')
        if self.max_retries < 0:
            raise ValueError('COS max_retries must not be negative')
        if self.retry_backoff < 0:
            raise ValueError('COS retry_backoff must not be negative')
        if self.large_file_threshold <= 0:
            raise ValueError('COS large_file_threshold must be greater than zero')
        if self.part_size < _MIN_MULTIPART_PART_SIZE:
            raise ValueError(f'COS part_size must be at least {_MIN_MULTIPART_PART_SIZE} bytes')
        if self.part_size > _MAX_MULTIPART_PART_SIZE:
            raise ValueError(f'COS part_size must not exceed {_MAX_MULTIPART_PART_SIZE} bytes')
        if self.max_concurrency <= 0:
            raise ValueError('COS max_concurrency must be greater than zero')

    @property
    def origin(self) -> str:
        """Return the bucket-qualified endpoint origin."""
        if self.endpoint is None:
            service = 'cos-internal' if self.use_internal_endpoint else 'cos'
            suffix = 'tencentcos.cn' if self.use_internal_endpoint else 'myqcloud.com'
            return f'{self.scheme}://{self.bucket}.{service}.{self.region}.{suffix}'

        endpoint = self.endpoint.format(bucket=self.bucket, region=self.region or '')
        if '://' not in endpoint:
            endpoint = f'{self.scheme}://{endpoint}'
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError('COS endpoint must be an HTTP(S) origin')
        if parsed.path not in {'', '/'} or parsed.query or parsed.fragment:
            raise ValueError('COS endpoint must not contain a path, query, or fragment')
        return urlunsplit((parsed.scheme, parsed.netloc, '', '', ''))


@dataclass(frozen=True)
class CosObject:
    """Metadata returned after a successful object transfer."""

    key: str
    size: int
    etag: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class CosLink:
    """An uploaded object together with a pre-signed URL that can fetch it."""

    url: str
    key: str
    bucket: str
    size: int
    etag: str | None = None
    expires: int = DEFAULT_COS_LINK_EXPIRES
    """Validity window requested for :attr:`url`, in seconds. Not a guarantee: a URL signed with
    temporary credentials also dies when the STS token expires, which may be sooner."""


@dataclass(frozen=True)
class CosLocation:
    """One object addressed by a COS URL."""

    bucket: str
    key: str
    region: str | None = None
    origin: str | None = None
    """Bucket-qualified origin taken verbatim from the URL, or ``None`` for ``cos://`` links."""
    internal: bool = False


@dataclass(frozen=True)
class _LocalFileState:
    size: int
    mtime_ns: int
    device: int
    inode: int


class CosV5Signer:
    """Pure-Python implementation of the COS XML API V5 signature."""

    def __init__(self, secret_id: str, secret_key: str, *, token: str | None = None) -> None:
        if not secret_id or not secret_key:
            raise ValueError('COS secret_id and secret_key must be non-empty')
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._token = token

    def authorization(
        self,
        method: str,
        path: str,
        *,
        host: str,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, object] | None = None,
        expires: int = 10_000,
        timestamp: int | None = None,
    ) -> str:
        """Build an Authorization header value for one COS request."""
        fields = self._signature_fields(
            method,
            path,
            host=host,
            params=params,
            headers=headers,
            expires=expires,
            timestamp=timestamp,
        )
        return '&'.join(f'{key}={value}' for key, value in fields)

    def presign_url(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, object] | None = None,
        expires: int = 3600,
        timestamp: int | None = None,
    ) -> str:
        """Place the V5 signature in a URL query string."""
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError('COS URL must be HTTP(S)')
        if parsed.fragment:
            raise ValueError('COS URL must not contain a fragment')

        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        for key, value in (params or {}).items():
            if key in query:
                raise ValueError(f'Duplicate COS query parameter: {key}')
            query[key] = _string_value(value)
        if self._token:
            query.setdefault('x-cos-security-token', self._token)
        conflict = _AUTH_QUERY_KEYS.intersection(query)
        if conflict:
            names = ', '.join(sorted(conflict))
            raise ValueError(f'COS URL already contains signature parameters: {names}')

        fields = self._signature_fields(
            method,
            unquote(parsed.path),
            host=parsed.netloc,
            params=query,
            headers=headers,
            expires=expires,
            timestamp=timestamp,
        )
        encoded_query = urlencode(
            [*query.items(), *fields],
            doseq=False,
            safe='-_.~',
            quote_via=quote,
        )
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, encoded_query, ''))

    def _signature_fields(
        self,
        method: str,
        path: str,
        *,
        host: str,
        params: Mapping[str, object] | None,
        headers: Mapping[str, object] | None,
        expires: int,
        timestamp: int | None,
    ) -> list[tuple[str, str]]:
        if not method:
            raise ValueError('COS request method must be non-empty')
        if not path.startswith('/'):
            path = f'/{path}'
        if not host:
            raise ValueError('COS request host must be non-empty')
        if expires <= 0:
            raise ValueError('COS signature expiry must be greater than zero')

        signable_headers = _signable_headers(headers or {})
        signable_headers['host'] = host
        canonical_params, param_names = _canonical_mapping(params or {}, lowercase_keys=True)
        canonical_headers, header_names = _canonical_mapping(signable_headers, lowercase_keys=True)
        canonical_request = f'{method.lower()}\n{path}\n{canonical_params}\n{canonical_headers}\n'

        now = int(time.time()) if timestamp is None else timestamp
        key_time = f'{now - 60};{now + expires}'
        request_hash = hashlib.sha1(canonical_request.encode()).hexdigest()
        string_to_sign = f'sha1\n{key_time}\n{request_hash}\n'
        sign_key = hmac.new(self._secret_key.encode(), key_time.encode(), hashlib.sha1).hexdigest()
        signature = hmac.new(sign_key.encode(), string_to_sign.encode(), hashlib.sha1).hexdigest()
        return [
            ('q-sign-algorithm', 'sha1'),
            ('q-ak', self._secret_id),
            ('q-sign-time', key_time),
            ('q-key-time', key_time),
            ('q-header-list', ';'.join(header_names)),
            ('q-url-param-list', ';'.join(param_names)),
            ('q-signature', signature),
        ]


class AsyncCosClient:
    """Streaming COS upload and download client using aiohttp."""

    def __init__(
        self,
        config: CosConfig,
        credentials: CosCredentials,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config
        self.credentials = credentials
        self._session = session
        self._owns_session = False
        self._signer = CosV5Signer(
            credentials.secret_id.get_secret_value(),
            credentials.secret_key.get_secret_value(),
            token=credentials.token.get_secret_value() if credentials.token is not None else None,
        )

    @classmethod
    async def from_runtime(
        cls,
        config: CosConfig,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> AsyncCosClient:
        """Create a client from COS credentials requested once by the Runtime."""

        from nexus.runtime.context import require_runtime_control

        credentials = await require_runtime_control().get_shared_value(
            COS_SHARED_VALUE_NAME,
            CosCredentials,
            decode=CosCredentials.from_shared_payload,
        )
        return cls(config, credentials, session=session)

    async def __aenter__(self) -> AsyncCosClient:
        self._get_session()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close a session created by this client."""
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()
        if self._owns_session:
            self._session = None
            self._owns_session = False

    def presign_url(
        self,
        key: str,
        *,
        method: str = 'GET',
        expires: int = 3600,
        params: Mapping[str, object] | None = None,
        headers: Mapping[str, object] | None = None,
    ) -> str:
        """Generate a pre-signed URL for one object."""
        return self._signer.presign_url(
            method,
            self._object_url(key),
            params=params,
            headers=headers,
            expires=expires,
        )

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        *,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> CosObject:
        """Upload an in-memory object with one PUT request."""
        request_headers = _upload_headers(len(data), content_type, headers)

        async def operation() -> CosObject:
            return await self._upload_once(data, key, request_headers, len(data))

        return await self._with_retries('PUT', key, operation)

    async def upload_file(
        self,
        local_path: str | Path,
        key: str,
        *,
        content_type: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> CosObject:
        """Upload a file, using multipart upload for large files."""
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size > self.config.large_file_threshold:
            return await self._multipart_upload_file(path, key, size, content_type, headers)

        request_headers = _upload_headers(size, content_type, headers)

        async def operation() -> CosObject:
            with path.open('rb') as body:
                return await self._upload_once(body, key, request_headers, size)

        return await self._with_retries('PUT', key, operation)

    async def download_bytes(self, key: str, *, headers: Mapping[str, str] | None = None) -> bytes:
        """Download an object into memory."""

        async def operation() -> bytes:
            response = await self._open_download(key, headers)
            async with response:
                return await response.read()

        return await self._with_retries('GET', key, operation)

    async def stat(self, key: str) -> CosObject:
        """Read object size and identity without downloading its content."""
        return await self._head_object(key, None)

    async def bucket_versioning(self) -> str | None:
        """Read bucket versioning before relying on forbid-overwrite semantics."""
        response = await self._request('GET', '', params={'versioning': ''})
        async with response:
            body = await response.content.read(64 * 1024 + 1)
        if len(body) > 64 * 1024:
            raise CosClientError('COS versioning response exceeds size limit')
        root = ElementTree.fromstring(body)
        if root.tag.rsplit('}', 1)[-1] != 'VersioningConfiguration':
            raise CosClientError('Invalid COS versioning response')
        for element in root.iter():
            if element.tag.rsplit('}', 1)[-1] == 'Status':
                return element.text
        return None

    async def read_range(self, key: str, start: int, end: int, *, etag: str, size: int) -> bytes:
        """Read an exclusive-end range pinned to an observed object identity."""
        if not etag or start < 0 or end < start or end > size:
            raise ValueError('Invalid COS range or missing object identity')
        if start == end:
            return b''

        async def operation() -> bytes:
            response = await self._open_download(key, {'Range': f'bytes={start}-{end - 1}', 'If-Match': etag})
            async with response:
                expected = f'bytes {start}-{end - 1}/{size}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise CosClientError('COS returned an invalid Content-Range')
                if response.headers.get('ETag') != etag:
                    raise CosClientError('COS object identity changed while reading')
                data = await response.content.readexactly(end - start)
                if await response.content.read(1):
                    raise CosClientError('COS range response exceeded requested length')
                return data

        return await self._with_retries('GET', key, operation)

    async def download_file(
        self,
        key: str,
        local_path: str | Path,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> CosObject:
        """Download an object to a local file, using ranges for large objects."""
        target = Path(local_path)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')

        try:
            if _has_header(headers, 'range'):
                result = await self._download_file_once(key, temporary, headers)
            else:
                metadata = await self._head_object(key, headers)
                if metadata.size > self.config.large_file_threshold:
                    result = await self._download_file_ranges(key, temporary, metadata, headers)
                else:
                    result = await self._download_file_once(key, temporary, headers)
            await asyncio.to_thread(os.replace, temporary, target)
            return result
        finally:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)

    async def delete_object(self, key: str) -> CosObject:
        """Delete one object from COS."""

        async def operation() -> CosObject:
            response = await self._request('DELETE', key)
            async with response:
                await response.read()
                return _object_metadata(key, 0, response.headers)

        return await self._with_retries('DELETE', key, operation)

    async def _upload_once(
        self,
        data: object,
        key: str,
        headers: Mapping[str, str],
        size: int,
    ) -> CosObject:
        response = await self._request('PUT', key, headers=headers, data=data)
        async with response:
            await response.read()
            return _object_metadata(key, size, response.headers)

    async def _multipart_upload_file(
        self,
        path: Path,
        key: str,
        size: int,
        content_type: str | None,
        headers: Mapping[str, str] | None,
    ) -> CosObject:
        source_state = _local_file_state(path)
        if source_state.size != size:
            raise _CosSourceChangedError(f'Local file {path} changed before multipart upload started')
        upload_id = await self._create_multipart_upload(key, content_type, headers)

        async def upload_part(part_number: int, offset: int, length: int) -> str:
            _ensure_file_unchanged(path, source_state)
            return await self._upload_file_part(path, key, upload_id, part_number, offset, length, headers)

        async def complete(uploaded_parts: list[tuple[int, str]]) -> CosObject:
            _ensure_file_unchanged(path, source_state)
            result = await self._complete_multipart_upload(key, upload_id, size, uploaded_parts, headers=headers)
            _ensure_file_unchanged(path, source_state)
            return result

        return await _run_multipart_upload(
            part_ranges=_part_ranges(size, self.config.part_size),
            max_concurrency=self.config.max_concurrency,
            upload_part=upload_part,
            complete=complete,
            abort=lambda: self._abort_multipart_upload(key, upload_id),
        )

    async def _create_multipart_upload(
        self,
        key: str,
        content_type: str | None,
        headers: Mapping[str, str] | None,
    ) -> str:
        request_headers = _object_upload_headers(content_type, headers)

        async def operation() -> str:
            response = await self._request('POST', key, params={'uploads': ''}, headers=request_headers)
            async with response:
                body = await response.read()
            upload_id = _xml_value(body, 'UploadId')
            if not upload_id:
                raise CosClientError(f'COS multipart upload for {key!r} returned no UploadId')
            return upload_id

        return await self._with_retries('POST', key, operation)

    async def _upload_file_part(
        self,
        path: Path,
        key: str,
        upload_id: str,
        part_number: int,
        offset: int,
        length: int,
        headers: Mapping[str, str] | None,
    ) -> str:
        request_headers = _multipart_part_headers(length, headers)
        params = {'partNumber': part_number, 'uploadId': upload_id}

        async def operation() -> str:
            response = await self._request(
                'PUT',
                key,
                params=params,
                headers=request_headers,
                data=_iter_file_segment(path, offset, length),
            )
            async with response:
                await response.read()
                etag = response.headers.get('ETag')
            if not etag:
                raise CosClientError(f'COS multipart upload part {part_number} for {key!r} returned no ETag')
            return etag

        return await self._with_retries('PUT', key, operation)

    async def _complete_multipart_upload(
        self,
        key: str,
        upload_id: str,
        size: int,
        uploaded_parts: list[tuple[int, str]],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> CosObject:
        body = _complete_multipart_body(uploaded_parts)
        request_headers = {'Content-Length': str(len(body)), 'Content-Type': 'application/xml'}
        for name, value in (headers or {}).items():
            if name.lower() == 'x-cos-forbid-overwrite':
                request_headers[name] = value

        async def operation() -> CosObject:
            response = await self._request(
                'POST',
                key,
                params={'uploadId': upload_id},
                headers=request_headers,
                data=body,
            )
            async with response:
                response_body = await response.read()
                etag = response.headers.get('ETag') or _xml_value(response_body, 'ETag')
                return CosObject(
                    key=key,
                    size=size,
                    etag=etag,
                    request_id=response.headers.get('x-cos-request-id'),
                )

        return await self._with_retries('POST', key, operation)

    async def _abort_multipart_upload(self, key: str, upload_id: str) -> None:
        response = await self._request('DELETE', key, params={'uploadId': upload_id})
        async with response:
            await response.read()

    async def _head_object(
        self,
        key: str,
        headers: Mapping[str, str] | None,
    ) -> CosObject:
        request_headers = _without_headers(headers, {'range', 'x-cos-traffic-limit'})

        async def operation() -> CosObject:
            response = await self._request('HEAD', key, headers=request_headers)
            async with response:
                size_value = response.headers.get('Content-Length')
                if size_value is None:
                    raise CosClientError(f'COS HEAD {key!r} returned no Content-Length')
                return _object_metadata(key, int(size_value), response.headers)

        return await self._with_retries('HEAD', key, operation)

    async def _download_file_once(
        self,
        key: str,
        temporary: Path,
        headers: Mapping[str, str] | None,
    ) -> CosObject:
        async def operation() -> CosObject:
            response = await self._open_download(key, headers)
            size = 0
            async with response, aiofiles.open(temporary, 'wb') as output:
                async for chunk in response.content.iter_chunked(_STREAM_CHUNK_SIZE):
                    await output.write(chunk)
                    size += len(chunk)
            expected_size = response.headers.get('Content-Length')
            if expected_size is not None and size != int(expected_size):
                raise CosClientError(f'COS GET {key!r} returned {size} bytes, expected {expected_size}')
            return _object_metadata(key, size, response.headers)

        return await self._with_retries('GET', key, operation)

    async def _download_file_ranges(
        self,
        key: str,
        temporary: Path,
        metadata: CosObject,
        headers: Mapping[str, str] | None,
    ) -> CosObject:
        async with aiofiles.open(temporary, 'wb') as output:
            await output.truncate(metadata.size)

        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def download_part(part_number: int, offset: int, length: int) -> None:
            async with semaphore:
                await self._download_file_part(key, temporary, metadata, part_number, offset, length, headers)

        tasks = [
            asyncio.create_task(download_part(part_number, offset, length))
            for part_number, offset, length in _part_ranges(metadata.size, self.config.part_size)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return metadata

    async def _download_file_part(
        self,
        key: str,
        temporary: Path,
        metadata: CosObject,
        part_number: int,
        offset: int,
        length: int,
        headers: Mapping[str, str] | None,
    ) -> None:
        end = offset + length - 1
        request_headers = dict(headers or {})
        request_headers['Range'] = f'bytes={offset}-{end}'
        if metadata.etag and not _has_header(request_headers, 'if-match'):
            request_headers['If-Match'] = metadata.etag

        async def operation() -> None:
            response = await self._open_download(key, request_headers)
            written = 0
            async with response:
                if response.status != 206:
                    await response.read()
                    raise CosClientError(
                        f'COS ranged GET part {part_number} for {key!r} returned HTTP {response.status}, expected 206'
                    )
                content_range = response.headers.get('Content-Range')
                if _parse_content_range(content_range) != (offset, end, metadata.size):
                    await response.read()
                    raise CosClientError(
                        f'COS ranged GET part {part_number} for {key!r} returned invalid Content-Range {content_range!r}'
                    )
                async with aiofiles.open(temporary, 'r+b') as output:
                    await output.seek(offset)
                    async for chunk in response.content.iter_chunked(_STREAM_CHUNK_SIZE):
                        await output.write(chunk)
                        written += len(chunk)
            if written != length:
                raise CosClientError(
                    f'COS ranged GET part {part_number} for {key!r} returned {written} bytes, expected {length}'
                )

        await self._with_retries('GET', key, operation)

    async def _open_download(
        self,
        key: str,
        headers: Mapping[str, str] | None,
    ) -> aiohttp.ClientResponse:
        return await self._request('GET', key, headers=headers)

    async def _request(
        self,
        method: str,
        key: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, object] | None = None,
        data: object = None,
    ) -> aiohttp.ClientResponse:
        bucket_request = key == '' and method == 'GET' and params == {'versioning': ''}
        url = self.config.origin + '/' if bucket_request else self._object_url(key)
        parsed = urlsplit(url)
        request_headers = dict(headers or {})
        _validate_caller_headers(request_headers)
        request_headers['Host'] = parsed.netloc
        if self.credentials.token is not None:
            request_headers['x-cos-security-token'] = self.credentials.token.get_secret_value()
        request_headers['Authorization'] = self._signer.authorization(
            method,
            '/' if bucket_request else _canonical_path(key),
            host=parsed.netloc,
            params=params,
            headers=request_headers,
        )
        request_headers.setdefault('User-Agent', 'nexus-cos-client')

        response = await self._get_session().request(
            method,
            url,
            params={key: _string_value(value) for key, value in (params or {}).items()},
            headers=request_headers,
            data=data,
            allow_redirects=False,
            ssl=self.config.verify_ssl,
            timeout=aiohttp.ClientTimeout(
                total=None,
                connect=self.config.connect_timeout,
                sock_read=self.config.timeout,
            ),
            auto_decompress=False,
        )
        if 200 <= response.status < 300:
            return response
        try:
            body = await response.content.read(64 * 1024)
            error = _service_error(method, key, response.status, response.headers, body)
        finally:
            response.release()
        raise error

    async def _with_retries(
        self,
        method: str,
        key: str,
        operation: Callable[[], Awaitable[_T]],
    ) -> _T:
        return await _with_retries(
            method,
            key,
            operation,
            max_retries=self.config.max_retries,
            retry_backoff=self.config.retry_backoff,
        )

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            if self._session.closed:
                raise RuntimeError('COS aiohttp session is closed')
            return self._session
        self._session = aiohttp.ClientSession()
        self._owns_session = True
        return self._session

    def _object_url(self, key: str) -> str:
        canonical_path = _canonical_path(key)
        encoded_path = quote(canonical_path[1:], safe='/-_.~').replace('./', '.%2F')
        return f'{self.config.origin}/{encoded_path}'


async def upload_presigned_file(
    session: aiohttp.ClientSession,
    local_path: str | Path,
    signed_url: str,
    *,
    key: str,
    content_type: str | None = None,
    extra_headers: Mapping[str, str] | None = None,
    timeout: float | None = 300,
    connect_timeout: float = 30,
    max_retries: int = 3,
    retry_backoff: float = 0.25,
    verify_ssl: bool | SSLContext = True,
    expected_stat: os.stat_result | None = None,
) -> CosObject:
    """Stream a local file to one pre-signed COS PUT URL with retries.

    ``extra_headers`` are forwarded verbatim; callers should pass here any
    headers the URL was signed with (for example the ``content-type`` returned
    by the artifact service) so the signature matches.
    """

    _validate_transfer_options(
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    path = Path(local_path)
    source_state = _local_file_state(path)
    if expected_stat is not None and source_state != _file_state_from_stat(expected_stat):
        raise _CosSourceChangedError(f'Local file {path} changed before upload started')
    _validate_presigned_url(signed_url, field='signed_url')
    caller_headers: dict[str, str] = dict(extra_headers or {})
    if not _has_header(caller_headers, 'expect'):
        caller_headers['Expect'] = ''
    request_headers = _upload_headers(source_state.size, content_type, caller_headers)

    async def operation() -> CosObject:
        _ensure_file_unchanged(path, source_state)
        response = await _request_presigned(
            session,
            'PUT',
            key,
            signed_url,
            headers=request_headers,
            data=_iter_file_segment(path, 0, source_state.size),
            timeout=timeout,
            connect_timeout=connect_timeout,
            verify_ssl=verify_ssl,
        )
        async with response:
            await response.read()
            result = _object_metadata(key, source_state.size, response.headers)
        _ensure_file_unchanged(path, source_state)
        return result

    return await _with_retries(
        'PUT',
        key,
        operation,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )


async def upload_presigned_multipart_file(
    session: aiohttp.ClientSession,
    local_path: str | Path,
    upload: Mapping[str, Any],
    *,
    key: str,
    max_concurrency: int = 5,
    timeout: float | None = 300,
    connect_timeout: float = 30,
    max_retries: int = 3,
    retry_backoff: float = 0.25,
    verify_ssl: bool | SSLContext = True,
    expected_stat: os.stat_result | None = None,
) -> CosObject:
    """Upload a file with multipart URLs initialized and signed by another service.

    ``upload`` must contain ``part_urls``, ``complete_url``, ``abort_url``, and
    ``part_size``. The same streaming, concurrency, retry, completion, and abort
    machinery used by :class:`AsyncCosClient` performs the transfer.
    """

    _validate_transfer_options(
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    abort_url = upload.get('abort_url')

    async def abort() -> None:
        if not isinstance(abort_url, str) or not abort_url:
            return

        async def operation() -> None:
            response = await _request_presigned(
                session,
                'DELETE',
                key,
                abort_url,
                timeout=timeout,
                connect_timeout=connect_timeout,
                verify_ssl=verify_ssl,
            )
            async with response:
                await response.read()

        await _with_retries(
            'DELETE',
            key,
            operation,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

    transfer_started = False
    try:
        path = Path(local_path)
        source_state = _local_file_state(path)
        if expected_stat is not None and source_state != _file_state_from_stat(expected_stat):
            raise _CosSourceChangedError(f'Local file {path} changed before upload started')
        if max_concurrency <= 0:
            raise ValueError('COS max_concurrency must be greater than zero')

        raw_part_size = upload.get('part_size')
        if isinstance(raw_part_size, bool) or not isinstance(raw_part_size, int):
            raise CosClientError('COS multipart signing response has an invalid part_size')
        selected_part_size = select_multipart_part_size(source_state.size, raw_part_size)
        if selected_part_size != raw_part_size:
            raise CosClientError(
                f'COS multipart part_size {raw_part_size} would exceed the {_MAX_MULTIPART_PARTS}-part limit '
                f'for a {source_state.size}-byte file; at least {selected_part_size} bytes is required'
            )

        raw_part_urls = upload.get('part_urls')
        if not isinstance(raw_part_urls, list) or not all(isinstance(url, str) and url for url in raw_part_urls):
            raise CosClientError('COS multipart signing response has invalid part_urls')
        part_urls = list(raw_part_urls)
        part_ranges = _part_ranges(source_state.size, raw_part_size)
        if len(part_urls) != len(part_ranges):
            raise CosClientError(
                f'COS multipart signing response returned {len(part_urls)} part URLs, '
                f'but {len(part_ranges)} are required'
            )

        complete_url = upload.get('complete_url')
        if not isinstance(complete_url, str) or not complete_url:
            raise CosClientError('COS multipart signing response has no complete_url')
        if not isinstance(abort_url, str) or not abort_url:
            raise CosClientError('COS multipart signing response has no abort_url')
        for index, url in enumerate(part_urls, start=1):
            _validate_presigned_url(url, field=f'part_urls[{index}]')
        _validate_presigned_url(complete_url, field='complete_url')
        _validate_presigned_url(abort_url, field='abort_url')

        async def upload_part(part_number: int, offset: int, length: int) -> str:
            part_url = part_urls[part_number - 1]
            request_headers = {'Content-Length': str(length)}

            async def operation() -> str:
                _ensure_file_unchanged(path, source_state)
                response = await _request_presigned(
                    session,
                    'PUT',
                    key,
                    part_url,
                    headers=request_headers,
                    data=_iter_file_segment(path, offset, length),
                    timeout=timeout,
                    connect_timeout=connect_timeout,
                    verify_ssl=verify_ssl,
                )
                async with response:
                    await response.read()
                    etag = response.headers.get('ETag')
                if not etag:
                    raise CosClientError(f'COS multipart upload part {part_number} for {key!r} returned no ETag')
                return etag

            return await _with_retries(
                'PUT',
                key,
                operation,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )

        async def complete(uploaded_parts: list[tuple[int, str]]) -> CosObject:
            _ensure_file_unchanged(path, source_state)
            body = _complete_multipart_body(uploaded_parts)
            request_headers = {'Content-Length': str(len(body)), 'Content-Type': 'application/xml'}

            async def operation() -> CosObject:
                response = await _request_presigned(
                    session,
                    'POST',
                    key,
                    complete_url,
                    headers=request_headers,
                    data=body,
                    timeout=timeout,
                    connect_timeout=connect_timeout,
                    verify_ssl=verify_ssl,
                )
                async with response:
                    response_body = await response.read()
                    error_code = _xml_error_code(response_body)
                    if error_code:
                        raise CosClientError(f'COS multipart completion for {key!r} failed: {error_code}')
                    etag = response.headers.get('ETag') or _xml_value(response_body, 'ETag')
                    return CosObject(
                        key=key,
                        size=source_state.size,
                        etag=etag,
                        request_id=response.headers.get('x-cos-request-id'),
                    )

            result = await _with_retries(
                'POST',
                key,
                operation,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
            _ensure_file_unchanged(path, source_state)
            return result

        transfer_started = True
        return await _run_multipart_upload(
            part_ranges=part_ranges,
            max_concurrency=max_concurrency,
            upload_part=upload_part,
            complete=complete,
            abort=abort,
        )
    except BaseException:
        if not transfer_started:
            with suppress(Exception):
                await asyncio.shield(abort())
        raise


def is_presigned_cos_url(url: str) -> bool:
    """Return whether ``url`` already carries a COS V5 signature in its query string."""

    query = urlsplit(url).query
    return any(key == 'q-signature' for key, _ in parse_qsl(query, keep_blank_values=True))


def parse_cos_url(url: str) -> CosLocation:
    """Resolve a COS URL into the bucket, region, and object key it addresses.

    Accepts the public ``https://<bucket>.cos.<region>.myqcloud.com/<key>`` form, the internal
    ``cos-internal``/``tencentcos.cn`` variants, and the ``cos://<bucket>/<key>`` short form (which
    carries no region). Signature parameters are ignored; any other query parameter is rejected
    rather than silently dropped, because dropping ``versionId`` would return a different object.
    """

    parsed = urlsplit(url)
    if parsed.scheme == COS_URL_SCHEME:
        if parsed.username is not None or parsed.password is not None or ':' in parsed.netloc:
            raise CosClientError(f'COS URL must not contain credentials or a port: {_redacted_url(url)!r}')
        bucket = parsed.netloc
        if not bucket:
            raise CosClientError(f'COS URL is missing a bucket: {_redacted_url(url)!r}')
        _reject_unsupported_query(parsed.query, url)
        return CosLocation(bucket=bucket, key=_url_object_key(parsed.path, url))

    if parsed.scheme not in {'http', 'https'}:
        raise CosClientError(f'COS URL must be http(s) or {COS_URL_SCHEME}://: {_redacted_url(url)!r}')
    if parsed.username is not None or parsed.password is not None:
        raise CosClientError(f'COS URL must not contain credentials: {_redacted_url(url)!r}')

    hostname = parsed.hostname or ''
    match = _COS_HOST_PATTERN.match(hostname)
    if match is None:
        raise CosClientError(f'Not a COS URL: {_redacted_url(url)!r}')
    _reject_unsupported_query(parsed.query, url)
    return CosLocation(
        bucket=match['bucket'],
        key=_url_object_key(parsed.path, url),
        region=match['region'],
        # netloc, not hostname: it keeps any explicit port so the request hits the same origin.
        origin=urlunsplit((parsed.scheme, parsed.netloc, '', '', '')),
        internal=match['service'] == 'cos-internal',
    )


async def upload_to_cos(
    local_path: str | Path,
    key: str | None = None,
    *,
    bucket: str | None = None,
    region: str | None = None,
    endpoint: str | None = None,
    internal: bool = False,
    prefix: str | None = None,
    content_type: str | None = None,
    expires: int = DEFAULT_COS_LINK_EXPIRES,
    link_internal: bool | None = None,
    link_endpoint: str | None = None,
    credentials: CosCredentials | None = None,
    verify_ssl: bool | SSLContext | None = None,
    session: aiohttp.ClientSession | None = None,
) -> CosLink:
    """Upload one local file to COS and return it with a pre-signed download URL.

    ``bucket``/``region``/``endpoint`` fall back to ``$COS_BUCKET``/``$COS_REGION``/``$COS_ENDPOINT``.
    When ``key`` is omitted one is derived as ``<prefix>/<uuid4>/<filename>``. Large files upload in
    parts automatically. The returned URL is signed, so a private bucket can be fetched with no
    credentials — pass it straight to :func:`download_from_cos_url`.

    The link is always signed against the canonical bucket origin, never against ``endpoint``: bytes
    often travel over an internal or accelerated endpoint that the recipient cannot reach. Use
    ``link_internal`` to sign an internal-domain link, or ``link_endpoint`` for a custom domain.
    """

    if expires <= 0:
        raise ValueError('COS link expires must be greater than zero')
    path = Path(resolve_workdir_path(local_path))
    if not path.is_file():
        raise FileNotFoundError(path)

    resolved_bucket = _resolved_option(bucket, ENV_COS_BUCKET)
    resolved_region = _resolved_option(region, ENV_COS_REGION)
    resolved_endpoint = _resolved_option(endpoint, ENV_COS_ENDPOINT)
    if not resolved_bucket:
        raise CosConfigurationError(f'COS bucket is required: pass bucket= or set {ENV_COS_BUCKET}')
    if not resolved_region and not resolved_endpoint:
        raise CosConfigurationError(
            f'COS region is required: pass region= or endpoint=, or set {ENV_COS_REGION} or {ENV_COS_ENDPOINT}'
        )
    # Checked before uploading: the link is signed against the canonical bucket origin, which needs a
    # region even when the bytes travel over a custom endpoint.
    if link_endpoint is not None:
        try:
            _ = _cos_config(
                bucket=resolved_bucket,
                region=resolved_region,
                endpoint=link_endpoint,
                scheme='https',
                allow_placeholders=True,
            ).origin
        except (ValueError, IndexError, KeyError) as exc:
            raise CosConfigurationError(f'Invalid link_endpoint {link_endpoint!r}: {exc}') from exc
    elif not resolved_region:
        raise CosConfigurationError(
            f'COS region is required to sign a download link: pass region= or link_endpoint=, or set {ENV_COS_REGION}'
        )

    object_key = key if key is not None else _derive_cos_key(path, prefix)
    if content_type is None:
        content_type = mimetypes.guess_type(path.name)[0]
    if credentials is None:
        credentials = await resolve_cos_credentials()

    config = _cos_config(
        bucket=resolved_bucket,
        region=resolved_region,
        endpoint=resolved_endpoint,
        use_internal_endpoint=internal,
        scheme='http' if internal else 'https',
        verify_ssl=_default_verify_ssl(verify_ssl, internal=internal),
        allow_placeholders=True,
    )
    async with AsyncCosClient(config, credentials, session=session) as client:
        uploaded = await client.upload_file(path, object_key, content_type=content_type)

    link_url = _presign_link(
        config,
        credentials,
        object_key,
        expires=expires,
        internal=internal if link_internal is None else link_internal,
        link_endpoint=link_endpoint,
    )
    return CosLink(
        url=link_url,
        key=object_key,
        bucket=resolved_bucket,
        size=uploaded.size,
        etag=uploaded.etag,
        expires=expires,
    )


async def download_from_cos_url(
    url: str,
    dest: str | Path | None = None,
    *,
    region: str | None = None,
    endpoint: str | None = None,
    credentials: CosCredentials | None = None,
    verify_ssl: bool | SSLContext | None = None,
    timeout: float | None = 300,
    connect_timeout: float = 30,
    max_retries: int = 3,
    retry_backoff: float = 0.25,
    session: aiohttp.ClientSession | None = None,
) -> Path:
    """Download the object addressed by ``url`` and return the local path it was written to.

    Three link forms are accepted. A pre-signed URL is fetched as-is and needs no credentials; a
    bare URL on a private bucket is signed (large objects still download as parallel ranges); and
    ``cos://<bucket>/<key>`` resolves its region from ``region=`` or ``$COS_REGION``.

    ``dest`` may be a file path, an existing directory, or omitted — in the latter two cases the
    file name comes from the object key. Relative paths resolve against ``$NEXUS_WORKDIR``.
    """

    if is_presigned_cos_url(url):
        # Already signed: the signature covers the exact method and query, so the URL is fetched
        # verbatim. Deliberately not parsed as a COS host either — a pre-signed link may be served
        # from a custom domain, and nothing here needs the bucket or region.
        parsed = urlsplit(url)
        key = _url_object_key(parsed.path, url)
        target = _resolve_download_target(dest, key)
        owns_session = session is None
        http = session if session is not None else aiohttp.ClientSession()
        try:
            await download_presigned_file(
                http,
                url,
                target,
                key=key,
                timeout=timeout,
                connect_timeout=connect_timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                verify_ssl=_default_verify_ssl(verify_ssl, internal=_is_internal_host(parsed.hostname or '')),
            )
        finally:
            if owns_session:
                await http.close()
        return target

    location = parse_cos_url(url)
    target = _resolve_download_target(dest, location.key)

    resolved_region = location.region or _resolved_option(region, ENV_COS_REGION)
    # The host named in the URL outranks $COS_ENDPOINT: an ambient env var must not silently redirect
    # a fully-qualified request. Only an explicit endpoint= argument overrides it.
    resolved_endpoint = endpoint or location.origin or _resolved_option(None, ENV_COS_ENDPOINT)
    if not resolved_region and not resolved_endpoint:
        raise CosConfigurationError(
            f'COS region is required to resolve {_redacted_url(url)!r}: pass region= or set {ENV_COS_REGION}'
        )
    if credentials is None:
        credentials = await resolve_cos_credentials()

    config = _cos_config(
        bucket=location.bucket,
        region=resolved_region,
        endpoint=resolved_endpoint,
        use_internal_endpoint=location.internal,
        scheme='http' if location.internal else 'https',
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        verify_ssl=_default_verify_ssl(verify_ssl, internal=location.internal),
    )
    async with AsyncCosClient(config, credentials, session=session) as client:
        await client.download_file(location.key, target)
    return target


async def download_presigned_file(
    session: aiohttp.ClientSession,
    signed_url: str,
    local_path: str | Path,
    *,
    key: str,
    headers: Mapping[str, str] | None = None,
    timeout: float | None = 300,
    connect_timeout: float = 30,
    max_retries: int = 3,
    retry_backoff: float = 0.25,
    verify_ssl: bool | SSLContext = True,
) -> CosObject:
    """Stream one pre-signed COS GET URL to a local file with retries.

    A single streaming GET, never a ranged one: the signature covers the HTTP method, so probing the
    object with HEAD first would be rejected.
    """

    _validate_transfer_options(
        timeout=timeout,
        connect_timeout=connect_timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
    )
    _validate_presigned_url(signed_url, field='signed_url')
    # urlsplit defers port parsing, so an out-of-range port only fails later inside aiohttp — as an
    # InvalidURL whose text is the full signed URL, which _with_retries would put in its message.
    try:
        _ = urlsplit(signed_url).port
    except ValueError as exc:
        raise CosClientError(f'COS URL has an invalid port: {_redacted_url(signed_url)!r}') from exc
    target = Path(local_path)
    await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{uuid.uuid4().hex}.part')

    async def operation() -> CosObject:
        response = await _request_presigned(
            session,
            'GET',
            key,
            signed_url,
            headers=headers,
            timeout=timeout,
            connect_timeout=connect_timeout,
            verify_ssl=verify_ssl,
        )
        async with response:
            written = 0
            # 'wb' truncates, so a retried attempt replaces the partial body instead of appending.
            async with aiofiles.open(temporary, 'wb') as sink:
                async for chunk in response.content.iter_chunked(_STREAM_CHUNK_SIZE):
                    await sink.write(chunk)
                    written += len(chunk)
            status = response.status
            expected_size = response.headers.get('Content-Length')
            result = _object_metadata(key, written, response.headers)

        # Validated after the response is released: raising inside the context would leave the
        # connection unread and the next attempt waiting on it.
        #
        # A 206 means only part of the object arrived; without an explicit Range the caller asked for
        # the whole thing, so accepting it would silently truncate the file.
        if status == 206 and not _has_header(headers, 'range'):
            raise CosClientError(f'COS GET {key!r} returned a partial response for a whole-object request')
        if expected_size is not None and written != int(expected_size):
            raise CosClientError(f'COS GET {key!r} returned {written} bytes, expected {expected_size}')
        return result

    try:
        # ``key`` and never the URL: _with_retries embeds it in its error message, and a signed URL
        # would leak q-signature into logs.
        result = await _with_retries('GET', key, operation, max_retries=max_retries, retry_backoff=retry_backoff)
        await asyncio.to_thread(os.replace, temporary, target)
        return result
    finally:
        await asyncio.to_thread(temporary.unlink, missing_ok=True)


def _reject_unsupported_query(query: str, url: str) -> None:
    # Signature parameters (and the STS token that travels with them) are meaningful only on the
    # pre-signed path, which never reaches here. Anything else (versionId, response-*) would change
    # which bytes come back, so refuse instead of silently fetching a different object.
    allowed = _AUTH_QUERY_KEYS | {'x-cos-security-token'}
    unsupported = sorted(name for name, _ in parse_qsl(query, keep_blank_values=True) if name not in allowed)
    if unsupported:
        raise CosClientError(
            f'COS URL query parameters are not supported: {", ".join(unsupported)} in {_redacted_url(url)!r}'
        )


def _redacted_url(url: str) -> str:
    """Strip signature and token parameters so a URL can appear in an error message or log."""

    parsed = urlsplit(url)
    if not parsed.query:
        return url
    kept = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in _AUTH_QUERY_KEYS and key != 'x-cos-security-token'
    ]
    redacted = urlencode(kept) if kept else ''
    if len(kept) != len(parse_qsl(parsed.query, keep_blank_values=True)):
        redacted = f'{redacted}&<redacted>' if redacted else '<redacted>'
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, redacted, ''))


def _url_object_key(path: str, url: str) -> str:
    # Decoded exactly once: _object_url re-encodes the key, so a surviving %20 would become %2520.
    key = unquote(path).lstrip('/')
    if not key:
        raise CosClientError(f'COS URL is missing an object key: {_redacted_url(url)!r}')
    return key


def _resolved_option(value: str | None, env_name: str) -> str | None:
    if value is not None:
        return value or None
    return os.getenv(env_name) or None


def _default_verify_ssl(verify_ssl: bool | SSLContext | None, *, internal: bool) -> bool | SSLContext:
    # Internal COS endpoints serve certificates that do not validate against the public trust store.
    return (not internal) if verify_ssl is None else verify_ssl


def _cos_config(*, endpoint: str | None, allow_placeholders: bool = False, **options: Any) -> CosConfig:
    # CosConfig.origin runs str.format() on endpoint to expand {bucket}/{region}. An origin parsed out
    # of a URL must never be reinterpreted that way, so escape its braces into literals.
    if endpoint is not None and not allow_placeholders:
        endpoint = endpoint.replace('{', '{{').replace('}', '}}')
    return CosConfig(endpoint=endpoint, **options)


def _derive_cos_key(local_path: Path, prefix: str | None) -> str:
    resolved = prefix if prefix is not None else os.getenv(ENV_COS_UPLOAD_PREFIX) or DEFAULT_COS_UPLOAD_PREFIX
    return '/'.join(part for part in (resolved.strip('/'), uuid.uuid4().hex, local_path.name) if part)


def _is_internal_host(hostname: str) -> bool:
    # Anchored on the full COS host grammar, never a bare substring: 'x.cos-internal.attacker.example'
    # must not be able to turn off certificate verification.
    match = _COS_HOST_PATTERN.match(hostname.lower())
    return match is not None and match['service'] == 'cos-internal'


def _resolve_download_target(dest: str | Path | None, key: str) -> Path:
    # PurePosixPath: COS keys are always '/'-separated, so a literal backslash is part of the name.
    # A key ending in '/' names no file at all, and PurePosixPath would hide that by normalising it.
    filename = '' if key.endswith('/') else PurePosixPath(key).name
    if filename in {'', '.', '..'}:
        filename = ''
    if dest is None:
        if not filename:
            raise CosClientError(f'COS key {key!r} has no file name; pass an explicit dest')
        return resolve_workdir_path(filename)
    text = dest if isinstance(dest, str) else str(dest)
    target = resolve_workdir_path(dest)
    # Checked on the original string: Path() drops a trailing separator. A Path built from 'outdir/'
    # has already lost it, so only a str can express "directory that does not exist yet".
    if text.endswith(('/', os.sep)) or target.is_dir():
        if not filename:
            raise CosClientError(f'COS key {key!r} has no file name; pass an explicit dest')
        return target / filename
    return target


def _presign_link(
    config: CosConfig,
    credentials: CosCredentials,
    key: str,
    *,
    expires: int,
    internal: bool,
    link_endpoint: str | None,
) -> str:
    # Never inherit the transport endpoint: the bytes may travel over an internal or accelerated
    # origin, while the link has to be reachable by whoever receives it.
    link_config = (
        # scheme='https' so a scheme-less link_endpoint never hands a signed URL out over plaintext.
        _cos_config(
            bucket=config.bucket,
            region=config.region,
            endpoint=link_endpoint,
            scheme='https',
            allow_placeholders=True,
        )
        if link_endpoint is not None
        else replace(
            config,
            endpoint=None,
            use_internal_endpoint=internal,
            scheme='http' if internal else 'https',
        )
    )
    # presign_url is synchronous and touches no session, so this client is never entered.
    return AsyncCosClient(link_config, credentials).presign_url(key, expires=expires)


async def _run_multipart_upload(
    *,
    part_ranges: list[tuple[int, int, int]],
    max_concurrency: int,
    upload_part: Callable[[int, int, int], Awaitable[str]],
    complete: Callable[[list[tuple[int, str]]], Awaitable[_T]],
    abort: Callable[[], Awaitable[None]],
) -> _T:
    try:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def upload_one(part_number: int, offset: int, length: int) -> tuple[int, str]:
            async with semaphore:
                return part_number, await upload_part(part_number, offset, length)

        tasks = [
            asyncio.create_task(upload_one(part_number, offset, length)) for part_number, offset, length in part_ranges
        ]
        try:
            uploaded_parts = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return await complete(uploaded_parts)
    except BaseException:
        with suppress(Exception):
            await asyncio.shield(abort())
        raise


async def _request_presigned(
    session: aiohttp.ClientSession,
    method: str,
    key: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    data: object = None,
    timeout: float | None,
    connect_timeout: float,
    verify_ssl: bool | SSLContext,
) -> aiohttp.ClientResponse:
    response = await session.request(
        method,
        url,
        headers=headers,
        data=data,
        allow_redirects=False,
        ssl=verify_ssl,
        timeout=aiohttp.ClientTimeout(total=None, connect=connect_timeout, sock_read=timeout),
        auto_decompress=False,
    )
    if 200 <= response.status < 300:
        return response
    try:
        body = await response.content.read(64 * 1024)
        error = _service_error(method, key, response.status, response.headers, body)
    finally:
        response.release()
    raise error


async def _with_retries(
    method: str,
    key: str,
    operation: Callable[[], Awaitable[_T]],
    *,
    max_retries: int,
    retry_backoff: float,
) -> _T:
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            return await operation()
        except CosServiceError as exc:
            retryable = exc.status in {408, 429} or exc.status >= 500 or exc.code in _RETRYABLE_COS_ERROR_CODES
            if not retryable or attempt + 1 >= attempts:
                raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt + 1 >= attempts:
                raise CosClientError(
                    f'COS {method} {key!r} failed after {attempts} attempts: {exc.__class__.__name__}: {exc}'
                ) from exc
        except _CosSourceChangedError:
            raise
        except CosClientError:
            if attempt + 1 >= attempts:
                raise
        if retry_backoff:
            await asyncio.sleep(min(retry_backoff * (2**attempt), 5.0))
    raise AssertionError('unreachable')


def _canonical_path(key: str) -> str:
    if not key:
        raise ValueError('COS object key must be non-empty')
    return key if key.startswith('/') else f'/{key}'


def _signable_headers(headers: Mapping[str, object]) -> dict[str, str]:
    result = {}
    for key, value in headers.items():
        name = key.lower()
        if name in _SIGNABLE_HEADERS or name.startswith(('x-cos-', 'x-ci-')):
            result[name] = _string_value(value)
    return result


def _canonical_mapping(values: Mapping[str, object], *, lowercase_keys: bool) -> tuple[str, list[str]]:
    encoded = []
    for key, value in values.items():
        name = quote(str(key), safe='-_.~')
        if lowercase_keys:
            name = name.lower()
        encoded.append((name, quote(_string_value(value), safe='-_.~')))
    encoded.sort()
    return '&'.join(f'{key}={value}' for key, value in encoded), [key for key, _value in encoded]


def _string_value(value: object) -> str:
    return '' if value is None else str(value)


def _local_file_state(path: Path) -> _LocalFileState:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise CosClientError(f'Failed to inspect local file {path}: {exc}') from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(path)
    return _file_state_from_stat(metadata)


def _file_state_from_stat(metadata: os.stat_result) -> _LocalFileState:
    return _LocalFileState(
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _ensure_file_unchanged(path: Path, expected: _LocalFileState) -> None:
    try:
        current = _local_file_state(path)
    except (FileNotFoundError, CosClientError) as exc:
        raise _CosSourceChangedError(f'Local file {path} changed during upload') from exc
    if current != expected:
        raise _CosSourceChangedError(f'Local file {path} changed during upload')


def _validate_transfer_options(
    *,
    timeout: float | None,
    connect_timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> None:
    if timeout is not None and timeout <= 0:
        raise ValueError('COS timeout must be greater than zero')
    if connect_timeout <= 0:
        raise ValueError('COS connect_timeout must be greater than zero')
    if max_retries < 0:
        raise ValueError('COS max_retries must not be negative')
    if retry_backoff < 0:
        raise ValueError('COS retry_backoff must not be negative')


def _validate_presigned_url(url: str, *, field: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.fragment:
        raise CosClientError(f'COS multipart signing response has an invalid {field}')


def _validate_caller_headers(headers: Mapping[str, str]) -> None:
    reserved = {'authorization', 'host', 'x-cos-security-token'}
    invalid = reserved.intersection(key.lower() for key in headers)
    if invalid:
        names = ', '.join(sorted(invalid))
        raise ValueError(f'COS request headers are managed by the client: {names}')


def _has_header(headers: Mapping[str, str] | None, name: str) -> bool:
    expected = name.lower()
    return headers is not None and any(key.lower() == expected for key in headers)


def _without_headers(headers: Mapping[str, str] | None, excluded: set[str]) -> dict[str, str]:
    names = {name.lower() for name in excluded}
    return {key: value for key, value in (headers or {}).items() if key.lower() not in names}


def _object_upload_headers(
    content_type: str | None,
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    result = _without_headers(headers, {'content-length'})
    _validate_caller_headers(result)
    if content_type:
        result['Content-Type'] = content_type
    return result


def _upload_headers(
    size: int,
    content_type: str | None,
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    result = _object_upload_headers(content_type, headers)
    result['Content-Length'] = str(size)
    return result


def _multipart_part_headers(size: int, headers: Mapping[str, str] | None) -> dict[str, str]:
    result = {'Content-Length': str(size)}
    for key, value in (headers or {}).items():
        name = key.lower()
        if name == 'x-cos-traffic-limit' or name.startswith('x-cos-server-side-encryption-customer-'):
            result[key] = value
    return result


def select_multipart_part_size(size: int, preferred_part_size: int) -> int:
    """Select a COS-valid part size while staying within the 10,000-part limit."""

    if size < 0:
        raise ValueError('COS object size must not be negative')
    if preferred_part_size < _MIN_MULTIPART_PART_SIZE:
        raise ValueError(f'COS part_size must be at least {_MIN_MULTIPART_PART_SIZE} bytes')
    if preferred_part_size > _MAX_MULTIPART_PART_SIZE:
        raise ValueError(f'COS part_size must not exceed {_MAX_MULTIPART_PART_SIZE} bytes')
    minimum_size = (size + _MAX_MULTIPART_PARTS - 1) // _MAX_MULTIPART_PARTS
    minimum_size = ((minimum_size + _MIB - 1) // _MIB) * _MIB
    part_size = max(preferred_part_size, minimum_size)
    if part_size > _MAX_MULTIPART_PART_SIZE:
        raise CosClientError(f'COS object size {size} requires parts larger than {_MAX_MULTIPART_PART_SIZE} bytes')
    return part_size


def _part_ranges(size: int, preferred_part_size: int) -> list[tuple[int, int, int]]:
    part_size = select_multipart_part_size(size, preferred_part_size)
    if size == 0:
        return []
    ranges = []
    for part_number, offset in enumerate(range(0, size, part_size), start=1):
        ranges.append((part_number, offset, min(part_size, size - offset)))
    return ranges


async def _iter_file_segment(path: Path, offset: int, length: int) -> AsyncIterator[bytes]:
    remaining = length
    async with aiofiles.open(path, 'rb') as source:
        await source.seek(offset)
        while remaining:
            chunk = await source.read(min(_STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                raise _CosSourceChangedError(
                    f'Local file {path} ended before reading {length} bytes at offset {offset}'
                )
            remaining -= len(chunk)
            yield chunk


# Public aliases so other SDKs (e.g. the artifact client) can reuse the same
# streaming, retry, and source-integrity machinery instead of duplicating it.
iter_file_segment = _iter_file_segment
with_retries = _with_retries
run_multipart_upload = _run_multipart_upload
part_ranges = _part_ranges
CosSourceChangedError = _CosSourceChangedError


def _complete_multipart_body(uploaded_parts: list[tuple[int, str]]) -> bytes:
    root = ElementTree.Element('CompleteMultipartUpload')
    for part_number, etag in sorted(uploaded_parts):
        part = ElementTree.SubElement(root, 'Part')
        ElementTree.SubElement(part, 'PartNumber').text = str(part_number)
        ElementTree.SubElement(part, 'ETag').text = etag
    return ElementTree.tostring(root, encoding='utf-8', xml_declaration=True)


def _xml_value(body: bytes, name: str) -> str | None:
    if not body:
        return None
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None
    element = root.find(f'.//{{*}}{name}')
    return element.text if element is not None else None


def _xml_error_code(body: bytes) -> str | None:
    if not body:
        return None
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return None
    if root.tag.rsplit('}', 1)[-1] != 'Error':
        return None
    code = root.find('.//{*}Code')
    return code.text if code is not None else 'UnknownError'


def _parse_content_range(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    match = _CONTENT_RANGE_PATTERN.fullmatch(value)
    if match is None:
        return None
    start, end, size = match.groups()
    return int(start), int(end), int(size)


def _object_metadata(key: str, size: int, headers: Mapping[str, str]) -> CosObject:
    return CosObject(
        key=key,
        size=size,
        etag=headers.get('ETag'),
        request_id=headers.get('x-cos-request-id'),
    )


def _service_error(
    method: str,
    key: str,
    status: int,
    headers: Mapping[str, str],
    body: bytes,
) -> CosServiceError:
    code = None
    message = None
    request_id = headers.get('x-cos-request-id')
    if body:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError:
            message = body.decode('utf-8', errors='replace')[:300]
        else:
            values = {element.tag.rsplit('}', 1)[-1]: element.text for element in root.iter()}
            code = values.get('Code')
            message = values.get('Message')
            request_id = request_id or values.get('RequestId')
    return CosServiceError(
        method=method,
        key=key,
        status=status,
        code=code,
        message=message,
        request_id=request_id,
    )


__all__ = [
    'AsyncCosClient',
    'COS_SHARED_VALUE_NAME',
    'DEFAULT_COS_LINK_EXPIRES',
    'DEFAULT_COS_UPLOAD_PREFIX',
    'DEFAULT_MULTIPART_PART_SIZE',
    'CosClientError',
    'CosConfig',
    'CosConfigurationError',
    'CosCredentials',
    'CosLink',
    'CosLocation',
    'CosObject',
    'CosServiceError',
    'CosSourceChangedError',
    'CosV5Signer',
    'clear_cos_credentials_cache',
    'download_from_cos_url',
    'download_presigned_file',
    'is_presigned_cos_url',
    'iter_file_segment',
    'load_cos_credentials',
    'parse_cos_url',
    'part_ranges',
    'resolve_cos_credentials',
    'run_multipart_upload',
    'select_multipart_part_size',
    'upload_presigned_file',
    'upload_presigned_multipart_file',
    'upload_to_cos',
    'with_retries',
]
