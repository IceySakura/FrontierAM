"""COS upload and download helpers for artifact archives.

Provide synchronous and thread-backed asynchronous APIs. Async timeouts do
not terminate an in-flight SDK thread. Base64 encoding does not protect secrets.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import threading

from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


logger = logging.getLogger(__name__)

_DEFAULT_BUCKET = 'fitzhu-cos-1257943044'
_DEFAULT_REGION = 'ap-nanjing'
_DEFAULT_PREFIX = 'nexus/frontend-task'
_DEFAULT_TIMEOUT_S = 1800  # total async upload ceiling
# 'ioa' (default) resolves the -internal endpoint alias; 'idc' the plain one.
_COS_ENV = 'ioa'

# Base64 encoding is not encryption; do not expose credential values.
_DEFAULT_SECRET_ID_B64 = 'QUtJRGRFbkdqNUdEeFh2NjRvU3E5R3pXcjE2OEE2NHprSWVv'  # noqa: S105
_DEFAULT_SECRET_KEY_B64 = 'MXlvMkdQV3ZBTWJyQVp4WUFqQVRyRVNZVkp5VG10b1g='  # noqa: S105
_TRUTHY = {'1', 'true', 'yes', 'on'}

# Create the shared COS client lazily.
_client: Any = None
_client_lock = threading.Lock()


def _is_transient_io_error(exc: BaseException) -> bool:
    """Return whether an I/O error is eligible for retry."""
    # Inspect wrapped exceptions when classifying transient transport errors.
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (OSError, ConnectionError, TimeoutError)):
            return True
        cur = cur.__cause__ or cur.__context__

    cls = type(exc).__name__
    if cls == 'CosClientError':
        return True
    if cls == 'CosServiceError':
        # qcloud_cos exposes both `get_status_code()` (method) on some versions
        # and `status_code` (attribute) on others; tolerate either.
        status: Any = None
        getter = getattr(exc, 'get_status_code', None)
        if callable(getter):
            try:
                status = getter()
            except Exception:  # noqa: BLE001 — best-effort introspection
                status = None
        if status is None:
            status = getattr(exc, 'status_code', None)
        return isinstance(status, int) and 500 <= status < 600
    return False


# Retry transient transport failures with exponential jitter; four total attempts.
_io_retry = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1.0, max=8.0),
    retry=retry_if_exception(_is_transient_io_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def is_upload_enabled() -> bool:
    """Whether to back the archive up to COS. See ``config.CosCfg.enabled``.

    Upload is on by default; turn it off with ``cos.enabled: false`` in a profile (or the
    legacy ``FRONTEND_UPLOAD_TO_COS=0``, still honoured by the config layer).
    """
    from nexus.extensions.tasks.artifact import config

    return config.get().cos.enabled


def _decode_default_secret_id() -> str:
    return base64.b64decode(_DEFAULT_SECRET_ID_B64).decode()


def _decode_default_secret_key() -> str:
    return base64.b64decode(_DEFAULT_SECRET_KEY_B64).decode()


def get_prefix() -> str:
    """Return the COS key prefix, with no trailing slash."""
    return _DEFAULT_PREFIX


def get_bucket() -> str:
    return _DEFAULT_BUCKET


def get_region() -> str:
    return _DEFAULT_REGION


def get_endpoint() -> str:
    region = get_region()
    # IDC routes through plain cos.<region>.tencentcos.cn; iOA / external uses the -internal alias.
    if _COS_ENV == 'idc':
        return f'cos.{region}.tencentcos.cn'
    return f'cos-internal.{region}.tencentcos.cn'


def cos_url(key: str) -> str:
    """Build a `cos://<bucket>/<key>` URL for logging / output."""
    return f'cos://{get_bucket()}/{key.lstrip("/")}'


def _get_client() -> Any:
    """Lazy-build a CosS3Client. Raises RuntimeError when prereqs are missing.

    Thread-safe via double-checked locking so concurrent `asyncio.to_thread`
    upload workers don't race to build duplicate clients.
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as exc:
            raise RuntimeError(
                'qcloud_cos is not installed. Install with `pip install cos-python-sdk-v5` to enable COS uploads.'
            ) from exc

        # Credentials: the built-in cluster account, overridable by env for a personal bucket.
        # Empty string counts as "not set" so a partial override still falls back per-field.
        secret_id = os.environ.get('COS_SECRET_ID', '') or _decode_default_secret_id()
        secret_key = os.environ.get('COS_SECRET_KEY', '') or _decode_default_secret_key()
        if not secret_id or not secret_key:
            raise RuntimeError('COS credentials missing: set COS_SECRET_ID and COS_SECRET_KEY.')

        config = CosConfig(
            Region=get_region(),
            SecretId=secret_id,
            SecretKey=secret_key,
            Scheme='https',
            Endpoint=get_endpoint(),
        )
        _client = CosS3Client(config)
        return _client


def _get_upload_timeout() -> float:
    """Total async upload ceiling in seconds.

    The async wrappers apply this around the whole worker-thread call, so
    base64 decoding, the PUT and tenacity retries all consume the same budget.
    """
    return float(_DEFAULT_TIMEOUT_S)


@_io_retry
def upload_bytes(key: str, data: bytes | io.IOBase) -> str:
    """Upload raw bytes (or a file-like object) to COS at `key`.

    Accepts either a ``bytes`` blob or a binary file-like object (anything
    qcloud_cos can stream — ``io.BytesIO``, an open file in ``rb`` mode, or
    a ``tarfile``-produced reader). Streaming via a file-like avoids a
    second copy of the payload in memory.

    Returns ``cos://<bucket>/<key>`` on success. Synchronous; for asyncio
    use ``upload_bytes_async``.

    Retries: wrapped in ``_io_retry`` so transient errors (NFS-stat hiccup on
    certifi/cacert.pem, network blip, cos 5xx) get up to 4 attempts. To make
    retries idempotent for stream payloads, we ``seek(0)`` on each entry —
    otherwise a second attempt would PUT zero bytes because qcloud_cos drains
    the stream during the first failed PUT.
    """
    seek = getattr(data, 'seek', None)
    if callable(seek):
        seek(0)
    client = _get_client()
    bucket = get_bucket()
    client.put_object(Bucket=bucket, Key=key, Body=data)
    return cos_url(key)


def upload_archive_b64(key: str, archive_b64: str) -> str:
    """Decode a base64 archive and PUT it to COS at `key`. Returns `cos://...` URL.

    The decoded bytes are streamed via ``io.BytesIO`` instead of being
    handed in as a second ``bytes`` reference, so the qcloud_cos SDK can
    chunk the upload without us holding two large copies.

    Synchronous; prefer ``upload_archive_b64_async`` from async code.
    """
    if not archive_b64:
        raise ValueError('archive_b64 is empty; nothing to upload')
    return upload_bytes(key, io.BytesIO(base64.b64decode(archive_b64)))


async def upload_bytes_async(key: str, data: bytes | io.IOBase) -> str:
    """Upload in a worker thread with a timeout; timeout does not cancel the SDK thread."""
    return await asyncio.wait_for(
        asyncio.to_thread(upload_bytes, key, data),
        timeout=_get_upload_timeout(),
    )


async def upload_archive_b64_async(key: str, archive_b64: str) -> str:
    """Decode and upload an archive in a worker thread with the configured timeout."""
    if not archive_b64:
        raise ValueError('archive_b64 is empty; nothing to upload')
    return await asyncio.wait_for(
        asyncio.to_thread(upload_archive_b64, key, archive_b64),
        timeout=_get_upload_timeout(),
    )


def parse_cos_url(url: str) -> tuple[str, str]:
    """Split a `cos://<bucket>/<key>` URL produced by `cos_url()` into (bucket, key).

    Use this to feed the bucket back into `download_*` when the URL is for a
    bucket other than the configured default (e.g. shared archives written by
    a teammate using a different `COS_BUCKET`). Raises `ValueError` on a
    malformed URL.
    """
    if not isinstance(url, str) or not url.startswith('cos://'):
        raise ValueError(f'not a cos:// URL: {url!r}')
    rest = url[len('cos://') :]
    bucket, _, key = rest.partition('/')
    if not bucket or not key:
        raise ValueError(f'cos URL missing bucket or key: {url!r}')
    return bucket, key


@_io_retry
def download_bytes(key: str, *, bucket: str | None = None) -> bytes:
    """GET an object from COS at `key` and return its raw bytes.

    Reads the entire body into memory in one shot — fine for tar.gz archives
    the frontend task produces (typically tens of MB) but **don't** use this
    on multi-GB objects: prefer `download_to_file` which streams onto disk.

    Synchronous; prefer `download_bytes_async` from async code so the
    event loop is not blocked for the duration of the GET.
    """
    client = _get_client()
    response = client.get_object(Bucket=bucket or get_bucket(), Key=key)
    return response['Body'].get_raw_stream().read()


@_io_retry
def download_to_file(key: str, dest_path: str | os.PathLike[str], *, bucket: str | None = None) -> str:
    """GET an object from COS at `key` and stream it onto disk at `dest_path`.

    Uses qcloud_cos's chunked downloader (`CosS3Client.download_file`) so
    memory stays bounded even for multi-GB objects. Creates the parent
    directory if missing. Returns the destination path as a str.

    Synchronous; prefer `download_to_file_async` from async code.
    """
    client = _get_client()
    dest = os.fspath(dest_path)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)
    client.download_file(Bucket=bucket or get_bucket(), Key=key, DestFilePath=dest)
    return dest


def download_archive_b64(key: str, *, bucket: str | None = None) -> str:
    """GET an object from COS and re-encode it as base64 (mirror of `upload_archive_b64`).

    Useful when downstream code expects the same shape as
    `task_output.archive_b64` / `claude_home_archive_b64` — e.g. feeding the
    artifact back into another batch as a fixture without round-tripping
    through disk.

    Synchronous; prefer `download_archive_b64_async` from async code.
    """
    return base64.b64encode(download_bytes(key, bucket=bucket)).decode()


async def download_bytes_async(key: str, *, bucket: str | None = None) -> bytes:
    """Async wrapper around `download_bytes`; runs the GET in a worker thread."""
    return await asyncio.to_thread(download_bytes, key, bucket=bucket)


async def download_to_file_async(key: str, dest_path: str | os.PathLike[str], *, bucket: str | None = None) -> str:
    """Async wrapper around `download_to_file`; streams onto disk in a worker thread."""
    return await asyncio.to_thread(download_to_file, key, dest_path, bucket=bucket)


async def download_archive_b64_async(key: str, *, bucket: str | None = None) -> str:
    """Async wrapper around `download_archive_b64`.

    Both the GET and the base64 encode happen in a worker thread so a
    multi-MB download does not block the asyncio event loop.
    """
    return await asyncio.to_thread(download_archive_b64, key, bucket=bucket)


def build_archive_key(
    *,
    instance_id: str,
    archive_name: str,
    cache_task_id: str = '',
    span_id: str = '',
    timestamp: str | None = None,
    date_str: str | None = None,
) -> str:
    """Build a per-instance, per-run archive key under the configured prefix."""
    from datetime import datetime

    now = datetime.now()
    ts = timestamp or now.strftime('%Y%m%d-%H%M%S')
    # Keep the date folder consistent with ts when a timestamp is supplied
    # (avoids a midnight-boundary split between folder day and filename ts).
    day = date_str or (ts[:8] if ts[:8].isdigit() else now.strftime('%Y%m%d'))
    safe_ts = _sanitise(ts) or 'unknown'
    safe_instance = _sanitise(instance_id) or 'unknown'
    safe_run = _sanitise(cache_task_id) or _sanitise(span_id) or 'unknown'
    base_name = archive_name.lstrip('/')
    leaf = f'{safe_run}__{safe_instance}__{safe_ts}__{base_name}'
    return f'{get_prefix()}/{day}/{safe_instance}/{safe_run}/{leaf}'


def _sanitise(value: str) -> str:
    """Replace anything outside `[A-Za-z0-9._-]` with '_' so the key is URL-safe."""
    out_chars: list[str] = []
    for ch in (value or '').strip():
        if ch.isalnum() or ch in {'.', '_', '-'}:
            out_chars.append(ch)
        else:
            out_chars.append('_')
    return ''.join(out_chars)


__all__ = [
    'build_archive_key',
    'cos_url',
    'download_archive_b64',
    'download_archive_b64_async',
    'download_bytes',
    'download_bytes_async',
    'download_to_file',
    'download_to_file_async',
    'get_bucket',
    'get_endpoint',
    'get_prefix',
    'get_region',
    'is_upload_enabled',
    'parse_cos_url',
    'upload_archive_b64',
    'upload_archive_b64_async',
    'upload_bytes',
    'upload_bytes_async',
]
