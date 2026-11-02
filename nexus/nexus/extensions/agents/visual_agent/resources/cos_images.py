"""Download HTTP(S) COS images before the normal media ingestion pipeline."""

import asyncio
import base64
import re

from collections.abc import Sequence
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import aiohttp

from yarl import URL


_COS_HOST = re.compile(r'[^.]+\.cos(?:-internal)?\.[a-z0-9-]+\.(?:myqcloud\.com|tencentcos\.cn)', re.IGNORECASE)
_DOWNLOAD_TIMEOUT = 60
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_CONCURRENCY = 4


def is_cos_image_url(ref: str) -> bool:
    """Recognize bucket-qualified COS endpoints, including internal/signed URLs."""
    try:
        parsed = urlsplit(ref)
        return parsed.scheme in {'http', 'https'} and _COS_HOST.fullmatch(parsed.hostname or '') is not None
    except ValueError:
        return False


def _as_image_data_uri(payload: bytes, ref: str) -> str:
    """Wrap downloaded bytes without decoding, validating or re-encoding pixels."""
    suffix = PurePosixPath(urlsplit(ref).path).suffix.lower()
    mime = {
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
    }.get(suffix, 'image/jpeg')
    return f'data:{mime};base64,{base64.b64encode(payload).decode("ascii")}'


async def _download_image(session: aiohttp.ClientSession, ref: str) -> str:
    host = urlsplit(ref).hostname or 'COS'
    try:
        # Do not re-quote a signed URL: COS may sign the escaped path/query.
        # Redirects can change the signed host and are not needed for object URLs.
        async with session.get(URL(ref, encoded=True), allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f'COS image download from {host} failed: HTTP {response.status}')
            if response.content_length is not None and response.content_length > _MAX_IMAGE_BYTES:
                raise ValueError(f'COS image from {host} exceeds the 64 MiB download limit')
            data = bytearray()
            async for chunk in response.content.iter_chunked(1024 * 1024):
                data.extend(chunk)
                if len(data) > _MAX_IMAGE_BYTES:
                    raise ValueError(f'COS image from {host} exceeds the 64 MiB download limit')
    except (aiohttp.ClientError, asyncio.TimeoutError):
        # Signed query strings and tokens must not appear in error traces.
        raise ValueError(f'COS image download from {host} failed; check connectivity and URL validity') from None
    if not data:
        raise ValueError(f'COS image download from {host} returned an empty response')
    return await asyncio.to_thread(_as_image_data_uri, bytes(data), ref)


async def download_cos_images(refs: Sequence[str]) -> dict[str, str]:
    """Return URL -> data URI for unique COS sources; any failure stops ingestion."""
    sources = list(dict.fromkeys(refs))
    if not sources:
        return {}
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    # Match existing COS/spatial clients: internal endpoints must not inherit
    # a host's outbound HTTP proxy. Use each supplied object's URL as-is.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=_DOWNLOAD_TIMEOUT), trust_env=False
    ) as session:

        async def download(ref: str) -> str:
            async with semaphore:
                return await _download_image(session, ref)

        results = await asyncio.gather(*(download(ref) for ref in sources), return_exceptions=True)
    resolved: dict[str, str] = {}
    for ref, result in zip(sources, results, strict=True):
        if isinstance(result, BaseException):
            raise result
        resolved[ref] = result
    return resolved
