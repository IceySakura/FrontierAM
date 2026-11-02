"""
工具名称: 图片下载工具
工具类型: LOCAL
说明: 从URL下载图片并以base64格式返回给模型，支持单个URL或URL列表，下载通过代理完成
"""

import asyncio
import base64
import logging
import mimetypes
import os
import ssl

import aiohttp

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.extensions.tools.web_search._config_utils import as_bool as _as_bool
from nexus.extensions.tools.web_search._config_utils import as_float as _as_float
from nexus.extensions.tools.web_search._config_utils import as_int as _as_int
from nexus.extensions.tools.web_search._sandbox_files import fetch_enabled, is_remote_uri, read_sandbox_bytes


logger = logging.getLogger(__name__)

# ============== Constants ==============
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 1
RETRY_DELAY = 2.0
# Maximum image size to prevent OOM (default 20MB)
MAX_IMAGE_SIZE = int(os.environ.get('IMAGE_DOWNLOAD_MAX_SIZE', 20 * 1024 * 1024))


def _build_img_ssl(ca_file: str | None = None, insecure: bool = False) -> ssl.SSLContext | bool | None:
    if insecure:
        logger.warning('[ImageDownload] TLS verification disabled for image downloads.')
        return False
    if not ca_file:
        return None
    if not os.path.isfile(ca_file):
        logger.warning(f'[ImageDownload] CA file not found: {ca_file}')
        return None
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=ca_file)
    return context


def _download_config(config: dict | None = None) -> dict:
    config = config or {}
    proxy = config.get('image_proxy') or os.environ.get('IMG_HTTP_PROXY') or os.environ.get('IMG_HTTPS_PROXY')
    ca_file = config.get('image_ca_file') or os.environ.get('IMG_HTTPS_CA_FILE') or os.environ.get('SANDBOX_PROXY_CA')
    insecure = _as_bool(config.get('image_ssl_insecure'), _as_bool(os.environ.get('IMG_SSL_INSECURE')))
    return {
        'timeout': _as_float(config.get('timeout', config.get('image_download_timeout')), DEFAULT_TIMEOUT),
        'max_retries': _as_int(config.get('max_retries', config.get('image_download_retries')), MAX_RETRIES),
        'max_size': _as_int(config.get('image_download_max_size'), MAX_IMAGE_SIZE),
        'proxy': proxy,
        'ssl_arg': _build_img_ssl(ca_file, insecure),
    }


# Magic bytes for the raster formats a vision model can consume. A local file is
# matched against these because the model chooses the path, so neither the
# extension nor the absence of one can be taken as evidence of the contents.
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
    (b'BM', 'image/bmp'),
)
_MAGIC_PREFIX_BYTES = 32


def _sniff_image_mime(data: bytes) -> str | None:
    """Identify an image format from its leading bytes.

    Args:
        data (bytes): Start of the file; ``_MAGIC_PREFIX_BYTES`` is enough.

    Returns:
        The mime type, or None when the bytes are not a supported image.
    """
    for signature, mime in _IMAGE_MAGIC:
        if data.startswith(signature):
            return mime
    # RIFF....WEBP carries its marker at offset 8, past the chunk size.
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


# ============== Helper Functions ==============
def _guess_mime_type(url: str, content_type: str | None = None) -> str:
    """Guess MIME type from URL or Content-Type header."""
    if content_type:
        # Extract main type from header like 'image/png; charset=utf-8'
        mime = content_type.split(';')[0].strip().lower()
        if mime.startswith('image/'):
            return mime

    # Fallback: guess from URL path
    mime, _ = mimetypes.guess_type(url.split('?')[0].split('#')[0])
    if mime and mime.startswith('image/'):
        return mime

    # Default
    return 'image/png'


async def download_image(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
    max_size: int = MAX_IMAGE_SIZE,
    proxy: str | None = None,
    ssl_arg: ssl.SSLContext | bool | None = None,
) -> tuple[bytes | None, str | None]:
    """
    Download an image from a URL using a proxy-enabled aiohttp session.

    Uses trust_env=True so that only image download requests go through
    the proxy configured via http_proxy/https_proxy environment variables.

    Args:
        url: The image URL to download.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        A tuple of (image_bytes, mime_type), or (None, None) on failure.
    """
    ct = aiohttp.ClientTimeout(total=timeout)
    get_kwargs: dict = {}
    if proxy:
        get_kwargs['proxy'] = proxy
    if ssl_arg is not None:
        get_kwargs['ssl'] = ssl_arg

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession(timeout=ct, trust_env=True) as session:
                async with session.get(url, **get_kwargs) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get('Content-Type', '')
                        data = await resp.read()

                        if len(data) > max_size:
                            logger.warning(f'[ImageDownload] Image too large ({len(data)} bytes > {max_size}): {url}')
                            return None, None

                        mime_type = _guess_mime_type(url, content_type)
                        logger.debug(f'[ImageDownload] Downloaded {len(data)} bytes ({mime_type}) from: {url}')
                        return data, mime_type
                    body = await resp.text()
                    logger.warning(f'[ImageDownload] HTTP {resp.status} for {url}: {body[:200]}')
        except asyncio.TimeoutError:
            logger.warning(
                f'[ImageDownload] Timeout ({timeout}s) downloading {url} (attempt {attempt + 1}/{max_retries})'
            )
        except Exception as e:
            logger.warning(
                f'[ImageDownload] Error downloading {url} (attempt {attempt + 1}/{max_retries}): '
                f'{type(e).__name__}: {e}'
            )

        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)

    logger.error(f'[ImageDownload] Failed to download after {max_retries} attempts: {url}')
    return None, None


async def read_local_image(path: str, max_size: int = MAX_IMAGE_SIZE) -> tuple[bytes | None, str | None]:
    """Read an image the model produced in the sandbox.

    The model routinely crops or zooms with ``code_interpreter`` and then asks to
    view the result, which is a filesystem path rather than a URL. HTTP download
    cannot serve those, so the bytes are taken from the sandbox instead.

    The model chooses this path, so the contents are validated rather than
    trusted: the bytes must carry a known image signature. Nothing else may be
    read, or the tool would hand the model any file the agent process can open
    and base64 it into the trajectory.

    Args:
        path (str): Absolute path as supplied by the model.
        max_size (int): Reject anything larger, matching the download limit.

    Returns:
        A tuple of (image_bytes, mime_type), or (None, None) when the path is
        unavailable or does not hold a supported image.
    """
    data = await read_sandbox_bytes(path, max_bytes=max_size)
    if data is None:
        return None, None
    # NOTE(jaykaizhang): sniff the payload instead of reading the extension.
    # _guess_mime_type falls back to image/png for anything it cannot place, so
    # trusting it would label a .txt (or an extensionless file) as an image.
    mime_type = _sniff_image_mime(data[:_MAGIC_PREFIX_BYTES])
    if mime_type is None:
        logger.warning('[ImageDownload] Not a supported image, refusing to return it: %s', path)
        return None, None
    logger.debug(f'[ImageDownload] Read {len(data)} bytes ({mime_type}) from sandbox file: {path}')
    return data, mime_type


async def download_images(
    urls: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    max_size: int = MAX_IMAGE_SIZE,
    proxy: str | None = None,
    ssl_arg: ssl.SSLContext | bool | None = None,
) -> list[tuple[str, bytes | None, str | None]]:
    """
    Download multiple images concurrently.

    With ``NEXUS_FETCH_SANDBOX_IMAGES`` set, a filesystem path is served from
    the run's sandbox instead of being fetched over HTTP, which is the only way
    to view an image ``code_interpreter`` just produced. Without the flag every
    argument goes to HTTP as before, so a path fails the way it always did.

    Args:
        urls: List of image URLs.
        timeout: Timeout per request.
        max_retries: Max retries per request.

    Returns:
        List of (url, image_bytes, mime_type) tuples.
    """
    # NOTE(jaykaizhang): read the flag once so every argument in one call is
    # routed the same way even if the environment changes underneath us.
    serve_from_sandbox = fetch_enabled()
    tasks = [
        read_local_image(url, max_size=max_size)
        if serve_from_sandbox and not is_remote_uri(url)
        else download_image(
            url,
            timeout=timeout,
            max_retries=max_retries,
            max_size=max_size,
            proxy=proxy,
            ssl_arg=ssl_arg,
        )
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for url, result in zip(urls, results, strict=False):
        if isinstance(result, Exception):
            logger.error(f'[ImageDownload] Unexpected error for {url}: {result}')
            output.append((url, None, None))
        else:
            data, mime = result
            output.append((url, data, mime))
    return output


def _image_to_base64_url(data: bytes, mime_type: str) -> str:
    """Convert image bytes to a data URI string."""
    b64 = base64.b64encode(data).decode('utf-8')
    return f'data:{mime_type};base64,{b64}'


# ============== Nexus Tool Adapter ==============
class ImageDownloadInput(BaseModel):
    """Input parameters for image_visit tool."""

    url: str | list[str] = Field(
        description='The URL(s) of the image(s) to download. Can be a single URL or an array of URLs.'
    )


@tool_registry.register(
    'image_visit',
    meta={
        'name': 'image_visit',
        'description': (
            'Download and return images from the web. '
            'Accepts a single image URL or a list of image URLs. '
            'Returns the downloaded images in base64 format for direct viewing.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def image_visit_tool(params: ImageDownloadInput, context: ToolContext) -> ToolResponse:
    """Download images from URLs and return them as base64-encoded content."""
    urls = params.url

    if not urls:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty URL provided.'}],
            meta={'success': False},
        )

    # Normalize to list
    if isinstance(urls, str):
        urls = [urls]

    # Filter out empty strings
    urls = [u.strip() for u in urls if u and u.strip()]
    if not urls:
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: No valid URLs provided.'}],
            meta={'success': False},
        )

    try:
        cfg = context.tool_configs.get('image_visit', {})
        results = await download_images(urls, **_download_config(dict(cfg)))

        output_parts: list[ChatCompletionContentPartParam | dict] = []
        success_count = 0
        fail_urls = []

        for url, data, mime_type in results:
            if data and mime_type:
                success_count += 1
                data_uri = _image_to_base64_url(data, mime_type)
                output_parts.append({
                    'type': 'text',
                    'text': f'\nImage from: {url}',
                })
                output_parts.append({
                    'type': 'image_url',
                    'image_url': {'url': data_uri},
                    # 'image_url': data_uri,
                })
            else:
                fail_urls.append(url)
                # NOTE(jaykaizhang): only when serving paths from the sandbox is a
                # failure not a download failure; calling it one would misdirect
                # whoever reads the trajectory. Without the flag the request did go
                # to HTTP, so the original wording stays accurate.
                served_locally = fetch_enabled() and not is_remote_uri(url)
                reason = 'read image file' if served_locally else 'download image from'
                output_parts.append({
                    'type': 'text',
                    'text': f'\nFailed to {reason}: {url}',
                })

        if len(output_parts) == 0:
            output_parts.append({'type': 'text', 'text': 'No images were successfully downloaded.'})

        return ToolResponse(
            output=output_parts,
            meta={
                'success': success_count > 0,
                'total': len(urls),
                'downloaded': success_count,
                'failed': len(fail_urls),
            },
        )

    except Exception as e:
        logger.error(f'[ImageDownload] Tool error: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error during image download: {str(e)}'}],
            meta={'success': False},
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image download tool - download images from URLs')
    parser.add_argument(
        '--url',
        '-u',
        type=str,
        nargs='+',
        default=[
            'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ-XBO1_qby1o_bZAXYab4k53-jPmJFzJYddZb7OLCx4dEK4ZWX',
            'https://lookaside.fbsbx.com/lookaside/crawler/media/?media_id=122098318473211255',
        ],
        help='One or more image URLs to download',
    )
    parser.add_argument('--timeout', '-t', type=float, default=30.0, help='Timeout in seconds')
    parser.add_argument('--output-dir', '-o', type=str, default='.', help='Directory to save images')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def main():
        results = await download_images(args.url, timeout=args.timeout)
        os.makedirs(args.output_dir, exist_ok=True)

        for url, data, mime_type in results:
            if data and mime_type:
                # Determine file extension
                ext = mimetypes.guess_extension(mime_type) or '.png'
                fname = os.path.basename(url.split('?')[0].split('#')[0])
                if not fname or '.' not in fname:
                    fname = f'image{ext}'
                save_path = os.path.join(args.output_dir, fname)
                with open(save_path, 'wb') as f:
                    f.write(data)
                print(f'[OK] {url} -> {save_path} ({len(data)} bytes, {mime_type})')
            else:
                print(f'[FAIL] {url}')

    asyncio.run(main())
