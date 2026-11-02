"""
工具名称: Google Serper 图搜图
工具类型: LOCAL
说明: 使用 Serper.dev API 提供的低成本 Google 搜索服务(图搜图)
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os

from dataclasses import dataclass
from urllib.parse import quote

import aiohttp

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.extensions.tools.web_search._config_utils import as_bool as _as_bool
from nexus.extensions.tools.web_search._config_utils import as_float as _as_float
from nexus.extensions.tools.web_search._config_utils import as_int as _as_int
from nexus.extensions.tools.web_search._sandbox_files import discard_staged_copy, is_remote_uri, stage_image_path
from nexus.utils.async_http import http_client

from .text_to_image import _download_images, _image_download_config, _image_to_base64_url


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class CosUploadConfig:
    secret_id: str
    secret_key: str
    bucket: str
    prefix: str
    region: str | None = None
    token: str | None = None
    endpoint: str | None = None
    service_domain: str | None = None
    scheme: str = 'https'
    public_base_url: str | None = None
    public_url_template: str | None = None
    storage_class: str | None = None
    acl: str | None = None


_cos_clients: dict[tuple[str, str, str, str, str, str], object] = {}


def _cfg_or_env(config: dict, config_key: str, *env_keys: str, default: object = None) -> object:
    value = config.get(config_key)
    if value not in (None, ''):
        return value
    for env_key in env_keys:
        env_value = os.environ.get(env_key)
        if env_value not in (None, ''):
            return env_value
    return default


def _cos_upload_config(config: dict | None = None) -> CosUploadConfig | None:
    config = config or {}
    enabled = _as_bool(
        _cfg_or_env(config, 'cos_enabled', 'I2I_COS_ENABLED', 'COS_UPLOAD_ENABLED'),
        False,
    )
    if not enabled:
        return None

    secret_id = str(_cfg_or_env(config, 'cos_secret_id', 'COS_SECRET_ID', default='') or '')
    secret_key = str(_cfg_or_env(config, 'cos_secret_key', 'COS_SECRET_KEY', default='') or '')
    bucket = str(_cfg_or_env(config, 'cos_bucket', 'COS_BUCKET', default='') or '')
    prefix = str(_cfg_or_env(config, 'cos_prefix', 'COS_UPLOAD_PREFIX', default='') or '').strip('/')
    region = str(_cfg_or_env(config, 'cos_region', 'COS_REGION', default='') or '') or None
    public_base_url = str(_cfg_or_env(config, 'cos_public_base_url', 'COS_PUBLIC_BASE_URL', default='') or '') or None
    public_url_template = (
        str(_cfg_or_env(config, 'cos_public_url_template', 'COS_PUBLIC_URL_TEMPLATE', default='') or '') or None
    )

    missing = []
    if not secret_id:
        missing.append('cos_secret_id/COS_SECRET_ID')
    if not secret_key:
        missing.append('cos_secret_key/COS_SECRET_KEY')
    if not bucket:
        missing.append('cos_bucket/COS_BUCKET')
    if not prefix:
        missing.append('cos_prefix/COS_UPLOAD_PREFIX')
    if not region and not _cfg_or_env(config, 'cos_endpoint', 'COS_ENDPOINT'):
        missing.append('cos_region/COS_REGION or cos_endpoint/COS_ENDPOINT')
    if not region and not public_base_url and not public_url_template:
        missing.append('cos_region/COS_REGION or cos_public_base_url/COS_PUBLIC_BASE_URL')
    if missing:
        logger.info(f'[COS] COS upload enabled but missing config: {", ".join(missing)}; skipping COS upload')
        return None

    return CosUploadConfig(
        secret_id=secret_id,
        secret_key=secret_key,
        token=str(_cfg_or_env(config, 'cos_token', 'COS_TOKEN', default='') or '') or None,
        bucket=bucket,
        region=region,
        prefix=prefix,
        endpoint=str(_cfg_or_env(config, 'cos_endpoint', 'COS_ENDPOINT', default='') or '') or None,
        service_domain=str(_cfg_or_env(config, 'cos_service_domain', 'COS_SERVICE_DOMAIN', default='') or '') or None,
        scheme=str(_cfg_or_env(config, 'cos_scheme', 'COS_SCHEME', default='https') or 'https'),
        public_base_url=public_base_url.rstrip('/') if public_base_url else None,
        public_url_template=public_url_template,
        storage_class=str(_cfg_or_env(config, 'cos_storage_class', 'COS_STORAGE_CLASS', default='') or '') or None,
        acl=str(_cfg_or_env(config, 'cos_acl', 'COS_ACL', default='') or '') or None,
    )


def _get_cos_client(config: CosUploadConfig):
    try:
        from qcloud_cos import CosConfig, CosS3Client
    except ImportError:
        logger.info('[COS] qcloud_cos is not installed; install cos-python-sdk-v5>=1.9.42 to enable COS upload')
        return None

    cache_key = (
        config.secret_id,
        config.token or '',
        config.endpoint or '',
        config.service_domain or '',
        config.scheme,
        config.secret_key,
    )
    if cache_key in _cos_clients:
        return _cos_clients[cache_key]

    kwargs = {
        'Secret_id': config.secret_id,
        'Secret_key': config.secret_key,
        'Token': config.token,
        'Scheme': config.scheme,
    }
    if config.region:
        kwargs['Region'] = config.region
    if config.endpoint:
        kwargs['Endpoint'] = config.endpoint
    if config.service_domain:
        kwargs['ServiceDomain'] = config.service_domain

    cos_config = CosConfig(**kwargs)
    client = CosS3Client(cos_config)
    _cos_clients[cache_key] = client
    return client


def _cos_public_url(config: CosUploadConfig, key: str) -> str:
    escaped_key = quote(key)
    if config.public_url_template:
        return config.public_url_template.format(
            bucket=config.bucket,
            region=config.region or '',
            key=escaped_key,
        )
    if config.public_base_url:
        return f'{config.public_base_url}/{escaped_key}'
    return f'https://{config.bucket}.cos.{config.region}.myqcloud.com/{escaped_key}'


def _cos_upload_image(image_path: str, config: CosUploadConfig) -> str | None:
    client = _get_cos_client(config)
    if client is None:
        return None

    mime_type, _ = mimetypes.guess_type(image_path)
    content_type = mime_type if mime_type and mime_type.startswith('image/') else 'application/octet-stream'
    ext = os.path.splitext(image_path)[1].lstrip('.').lower() or 'bin'

    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    digest = hashlib.md5(image_data, usedforsecurity=False).hexdigest()
    key = f'{config.prefix}/{digest}.{ext}'
    put_kwargs = {
        'Bucket': config.bucket,
        'Body': io.BytesIO(image_data),
        'Key': key,
        'ContentType': content_type,
    }
    if config.storage_class:
        put_kwargs['StorageClass'] = config.storage_class
    client.put_object(**put_kwargs)

    if config.acl:
        client.put_object_acl(Bucket=config.bucket, Key=key, ACL=config.acl)

    return _cos_public_url(config, key)


def _parse_cache_paths(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(path).strip() for path in value if str(path).strip()]
    raw_value = str(value).strip()
    if not raw_value:
        return []
    if raw_value.startswith('['):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(path).strip() for path in parsed if str(path).strip()]
    separator = ',' if ',' in raw_value else ':'
    if separator in raw_value:
        return [path.strip() for path in raw_value.split(separator) if path.strip()]
    return [raw_value]


class ImageSearchCache:
    """Read-only JSONL cache for reverse image search results."""

    def __init__(self, cache_paths: object = None) -> None:
        self._cache: dict[str, dict] = {}
        for path in _parse_cache_paths(cache_paths):
            if os.path.isfile(path):
                self._load(path)
            else:
                logger.debug(f'[ImageSearchCache] Skipping missing cache file: {path}')

    def _load(self, path: str) -> None:
        loaded = 0
        with open(path, encoding='utf-8') as cache_file:
            for line in cache_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = entry.get('key') or entry.get('image_path')
                if key:
                    self._cache[str(key)] = entry
                    loaded += 1
        logger.debug(f'[ImageSearchCache] Loaded {loaded} entries from {path}')

    def get_raw_result(self, image_path: str) -> dict | None:
        entry = self._cache.get(image_path)
        return entry.get('raw_result') if entry else None

    def get_uploaded_url(self, image_path: str) -> str | None:
        entry = self._cache.get(image_path)
        return entry.get('uploaded_url') if entry else None


_global_cache: tuple[tuple[str, ...], ImageSearchCache] | None = None


def _get_cache(cache_paths: object = None) -> ImageSearchCache:
    global _global_cache
    paths = tuple(_parse_cache_paths(cache_paths or os.environ.get('IMAGE_SEARCH_CACHE_FILE')))
    if _global_cache is None or _global_cache[0] != paths:
        _global_cache = (paths, ImageSearchCache(list(paths)))
    return _global_cache[1]


# ============== Image Upload Helpers ==============
async def _upload_to_service(
    service_name: str,
    url: str,
    form: aiohttp.FormData,
    parse_fn,
    timeout: float = 30.0,
) -> str | None:
    """
    Upload an image to a single hosting service.
    Uses a dedicated aiohttp session with proxy support (trust_env=True)
    so that only upload requests go through the proxy.

    Args:
        service_name: Name of the service (for logging).
        url: Upload endpoint URL.
        form: aiohttp.FormData with file and extra fields.
        parse_fn: Callable(aiohttp.ClientResponse) -> Optional[str] to extract image URL.
        timeout: Request timeout in seconds.

    Returns:
        Image URL, or None on failure.
    """
    ct = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=ct, trust_env=True) as session:
            async with session.post(url, data=form) as resp:
                status = resp.status
                body = await resp.text()
                if status == 200:
                    uploaded_url = parse_fn(status, body)
                    if uploaded_url:
                        logger.debug(f'[ImageSearch] Successfully uploaded to {service_name}: {uploaded_url}')
                        return uploaded_url
                    logger.warning(f'[ImageSearch] {service_name}: invalid response format - {body[:200]}')
                else:
                    logger.warning(f'[ImageSearch] {service_name}: HTTP {status} - {body[:200]}')
    except asyncio.TimeoutError:
        logger.warning(f'[ImageSearch] {service_name}: upload timeout ({timeout}s)')
    except Exception as e:
        logger.warning(f'[ImageSearch] {service_name}: upload error - {type(e).__name__}: {e}')
    return None


async def upload_local_image(
    image_path: str,
    imbb_api_key: str | None = None,
    freeimage_api_key: str | None = None,
    cos_config: CosUploadConfig | None = None,
    max_retries: int = 1,
    timeout: float = 30.0,
) -> str | None:
    """
    Upload a local image to an image hosting service.
    Tries multiple services in priority order with automatic fallback.

    Args:
        image_path: Local image file path.
        max_retries: Max retries per service.
        timeout: Timeout in seconds per request.

    Returns:
        Public image URL, or None on failure.
    """
    # Validate file
    if not os.path.isfile(image_path):
        logger.error(f'[ImageSearch] File not found: {image_path}')
        return None

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and not mime_type.startswith('image/'):
        logger.error(f'[ImageSearch] Not an image file (MIME: {mime_type}): {image_path}')
        return None

    mime_type = mime_type or 'image/png'
    filename = os.path.basename(image_path)

    if cos_config:
        try:
            logger.debug('[ImageSearch] Trying Tencent COS upload...')
            loop = asyncio.get_running_loop()
            cos_url = await loop.run_in_executor(None, _cos_upload_image, image_path, cos_config)
            if cos_url:
                logger.debug(f'[ImageSearch] Successfully uploaded to Tencent COS: {cos_url}')
                return cos_url
            logger.info('[ImageSearch] COS upload unavailable, trying next configured image host')
        except Exception as exc:
            logger.warning(f'[ImageSearch] COS upload failed: {type(exc).__name__}: {exc}')

    has_imgbb = bool(imbb_api_key and len(imbb_api_key) > 1)
    has_freeimage = bool(freeimage_api_key and len(freeimage_api_key) > 1)
    if not has_imgbb and not has_freeimage:
        logger.error('[ImageSearch] Local upload enabled but no image hosting credentials were configured')
        return None

    with open(image_path, 'rb') as f:
        image_data = f.read()

    logger.debug(f'[ImageSearch] Uploading local image ({len(image_data)} bytes): {image_path}')

    # ---- Define service builders ----
    # Each builder returns (service_name, url, FormData, parse_fn)
    # We use builders (callables) because aiohttp.FormData is single-use.

    def _build_imgbb(data: bytes, fname: str, mime: str, api_key: str):
        image_b64 = base64.b64encode(data).decode('utf-8')
        form = aiohttp.FormData()
        form.add_field('key', api_key)
        form.add_field('image', image_b64)

        def parse(status, body):
            try:
                result = json.loads(body)
                if result.get('success'):
                    return result.get('data', {}).get('url')
            except Exception:
                pass
            return None

        return ('ImgBB', 'https://api.imgbb.com/1/upload', form, parse)

    def _build_freeimage(data: bytes, fname: str, mime: str):
        form = aiohttp.FormData()
        form.add_field('key', freeimage_api_key)
        form.add_field('source', data, filename=fname, content_type=mime)
        form.add_field('format', 'json')

        def parse(status, body):
            try:
                result = json.loads(body)
                return result.get('image', {}).get('url')
            except Exception:
                pass
            return None

        return ('Freeimage.host', 'https://freeimage.host/api/1/upload', form, parse)

    # ---- Build service list in priority order ----
    service_builders = []

    if has_imgbb and imbb_api_key is not None:
        service_builders.append(lambda d=image_data, f=filename, m=mime_type, k=imbb_api_key: _build_imgbb(d, f, m, k))

    if has_freeimage:
        service_builders.append(lambda d=image_data, f=filename, m=mime_type: _build_freeimage(d, f, m))

    # ---- Try each service with retries ----
    for builder in service_builders:
        svc_name = None
        for attempt in range(max_retries):
            # Rebuild FormData for each attempt (FormData is consumed after use)
            svc_name, svc_url, svc_form, svc_parse = builder()
            if attempt == 0:
                logger.debug(f'[ImageSearch] Trying {svc_name}...')

            result = await _upload_to_service(svc_name, svc_url, svc_form, svc_parse, timeout=timeout)
            if result:
                return result

            if attempt < max_retries - 1:
                await asyncio.sleep(2)

        logger.info(f'[ImageSearch] {svc_name} failed after {max_retries} attempts, trying next...')

    logger.error(f'[ImageSearch] All upload services failed for: {image_path}')
    return None


async def format_search_results(
    results: dict,
    query: str,
    num: int,
    *,
    return_images: bool = False,
    return_images_num: int | None = None,
    image_download_config: dict | None = None,
) -> str | list[ChatCompletionContentPartParam | dict]:
    image_items = results.get('organic', [])[:num] if results else []
    if not image_items:
        logger.error(f"No results found for '{query}'. results: {results}")
        return '[Image Search] No available image results found.'

    if not return_images:
        web_snippets = []
        for idx, page in enumerate(image_items, 1):
            source = page.get('source', '')
            image_url = page.get('thumbnailUrl') or page.get('imageUrl', '')
            redacted_version = (
                f'{idx}. [{page.get("title", "")}] Source: {source}  '
                f'image_url: {image_url} '
                f'Link: {page.get("link", "")}'
            )
            web_snippets.append(redacted_version)
        return f'Image search found {len(web_snippets)} results:\n\n## Image Results\n' + '\n\n'.join(web_snippets)

    embed_count = len(image_items) if return_images_num is None else max(0, min(return_images_num, len(image_items)))
    image_urls = [str(page.get('thumbnailUrl') or page.get('imageUrl') or '') for page in image_items[:embed_count]]
    image_urls = [image_url for image_url in image_urls if image_url]
    downloaded = await _download_images(image_urls, **(image_download_config or {})) if image_urls else []
    url_to_image = {url: (data, mime_type) for url, data, mime_type in downloaded}

    output_parts: list[ChatCompletionContentPartParam | dict] = [
        {'type': 'text', 'text': f'Image search found {len(image_items)} results:\n'}
    ]
    for idx, page in enumerate(image_items, 1):
        image_url = str(page.get('thumbnailUrl') or page.get('imageUrl') or '')
        output_parts.append({
            'type': 'text',
            'text': (
                f'\n\n{idx}. [{page.get("title", "")}]\nSource: {page.get("source", "")}\nLink: {page.get("link", "")}'
            ),
        })
        image_data, mime_type = url_to_image.get(image_url, (None, None))
        if image_data and mime_type:
            output_parts.append({
                'type': 'image_url',
                'image_url': {'url': _image_to_base64_url(image_data, mime_type)},
            })
    return output_parts


# ============== 核心 google_serper_search 实现 ==============
async def google_serper_search_raw(
    query: str,
    num: int,
    url: str,
    app_id: str,
    app_key: str,
    timeout: float = 30.0,
    retry_delay: float = 5.0,
    max_retries: int = 3,
):
    """
    执行 Google 搜索

    Args:
        query: 搜索查询字符串
        app_id: 应用 ID
        app_key: 应用密钥
        timeout: 每次请求的超时时间（秒），默认 10 秒
        retry_delay: 重试之间的延迟时间（秒），默认 5 秒
        max_retries: 最大重试次数，默认 3 次

    Returns:
        搜索结果字符串或错误信息
    """
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {app_id}:{app_key}?provider=serper'}
    results = None
    response = None

    for attempt in range(max_retries):
        try:
            response = await http_client.post(url, headers=headers, json={'url': query, 'n': num}, timeout=timeout)
            if not response.ok:
                logger.warning(
                    f'Google search attempt {attempt + 1}/{max_retries} failed with HTTP {response.status_code}, '
                    f'response body: {response.text[:500]}'
                )
                if attempt < max_retries - 1:
                    logger.debug(f'Retrying in {retry_delay} seconds...')
                    await asyncio.sleep(retry_delay)
                    continue
                response.raise_for_status()
            results = response.json()
            break
        except Exception as e:
            status_code = getattr(response, 'status_code', 'N/A') if response else 'N/A'
            logger.warning(
                f'Google search attempt {attempt + 1}/{max_retries} failed: {e}, response: {response} (status_code: {status_code})'
            )
            if attempt < max_retries - 1:
                logger.debug(f'Retrying in {retry_delay} seconds...')
                await asyncio.sleep(retry_delay)
            else:
                raise e

    if results is None:
        raise Exception('Search Timeout, return None, Please try again later.')

    return results


async def google_serper_search(
    query: str,
    num: int,
    url: str,
    app_id: str,
    app_key: str,
    timeout: float = 30.0,
    retry_delay: float = 5.0,
    max_retries: int = 3,
) -> str:
    results = await google_serper_search_raw(query, num, url, app_id, app_key, timeout, retry_delay, max_retries)
    formatted = await format_search_results(results, query, num)
    assert isinstance(formatted, str)
    return formatted


# ============== Nexus 工具适配层 ==============
class Image2ImageSearchInput(RobustBaseModel):
    """image to image search 工具输入参数"""

    image_path: str = Field(description='The search image to execute.')
    num_results: int = Field(default=10, description='Max search results to return, default is 10')


@tool_registry.register(
    'image_to_image_search',
    meta={
        'name': 'image_to_image_search',
        'description': (
            'Use this tool to search the web for images using image prompts. '
            'Returns relevant image results with titles, sources, and URLs.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def image_to_image_search_tool(params: Image2ImageSearchInput, context: ToolContext) -> ToolResponse:
    image_path = params.image_path
    num_results = params.num_results
    logger.debug(f'[log] Image to Image search query: {image_path}')
    if not image_path or not image_path.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty search query provided.'}],
            meta={'success': False},
        )

    # 从 task_config 读取配置，如果没有则使用环境变量
    cfg = context.tool_configs.get('image_to_image_search', {})
    url = cfg.get('serper_url', os.environ.get('I2I_SEARCH_SERPER_URL', ''))
    app_id = cfg.get('serper_app_id', os.environ.get('SEARCH_SERPER_APP_ID', ''))
    app_key = cfg.get('serper_app_key', os.environ.get('SEARCH_SERPER_APP_KEY', ''))
    imbb_api_key = cfg.get('imgbb_api_key', os.environ.get('IMGBB_API_KEY', ''))
    freeimage_api_key = cfg.get('freeimage_api_key', os.environ.get('FREEIMAGE_API_KEY', ''))
    cache_file = cfg.get('image_search_cache_file', os.environ.get('IMAGE_SEARCH_CACHE_FILE', ''))
    image_search_timeout = _as_float(cfg.get('image_search_timeout', os.environ.get('IMAGE_SEARCH_TIMEOUT')), 30.0)
    upload_timeout = _as_float(cfg.get('upload_timeout', os.environ.get('I2I_UPLOAD_TIMEOUT')), 30.0)
    upload_retries = _as_int(cfg.get('upload_retries', os.environ.get('I2I_UPLOAD_RETRIES')), 1)
    allow_local_upload = _as_bool(cfg.get('allow_local_upload'), _as_bool(os.environ.get('I2I_ALLOW_LOCAL_UPLOAD')))
    return_images = _as_bool(cfg.get('return_images'), _as_bool(os.environ.get('I2I_RETURN_IMAGES')))
    return_images_num = _as_int(cfg.get('return_images_num', os.environ.get('I2I_RETURN_IMAGES_NUM')), num_results)
    image_download_config = _image_download_config(dict(cfg))
    cos_config = _cos_upload_config(dict(cfg))

    cache = _get_cache(cache_file)
    cached_raw = cache.get_raw_result(image_path)
    if cached_raw:
        formatted = await format_search_results(
            cached_raw,
            image_path,
            num_results,
            return_images=return_images,
            return_images_num=return_images_num,
            image_download_config=image_download_config,
        )
        if isinstance(formatted, str):
            return ToolResponse(output=[{'type': 'text', 'text': formatted}], meta={'success': True, 'cached': True})
        return ToolResponse(output=formatted, meta={'success': True, 'cached': True})

    uploaded_url = image_path
    # NOTE(jaykaizhang): the path may name a sandbox-only file, either a task
    # image uploaded by mm_agent or a crop written by code_interpreter, so it is
    # resolved against the sandbox before the upload chain opens it. A staged
    # copy is ours to delete: only the resulting public URL is needed once the
    # upload succeeds, and keeping the file would leave one behind per call.
    staged: tuple[str, bool] | None = None
    try:
        is_remote_image = is_remote_uri(image_path)
        staged = None if is_remote_image else await stage_image_path(image_path)
        if staged is not None:
            if not allow_local_upload:
                raise ValueError(
                    'Local image upload is disabled. Provide a public image URL or set allow_local_upload=true.'
                )
            # NOTE(jaykaizhang): keyed by the model-supplied path, not the staged
            # one, because this cache is prebuilt offline against dataset paths.
            uploaded_url = cache.get_uploaded_url(image_path) or ''
            if not uploaded_url:
                uploaded_url = (
                    await upload_local_image(
                        staged[0],
                        imbb_api_key=imbb_api_key,
                        freeimage_api_key=freeimage_api_key,
                        cos_config=cos_config,
                        max_retries=upload_retries,
                        timeout=upload_timeout,
                    )
                    or ''
                )
            if not uploaded_url:
                raise RuntimeError(
                    f'Failed to upload local image file {image_path}. Provide a public image URL instead.'
                )
        elif not is_remote_image:
            # NOTE(jaykaizhang): distinguish "no such file" from "uploads disabled".
            # Conflating the two sends readers to check a flag that is already on.
            raise ValueError(
                f'No image file at {image_path} and it is not a public URL. '
                'Pass a path that exists, or a public image URL.'
            )
        logger.debug(f'[ImageSearch] Reverse image search for: {uploaded_url[:100]}')
    except Exception as e:
        logger.error(f'[log] Google Serper search failed: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: {e}'}],
            meta={'success': False},
        )
    finally:
        if staged is not None and staged[1]:
            discard_staged_copy(staged[0])

    try:
        raw_result = await google_serper_search_raw(
            uploaded_url,
            num_results,
            url,
            app_id,
            app_key,
            timeout=image_search_timeout,
        )
    except Exception as e:
        logger.error(f'[ImageSearch] Search error: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Search error: {e}'}],
            meta={'success': False},
        )

    result = await format_search_results(
        raw_result,
        image_path,
        num_results,
        return_images=return_images,
        return_images_num=return_images_num,
        image_download_config=image_download_config,
    )
    if isinstance(result, list):
        return ToolResponse(output=result, meta={'success': True})

    if result.startswith('Search Timeout') or result == '':
        return ToolResponse(
            output=[{'type': 'text', 'text': result or 'Search failed.'}],
            meta={'success': False},
        )

    return ToolResponse(
        output=[{'type': 'text', 'text': result}],
        meta={'success': True},
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image upload & reverse image search tool')
    parser.add_argument(
        '--image',
        '-i',
        type=str,
        default='',
        help='Local image path to upload and search',
    )
    parser.add_argument('--upload-only', action='store_true', help='Only upload, skip search')
    parser.add_argument('--num', '-n', type=int, default=4, help='Number of search results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    async def main():
        if not args.image:
            print('Error: --image is required')
            return
        # Step 1: Upload image
        img_url = await upload_local_image(
            args.image,
            imbb_api_key=os.environ.get('IMGBB_API_KEY', ''),
            freeimage_api_key=os.environ.get('FREEIMAGE_API_KEY', ''),
        )
        print(f'Uploaded URL: {img_url}')
        if not img_url:
            print('Upload failed, exiting.')
            return

        if args.upload_only:
            return

        # Step 2: Reverse image search
        url = os.environ.get('I2I_SEARCH_SERPER_URL', '')
        app_id = os.environ.get('SEARCH_SERPER_APP_ID', '')
        app_key = os.environ.get('SEARCH_SERPER_APP_KEY', '')
        print(f'app_id: {app_id}, app_key: {app_key}')
        result = await google_serper_search(img_url, args.num, url, app_id, app_key)
        print(result)

    asyncio.run(main())
