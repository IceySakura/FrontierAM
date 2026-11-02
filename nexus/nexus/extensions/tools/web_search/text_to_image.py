"""
工具名称: Google Serper 文搜图
工具类型: LOCAL
说明: 使用 Serper.dev API 提供的低成本 Google 搜索服务(文搜图)
"""

import asyncio
import base64
import logging
import mimetypes
import os
import ssl

from openai.types.chat import ChatCompletionContentPartParam
from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.extensions.tools.web_search._config_utils import as_bool as _as_bool
from nexus.extensions.tools.web_search._config_utils import as_float as _as_float
from nexus.extensions.tools.web_search._config_utils import as_int as _as_int
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_IMAGE_DOWNLOAD_TIMEOUT = 10.0
DEFAULT_IMAGE_DOWNLOAD_MAX_SIZE = 4 * 1024 * 1024


def _guess_mime_type(url: str, content_type: str | None = None) -> str:
    if content_type:
        mime = content_type.split(';')[0].strip().lower()
        if mime.startswith('image/'):
            return mime
    mime, _ = mimetypes.guess_type(url.split('?')[0].split('#')[0])
    if mime and mime.startswith('image/'):
        return mime
    return 'image/png'


def _build_image_ssl(ca_file: str | None = None, insecure: bool = False) -> ssl.SSLContext | bool | None:
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


def _image_download_config(config: dict | None = None) -> dict:
    config = config or {}
    proxy = config.get('image_proxy') or os.environ.get('IMG_HTTP_PROXY') or os.environ.get('IMG_HTTPS_PROXY')
    ca_file = config.get('image_ca_file') or os.environ.get('IMG_HTTPS_CA_FILE') or os.environ.get('SANDBOX_PROXY_CA')
    insecure = _as_bool(config.get('image_ssl_insecure'), _as_bool(os.environ.get('IMG_SSL_INSECURE')))
    return {
        'timeout': _as_float(
            config.get('image_download_timeout', os.environ.get('IMAGE_DOWNLOAD_TIMEOUT')),
            DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
        ),
        'max_retries': _as_int(config.get('image_download_retries'), 1),
        'retry_delay': _as_float(config.get('image_download_retry_delay'), 2.0),
        'max_size': _as_int(
            config.get('image_download_max_size', os.environ.get('IMAGE_DOWNLOAD_MAX_SIZE')),
            DEFAULT_IMAGE_DOWNLOAD_MAX_SIZE,
        ),
        'proxy': proxy,
        'ssl': _build_image_ssl(ca_file, insecure),
    }


async def _download_image(
    url: str,
    *,
    timeout: float = DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    max_retries: int = 1,
    retry_delay: float = 2.0,
    max_size: int = DEFAULT_IMAGE_DOWNLOAD_MAX_SIZE,
    proxy: str | None = None,
    ssl: ssl.SSLContext | bool | None = None,
) -> tuple[bytes | None, str | None]:
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        ),
        'Referer': 'https://www.google.com/',
        'Accept': 'image/avif,image/webp,image/apng,image/png,image/*,*/*;q=0.8',
    }
    request_kwargs = {'headers': headers, 'timeout': timeout}
    if proxy:
        request_kwargs['proxy'] = proxy
    if ssl is not None:
        request_kwargs['ssl'] = ssl

    for attempt in range(max_retries):
        try:
            response = await http_client.get(url, **request_kwargs)
            if response.status_code == 200:
                if len(response.content) > max_size:
                    logger.warning(f'[ImageDownload] Image too large ({len(response.content)} bytes): {url}')
                    return None, None
                return response.content, _guess_mime_type(url, response.headers.get('Content-Type', ''))
            logger.warning(f'[ImageDownload] HTTP {response.status_code} for {url}: {(response.text or "")[:200]}')
        except asyncio.TimeoutError:
            logger.warning(
                f'[ImageDownload] Timeout ({timeout}s) downloading {url} (attempt {attempt + 1}/{max_retries})'
            )
        except Exception as exc:
            logger.warning(
                f'[ImageDownload] Error downloading {url} (attempt {attempt + 1}/{max_retries}): '
                f'{type(exc).__name__}: {exc}'
            )
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
    return None, None


async def _download_images(
    urls: list[str],
    *,
    timeout: float = DEFAULT_IMAGE_DOWNLOAD_TIMEOUT,
    max_retries: int = 1,
    retry_delay: float = 2.0,
    max_size: int = DEFAULT_IMAGE_DOWNLOAD_MAX_SIZE,
    proxy: str | None = None,
    ssl: ssl.SSLContext | bool | None = None,
) -> list[tuple[str, bytes | None, str | None]]:
    tasks = [
        _download_image(
            url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_size=max_size,
            proxy=proxy,
            ssl=ssl,
        )
        for url in urls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for url, result in zip(urls, results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f'[ImageDownload] Unexpected error for {url}: {result}')
            output.append((url, None, None))
        else:
            data, mime_type = result
            output.append((url, data, mime_type))
    return output


def _image_to_base64_url(data: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(data).decode('utf-8')
    return f'data:{mime_type};base64,{b64}'


# ============== 核心 google_serper_search 实现 ==============
async def google_serper_search(
    query: str,
    num: int,
    url: str,
    app_id: str,
    app_key: str,
    timeout: float = 30.0,
    retry_delay: float = 5.0,
    max_retries: int = 3,
    return_images: bool = False,
    return_images_num: int | None = None,
    image_download_config: dict | None = None,
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
            response = await http_client.post(url, headers=headers, json={'q': query, 'num': num}, timeout=timeout)
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

    image_items = results.get('images', [])[:num] if results else []
    if image_items:
        if return_images:
            embed_count = (
                len(image_items) if return_images_num is None else max(0, min(return_images_num, len(image_items)))
            )
            image_urls = [page.get('thumbnailUrl', '') for page in image_items[:embed_count]]
            image_urls = [image_url for image_url in image_urls if image_url]
            downloaded = await _download_images(image_urls, **(image_download_config or {})) if image_urls else []
            url_to_image = {url: (data, mime_type) for url, data, mime_type in downloaded}

            output_parts: list[ChatCompletionContentPartParam | dict] = [
                {
                    'type': 'text',
                    'text': f"A Google text image search for '{query}' found {len(image_items)} results:\n",
                }
            ]
            for idx, page in enumerate(image_items, 1):
                thumbnail_url = page.get('thumbnailUrl', '')
                output_parts.append({
                    'type': 'text',
                    'text': (
                        f'\n\n{idx}. [{page.get("title", "")}] '
                        f'Source: {page.get("source", "")}  image_url: {thumbnail_url}'
                    ),
                })
                image_data, mime_type = url_to_image.get(thumbnail_url, (None, None))
                if image_data and mime_type:
                    output_parts.append({
                        'type': 'image_url',
                        'image_url': {'url': _image_to_base64_url(image_data, mime_type)},
                    })
            return output_parts

        web_snippets = []
        for idx, page in enumerate(image_items, 1):
            redacted_version = (
                f'{idx}. [{page.get("title", "")}] Source: {page.get("source", "")}  '
                f'image_url: {page.get("thumbnailUrl", "")}'
            )
            web_snippets.append(redacted_version)

        return (
            f"A Google text image search for '{query}' found {len(web_snippets)} results:\n\n## Image Results\n"
            + '\n\n'.join(web_snippets)
        )
    logger.error(f'No results found for "{query}". results: {results}')
    return f"No results found for '{query}'. Try with a more general query."


# ============== Nexus 工具适配层 ==============
class Text2ImageSearchInput(RobustBaseModel):
    """Text to Image Search 工具输入参数"""

    query: str = Field(description='The search query to execute.')
    num_results: int = Field(default=10, description='Max search results to return, default is 10')


@tool_registry.register(
    'text_to_image_search',
    meta={
        'name': 'text_to_image_search',
        'description': (
            # "A web image search tool that returns information such as images and corresponding original webpage URLs based on input queries. Queries should be concise and clear. Break down complex questions into multiple steps. If no useful results are found, adjust the query by reducing qualifiers or changing the search approach. For better results, use Chinese queries for Chinese resources and English for non-Chinese resources."
            'Use this tool to search the web for images using text prompts. '
            'Returns relevant image results with titles, sources, and URLs.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def text_to_image_search_tool(params: Text2ImageSearchInput, context: ToolContext) -> ToolResponse:
    query = params.query
    num_results = params.num_results
    logger.debug(f'[log] Text to Image search query: {query}')
    if not query or not query.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty search query provided.'}],
            meta={'success': False},
        )

    # 从 task_config 读取配置，如果没有则使用环境变量
    cfg = context.tool_configs.get('text_to_image_search', {})
    url = cfg.get('serper_url', os.environ.get('T2I_SEARCH_SERPER_URL', ''))
    app_id = cfg.get('serper_app_id', os.environ.get('SEARCH_SERPER_APP_ID', ''))
    app_key = cfg.get('serper_app_key', os.environ.get('SEARCH_SERPER_APP_KEY', ''))
    text_to_image_timeout = _as_float(cfg.get('text_to_image_timeout', os.environ.get('T2I_SEARCH_TIMEOUT')), 30.0)
    return_images = _as_bool(cfg.get('return_images'), _as_bool(os.environ.get('T2I_RETURN_IMAGES')))
    return_images_num = _as_int(cfg.get('return_images_num', os.environ.get('T2I_RETURN_IMAGES_NUM')), num_results)

    # 调用核心搜索逻辑，传入配置的 app_id 和 app_key
    result = await google_serper_search(
        query,
        num_results,
        url,
        app_id,
        app_key,
        timeout=text_to_image_timeout,
        return_images=return_images,
        return_images_num=return_images_num,
        image_download_config=_image_download_config(dict(cfg)),
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
    import asyncio

    async def main():
        url = os.environ.get('T2I_SEARCH_SERPER_URL', '')
        app_id = os.environ.get('SEARCH_SERPER_APP_ID', '')
        app_key = os.environ.get('SEARCH_SERPER_APP_KEY', '')
        print(f'app_id: {app_id}, app_key: {app_key}')
        result = await google_serper_search('天安门', 4, url, app_id, app_key)
        print(result)

    asyncio.run(main())
