"""
开发信息: DrakosXiao(2025.12), mannaliang(2025.12.23 重构)
工具名称: Google Serper 搜索工具
工具类型: LOCAL
说明: 使用 Serper.dev API 提供的低成本 Google 搜索服务
"""

import asyncio
import logging
import os

from urllib.parse import urlparse

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel
from nexus.extensions.agents.base.errors import (
    NonRetryableProviderError,
    raise_for_non_retryable_provider_error,
)
from nexus.extensions.tools.web_search._config_utils import as_bool as _as_bool
from nexus.extensions.tools.web_search._config_utils import as_float as _as_float
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXCLUDED_SEARCH_SITES = ('huggingface.co', 'hf.co')


def _parse_url_blacklist(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(',') if item.strip()]


def _is_blacklisted_result_url(url: str, blacklist: list[str]) -> bool:
    if not url or not blacklist:
        return False

    parsed_url = urlparse(url if '://' in url else f'https://{url}')
    hostname = (parsed_url.hostname or '').lower().rstrip('.')
    for blocked in blacklist:
        blocked = blocked.strip()
        if not blocked:
            continue
        if '://' in blocked:
            if url.startswith(blocked):
                return True
            continue

        parsed_blocked = urlparse(f'https://{blocked}')
        blocked_hostname = (parsed_blocked.hostname or '').lower().rstrip('.')
        if blocked_hostname and (hostname == blocked_hostname or hostname.endswith(f'.{blocked_hostname}')):
            return True
    return False


def _is_excluded_search_result_url(url: str, block_huggingface: bool = True) -> bool:
    """Check whether the result URL belongs to an excluded domain."""
    if not block_huggingface:
        return False
    if not url:
        return False
    parsed = urlparse(url if '://' in url else f'https://{url}')
    hostname = (parsed.hostname or '').lower().removeprefix('www.')
    return any(hostname == site or hostname.endswith(f'.{site}') for site in EXCLUDED_SEARCH_SITES)


# ============== 核心 google_serper_search 实现 ==============
async def google_serper_search(
    query: str,
    num: int,
    url: str,
    app_id: str,
    app_key: str,
    timeout: float = 120.0,
    retry_delay: float = 10.0,
    max_retries: int = 3,
    block_huggingface: bool = True,
    use_cache: bool = False,
    cache_dir: str | None = None,
    url_blacklist: list[str] | None = None,
):
    """
    执行 Google 搜索

    Args:
        query: 搜索查询字符串
        app_id: 应用 ID
        app_key: 应用密钥
        timeout: 每次请求的超时时间（秒），默认 60 秒
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
            raise_for_non_retryable_provider_error(getattr(response, 'text', ''))
            response.raise_for_status()
            results = response.json()
            break
        except NonRetryableProviderError:
            logger.error(
                f'[google_serper_search] stopped on terminal platform error for query={query!r}', exc_info=True
            )
            raise
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

    web_snippets = []
    idx = 0
    if len(results) > 0 and 'organic' in results:
        for page in results['organic']:
            page_link = page.get('link', '')
            if _is_excluded_search_result_url(page_link, block_huggingface=block_huggingface):
                logger.debug(f'[google_serper_search] filtered excluded result url: {page_link}')
                continue
            if _is_blacklisted_result_url(page_link, url_blacklist or []):
                logger.debug(f'[google_serper_search] filtered blacklisted result url: {page_link}')
                continue
            idx += 1
            date_published = ''
            if 'date' in page:
                date_published = '\nDate published: ' + page['date']

            source = ''
            if 'source' in page:
                source = '\nSource: ' + page['source']

            snippet = ''
            if 'snippet' in page:
                snippet = '\n' + page['snippet']

            redacted_version = f'{idx}. [{page["title"]}]({page["link"]}){date_published}{source}\n{snippet}'
            redacted_version = redacted_version.replace("Your browser can't play this video.", '')
            web_snippets.append(redacted_version)

        if web_snippets:
            return f"A web search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + '\n\n'.join(
                web_snippets
            )

    logger.error(f'No results found for "{query}". results: {results}')
    return f"No results found for '{query}'. Try with a more general query."


# ============== Nexus 工具适配层 ==============
class GoogleSerperSearchInput(RobustBaseModel):
    query: str = Field(description='The search query to execute.')
    num_results: int = Field(default=10, description='Max search results to return, default is 10')


@tool_registry.register(
    'google_serper_search',
    meta={
        'name': 'google_search',
        'description': (
            'Use this tool to search the web for information using Google. '
            'Returns relevant web pages with titles, links, and snippets.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def google_serper_search_tool(params: GoogleSerperSearchInput, context: ToolContext) -> ToolResponse:
    query = params.query
    num_results = params.num_results
    logger.debug(f'[log] Google Serper search query: {query}')
    if not query or not query.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty search query provided.'}],
            meta={'success': False},
        )

    # 从 task_config 读取配置，如果没有则使用环境变量
    cfg = context.tool_configs.get('google_serper_search', {})
    url = cfg.get('serper_url', os.environ.get('SEARCH_SERPER_URL', ''))
    app_id = cfg.get('serper_app_id', os.environ.get('SEARCH_SERPER_APP_ID', ''))
    app_key = cfg.get('serper_app_key', os.environ.get('SEARCH_SERPER_APP_KEY', ''))
    text_search_timeout = _as_float(cfg.get('text_search_timeout', os.environ.get('TEXT_SEARCH_TIMEOUT')), 120.0)
    default_block_huggingface = _as_bool(os.getenv('DEFAULT_SERPER_BLOCK_HUGGINGFACE'), True)
    block_huggingface = _as_bool(cfg.get('block_huggingface'), default_block_huggingface)
    use_cache = _as_bool(cfg.get('use_cache'), _as_bool(os.environ.get('SEARCH_SERPER_USE_CACHE')))
    cache_dir = cfg.get('cache_dir', os.environ.get('SEARCH_CACHE_DIR', None))
    url_blacklist = _parse_url_blacklist(cfg.get('url_blacklist', os.environ.get('SEARCH_URL_BLACKLIST')))

    # 调用核心搜索逻辑，传入配置的 app_id 和 app_key
    try:
        result = await google_serper_search(
            query,
            num_results,
            url,
            app_id,
            app_key,
            timeout=text_search_timeout,
            block_huggingface=block_huggingface,
            use_cache=use_cache,
            cache_dir=cache_dir,
            url_blacklist=url_blacklist,
        )
    except NonRetryableProviderError:
        raise
    except Exception as e:
        logger.error(f'[google_serper_search] tool error for query={query!r}: {e}', exc_info=True)
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error during search: {str(e)}'}],
            meta={'success': False},
        )

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
        url = os.environ.get('SEARCH_SERPER_URL', '')
        app_id = os.environ.get('SEARCH_SERPER_APP_ID', '')
        app_key = os.environ.get('SEARCH_SERPER_APP_KEY', '')
        print(f'app_id: {app_id}, app_key: {app_key}')
        result = await google_serper_search('天安门', 4, url, app_id, app_key)
        print(result)

    asyncio.run(main())
