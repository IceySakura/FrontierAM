"""
Backend for search and fetch operations.
Simplified version based on verl/tools/backend.py
"""

import asyncio
import hashlib
import json
import logging
import os
import time

from typing import Any

from nexus.utils.async_http import HTTPResponse, http_client
from nexus.utils.common import configurable_retry

from .file_cache import initialize_caches
from .page_contents import PageContents, SearchResult, process_html


logger = logging.getLogger(__name__)

# Forward-proxy URL of the shared ATS POST cache (iwiki 4047616988), e.g. http://21.133.244.13:3080.
ATS_PROXY_ENV = 'NEXUS_SEARCH_ATS_PROXY'
ATS_CACHE_KEY_HEADER = 'X-Nexus-Cache-Key'


class BackendError(Exception):
    """Backend operation error."""

    pass


def ats_cache_key(path: str, body: dict[str, Any], provider: str, extra: dict[str, Any]) -> str:
    """Return the ATS cache key; it excludes credentials so every account shares one entry per request."""
    payload = {'method': 'POST', 'path': path, 'body': body, 'provider': provider, 'extra': extra}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class GoogleApiBackend:
    """Backend that uses Google Search API."""

    BASE_URL: str = 'http://trpc-gpt-eval.production.polaris:8080'
    _cache_logged: bool = False

    def __init__(
        self,
        source: str = 'google api',
        url_block_list_file: str = '',
        # Legacy 共享凭据（同时用于 serper 与 jina）——保留以向后兼容.
        user: str = '',
        apikey: str = '',
        # Serper 专用凭据（若未提供则回退到 user/apikey）.
        serper_app_id: str = '',
        serper_app_key: str = '',
        # Jina 专用凭据（若未提供则回退到 user/apikey）.
        jina_app_id: str = '',
        jina_app_key: str = '',
        jina_provider: str = 'aws_jina',
        use_cache: bool = True,
        search_cache_path: str | None = None,
        visit_cache_path: str | None = None,
        # anti-hack：命中列表中任何条目的搜索结果（title/snippet/link）会被丢弃.
        testset_filter_list: list[str] | None = None,
        # None reads NEXUS_SEARCH_ATS_PROXY; empty disables the shared ATS cache.
        ats_proxy: str | None = None,
    ):
        self.source = source
        self.ats_proxy = os.getenv(ATS_PROXY_ENV, '') if ats_proxy is None else ats_proxy
        if self.ats_proxy:
            # ATS is the shared cache; a sandbox-local FileCache would only hide repeated requests from it.
            use_cache = False
        self.url_block_list_file = url_block_list_file

        self.serper_user = serper_app_id or user
        self.serper_apikey = serper_app_key or apikey
        self.jina_user = jina_app_id or user
        self.jina_apikey = jina_app_key or apikey

        self.jina_provider = jina_provider
        self.use_cache = use_cache
        self.search_cache = None
        self.visit_cache = None
        # 保留原样（不 strip）：filter 文件里的前导/尾随空格是有意为之，用于避免误伤匹配.
        # 只在下面的循环里跳过空字符串（避免 split 尾部 '' 命中所有内容）.
        self.testset_filter_list = list(testset_filter_list or [])

        if not self.serper_user or not self.serper_apikey:
            raise ValueError(
                'GoogleApiBackend requires Serper credentials. Please set SEARCH_SERPER_APP_ID / '
                'SEARCH_SERPER_APP_KEY (or legacy SEARCH_STATEFUL_USER / SEARCH_STATEFUL_APIKEY), '
                'or configure serper_app_id/serper_app_key in tool_configs.search_stateful'
            )
        if not self.jina_user or not self.jina_apikey:
            raise ValueError(
                'GoogleApiBackend requires Jina credentials. Please set DEFAULT_JINA_APP_ID / '
                'DEFAULT_JINA_APP_KEY (or legacy SEARCH_STATEFUL_USER / SEARCH_STATEFUL_APIKEY), '
                'or configure jina_app_id/jina_app_key in tool_configs.search_stateful'
            )

        # Initialize caches lazily when the backend is actually used.
        if self.use_cache:
            self.search_cache, self.visit_cache = initialize_caches(search_cache_path, visit_cache_path)
            if not GoogleApiBackend._cache_logged:
                logger.debug(
                    'File cache已开启，存储路径为: search=%s, visit=%s',
                    self.search_cache.base_dir,
                    self.visit_cache.base_dir,
                )
                GoogleApiBackend._cache_logged = True
        elif self.ats_proxy and not GoogleApiBackend._cache_logged:
            logger.info('ATS cache enabled via %s; file cache disabled', self.ats_proxy)
            GoogleApiBackend._cache_logged = True

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str],
        provider: str,
        extra: dict[str, Any],
        **kwargs: Any,
    ) -> HTTPResponse:
        """POST to BASE_URL, through the ATS forward proxy with a cache key when it is configured."""
        if not self.ats_proxy:
            return await http_client.post(url=self.BASE_URL + path, json=body, headers=headers, **kwargs)
        key = ats_cache_key(path, body, provider, extra)
        start = time.monotonic()
        response = await http_client.post(
            url=self.BASE_URL + path,
            json=body,
            # X-Debug asks ATS to report X-Cache (miss / hit-fresh / skipped) on the response.
            headers={**headers, ATS_CACHE_KEY_HEADER: key, 'X-Debug': 'x-cache'},
            proxy=self.ats_proxy,
            **kwargs,
        )
        logger.info(
            '[ats] path=%s key=%s status=%s x_cache=%s ms=%.0f',
            path,
            key[:16],
            response.status_code,
            response.headers.get('X-Cache') or response.headers.get('x-cache', ''),
            (time.monotonic() - start) * 1000,
        )
        return response

    async def search(self, query: str, topn: int) -> PageContents:
        """Perform Google search using Serper API and return results as PageContents."""
        # Check cache first
        cached = await self.search_cache.load_async(query) if self.use_cache and self.search_cache is not None else None

        if cached is not None:
            # logger.warning(f'[CACHE HIT][SEARCH] {query}')
            results = cached.get('results', [])
            data = cached.get('raw_response', {})
        else:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.serper_user}:{self.serper_apikey}?provider=serper&timeout=60',
            }
            payload = {
                'q': query,
                'num': topn,
            }

            # Retry mechanism
            data = None
            for attempt in range(3):
                try:
                    response = await self._post('/search', payload, headers, 'serper', {})
                    if response.status_code != 200:
                        raise BackendError(f'GoogleApiBackend error {response.status_code}: {response.text}')
                    data = response.json()
                    if self.ats_proxy and not (data or {}).get('organic'):
                        # ATS stores any 200, so an empty answer may be pinned under the base key;
                        # one request under a derived key reaches the origin again.
                        response = await self._post('/search', payload, headers, 'serper', {'retry': 1})
                        if response.status_code == 200:
                            data = response.json()
                    break
                except Exception as e:
                    if attempt == 2:
                        if not isinstance(e, BackendError):
                            raise BackendError(f'GoogleApiBackend search request error: {str(e)}') from e
                        raise
                    logger.warning(f'Search attempt {attempt + 1} failed: {e}')
                    await asyncio.sleep(1)

            if not data:
                raise BackendError('No results searched!')
            results = data.get('organic', [])

            # Save to cache
            if self.use_cache and self.search_cache is not None:
                self.search_cache.save_in_background(query, {'results': results, 'raw_response': data})

        if not results:
            raise BackendError('No results searched!')

        # Load URL block list
        url_block_list = []
        if self.url_block_list_file:
            try:
                with open(self.url_block_list_file, encoding='utf-8') as file:
                    content = file.read()
                    url_block_list = [url.strip() for url in content.split('\n')]
            except Exception as e:
                logger.warning(f'Failed to read URL block list: {e}')

        titles_and_urls = []
        for result in results:
            ban_flag = False
            title = result.get('title', '')
            link = result.get('link', '')
            snippet = result.get('snippet', '')

            for url_block in url_block_list:
                if url_block and url_block in link:
                    logger.warning(f'URL blocked: {url_block}')
                    ban_flag = True
                    break

            if not ban_flag and self.testset_filter_list:
                # anti-hack：如果搜索结果 summary（title/snippet/link）中泄露了测试集名称，直接跳过该条结果.
                combined = f'{title}\n{snippet}\n{link}'
                for testset_name in self.testset_filter_list:
                    if testset_name and testset_name in combined:
                        logger.warning(f'Search result blocked due to testset name leak: {testset_name}')
                        ban_flag = True
                        break

            if not ban_flag:
                titles_and_urls.append((title, link, snippet))

        # Create HTML page from results
        html_page = f"""<html><body>
<h1>Search Results for: {query}</h1>
<ul>
{''.join([f"<li><a href='{url}'>{title}</a><p>{snippet}</p></li>" for title, url, snippet in titles_and_urls[:topn]])}
</ul>
</body></html>
"""

        page = process_html(
            html=html_page,
            url='',
            title=f'Search Results for: {query}',
            display_urls=True,
            session=None,
        )

        # Keep the hits themselves on the page. Rendering them out of page.text would mean parsing
        # html2text output back apart, and date/source never reach the HTML at all.
        by_link = {result.get('link', ''): result for result in results}
        page.search_results = [
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                date=by_link.get(url, {}).get('date', ''),
                source=by_link.get(url, {}).get('source', ''),
            )
            for title, url, snippet in titles_and_urls[:topn]
        ]
        return page

    @configurable_retry(max_attempts=3)
    async def fetch(self, url: str) -> PageContents:
        """Fetch a URL and return as PageContents."""
        # Check cache first
        cache = await self.visit_cache.load_async(url) if self.use_cache and self.visit_cache is not None else None

        if cache is not None:
            # logger.warning(f'[CACHE HIT][VISIT] {url}')
            data = cache.get('data', {})
            if data:
                # Extract text and title from cached data
                text = data.get('data', {}).get('text', None)
                title = data.get('data', {}).get('title', None)
                if text:
                    return PageContents(
                        url=url,
                        text=text,
                        urls={},
                        title=title,
                    )
                logger.warning(f'Cache data text is empty for URL: {url}')
            else:
                logger.warning(f'Cache data is empty for URL: {url}')

        # Fetch from API if not in cache or cache miss
        # Use Jina API to fetch page content
        jina_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.jina_user}:{self.jina_apikey}?provider={self.jina_provider}',
            'X-Return-Format': 'text',
        }
        path = '/api/jina_ai/search'
        extra = {'X-Return-Format': 'text'}

        try:
            response = await self._post(path, {'url': url}, jina_headers, self.jina_provider, extra, timeout=60)
            if response.status_code != 200:
                raise BackendError(f'Failed to fetch URL {url}: {response.status_code} - {response.text}')
            data = response.json()
            if self.ats_proxy and not ((data or {}).get('data') or {}).get('text'):
                response = await self._post(
                    path, {'url': url}, jina_headers, self.jina_provider, {**extra, 'retry': 1}, timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
        except Exception as e:
            if not isinstance(e, BackendError):
                raise BackendError(f'Error fetching URL `{url}`: {str(e)}') from e
            raise

        if not data:
            raise BackendError(f'No contents returned for {url}')

        text = data.get('data', {}).get('text', None)
        title = data.get('data', {}).get('title', None)

        if not text:
            raise BackendError(f'No contents returned for {url}, text为空')

        # Save to cache
        if self.use_cache and self.visit_cache is not None:
            self.visit_cache.save_in_background(url, {'data': data})

        return PageContents(
            url=url,
            text=text,
            urls={},
            title=title,
        )
