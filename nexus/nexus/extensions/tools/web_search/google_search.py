from __future__ import annotations

import logging
import os

from typing import Any

import aiohttp

from pydantic import Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.core.types import RobustBaseModel


logger = logging.getLogger(__name__)


# Error messages
_ERR_CONFIG = 'Configuration Error: GOOGLE_API_URL and GOOGLE_API_KEY required'
_ERR_TIMEOUT = 'Search request timed out.'
_ERR_NO_RESULTS = 'No good Google Search Result was found'
_ERR_NO_CONTENT = 'No content available for these results.'
_ERR_UNAVAILABLE = 'Service temporarily unavailable.'
_HTTP_ERRORS = {401: 'Authentication failed. Check your API key.', 429: 'Rate limit exceeded. Try again later.'}


_WEB_SEARCH_DESCRIPTION = """\
A search engine optimized for comprehensive, accurate, and trusted results.
Useful for when you need to answer questions about current events. Input should be a search query."""


class GoogleSearchInput(RobustBaseModel):
    query: str = Field(description='search query to look up')
    num_results: int = Field(default=10, description='Max search results to return, default is 10')


@tool_registry.register(
    'google_search',
    meta={
        'name': 'google_search',
        'description': _WEB_SEARCH_DESCRIPTION,
        'tool_type': ToolType.LOCAL,
    },
)
async def google_search_tool(params: GoogleSearchInput, context: ToolContext) -> ToolResponse:
    """Execute a Google search and return formatted results."""
    query = params.query
    cfg = context.tool_configs.get('google_search', {})
    api_url = cfg.get('api_url') or os.environ.get('GOOGLE_API_URL')
    api_key = cfg.get('api_key') or os.environ.get('GOOGLE_API_KEY')
    timeout = cfg.get('timeout_seconds', 60)

    if not api_url or not api_key:
        return ToolResponse(
            output=[{'type': 'text', 'text': _ERR_CONFIG}], meta={'error': 'config_error', 'query': query}
        )

    try:
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(
                f'{api_url.rstrip("/")}/customsearch/v1',
                headers={'Authorization': f'Bearer {api_key}'},
                params={'q': query, 'num': params.num_results},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                items: list[dict[str, Any]] = data.get('items', [])

    except aiohttp.ClientError as e:
        if isinstance(e, aiohttp.ServerTimeoutError):
            return ToolResponse(
                output=[{'type': 'text', 'text': _ERR_TIMEOUT}], meta={'error': 'timeout', 'query': query}
            )
        if isinstance(e, aiohttp.ClientResponseError):
            code = e.status
            msg = _HTTP_ERRORS.get(code) or (_ERR_UNAVAILABLE if code >= 500 else f'API error: {code}')
            return ToolResponse(
                output=[{'type': 'text', 'text': msg}],
                meta={'error': 'http_error', 'status_code': code, 'query': query},
            )
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Request failed: {str(e)}'}],
            meta={'error': 'client_error', 'query': query},
        )

    if not items:
        return ToolResponse(output=[{'type': 'text', 'text': _ERR_NO_RESULTS}], meta={'query': query, 'num_results': 0})

    results = [{'title': item.get('title'), 'link': item.get('link'), 'snippet': item.get('snippet')} for item in items]
    snippets = [r['snippet'] for r in results if r.get('snippet')]
    links = [r['link'] for r in results if r.get('link')]

    # Fallback: if no snippets, show titles with links
    output_text = (
        ' '.join(snippets)
        if snippets
        else ' | '.join(f'{r["title"]} ({r["link"]})' for r in results if r.get('title') and r.get('link'))
    )

    return ToolResponse(
        output=[{'type': 'text', 'text': output_text or _ERR_NO_CONTENT}],
        meta={'query': query, 'num_results': len(results), 'links': links, 'results': results},
    )


if __name__ == '__main__':
    import asyncio

    async def main():
        api_url = os.environ.get('GOOGLE_API_URL')
        api_key = os.environ.get('GOOGLE_API_KEY')
        print(f'API URL: {api_url}, API Key: {api_key}')
        context = ToolContext(tool_configs={'google_search': {'api_url': api_url, 'api_key': api_key}})
        result = await google_search_tool(GoogleSearchInput(query='Python programming', num_results=3), context=context)
        print(result)

    asyncio.run(main())
