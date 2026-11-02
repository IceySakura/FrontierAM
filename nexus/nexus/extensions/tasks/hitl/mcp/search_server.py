"""MCP stdio server exposing web search and fetch to Claude Code inside sandbox.

Run as: python -m nexus.extensions.tasks.hitl.mcp.search_server

Environment variables:
  SEARCH_API_KEY    Pre-assembled Authorization header value (required)
  SEARCH_API_URL    Search API endpoint (default: internal polaris gateway)

``web_fetch`` reaches the open internet and works. ``web_search`` answered
``402 Payment Required`` from the default gateway on 0822, with the key the run
scripts carry -- so a run that needs search results has to bring a key that gateway
accepts, or point ``SEARCH_API_URL`` elsewhere. The tool reports the status and the
model carries on, which is why this reads as a worker that chose not to search.
"""

import asyncio
import logging
import os

from typing import Any

import httpx

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

server = Server(name='nexus-search')

DEFAULT_SEARCH_URL = 'http://trpc-gpt-eval-search.production.polaris:8080/search'


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


async def _web_search(query: str, max_results: int) -> str:
    api_key = os.environ.get('SEARCH_API_KEY', '')
    if not api_key:
        return 'Error: SEARCH_API_KEY environment variable is required.'

    url = os.environ.get('SEARCH_API_URL', DEFAULT_SEARCH_URL)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json={'q': query, 'num': max_results}, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for idx, page in enumerate(data.get('organic', [])):
        title = page.get('title', '')
        link = page.get('link', '')
        snippet = page.get('snippet', '')
        date = page.get('date', '')
        entry = f'{idx + 1}. [{title}]({link})'
        if date:
            entry += f'\n   Date: {date}'
        if snippet:
            entry += f'\n   {snippet}'
        results.append(entry)

    if not results:
        return f'No results found for: {query}'
    return f'Found {len(results)} results for "{query}":\n\n' + '\n\n'.join(results)


# ---------------------------------------------------------------------------
# Web fetch
# ---------------------------------------------------------------------------


async def _web_fetch(url: str, max_chars: int) -> str:
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; NexusBot/1.0)'}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(url, headers=headers, timeout=20.0)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        if 'text/html' in content_type:
            return _extract_text_from_html(resp.text[: max_chars * 3])[:max_chars]
        return resp.text[:max_chars]


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML, stripping tags and scripts."""
    import re

    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r'<[^>]+>', ' ', html)
    html = re.sub(r'\s+', ' ', html)
    return html.strip()


# ---------------------------------------------------------------------------
# MCP tool registration
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name='web_search',
            description=(
                'Search the web and use the results to inform responses. '
                'Provides up-to-date information for current events, recent documentation, '
                'and data beyond the knowledge cutoff. Returns search results with titles, '
                'links, and snippets. Use the current year (2026) in queries when searching '
                'for recent information.'
            ),
            inputSchema={
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The search query. Use specific terms and include the year for recent topics.',
                    },
                    'max_results': {
                        'type': 'integer',
                        'description': 'Maximum number of results to return (default: 10).',
                        'default': 10,
                    },
                },
                'required': ['query'],
            },
        ),
        Tool(
            name='web_fetch',
            description=(
                'Fetch content from a URL, convert HTML to readable text, and return it. '
                'Use this to retrieve and analyze web content: documentation pages, API '
                'references, blog posts, code examples, etc. The URL must be fully-formed. '
                'Results may be truncated for very large pages.'
            ),
            inputSchema={
                'type': 'object',
                'properties': {
                    'url': {
                        'type': 'string',
                        'description': 'The fully-formed URL to fetch content from.',
                    },
                    'max_chars': {
                        'type': 'integer',
                        'description': 'Maximum characters to return (default: 15000).',
                        'default': 15000,
                    },
                },
                'required': ['url'],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == 'web_search':
        query = arguments.get('query', '').strip()
        if not query:
            return [TextContent(type='text', text='Error: empty query')]
        max_results = min(int(arguments.get('max_results', 10)), 10)
        try:
            text = await _web_search(query, max_results)
        except Exception as exc:
            text = f'Search failed: {exc}'
        return [TextContent(type='text', text=text)]

    if name == 'web_fetch':
        url = arguments.get('url', '').strip()
        if not url:
            return [TextContent(type='text', text='Error: empty url')]
        max_chars = min(int(arguments.get('max_chars', 15000)), 50000)
        try:
            text = await _web_fetch(url, max_chars)
        except Exception as exc:
            text = f'Fetch failed: {exc}'
        return [TextContent(type='text', text=text)]

    return [TextContent(type='text', text=f'Unknown tool: {name}')]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
