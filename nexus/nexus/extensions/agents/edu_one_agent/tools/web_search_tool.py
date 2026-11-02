"""web_search_tool for edu_one_agent."""

from __future__ import annotations

import os

import aiohttp

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


WEB_SEARCH_TOOL_ENDPOINT = os.environ.get('WEB_SEARCH_TOOL_ENDPOINT', 'http://11.141.126.149:8003/batch_tool_call')
WEB_SEARCH_TOOL_TIMEOUT = float(os.environ.get('WEB_SEARCH_TOOL_TIMEOUT', '30'))


class WebSearchInput(BaseModel):
    query: list[str] = Field(description='一组用于文本检索的子任务查询query')


@tool_registry.register(
    'edu_one_agent/web_search',
    meta={
        'name': 'web_search',
        'description': '通用搜索工具，可以根据query检索网络信息，返回搜到的网页内容。',
        'tool_type': ToolType.LOCAL,
    },
)
async def web_search_tool(params: WebSearchInput, context: ToolContext) -> ToolResponse:
    query = params.query
    if isinstance(query, str):
        query = [query]

    timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_TOOL_TIMEOUT)

    payload = {
        'calls': [
            {
                'call_id': 'nexus-edu-one-agent',
                'name': 'web_search',
                'version': 'v1.0',
                'arguments': {
                    'query': query,
                },
            }
        ],
        'options': {},
        'context_in': {
            'user': {'user_id': 'nexus-edu-one-agent-trainer'},
            'session': {'round': 1, 'history': [], 'raw_query': ''},
            'caller': {'call_source': 'training'},
        },
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            WEB_SEARCH_TOOL_ENDPOINT,
            json=payload,
            headers={
                'Content-Type': 'application/json',
                'User-Agent': 'nexus-edu-one-agent',
            },
        ) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)

    results = data.get('results', []) if isinstance(data, dict) else []
    texts: list[str] = []
    if results:
        chunks = results[0].get('for_model', []) or []
        texts = [c.get('content', '') for c in chunks if c.get('type') == 'text' and c.get('content')]
    text = '\n'.join(texts) if texts else '(无内容)'

    return ToolResponse(output=[{'type': 'text', 'text': text}])


if __name__ == '__main__':
    import asyncio
    import sys

    async def _main() -> int:
        queries = sys.argv[1:] or ['python asyncio 教程', '什么是大语言模型']

        print(f'[web_search_tool] queries={queries}')
        print(f'[web_search_tool] endpoint={WEB_SEARCH_TOOL_ENDPOINT}')

        params = WebSearchInput(query=queries)
        context = ToolContext()
        response = await web_search_tool(params, context)

        print('-' * 60)
        for part in response.output:
            if isinstance(part, dict) and part.get('type') == 'text':
                print(part.get('text', ''))
        print('-' * 60)
        return 0

    raise SystemExit(asyncio.run(_main()))
