"""
Development Info: Nexus Framework Adapter (2025)
Tool Name: Hunyuan Search
Tool Type: LOCAL
"""

import logging
import traceback
import uuid

from datetime import datetime
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry
from nexus.utils.async_http import http_client


# Setup Logger
logger = logging.getLogger(__name__)

# --- Constants & Configuration ---
API_URL = 'http://hunyuanapi.woa.com/api/workflow/search'
DEFAULT_COOKIE = (
    'Cookie: x-client-ssid=4d622415:0198f4fcd5ae:10cdaf; x_host_key_access=8861ce0ec6d4dedb02e4c39f2d6cd73a6316b309_s'
)


# --- Internal Data Structures ---
class SearchResult(NamedTuple):
    """Immutable, lightweight structure for internal result passing."""

    title: str
    link: str
    snippet: str
    position: int
    publish_time: datetime | None


class HunyuanSearchInput(BaseModel):
    """Input schema for the Nexus tool registry."""

    query: str = Field(
        ..., description='A natural language search query string describing what information or context to retrieve.'
    )
    max_results: int = Field(
        10,
        ge=1,
        le=20,
        description='The maximum number of search results to return, controlling how many top items the search should fetch.',
    )


def _parse_publish_datetime(datetime_str: str):
    try:
        return datetime.strptime(datetime_str, '%Y年%m月%d日')
    except ValueError:
        return None


def _create_payload(query: str) -> dict[str, Any]:
    """Generates the massive API payload with fresh UUIDs."""
    req_id = uuid.uuid4().hex
    return {
        'model': 'hunyuan-search',
        'query': query,
        'request_id': req_id,
        'extra_params': {
            'business_id': '0',
            'doc_id_list': [],
            'entry': 'toc',
            'search_engine_types': ['mp', 'web', 'sogou', 'wx_extend'],
            'profile': {
                'age': 0,
                'gender': 0,
                'marriage': '',
                'eduaction': 0,  # Typo in API: eduaction
                'artical_ecate_level1': '',
                'artical_ecate_level2': '',
                'artical_ecate_level3': '',
                'prompt_interest': '',
                'residence_city_id': '',
            },
            'source': None,
            'uin_list': [],
            'use_quote': 0,
            'xinan_prompt_code': 100,
            'event_app_name': '',
            'event_name': '',
            'event_scene': '',
            'hongyi_shixin': False,
            'is_multimedia': False,
            'is_news_bro_request': False,
            'is_phone': False,
            'is_sport': False,
        },
        'se_params': {
            'search_type': 0,
            'session_id': '11111',
            'user_request_id': req_id,
            'event_report_req_id': uuid.uuid4().hex,
            'event_multiple': 6,
            'event_report': True,
            'is_extend': False,
            'sites': [],
            'recall_strategy': {'is_q_prime': True, 'recall_expand_num': 5},
        },
    }


async def _execute_search(query: str, api_key: str, cookie: str, max_results: int) -> list[SearchResult]:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'Cookie': cookie,
    }

    try:
        response = await http_client.post(API_URL, headers=headers, json=_create_payload(query), timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.error(f'Hunyuan API IO Error: {e}')
        raise RuntimeError(f'Search API connection failed: {e}') from e

    results: list[SearchResult] = []
    # Using 'get' with defaults to prevent KeyErrors on unstable API responses
    raw_list = data.get('search_results', [])

    for idx, item in enumerate(raw_list):
        if len(results) >= max_results:
            break

        url = item.get('url', '')
        if not url:
            continue

        # Quick fix for protocol
        if not url.startswith('http'):
            url = f'http://{url}'

        publish_time = _parse_publish_datetime(item.get('publish_time', ''))
        results.append(
            SearchResult(
                title=item.get('title', 'Untitled'),
                link=url,
                snippet=item.get('abs') or item.get('quote') or 'No description available.',
                position=idx + 1,
                publish_time=publish_time,
            )
        )

    return results


def format_results_for_llm(results: list[SearchResult]) -> str:
    if not results:
        return 'No relevant search results found.'

    return f'Found {len(results)} search results:\n' + '\n\n'.join(
        f'{r.position}. Title: {r.title}\n    URL: {r.link}\n    '
        f'Published on: {r.publish_time.strftime("%Y-%m-%d") if r.publish_time else "Unknown"}\n    '
        f'Summary: {r.snippet.replace(chr(10), " ").strip()}'
        for r in results
    )


@tool_registry.register(
    'hunyuan_search',
    meta={
        'name': 'hunyuan_search',
        'description': 'Search the web using Hunyuan Search. Returns titles, links, and snippets.',
        'tool_type': ToolType.LOCAL,
    },
)
async def hunyuan_search_tool(params: HunyuanSearchInput, context: ToolContext) -> ToolResponse:
    """Entry point for the Hunyuan Search tool."""
    query = params.query.strip()
    if not query:
        return ToolResponse(output=[{'type': 'text', 'text': 'Error: Empty query.'}], meta={'success': False})

    config = context.tool_configs.get('hunyuan_search', {})
    if not (api_key := config.get('api_key')):
        return ToolResponse(
            output=[{'type': 'text', 'text': "Error: Missing 'api_key' in tool_config."}], meta={'success': False}
        )

    cookie = config.get('cookie', DEFAULT_COOKIE)

    try:
        results = await _execute_search(query, api_key, cookie, params.max_results)
        return ToolResponse(
            output=[{'type': 'text', 'text': format_results_for_llm(results)}],
            meta={'success': True, 'count': len(results)},
        )
    except Exception as e:
        # Log full trace for devs, return simple error to agent
        logger.error(f'Search tool critical failure: {traceback.format_exc()}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Search failed: {str(e)}'}], meta={'success': False, 'error': str(e)}
        )
