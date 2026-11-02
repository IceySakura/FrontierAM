"""
Stateful search tool for web search.
Based on WebSearchTool from verl/tools/search_tool_stateful_woray.py
"""

import logging

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .backend import BackendError, GoogleApiBackend
from .config import get_default_config
from .state import SIMPLE_BROWSER_STATE_LIST


logger = logging.getLogger(__name__)


class SearchInput(BaseModel):
    """Search tool input parameters"""

    query: str = Field(description='The search query to execute.')
    topn: int = Field(default=10, description='Max search results to return, default is 10')
    source: Literal['web', 'news'] = Field(
        default='web',
        description='Source to search within',
    )


@tool_registry.register(
    'search_stateful/search',
    meta={
        'name': 'search',
        'description': (
            'Use this tool to search the web for information using Google. '
            'Returns relevant web pages with titles, links, and snippets.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def search_tool(params: SearchInput, context: ToolContext) -> ToolResponse:
    """Execute web search and store results in browser state."""
    query = params.query
    if not query or not query.strip():
        return ToolResponse(
            output=[{'type': 'text', 'text': 'Error: Empty search query provided.'}],
            meta={'success': False},
        )

    request_id = context.run_id
    if not request_id:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': 'Error: No run_id available in context. Cannot maintain state.',
                }
            ],
            meta={'success': False},
        )

    # Get or create browser state
    browser_state = await SIMPLE_BROWSER_STATE_LIST.get_browser_state(request_id)
    if not browser_state:
        # Get origin question from tool_state if available
        origin_question = context.tool_state.get('origin_question', '')
        tool_config = context.tool_configs.get('search_stateful', {})
        default_config = get_default_config()
        config = {
            'testset_name_to_filter_file': tool_config.get(
                'testset_name_to_filter_file',
                default_config['testset_name_to_filter_file'],
            ),
            'url_block_list_file': tool_config.get(
                'url_block_list_file',
                default_config['url_block_list_file'],
            ),
        }
        browser_state = await SIMPLE_BROWSER_STATE_LIST.add_browser_state(request_id, origin_question, config)

    # Get backend configuration
    tool_config = context.tool_configs.get('search_stateful', {})
    default_config = get_default_config()
    url_block_list_file = tool_config.get(
        'url_block_list_file',
        default_config['url_block_list_file'],
    )
    max_search_results = tool_config.get('max_search_results', 10)

    # Legacy 共享凭据（回退用）.
    user = tool_config.get('user', default_config.get('user', ''))
    apikey = tool_config.get('apikey', default_config.get('apikey', ''))

    # Serper / Jina 分开的凭据（未配置时由 backend 内部回退到 user/apikey）.
    serper_app_id = tool_config.get('serper_app_id', default_config.get('serper_app_id', ''))
    serper_app_key = tool_config.get('serper_app_key', default_config.get('serper_app_key', ''))
    jina_app_id = tool_config.get('jina_app_id', default_config.get('jina_app_id', ''))
    jina_app_key = tool_config.get('jina_app_key', default_config.get('jina_app_key', ''))

    # Get cache configuration
    use_cache = tool_config.get('use_cache', default_config.get('use_cache', True))
    search_cache_path = tool_config.get(
        'search_cache_path',
        default_config.get('search_cache_path', None),
    )
    visit_cache_path = tool_config.get(
        'visit_cache_path',
        default_config.get('visit_cache_path', None),
    )
    jina_provider = tool_config.get(
        'jina_provider',
        default_config.get('jina_provider', 'aws_jina'),
    )

    backend = GoogleApiBackend(
        source='google api',
        url_block_list_file=url_block_list_file,
        user=user,
        apikey=apikey,
        serper_app_id=serper_app_id,
        serper_app_key=serper_app_key,
        jina_app_id=jina_app_id,
        jina_app_key=jina_app_key,
        jina_provider=jina_provider,
        use_cache=use_cache,
        search_cache_path=search_cache_path,
        visit_cache_path=visit_cache_path,
        testset_filter_list=browser_state.testset_name_to_filter_list,
    )

    # Use topn from params if provided, otherwise use max_search_results from config
    topn = params.topn if params.topn > 0 else max_search_results

    try:
        search_page = await backend.search(query=query, topn=topn)
    except BackendError as e:
        logger.error(f'Search error: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Search failed: {str(e)}'}],
            meta={'success': False},
        )
    except Exception as e:
        logger.error(f'Unexpected search error: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Search error: {str(e)}'}],
            meta={'success': False},
        )

    # Add page to state and show it
    browser_state.add_page(search_page)
    result_format = tool_config.get('search_result_format', 'pager')
    tool_response, score, metrics = await browser_state.show_search_results(query, result_format)
    return tool_response
