"""
Stateful open tool for opening web pages.
Based on WebReadTool from verl/tools/search_tool_stateful_woray.py
"""

import logging

from typing import Literal

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .backend import BackendError, GoogleApiBackend
from .config import get_default_config
from .state import SIMPLE_BROWSER_STATE_LIST


logger = logging.getLogger(__name__)

VIEW_SOURCE_PREFIX = 'view-source:'


class OpenInput(BaseModel):
    """Open tool input parameters."""

    id: int | str = Field(
        default=-1,
        description='Link ID (number) or fully qualified URL (string)',
    )
    cursor: int = Field(default=-1, description='Page cursor indicator')
    loc: int = Field(default=-1, description='Starting line number')
    num_lines: int = Field(default=-1, description='Number of lines to show')
    view_source: bool = Field(default=False, description='Whether to view source')
    source: Literal['web', 'news'] = Field(
        default='web',
        description='Source associated with the URL',
    )


@tool_registry.register(
    'search_stateful/open',
    meta={
        'name': 'open',
        'description': (
            'Opens the link id from the page indicated by cursor starting at line number loc, showing num_lines lines. '
            'Valid link ids are displayed with the formatting: 【{id}†.*】. '
            'If cursor is not provided, the most recent page is implied. '
            'If id is a string, it is treated as a fully qualified URL associated with source. '
            'If loc is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available. '
            'Use this function without id to scroll to a new location of an opened page.'
        ),
        'tool_type': ToolType.LOCAL,
    },
)
async def open_tool(params: OpenInput, context: ToolContext) -> ToolResponse:
    """Open a web page and add it to browser state."""
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

    # Get browser state
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

    # Determine URL to open
    curr_page = None
    stay_on_current_page = False
    url = None
    snippet = None

    if isinstance(params.id, str):
        # Direct URL open
        url = params.id
    else:
        # Operate on a previously opened page
        try:
            curr_page = browser_state.get_page(params.cursor)
        except ValueError as e:
            return ToolResponse(
                output=[{'type': 'text', 'text': f'Error: {str(e)}'}],
                meta={'success': False},
            )

        if params.id >= 0:
            # Click a link
            try:
                url = curr_page.urls[str(params.id)]
            except KeyError:
                return ToolResponse(
                    output=[{'type': 'text', 'text': f'Invalid link id `{params.id}`.'}],
                    meta={'success': False},
                )
            snippet = (curr_page.snippets or {}).get(str(params.id))
        else:
            # Navigate to new position on the current page
            if not params.view_source:
                stay_on_current_page = True
            url = curr_page.url
            snippet = None

    # Handle view source
    if params.view_source:
        url = f'{VIEW_SOURCE_PREFIX}{url}'
        snippet = None

    # Fetch new page if needed
    new_page = None
    if stay_on_current_page:
        assert curr_page is not None
        new_page = curr_page
    else:
        # Get backend configuration
        tool_config = context.tool_configs.get('search_stateful', {})
        default_config = get_default_config()
        url_block_list_file = tool_config.get(
            'url_block_list_file',
            default_config['url_block_list_file'],
        )

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

        try:
            new_page = await backend.fetch(url)
        except BackendError as e:
            logger.error(f'Fetch error: {e}')
            return ToolResponse(
                output=[{'type': 'text', 'text': f'Failed to fetch URL: {str(e)}'}],
                meta={'success': False},
            )
        except Exception as e:
            logger.error(f'Unexpected fetch error: {e}')
            return ToolResponse(
                output=[{'type': 'text', 'text': f'Fetch error: {str(e)}'}],
                meta={'success': False},
            )

        # Check if page should be blocked
        if browser_state.check_fetch_page(new_page):
            new_page.text = 'This page has been blocked!'
            logger.warning('Page blocked due to question leak detection')

    # Add page to state
    browser_state.add_page(new_page)

    # Determine location to show
    loc = params.loc
    if loc < 0:
        if snippet is not None and snippet.line_idx is not None:
            loc = snippet.line_idx
            if loc > 4:
                loc -= 4
        else:
            loc = 0

    # Show page
    tool_response, score, metrics = await browser_state.show_page_safely(loc=loc, num_lines=params.num_lines)
    return tool_response
