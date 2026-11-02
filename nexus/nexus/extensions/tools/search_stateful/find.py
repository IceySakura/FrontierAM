"""
Stateful find tool for searching within a page.
Based on WebFindTool from verl/tools/search_tool_stateful_woray.py
"""

import logging

from urllib.parse import quote

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .page_contents import Extract, PageContents
from .state import (
    FIND_PAGE_LINK_FORMAT,
    SIMPLE_BROWSER_STATE_LIST,
    join_lines,
    strip_links,
    wrap_lines,
)


logger = logging.getLogger(__name__)


class FindInput(BaseModel):
    """Find tool input parameters."""

    pattern: str = Field(description='The exact pattern to search for')
    cursor: int = Field(default=-1, description='Page cursor to search in')


async def run_find_in_page(
    pattern: str,
    page: PageContents,
    max_results: int = 50,
    num_show_lines: int = 4,
) -> PageContents:
    """Find pattern in page and return results as a new PageContents."""
    lines = wrap_lines(text=page.text)
    txt = join_lines(lines, add_line_numbers=False)
    without_links = strip_links(txt)
    lines = without_links.split('\n')

    result_chunks, snippets = [], []
    line_idx, match_idx = 0, 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if pattern not in line.lower():
            line_idx += 1
            continue
        snippet = '\n'.join(lines[line_idx : line_idx + num_show_lines])
        link_title = FIND_PAGE_LINK_FORMAT.format(idx=f'{match_idx}', title=f'match at L{line_idx}')
        result_chunks.append(f'{link_title}\n{snippet}')
        snippets.append(Extract(url=page.url, text=snippet, title=f'#{match_idx}', line_idx=line_idx))
        if len(result_chunks) == max_results:
            break
        match_idx += 1
        line_idx += num_show_lines

    urls = [page.url for _ in result_chunks]

    display_text = '\n\n'.join(result_chunks) if result_chunks else f'No `find` results for pattern: `{pattern}`'

    return PageContents(
        url=f'{page.url}/find?pattern={quote(pattern)}',
        title=f'Find results for text: `{pattern}` in `{page.title}`',
        text=display_text,
        urls={str(i): url for i, url in enumerate(urls)},
        snippets={str(i): snip for i, snip in enumerate(snippets)},
    )


@tool_registry.register(
    'search_stateful/find',
    meta={
        'name': 'find',
        'description': 'Finds exact matches of pattern in the current page, or the page given by cursor.',
        'tool_type': ToolType.LOCAL,
    },
)
async def find_tool(params: FindInput, context: ToolContext) -> ToolResponse:
    """Find pattern in current page and add results to browser state."""
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
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': 'Error: No browser state found. Please use search or open first.',
                }
            ],
            meta={'success': False},
        )

    # Get current page
    try:
        page = browser_state.get_page(params.cursor)
    except ValueError as e:
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Error: {str(e)}'}],
            meta={'success': False},
        )

    # Check if page is a search results page or find results page
    if page.snippets is not None:
        return ToolResponse(
            output=[
                {
                    'type': 'text',
                    'text': 'Cannot run `find` on search results page or find results page',
                }
            ],
            meta={'success': False},
        )

    # Run find
    try:
        result_page = await run_find_in_page(
            pattern=str(params.pattern).lower(),
            page=page,
        )
    except Exception as e:
        logger.error(f'Find error: {e}')
        return ToolResponse(
            output=[{'type': 'text', 'text': f'Find error: {str(e)}'}],
            meta={'success': False},
        )

    # Add result page to state
    browser_state.add_page(result_page)

    # Show results
    tool_response, score, metrics = await browser_state.show_page_safely(loc=0)
    return tool_response
