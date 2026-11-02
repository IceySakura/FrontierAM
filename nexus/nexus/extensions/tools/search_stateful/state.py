"""
State management for stateful search tools.
Based on SimpleBrowserStateActor from verl/tools/search_tool_stateful_woray.py
"""

import asyncio
import itertools
import logging
import re
import textwrap

from collections import OrderedDict
from typing import Optional
from urllib.parse import unquote

import pydantic

from .check_leak_question_n_gram import contains_question
from .config import get_default_config
from .page_contents import PageContents, SearchResult


logger = logging.getLogger(__name__)

ENC_NAME = 'o200k_base'
FIND_PAGE_LINK_FORMAT = '# 【{idx}†{title}】'
PARTIAL_INITIAL_LINK_PATTERN = re.compile(r'^[^【】]*】')
PARTIAL_FINAL_LINK_PATTERN = re.compile(r'【\d*(?:†(?P<content>[^†】]*)(?:†[^†】]*)?)?$')
LINK_PATTERN = re.compile(r'【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】')


class LRUDict(OrderedDict):
    """LRU cache dictionary with capacity limit."""

    def __init__(self, capacity: int = 8192, *args, **kwargs):
        self.capacity = capacity
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)


class SimpleBrowserStateList(pydantic.BaseModel):
    """Global state manager for browser states with thread-safe operations."""

    browser_states_dict: LRUDict = pydantic.Field(default_factory=lambda: LRUDict(capacity=16384))
    capacity: int = 16384

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.browser_states_dict = LRUDict(capacity=self.capacity)
        # Initialize lock as a private attribute (not a Pydantic field)
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        """Get or create the async lock (lazy initialization)."""
        if self._lock is None:
            try:
                self._lock = asyncio.Lock()
            except RuntimeError:
                # If no event loop is running, we'll create it when needed
                # This should not happen in normal async execution
                self._lock = asyncio.Lock()
        return self._lock

    async def add_browser_state(self, request_id: str, origin_question: str, config: dict) -> 'SimpleBrowserStateActor':
        """Add a new browser state actor and return it (thread-safe)."""
        lock = self._get_lock()
        async with lock:
            # Double-check pattern: check again inside lock to avoid race condition
            if request_id in self.browser_states_dict:
                return self.browser_states_dict[request_id]

            default_config = get_default_config()
            new_browser_state = SimpleBrowserStateActor(
                origin_question=origin_question,
                testset_name_to_filter_file=config.get(
                    'testset_name_to_filter_file',
                    default_config['testset_name_to_filter_file'],
                ),
            )
            new_browser_state.get_testset_name_to_filter_list()
            self.browser_states_dict[request_id] = new_browser_state
            return new_browser_state

    async def get_browser_state(self, request_id: str) -> Optional['SimpleBrowserStateActor']:
        """Get the browser state actor handle at the specified request_id (thread-safe)."""
        lock = self._get_lock()
        async with lock:
            if request_id in self.browser_states_dict:
                return self.browser_states_dict[request_id]
            return None


class SimpleBrowserStateActor(pydantic.BaseModel):
    """Browser state for a single request/session."""

    pages: dict[str, PageContents] = pydantic.Field(default_factory=dict)
    page_stack: list[str] = pydantic.Field(default_factory=list)
    view_tokens: int = 1024
    encoding_name: str = ENC_NAME
    origin_question: str = pydantic.Field(default_factory=str)
    testset_name_to_filter_file: str = pydantic.Field(default_factory=str)
    testset_name_to_filter_list: list[str] = pydantic.Field(default_factory=list)

    def get_testset_name_to_filter_list(self):
        """Read filter list from file."""
        try:
            with open(self.testset_name_to_filter_file, encoding='utf-8') as file:
                content = file.read()
                # 保留原样（不 strip）：filter 文件里的前导/尾随空格是有意为之，用于避免误伤（例如 " HLE" 只
                # 匹配带空格的整词，而不是任意包含 "hle" 三个字母的子串）.
                self.testset_name_to_filter_list = content.split('\n')
        except Exception as e:
            logger.warning(f'Failed to read filter file: {e}')
            self.testset_name_to_filter_list = []

    @property
    def current_cursor(self) -> int:
        return len(self.page_stack) - 1

    def add_page(self, page: PageContents) -> None:
        """Add a new page to the state."""
        self.pages[page.url] = page
        self.page_stack.append(page.url)

    def get_page(self, cursor: int = -1) -> PageContents:
        """Get page by cursor index."""
        if self.current_cursor < 0:
            raise ValueError('No pages to access!')
        if cursor == -1 or cursor == self.current_cursor:
            return self.pages[self.page_stack[-1]]
        try:
            page_url = self.page_stack[cursor]
        except TypeError as e:
            raise ValueError(f'`cursor` should be an integer, not `{type(cursor).__name__}`') from e
        except IndexError as e:
            raise ValueError(
                f'Cursor `{cursor}` is out of range. Available cursor indices: [0 - {self.current_cursor}].'
            ) from e
        return self.pages[page_url]

    def get_page_by_url(self, url: str) -> PageContents | None:
        """Get page by URL if exists."""
        if url in self.pages:
            return self.pages[url]
        return None

    def pop_page_stack(self) -> None:
        """Pop the last page from stack."""
        assert self.current_cursor >= 0, 'No page to pop!'
        self.page_stack.pop()

    def check_fetch_page(self, page: PageContents) -> bool:
        """Check if page should be blocked (contains question or testset name)."""
        # Test set detection
        for testset_name in self.testset_name_to_filter_list:
            if testset_name in page.text:
                return True

        # Question leak detection
        result, similarity_cn = contains_question(question=self.origin_question, doc=page.text, threshold=0.6)
        return bool(result)

    def _make_response(self, page: PageContents, cursor: int, body: str, scrollbar: str) -> tuple:
        """Create a tool response from page content."""
        from nexus.core import ToolResponse

        domain = maybe_truncate(unquote(page.url))
        header = f'{page.title}'
        if domain:
            header += f' ({domain})'
        header += f'\n**{scrollbar}**\n\n'

        content = f'[{cursor}] {header}{body}'
        return ToolResponse(output=[{'type': 'text', 'text': content}]), 0.0, {'success': True}

    async def show_page(self, loc: int = 0, num_lines: int = -1) -> tuple:
        """Show page content starting from loc."""
        page = self.get_page()
        cursor = self.current_cursor
        lines = wrap_lines(text=page.text)
        total_lines = len(lines)

        if loc >= total_lines:
            raise ValueError(f'Invalid location parameter: `{loc}`. Cannot exceed page maximum of {total_lines - 1}.')

        # Calculate end location
        if num_lines <= 0:
            # Compute number of lines to show based on view_tokens
            txt = join_lines(lines[loc:], add_line_numbers=True, offset=loc)
            if len(txt) > self.view_tokens:
                # Simple approximation: assume average 4 chars per token
                max_chars = self.view_tokens * 4
                end_loc = loc
                char_count = 0
                for i, line in enumerate(lines[loc:], start=loc):
                    char_count += len(line) + 1  # +1 for newline
                    if char_count > max_chars:
                        end_loc = i
                        break
                else:
                    end_loc = total_lines
            else:
                end_loc = total_lines
        else:
            end_loc = min(loc + num_lines, total_lines)

        lines_to_show = lines[loc:end_loc]
        body = join_lines(lines_to_show, add_line_numbers=True, offset=loc)

        scrollbar = f'viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}'

        return self._make_response(page, cursor, body, scrollbar)

    async def show_page_safely(self, loc: int = 0, num_lines: int = -1) -> tuple:
        """Show page content safely, handling errors."""
        try:
            return await self.show_page(loc=loc, num_lines=num_lines)
        except ValueError as e:
            self.pop_page_stack()
            from nexus.core import ToolResponse

            return (
                ToolResponse(output=[{'type': 'text', 'text': f'Error: {str(e)}'}]),
                0.0,
                {'success': False},
            )

    async def show_search_results(self, query: str, result_format: str = 'pager') -> tuple:
        """Show the search page on top of the stack, in whichever shape the caller asked for.

        'pager' is the viewport the browser tools share: numbered lines, a scrollbar, and links
        reduced to 【{id}†title†domain】 so the model follows them through ``open``. That hides
        full URLs, which leaves ``visit`` -- and any other tool taking a URL -- nothing to work
        on, so 'links' renders the hits as the flat markdown list web_search's search returns.
        Either way the page stays on the stack, so open/find keep working.
        """
        if result_format != 'links':
            return await self.show_page_safely(loc=0)

        from nexus.core import ToolResponse

        text = render_links(self.get_page().search_results, query)
        return ToolResponse(output=[{'type': 'text', 'text': text}]), 0.0, {'success': True}


# Global state manager instance
SIMPLE_BROWSER_STATE_LIST = SimpleBrowserStateList()


def join_lines(lines: list[str], add_line_numbers: bool = False, offset: int = 0) -> str:
    """Join lines with optional line numbers."""
    if add_line_numbers:
        return '\n'.join([f'L{i + offset}: {line}' for i, line in enumerate(lines)])
    return '\n'.join(lines)


def wrap_lines(text: str, width: int = 80) -> list[str]:
    """Wrap text into lines."""
    lines = text.split('\n')
    wrapped = itertools.chain.from_iterable(
        (textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False) if line else [''])
        for line in lines
    )
    return list(wrapped)


def strip_links(text: str) -> str:
    """Strip link markers from text."""
    text = re.sub(PARTIAL_INITIAL_LINK_PATTERN, '', text)
    text = re.sub(PARTIAL_FINAL_LINK_PATTERN, lambda mo: mo.group('content'), text)
    return re.sub(LINK_PATTERN, lambda mo: mo.group('content'), text)


def maybe_truncate(text: str, num_chars: int = 1024) -> str:
    """Truncate text if too long."""
    if len(text) > num_chars:
        text = text[: (num_chars - 3)] + '...'
    return text


def _render_link(index: int, result: SearchResult) -> str:
    date = f'\nDate published: {result.date}' if result.date else ''
    source = f'\nSource: {result.source}' if result.source else ''
    snippet = f'\n{result.snippet}' if result.snippet else ''
    return f'{index}. [{result.title}]({result.url}){date}{source}\n{snippet}'


def render_links(results: list[SearchResult] | None, query: str) -> str:
    """Render search hits as the flat markdown list web_search's search returns."""
    if not results:
        return f"No results found for '{query}'. Try with a more general query."

    entries = [_render_link(index, result) for index, result in enumerate(results, 1)]
    return f"A web search for '{query}' found {len(entries)} results:\n\n## Web Results\n" + '\n\n'.join(entries)
