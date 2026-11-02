"""Search Stateful MCP Server — pure JSON-RPC 2.0 over stdio.

A standalone MCP server for Claude Code that exposes search/open/find/visit tools,
reusing nexus's ``GoogleApiBackend`` and ``SimpleBrowserStateActor``. Clients pick
a page-reading style with their own tool allow-list (e.g. Codex ``enabled_tools``):
``open`` + ``find`` page through raw text, ``visit`` returns a goal-oriented summary.

This file does not import the ``mcp`` Python SDK directly. Instead it
implements the minimal MCP protocol subset (newline-delimited JSON-RPC 2.0
over stdio, per MCP spec) by hand. However, it imports nexus modules
(``backend``, ``state``, ``find``, etc.) which transitively pull in parts of
``nexus.core``. The ``nexus.core`` package *does* depend on the ``mcp`` SDK
at import time (via ``nexus.core.tool`` lazy exports), so the runtime venv
must have ``mcp`` installed (it is listed in pyproject.toml).

Env vars (set by Claude Code from the MCP server config):
  Credentials (需要 serper 和 jina 各一对，二者可以不同)：
    - Serper: ``SEARCH_SERPER_APP_ID`` / ``SEARCH_SERPER_APP_KEY``
    - Jina:   ``DEFAULT_JINA_APP_ID``  / ``DEFAULT_JINA_APP_KEY``
    - Legacy 共享凭据（作为回退）：``SEARCH_USER`` / ``SEARCH_APIKEY``
      以及 ``SEARCH_STATEFUL_USER`` / ``SEARCH_STATEFUL_APIKEY`` （在 config.py 中生效）
  Summariser for ``visit`` (same keys as the native visit tool, see ``visit.py``):
    - ``SEARCH_STATEFUL_SUMMARY_CLIENT=openai`` + ``DEFAULT_JINA_MODEL_APP_KEY``
      (+ optional ``DEFAULT_JINA_MODEL`` / ``SEARCH_STATEFUL_SUMMARY_BASE_URL``) for AIHub 2.0;
    - otherwise the legacy gpt-eval path via ``DEFAULT_JINA_MODEL`` and
      ``DEFAULT_JINA_MODEL_APP_ID/KEY``, falling back to the search credentials.
  ORIGIN_QUESTION      — optional, enables question-leak detection
  NEXUS_SEARCH_ATS_PROXY — optional ATS forward proxy; replaces the file cache with the shared ATS cache
  LD_LIBRARY_PATH      — required on Alpine, ``/nix/runtime/nix-env/lib``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sys

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from nexus.core import OpenAICompatibleClient
    from nexus.extensions.tools.search_stateful.backend import GoogleApiBackend
    from nexus.extensions.tools.search_stateful.state import SimpleBrowserStateActor
    from nexus.extensions.tools.search_stateful.visit import SummaryConfig


# ---------------------------------------------------------------------------
# Logging — stderr only (stdout is the JSON-RPC transport)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='[mcp-search] %(levelname)s %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports — deferred so the module header stays lightweight
# ---------------------------------------------------------------------------

_backend: GoogleApiBackend | None = None
_state: SimpleBrowserStateActor | None = None
_url_block_list: list[str] = []
_summariser: tuple[SummaryConfig, OpenAICompatibleClient | None] | None = None


def _get_backend() -> GoogleApiBackend:
    if _backend is None:
        raise RuntimeError('Search backend has not been initialised')
    return _backend


def _get_state() -> SimpleBrowserStateActor:
    if _state is None:
        raise RuntimeError('Browser state has not been initialised')
    return _state


def _get_summariser() -> tuple[SummaryConfig, OpenAICompatibleClient | None]:
    """Resolve the visit summariser once; raises ValueError while credentials are missing."""
    global _summariser
    if _summariser is None:
        from nexus.extensions.tools.search_stateful.visit import resolve_summariser

        _summariser = resolve_summariser({})
    return _summariser


def _init() -> None:
    """Initialise the search backend and browser state from env vars."""
    global _backend, _state, _url_block_list

    from nexus.extensions.tools.search_stateful.backend import GoogleApiBackend
    from nexus.extensions.tools.search_stateful.config import get_default_config
    from nexus.extensions.tools.search_stateful.state import SimpleBrowserStateActor
    from nexus.extensions.tools.search_stateful.visit import _load_url_block_list

    cfg = get_default_config()

    # 兼容旧的 SEARCH_USER/SEARCH_APIKEY 以及旧的 config user/apikey.
    user = os.environ.get('SEARCH_USER', cfg.get('user', ''))
    apikey = os.environ.get('SEARCH_APIKEY', cfg.get('apikey', ''))

    # Serper / Jina 独立凭据；未提供时回退到共享 user/apikey.
    serper_app_id = cfg.get('serper_app_id', '') or user
    serper_app_key = cfg.get('serper_app_key', '') or apikey
    jina_app_id = cfg.get('jina_app_id', '') or user
    jina_app_key = cfg.get('jina_app_key', '') or apikey

    if not serper_app_id or not serper_app_key or not jina_app_id or not jina_app_key:
        logger.error(
            'Search stateful MCP server requires Serper and Jina credentials via '
            'SEARCH_SERPER_APP_ID/KEY, DEFAULT_JINA_APP_ID/KEY, or legacy SEARCH_USER/SEARCH_APIKEY'
        )
        sys.exit(1)

    origin_question = os.environ.get('ORIGIN_QUESTION', '')
    _state = SimpleBrowserStateActor(
        origin_question=origin_question,
        testset_name_to_filter_file=cfg.get('testset_name_to_filter_file', ''),
    )
    _state.get_testset_name_to_filter_list()

    _backend = GoogleApiBackend(
        source='google api',
        url_block_list_file=cfg.get('url_block_list_file', ''),
        serper_app_id=serper_app_id,
        serper_app_key=serper_app_key,
        jina_app_id=jina_app_id,
        jina_app_key=jina_app_key,
        jina_provider=cfg.get('jina_provider', 'aws_jina'),
        testset_filter_list=_state.testset_name_to_filter_list,
    )
    _url_block_list = _load_url_block_list(cfg.get('url_block_list_file', ''))

    logger.debug(
        'Backend and state initialised (serper_user=%s, jina_user=%s, origin_q=%d chars)',
        serper_app_id,
        jina_app_id,
        len(origin_question),
    )


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 over stdio — newline-delimited (MCP spec)
#
# Per https://modelcontextprotocol.io/specification/2025-03-26/basic/transports:
#   "Messages are delimited by newlines, and MUST NOT contain embedded newlines."
# ---------------------------------------------------------------------------


async def _read_message(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one newline-delimited JSON-RPC message. Returns ``None`` on EOF."""
    while True:
        line = await reader.readline()
        if not line:
            return None  # EOF
        line = line.strip()
        if not line:
            continue  # skip blank lines
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning('Skipping malformed JSON line: %s', exc)
            continue


def _write_message(msg: dict[str, Any]) -> None:
    """Write one newline-delimited JSON-RPC message to stdout."""
    # Ensure no embedded newlines (MCP spec requirement)
    line = json.dumps(msg, ensure_ascii=False, separators=(',', ':'))
    sys.stdout.write(line + '\n')
    sys.stdout.flush()


def _ok(request_id: Any, result: Any) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': request_id, 'result': result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}}


# ---------------------------------------------------------------------------
# Tool schemas — descriptions aligned with the training-data schema overrides
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        'name': 'search',
        'description': (
            'Use this tool to search the web for information using Google. '
            'Returns relevant web pages with titles, links, and snippets.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string', 'description': 'The search query to execute.'},
                'topn': {
                    'type': 'integer',
                    'description': 'Max search results to return, default is 10',
                    'default': 10,
                },
            },
            'required': ['query'],
        },
    },
    {
        'name': 'open',
        'description': (
            'Opens the link id from the page indicated by cursor starting at line '
            'number loc, showing num_lines lines. Valid link ids are displayed with '
            'the formatting: 【{id}†.*】. If cursor is not provided, the most recent '
            'page is implied. If id is a string, it is treated as a fully qualified '
            'URL associated with source. If loc is not provided, the viewport will '
            'be positioned at the beginning of the document or centered on the most '
            'relevant passage, if available. Use this function without id to scroll '
            'to a new location of an opened page.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'id': {
                    'anyOf': [{'type': 'integer'}, {'type': 'string'}],
                    'description': 'Link ID (number) or fully qualified URL (string)',
                    'default': -1,
                },
                'cursor': {
                    'type': 'integer',
                    'description': 'Page cursor indicator',
                    'default': -1,
                },
                'loc': {
                    'type': 'integer',
                    'description': 'Starting line number',
                    'default': -1,
                },
                'num_lines': {
                    'type': 'integer',
                    'description': 'Number of lines to show',
                    'default': -1,
                },
            },
        },
    },
    {
        'name': 'find',
        'description': ('Finds exact matches of pattern in the current page, or the page given by cursor.'),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'pattern': {'type': 'string', 'description': 'The exact pattern to search for'},
                'cursor': {
                    'type': 'integer',
                    'description': 'Page cursor to search in',
                    'default': -1,
                },
            },
            'required': ['pattern'],
        },
    },
    {
        'name': 'visit',
        'description': (
            'Visit webpage(s) and return the summary of the content. '
            'Use this tool to extract relevant information from web pages based on a specific goal.'
        ),
        'inputSchema': {
            'type': 'object',
            'properties': {
                'url': {
                    'anyOf': [{'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}],
                    'description': 'The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.',
                },
                'goal': {'type': 'string', 'description': 'The goal of the visit for webpage(s).'},
            },
            'required': ['url', 'goal'],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_text(tool_response: object) -> str:
    """Extract plain text from a nexus ``ToolResponse``."""
    output = getattr(tool_response, 'output', None)
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(item.get('text', ''))
        return '\n'.join(parts)
    return str(output) if output else ''


def _text_content(text: str) -> list[dict[str, Any]]:
    """Wrap *text* in MCP ``TextContent`` format."""
    return [{'type': 'text', 'text': text}]


def _error_content(text: str) -> dict[str, Any]:
    """Return MCP call_tool error result."""
    return {'content': _text_content(text), 'isError': True}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _do_search(args: dict[str, Any]) -> dict[str, Any]:
    """Execute search and return MCP result."""
    from nexus.extensions.tools.search_stateful.backend import BackendError

    backend = _get_backend()
    state = _get_state()

    query = args.get('query', '')
    topn = args.get('topn', 10)
    if not query or not query.strip():
        return _error_content('Error: Empty search query provided.')

    try:
        search_page = await backend.search(query=query, topn=topn)
    except BackendError as exc:
        return _error_content(f'Search failed: {exc}')
    except Exception as exc:
        return _error_content(f'Search error: {exc}')

    state.add_page(search_page)
    resp, _, _ = await state.show_page_safely(loc=0)
    return {'content': _text_content(_extract_text(resp))}


async def _do_open(args: dict[str, Any]) -> dict[str, Any]:
    """Open a page by link id or URL, or scroll the current page."""
    from nexus.extensions.tools.search_stateful.backend import BackendError

    backend = _get_backend()
    state = _get_state()

    link_id = args.get('id', -1)
    cursor = args.get('cursor', -1)
    loc = args.get('loc', -1)
    num_lines = args.get('num_lines', -1)

    # Determine URL to open
    stay_on_current_page = False
    url = None
    snippet = None

    if isinstance(link_id, str):
        url = link_id
    else:
        try:
            curr_page = state.get_page(cursor)
        except ValueError as exc:
            return _error_content(f'Error: {exc}')

        if link_id >= 0:
            try:
                url = curr_page.urls[str(link_id)]
            except KeyError:
                return _error_content(f'Invalid link id `{link_id}`.')
            snippet = (curr_page.snippets or {}).get(str(link_id))
        else:
            stay_on_current_page = True
            url = curr_page.url
            snippet = None

    if stay_on_current_page:
        new_page = state.get_page(cursor)
    else:
        try:
            new_page = await backend.fetch(url)
        except BackendError as exc:
            return _error_content(f'Failed to fetch URL: {exc}')
        except Exception as exc:
            return _error_content(f'Fetch error: {exc}')

        if state.check_fetch_page(new_page):
            new_page.text = 'This page has been blocked!'
            logger.warning('Page blocked due to question leak detection')

    state.add_page(new_page)

    # Determine display location
    if loc < 0:
        if snippet is not None and hasattr(snippet, 'line_idx') and snippet.line_idx is not None:
            loc = max(snippet.line_idx - 4, 0)
        else:
            loc = 0

    resp, _, _ = await state.show_page_safely(loc=loc, num_lines=num_lines)
    return {'content': _text_content(_extract_text(resp))}


async def _do_find(args: dict[str, Any]) -> dict[str, Any]:
    """Find a pattern in the current page."""
    from nexus.extensions.tools.search_stateful.find import run_find_in_page

    pattern = args.get('pattern', '')
    cursor = args.get('cursor', -1)
    if not pattern:
        return _error_content('Error: Empty pattern provided.')

    browser_state = _get_state()
    if browser_state.current_cursor < 0:
        return _error_content('Error: No browser state found. Please use search or open first.')

    try:
        page = browser_state.get_page(cursor)
    except ValueError as exc:
        return _error_content(f'Error: {exc}')

    if page.snippets is not None:
        return _error_content('Cannot run `find` on search results page or find results page')

    try:
        result_page = await run_find_in_page(pattern=str(pattern).lower(), page=page)
    except Exception as exc:
        return _error_content(f'Find error: {exc}')

    browser_state.add_page(result_page)
    resp, _, _ = await browser_state.show_page_safely(loc=0)
    return {'content': _text_content(_extract_text(resp))}


async def _do_visit(args: dict[str, Any]) -> dict[str, Any]:
    """Fetch page(s) and return goal-oriented summaries, as the native visit tool does."""
    from pydantic import ValidationError

    from nexus.extensions.tools.search_stateful.visit import VisitInput, visit_pages

    try:
        params = VisitInput.model_validate(args)
    except ValidationError as exc:
        return _error_content(f'Error: Invalid visit arguments: {exc}')
    goal = params.goal
    if not goal.strip():
        return _error_content('Error: Empty goal provided.')
    urls = [params.url] if isinstance(params.url, str) else params.url
    if not urls:
        return _error_content('Error: Empty url provided.')

    try:
        summary_config, openai_summary_client = _get_summariser()
    except ValueError as exc:
        return _error_content(f'Error: {exc}')

    text, success = await visit_pages(
        urls, goal, _get_backend(), _get_state(), _url_block_list, summary_config, openai_summary_client
    )
    if not success:
        return _error_content(text)
    return {'content': _text_content(text)}


_TOOL_DISPATCH = {
    'search': _do_search,
    'open': _do_open,
    'find': _do_find,
    'visit': _do_visit,
}


# ---------------------------------------------------------------------------
# MCP protocol dispatch
# ---------------------------------------------------------------------------

SERVER_INFO = {
    'name': 'search-stateful',
    'version': '0.1.0',
}

CAPABILITIES = {
    'tools': {},
}


async def _dispatch(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message. Returns a response dict, or ``None`` for notifications."""
    method = msg.get('method', '')
    msg_id = msg.get('id')  # None for notifications
    params = msg.get('params', {})

    # --- Notifications (no id → no response) ---
    if msg_id is None:
        if method == 'notifications/initialized':
            logger.debug('Client sent notifications/initialized')
        elif method == 'notifications/cancelled':
            logger.info('Client cancelled request: %s', params.get('requestId'))
        else:
            logger.debug('Ignoring notification: %s', method)
        return None

    # --- Requests (have id → must respond) ---
    if method == 'initialize':
        return _ok(
            msg_id,
            {
                'protocolVersion': params.get('protocolVersion', '2024-11-05'),
                'capabilities': CAPABILITIES,
                'serverInfo': SERVER_INFO,
            },
        )

    if method == 'ping':
        return _ok(msg_id, {})

    if method == 'tools/list':
        return _ok(msg_id, {'tools': TOOL_SCHEMAS})

    if method == 'tools/call':
        tool_name = params.get('name', '')
        tool_args = params.get('arguments', {})
        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return _ok(msg_id, _error_content(f'Unknown tool: {tool_name}'))
        try:
            result = await handler(tool_args)
        except Exception as exc:
            logger.exception('Tool %s failed', tool_name)
            result = _error_content(f'Internal error: {exc}')
        return _ok(msg_id, result)

    # resources/list, prompts/list — return empty
    if method in ('resources/list', 'resources/templates/list', 'prompts/list'):
        items_key = 'prompts' if 'prompts' in method else 'resources' if 'resources' in method else 'items'
        return _ok(msg_id, {items_key: []})

    # Unknown method
    return _error(msg_id, -32601, f'Method not found: {method}')


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def _main() -> None:
    _init()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    logger.debug('MCP search-stateful server ready')

    while True:
        msg = await _read_message(reader)
        if msg is None:
            logger.debug('EOF on stdin, shutting down')
            break
        logger.debug('← %s', msg.get('method', '?'))

        response = await _dispatch(msg)
        if response is not None:
            _write_message(response)
            logger.debug('→ id=%s', response.get('id'))


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt, BrokenPipeError):
        asyncio.run(_main())
