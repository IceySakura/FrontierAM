"""Unit tests for the visit tool exposed by the search_stateful stdio MCP server."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tools.search_stateful import mcp_server
from nexus.extensions.tools.search_stateful import visit as visit_module
from nexus.extensions.tools.search_stateful.backend import BackendError, GoogleApiBackend
from nexus.extensions.tools.search_stateful.page_contents import PageContents
from nexus.extensions.tools.search_stateful.state import SimpleBrowserStateActor


QUESTION = 'What is the minimum number of moves required to solve a 3x3x3 Rubik cube in the worst case?'
SUMMARY_TEXT = 'EVIDENCE: the cube needs 20 moves.\n\nSUMMARY: God number is 20.'


def _page(url: str, text: str) -> PageContents:
    return PageContents(url=url, text=text, title='A page', urls={})


@pytest.fixture(autouse=True)
def server(monkeypatch: pytest.MonkeyPatch) -> SimpleBrowserStateActor:
    """Install a fresh backend, browser state and AIHub summariser on the server module."""
    for name in ('SEARCH_STATEFUL_SUMMARY_CLIENT', 'DEFAULT_JINA_MODEL', 'SEARCH_STATEFUL_SUMMARY_BASE_URL'):
        monkeypatch.delenv(name, raising=False)
    state = SimpleBrowserStateActor(origin_question=QUESTION)
    backend = GoogleApiBackend(
        source='google api',
        serper_app_id='serper-id',
        serper_app_key='serper-key',
        jina_app_id='jina-id',
        jina_app_key='jina-key',
        use_cache=False,
    )
    monkeypatch.setattr(mcp_server, '_backend', backend)
    monkeypatch.setattr(mcp_server, '_state', state)
    monkeypatch.setattr(mcp_server, '_summariser', None)
    return state


@pytest.fixture
def summary_client(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    monkeypatch.setenv('SEARCH_STATEFUL_SUMMARY_CLIENT', 'openai')
    monkeypatch.setenv('DEFAULT_JINA_MODEL_APP_KEY', 'test-aihub-key')
    create_completion = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=SUMMARY_TEXT))])
    )

    class FakeOpenAIClient:
        def __init__(self, config):
            self.config = config
            self.create_completion = create_completion

    monkeypatch.setattr(visit_module, 'OpenAIClient', FakeOpenAIClient)
    return create_completion


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, result) -> AsyncMock:
    mock = AsyncMock(side_effect=result) if isinstance(result, BaseException) else AsyncMock(return_value=result)
    monkeypatch.setattr(GoogleApiBackend, 'fetch', mock)
    return mock


async def _call(arguments: dict) -> dict:
    response = await mcp_server._dispatch({
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': 'visit', 'arguments': arguments},
    })
    return response['result']


@pytest.mark.asyncio
async def test_tools_list_exposes_visit_with_url_and_goal() -> None:
    response = await mcp_server._dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'})

    tools = {tool['name']: tool for tool in response['result']['tools']}
    assert set(tools) == {'search', 'open', 'find', 'visit'}
    assert tools['visit']['inputSchema']['required'] == ['url', 'goal']


@pytest.mark.asyncio
async def test_visit_summarises_page_and_reuses_the_summariser(
    monkeypatch: pytest.MonkeyPatch, server: SimpleBrowserStateActor, summary_client: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'A long page about cubes.'))

    result = await _call({'url': 'https://example.com/a', 'goal': 'find the god number'})
    first_summariser = mcp_server._summariser
    await _call({'url': 'https://example.com/a', 'goal': 'second visit'})

    assert 'isError' not in result
    text = result['content'][0]['text']
    assert text.startswith('The useful information in https://example.com/a for user goal find the god number')
    assert 'God number is 20' in text
    assert server.page_stack[0] == 'https://example.com/a'
    assert mcp_server._summariser is first_summariser
    assert summary_client.await_count == 2


@pytest.mark.asyncio
async def test_visit_reports_error_when_the_page_fails(
    monkeypatch: pytest.MonkeyPatch, summary_client: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, BackendError('HTTP 422'))

    result = await _call({'url': 'https://example.com/a', 'goal': 'anything'})

    assert result['isError'] is True
    assert 'Failed to fetch URL https://example.com/a' in result['content'][0]['text']
    assert summary_client.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('arguments', 'message'),
    [
        ({'url': 'https://example.com/a', 'goal': '  '}, 'Error: Empty goal provided.'),
        ({'url': [], 'goal': 'anything'}, 'Error: Empty url provided.'),
        ({'url': ['https://example.com/a', 123], 'goal': 'anything'}, 'Error: Invalid visit arguments:'),
    ],
)
async def test_visit_rejects_invalid_arguments(monkeypatch: pytest.MonkeyPatch, arguments: dict, message: str) -> None:
    fetch = _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))

    result = await _call(arguments)

    assert result['isError'] is True
    assert result['content'][0]['text'].startswith(message)
    assert fetch.await_count == 0


@pytest.mark.asyncio
async def test_visit_reports_missing_aihub_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SEARCH_STATEFUL_SUMMARY_CLIENT', 'openai')
    monkeypatch.delenv('DEFAULT_JINA_MODEL_APP_KEY', raising=False)
    fetch = _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))

    result = await _call({'url': 'https://example.com/a', 'goal': 'anything'})

    assert result['isError'] is True
    assert 'Missing AIHub 2.0 summary API key' in result['content'][0]['text']
    assert fetch.await_count == 0
