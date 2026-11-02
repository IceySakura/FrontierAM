"""Unit tests for rendering search results as a flat markdown list."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.search_stateful import backend as backend_module
from nexus.extensions.tools.search_stateful import search as search_module
from nexus.extensions.tools.search_stateful.backend import GoogleApiBackend
from nexus.extensions.tools.search_stateful.state import SIMPLE_BROWSER_STATE_LIST, render_links


QUERY = 'cortisol epimer'

ORGANIC_RESULTS = [
    {
        'title': 'Cortisol',
        'link': 'https://en.wikipedia.org/wiki/Cortisol',
        'snippet': 'Cortisol is a steroid hormone in the glucocorticoid class.',
    },
    {
        'title': 'Showing Compound Cortisol (FDB021888)',
        'link': 'https://foodb.ca/compounds/FDB021888',
        'snippet': 'Belongs to the class of organic compounds known as 21-hydroxysteroids.',
        'date': 'Sep 21, 2011',
        'source': 'FooDB',
    },
]

CREDENTIALS = {'user': 'test-user', 'apikey': 'test-apikey', 'use_cache': False}


@pytest.fixture
def run_id(request: pytest.FixtureRequest) -> str:
    """Give each test its own browser state and drop it afterwards."""
    identifier = f'test-{request.node.name}'
    yield identifier
    SIMPLE_BROWSER_STATE_LIST.browser_states_dict.pop(identifier, None)


def _serper(monkeypatch: pytest.MonkeyPatch, organic: list[dict]) -> None:
    """Answer the Serper call with the given organic results."""
    response = SimpleNamespace(status_code=200, text='', json=lambda: {'organic': organic})
    monkeypatch.setattr(backend_module.http_client, 'post', AsyncMock(return_value=response))


async def _search(monkeypatch: pytest.MonkeyPatch, organic: list[dict], topn: int = 10, **backend_kwargs):
    _serper(monkeypatch, organic)
    backend = GoogleApiBackend(**CREDENTIALS, **backend_kwargs)
    return await backend.search(query=QUERY, topn=topn)


@pytest.mark.asyncio
async def test_the_search_page_carries_its_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    page = await _search(monkeypatch, ORGANIC_RESULTS)

    assert [(hit.title, hit.url) for hit in page.search_results] == [
        ('Cortisol', 'https://en.wikipedia.org/wiki/Cortisol'),
        ('Showing Compound Cortisol (FDB021888)', 'https://foodb.ca/compounds/FDB021888'),
    ]
    assert page.search_results[1].date == 'Sep 21, 2011'
    assert page.search_results[1].source == 'FooDB'


@pytest.mark.asyncio
async def test_the_serper_call_asks_the_gateway_for_a_60s_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway takes its per-request timeout from the Authorization query string, not a header."""
    _serper(monkeypatch, ORGANIC_RESULTS)
    backend = GoogleApiBackend(**CREDENTIALS)

    await backend.search(query=QUERY, topn=3)

    call = backend_module.http_client.post.await_args
    assert call.kwargs['url'] == 'http://trpc-gpt-eval.production.polaris:8080/search'
    assert call.kwargs['headers']['Authorization'] == 'Bearer test-user:test-apikey?provider=serper&timeout=60'
    assert call.kwargs['json'] == {'q': QUERY, 'num': 3}


@pytest.mark.asyncio
async def test_a_fetched_page_carries_no_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    """search_results belongs to search pages only; open must not see a stray empty list."""
    monkeypatch.setattr(
        backend_module.http_client,
        'post',
        AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                text='',
                json=lambda: {'data': {'text': 'Some article body.', 'title': 'An article'}},
            )
        ),
    )
    backend = GoogleApiBackend(**CREDENTIALS)

    page = await backend.fetch('https://example.com/article')

    assert page.search_results is None


@pytest.mark.asyncio
async def test_rendering_matches_what_web_search_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    page = await _search(monkeypatch, ORGANIC_RESULTS)

    assert render_links(page.search_results, QUERY) == (
        "A web search for 'cortisol epimer' found 2 results:\n"
        '\n'
        '## Web Results\n'
        '1. [Cortisol](https://en.wikipedia.org/wiki/Cortisol)\n'
        '\n'
        'Cortisol is a steroid hormone in the glucocorticoid class.\n'
        '\n'
        '2. [Showing Compound Cortisol (FDB021888)](https://foodb.ca/compounds/FDB021888)\n'
        'Date published: Sep 21, 2011\n'
        'Source: FooDB\n'
        '\n'
        'Belongs to the class of organic compounds known as 21-hydroxysteroids.'
    )


@pytest.mark.asyncio
async def test_a_result_without_a_snippet_still_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    page = await _search(monkeypatch, [{'title': 'Bare', 'link': 'https://example.com/bare'}])

    assert render_links(page.search_results, QUERY).endswith('1. [Bare](https://example.com/bare)\n')


def test_an_empty_result_set_is_reported() -> None:
    assert render_links([], QUERY) == "No results found for 'cortisol epimer'. Try with a more general query."


@pytest.mark.asyncio
async def test_blocked_results_stay_out_of_the_list(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The list inherits the block list because it renders what the page was built from."""
    block_list = tmp_path / 'blocked.txt'
    # Written without a trailing newline on purpose: splitting one off leaves an empty entry that
    # matches every URL and blocks the lot. That is a standing backend bug, fixed by the
    # account/summary MR, and not what this test is here to pin down.
    block_list.write_text('foodb.ca', encoding='utf-8')

    page = await _search(monkeypatch, ORGANIC_RESULTS, url_block_list_file=str(block_list))

    assert [hit.url for hit in page.search_results] == ['https://en.wikipedia.org/wiki/Cortisol']
    assert 'foodb.ca' not in render_links(page.search_results, QUERY)


@pytest.mark.asyncio
async def test_topn_trims_the_list_like_it_trims_the_page(monkeypatch: pytest.MonkeyPatch) -> None:
    page = await _search(monkeypatch, ORGANIC_RESULTS, topn=1)

    assert [hit.url for hit in page.search_results] == ['https://en.wikipedia.org/wiki/Cortisol']
    assert len(page.urls) == 1


@pytest.mark.asyncio
async def test_search_prints_full_urls_when_configured(monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    _serper(monkeypatch, ORGANIC_RESULTS)
    context = ToolContext(
        run_id=run_id,
        tool_configs={'search_stateful': {**CREDENTIALS, 'search_result_format': 'links'}},
        tool_state={},
    )

    response = await search_module.search_tool(search_module.SearchInput(query=QUERY), context)

    text = response.output[0]['text']
    assert text.startswith("A web search for 'cortisol epimer' found 2 results:")
    assert '1. [Cortisol](https://en.wikipedia.org/wiki/Cortisol)' in text
    assert 'viewing lines' not in text


@pytest.mark.asyncio
async def test_search_keeps_the_paged_view_by_default(monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    _serper(monkeypatch, ORGANIC_RESULTS)
    context = ToolContext(run_id=run_id, tool_configs={'search_stateful': CREDENTIALS}, tool_state={})

    response = await search_module.search_tool(search_module.SearchInput(query=QUERY), context)

    text = response.output[0]['text']
    assert '【0†Cortisol†en.wikipedia.org】' in text
    assert 'viewing lines' in text
    assert 'https://en.wikipedia.org/wiki/Cortisol' not in text


@pytest.mark.asyncio
async def test_the_links_view_still_registers_the_page_for_open(monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    """open works off the browser state, which the flat rendering must not bypass."""
    _serper(monkeypatch, ORGANIC_RESULTS)
    context = ToolContext(
        run_id=run_id,
        tool_configs={'search_stateful': {**CREDENTIALS, 'search_result_format': 'links'}},
        tool_state={},
    )

    await search_module.search_tool(search_module.SearchInput(query=QUERY), context)

    browser_state = await SIMPLE_BROWSER_STATE_LIST.get_browser_state(run_id)
    assert browser_state.get_page().urls['0'] == 'https://en.wikipedia.org/wiki/Cortisol'
