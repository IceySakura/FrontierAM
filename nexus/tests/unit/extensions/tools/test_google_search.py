from types import SimpleNamespace

import aiohttp
import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.web_search import google_search as google_search_module


QUERY = 'nexus'
API_URL = 'https://search.example'
API_KEY = 'search-key'
ENV_API_URL = 'https://env-search.example'
ENV_API_KEY = 'env-key'
RESULT_A_LINK = 'https://a.example'
RESULT_B_LINK = 'https://b.example'
TIMEOUT_TOTAL_KEY = 'timeout_total'
URL_KEY = 'url'
HEADERS_KEY = 'headers'
PARAMS_KEY = 'params'


class _FakeRequestContext:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeClientSession:
    def __init__(self, *, timeout, response=None, error: Exception | None = None, call_record: dict[str, object] | None = None):
        self._response = response
        self._error = error
        self._call_record = call_record if call_record is not None else {}
        self._call_record[TIMEOUT_TOTAL_KEY] = timeout.total

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def get(self, url, headers, params):
        self._call_record[URL_KEY] = url
        self._call_record[HEADERS_KEY] = headers
        self._call_record[PARAMS_KEY] = params
        return _FakeRequestContext(response=self._response, error=self._error)


@pytest.mark.asyncio
async def test_google_search_tool_falls_back_to_titles_when_snippets_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_record: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def json(self):
            return {
                'items': [
                    {'title': 'Result A', 'link': RESULT_A_LINK, 'snippet': None},
                    {'title': 'Result B', 'link': RESULT_B_LINK},
                ]
            }

    monkeypatch.setattr(
        google_search_module.aiohttp,
        'ClientSession',
        lambda *, timeout: _FakeClientSession(timeout=timeout, response=_FakeResponse(), call_record=call_record),
    )

    response = await google_search_module.google_search_tool(
        google_search_module.GoogleSearchInput(query=QUERY, num_results=2),
        ToolContext(
            tool_configs={
                'google_search': {
                    'api_url': f'{API_URL}/',
                    'api_key': API_KEY,
                    'timeout_seconds': 15,
                }
            }
        ),
    )

    assert response.output[0]['text'] == f'Result A ({RESULT_A_LINK}) | Result B ({RESULT_B_LINK})'
    assert response.meta['query'] == QUERY
    assert response.meta['num_results'] == 2
    assert response.meta['links'] == [RESULT_A_LINK, RESULT_B_LINK]
    assert call_record == {
        TIMEOUT_TOTAL_KEY: 15,
        URL_KEY: f'{API_URL}/customsearch/v1',
        HEADERS_KEY: {'Authorization': f'Bearer {API_KEY}'},
        PARAMS_KEY: {'q': QUERY, 'num': 2},
    }


@pytest.mark.asyncio
async def test_google_search_tool_returns_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        google_search_module.aiohttp,
        'ClientSession',
        lambda *, timeout: _FakeClientSession(timeout=timeout, error=aiohttp.ServerTimeoutError()),
    )

    response = await google_search_module.google_search_tool(
        google_search_module.GoogleSearchInput(query=QUERY),
        ToolContext(tool_configs={'google_search': {'api_url': API_URL, 'api_key': API_KEY}}),
    )

    assert response.meta == {'error': 'timeout', 'query': QUERY}
    assert response.output[0]['type'] == 'text'
    assert response.output[0]['text']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'status_code',
    [
        401,
        429,
    ],
)
async def test_google_search_tool_maps_http_errors(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    response_error = aiohttp.ClientResponseError(
        request_info=SimpleNamespace(real_url=API_URL),
        history=(),
        status=status_code,
        message='http error',
        headers=None,
    )
    monkeypatch.setattr(
        google_search_module.aiohttp,
        'ClientSession',
        lambda *, timeout: _FakeClientSession(timeout=timeout, error=response_error),
    )

    response = await google_search_module.google_search_tool(
        google_search_module.GoogleSearchInput(query=QUERY),
        ToolContext(tool_configs={'google_search': {'api_url': API_URL, 'api_key': API_KEY}}),
    )

    assert response.meta == {'error': 'http_error', 'status_code': status_code, 'query': QUERY}
    assert response.output[0]['type'] == 'text'
    assert response.output[0]['text']


@pytest.mark.asyncio
async def test_google_search_tool_uses_env_fallback_and_reports_no_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_record: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def json(self):
            return {'items': []}

    monkeypatch.setenv('GOOGLE_API_URL', ENV_API_URL)
    monkeypatch.setenv('GOOGLE_API_KEY', ENV_API_KEY)
    monkeypatch.setattr(
        google_search_module.aiohttp,
        'ClientSession',
        lambda *, timeout: _FakeClientSession(timeout=timeout, response=_FakeResponse(), call_record=call_record),
    )

    response = await google_search_module.google_search_tool(
        google_search_module.GoogleSearchInput(query=QUERY, num_results=4),
        ToolContext(tool_configs={'google_search': {}}),
    )

    assert response.output[0]['type'] == 'text'
    assert response.output[0]['text']
    assert response.meta == {'query': QUERY, 'num_results': 0}
    assert call_record[URL_KEY] == f'{ENV_API_URL}/customsearch/v1'
    assert call_record[HEADERS_KEY] == {'Authorization': f'Bearer {ENV_API_KEY}'}
    assert call_record[PARAMS_KEY] == {'q': QUERY, 'num': 4}


@pytest.mark.asyncio
async def test_google_search_tool_returns_no_content_when_results_lack_snippets_and_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        async def json(self):
            return {'items': [{'title': 'Title without link or snippet'}]}

    monkeypatch.setattr(
        google_search_module.aiohttp,
        'ClientSession',
        lambda *, timeout: _FakeClientSession(timeout=timeout, response=_FakeResponse()),
    )

    response = await google_search_module.google_search_tool(
        google_search_module.GoogleSearchInput(query=QUERY),
        ToolContext(tool_configs={'google_search': {'api_url': API_URL, 'api_key': API_KEY}}),
    )

    assert response.output[0]['type'] == 'text'
    assert response.output[0]['text']
    assert response.meta['query'] == QUERY
    assert response.meta['num_results'] == 1
    assert response.meta['links'] == []
