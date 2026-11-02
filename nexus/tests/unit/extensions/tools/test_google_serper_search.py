from unittest.mock import AsyncMock

import pytest

from nexus.core import ToolContext
from nexus.extensions.agents.base.errors import NonRetryableProviderError
from nexus.extensions.tools.web_search import google_serper_search as google_serper_search_module


QUERY = 'nexus'
BLANK_QUERY = '   '
SERPER_URL = 'https://serper.example'
TOOL_CONFIG_URL = 'https://tool-config.example'
APP_ID = 'app-id'
APP_KEY = 'app-key'
TOOL_APP_ID = 'tool-id'
TOOL_APP_KEY = 'tool-key'
TEMPORARY_FAILURE = 'temporary failure'
BACKEND_UNAVAILABLE = 'serper backend unavailable'
ARGS_KEY = 'args'


@pytest.mark.asyncio
async def test_google_serper_search_tool_uses_tool_config_and_wraps_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_record: dict[str, object] = {}
    monkeypatch.setenv('DEFAULT_SERPER_BLOCK_HUGGINGFACE', 'true')
    monkeypatch.setenv('SEARCH_SERPER_USE_CACHE', 'false')
    monkeypatch.delenv('SEARCH_CACHE_DIR', raising=False)

    async def fake_google_serper_search(
        query,
        num,
        url,
        app_id,
        app_key,
        timeout=120.0,
        retry_delay=10.0,
        max_retries=3,
        block_huggingface=True,
        use_cache=False,
        cache_dir=None,
        url_blacklist=None,
    ):
        call_record[ARGS_KEY] = {
            'query': query,
            'num': num,
            'url': url,
            'app_id': app_id,
            'app_key': app_key,
            'timeout': timeout,
            'retry_delay': retry_delay,
            'max_retries': max_retries,
            'block_huggingface': block_huggingface,
            'use_cache': use_cache,
            'cache_dir': cache_dir,
            'url_blacklist': url_blacklist,
        }
        return f"A web search for '{QUERY}' found 1 results:\n\n## Web Results\n\n1. [Nexus](https://example.com)"

    monkeypatch.setattr(google_serper_search_module, 'google_serper_search', fake_google_serper_search)

    response = await google_serper_search_module.google_serper_search_tool(
        google_serper_search_module.GoogleSerperSearchInput(query=QUERY, num_results=3),
        ToolContext(
            tool_configs={
                'google_serper_search': {
                    'serper_url': TOOL_CONFIG_URL,
                    'serper_app_id': TOOL_APP_ID,
                    'serper_app_key': TOOL_APP_KEY,
                    'block_huggingface': False,
                    'text_search_timeout': '12.5',
                    'url_blacklist': ['https://blocked.example/'],
                }
            }
        ),
    )

    assert response.meta == {'success': True}
    assert response.output[0]['type'] == 'text'
    assert response.output[0]['text']
    assert call_record[ARGS_KEY] == {
        'query': QUERY,
        'num': 3,
        'url': TOOL_CONFIG_URL,
        'app_id': TOOL_APP_ID,
        'app_key': TOOL_APP_KEY,
        'timeout': 12.5,
        'retry_delay': 10.0,
        'max_retries': 3,
        'block_huggingface': False,
        'use_cache': False,
        'cache_dir': None,
        'url_blacklist': ['https://blocked.example/'],
    }


@pytest.mark.asyncio
async def test_google_serper_search_tool_rejects_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    search_mock = AsyncMock()
    monkeypatch.setattr(google_serper_search_module, 'google_serper_search', search_mock)
    response = await google_serper_search_module.google_serper_search_tool(
        google_serper_search_module.GoogleSerperSearchInput(query=BLANK_QUERY),
        ToolContext(tool_configs={}),
    )

    assert response.meta == {'success': False}
    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_google_serper_search_tool_wraps_search_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    search_mock = AsyncMock(side_effect=RuntimeError(BACKEND_UNAVAILABLE))

    monkeypatch.setattr(google_serper_search_module, 'google_serper_search', search_mock)

    response = await google_serper_search_module.google_serper_search_tool(
        google_serper_search_module.GoogleSerperSearchInput(query=QUERY),
        ToolContext(tool_configs={'google_serper_search': {}}),
    )

    assert response.meta == {'success': False}
    search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_google_serper_search_tool_propagates_terminal_error(monkeypatch: pytest.MonkeyPatch) -> None:
    search_mock = AsyncMock(side_effect=NonRetryableProviderError('no_budget_error'))

    monkeypatch.setattr(google_serper_search_module, 'google_serper_search', search_mock)

    with pytest.raises(NonRetryableProviderError, match='no_budget_error'):
        await google_serper_search_module.google_serper_search_tool(
            google_serper_search_module.GoogleSerperSearchInput(query=QUERY),
            ToolContext(tool_configs={'google_serper_search': {}}),
        )

    search_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_google_serper_search_retries_then_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []
    sleep_delays: list[float] = []

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    async def fake_post(url, headers, json, timeout):
        attempts.append(json['q'])
        if len(attempts) == 1:
            raise RuntimeError(TEMPORARY_FAILURE)
        return _FakeResponse(
            {
                'organic': [
                    {
                        'title': 'HF Result',
                        'link': 'https://huggingface.co/demo/model',
                        'snippet': 'Should be filtered.',
                    },
                    {
                        'title': 'Nexus Result',
                        'link': 'https://example.com/nexus',
                        'date': '2026-04-02',
                        'source': 'Example',
                        'snippet': "Useful snippet. Your browser can't play this video.",
                    },
                ]
            }
        )

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr(google_serper_search_module.http_client, 'post', fake_post)
    monkeypatch.setattr(google_serper_search_module.asyncio, 'sleep', fake_sleep)

    result = await google_serper_search_module.google_serper_search(
        query=QUERY,
        num=2,
        url=SERPER_URL,
        app_id=APP_ID,
        app_key=APP_KEY,
        retry_delay=0.25,
        max_retries=2,
    )

    assert attempts == [QUERY, QUERY]
    assert sleep_delays == [0.25]
    assert f"A web search for '{QUERY}' found 1 results" in result
    assert 'Date published: 2026-04-02' in result
    assert 'Source: Example' in result
    assert "Your browser can't play this video." not in result
    assert 'huggingface.co' not in result


@pytest.mark.asyncio
async def test_google_serper_search_raises_without_retry_for_no_budget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []
    sleep_mock = AsyncMock()

    class _FakeResponse:
        status_code = 402
        text = '{"error":{"message":"预算不足","type":"no_budget_error","code":"1029"},"Status":402}'

        def raise_for_status(self) -> None:
            raise AssertionError('terminal platform errors should stop before status retry handling')

    async def fake_post(url, headers, json, timeout):
        attempts.append(json['q'])
        return _FakeResponse()

    monkeypatch.setattr(google_serper_search_module.http_client, 'post', fake_post)
    monkeypatch.setattr(google_serper_search_module.asyncio, 'sleep', sleep_mock)

    with pytest.raises(NonRetryableProviderError, match='no_budget_error'):
        await google_serper_search_module.google_serper_search(
            query=QUERY,
            num=2,
            url=SERPER_URL,
            app_id=APP_ID,
            app_key=APP_KEY,
            max_retries=3,
        )

    assert attempts == [QUERY]
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_google_serper_search_keeps_huggingface_results_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    async def fake_post(url, headers, json, timeout):
        assert json['q'] == QUERY
        return _FakeResponse(
            {
                'organic': [
                    {
                        'title': 'HF Result',
                        'link': 'https://huggingface.co/demo/model',
                        'snippet': 'Should remain when filtering is disabled.',
                    }
                ]
            }
        )

    monkeypatch.setattr(google_serper_search_module.http_client, 'post', fake_post)

    result = await google_serper_search_module.google_serper_search(
        query=QUERY,
        num=2,
        url=SERPER_URL,
        app_id=APP_ID,
        app_key=APP_KEY,
        block_huggingface=False,
    )

    assert 'https://huggingface.co/demo/model' in result


@pytest.mark.asyncio
async def test_google_serper_search_filters_url_blacklist(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                'organic': [
                    {
                        'title': 'Blocked',
                        'link': 'https://blocked.example/a',
                        'snippet': 'Should be filtered.',
                    },
                    {
                        'title': 'Allowed',
                        'link': 'https://allowed.example/a',
                        'snippet': 'Should remain.',
                    },
                ]
            }

    async def fake_post(url, headers, json, timeout):
        return _FakeResponse()

    monkeypatch.setattr(google_serper_search_module.http_client, 'post', fake_post)

    result = await google_serper_search_module.google_serper_search(
        query=QUERY,
        num=2,
        url=SERPER_URL,
        app_id=APP_ID,
        app_key=APP_KEY,
        url_blacklist=['https://blocked.example/'],
    )

    assert 'https://blocked.example/a' not in result
    assert 'https://allowed.example/a' in result


@pytest.mark.asyncio
async def test_google_serper_search_filters_domain_blacklist(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                'organic': [
                    {'title': 'Blocked', 'link': 'https://www.blocked.example/a'},
                    {'title': 'Blocked subdomain', 'link': 'http://sub.blocked.example/b'},
                    {'title': 'Allowed lookalike', 'link': 'https://blocked.example.evil/a'},
                ]
            }

    async def fake_post(url, headers, json, timeout):
        return _FakeResponse()

    monkeypatch.setattr(google_serper_search_module.http_client, 'post', fake_post)

    result = await google_serper_search_module.google_serper_search(
        query=QUERY,
        num=3,
        url=SERPER_URL,
        app_id=APP_ID,
        app_key=APP_KEY,
        url_blacklist=['blocked.example'],
    )

    assert 'https://www.blocked.example/a' not in result
    assert 'http://sub.blocked.example/b' not in result
    assert 'https://blocked.example.evil/a' in result
