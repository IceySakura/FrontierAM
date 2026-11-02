"""Unit tests for the search_stateful visit tool."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.search_stateful import backend as backend_module
from nexus.extensions.tools.search_stateful import config as config_module
from nexus.extensions.tools.search_stateful import visit as visit_module
from nexus.extensions.tools.search_stateful.backend import BackendError
from nexus.extensions.tools.search_stateful.page_contents import PageContents
from nexus.extensions.tools.search_stateful.state import SIMPLE_BROWSER_STATE_LIST


QUESTION = 'What is the minimum number of moves required to solve a 3x3x3 Rubik cube in the worst case?'
LEAKING_PAGE = f'Solutions archive. {QUESTION} The answer is 20.'
SUMMARY_TEXT = 'EVIDENCE: the cube needs 20 moves.\n\nSUMMARY: God number is 20.'

# jina_model* are web_search's names for the summariser, kept so one host config serves both
# tool families. The values say what they actually are.
CREDENTIALS = {
    'user': 'test-user',
    'apikey': 'test-apikey',
    'use_cache': False,
    'jina_model': 'api_naci_default_deepseek-v3.2',
    'jina_model_app_id': 'test-summary-app-id',
    'jina_model_app_key': 'test-summary-app-key',
}


def _page(url: str, text: str) -> PageContents:
    return PageContents(url=url, text=text, title='A page', urls={})


def _deepseek_response(content: str = SUMMARY_TEXT) -> dict:
    return {'request_detail': {'response': {'output': {'choices': [{'message': {'content': content}}]}}}}


def _doubao_response(content: str = SUMMARY_TEXT) -> dict:
    return {'request_detail': {'response': {'choices': [{'message': {'content': content}}]}}}


def _claude_response(content: str = SUMMARY_TEXT) -> dict:
    return {'request_detail': {'response': {'content': [{'text': content}]}}}


@pytest.fixture
def summarizer(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Stub the gpt-eval summariser endpoint."""
    mock = AsyncMock(return_value=SimpleNamespace(json=lambda: _deepseek_response()))
    monkeypatch.setattr(visit_module.http_client, 'post', mock)
    return mock


@pytest.fixture
def run_id(request: pytest.FixtureRequest) -> str:
    """Give each test its own browser state and drop it afterwards."""
    identifier = f'test-{request.node.name}'
    yield identifier
    SIMPLE_BROWSER_STATE_LIST.browser_states_dict.pop(identifier, None)


@pytest.fixture(autouse=True)
def isolate_summary_client_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('SEARCH_STATEFUL_SUMMARY_CLIENT', raising=False)
    monkeypatch.delenv('SEARCH_STATEFUL_SUMMARY_BASE_URL', raising=False)


def _context(run_id: str, *, origin_question: str = QUESTION, **overrides) -> ToolContext:
    return ToolContext(
        run_id=run_id,
        tool_configs={'search_stateful': {**CREDENTIALS, **overrides}},
        tool_state={'origin_question': origin_question},
    )


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, result) -> None:
    raises_or_dispatches = isinstance(result, BaseException) or callable(result)
    mock = AsyncMock(side_effect=result) if raises_or_dispatches else AsyncMock(return_value=result)
    monkeypatch.setattr(visit_module.GoogleApiBackend, 'fetch', mock)


@pytest.mark.asyncio
async def test_visit_summarizes_a_page(monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'A long page about cubes.'))

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find the god number'), _context(run_id)
    )

    assert response.meta['success'] is True
    assert 'God number is 20' in response.output[0]['text']
    assert summarizer.await_count == 1


@pytest.mark.asyncio
async def test_openai_summary_client_takes_precedence_over_legacy_hmac(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))
    create_completion = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=SUMMARY_TEXT))],
        )
    )
    summary_client = SimpleNamespace(create_completion=create_completion)
    execution_context = SimpleNamespace(
        get_openai_client=lambda name: summary_client if name == 'summary' else None,
    )
    monkeypatch.setattr(visit_module, 'require_context', lambda: execution_context)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    assert response.meta['success'] is True
    assert 'God number is 20' in response.output[0]['text']
    assert create_completion.await_count == 1
    assert isinstance(create_completion.await_args.kwargs['messages'][0]['content'], str)
    assert summarizer.await_count == 0


@pytest.mark.asyncio
async def test_search_stateful_lazily_builds_existing_openai_client_for_aihub2(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))
    monkeypatch.setenv('SEARCH_STATEFUL_SUMMARY_CLIENT', 'unsupported')
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, config):
            captured['config'] = config

        async def create_completion(self, **kwargs):
            captured['request'] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=SUMMARY_TEXT))])

    execution_context = SimpleNamespace(openai_clients={})

    def get_openai_client(name):
        if name not in execution_context.openai_clients:
            raise KeyError(name)
        return execution_context.openai_clients[name]

    execution_context.get_openai_client = get_openai_client
    monkeypatch.setattr(visit_module, 'OpenAIClient', FakeOpenAIClient)
    monkeypatch.setattr(visit_module, 'require_context', lambda: execution_context)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'),
        _context(
            run_id,
            summary_client='openai',
            jina_model='deepseek-v4-flash',
            jina_model_app_key='aihub2-key',
        ),
    )

    assert response.meta['success'] is True
    assert summarizer.await_count == 0
    assert execution_context.openai_clients['summary']._client_key == 'summary'
    assert captured['config'].client_args == {
        'api_key': 'aihub2-key',
        'base_url': 'http://api.aihub.woa.com/standard/v1',
        'max_retries': 3,
        'timeout': 1200,
    }
    assert captured['config'].request_args == {
        'model': 'deepseek-v4-flash',
        'extra_body': {'thinking': {'type': 'disabled'}},
    }
    assert isinstance(captured['request']['messages'][0]['content'], str)


def test_search_stateful_summary_client_can_be_selected_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SEARCH_STATEFUL_SUMMARY_CLIENT', 'openai')
    monkeypatch.setenv('SEARCH_STATEFUL_SUMMARY_BASE_URL', 'http://aihub2.example/v1')
    monkeypatch.setenv('DEFAULT_JINA_MODEL', 'deepseek-v4-flash')
    monkeypatch.setenv('DEFAULT_JINA_MODEL_APP_KEY', 'aihub2-key')
    captured = {}

    class FakeOpenAIClient:
        def __init__(self, config):
            captured['config'] = config

    def no_execution_context():
        raise RuntimeError

    monkeypatch.setattr(visit_module, 'OpenAIClient', FakeOpenAIClient)
    monkeypatch.setattr(visit_module, 'require_context', no_execution_context)

    client = visit_module._get_openai_summary_client({})

    assert client is not None
    assert client._client_key == 'summary'
    assert captured['config'].client_args['api_key'] == 'aihub2-key'
    assert captured['config'].client_args['base_url'] == 'http://aihub2.example/v1'
    assert captured['config'].request_args['model'] == 'deepseek-v4-flash'


@pytest.mark.asyncio
async def test_openai_summary_failure_does_not_fall_back_to_legacy_hmac(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))
    create_completion = AsyncMock(side_effect=RuntimeError('AIHub 2.0 is unavailable'))
    summary_client = SimpleNamespace(create_completion=create_completion)
    execution_context = SimpleNamespace(
        get_openai_client=lambda name: summary_client if name == 'summary' else None,
    )
    monkeypatch.setattr(visit_module, 'require_context', lambda: execution_context)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    assert response.meta['success'] is False
    assert visit_module.SUMMARY_FAILURE_TEXT in response.output[0]['text']
    assert create_completion.await_count == visit_module.SUMMARY_MAX_ATTEMPTS
    assert summarizer.await_count == 0


@pytest.mark.asyncio
async def test_summary_request_matches_the_legacy_contract(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    """The gpt-eval call must stay byte-compatible with the retired visit_cache tool."""
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))

    await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    call = summarizer.await_args
    assert call.args[0] == 'http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval'

    headers = call.kwargs['headers']
    assert headers['Source'] == 'nexus_tools_search_visit'
    assert headers['Apiversion'] == 'v2.03'
    assert headers['Authorization'].startswith('hmac id="test-summary-app-id", algorithm="hmac-sha1"')

    payload = call.kwargs['json']
    assert payload['model_marker'] == 'api_naci_default_deepseek-v3.2'
    assert payload['params']['model'] == 'api_naci_default_deepseek-v3.2'
    assert payload['params']['enable_thinking'] is False
    assert payload['messages'][0]['content'][0]['type'] == 'text'
    assert 'Body text.' in payload['messages'][0]['content'][0]['value']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'builder', [_deepseek_response, _doubao_response, _claude_response], ids=['deepseek', 'doubao', 'claude']
)
async def test_visit_parses_every_provider_response_shape(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock, builder
) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))
    summarizer.return_value = SimpleNamespace(json=lambda: builder())

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    assert response.meta['success'] is True
    assert 'God number is 20' in response.output[0]['text']


@pytest.mark.asyncio
async def test_summary_credentials_come_from_the_config(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    """The summariser is its own service, so it can be pointed at its own app."""
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Body text.'))

    context = ToolContext(
        run_id=run_id,
        tool_configs={
            'search_stateful': {
                'user': 'u',
                'apikey': 'k',
                'use_cache': False,
                'jina_model': 'api_naci_default_deepseek-v3.2',
                'jina_model_app_id': 'summary-id',
                'jina_model_app_key': 'summary-key',
            }
        },
        tool_state={'origin_question': QUESTION},
    )
    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), context
    )

    assert response.meta['success'] is True
    call = summarizer.await_args
    assert call.kwargs['json']['model_marker'] == 'api_naci_default_deepseek-v3.2'
    assert call.kwargs['headers']['Authorization'].startswith('hmac id="summary-id"')


def test_summary_falls_back_to_the_shared_credentials() -> None:
    """Most deployments run one app for everything and should not have to say so three times."""
    config = visit_module._build_summary_config({'user': 'shared-id', 'apikey': 'shared-key'})

    assert (config.app_id, config.app_key) == ('shared-id', 'shared-key')


def test_legacy_summary_model_keeps_its_backward_compatible_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DEFAULT_JINA_MODEL', raising=False)

    config = visit_module._build_summary_config({'user': 'shared-id', 'apikey': 'shared-key'})

    assert config.model == 'api_doubao_DeepSeek-V3.1-250821'


def test_summary_reads_its_own_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same env vars web_search's visit reads, so one host config serves both."""
    monkeypatch.setenv('DEFAULT_JINA_MODEL', 'env-model')
    monkeypatch.setenv('DEFAULT_JINA_MODEL_APP_ID', 'env-id')
    monkeypatch.setenv('DEFAULT_JINA_MODEL_APP_KEY', 'env-key')

    config = visit_module._build_summary_config({'user': 'shared-id', 'apikey': 'shared-key'})

    assert (config.model, config.app_id, config.app_key) == ('env-model', 'env-id', 'env-key')


@pytest.mark.asyncio
async def test_configured_credentials_reach_the_fetch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert on the wire rather than on backend attributes, which credential work renames."""
    post = AsyncMock(
        return_value=SimpleNamespace(status_code=200, json=lambda: {'data': {'text': 'Body.', 'title': 'A page'}})
    )
    monkeypatch.setattr(backend_module.http_client, 'post', post)

    backend = visit_module._build_backend({'user': 'shared-id', 'apikey': 'shared-key', 'use_cache': False})
    await backend.fetch('https://example.com/a')

    assert post.await_args.kwargs['headers']['Authorization'].startswith('Bearer shared-id:shared-key?')


@pytest.mark.asyncio
async def test_leaking_page_is_blocked_without_reaching_the_summarizer(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    """A leaking page must not be summarised, or the summary would launder the answer."""
    _patch_fetch(monkeypatch, _page('https://example.com/leak', LEAKING_PAGE))

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/leak', goal='find the answer'), _context(run_id)
    )

    assert 'This page has been blocked!' in response.output[0]['text']
    assert summarizer.await_count == 0


@pytest.mark.asyncio
async def test_blocked_url_is_refused_before_fetching(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock, tmp_path
) -> None:
    """The backend only block-lists search hits, so visit must screen direct URLs itself."""
    block_file = tmp_path / 'url_block_list.txt'
    block_file.write_text('huggingface.co/datasets\n', encoding='utf-8')
    fetch = AsyncMock(return_value=_page('https://huggingface.co/datasets/hle', 'leaked answers'))
    monkeypatch.setattr(visit_module.GoogleApiBackend, 'fetch', fetch)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://huggingface.co/datasets/hle', goal='find the answer'),
        _context(run_id, url_block_list_file=str(block_file)),
    )

    assert 'This page has been blocked!' in response.output[0]['text']
    assert fetch.await_count == 0
    assert summarizer.await_count == 0


@pytest.mark.asyncio
async def test_unlisted_url_passes_the_block_list(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock, tmp_path
) -> None:
    block_file = tmp_path / 'url_block_list.txt'
    block_file.write_text('huggingface.co/datasets\n', encoding='utf-8')
    _patch_fetch(monkeypatch, _page('https://arxiv.org/abs/1234', 'A paper.'))

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://arxiv.org/abs/1234', goal='find it'),
        _context(run_id, url_block_list_file=str(block_file)),
    )

    assert response.meta['success'] is True
    assert summarizer.await_count == 1


@pytest.mark.asyncio
async def test_visit_without_origin_question_still_summarizes(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    """Leak detection is inert without a seeded question; the tool must still work."""
    _patch_fetch(monkeypatch, _page('https://example.com/leak', LEAKING_PAGE))

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/leak', goal='find the answer'),
        _context(run_id, origin_question=''),
    )

    assert response.meta['success'] is True
    assert summarizer.await_count == 1


@pytest.mark.asyncio
async def test_visit_handles_multiple_urls(monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock) -> None:
    pages = {
        'https://example.com/a': _page('https://example.com/a', 'Page A'),
        'https://example.com/b': _page('https://example.com/b', 'Page B'),
    }
    _patch_fetch(monkeypatch, lambda url: pages[url])

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=list(pages), goal='compare them'), _context(run_id)
    )

    assert response.output[0]['text'].count('=======') == 1
    assert summarizer.await_count == 2


@pytest.mark.asyncio
async def test_visited_pages_are_registered_for_find(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    """visit and open/find share one browser state, so an agent can mix the two styles."""
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'A long page about cubes.'))

    await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    browser_state = await SIMPLE_BROWSER_STATE_LIST.get_browser_state(run_id)
    assert browser_state is not None
    assert browser_state.get_page().url == 'https://example.com/a'


@pytest.mark.asyncio
async def test_fetch_failure_is_reported_per_url(
    monkeypatch: pytest.MonkeyPatch, run_id: str, summarizer: AsyncMock
) -> None:
    _patch_fetch(monkeypatch, BackendError('upstream is down'))

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    assert response.meta['success'] is False
    assert 'upstream is down' in response.output[0]['text']


@pytest.mark.asyncio
async def test_summary_failure_is_reported(monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'Raw body of the page.'))
    mock = AsyncMock(side_effect=RuntimeError('summariser exploded'))
    monkeypatch.setattr(visit_module.http_client, 'post', mock)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), _context(run_id)
    )

    assert response.meta['success'] is False
    assert visit_module.SUMMARY_FAILURE_TEXT in response.output[0]['text']
    assert mock.await_count == visit_module.SUMMARY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_missing_summary_credentials_are_reported(monkeypatch: pytest.MonkeyPatch, run_id: str) -> None:
    """Reachable only when nothing is configured at all, since summary falls back to user/apikey."""
    monkeypatch.delenv('DEFAULT_JINA_MODEL_APP_ID', raising=False)
    monkeypatch.delenv('DEFAULT_JINA_MODEL_APP_KEY', raising=False)
    monkeypatch.setattr(config_module, 'DEFAULT_USER', '')
    monkeypatch.setattr(config_module, 'DEFAULT_APIKEY', '')
    _patch_fetch(monkeypatch, _page('https://example.com/a', 'body'))

    context = ToolContext(
        run_id=run_id,
        tool_configs={'search_stateful': {'user': '', 'apikey': '', 'use_cache': False}},
        tool_state={'origin_question': QUESTION},
    )
    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='find it'), context
    )

    assert response.meta['success'] is False
    assert 'Missing summariser credentials' in response.output[0]['text']


@pytest.mark.asyncio
async def test_empty_goal_is_rejected(run_id: str) -> None:
    response = await visit_module.visit_tool(
        visit_module.VisitInput(url='https://example.com/a', goal='   '), _context(run_id)
    )

    assert response.meta['success'] is False
