from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core import ContextLengthExceededError, ToolContext
from nexus.extensions.agents.base.errors import NonRetryableProviderError
from nexus.extensions.tools.web_search import visit as visit_module


BLANK_GOAL = '   '
DEFAULT_URL = 'https://example.com'
PAGE_URL = 'https://example.com/page'
GOOD_URL = 'https://ok.example'
BAD_URL = 'https://bad.example'
BATCH_URLS = ['https://a.example', 'https://b.example']
GOAL = 'find revenue'
VISIT_APP_ID = 'visit-id'
VISIT_APP_KEY = 'visit-key'
MODEL_APP_ID = 'model-id'
MODEL_APP_KEY = 'model-key'
VISIT_BASE_URL = 'https://visit.example'
CUSTOM_PROVIDER = 'custom-provider'
CUSTOM_MODEL = 'custom-model'
FAILED_VISIT_RESULT = '[visit error] Failed to read page. error=timeout'
FAILED_LLM_VISIT_RESULT = '[visit llm error] LLM extraction failed after 3 attempts.'
HF_DATASET_URL = 'https://huggingface.co/datasets/demo/corpus'
HF_SPACE_URL = 'https://huggingface.co/spaces/demo/app'
SUCCESS_SUMMARY = 'useful summary from batch visit'
MASKED_BEARER_PREFIX = 'Bearer '
MASK_MARKER = '****'
JINA_ENV_VARS = (
    'DEFAULT_JINA_APP_ID',
    'DEFAULT_JINA_APP_KEY',
    'DEFAULT_JINA_MODEL_APP_ID',
    'DEFAULT_JINA_MODEL_APP_KEY',
)
ARGS_KEY = 'args'
VISIT_SUMMARY_ENV_VARS = (
    'VISIT_SUMMARY_CLIENT',
    'VISIT_SUMMARY_BASE_URL',
    'VISIT_SUMMARY_PROVIDER',
    'VISIT_SUMMARY_TIMEOUT',
    'AIHUB_V2_MAX_OUTPUT_TOKENS',
    'VISIT_SUMMARY_REASONING_EFFORT',
)


@pytest.fixture(autouse=True)
def clear_visit_summary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop an ambient AIHub setting from switching the summariser under every test."""
    for env_name in VISIT_SUMMARY_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)


def _patch_without_execution_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a visit running outside any ExecutionContext, as the unit tests do."""

    def no_execution_context():
        raise RuntimeError('no active ExecutionContext')

    monkeypatch.setattr(visit_module, 'require_context', no_execution_context)


@pytest.mark.asyncio
async def test_visit_tool_rejects_empty_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_readpage(*args, **kwargs):
        raise AssertionError('readpage should not be called for an empty goal')

    monkeypatch.setattr(visit_module, 'readpage', fail_readpage)
    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=DEFAULT_URL, goal=BLANK_GOAL),
        ToolContext(tool_configs={}),
    )

    assert response.meta == {'success': False}


@pytest.mark.asyncio
async def test_visit_tool_returns_error_when_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_readpage(*args, **kwargs):
        raise AssertionError('readpage should not be called when credentials are missing')

    monkeypatch.setattr(visit_module, 'readpage', fail_readpage)
    for env_name in JINA_ENV_VARS:
        monkeypatch.delenv(env_name, raising=False)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=DEFAULT_URL, goal=GOAL),
        ToolContext(tool_configs={'visit': {}}),
    )

    assert response.meta == {'success': False}


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_result', [FAILED_VISIT_RESULT, FAILED_LLM_VISIT_RESULT])
async def test_visit_tool_returns_failed_response_for_failed_visit_result(
    monkeypatch: pytest.MonkeyPatch, failed_result: str
) -> None:
    async def fake_readpage(*args, **kwargs):
        return failed_result

    monkeypatch.setattr(visit_module, 'readpage', fake_readpage)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=DEFAULT_URL, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_model_app_id': MODEL_APP_ID,
                    'jina_model_app_key': MODEL_APP_KEY,
                }
            }
        ),
    )

    assert response.meta == {'success': False}
    assert response.output[0]['text'] == failed_result


@pytest.mark.asyncio
@pytest.mark.parametrize('url', [HF_DATASET_URL, [DEFAULT_URL, HF_SPACE_URL]])
async def test_visit_tool_blocks_huggingface_urls_at_entry(monkeypatch: pytest.MonkeyPatch, url: str | list[str]) -> None:
    async def fail_read(*args, **kwargs):
        raise AssertionError('underlying visit functions should not be called for blocked Hugging Face URLs')

    monkeypatch.setattr(visit_module, 'readpage', fail_read)
    monkeypatch.setattr(visit_module, 'visit_pages', fail_read)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=url, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_model_app_id': MODEL_APP_ID,
                    'jina_model_app_key': MODEL_APP_KEY,
                    'block_huggingface': True,
                }
            }
        ),
    )

    assert response.meta == {'success': False}
    assert 'Hugging Face dataset or space URL' in response.output[0]['text']


@pytest.mark.asyncio
async def test_visit_tool_uses_batch_visit_and_strips_output(monkeypatch: pytest.MonkeyPatch) -> None:
    call_record: dict[str, object] = {}

    async def fake_visit_pages(urls, goal, app_id, app_key, provider, summary_models, base_url, **kwargs):
        call_record[ARGS_KEY] = {
            'urls': urls,
            'goal': goal,
            'app_id': app_id,
            'app_key': app_key,
            'provider': provider,
            'summary_models': summary_models,
            'base_url': base_url,
        }
        return f'  {SUCCESS_SUMMARY}  \n'

    monkeypatch.setattr(visit_module, 'visit_pages', fake_visit_pages)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=BATCH_URLS, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_provider': CUSTOM_PROVIDER,
                    'jina_url': VISIT_BASE_URL,
                    'jina_model': CUSTOM_MODEL,
                    'jina_model_app_id': MODEL_APP_ID,
                    'jina_model_app_key': MODEL_APP_KEY,
                }
            }
        ),
    )

    assert response.meta == {'success': True}
    assert response.output == [{'type': 'text', 'text': SUCCESS_SUMMARY}]
    assert call_record[ARGS_KEY] == {
        'urls': BATCH_URLS,
        'goal': GOAL,
        'app_id': VISIT_APP_ID,
        'app_key': VISIT_APP_KEY,
        'provider': CUSTOM_PROVIDER,
        'summary_models': [visit_module.SummaryModelConfig(CUSTOM_MODEL, MODEL_APP_ID, MODEL_APP_KEY)],
        'base_url': VISIT_BASE_URL,
    }


def test_mask_auth_bearer_masks_credentials_but_keeps_provider() -> None:
    original = 'Bearer abcdef123456:uvwxyz987654?provider=aws_jina'
    original_app_id = 'abcdef123456'
    original_app_key = 'uvwxyz987654'
    masked = visit_module._mask_auth_bearer(original)

    assert masked.startswith(MASKED_BEARER_PREFIX)
    assert MASK_MARKER in masked
    assert original_app_id not in masked
    assert original_app_key not in masked
    assert masked.endswith('?provider=aws_jina')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('response_payload', 'expected'),
    [
        ({'request_detail': {'response': {'output': {'choices': [{'message': {'content': 'from-output'}}]}}}}, 'from-output'),
        ({'request_detail': {'response': {'choices': [{'message': {'content': 'from-choices'}}]}}}, 'from-choices'),
        ({'request_detail': {'response': {'content': [{'text': 'from-content'}]}}}, 'from-content'),
    ],
)
async def test_call_llm_server_supports_multiple_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
    response_payload: dict,
    expected: str,
) -> None:
    class _FakeResponse:
        status_code = 200
        text = 'ok'

        def json(self):
            return response_payload

    async def fake_post(url, headers, json, timeout):
        return _FakeResponse()

    monkeypatch.setattr(visit_module.http_client, 'post', fake_post)

    result = await visit_module.call_llm_server(
        messages=[{'role': 'user', 'content': [{'type': 'text', 'value': 'question'}]}],
        app_id=MODEL_APP_ID,
        app_key=MODEL_APP_KEY,
        model='demo-model',
        max_tries=1,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_call_llm_server_raises_without_retry_for_no_budget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sleep_mock = AsyncMock()

    class _FakeResponse:
        status_code = 402
        text = '{"error":{"message":"预算不足","type":"no_budget_error","code":"1029"},"Status":402}'

    async def fake_post(url, headers, json, timeout):
        calls.append(json['request_id'])
        return _FakeResponse()

    monkeypatch.setattr(visit_module.http_client, 'post', fake_post)
    monkeypatch.setattr(visit_module.asyncio, 'sleep', sleep_mock)

    with pytest.raises(NonRetryableProviderError, match='no_budget_error'):
        await visit_module.call_llm_server(
            messages=[{'role': 'user', 'content': [{'type': 'text', 'value': 'question'}]}],
            app_id=MODEL_APP_ID,
            app_key=MODEL_APP_KEY,
            model='demo-model',
            max_tries=3,
        )

    assert len(calls) == 1
    sleep_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_jina_readpage_returns_failure_string_without_retry_for_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FakeResponse:
        status_code = 422
        text = 'unexpected content type'

    async def fake_post(base_url, json, headers, timeout):
        calls.append(json['url'])
        return _FakeResponse()

    monkeypatch.setattr(visit_module.http_client, 'post', fake_post)

    result = await visit_module.jina_readpage(
        url=PAGE_URL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        base_url=VISIT_BASE_URL,
    )

    assert calls == [PAGE_URL]
    assert result.startswith(f'{visit_module.VISIT_ERROR_TAG} Failed to read page (HTTP 422).')
    assert 'Reason: unexpected content type.' in result


@pytest.mark.asyncio
async def test_jina_readpage_raises_without_retry_for_no_budget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeResponse:
        status_code = 402
        text = '{"error":{"message":"预算不足","type":"no_budget_error","code":"1029"},"Status":402}'

    async def fake_post(base_url, json, headers, timeout):
        calls.append(json['url'])
        return _FakeResponse()

    monkeypatch.setattr(visit_module.http_client, 'post', fake_post)

    with pytest.raises(NonRetryableProviderError, match='no_budget_error'):
        await visit_module.jina_readpage(
            url=PAGE_URL,
            app_id=VISIT_APP_ID,
            app_key=VISIT_APP_KEY,
            provider='aws_jina',
            base_url=VISIT_BASE_URL,
        )

    assert calls == [PAGE_URL]


@pytest.mark.asyncio
async def test_jina_readpage_422_hint_includes_readable_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    readable_message = 'reader got unsupported content type application/xml from upstream'

    class _FakeResponse:
        status_code = 422
        text = (
            '{"readableMessage": "reader got unsupported content type application/xml from upstream", '
            '"message": "raw backend error"}'
        )

    async def fake_post(base_url, json, headers, timeout):
        calls.append(json['url'])
        return _FakeResponse()

    monkeypatch.setattr(visit_module.http_client, 'post', fake_post)

    result = await visit_module.jina_readpage(
        url=PAGE_URL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        base_url=VISIT_BASE_URL,
    )

    assert calls == [PAGE_URL]
    assert result.startswith(f'{visit_module.VISIT_ERROR_TAG} Failed to read page (HTTP 422).')
    assert f'Reason: {readable_message}.' in result


@pytest.mark.asyncio
async def test_visit_pages_joins_successes_and_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_readpage(url, goal, app_id, app_key, provider, summary_models, base_url, **kwargs):
        if url == BAD_URL:
            raise RuntimeError('boom')
        return f'content for {url}'

    monkeypatch.setattr(visit_module, 'readpage', fake_readpage)

    summary_models = [visit_module.SummaryModelConfig('demo-model', MODEL_APP_ID, MODEL_APP_KEY)]
    result = await visit_module.visit_pages(
        urls=[GOOD_URL, BAD_URL],
        goal=GOAL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        summary_models=summary_models,
        base_url=VISIT_BASE_URL,
    )

    assert f'content for {GOOD_URL}' in result
    assert f'Error fetch content from {BAD_URL}' in result
    assert 'boom' in result


@pytest.mark.asyncio
async def test_visit_pages_raises_terminal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_readpage(url, goal, app_id, app_key, provider, summary_models, base_url, **kwargs):
        if url == BAD_URL:
            raise NonRetryableProviderError('no_budget_error')
        return f'content for {url}'

    monkeypatch.setattr(visit_module, 'readpage', fake_readpage)

    summary_models = [visit_module.SummaryModelConfig('demo-model', MODEL_APP_ID, MODEL_APP_KEY)]
    with pytest.raises(NonRetryableProviderError, match='no_budget_error'):
        await visit_module.visit_pages(
            urls=[GOOD_URL, BAD_URL],
            goal=GOAL,
            app_id=VISIT_APP_ID,
            app_key=VISIT_APP_KEY,
            provider='aws_jina',
            summary_models=summary_models,
            base_url=VISIT_BASE_URL,
        )


# ============== Dual Summary Model Tests ==============

MODEL_A = visit_module.SummaryModelConfig('model-a', 'a-id', 'a-key')
MODEL_B = visit_module.SummaryModelConfig('model-b', 'b-id', 'b-key')
JINA_PAGE_CONTENT = 'some webpage content from jina'
LLM_SUCCESS_RESULT = 'extracted useful info'


@pytest.mark.asyncio
async def test_readpage_single_model_uses_same_model_for_all_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single model: all 3 retry attempts must use the same (only) model."""
    models_used: list[str] = []
    monkeypatch.setattr(visit_module.asyncio, 'sleep', AsyncMock())

    async def fake_jina_readpage(url, app_id, app_key, provider, base_url, **kwargs):
        return JINA_PAGE_CONTENT

    async def fake_call_llm(messages, app_id, app_key, model, max_tries=3, **kwargs):
        models_used.append(model)
        return visit_module.VISIT_LLM_ERROR_TAG

    monkeypatch.setattr(visit_module, 'jina_readpage', fake_jina_readpage)
    monkeypatch.setattr(visit_module, 'call_llm_server', fake_call_llm)

    result = await visit_module.readpage(
        url=PAGE_URL,
        goal=GOAL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        summary_models=[MODEL_A],
        base_url=VISIT_BASE_URL,
    )

    assert len(models_used) == 3
    assert all(m == 'model-a' for m in models_used)
    assert result.startswith(visit_module.VISIT_LLM_ERROR_TAG)
    assert JINA_PAGE_CONTENT in result


@pytest.mark.asyncio
async def test_readpage_dual_model_failover_switches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dual models: when the first model fails, the second attempt must use a different model."""
    models_used: list[str] = []
    monkeypatch.setattr(visit_module.asyncio, 'sleep', AsyncMock())

    async def fake_jina_readpage(url, app_id, app_key, provider, base_url, **kwargs):
        return JINA_PAGE_CONTENT

    async def fake_call_llm(messages, app_id, app_key, model, max_tries=3, **kwargs):
        models_used.append(model)
        if len(models_used) == 1:
            return visit_module.VISIT_LLM_ERROR_TAG
        return LLM_SUCCESS_RESULT

    monkeypatch.setattr(visit_module, 'jina_readpage', fake_jina_readpage)
    monkeypatch.setattr(visit_module, 'call_llm_server', fake_call_llm)
    monkeypatch.setattr(visit_module.random, 'sample', lambda pop, k: list(pop))

    result = await visit_module.readpage(
        url=PAGE_URL,
        goal=GOAL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        summary_models=[MODEL_A, MODEL_B],
        base_url=VISIT_BASE_URL,
    )

    assert len(models_used) == 2
    assert models_used[0] != models_used[1]
    assert LLM_SUCCESS_RESULT in result


@pytest.mark.asyncio
async def test_readpage_dual_model_all_fail_returns_jina_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dual models: if all attempts fail, should return truncated Jina content."""
    models_used: list[str] = []
    monkeypatch.setattr(visit_module.asyncio, 'sleep', AsyncMock())
    overflow = 10
    fallback_len = visit_module.JINA_FALLBACK_CONTENT_LENGTH
    long_jina_page_content = 'x' * (fallback_len + overflow)

    async def fake_jina_readpage(url, app_id, app_key, provider, base_url, **kwargs):
        return long_jina_page_content

    async def fake_call_llm(messages, app_id, app_key, model, max_tries=3, **kwargs):
        models_used.append(model)
        return visit_module.VISIT_LLM_ERROR_TAG

    monkeypatch.setattr(visit_module, 'jina_readpage', fake_jina_readpage)
    monkeypatch.setattr(visit_module, 'call_llm_server', fake_call_llm)
    monkeypatch.setattr(visit_module.random, 'sample', lambda pop, k: list(pop))

    result = await visit_module.readpage(
        url=PAGE_URL,
        goal=GOAL,
        app_id=VISIT_APP_ID,
        app_key=VISIT_APP_KEY,
        provider='aws_jina',
        summary_models=[MODEL_A, MODEL_B],
        base_url=VISIT_BASE_URL,
    )

    assert len(models_used) == 3
    assert models_used[0] == 'model-a'
    assert models_used[1] == 'model-b'
    assert models_used[2] == 'model-b'
    assert result.startswith(visit_module.VISIT_LLM_ERROR_TAG)
    assert f'{fallback_len}/{len(long_jina_page_content)}' in result
    assert f'[Content truncated. {overflow} more characters not shown.]' in result
    assert long_jina_page_content[:fallback_len] in result
    assert long_jina_page_content not in result


@pytest.mark.asyncio
async def test_visit_tool_builds_dual_summary_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """visit_tool should build a list with two SummaryModelConfig when jina_model_2 is set."""
    captured_models: list[visit_module.SummaryModelConfig] = []

    async def fake_readpage(url, goal, app_id, app_key, provider, summary_models, base_url, **kwargs):
        captured_models.extend(summary_models)
        return 'ok'

    monkeypatch.setattr(visit_module, 'readpage', fake_readpage)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=DEFAULT_URL, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_model': 'primary-model',
                    'jina_model_app_id': MODEL_APP_ID,
                    'jina_model_app_key': MODEL_APP_KEY,
                    'jina_model_2': 'secondary-model',
                    'jina_model_app_id_2': 'second-id',
                    'jina_model_app_key_2': 'second-key',
                }
            }
        ),
    )

    assert response.meta == {'success': True}
    assert len(captured_models) == 2
    assert captured_models[0] == visit_module.SummaryModelConfig('primary-model', MODEL_APP_ID, MODEL_APP_KEY)
    assert captured_models[1] == visit_module.SummaryModelConfig('secondary-model', 'second-id', 'second-key')


@pytest.mark.asyncio
async def test_visit_tool_model_2_reuses_primary_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """When model_2 credentials are missing, they should fall back to primary model credentials."""
    captured_models: list[visit_module.SummaryModelConfig] = []

    async def fake_readpage(url, goal, app_id, app_key, provider, summary_models, base_url, **kwargs):
        captured_models.extend(summary_models)
        return 'ok'

    monkeypatch.setattr(visit_module, 'readpage', fake_readpage)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=DEFAULT_URL, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_model': 'primary-model',
                    'jina_model_app_id': MODEL_APP_ID,
                    'jina_model_app_key': MODEL_APP_KEY,
                    'jina_model_2': 'secondary-model',
                }
            }
        ),
    )

    assert response.meta == {'success': True}
    assert len(captured_models) == 2
    assert captured_models[1] == visit_module.SummaryModelConfig('secondary-model', MODEL_APP_ID, MODEL_APP_KEY)


class _FakeOpenAIClient:
    """Stand-in for the AIHub 2.0 client that records how it was built and called."""

    def __init__(self, config, captured: dict, content: str | None = SUCCESS_SUMMARY):
        self.config = config
        self._captured = captured
        self._content = content
        captured['config'] = config

    async def create_completion(self, **kwargs):
        self._captured['request'] = kwargs
        if self._content is None:
            return SimpleNamespace(choices=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


def _patch_fake_openai_client(monkeypatch: pytest.MonkeyPatch, content: str | None = SUCCESS_SUMMARY) -> dict:
    """Replace OpenAIClient with a recording fake and return the capture dict."""
    captured: dict = {}
    monkeypatch.setattr(
        visit_module, 'OpenAIClient', lambda config: _FakeOpenAIClient(config, captured, content=content)
    )
    return captured


@pytest.mark.asyncio
async def test_visit_tool_summarizes_through_aihub_client_without_model_app_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opting into AIHub 2.0 must bypass data_eval and not require the data_eval model app id."""
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)

    async def fake_jina_readpage(url, app_id, app_key, provider, base_url, **kwargs):
        return JINA_PAGE_CONTENT

    async def fail_data_eval(*args, **kwargs):
        raise AssertionError('data_eval must not be called once the AIHub client is enabled')

    monkeypatch.setattr(visit_module, 'jina_readpage', fake_jina_readpage)
    monkeypatch.setattr(visit_module.http_client, 'post', fail_data_eval)

    response = await visit_module.visit_tool(
        visit_module.VisitInput(url=PAGE_URL, goal=GOAL),
        ToolContext(
            tool_configs={
                'visit': {
                    'jina_app_id': VISIT_APP_ID,
                    'jina_app_key': VISIT_APP_KEY,
                    'jina_model': CUSTOM_MODEL,
                    'jina_model_app_key': MODEL_APP_KEY,
                    'summary_client': 'openai',
                }
            }
        ),
    )

    assert response.meta == {'success': True}
    assert SUCCESS_SUMMARY in response.output[0]['text']
    # The page text and goal must reach the model as one plain string, not data_eval parts.
    sent_content = captured['request']['messages'][0]['content']
    assert isinstance(sent_content, str)
    assert JINA_PAGE_CONTENT in sent_content
    assert GOAL in sent_content
    assert captured['request']['model'] == CUSTOM_MODEL
    assert captured['config'].client_args['api_key'] == MODEL_APP_KEY
    assert captured['config'].client_args['base_url'] == visit_module.DEFAULT_SUMMARY_BASE_URL
    # model must stay out of request_args or it would pin every retry to one model.
    assert 'model' not in captured['config'].request_args
    assert captured['config'].request_args['max_tokens'] == visit_module.AIHUB_V2_MAX_OUTPUT_TOKENS


def test_get_openai_summary_client_returns_none_without_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_without_execution_context(monkeypatch)

    assert visit_module.get_openai_summary_client({}) is None


def test_get_openai_summary_client_rejects_unsupported_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_without_execution_context(monkeypatch)

    with pytest.raises(ValueError, match='Unsupported visit summary_client'):
        visit_module.get_openai_summary_client({'summary_client': 'responses'})


def test_get_openai_summary_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_without_execution_context(monkeypatch)
    monkeypatch.delenv('DEFAULT_JINA_MODEL_APP_KEY', raising=False)

    with pytest.raises(ValueError, match='Missing AIHub 2.0 summary API key'):
        visit_module.get_openai_summary_client({'summary_client': 'openai'})


def test_get_openai_summary_client_appends_provider_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
        'summary_provider': 'ali',
    })

    assert captured['config'].client_args['api_key'] == f'{MODEL_APP_KEY}?provider=ali'


def test_get_openai_summary_client_keeps_provider_already_in_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': f'{MODEL_APP_KEY}?provider=aws_third',
        'summary_provider': 'ali',
    })

    assert captured['config'].client_args['api_key'] == f'{MODEL_APP_KEY}?provider=aws_third'


def test_get_openai_summary_client_builds_and_caches_when_context_lacks_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context without a 'summary' client must build one and cache it, not raise."""
    captured = _patch_fake_openai_client(monkeypatch)
    execution_context = SimpleNamespace(openai_clients={})

    def get_openai_client(name):
        if name not in execution_context.openai_clients:
            raise KeyError(name)
        return execution_context.openai_clients[name]

    execution_context.get_openai_client = get_openai_client
    monkeypatch.setattr(visit_module, 'require_context', lambda: execution_context)

    client = visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert client is not None
    assert captured['config'].client_args['api_key'] == MODEL_APP_KEY
    # The built client is cached on the context so one SDK client serves the trajectory.
    assert execution_context.openai_clients[visit_module.SUMMARY_CLIENT_KEY] is client
    assert client._client_key == visit_module.SUMMARY_CLIENT_KEY


def test_get_openai_summary_client_survives_keyerror_from_require_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KeyError out of require_context itself must not leave execution_context unbound.

    require_context only raises RuntimeError today, so this guards the refactor that would
    otherwise turn the later `if execution_context is not None` into a NameError.
    """
    captured = _patch_fake_openai_client(monkeypatch)

    def raise_key_error():
        raise KeyError('no context registry entry')

    monkeypatch.setattr(visit_module, 'require_context', raise_key_error)

    client = visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert client is not None
    assert captured['config'].client_args['api_key'] == MODEL_APP_KEY


def test_get_openai_summary_client_reuses_client_already_on_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner-provided 'summary' client wins over any local AIHub configuration."""
    existing = object()
    execution_context = SimpleNamespace(openai_clients={visit_module.SUMMARY_CLIENT_KEY: existing})
    execution_context.get_openai_client = lambda name: execution_context.openai_clients[name]

    def fail_build(config):
        raise AssertionError('must not build a client when the context already provides one')

    monkeypatch.setattr(visit_module, 'OpenAIClient', fail_build)
    monkeypatch.setattr(visit_module, 'require_context', lambda: execution_context)

    assert visit_module.get_openai_summary_client({}) is existing


def test_get_openai_summary_client_defaults_to_no_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default must switch reasoning off: deepseek-v4.1-flash would otherwise use 'high'.

    Only standard chat/completions fields may be sent, so a vendor extra_body is not an option.
    """
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert captured['config'].request_args['reasoning_effort'] == 'none'
    assert 'extra_body' not in captured['config'].request_args


def test_get_openai_summary_client_sends_configured_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """hy3 accepts only its own dialect, so the value must be overridable per model."""
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)
    monkeypatch.setenv('VISIT_SUMMARY_REASONING_EFFORT', 'no_think')

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert captured['config'].request_args['reasoning_effort'] == 'no_think'
    assert 'extra_body' not in captured['config'].request_args


def test_get_openai_summary_client_empty_reasoning_effort_omits_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty string is an explicit request to send no reasoning_effort at all."""
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)
    monkeypatch.setenv('VISIT_SUMMARY_REASONING_EFFORT', '')

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert 'reasoning_effort' not in captured['config'].request_args


def test_get_openai_summary_client_defaults_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert captured['config'].request_args['max_tokens'] == 40960


def test_get_openai_summary_client_max_output_tokens_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)
    monkeypatch.setenv('AIHUB_V2_MAX_OUTPUT_TOKENS', '1234')

    visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    assert captured['config'].request_args['max_tokens'] == 1234


def test_get_openai_summary_client_can_be_selected_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch)
    _patch_without_execution_context(monkeypatch)
    monkeypatch.setenv('VISIT_SUMMARY_CLIENT', 'openai')
    monkeypatch.setenv('VISIT_SUMMARY_BASE_URL', 'http://aihub2.example/v1')
    monkeypatch.setenv('DEFAULT_JINA_MODEL_APP_KEY', MODEL_APP_KEY)

    client = visit_module.get_openai_summary_client({})

    assert client is not None
    assert captured['config'].client_args['base_url'] == 'http://aihub2.example/v1'


@pytest.mark.asyncio
async def test_call_llm_server_maps_context_length_error_to_tag() -> None:
    class ExplodingClient:
        async def create_completion(self, **kwargs):
            raise ContextLengthExceededError('prompt is longer than the model context length')

    result = await visit_module.call_llm_server(
        [{'role': 'user', 'content': [{'value': 'hi', 'type': 'text'}]}],
        MODEL_APP_ID,
        MODEL_APP_KEY,
        CUSTOM_MODEL,
        summary_client=ExplodingClient(),
    )

    assert result == visit_module.VISIT_CONTEXT_LENGTH_ERROR_TAG


@pytest.mark.asyncio
async def test_call_llm_server_reports_unknown_format_for_empty_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_fake_openai_client(monkeypatch, content=None)
    _patch_without_execution_context(monkeypatch)
    client = visit_module.get_openai_summary_client({
        'summary_client': 'openai',
        'jina_model_app_key': MODEL_APP_KEY,
    })

    result = await visit_module.call_llm_server(
        [{'role': 'user', 'content': [{'value': 'hi', 'type': 'text'}]}],
        MODEL_APP_ID,
        MODEL_APP_KEY,
        CUSTOM_MODEL,
        summary_client=client,
    )

    assert result == visit_module.VISIT_LLM_UNKNOWN_FORMAT_TAG
    assert captured['request']['model'] == CUSTOM_MODEL


@pytest.mark.asyncio
async def test_call_llm_server_reraises_terminal_provider_error_from_aihub() -> None:
    class ExplodingClient:
        async def create_completion(self, **kwargs):
            raise NonRetryableProviderError('account has no budget left')

    with pytest.raises(NonRetryableProviderError):
        await visit_module.call_llm_server(
            [{'role': 'user', 'content': [{'value': 'hi', 'type': 'text'}]}],
            MODEL_APP_ID,
            MODEL_APP_KEY,
            CUSTOM_MODEL,
            summary_client=ExplodingClient(),
        )
