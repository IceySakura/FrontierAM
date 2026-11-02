from __future__ import annotations

import json

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from hylive.cerebellum_rules import injected, search_failure

from nexus.extensions.projects.hy_live.agent import LiveAgentAgentInput
from nexus.extensions.projects.hy_live.coordination.backend import WorkAgentBudget
from nexus.extensions.projects.hy_live.episode_clock import EpisodeClock
from nexus.extensions.projects.hy_live.hylive_shell import LiveShell
from nexus.extensions.projects.hy_live.task import LiveAgentTaskInput, live_agent_task
from nexus.extensions.projects.hy_live.user_simulator import ScriptedUserSimulator, ScriptedUserSimulatorConfig
from nexus.extensions.projects.hy_live.web_search import (
    EMPTY_TEXT,
    MAX_DOCS,
    MAX_TOTAL_CHARS,
    NOT_CONNECTED_TEXT,
    SEARCH_URL,
    WebSearchBackend,
    YuanbaoWebSearch,
    build_web_search_backend,
    extract_docs,
    latest_user_text,
    render_docs,
)


WEATHER_DOC = {
    'url': 'https://tianqi.example/shanghai',
    'title': '墨迹天气',
    'site_name': '墨迹天气',
    'recall_query': '上海今天天气 温度 2026年10月9日',
    'doc_authority_level': 3,
    'publish_time': '2026/10/09 13:00:25',
    'content': '上海天气\n当前多云,25.0°C',
}

WEATHER_BLOCK = (
    '[web_doc 1 begin]\n'
    '[search query] 上海今天天气 温度 2026年10月9日\n'
    '[web_doc_title] 墨迹天气\n'
    '[web_doc_site] 墨迹天气\n'
    '[web_doc_authority_level] 权威性较高\n'
    '[web_doc_publish_time] 2026/10/09 13:00:25\n'
    '[web_doc_content begin]\n'
    '上海天气\n当前多云,25.0°C\n'
    '[web_doc_content end]\n'
    '[web_doc 1 end]'
)
"""hy-live-demo's ``render_docs`` output for ``WEATHER_DOC``, the layout its v3 weather search shows."""


def _search_with(handler: Any) -> tuple[YuanbaoWebSearch, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return YuanbaoWebSearch(transport=httpx.MockTransport(record)), requests


def test_render_matches_the_demo_block_layout() -> None:
    assert render_docs(extract_docs(json.dumps({'docs': [WEATHER_DOC]}))) == WEATHER_BLOCK


def test_render_falls_back_to_abstract_and_unknown_fields() -> None:
    raw = json.dumps({'docs': [{'abstract': '摘要', 'doc_authority_level': 'high', 'pub_time': ''}]})
    assert render_docs(extract_docs(raw)) == (
        '[web_doc 1 begin]\n[search query] \n[web_doc_title] \n[web_doc_site] \n'
        '[web_doc_authority_level] 权威性未知\n[web_doc_publish_time] 发布时间未知\n'
        '[web_doc_content begin]\n摘要\n[web_doc_content end]\n[web_doc 1 end]'
    )


def test_extract_skips_repeated_urls_and_caps_the_doc_count() -> None:
    docs = [{**WEATHER_DOC, 'url': f'https://x/{index % 15}', 'content': str(index)} for index in range(30)]
    hits = extract_docs(json.dumps({'docs': docs}))
    assert [hit['content'] for hit in hits] == [str(index) for index in range(MAX_DOCS)]


@pytest.mark.parametrize('raw', ['not json', '[]', '{"docs": {}}', '{"docs": [{"title": "无正文"}]}'])
def test_unusable_responses_render_the_demo_empty_text(raw: str) -> None:
    assert render_docs(extract_docs(raw)) == EMPTY_TEXT


def test_render_truncates_one_oversized_first_doc_and_stops_at_the_char_budget() -> None:
    oversized = render_docs([{'content': 'a' * (MAX_TOTAL_CHARS + 5)}, {'content': 'b'}])
    assert oversized.count('[web_doc 1 begin]') == 1
    assert 'a' * MAX_TOTAL_CHARS + '......\n[web_doc_content end]' in oversized
    assert 'b\n' not in oversized

    half = MAX_TOTAL_CHARS // 2
    budget = render_docs([{'content': 'a' * half}, {'content': 'b' * half}, {'content': 'c'}])
    assert '[web_doc 2 end]' in budget
    assert '[web_doc 3 begin]' not in budget, 'A doc that would overflow the budget ends the list'


async def test_search_posts_the_demo_payload_and_returns_rendered_docs() -> None:
    search, requests = _search_with(lambda request: httpx.Response(200, json={'docs': [WEATHER_DOC]}))

    text = await search.search(['上海今天天气 温度 2026年10月9日', ' '], prompt='查一下上海今天的温度')

    assert text == WEATHER_BLOCK
    assert str(requests[0].url) == SEARCH_URL
    payload = json.loads(requests[0].content)
    request_id = payload['header']['session_id']
    assert payload == {
        'header': {'session_id': request_id, 'uuid': request_id, 'trace_id': request_id, 'yb_traffic_type': 0},
        'prompt': '查一下上海今天的温度',
        'source': 'agentic_search',
        'sub_query_info': [{'subq': '上海今天天气 温度 2026年10月9日', 'search_type': 0}],
        'messages': [{'role': 'user', 'content': '查一下上海今天的温度'}],
        'params': {
            'ext_params': {
                'open_google_api': '0',
                'open_wx_video': '0',
                'filter_level': '0',
                'enable_full_content': '1',
            }
        },
    }
    assert 'authorization' not in {key.lower() for key in requests[0].headers}


async def test_search_without_a_prompt_sends_the_first_query() -> None:
    search, requests = _search_with(lambda request: httpx.Response(200, json={'docs': []}))
    assert await search.search('上海 天气') == EMPTY_TEXT
    assert json.loads(requests[0].content)['prompt'] == '上海 天气'


async def _tool_text(search: YuanbaoWebSearch | None, raw_query: Any) -> str:
    """The ``web_search`` tool result the cerebellum gets for ``raw_query``: the runner hands hylive the text or the error."""
    if search is None:
        return NOT_CONNECTED_TEXT
    try:
        return await search.search(raw_query)
    except Exception as error:  # noqa: BLE001
        return search_failure(str(error))


async def test_search_retries_three_times_then_reports_the_demo_failure_text() -> None:
    search, requests = _search_with(lambda request: httpx.Response(503))

    text = await _tool_text(search, ['上海 天气'])

    assert len(requests) == 3
    assert text.startswith('搜索失败：搜索失败：Server error')
    assert '503' in text


async def test_search_recovers_when_a_retry_succeeds() -> None:
    responses = iter([httpx.Response(500), httpx.Response(200, json={'docs': [WEATHER_DOC]})])
    search, requests = _search_with(lambda request: next(responses))
    assert await search.search(['上海 天气']) == WEATHER_BLOCK
    assert len(requests) == 2


async def test_unreachable_endpoint_reports_the_connection_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('All connection attempts failed', request=request)

    search, _ = _search_with(refuse)
    assert await _tool_text(search, ['上海 天气']) == '搜索失败：搜索失败：All connection attempts failed'


async def test_empty_query_and_missing_backend_use_the_demo_failure_text() -> None:
    search, requests = _search_with(lambda request: httpx.Response(200, json={'docs': []}))
    assert await _tool_text(search, [' ']) == '搜索失败：web_search: query 不能为空'
    assert await _tool_text(search, None) == '搜索失败：web_search: query 不能为空'
    assert not requests
    assert await _tool_text(None, ['上海']) == NOT_CONNECTED_TEXT == '搜索失败：没有接上搜索'


def test_provider_builds_yuanbao_or_nothing() -> None:
    backend = build_web_search_backend('yuanbao')
    assert isinstance(backend, YuanbaoWebSearch)
    assert isinstance(backend, WebSearchBackend)
    assert build_web_search_backend('none') is None


async def test_task_forwards_a_serializable_provider_to_the_agent() -> None:
    forwarded: dict[str, Any] = {}

    async def run_agent(**kwargs: Any) -> Any:
        forwarded.update(kwargs)
        raise RuntimeError('stop after the call')

    task_input = LiveAgentTaskInput.model_validate({
        'prompt': '你好',
        'web_search_provider': 'yuanbao',
        'run_agent': run_agent,
    })
    assert task_input.model_dump(exclude={'run_agent'})['web_search_provider'] == 'yuanbao'
    with pytest.raises(RuntimeError, match='stop after the call'):
        await live_agent_task(task_input)
    assert forwarded['web_search_provider'] == 'yuanbao'
    assert LiveAgentAgentInput(prompt='你好').web_search_provider == 'none'
    with pytest.raises(ValueError, match='web_search_provider'):
        LiveAgentTaskInput.model_validate({'prompt': '你好', 'web_search_provider': 'google', 'run_agent': run_agent})


def test_latest_user_text_skips_injected_notices() -> None:
    log = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': '查一下上海今天的温度'},
        injected('[通知]\n有一项工作做完了。'),
        {'role': 'assistant', 'content': ''},
    ]
    assert latest_user_text(log) == '查一下上海今天的温度'
    assert latest_user_text([{'role': 'user', 'content': [{'type': 'text', 'text': '天气'}]}]) == '天气'
    assert latest_user_text([]) == ''


class _ScriptedClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = list(messages)

    async def create_completion(self, **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(message=self.messages.pop(0))])


async def test_cerebellum_tool_result_is_the_rendered_search_text() -> None:
    search, requests = _search_with(lambda request: httpx.Response(200, json={'docs': [WEATHER_DOC]}))
    call = {
        'id': 'call-1',
        'type': 'function',
        'function': {'name': 'web_search', 'arguments': json.dumps({'query': ['上海今天天气 温度 2026年10月9日']})},
    }
    client = _ScriptedClient([{'role': 'assistant', 'content': '', 'tool_calls': [call]}, {'content': '25℃。'}])

    async def get_runtime() -> Any:
        return SimpleNamespace(on=lambda *_: None)

    shell = LiveShell(
        ctx=SimpleNamespace(run_id='run-1', get_runtime=get_runtime, openai_clients={}),  # type: ignore[arg-type]
        owner_id='owner',
        clients={'cerebellum': client, 'midbrain': _ScriptedClient([])},
        cerebellum_system='sys',
        trusted_config={},
        budget=WorkAgentBudget(),
        search_backend=search,
        clock=EpisodeClock(),
    )
    try:
        live = await shell.run(
            simulator=ScriptedUserSimulator(ScriptedUserSimulatorConfig(turns=['查一下上海今天的温度'])),
            max_user_turns=1,
            task_delivery_timeout=5,
        )
    finally:
        await shell.close()

    assert live.episode.summary == '25℃。'
    tool_message = next(message for message in live.state.cerebellum.messages if message.get('role') == 'tool')
    assert tool_message['content'] == WEATHER_BLOCK
    assert json.loads(requests[0].content)['prompt'] == '查一下上海今天的温度'
