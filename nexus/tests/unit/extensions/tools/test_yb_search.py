from __future__ import annotations

import asyncio
import base64
import io

from unittest.mock import AsyncMock

import pytest

from nexus.core import ToolContext
from nexus.extensions.tools.yb_search import _yb_utils, yb_vision_search, yb_web_search
from nexus.utils.rate_limiter import (
    QpsLimiter,
    RateLimiter,
    get_qps_limiter,
    get_rate_limiter,
    reset_all_qps_limiters,
    reset_all_rate_limiters,
    reset_qps_limiter,
    reset_rate_limiter,
)


@pytest.fixture
def pil_image():
    return pytest.importorskip(
        'PIL.Image',
        minversion='10.0',
        reason='YB image operation tests require the yb_search Pillow dependency',
    )


@pytest.mark.asyncio
async def test_yb_web_search_preserves_queries_and_returns_doc_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search = AsyncMock(
        return_value={
            'docs': [
                {
                    'recall_query': ' first query ',
                    'title': 'First result',
                    'site_name': 'Example',
                    'url': 'https://example.test/first',
                    'publish_time': '2026-06-23',
                    'content': 'Useful content',
                    'doc_authority_level': 4,
                },
                {
                    'recall_query': 'second query',
                    'title': 'Hidden result',
                    'url': 'https://example.test/hidden',
                    'abstract': 'Hidden content',
                    'invisible': 1,
                },
            ]
        }
    )
    monkeypatch.setattr(yb_web_search, '_prompt_search', search)
    monkeypatch.setattr(yb_web_search, 'get_current_time_str', lambda: 'fixed time')

    response = await yb_web_search.yb_web_search_tool(
        yb_web_search.YbWebSearchInput(query=[' first query ', '', 'second query']),
        ToolContext(
            tool_configs={'yb_web_search': {'max_queries': 2}},
            tool_state={'yb_search.user_question': 'original question'},
        ),
    )

    assert search.await_args is not None
    assert search.await_args.args[:2] == ('original question', [' first query ', 'second query'])
    assert response.meta['count'] == 2
    assert len(response.meta['doc_segments']) == 2
    assert '[web_doc 1 begin]' in response.output[0]['text']
    assert '[web_doc_displayable]False' in response.meta['doc_segments'][1]
    assert response.output[0]['text'].startswith('当前时间：fixed time\n')
    assert response.meta['no_resp_queries'] == []


@pytest.mark.asyncio
async def test_yb_web_search_preserves_oneagent_no_hit_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        yb_web_search,
        '_prompt_search',
        AsyncMock(return_value={'docs': [{'recall_query': 'hit', 'title': 'title', 'content': 'x' * 10001}]}),
    )
    monkeypatch.setattr(yb_web_search, 'get_current_time_str', lambda: 'fixed time')

    response = await yb_web_search.yb_web_search_tool(
        yb_web_search.YbWebSearchInput(query=['hit', 'miss']),
        ToolContext(),
    )

    assert response.meta['no_resp_queries'] == ['miss']
    assert '使用web_search检索工具， "miss"未能检索到相关信息；' in response.output[0]['text']
    assert '通过其它search query获取的检索信息如下：' in response.output[0]['text']
    assert 'x' * 10001 in response.meta['doc_segments'][0]


@pytest.mark.asyncio
async def test_yb_web_search_preserves_backend_exception_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yb_web_search, '_prompt_search', AsyncMock(side_effect=RuntimeError('discovery failed')))
    monkeypatch.setattr(yb_web_search, 'get_current_time_str', lambda: 'fixed time')

    response = await yb_web_search.yb_web_search_tool(
        yb_web_search.YbWebSearchInput(query=['query']),
        ToolContext(),
    )

    assert response.output == [
        {'type': 'text', 'text': '当前时间：fixed time\nWeb search failed: discovery failed'}
    ]
    assert response.meta == {'success': False, 'error': 'discovery failed', 'doc_segments': []}


@pytest.mark.asyncio
async def test_yb_web_search_preserves_backend_none_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yb_web_search, '_prompt_search', AsyncMock(return_value=None))
    monkeypatch.setattr(yb_web_search, 'get_current_time_str', lambda: 'fixed time')

    response = await yb_web_search.yb_web_search_tool(
        yb_web_search.YbWebSearchInput(query=['query']),
        ToolContext(),
    )

    assert response.output == [
        {
            'type': 'text',
            'text': (
                '当前时间：fixed time\n'
                'Web search failed: backend returned None (network or service error).'
            ),
        }
    ]
    assert response.meta == {'success': False, 'error': 'backend_none', 'doc_segments': []}


class _NoopLimiter:
    def __init__(self, name: str | None = None, events: list[str] | None = None):
        self.name = name
        self.events = events

    async def __aenter__(self):
        if self.events is not None:
            self.events.append(f'enter:{self.name}')
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.events is not None:
            self.events.append(f'exit:{self.name}')
        return False


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_yb_web_search_backend_uses_mocked_discovery_and_httpx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery = type(
        'Discovery',
        (),
        {'get_url': lambda self, path: f'http://search.internal{path}', 'invalidate': lambda self: None},
    )()
    client = _FakeClient({'docs': []})
    limiter_calls = []
    limiter_events = []
    monkeypatch.setattr(yb_web_search, 'get_service_discovery', lambda *args: discovery)
    monkeypatch.setattr(yb_web_search, 'get_shared_httpx_client', lambda *args, **kwargs: client)
    monkeypatch.setattr(
        yb_web_search,
        'get_rate_limiter',
        lambda name, value: (
            limiter_calls.append(('concurrency', name, value)) or _NoopLimiter('concurrency', limiter_events)
        ),
    )
    monkeypatch.setattr(
        yb_web_search,
        'get_qps_limiter',
        lambda name, value: limiter_calls.append(('qps', name, value)) or _NoopLimiter('qps', limiter_events),
    )
    config = yb_web_search._runtime_config({
        'service_namespace': 'Test',
        'service_name': 'Search',
        'path': '/prompt-search',
        'retries': 1,
    })

    result = await yb_web_search._prompt_search('original', ['query'], config)

    assert result == {'docs': []}
    assert client.calls[0][0] == 'http://search.internal/prompt-search'
    assert client.calls[0][1]['json']['prompt'] == 'original'
    assert client.calls[0][1]['json']['source'] == 'vlm-agent-rl'
    assert client.calls[0][1]['json']['sub_query_info'][0]['subq'] == 'query'
    assert limiter_calls == [
        ('concurrency', 'yb_web_search:Test:Search:1.0:2', 2),
        ('qps', 'yb_web_search:Test:Search:1.0:2', 1.0),
    ]
    assert limiter_events == ['enter:concurrency', 'enter:qps', 'exit:qps', 'exit:concurrency']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('module', 'search', 'config_overrides', 'expected_result'),
    [
        (yb_web_search, yb_web_search._prompt_search, {}, {'docs': []}),
        (yb_vision_search, yb_vision_search._image_search, {}, []),
    ],
)
async def test_yb_search_qps_zero_disables_qps_limiter(
    monkeypatch: pytest.MonkeyPatch,
    module,
    search,
    config_overrides,
    expected_result,
) -> None:
    discovery = type(
        'Discovery',
        (),
        {'get_url': lambda self, path: f'http://search.internal{path}', 'invalidate': lambda self: None},
    )()
    payload = {'docs': []} if module is yb_web_search else {'code': 0, 'imgs': []}
    client = _FakeClient(payload)
    limiter_events = []
    monkeypatch.setattr(module, 'get_service_discovery', lambda *args: discovery)
    monkeypatch.setattr(module, 'get_shared_httpx_client', lambda *args, **kwargs: client)
    monkeypatch.setattr(module, 'get_rate_limiter', lambda *args: _NoopLimiter('concurrency', limiter_events))
    monkeypatch.setattr(
        module,
        'get_qps_limiter',
        lambda *args: pytest.fail('get_qps_limiter must not be called when qps is zero'),
    )
    config = module._runtime_config({'qps': 0, 'retries': 1, **config_overrides})

    if module is yb_web_search:
        result = await search('original', ['query'], config)
    else:
        result = await search(b'jpeg-bytes', 'prompt', config)

    assert result == expected_result
    assert config['qps'] == 0
    assert limiter_events == ['enter:concurrency', 'exit:concurrency']


@pytest.mark.asyncio
async def test_yb_vision_search_resolves_first_user_image_and_returns_mixed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare = AsyncMock(return_value=(object(), 95, b'jpeg-bytes'))
    search = AsyncMock(
        return_value=[
            {
                'image_url': 'https://images.example.test/result.jpg',
                'caption': 'matching image',
                'date_published': '2026-06-23',
            }
        ]
    )
    monkeypatch.setattr(yb_vision_search, '_prepare_image', prepare)
    monkeypatch.setattr(yb_vision_search, '_image_search', search)
    monkeypatch.setattr(yb_vision_search, 'get_current_time_str', lambda: 'fixed time')
    download = AsyncMock(return_value=('data:image/jpeg;base64,encoded', 320, 256))
    monkeypatch.setattr(yb_vision_search, 'download_image_as_data_uri', download)

    response = await yb_vision_search.yb_vision_search_tool(
        yb_vision_search.YbVisionSearchInput(
            query=[yb_vision_search.VisionSearchQueryItem(source='user_image 1', text='identify this')]
        ),
        ToolContext(
            tool_configs={},
            tool_state={
                'yb_search.images': ['/tmp/input.png'],
                'yb_search.round_image_tokens': 10,
                'token_counter': type('TokenCounter', (), {'count_image_tokens': lambda self, w, h: w + h})(),
            },
        ),
    )

    prepare.assert_awaited_once()
    assert search.await_count == 1
    assert response.meta['rendered_count'] == 1
    assert [part['type'] for part in response.output] == ['text', 'text', 'image_url', 'text']
    assert response.output[0]['text'] == '当前时间：fixed time\n'
    assert response.output[2]['image_url'] == {'url': 'data:image/jpeg;base64,encoded'}
    assert '[search query]{"source": "user_image 1", "text": "identify this"}' in response.output[1]['text']
    assert response.meta['image_sizes'] == [(320, 256)]
    assert response.meta['image_download_failures'] == 0
    assert response.meta['image_tokens'] == 576
    assert response.context_updates == {
        'yb_search.next_image_idx': 2,
        'yb_search.round_image_tokens': 586,
    }


@pytest.mark.asyncio
async def test_yb_vision_search_continues_image_index_from_tool_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yb_vision_search, '_prepare_image', AsyncMock(return_value=(object(), 95, b'jpeg-bytes')))
    monkeypatch.setattr(
        yb_vision_search,
        '_image_search',
        AsyncMock(return_value=[{'image_url': 'https://images.example.test/result.jpg'}]),
    )
    monkeypatch.setattr(
        yb_vision_search,
        'download_image_as_data_uri',
        AsyncMock(return_value=('data:image/jpeg;base64,encoded', 320, 256)),
    )

    response = await yb_vision_search.yb_vision_search_tool(
        yb_vision_search.YbVisionSearchInput(
            query=[yb_vision_search.VisionSearchQueryItem(source='user_image 1', text='identify this')]
        ),
        ToolContext(
            tool_configs={'yb_vision_search': {'start_idx': 2}},
            tool_state={
                'yb_search.images': ['/tmp/input.png'],
                'yb_search.next_image_idx': 7,
            },
        ),
    )

    assert '[web_image 7 begin]' in response.output[1]['text']
    assert '[web_image 7 end]' in response.output[-1]['text']
    assert response.meta['success'] is True
    assert response.meta['rendered_count'] == 1
    assert response.context_updates == {'yb_search.next_image_idx': 8}


@pytest.mark.asyncio
async def test_yb_vision_search_rejects_out_of_range_user_image(monkeypatch: pytest.MonkeyPatch) -> None:
    search = AsyncMock()
    monkeypatch.setattr(yb_vision_search, '_image_search', search)
    monkeypatch.setattr(yb_vision_search, 'get_current_time_str', lambda: 'fixed time')
    response = await yb_vision_search.yb_vision_search_tool(
        yb_vision_search.YbVisionSearchInput(
            query=[yb_vision_search.VisionSearchQueryItem(source='user_image 2', text='identify this')]
        ),
        ToolContext(tool_configs={}, tool_state={'yb_search.images': ['/tmp/only-one.png']}),
    )

    search.assert_not_awaited()
    assert response.meta['no_hit_sources'] == ['user_image 2']
    assert (
        '使用vision_search检索工具，"{"source": "user_image 2", "text": "identify this"}"' in response.output[0]['text']
    )


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], content_length: str | None = None):
        self.chunks = chunks
        self.headers = {} if content_length is None else {'Content-Length': content_length}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk


class _FakeStreamClient:
    def __init__(self, response: _FakeStreamResponse):
        self.response = response

    def stream(self, method: str, url: str, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_yb_image_download_enforces_streaming_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeStreamClient(_FakeStreamResponse([b'1234', b'5678']))
    monkeypatch.setattr(_yb_utils, 'get_shared_httpx_client', lambda *args, **kwargs: client)

    with pytest.raises(ValueError, match='exceeds 6 bytes'):
        await _yb_utils.download_image('https://images.example.test/image.jpg', max_bytes=6)

    declared_large = _FakeStreamClient(_FakeStreamResponse([], content_length='7'))
    monkeypatch.setattr(_yb_utils, 'get_shared_httpx_client', lambda *args, **kwargs: declared_large)
    with pytest.raises(ValueError, match='exceeds 6 bytes'):
        await _yb_utils.download_image('https://images.example.test/image.jpg', max_bytes=6)


def test_yb_compress_image_converts_rgba_to_bounded_jpeg(pil_image) -> None:
    source = io.BytesIO()
    pil_image.new('RGBA', (256, 256), (255, 0, 0, 128)).save(source, format='PNG')

    _, quality, compressed, jpeg_bytes = _yb_utils.compress_image(
        source.getvalue(),
        max_size=20_000,
        quality=95,
    )

    assert compressed is not None
    assert compressed.mode == 'RGB'
    assert 5 <= quality <= 95
    assert jpeg_bytes is not None
    assert jpeg_bytes.startswith(b'\xff\xd8')
    assert len(jpeg_bytes) <= 20_000

    with pil_image.open(io.BytesIO(jpeg_bytes)) as decoded:
        assert decoded.mode == 'RGB'
        assert decoded.size == (256, 256)


def test_yb_image_bytes_to_data_uri_resizes_and_reports_encoded_size(pil_image) -> None:
    source = io.BytesIO()
    pil_image.new('RGB', (400, 200), (0, 127, 255)).save(source, format='PNG')

    data_uri, width, height = _yb_utils.image_bytes_to_data_uri(
        source.getvalue(),
        min_pixels=32 * 32,
        max_pixels=64 * 64,
        factor=32,
        max_ratio=200,
    )

    assert data_uri.startswith('data:image/jpeg;base64,')
    assert (width, height) == (64, 32)
    encoded = data_uri.split(',', 1)[1]
    with pil_image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
        assert decoded.mode == 'RGB'
        assert decoded.size == (width, height)


@pytest.mark.asyncio
async def test_rate_limiters_release_state_and_reset() -> None:
    reset_all_rate_limiters()
    concurrency = RateLimiter('test-concurrency', max_concurrent=2)
    pacing = QpsLimiter('test-qps', qps=10000)

    async def run_once() -> None:
        async with pacing, concurrency:
            await asyncio.sleep(0)

    await asyncio.gather(*(run_once() for _ in range(4)))
    assert concurrency.current_count == 0
    assert pacing.snapshot()['waiting'] == 0
    assert pacing.snapshot()['issued'] == 4

    slow_pacing = QpsLimiter('cancelled-qps', qps=0.1)
    await slow_pacing.acquire()
    waiting = asyncio.create_task(slow_pacing.acquire())
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert slow_pacing.snapshot()['waiting'] == 0

    old_concurrency = get_rate_limiter('registered', 1)
    old_pacing = get_qps_limiter('registered', 10000)
    reset_rate_limiter('registered')
    reset_qps_limiter('registered')
    assert get_rate_limiter('registered', 1) is not old_concurrency
    assert get_qps_limiter('registered', 10000) is not old_pacing
    reset_all_qps_limiters()
    reset_all_rate_limiters()
