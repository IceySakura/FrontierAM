"""COS image download and integration with the visual agent's existing pipeline."""

import base64
import copy
import sys
import traceback

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from nexus.extensions.agents.main.agent import MainAgentOutput
from nexus.extensions.agents.visual_agent import agent as agent_module
from nexus.extensions.agents.visual_agent.resources import cos_images, images
from nexus.runner.single import RunInput, run


PUBLIC = 'https://example-1250000000.cos.ap-guangzhou.myqcloud.com/photos/image.png?q-signature=test-only&x=%2F'
INTERNAL = 'http://example-1250000000.cos-internal.ap-guangzhou.tencentcos.cn/photo.png?q-signature=test-only'


def image_bytes() -> bytes:
    return base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAABgAAAAQCAIAAACDRijCAAAAGklEQVR4nGP8z0AdwEQlc0YNGjVo1KARaxAA2D0BHwIRnKYAAAAASUVORK5CYII='
    )


@pytest.mark.parametrize(
    'url',
    [
        PUBLIC,
        INTERNAL,
        'https://example-1250000000.cos.ap-shanghai.tencentcos.cn/a',
        'https://example-1250000000.cos-internal.ap-beijing.myqcloud.com/a',
    ],
)
def test_standard_cos_hosts_are_recognized(url: str) -> None:
    assert cos_images.is_cos_image_url(url)


@pytest.mark.parametrize(
    'url',
    [
        'https://example.com/a.png',
        'https://bucket.cos.ap-guangzhou.myqcloud.com.evil.example/a',
        'https://myqcloud.com/a',
        'https://example.com/bucket.cos.ap-guangzhou.myqcloud.com/a',
        'cos://bucket/key',
        'data:image/png;base64,AAAA',
        '/tmp/image.png',
        'https://[invalid',
    ],
)
def test_other_references_are_not_cos_downloads(url: str) -> None:
    assert not cos_images.is_cos_image_url(url)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    state = SimpleNamespace(payload=image_bytes(), status=200, calls=[], sessions=[], fail=False, length=None)

    class Content:
        async def iter_chunked(self, size):
            for index in range(0, len(state.payload), 7):
                yield state.payload[index : index + 7]

    class Response:
        async def __aenter__(self):
            return SimpleNamespace(status=state.status, content_length=state.length, content=Content())

        async def __aexit__(self, *args):
            return False

    class Session:
        def __init__(self, **kwargs):
            state.sessions.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url, **kwargs):
            state.calls.append((str(url), kwargs))
            if state.fail:
                raise aiohttp.ClientConnectionError(str(url))
            return Response()

    monkeypatch.setattr(cos_images.aiohttp, 'ClientSession', Session)
    return state


@pytest.mark.parametrize(
    ('suffix', 'mime'),
    [
        ('.png', 'image/png'),
        ('.PNG', 'image/png'),
        ('.jpg', 'image/jpeg'),
        ('.jpeg', 'image/jpeg'),
        ('.webp', 'image/webp'),
        ('.gif', 'image/gif'),
        ('.bmp', 'image/bmp'),
        ('', 'image/jpeg'),
        ('.unknown', 'image/jpeg'),
    ],
)
async def test_signed_url_is_preserved_and_mime_comes_from_path(transport, suffix: str, mime: str) -> None:
    url = PUBLIC.replace('image.png', f'image{suffix}') + '&filename=misleading.gif'
    resolved = await cos_images.download_cos_images([url, url])
    assert transport.calls == [(url, {'allow_redirects': False})]
    assert transport.sessions[0]['trust_env'] is False
    assert transport.sessions[0]['timeout'].total == 60
    assert resolved[url].startswith(f'data:{mime};base64,')
    assert base64.b64decode(resolved[url].split(',', 1)[1]) == transport.payload


@pytest.mark.parametrize('status', [302, 403, 404, 500])
async def test_http_failures_do_not_expose_signed_query(transport, status: int) -> None:
    transport.status = status
    with pytest.raises(ValueError, match=f'HTTP {status}') as exc:
        await cos_images.download_cos_images([PUBLIC])
    assert 'test-only' not in str(exc.value)
    assert 'q-signature' not in str(exc.value)


async def test_connection_failure_hides_underlying_signed_url(transport) -> None:
    transport.fail = True
    with pytest.raises(ValueError, match='check connectivity') as exc:
        await cos_images.download_cos_images([PUBLIC])
    assert PUBLIC not in ''.join(traceback.format_exception(exc.value))


async def test_download_preserves_payload_without_image_validation(transport) -> None:
    transport.payload = b'opaque downloaded bytes'
    resolved = await cos_images.download_cos_images([PUBLIC])
    assert base64.b64decode(resolved[PUBLIC].split(',', 1)[1]) == transport.payload


async def test_empty_response_is_rejected(transport) -> None:
    transport.payload = b''
    with pytest.raises(ValueError, match='empty response'):
        await cos_images.download_cos_images([PUBLIC])


@pytest.mark.parametrize('known_length', [False, True])
async def test_oversized_response_is_rejected(transport, monkeypatch: pytest.MonkeyPatch, known_length: bool) -> None:
    monkeypatch.setattr(cos_images, '_MAX_IMAGE_BYTES', 10)
    transport.length = len(transport.payload) if known_length else None
    with pytest.raises(ValueError, match='download limit'):
        await cos_images.download_cos_images([PUBLIC])


@pytest.mark.parametrize('upload', [False, True])
async def test_cos_image_inlines_without_runtime_and_preserves_original(transport, upload: bool) -> None:
    original = [
        {'type': 'text', 'text': 'before'},
        {'type': 'image_url', 'image_url': {'url': PUBLIC, 'detail': 'high'}},
        {'type': 'text', 'text': 'after'},
        {'type': 'image', 'image_url': PUBLIC},
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/ordinary.png'}},
    ]
    before = copy.deepcopy(original)
    resolved = await images.resolve_user_prompt(original, upload_to_runtime=upload)
    assert len(transport.calls) == 1
    assert original == before
    assert resolved[0] == original[0]
    assert resolved[2] == original[2]
    assert resolved[1]['image_url']['detail'] == 'high'
    assert resolved[1]['image_url']['url'].startswith('data:image/png;base64,')
    assert resolved[3]['image_url']['url'] == resolved[1]['image_url']['url']
    assert resolved[4] == original[4]


async def test_cos_download_failure_prevents_model_call(transport, main_delegate) -> None:
    transport.status = 403
    result = await run(
        RunInput(
            task_name='visual',
            task_config={'user_prompt': [{'type': 'image_url', 'image_url': {'url': INTERNAL}}]},
            agent_name='visual_agent',
        )
    )
    assert result.success is False
    assert 'HTTP 403' in result.error.message
    main_delegate.assert_not_awaited()


async def test_cos_upload_failure_stops_resolution(transport, monkeypatch):
    failing_upload = AsyncMock(side_effect=RuntimeError('sandbox unavailable'))
    monkeypatch.setattr(images, 'upload_images_to_runtime', failing_upload)
    monkeypatch.setattr(images, 'require_context', lambda: SimpleNamespace(runtime_provider=object()))
    with pytest.raises(RuntimeError, match='sandbox unavailable'):
        await images.resolve_user_prompt([{'type': 'image_url', 'image_url': {'url': INTERNAL}}])
    failing_upload.assert_awaited_once()


async def test_timeout_does_not_expose_signed_url(transport, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args, **kwargs):
        raise TimeoutError(PUBLIC)

    transport.sessions.clear()
    original_session = cos_images.aiohttp.ClientSession

    class TimeoutSession(original_session):
        get = timeout

    monkeypatch.setattr(cos_images.aiohttp, 'ClientSession', TimeoutSession)
    with pytest.raises(ValueError, match='check connectivity') as exc:
        await cos_images.download_cos_images([PUBLIC])
    assert PUBLIC not in ''.join(traceback.format_exception(exc.value))


@pytest.mark.parametrize('task_name', ['identity', 'visual'])
@pytest.mark.parametrize('inline', [False, True])
@pytest.mark.parametrize('position', ['before', 'after'])
async def test_cos_image_uploads_without_serializing_sources_in_tool_configs(
    transport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_name: str, inline: bool, position: str
) -> None:
    monkeypatch.setattr(images, 'RUNTIME_UPLOAD_DIR', str(tmp_path / 'sandbox'))
    prompt = [
        {'type': 'text', 'text': 'Inspect'},
        {'type': 'image_url', 'image_url': {'url': PUBLIC}},
        {'type': 'image_url', 'image_url': {'url': PUBLIC}},
    ]
    captured = []

    async def run_main(main_input):
        from nexus.core import require_context

        assert require_context().tool_configs == {}
        sandbox_files = list((tmp_path / 'sandbox').rglob('*.png'))
        assert len(sandbox_files) == 1
        assert sandbox_files[0].read_bytes() == transport.payload
        captured.append(main_input)
        return MainAgentOutput(main_trajectory=[], exit_status='finished', final_answer='ok')

    monkeypatch.setattr(agent_module, 'run_main_agent', run_main)
    config = {'show_image_dimensions': True, 'inline_images': inline, 'input_image_path_position': position}
    if task_name == 'identity':
        config['user_prompt'] = prompt
    result = await run(
        RunInput(
            task_name=task_name,
            task_config={'user_prompt': prompt} if task_name == 'visual' else {},
            agent_name='visual_agent',
            agent_config=config,
            runtime_provider_name='local',
        )
    )
    assert result.success, result.error
    assert len(transport.calls) == 1
    assert len(captured) == 1
    parts = captured[0].user_prompt
    if inline:
        for index, part in enumerate(parts):
            if part['type'] == 'image_url':
                note_index = index - 1 if position == 'before' else index + 1
                assert 0 <= note_index < len(parts)
                assert parts[note_index]['type'] == 'text'
                assert '![' in parts[note_index]['text']
                assert '(24x16 px)' in parts[note_index]['text']
    assert len([part for part in parts if part['type'] == 'image_url']) == (2 if inline else 0)
    notes = [part['text'] for part in parts if part['type'] == 'text']
    assert sum('(24x16 px)' in note for note in notes) == 2
    assert all('q-signature' not in str(part) for part in parts)


async def test_visual_task_delivers_cos_image_to_model_and_sandbox_without_pillow(
    transport, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, 'PIL', None)
    monkeypatch.setattr(images, 'RUNTIME_UPLOAD_DIR', str(tmp_path / 'sandbox'))
    prompt = [{'type': 'text', 'text': 'Original query  '}, {'type': 'image_url', 'image_url': {'url': PUBLIC}}]
    before = copy.deepcopy(prompt)
    captured = []

    async def run_main(main_input):
        from nexus.core import require_context

        assert require_context().tool_configs == {}
        sandbox_files = list((tmp_path / 'sandbox').rglob('*.png'))
        assert len(sandbox_files) == 1
        assert sandbox_files[0].read_bytes() == transport.payload
        parts = main_input.user_prompt
        assert parts[0] == prompt[0]
        image_parts = [part for part in parts if part['type'] == 'image_url']
        assert len(image_parts) == 1
        url = image_parts[0]['image_url']['url']
        assert url.startswith('data:image/png;base64,')
        assert base64.b64decode(url.partition(',')[2]) == transport.payload
        captured.append(main_input)
        return MainAgentOutput(main_trajectory=[], exit_status='finished', final_answer='ok')

    monkeypatch.setattr(agent_module, 'run_main_agent', run_main)
    result = await run(
        RunInput(
            task_name='visual',
            task_config={'user_prompt': prompt},
            agent_name='visual_agent',
            agent_config={'show_image_dimensions': False},
            runtime_provider_name='local',
        )
    )
    assert result.success, result.error
    assert len(captured) == 1
    assert len(transport.calls) == 1
    assert prompt == before


async def test_ordinary_urls_and_non_image_cos_parts_are_not_downloaded(transport) -> None:
    ordinary = [{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.png'}}]
    assert await images.resolve_user_prompt(ordinary) == ordinary
    with pytest.raises(ValueError, match='remote file URL'):
        await images.resolve_user_prompt([{'type': 'file', 'file': {'url': PUBLIC}}])
    with pytest.raises(ValueError, match='remote video URL'):
        await images.resolve_user_prompt([{'type': 'video_url', 'video_url': {'url': INTERNAL}}])
    assert transport.calls == []


async def test_download_works_with_real_http_transport() -> None:
    from aiohttp import web

    payload = image_bytes()
    seen = []

    async def serve(request):
        seen.append(request.raw_path)
        return web.Response(body=payload, content_type='application/octet-stream')

    app = web.Application()
    app.router.add_get('/image.png', serve)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = runner.addresses[0][1]
        async with aiohttp.ClientSession() as session:
            resolved = await cos_images._download_image(session, f'http://127.0.0.1:{port}/image.png?q-signature=a%2Fb')
        assert base64.b64decode(resolved.split(',', 1)[1]) == payload
        assert seen == ['/image.png?q-signature=a%2Fb']
    finally:
        await runner.cleanup()
