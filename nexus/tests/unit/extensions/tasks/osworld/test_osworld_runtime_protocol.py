"""OSWorld osworld runtime protocol regressions."""

from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest

import nexus.runtime.providers.ags.base as ags_base

from nexus.extensions.tasks.os_world.execution import runtime as ags_osworld
from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime as AGSOSWorldRuntime
from nexus.runtime.executors.models.file import FileItem
from nexus.runtime.providers.ags import AGSInstanceHandle
from nexus.utils.async_http import HTTPResponse


_PNG_BYTES = b'\x89PNG\r\n\x1a\n' + b'png'


_JPEG_BYTES = b'\xff\xd8\xff' + b'jpeg'


_HTML_BYTES = b'<html>not an image</html>'


@pytest.mark.asyncio
async def test_ags_image_cdp_handoff_suspends_and_restores_service(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-task-cdp',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        ),
        cdp_proxy_mode='image',
    )
    scripts: list[str] = []

    async def fake_run(_runtime, script, **kwargs):
        _ = kwargs
        scripts.append(script)
        if 'cdp-handoff-suspended' in script:
            return {'returncode': 0, 'output': 'cdp-handoff-suspended', 'error': ''}
        if 'cdp-handoff-restored' in script:
            return {'returncode': 0, 'output': 'cdp-handoff-restored', 'error': ''}
        raise AssertionError(f'unexpected script: {script}')

    monkeypatch.setattr(ags_osworld, '_run_osworld_bash_for_cdp_setup', fake_run)

    events = []
    async with runtime.task_owned_chromium_cdp(client_password='password'):
        events.append('body')

    assert events == ['body']
    assert len(scripts) == 2
    assert 'systemctl stop cdp_proxy.service' in scripts[0]
    assert '/usr/local/bin/osworld-cdp-proxy' in scripts[0]
    assert 'systemctl start cdp_proxy.service' in scripts[1]
    assert 'fuser -k 9222/tcp' in scripts[1]


@pytest.mark.asyncio
async def test_ags_image_cdp_handoff_restores_after_body_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-task-cdp-failure',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        ),
        cdp_proxy_mode='image',
    )
    markers: list[str] = []

    async def fake_run(_runtime, script, **kwargs):
        _ = kwargs
        marker = 'suspend' if 'cdp-handoff-suspended' in script else 'restore'
        markers.append(marker)
        output = 'cdp-handoff-suspended' if marker == 'suspend' else 'cdp-handoff-restored'
        return {'returncode': 0, 'output': output, 'error': ''}

    monkeypatch.setattr(ags_osworld, '_run_osworld_bash_for_cdp_setup', fake_run)

    with pytest.raises(RuntimeError, match='task failed'):
        async with runtime.task_owned_chromium_cdp(client_password='password'):
            raise RuntimeError('task failed')

    assert markers == ['suspend', 'restore']


@pytest.mark.asyncio
async def test_ags_image_cdp_handoff_restores_after_body_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-task-cdp-cancel',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        ),
        cdp_proxy_mode='image',
    )
    markers: list[str] = []
    body_started = asyncio.Event()

    async def fake_run(_runtime, script, **kwargs):
        _ = kwargs
        marker = 'suspend' if 'cdp-handoff-suspended' in script else 'restore'
        markers.append(marker)
        output = 'cdp-handoff-suspended' if marker == 'suspend' else 'cdp-handoff-restored'
        return {'returncode': 0, 'output': output, 'error': ''}

    async def cancelled_body() -> None:
        async with runtime.task_owned_chromium_cdp(client_password='password'):
            body_started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(ags_osworld, '_run_osworld_bash_for_cdp_setup', fake_run)
    task = asyncio.create_task(cancelled_body())
    await body_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert markers == ['suspend', 'restore']


def test_ags_osworld_screenshot_format_validates_payload() -> None:
    assert (
        ags_osworld._screenshot_format(
            HTTPResponse(status=200, headers={}, text='', content=_PNG_BYTES, url='/screenshot', encoding='utf-8')
        )
        == 'png'
    )
    assert (
        ags_osworld._screenshot_format(
            HTTPResponse(status=200, headers={}, text='', content=_JPEG_BYTES, url='/screenshot', encoding='utf-8')
        )
        == 'jpeg'
    )
    assert (
        ags_osworld._screenshot_format(
            HTTPResponse(
                status=200,
                headers={'Content-Type': 'image/png'},
                text='',
                content=b'opaque-image-bytes',
                url='/screenshot',
                encoding='utf-8',
            )
        )
        == 'png'
    )
    assert (
        ags_osworld._screenshot_format(
            HTTPResponse(
                status=200,
                headers={'Content-Type': 'text/html'},
                text='<html></html>',
                content=_HTML_BYTES,
                url='/screenshot',
                encoding='utf-8',
            )
        )
        is None
    )


@pytest.mark.asyncio
async def test_ags_osworld_runtime_download_file_uses_file_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-file',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_post(url: str, **kwargs: object) -> HTTPResponse:
        calls.append((url, dict(kwargs)))
        return HTTPResponse(
            status=200,
            headers={'Content-Type': 'application/octet-stream'},
            text='hello\n',
            content=b'hello\n',
            url=url,
            encoding='utf-8',
        )

    monkeypatch.setattr(ags_osworld.http_client, 'post', _fake_post)

    assert await runtime.get_file('/tmp/demo.txt') == b'hello\n'
    response = await runtime.download_file(files=[FileItem(path='/tmp/demo.txt', encoding='base64')])

    assert response.files[0].content == 'aGVsbG8K'
    assert response.files[0].encoding == 'base64'
    assert response.errors is None
    assert calls == [
        (
            runtime.handle.url(5000, '/file'),
            {
                'data': {'file_path': '/tmp/demo.txt'},
                'timeout': ags_osworld._FILE_REQUEST_TIMEOUT,
                'headers': {'X-Access-Token': 'token'},
            },
        ),
        (
            runtime.handle.url(5000, '/file'),
            {
                'data': {'file_path': '/tmp/demo.txt'},
                'timeout': ags_osworld._FILE_REQUEST_TIMEOUT,
                'headers': {'X-Access-Token': 'token'},
            },
        ),
    ]


@pytest.mark.asyncio
async def test_ags_osworld_upload_file_uses_setup_upload_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-upload',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    captured: list[tuple[str, str, aiohttp.FormData]] = []

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        captured.append((method, url, kwargs['data']))
        return HTTPResponse(
            status=200, headers={}, text='File Uploaded: 5 bytes', content=b'', url=url, encoding='utf-8'
        )

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    result = await runtime.call_executor(
        'upload_file',
        {
            'files': [
                {
                    'path': '/home/user/a.txt',
                    'content': base64.b64encode(b'hello').decode('ascii'),
                    'encoding': 'base64',
                }
            ],
            'overwrite': True,
        },
    )

    assert result['results'][0]['success'] is True
    assert len(captured) == 1
    method, url, form = captured[0]
    assert method == 'POST'
    assert url.endswith('/setup/upload')
    fields = {headers['name']: (headers, value) for headers, _, value in form._fields}
    assert fields['file_path'][1] == '/home/user/a.txt'
    file_headers, file_content = fields['file_data']
    assert file_headers['filename'] == 'a.txt'
    assert file_content == b'hello'


@pytest.mark.asyncio
async def test_ags_osworld_setup_upload_file_streams_cached_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-setup-upload',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    local_path = tmp_path / 'asset.txt'
    local_path.write_bytes(b'cached asset')
    captured: dict[str, object] = {}

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        form = kwargs['data']
        fields = {headers['name']: (headers, value) for headers, _, value in form._fields}
        captured['method'] = method
        captured['url'] = url
        captured['remote_path'] = fields['file_path'][1]
        captured['filename'] = fields['file_data'][0]['filename']
        captured['content'] = fields['file_data'][1].read()
        return HTTPResponse(
            status=200, headers={}, text='File Uploaded: 12 bytes', content=b'', url=url, encoding='utf-8'
        )

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    response = await runtime.setup_upload_file(local_path, '/home/user/asset.txt')

    assert response.status_code == 200
    assert captured == {
        'method': 'POST',
        'url': runtime.handle.url(5000, '/setup/upload'),
        'remote_path': '/home/user/asset.txt',
        'filename': 'asset.txt',
        'content': b'cached asset',
    }


@pytest.mark.asyncio
async def test_ags_osworld_setup_close_window_sends_official_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-close-window',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    captured: dict[str, object] = {}

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        captured['method'] = method
        captured['url'] = url
        captured['json'] = kwargs['json']
        return HTTPResponse(status=200, headers={}, text='', content=b'', url=url, encoding='utf-8')

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    response = await runtime.setup_close_window('Message Filters', strict=True, by_class=True)

    assert response.status_code == 200
    assert captured == {
        'method': 'POST',
        'url': runtime.handle.url(5000, '/setup/close_window'),
        'json': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
    }


@pytest.mark.asyncio
async def test_ags_osworld_setup_activate_window_sends_official_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-activate-window',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    captured: dict[str, object] = {}

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        captured['method'] = method
        captured['url'] = url
        captured['json'] = kwargs['json']
        return HTTPResponse(status=200, headers={}, text='', content=b'', url=url, encoding='utf-8')

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    response = await runtime.setup_activate_window('Message Filters', strict=True, by_class=True)

    assert response.status_code == 200
    assert captured == {
        'method': 'POST',
        'url': runtime.handle.url(5000, '/setup/activate_window'),
        'json': {'window_name': 'Message Filters', 'strict': True, 'by_class': True},
    }


@pytest.mark.asyncio
async def test_ags_osworld_setup_launch_posts_command_to_launch_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-launch',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    captured: dict[str, object] = {}

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        captured['method'] = method
        captured['url'] = url
        captured['json'] = kwargs['json']
        return HTTPResponse(
            status=200, headers={}, text='launched successfully', content=b'', url=url, encoding='utf-8'
        )

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    response = await runtime.setup_launch(['tinyproxy', '-c', '/tmp/tinyproxy.conf', '-d'])

    assert response.status_code == 200
    assert captured == {
        'method': 'POST',
        'url': runtime.handle.url(5000, '/setup/launch'),
        'json': {'command': ['tinyproxy', '-c', '/tmp/tinyproxy.conf', '-d'], 'shell': False},
    }


@pytest.mark.asyncio
async def test_ags_osworld_gui_screenshot_retries_payload_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-screenshot-retry',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)

    calls: list[tuple[str, str, dict[str, object]]] = []
    screenshot_attempts = 0

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        nonlocal screenshot_attempts
        calls.append((method, url, dict(kwargs)))
        if url.endswith('/screenshot'):
            screenshot_attempts += 1
            if screenshot_attempts == 1:
                raise aiohttp.ClientPayloadError('truncated payload')
            return HTTPResponse(status=200, headers={}, text='png', content=_PNG_BYTES, url=url, encoding='utf-8')
        if url.endswith('/screen_size'):
            return HTTPResponse(
                status=200,
                headers={'Content-Type': 'application/json'},
                text='{"width": 1280, "height": 720}',
                content=b'{"width": 1280, "height": 720}',
                url=url,
                encoding='utf-8',
            )
        raise AssertionError(f'unexpected URL: {url}')

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    result = await runtime.call_executor('gui_screenshot', {})

    assert result['success'] is True
    assert result['image'] == base64.b64encode(_PNG_BYTES).decode('utf-8')
    assert result['width'] == 1280
    assert result['height'] == 720
    assert [url.endswith('/screenshot') for _, url, _ in calls].count(True) == 2


@pytest.mark.asyncio
async def test_ags_osworld_gui_screenshot_retries_invalid_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-invalid-screenshot',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)

    calls: list[tuple[str, str]] = []

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        calls.append((method, url))
        if url.endswith('/screenshot'):
            if len([call for call in calls if call[1].endswith('/screenshot')]) == 1:
                return HTTPResponse(
                    status=200,
                    headers={'Content-Type': 'text/html'},
                    text='<html></html>',
                    content=_HTML_BYTES,
                    url=url,
                    encoding='utf-8',
                )
            return HTTPResponse(status=200, headers={}, text='png', content=_PNG_BYTES, url=url, encoding='utf-8')
        if url.endswith('/screen_size'):
            return HTTPResponse(
                status=200,
                headers={'Content-Type': 'application/json'},
                text='{"width": 1280, "height": 720}',
                content=b'{"width": 1280, "height": 720}',
                url=url,
                encoding='utf-8',
            )
        raise AssertionError(f'unexpected URL: {url}')

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    result = await runtime.call_executor('gui_screenshot', {})

    assert result['success'] is True
    assert result['image'] == base64.b64encode(_PNG_BYTES).decode('utf-8')
    assert [(method, url.endswith('/screenshot')) for method, url in calls].count(('GET', True)) == 2


@pytest.mark.asyncio
async def test_ags_osworld_gui_screenshot_returns_error_after_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-invalid-screenshot-final',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)

    calls = 0

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        nonlocal calls
        calls += 1
        assert method == 'GET'
        assert url.endswith('/screenshot')
        return HTTPResponse(
            status=200,
            headers={'Content-Type': 'text/html'},
            text='<html></html>',
            content=_HTML_BYTES,
            url=url,
            encoding='utf-8',
        )

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    result = await runtime.call_executor('gui_screenshot', {})

    assert result['success'] is False
    assert result['error'] == 'Invalid OSWorld screenshot payload'
    assert calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('succeed_on', [2, 3, None])
async def test_screenshot_uses_increasing_attempt_deadlines(monkeypatch, succeed_on) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='screenshot-deadline', access_token='token', domain='ap-guangzhou.tencentags.com')
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)
    http_deadlines = []
    wall_deadlines = []
    real_wait_for = asyncio.wait_for

    async def wait_for(awaitable, *, timeout):
        wall_deadlines.append(timeout)
        return await real_wait_for(awaitable, timeout=timeout)

    async def request(method, url, **kwargs):
        if url.endswith('/screen_size'):
            assert kwargs['timeout'].total == 1800
            return HTTPResponse(
                status=200,
                headers={},
                content=b'{"width":1280,"height":720}',
                text='{"width":1280,"height":720}',
                url=url,
                encoding='utf-8',
            )
        http_deadlines.append(kwargs['timeout'].total)
        if len(http_deadlines) != succeed_on:
            raise asyncio.TimeoutError('screenshot deadline')
        return HTTPResponse(status=200, headers={}, content=_PNG_BYTES, text='', url=url, encoding='utf-8')

    monkeypatch.setattr(ags_osworld.asyncio, 'wait_for', wait_for)
    monkeypatch.setattr(ags_base.http_client, 'request', request)
    if succeed_on is None:
        with pytest.raises(asyncio.TimeoutError, match='screenshot deadline'):
            await runtime.call_executor('gui_screenshot', {})
    else:
        result = await runtime.call_executor('gui_screenshot', {})
        assert result['success'] is True
    expected = [30, 60, 120][:succeed_on]
    assert http_deadlines == wall_deadlines == expected
    assert runtime.request_timeout == 1800


@pytest.mark.asyncio
async def test_screenshot_wall_deadline_cancels_stalled_transport(monkeypatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='screenshot-stalled', access_token='token', domain='ap-guangzhou.tencentags.com')
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)
    monkeypatch.setattr(ags_osworld, '_SCREENSHOT_ATTEMPT_TIMEOUTS', (0.01, 0.02, 0.04))
    cancelled = []

    async def request(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(kwargs['timeout'].total)

    monkeypatch.setattr(ags_base.http_client, 'request', request)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(runtime.call_executor('gui_screenshot', {}), timeout=2)
    assert cancelled == [0.01, 0.02, 0.04]


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 403, 502])
async def test_screenshot_status_retry_budget(monkeypatch, status) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='screenshot-status', access_token='token', domain='ap-guangzhou.tencentags.com')
    )
    monkeypatch.setattr(ags_osworld, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)
    calls = []

    async def request(method, url, **kwargs):
        calls.append(kwargs['timeout'].total)
        return HTTPResponse(status=status, headers={}, content=b'error', text='error', url=url, encoding='utf-8')

    monkeypatch.setattr(ags_base.http_client, 'request', request)
    result = await runtime.call_executor('gui_screenshot', {})
    assert result['success'] is False
    assert calls == ([30, 60, 120] if status == 502 else [30])


@pytest.mark.asyncio
async def test_screenshot_does_not_retry_cancellation(monkeypatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='screenshot-cancelled', access_token='token', domain='ap-guangzhou.tencentags.com'
        )
    )
    calls = []

    async def request(*args, **kwargs):
        calls.append(kwargs['timeout'].total)
        raise asyncio.CancelledError

    monkeypatch.setattr(ags_base.http_client, 'request', request)
    with pytest.raises(asyncio.CancelledError):
        await runtime.call_executor('gui_screenshot', {})
    assert calls == [30]


@pytest.mark.asyncio
async def test_ags_osworld_gui_click_does_not_retry_execute_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-execute-disconnect',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    calls = 0

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        nonlocal calls
        calls += 1
        assert method == 'POST'
        assert url.endswith('/execute')
        raise aiohttp.ClientPayloadError('truncated execute payload')

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    with pytest.raises(aiohttp.ClientPayloadError):
        await runtime.call_executor('gui_click', {'x': 10, 'y': 20})
    assert calls == 1


@pytest.mark.asyncio
async def test_ags_osworld_run_command_does_not_retry_execute_status(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-execute-status',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    calls = 0

    async def _fake_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        nonlocal calls
        calls += 1
        assert method == 'POST'
        assert url.endswith('/execute')
        return HTTPResponse(
            status=503,
            headers={'Content-Type': 'application/json'},
            text='{"error": "busy"}',
            content=b'{"error": "busy"}',
            url=url,
            encoding='utf-8',
        )

    monkeypatch.setattr(ags_base.http_client, 'request', _fake_request)

    response = await runtime.call_executor(
        'run_command',
        {
            'command': 'printf hello',
            'timeout': 12,
        },
    )

    assert response['status'] == 'Error'
    assert calls == 1
