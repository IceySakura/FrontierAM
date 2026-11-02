"""DevCloud tests with mocked HTTP; no real resources are created."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_PNG_BYTES = b'\x89PNG\r\n\x1a\npng-bytes'


def _make_provider(**kwargs):
    from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider

    defaults = {
        'app_id': 'app-123',
        'app_token': 'secret-token',
        'sandbox_api_base_url': 'http://sandbox.example',
    }
    defaults.update(kwargs)
    return DevCloudRuntimeProvider(**defaults)


def _response(status_code: int, body: dict | None = None, text: str | None = None, content: bytes | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {}
    resp.text = text if text is not None else __import__('json').dumps(body or {})
    resp.content = content if content is not None else resp.text.encode('utf-8')
    resp.json = MagicMock(return_value=body or {})
    return resp


@pytest.mark.asyncio
@pytest.mark.parametrize('proxy_url', [None, 'http://proxy.example:1080'])
@pytest.mark.parametrize('execute_mode', ['sync', 'async'])
async def test_devcloud_proxy_covers_lifecycle_and_health(proxy_url: str | None, execute_mode: str) -> None:
    base_url = 'http://sandbox.example'
    provider = _make_provider(sandbox_api_base_url=base_url + '/', proxy_url=proxy_url, execute_mode=execute_mode)
    responses = [
        _response(200, {'actionCode': 200, 'data': {'taskId': 'create-1'}}),
        _response(200, {'actionCode': 200, 'data': {'status': 'succeeded', 'ip': '10.0.0.9', 'id': -42}}),
        _response(200, {'actionCode': 200, 'data': {'taskId': 'delete-1'}}),
        _response(200, {'actionCode': 200, 'data': {'status': 'succeeded'}}),
    ]
    with (
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.post',
            new=AsyncMock(side_effect=responses),
        ) as post,
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.get',
            new=AsyncMock(return_value=_response(200, {'status': 'ok'}, content=b'ok')),
        ) as get,
    ):
        runtime = await provider.acquire('main', runtime_kind='macosworld_devcloud')
        await provider.release('main')

    assert runtime.base_url == 'http://10.0.0.9:5000'
    assert [(call.args[0], call.kwargs['json']) for call in post.await_args_list] == [
        (base_url + '/mac/sandboxCreate', {'leaseTime': 300}),
        (base_url + '/mac/sandboxTaskResult', {'taskId': 'create-1'}),
        (base_url + '/mac/sandboxDelete', {'id': '-42'}),
        (base_url + '/mac/sandboxTaskResult', {'taskId': 'delete-1'}),
    ]
    assert runtime.proxy_url == proxy_url
    assert runtime.execute_mode == execute_mode
    assert get.await_args.kwargs['proxy'] == proxy_url
    for call in post.await_args_list:
        assert call.kwargs.get('proxy') == proxy_url
        assert set(call.kwargs['headers']) == {'APPID', 'TIMESTP', 'RANDOM', 'ENCKEY'}


@pytest.mark.asyncio
@pytest.mark.parametrize('proxy_url', [None, 'http://proxy.example:1080'])
async def test_devcloud_proxy_covers_commands_screenshots_and_files(proxy_url: str | None) -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000', proxy_url=proxy_url)
    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
            new=AsyncMock(return_value=_response(200, {'returncode': 0, 'output': 'ok'}, content=b'ok')),
        ) as post,
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.get',
            new=AsyncMock(return_value=_response(200, content=_PNG_BYTES)),
        ) as get,
    ):
        await runtime.call_executor('run_command', {'command': 'echo ok'})
        await runtime.call_executor('gui_screenshot', {})
        await runtime.call_executor('download_file', {'files': [{'path': '/tmp/demo.txt', 'encoding': 'utf8'}]})
        await runtime.call_executor(
            'upload_file',
            {'files': [{'path': '/tmp/demo.txt', 'content': 'ok', 'encoding': 'utf8'}], 'overwrite': False},
        )
    assert [call.args[0] for call in post.await_args_list] == [
        'http://10.0.0.9:5000/execute',
        'http://10.0.0.9:5000/screen_size',
        'http://10.0.0.9:5000/file',
        'http://10.0.0.9:5000/setup/upload',
    ]
    get.assert_awaited_once()
    assert get.await_args.args[0] == 'http://10.0.0.9:5000/screenshot'
    for call in [*post.await_args_list, *get.await_args_list]:
        assert call.kwargs.get('proxy') == proxy_url


@pytest.mark.asyncio
@pytest.mark.parametrize('execute_mode', ['sync', 'async'])
async def test_acquire_reuses_instance_ip_and_release_does_not_call_control_plane(execute_mode: str) -> None:
    provider = _make_provider(app_id=None, app_token=None, execute_mode=execute_mode)
    provider.configure('main', runtime_credential={'instance_ip': '10.0.0.42', 'server_port': 5001})

    with (
        patch('nexus.runtime.providers.devcloud.provider.http_client.post', new=AsyncMock()) as post_mock,
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.get',
            new=AsyncMock(return_value=_response(200, {'status': 'ok'}, content=b'ok')),
        ) as get_mock,
    ):
        runtime = await provider.acquire('main', runtime_kind='macosworld_devcloud')
        await provider.release('main')

    assert runtime.base_url == 'http://10.0.0.42:5001'
    assert runtime.metadata == {'ip': '10.0.0.42', 'reused': True}
    assert runtime.execute_mode == execute_mode
    assert provider.get_runtime_id('main') is None
    assert provider.get_active_runtimes() == {}
    post_mock.assert_not_awaited()
    get_mock.assert_awaited_once()
    assert get_mock.await_args.args[0] == 'http://10.0.0.42:5001/health'


@pytest.mark.asyncio
async def test_gui_pyautogui_uses_call_timeout_for_http_rpc() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://runtime.example', request_timeout=5400)

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
        new=AsyncMock(return_value=_response(200, {'returncode': 0, 'output': '', 'error': ''})),
    ) as post:
        result = await runtime.gui_pyautogui('time.sleep(0.1)', timeout=2)

    assert result.success is True
    timeout = post.await_args.kwargs['timeout']
    assert timeout.total == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_runtime_run_command_uses_osworld_execute() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs):
        captured.append(kwargs['json'])
        return _response(200, {'returncode': 0, 'output': 'hello\n', 'error': ''})

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
    ):
        result = await runtime.call_executor(
            'run_command',
            {
                'command': 'printf hello',
                'cwd': '/tmp/work',
                'timeout': 12,
                'stdin': 'input text',
                'extra_env': {'DEMO_KEY': 'demo-value'},
            },
        )

    assert result['status'] == 'Finished'
    assert result['return_code'] == 0
    assert result['stdout'] == 'hello\n'
    assert captured[0]['command'][0:2] == ['bash', '-lc']
    assert 'timeout --kill-after=5s 12s bash -lc' in captured[0]['command'][2]
    assert 'export DEMO_KEY=demo-value' in captured[0]['command'][2]
    assert 'cd /tmp/work' in captured[0]['command'][2]


@pytest.mark.asyncio
@pytest.mark.parametrize('status_code', [400, 503])
async def test_runtime_run_command_rejects_http_errors_with_json_body(status_code: int) -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
        new=AsyncMock(return_value=_response(status_code, {'returncode': 0, 'output': 'True'})),
    ):
        with pytest.raises(RuntimeError, match=f'HTTP {status_code}'):
            await runtime.call_executor('run_command', {'command': 'echo True'})


@pytest.mark.asyncio
async def test_runtime_gui_screenshot_and_screen_size() -> None:
    import base64

    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    async def fake_post(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/screen_size'
        return _response(200, {'width': 1440, 'height': 900})

    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.get',
            new=AsyncMock(return_value=_response(200, content=_PNG_BYTES)),
        ),
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
        ),
    ):
        screenshot = await runtime.call_executor('gui_screenshot', {})

    assert screenshot['success'] is True
    assert screenshot['format'] == 'png'
    assert base64.b64decode(screenshot['image']) == _PNG_BYTES
    assert screenshot['width'] == 1440
    assert screenshot['height'] == 900


@pytest.mark.asyncio
async def test_runtime_gui_screenshot_retries_http_failure_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    monkeypatch.setattr(runtime, '_GUI_SCREENSHOT_RETRY_BACKOFF_SECONDS', 0.0)

    get_responses = [
        _response(503, text='screenshot backend busy', content=b'screenshot backend busy'),
        _response(200, content=_PNG_BYTES),
    ]

    async def fake_get(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/screenshot'
        return get_responses.pop(0)

    async def fake_post(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/screen_size'
        return _response(200, {'width': 1440, 'height': 900})

    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.get', new=AsyncMock(side_effect=fake_get)
        ) as get_mock,
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
        ) as post_mock,
    ):
        screenshot = await runtime.call_executor('gui_screenshot', {})

    assert screenshot['success'] is True
    assert base64.b64decode(screenshot['image']) == _PNG_BYTES
    assert screenshot['width'] == 1440
    assert screenshot['height'] == 900
    assert get_mock.await_count == 2
    post_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_gui_screenshot_repeated_empty_image_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    monkeypatch.setattr(runtime, '_GUI_SCREENSHOT_RETRY_ATTEMPTS', 2)
    monkeypatch.setattr(runtime, '_GUI_SCREENSHOT_RETRY_BACKOFF_SECONDS', 0.0)

    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.get',
            new=AsyncMock(return_value=_response(200, content=b'')),
        ) as get_mock,
        patch('nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock()) as post_mock,
    ):
        screenshot = await runtime.call_executor('gui_screenshot', {})

    assert screenshot['success'] is False
    assert screenshot['image'] is None
    assert screenshot['format'] == 'png'
    assert 'returned no image bytes' in screenshot['error']
    assert get_mock.await_count == 2
    post_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_runtime_gui_click_uses_pyautogui_execute() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    payloads: list[dict] = []

    async def fake_post(url: str, **kwargs):
        payloads.append(kwargs['json'])
        return _response(200, {'returncode': 0, 'output': '', 'error': ''})

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
    ):
        result = await runtime.call_executor('gui_click', {'x': 10, 'y': 20})

    assert result == {'success': True, 'error': None}
    assert payloads[0]['command'][0:2] == ['python3', '-c']
    assert "pyautogui.click(x=10, y=20, button='left', clicks=1)" in payloads[0]['command'][2]


@pytest.mark.asyncio
async def test_runtime_gui_launch_and_close_app_use_setup_routes() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    urls: list[str] = []

    async def fake_post(url: str, **kwargs):
        urls.append(url)
        if url.endswith('/setup/launch'):
            assert kwargs['json'] == {'command': ['open', '/Applications/Calculator.app'], 'shell': False}
        return _response(200, {'ok': True}, text='')

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
    ):
        launch = await runtime.call_executor('gui_launch_app', {'app': '/Applications/Calculator.app'})
        close = await runtime.call_executor('gui_close_app', {'app': 'Calculator'})

    assert launch == {'success': True, 'error': None}
    assert close == {'success': True, 'error': None}
    assert urls == ['http://10.0.0.9:5000/setup/launch', 'http://10.0.0.9:5000/setup/close_window']


@pytest.mark.asyncio
async def test_runtime_setup_route_uses_launch_response_for_launch_path() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    class DummyLaunchResponse:
        def __init__(self, *, success: bool, error: str | None) -> None:
            self.success = success
            self.error = error

        def model_dump(self) -> dict[str, object]:
            return {'kind': 'launch', 'success': self.success, 'error': self.error}

    class DummyCloseResponse:
        def __init__(self, *, success: bool, error: str | None) -> None:
            self.success = success
            self.error = error

        def model_dump(self) -> dict[str, object]:
            return {'kind': 'close', 'success': self.success, 'error': self.error}

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
            new=AsyncMock(return_value=_response(200)),
        ),
        patch('nexus.runtime.executors.gui.GuiLaunchAppResponse', new=DummyLaunchResponse),
        patch('nexus.runtime.executors.gui.GuiCloseAppResponse', new=DummyCloseResponse),
    ):
        launch = await runtime.call_executor('gui_launch_app', {'app': '/Applications/Calculator.app'})
        close = await runtime.call_executor('gui_close_app', {'app': 'Calculator'})

    assert launch == {'kind': 'launch', 'success': True, 'error': None}
    assert close == {'kind': 'close', 'success': True, 'error': None}


@pytest.mark.asyncio
async def test_runtime_download_file_posts_form_to_file_route() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    async def fake_post(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/file'
        assert kwargs['data'] == {'file_path': '/tmp/demo.txt'}
        return _response(200, text='hello', content=b'hello')

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
    ):
        result = await runtime.call_executor(
            'download_file', {'files': [{'path': '/tmp/demo.txt', 'encoding': 'utf8'}]}
        )

    assert result['files'] == [{'path': '/tmp/demo.txt', 'content': 'hello', 'encoding': 'utf8'}]
    assert result['errors'] is None


@pytest.mark.asyncio
async def test_runtime_download_file_base64_encodes_binary() -> None:
    import base64

    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
        new=AsyncMock(return_value=_response(200, text='ignored', content=b'\x00\x01')),
    ):
        result = await runtime.call_executor(
            'download_file', {'files': [{'path': '/tmp/blob.bin', 'encoding': 'base64'}]}
        )

    assert result['files'][0]['content'] == base64.b64encode(b'\x00\x01').decode('utf-8')


@pytest.mark.asyncio
async def test_runtime_upload_file_posts_multipart_to_setup_upload() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    captured: dict = {}

    async def fake_post(url: str, **kwargs):
        captured['url'] = url
        captured['data'] = kwargs['data']
        return _response(200, text='File Uploaded: 5 bytes', content=b'File Uploaded: 5 bytes')

    with patch(
        'nexus.runtime.providers.macosworld_devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)
    ):
        result = await runtime.call_executor(
            'upload_file',
            {'files': [{'path': '/tmp/demo.txt', 'content': 'hello', 'encoding': 'utf8'}], 'overwrite': False},
        )

    assert captured['url'] == 'http://10.0.0.9:5000/setup/upload'
    form = captured['data']
    assert form._fields[0][0]['name'] == 'file_path'
    assert form._fields[0][2] == '/tmp/demo.txt'
    assert form._fields[1][0]['name'] == 'file_data'
    assert form._fields[1][2] == b'hello'
    assert result['results'] == [{'path': '/tmp/demo.txt', 'success': True, 'message': 'File Uploaded: 5 bytes'}]


@pytest.mark.asyncio
async def test_runtime_upload_file_reports_missing_content() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    result = await runtime.call_executor('upload_file', {'files': [{'path': '/tmp/demo.txt', 'encoding': 'utf8'}]})

    assert result['results'][0]['success'] is False
    assert result['results'][0]['message'] == 'Content is required'


@pytest.mark.asyncio
async def test_screenshot_recovers_after_three_503s_with_bounded_requests() -> None:
    from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime as DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://runtime.example', request_timeout=5400)
    replies = [_response(503, text='busy')] * 3 + [_response(200, content=_PNG_BYTES)]
    with (
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.get', new=AsyncMock(side_effect=replies)
        ) as get,
        patch(
            'nexus.runtime.providers.macosworld_devcloud.client.http_client.post',
            new=AsyncMock(
                return_value=_response(200, {'width': 1440, 'height': 900}),
            ),
        ),
        patch('nexus.runtime.providers.macosworld_devcloud.client.asyncio.sleep', new=AsyncMock()) as sleep,
    ):
        result = await runtime.gui_screenshot()
    assert result.success
    assert get.await_count == 4
    assert [call.args[0] for call in sleep.await_args_list] == [1.0, 2.0, 4.0]
    assert all(call.kwargs['timeout'].total == 10 for call in get.await_args_list)
