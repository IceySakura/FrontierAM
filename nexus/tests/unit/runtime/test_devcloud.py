"""DevCloud tests with mocked HTTP; no real resources are created."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _response(status_code: int, body: dict | None = None, text: str | None = None, content: bytes | None = None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text if text is not None else __import__('json').dumps(body or {})
    resp.content = content if content is not None else resp.text.encode('utf-8')
    resp.json = MagicMock(return_value=body or {})
    return resp


@pytest.mark.asyncio
async def test_runtime_run_command_uses_osworld_execute() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    captured: list[dict] = []

    async def fake_post(url: str, **kwargs):
        captured.append(kwargs['json'])
        return _response(200, {'returncode': 0, 'output': 'hello\n', 'error': ''})

    with patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)):
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
async def test_runtime_gui_screenshot_and_screen_size() -> None:
    import base64

    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    async def fake_post(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/screen_size'
        return _response(200, {'width': 1440, 'height': 900})

    with (
        patch(
            'nexus.runtime.providers.devcloud.client.http_client.get',
            new=AsyncMock(return_value=_response(200, content=b'png-bytes')),
        ),
        patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)),
    ):
        screenshot = await runtime.call_executor('gui_screenshot', {})

    assert screenshot['success'] is True
    assert screenshot['format'] == 'png'
    assert base64.b64decode(screenshot['image']) == b'png-bytes'
    assert screenshot['width'] == 1440
    assert screenshot['height'] == 900


@pytest.mark.asyncio
async def test_runtime_gui_click_uses_pyautogui_execute() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    payloads: list[dict] = []

    async def fake_post(url: str, **kwargs):
        payloads.append(kwargs['json'])
        return _response(200, {'returncode': 0, 'output': '', 'error': ''})

    with patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)):
        result = await runtime.call_executor('gui_click', {'x': 10, 'y': 20})

    assert result == {'success': True, 'error': None}
    assert payloads[0]['command'][0:2] == ['python3', '-c']
    assert "pyautogui.click(x=10, y=20, button='left', clicks=1)" in payloads[0]['command'][2]


@pytest.mark.asyncio
async def test_runtime_gui_launch_and_close_app_use_setup_routes() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    urls: list[str] = []

    async def fake_post(url: str, **kwargs):
        urls.append(url)
        if url.endswith('/setup/launch'):
            assert kwargs['json'] == {'command': ['open', '/Applications/Calculator.app'], 'shell': False}
        return _response(200, {'ok': True}, text='')

    with patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)):
        launch = await runtime.call_executor('gui_launch_app', {'app': '/Applications/Calculator.app'})
        close = await runtime.call_executor('gui_close_app', {'app': 'Calculator'})

    assert launch == {'success': True, 'error': None}
    assert close == {'success': True, 'error': None}
    assert urls == ['http://10.0.0.9:5000/setup/launch', 'http://10.0.0.9:5000/setup/close_window']


@pytest.mark.asyncio
async def test_runtime_setup_route_uses_launch_response_for_launch_path() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

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
        patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(return_value=_response(200))),
        patch('nexus.runtime.executors.gui.GuiLaunchAppResponse', new=DummyLaunchResponse),
        patch('nexus.runtime.executors.gui.GuiCloseAppResponse', new=DummyCloseResponse),
    ):
        launch = await runtime.call_executor('gui_launch_app', {'app': '/Applications/Calculator.app'})
        close = await runtime.call_executor('gui_close_app', {'app': 'Calculator'})

    assert launch == {'kind': 'launch', 'success': True, 'error': None}
    assert close == {'kind': 'close', 'success': True, 'error': None}


@pytest.mark.asyncio
async def test_runtime_download_file_posts_form_to_file_route() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    async def fake_post(url: str, **kwargs):
        assert url == 'http://10.0.0.9:5000/file'
        assert kwargs['data'] == {'file_path': '/tmp/demo.txt'}
        return _response(200, text='hello', content=b'hello')

    with patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)):
        result = await runtime.call_executor(
            'download_file', {'files': [{'path': '/tmp/demo.txt', 'encoding': 'utf8'}]}
        )

    assert result['files'] == [{'path': '/tmp/demo.txt', 'content': 'hello', 'encoding': 'utf8'}]
    assert result['errors'] is None


@pytest.mark.asyncio
async def test_runtime_download_file_base64_encodes_binary() -> None:
    import base64

    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')

    with patch(
        'nexus.runtime.providers.devcloud.client.http_client.post',
        new=AsyncMock(return_value=_response(200, text='ignored', content=b'\x00\x01')),
    ):
        result = await runtime.call_executor(
            'download_file', {'files': [{'path': '/tmp/blob.bin', 'encoding': 'base64'}]}
        )

    assert result['files'][0]['content'] == base64.b64encode(b'\x00\x01').decode('utf-8')


@pytest.mark.asyncio
async def test_runtime_upload_file_posts_multipart_to_setup_upload() -> None:
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    captured: dict = {}

    async def fake_post(url: str, **kwargs):
        captured['url'] = url
        captured['data'] = kwargs['data']
        return _response(200, text='File Uploaded: 5 bytes', content=b'File Uploaded: 5 bytes')

    with patch('nexus.runtime.providers.devcloud.client.http_client.post', new=AsyncMock(side_effect=fake_post)):
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
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    runtime = DevCloudRuntime(base_url='http://10.0.0.9:5000')
    result = await runtime.call_executor('upload_file', {'files': [{'path': '/tmp/demo.txt', 'encoding': 'utf8'}]})

    assert result['results'][0]['success'] is False
    assert result['results'][0]['message'] == 'Content is required'
