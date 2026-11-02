import base64

import aiohttp
import pytest

from nexus.runtime.providers.ags import (
    AGSInstanceHandle,
    AGSOSWorldRuntime,
    AGSRawCodeInterpreterRuntime,
    AGSRawOSWorldRuntime,
    AGSRawRuntime,
)
from nexus.runtime.providers.ags.base import _extract_tagged_json
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, AGSWindowsAgentArenaRuntime
from nexus.utils.async_http import HTTPResponse


def test_extract_tagged_json_from_ags_code_interpreter_event_stream() -> None:
    payload = (
        '{"type": "number_of_executions", "execution_count": 3}\n'
        '{"type": "stdout", "text": "NEXUS_JSON_RESULT_START\\n'
        '{\\"status\\": \\"Finished\\", \\"return_code\\": 0, \\"stdout\\": \\"42\\\\n\\", \\"stderr\\": \\"\\"}'
        '\\nNEXUS_JSON_RESULT_END\\n", "timestamp": "2026-03-16T21:47:36.345321+00:00"}\n'
        '{"type": "end_of_execution"}\n'
    )

    result = _extract_tagged_json(payload)

    assert result == {
        'status': 'Finished',
        'return_code': 0,
        'stdout': '42\n',
        'stderr': '',
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(('provider_timeout', 'expected'), [(2400, 2400), (0, None)])
async def test_ags_osworld_unbounded_command_keeps_provider_http_budget(monkeypatch, provider_timeout, expected):
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='sandbox-123', access_token='token', domain='example.invalid'),
        request_timeout=provider_timeout,
    )

    async def post(path, payload, *, timeout=None):
        assert timeout == expected
        assert 'timeout --kill-after' not in str(payload)
        return {'returncode': 0, 'output': 'ok', 'error': ''}, '', 200

    monkeypatch.setattr(runtime, '_post_osworld_json', post)
    response = await runtime.run_command('echo ok', timeout=None)
    assert response.return_code == 0
    assert response.stdout == 'ok'


@pytest.mark.asyncio
async def test_ags_osworld_run_command_uses_execute_endpoint() -> None:
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    captured: list[tuple[str, dict[str, object]]] = []

    async def _fake_post(path: str, payload: dict[str, object], *, timeout=None) -> tuple[dict[str, object], str, int]:
        assert timeout == 12
        captured.append((path, payload))
        return (
            {
                'returncode': 0,
                'output': 'hello\n',
                'error': '',
            },
            '',
            200,
        )

    runtime._post_osworld_json = _fake_post  # type: ignore[method-assign]

    response = await runtime.call_executor(
        'run_command',
        {
            'command': 'printf hello',
            'cwd': '/workspace/demo',
            'timeout': 12,
            'stdin': 'ignored-stdin',
            'extra_env': {'DEMO_KEY': 'demo-value'},
        },
    )

    assert response['status'] == 'Finished'
    assert response['return_code'] == 0
    assert response['stdout'] == 'hello\n'
    assert captured[0][0] == '/execute'
    assert captured[0][1]['command'][0:2] == ['bash', '-lc']
    assert 'timeout --kill-after=5s 12s bash -lc' in captured[0][1]['command'][2]
    assert 'export DEMO_KEY=demo-value' in captured[0][1]['command'][2]
    assert 'cd /workspace/demo' in captured[0][1]['command'][2]
    assert 'cat <<' in captured[0][1]['command'][2]
    assert '| bash -lc' in captured[0][1]['command'][2]
    assert '\nignored-stdin\n__NEXUS_STDIN__' in captured[0][1]['command'][2]


@pytest.mark.asyncio
async def test_ags_raw_request_executor_serializes_http_response() -> None:
    runtime = AGSRawRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-raw-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        ),
        kind='custom',
        default_port=5000,
    )

    async def _fake_request(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> HTTPResponse:
        assert method == 'POST'
        assert path == '/info'
        assert port == 5001
        assert kwargs['json'] == {'hello': 'world'}
        return HTTPResponse(
            status=200,
            headers={'Content-Type': 'application/json'},
            text='{"ok": true}',
            content=b'{"ok": true}',
            url='https://5001-sandbox-raw-123.ap-guangzhou.tencentags.com/info',
            encoding='utf-8',
        )

    runtime.request = _fake_request  # type: ignore[method-assign]

    result = await runtime.call_executor(
        'raw_request',
        {
            'method': 'POST',
            'path': '/info',
            'port': 5001,
            'json': {'hello': 'world'},
            'include_content': True,
        },
    )

    assert result['status_code'] == 200
    assert result['json'] == {'ok': True}
    assert result['content_base64'] == base64.b64encode(b'{"ok": true}').decode('ascii')


@pytest.mark.asyncio
async def test_ags_waa_run_command_uses_command_timeout_as_request_timeout() -> None:
    runtime = AGSWindowsAgentArenaRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-waa-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    captured: list[tuple[str, str, dict[str, object]]] = []

    async def _fake_http_request(method: str, url: str, **kwargs: object) -> HTTPResponse:
        captured.append((method, url, kwargs))
        return HTTPResponse(
            status=200,
            headers={'Content-Type': 'application/json'},
            text='{"returncode":0,"output":"hello\\n","error":""}',
            content=b'{"returncode":0,"output":"hello\\n","error":""}',
            url=url,
            encoding='utf-8',
        )

    runtime._http_request = _fake_http_request  # type: ignore[method-assign]

    response = await runtime.run_command(command='echo hello', timeout=12)

    assert response.return_code == 0
    assert response.stdout == 'hello\n'
    assert captured
    method, url, kwargs = captured[0]
    assert method == 'POST'
    assert url == 'https://5000-sandbox-waa-123.ap-guangzhou.tencentags.com/execute'
    assert kwargs['json'] == {'command': 'echo hello', 'shell': True}
    assert isinstance(kwargs['timeout'], aiohttp.ClientTimeout)
    assert kwargs['timeout'].total == 12


@pytest.mark.asyncio
async def test_ags_raw_code_interpreter_execute_preserves_native_payload() -> None:
    runtime = AGSRawCodeInterpreterRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-raw-code-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    captured: list[tuple[str, str, int | None, dict[str, object]]] = []

    async def _fake_request(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> HTTPResponse:
        captured.append((method, path, port, dict(kwargs)))
        return HTTPResponse(
            status=200,
            headers={'Content-Type': 'application/json'},
            text='{"events": [{"type": "stdout", "text": "42\\n"}]}',
            content=b'{"events": [{"type": "stdout", "text": "42\\n"}]}',
            url='https://49999-sandbox-raw-code-123.ap-guangzhou.tencentags.com/execute',
            encoding='utf-8',
        )

    runtime.request = _fake_request  # type: ignore[method-assign]

    result = await runtime.execute('print(42)', language='python')

    assert result == {'events': [{'type': 'stdout', 'text': '42\n'}]}
    assert captured == [
        (
            'POST',
            '/execute',
            49999,
            {'json': {'code': 'print(42)', 'language': 'python'}},
        )
    ]


@pytest.mark.asyncio
async def test_ags_raw_osworld_runtime_preserves_original_endpoints() -> None:
    runtime = AGSRawOSWorldRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-raw-osworld-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    json_calls: list[tuple[str, str, int | None, dict[str, object]]] = []
    bytes_calls: list[tuple[str, str, int | None, dict[str, object]]] = []

    async def _fake_request_json(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> object:
        json_calls.append((method, path, port, dict(kwargs)))
        if path == '/accessibility':
            return {'AT': 'tree'}
        if path == '/screen_size':
            return {'width': 1920, 'height': 1080}
        if path == '/execute':
            return {'returncode': 0, 'output': 'ok', 'error': ''}
        if path == '/run_python':
            return {'status': 'ok'}
        if path == '/run_bash_script':
            return {'status': 'success', 'returncode': 0}
        if path == '/terminal':
            return {'output': 'prompt'}
        raise AssertionError(f'unexpected path: {path}')

    async def _fake_request_bytes(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> bytes:
        bytes_calls.append((method, path, port, dict(kwargs)))
        if path == '/screenshot':
            return b'png'
        if path == '/file':
            return b'file-bytes'
        raise AssertionError(f'unexpected path: {path}')

    request_calls: list[tuple[str, str, int | None, dict[str, object]]] = []

    async def _fake_request(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> HTTPResponse:
        request_calls.append((method, path, port, dict(kwargs)))
        # /setup/* 成功时返回空 body —— 这正是要验证的：setup_* 不能对空 body 调 .json()
        return HTTPResponse(status=200, headers={}, text='', content=b'', url=path, encoding='utf-8')

    runtime.request_json = _fake_request_json  # type: ignore[method-assign]
    runtime.request_bytes = _fake_request_bytes  # type: ignore[method-assign]
    runtime.request = _fake_request  # type: ignore[method-assign]

    assert await runtime.screenshot() == b'png'
    assert await runtime.accessibility_tree() == {'AT': 'tree'}
    assert await runtime.terminal_output() == {'output': 'prompt'}
    assert await runtime.get_file('/tmp/demo.txt') == b'file-bytes'
    assert await runtime.screen_size() == {'width': 1920, 'height': 1080}
    assert await runtime.execute(['bash', '-lc', 'pwd'], shell=False) == {'returncode': 0, 'output': 'ok', 'error': ''}
    assert await runtime.run_python('print(1)') == {'status': 'ok'}
    assert await runtime.run_bash_script('echo ok', timeout=20, working_dir='/tmp') == {
        'status': 'success',
        'returncode': 0,
    }
    # setup_* 返回原始 HTTPResponse（空 body 不报错），调用方自行决定如何处理
    open_resp = await runtime.setup_open_file('/tmp/demo.txt')
    close_resp = await runtime.setup_close_window('Firefox')
    assert isinstance(open_resp, HTTPResponse)
    assert open_resp.status_code == 200
    assert open_resp.text == ''
    assert isinstance(close_resp, HTTPResponse)
    assert close_resp.status_code == 200

    assert bytes_calls == [
        ('GET', '/screenshot', 5000, {}),
        ('POST', '/file', 5000, {'data': {'file_path': '/tmp/demo.txt'}}),
    ]
    assert json_calls == [
        ('GET', '/accessibility', 5000, {}),
        ('GET', '/terminal', 5000, {}),
        ('POST', '/screen_size', 5000, {'json': {}}),
        ('POST', '/execute', 5000, {'json': {'command': ['bash', '-lc', 'pwd'], 'shell': False}}),
        ('POST', '/run_python', 5000, {'json': {'code': 'print(1)'}}),
        ('POST', '/run_bash_script', 5000, {'json': {'script': 'echo ok', 'timeout': 20, 'working_dir': '/tmp'}}),
    ]
    assert request_calls == [
        ('POST', '/setup/open_file', 5000, {'json': {'path': '/tmp/demo.txt'}}),
        ('POST', '/setup/close_window', 5000, {'json': {'window_name': 'Firefox'}}),
    ]


@pytest.mark.asyncio
async def test_ags_raw_waa_setup_open_file_retries_transient_504(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = AGSRawWindowsAgentArenaRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-raw-waa-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )
    monkeypatch.setattr('nexus.runtime.providers.ags.waa._SETUP_OPEN_FILE_RETRY_DELAYS', (0.0, 0.0))

    calls: list[tuple[str, str]] = []

    async def _fake_request(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> HTTPResponse:
        calls.append((method, path))
        if path == '/setup/open_file' and calls.count(('POST', '/setup/open_file')) < 3:
            return HTTPResponse(
                status=504,
                headers={},
                text='{"error":"open_file_timeout","message":"GUI open helper failed"}',
                content=b'',
                url='https://5000-sandbox-raw-waa-123.ap-guangzhou.tencentags.com/setup/open_file',
                encoding='utf-8',
            )
        return HTTPResponse(
            status=200,
            headers={},
            text='opened',
            content=b'opened',
            url='https://5000-sandbox-raw-waa-123.ap-guangzhou.tencentags.com/setup/open_file',
            encoding='utf-8',
        )

    runtime.request = _fake_request  # type: ignore[method-assign]

    assert await runtime.setup_open_file('C:\\Users\\Docker\\Downloads\\demo.xlsx') == 'opened'
    assert calls == [
        ('POST', '/setup/open_file'),
        ('POST', '/setup/close_all'),
        ('POST', '/setup/open_file'),
        ('POST', '/setup/close_all'),
        ('POST', '/setup/open_file'),
    ]


@pytest.mark.asyncio
async def test_ags_raw_waa_setup_open_file_does_not_retry_non_transient_error() -> None:
    runtime = AGSRawWindowsAgentArenaRuntime(
        AGSInstanceHandle(
            instance_id='sandbox-raw-waa-123',
            access_token='token',
            domain='ap-guangzhou.tencentags.com',
        )
    )

    calls: list[tuple[str, str]] = []

    async def _fake_request(method: str, path: str = '', *, port: int | None = None, **kwargs: object) -> HTTPResponse:
        calls.append((method, path))
        return HTTPResponse(
            status=404,
            headers={},
            text='File not found: C:\\Users\\Docker\\Documents\\missing.txt',
            content=b'',
            url='https://5000-sandbox-raw-waa-123.ap-guangzhou.tencentags.com/setup/open_file',
            encoding='utf-8',
        )

    runtime.request = _fake_request  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='HTTP 404'):
        await runtime.setup_open_file('C:\\Users\\Docker\\Documents\\missing.txt')
    assert calls == [('POST', '/setup/open_file')]
