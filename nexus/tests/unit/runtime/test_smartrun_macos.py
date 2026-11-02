"""macOS compatibility at the public Runtime and MCP response boundaries."""

import json
import subprocess

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.runtime.providers.smartrun.client import SmartRunRuntime
from nexus.runtime.providers.smartrun.provider import SmartRunRuntimeProvider


@pytest.mark.parametrize('sse', [False, True])
@pytest.mark.parametrize('provider', [False, True])
async def test_mcp_response_formats(sse, provider):
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    body = json.dumps({'result': {'content': [{'type': 'text', 'text': '{"success": true}'}]}})
    response = MagicMock(status_code=200, text=f'data: {body}\n' if sse else body)
    module = 'provider' if provider else 'client'
    with patch(f'nexus.runtime.providers.smartrun.{module}.http_client') as http:
        http.post = AsyncMock(return_value=response)
        if provider:
            result = await SmartRunRuntimeProvider(token='test')._call_mcp('get_session_status', {})
        else:
            result = await runtime._call('get_session_status', {})
    assert result == {'success': True}


@pytest.mark.parametrize('payload', [{'return_code': 1, 'shell_success': False}, {'error': 'session unavailable'}])
async def test_shell_nonzero_exit_is_distinct_from_mcp_failure(payload):
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    body = {'result': {'isError': True, 'content': [{'type': 'text', 'text': json.dumps(payload)}]}}
    with patch('nexus.runtime.providers.smartrun.client.http_client') as http:
        http.post = AsyncMock(return_value=MagicMock(status_code=200, text=json.dumps(body)))
        if 'return_code' in payload:
            assert await runtime._call('shell', {}) == payload
        else:
            with pytest.raises(RuntimeError, match='session unavailable'):
                await runtime._call('shell', {})


async def test_macos_public_keyboard_and_shell():
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    runtime._connected = True
    with patch.object(runtime, '_call', new_callable=AsyncMock) as call:
        call.return_value = {'success': True, 'return_code': 0}
        assert (await runtime.gui_hotkey(['cmd', 'shift', 's'])).success
        assert call.call_args.args[0] == 'shell'
        assert 'pyautogui.hotkey' in call.call_args.args[1]['command']
        assert (await runtime.gui_press_key('Enter')).success
        assert 'pyautogui.press' in call.call_args.args[1]['command']
        call.reset_mock()
        assert not (await runtime.gui_press_key('Home')).success
        with pytest.raises(ValueError, match='at least 1 item'):
            await runtime.gui_hotkey([])
        call.assert_not_called()
        call.return_value = {'success': True, 'return_code': 0}
        assert (await runtime.gui_click(10, 20, button='right')).success
        assert call.call_args.args[0] == 'shell'
        assert (await runtime.gui_right_click(10, 20)).success
        assert call.call_args.args[0] == 'right_click'
        call.return_value = {'shell_success': True, 'return_code': 0, 'stdout': 'Darwin\n'}
        result = await runtime.run_command('uname -s', timeout=30)
        assert result.return_code == 0
        assert result.stdout == 'Darwin\n'
        assert call.call_args.args[1]['timeout'] == 30
        await runtime.run_command('pwd', cwd='/tmp')
        assert call.call_args.args[1]['command'] == 'cd /tmp && pwd'


async def test_recording_worker_result_and_mobile_guard():
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    runtime._connected = True
    from nexus.runtime.executors.models.command import RunCommandResponse

    with patch.object(runtime, 'run_command', new_callable=AsyncMock) as run:
        run.return_value = RunCommandResponse(
            status='Finished',
            return_code=0,
            stdout='NEXUS_RECORDING_RESULT:{"success":true,"output_path":"/tmp/test.mp4"}\n',
        )
        assert (await runtime.gui_start_recording(output_path='/tmp/test.mp4')).success
        run.return_value = RunCommandResponse(
            status='Finished',
            return_code=0,
            stdout='NEXUS_RECORDING_RESULT:{"success":false,"error":"No recording is in progress"}\n',
        )
        result = await runtime.gui_stop_recording()
        assert not result.success
        assert result.error == 'No recording is in progress'
        runtime.platform = 'ios'
        run.reset_mock()
        assert not (await runtime.gui_start_recording()).success
        run.assert_not_called()


async def test_shell_preserves_cwd_and_environment_literals(tmp_path):
    directory = tmp_path / "space and ' quote"
    directory.mkdir()
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform='macos', token='test'
    )
    runtime._connected = True
    value = 'literal $(echo injected); "quoted"'
    with patch.object(runtime, '_call', new_callable=AsyncMock) as call:
        call.return_value = {'shell_success': True, 'return_code': 0}
        await runtime.run_command('printf "%s\\n" "$PWD" "$PROOF"', cwd=str(directory), extra_env={'PROOF': value})
        command = call.call_args.args[1]['command']
    result = subprocess.run(['sh', '-c', command], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(directory), value]
