import shlex
import subprocess
import time

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.runtime.bridge.adapters.smartrun_macos import SmartRunMacOSBridgeAdapter, detached_shell_command
from nexus.runtime.bridge.runtime_kind import NativeBridgeBundleTarget, native_bridge_bundle_target_for_runtime
from nexus.runtime.providers.smartrun.client import SmartRunRuntime


def test_detached_shell_runs_without_console_and_records_pid(tmp_path):
    output = tmp_path / "output with ' quote"
    pid = tmp_path / 'pid'
    command = detached_shell_command(f'printf proof > {shlex.quote(str(output))}', str(pid))
    launched = subprocess.run(['sh', '-c', command], capture_output=True, text=True, check=True)
    assert launched.stdout.strip() == 'LAUNCHED'
    assert int(pid.read_text()) > 0
    deadline = time.monotonic() + 5
    while not output.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert output.read_text() == 'proof'


@pytest.mark.parametrize('platform', ['macos', 'ios', 'android'])
def test_only_macos_supports_native_bridge(platform):
    runtime = SmartRunRuntime(
        device_id='test', session_id='test', screen_size=(1920, 1080), platform=platform, token='test'
    )
    if platform == 'macos':
        assert native_bridge_bundle_target_for_runtime(runtime) == NativeBridgeBundleTarget.MACOS_ARM64_PY311
    else:
        with pytest.raises(TypeError, match='Unsupported native bridge runtime'):
            native_bridge_bundle_target_for_runtime(runtime)


@pytest.mark.parametrize('upload_fails', [False, True])
async def test_upload_always_stops_server_and_closes_proxy(tmp_path, monkeypatch, upload_fails):
    runtime = SimpleNamespace(
        device_id='test',
        session_id='test',
        run_command=AsyncMock(return_value=SimpleNamespace(return_code=0)),
        _call=AsyncMock(
            side_effect=[
                {'success': True, 'token': 'test-token', 'consumer_url': 'https://example.test/proxy/'},
                {'success': True},
            ]
        ),
    )
    upload = AsyncMock(return_value=SimpleNamespace(status=204))
    if upload_fails:
        upload.side_effect = RuntimeError('upload failed')
    monkeypatch.setattr('nexus.runtime.bridge.adapters.smartrun_macos.http_client.put', upload)
    artifact = tmp_path / 'bridge.tar.gz'
    artifact.write_bytes(b'a' * (512 * 1024 + 1))
    adapter = SmartRunMacOSBridgeAdapter(runtime)
    if upload_fails:
        with pytest.raises(RuntimeError, match='upload failed'):
            await adapter._upload_artifact(artifact, '/tmp/bridge.tar.gz')
    else:
        await adapter._upload_artifact(artifact, '/tmp/bridge.tar.gz')
        assert [call.kwargs['headers']['X-Upload-Offset'] for call in upload.call_args_list] == ['0', '524288']
        assert b''.join(call.kwargs['data'] for call in upload.call_args_list) == artifact.read_bytes()
    assert 'kill' in runtime.run_command.call_args.args[0]
    assert runtime._call.call_args.args[0] == 'close_tunnel'
    assert runtime._call.call_args.args[1]['token'] == 'test-token'


async def test_stop_closes_proxy_even_if_process_cleanup_fails(monkeypatch):
    runtime = SimpleNamespace(device_id='test', session_id='test', _call=AsyncMock(return_value={'success': True}))
    adapter = SmartRunMacOSBridgeAdapter(runtime)
    adapter._tunnel = {'token': 'test-token'}
    monkeypatch.setattr(
        'nexus.runtime.bridge.adapters.base.BasePosixBridgeAdapter.stop_bridge',
        AsyncMock(side_effect=RuntimeError('shell unavailable')),
    )
    with pytest.raises(RuntimeError, match='shell unavailable'):
        await adapter.stop_bridge()
    assert runtime._call.call_args.args[0] == 'close_tunnel'
    assert adapter._tunnel is None


async def test_upload_waits_for_listener_without_reverse_dns(tmp_path, monkeypatch):
    import asyncio
    import socket

    from nexus.runtime.bridge.adapters import smartrun_macos

    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    # A reverse lookup must not be required, even with a delayed server startup.
    monkeypatch.setattr(
        smartrun_macos,
        '_UPLOAD_SERVER',
        'import socket, time\n'
        'def forbidden_lookup(*args):\n'
        '    raise RuntimeError("Unexpected reverse DNS lookup")\n'
        'socket.getfqdn = forbidden_lookup\n'
        'time.sleep(0.2)\n' + smartrun_macos._UPLOAD_SERVER,
    )

    async def run_command(command, **kwargs):
        if command == smartrun_macos._PREPARE_PYTHON:
            return SimpleNamespace(return_code=0)
        result = await asyncio.to_thread(
            subprocess.run, ['sh', '-c', command], capture_output=True, text=True, timeout=45
        )
        return SimpleNamespace(return_code=result.returncode, stdout=result.stdout, stderr=result.stderr)

    runtime = SimpleNamespace(run_command=run_command)
    adapter = SmartRunMacOSBridgeAdapter(runtime)
    adapter.bridge_home = str(tmp_path / 'bridge')
    adapter._upload_port = port
    adapter._open_tunnel = AsyncMock(return_value={'consumer_url': f'http://127.0.0.1:{port}/'})
    adapter._close_tunnel = AsyncMock()
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.write_bytes(b'proof' * 150000)
    await adapter._upload_artifact(source, str(target))
    assert target.read_bytes() == source.read_bytes()
    assert not (tmp_path / 'bridge.upload.pid').exists()
    adapter._close_tunnel.assert_awaited_once()


async def test_upload_start_failure_cleans_up_without_opening_proxy(tmp_path):
    runtime = SimpleNamespace(
        run_command=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0),
                SimpleNamespace(return_code=1, stderr='Upload server exited before becoming ready'),
                SimpleNamespace(return_code=0),
            ]
        ),
        _call=AsyncMock(),
    )
    adapter = SmartRunMacOSBridgeAdapter(runtime)
    with pytest.raises(RuntimeError, match='exited before becoming ready'):
        await adapter._upload_artifact(tmp_path / 'unused', '/tmp/unused')
    runtime._call.assert_not_awaited()
    assert 'kill' in runtime.run_command.call_args.args[0]
