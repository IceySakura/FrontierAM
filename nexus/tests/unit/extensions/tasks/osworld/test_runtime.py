"""OSWorld runtime regressions."""

from __future__ import annotations

import asyncio
import subprocess
import sys

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

import nexus.extensions.tasks.os_world as os_world

from nexus.core.context import ExecutionContext
from nexus.extensions.tasks.os_world.execution.browser_launch import LOCAL_CDP_ORIGIN_ARG, add_local_cdp_origins
from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime, get_osworld_runtime
from nexus.runtime.providers.ags import AGSInstanceHandle, AGSOSWorldRuntime, AGSRuntimeProvider


def test_chrome_cdp_only_allows_fixed_loopback_origins():
    command = ['google-chrome', '--remote-debugging-port=1337', 'about:blank']
    result = add_local_cdp_origins(command)
    assert result == [*command, LOCAL_CDP_ORIGIN_ARG]
    assert '*' not in result[-1]
    assert 'http://127.0.0.1:9222' in result[-1]
    assert add_local_cdp_origins(result) == result


@pytest.mark.parametrize(
    'command',
    [
        ['xterm'],
        ['google-chrome', 'about:blank'],
        'google-chrome --remote-debugging-port=1337 && echo unchanged',
        ['google-chrome', '--remote-debugging-port=1337', '--remote-allow-origins=http://specific.test'],
    ],
)
def test_unrelated_or_explicit_launch_settings_are_preserved(command):
    assert add_local_cdp_origins(command) == command


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['auto', 'image', 'off'])
@pytest.mark.parametrize('ensure_target', [True, False])
async def test_cdp_endpoint_lookup_never_starts_browser_or_proxy(mode, ensure_target):
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime
    from nexus.runtime.providers.ags import AGSInstanceHandle

    runtime = OSWorldTaskRuntime(
        AGSInstanceHandle(instance_id='test', access_token='token', domain='example.invalid'),
        cdp_proxy_mode=mode,
    )
    runtime.run_command = AsyncMock(side_effect=AssertionError('must not mutate guest'))
    runtime.setup_upload_bytes = AsyncMock(side_effect=AssertionError('must not upload proxy'))
    endpoint, headers = await runtime.get_chromium_cdp_connection(ensure_target=ensure_target)
    assert endpoint == 'https://9222-test.example.invalid'
    assert headers == {'X-Access-Token': 'token'}
    runtime.run_command.assert_not_awaited()
    runtime.setup_upload_bytes.assert_not_awaited()


def test_guest_proxy_injection_mode_is_rejected():
    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime
    from nexus.runtime.providers.ags import AGSInstanceHandle

    with pytest.raises(ValueError, match='cdp_proxy_mode'):
        OSWorldTaskRuntime(
            AGSInstanceHandle(instance_id='test', access_token='token', domain='example.invalid'),
            cdp_proxy_mode='inject',
        )


@pytest.mark.asyncio
async def test_standard_provider_runtime_is_unmodified_by_task_selection(monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime, get_osworld_runtime
    from nexus.runtime.providers.ags import AGSInstanceHandle, AGSOSWorldRuntime, AGSRuntimeProvider

    provider = AGSRuntimeProvider(secret_id='test', secret_key='test')
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', AsyncMock())
    handle = AGSInstanceHandle(instance_id='test', access_token='token', domain='example.invalid')
    params = {'runtime_kind': 'osworld', 'provision_mode': 'control_plane', 'interface_mode': 'nexus'}
    standard = await provider._create_runtime(handle, params, {'ToolType': 'osworld'})
    assert type(standard) is AGSOSWorldRuntime
    initialize = AsyncMock()
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', initialize)
    provider._active_runtimes['main'] = standard
    ctx = ExecutionContext(runtime_provider=provider)
    task = await get_osworld_runtime(ctx, 'main')
    assert isinstance(task, OSWorldTaskRuntime)
    initialize.assert_awaited_once()
    assert task._build_osworld_execute_payload is AGSOSWorldRuntime._build_osworld_execute_payload
    assert not hasattr(standard, 'task_owned_chromium_cdp')


@pytest.mark.asyncio
async def test_official_socat_is_reused_only_for_verified_image_proxy():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.execution.setup import run_setup_action

    action = {
        'type': 'launch',
        'parameters': {
            'command': ['socat', 'tcp-listen:9222,fork', 'tcp:localhost:1337'],
        },
    }
    runtime = SimpleNamespace(
        cdp_proxy_mode='image',
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                return_code=0,
                stdout='/usr/local/bin/osworld-cdp-proxy --listen-port 9222 --upstream-port 1337',
                stderr='',
            )
        ),
        gui_get_screen_size=AsyncMock(return_value=SimpleNamespace(width=1920, height=1080)),
    )
    result = await run_setup_action(runtime, action, user_command_fn=lambda command: command)
    assert result['reused_image_cdp_proxy']
    assert all('nohup socat' not in call.kwargs['command'] for call in runtime.run_command.call_args_list)
    runtime.run_command.return_value = SimpleNamespace(return_code=1, stdout='', stderr='')
    with pytest.raises(RuntimeError, match='refusing duplicate socat'):
        await run_setup_action(runtime, action, user_command_fn=lambda command: command)


@pytest.mark.asyncio
@pytest.mark.parametrize('proxied', [False, True])
async def test_task_runtime_owns_cdp_relay_only_behind_proxy(monkeypatch, proxied):
    from nexus.extensions.tasks.os_world.execution import runtime as runtime_module

    relay = AsyncMock()
    relay.start.return_value = 'ws://127.0.0.1:1234/cdp'
    relays = []

    def relay_factory(endpoint, headers):
        relays.append((endpoint, headers))
        return relay

    monkeypatch.setattr(runtime_module, 'CDPRelay', relay_factory)
    monkeypatch.setattr(runtime_module, 'websocket_proxy', lambda url: 'http://proxy.invalid:3128' if proxied else None)
    handle = AGSInstanceHandle(instance_id='test', access_token='test-token', domain='example.test')
    runtime = OSWorldTaskRuntime(handle)
    chromium_url = handle.url(runtime.chromium_port)
    if proxied:
        assert [await runtime.chromium_cdp_url() for _ in range(2)] == ['ws://127.0.0.1:1234/cdp'] * 2
        assert relays == [(chromium_url, handle.headers)]
    else:
        assert await runtime.chromium_cdp_url() == chromium_url
        assert relays == []
    await runtime.close()
    assert relay.close.await_count == (1 if proxied else 0)


@pytest.mark.asyncio
async def test_connect_chromium_uses_runtime_cdp_url_and_closes_in_order(monkeypatch):
    import playwright.async_api

    from nexus.extensions.tasks.os_world.execution.cdp import connect_chromium

    events = []
    browser = AsyncMock()
    browser.close.side_effect = lambda: events.append('browser')
    driver = AsyncMock()
    driver.chromium.connect_over_cdp.return_value = browser
    driver.stop.side_effect = lambda: events.append('driver')
    starter = AsyncMock()
    starter.start.return_value = driver
    monkeypatch.setattr(playwright.async_api, 'async_playwright', lambda: starter)
    runtime = SimpleNamespace(
        chromium_cdp_url=AsyncMock(return_value='ws://127.0.0.1:1234/cdp'),
        handle=SimpleNamespace(headers={'X-Access-Token': 'test'}),
    )

    async with connect_chromium(runtime) as connected:
        assert connected is browser

    driver.chromium.connect_over_cdp.assert_awaited_once_with(
        'ws://127.0.0.1:1234/cdp', headers={'X-Access-Token': 'test'}, timeout=15000
    )
    assert events == ['browser', 'driver']


@pytest.mark.asyncio
@pytest.mark.parametrize('capacity', [64 * 1024**2, 512 * 1024**2, 1024**3])
async def test_shm_initialization_expands_only_small_mounts(capacity):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime
    from nexus.runtime.providers.ags import AGSInstanceHandle

    runtime = OSWorldTaskRuntime(AGSInstanceHandle(instance_id='test', access_token='test', domain='test'))
    responses = [SimpleNamespace(return_code=0, stdout=str(capacity))]
    if capacity < 512 * 1024**2:
        responses.extend(
            [
                SimpleNamespace(return_code=0, stdout=''),
                SimpleNamespace(return_code=0, stdout=str(512 * 1024**2)),
            ]
        )
    runtime.run_command = AsyncMock(side_effect=responses)
    await runtime.initialize()
    await runtime.initialize()
    assert runtime.run_command.await_count == len(responses)
    if len(responses) == 3:
        call = runtime.run_command.call_args_list[1]
        assert call.kwargs['command'] == "sudo -S -k -p '' /usr/bin/mount -o remount,size=512M /dev/shm"
        assert call.kwargs['stdin'] == 'password\n'
        assert call.kwargs['timeout'] == 30
    assert not hasattr(runtime, 'prepare_chrome_launch')


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['probe', 'remount', 'verify'])
async def test_shm_setup_failure_is_not_silently_ignored(failure):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime
    from nexus.runtime.providers.ags import AGSInstanceHandle

    runtime = OSWorldTaskRuntime(AGSInstanceHandle(instance_id='test', access_token='test', domain='test'))
    responses = [SimpleNamespace(return_code=0, stdout=str(64 * 1024**2))]
    if failure == 'probe':
        responses[0] = SimpleNamespace(return_code=1, stdout='')
    else:
        responses.append(SimpleNamespace(return_code=1 if failure == 'remount' else 0, stdout=''))
        if failure == 'verify':
            responses.append(SimpleNamespace(return_code=0, stdout=str(64 * 1024**2)))
    runtime.run_command = AsyncMock(side_effect=responses)
    with pytest.raises(RuntimeError, match='OSWorld AGS setup'):
        await runtime.initialize()
    assert not getattr(runtime, '_shm_initialized', False)


@pytest.mark.asyncio
async def test_failed_initialization_releases_new_instance(monkeypatch):
    from unittest.mock import AsyncMock

    from nexus.core.context import ExecutionContext
    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime, get_osworld_runtime
    from nexus.runtime.providers.ags import AGSRuntimeProvider

    provider = AGSRuntimeProvider(secret_id='test', secret_key='test')
    provider.configure(
        'main',
        provision_mode='control_plane',
        runtime_kind='osworld',
    )
    tool = {'ToolId': 'test-tool', 'ToolName': 'test', 'ToolType': 'osworld'}
    monkeypatch.setattr(provider, '_ensure_tool', AsyncMock(return_value=(tool, False)))
    monkeypatch.setattr(provider, '_start_instance', AsyncMock(return_value={'InstanceId': 'test-instance'}))
    monkeypatch.setattr(provider, '_acquire_token', AsyncMock(return_value='test-token'))
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', AsyncMock())
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', AsyncMock(side_effect=RuntimeError('remount failed')))
    stop = AsyncMock(return_value={})
    monkeypatch.setattr(provider, '_call_api', stop)
    with pytest.raises(RuntimeError, match='remount failed'):
        await get_osworld_runtime(ExecutionContext(runtime_provider=provider), 'main')
    stop.assert_awaited_once_with('StopSandboxInstance', {'InstanceId': 'test-instance'})
    assert 'main' not in provider._active_runtimes


@pytest.mark.asyncio
async def test_desktop_cache_refresh_targets_user_directory_without_sudo():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.os_world.execution.runtime import OSWorldTaskRuntime
    from nexus.runtime.providers.ags import AGSInstanceHandle

    runtime = OSWorldTaskRuntime(AGSInstanceHandle(instance_id='test', access_token='test', domain='test'))
    runtime.run_command = AsyncMock(return_value=SimpleNamespace(return_code=0, stdout='/home/user'))
    assert await runtime.prepare_setup_command(['update-desktop-database']) == [
        'update-desktop-database',
        '/home/user/.local/share/applications',
    ]
    explicit = ['update-desktop-database', '/usr/share/applications']
    assert await runtime.prepare_setup_command(explicit) == explicit
    assert await runtime.prepare_setup_command('update-desktop-database; echo other') == (
        'update-desktop-database; echo other'
    )
    runtime.run_command.assert_awaited_once()


def context(monkeypatch):
    provider = AGSRuntimeProvider(secret_id='test', secret_key='test', osworld_cdp_proxy_mode='image')
    provider.configure('main', runtime_kind='osworld', tunnel_port=32001, sudo_password='test-password')
    runtime = AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='test-instance', access_token='test', domain='example.invalid'),
        request_timeout=75,
    )
    provider._active_runtimes['main'] = runtime
    provider._instance_ids['main'] = 'test-instance'
    stop = AsyncMock(return_value={})
    monkeypatch.setattr(provider, '_call_api', stop)
    return ExecutionContext(runtime_provider=provider), runtime, stop


@pytest.mark.asyncio
async def test_task_agent_tools_and_provider_share_adapter_and_release(monkeypatch):
    ctx, standard, stop = context(monkeypatch)
    standard.default_venv_id = 'preserved'
    close = AsyncMock()
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', AsyncMock())
    monkeypatch.setattr(OSWorldTaskRuntime, 'close', close)
    adapter = await get_osworld_runtime(ctx, 'main')
    assert adapter is await ctx.get_runtime('main')  # Agent and tools use this same accessor.
    assert adapter is await ctx.runtime_provider.acquire('main')
    assert ctx.runtime_provider.get_active_runtimes()['main'] is adapter
    assert adapter.handle is standard.handle
    assert adapter.port_mappings is standard.port_mappings
    assert adapter.default_venv_id == 'preserved'
    assert adapter.request_timeout == 75
    assert adapter.tunnel_port == 32001
    assert adapter.cdp_proxy_mode == 'image'
    assert adapter._sudo_password == 'test-password'
    assert id(standard) not in ctx._runtime_shared_unsubs
    assert id(adapter) in ctx._runtime_shared_unsubs
    assert await ctx.release_runtime('main')
    close.assert_awaited_once()
    stop.assert_awaited_once_with('StopSandboxInstance', {'InstanceId': 'test-instance'})
    assert not ctx._runtimes
    assert not ctx.runtime_provider.get_active_runtimes()


@pytest.mark.asyncio
async def test_concurrent_adaptation_initializes_once(monkeypatch):
    ctx, _, _ = context(monkeypatch)

    async def initialize(self):
        await asyncio.sleep(0.01)

    initialized = AsyncMock(side_effect=initialize)
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', lambda self: initialized(self))
    results = await asyncio.gather(*(get_osworld_runtime(ctx, 'main') for _ in range(5)))
    assert all(item is results[0] for item in results)
    assert initialized.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_initialization_releases_via_provider(monkeypatch):
    ctx, _, stop = context(monkeypatch)
    entered = asyncio.Event()

    async def initialize(self):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', initialize)
    task = asyncio.create_task(get_osworld_runtime(ctx, 'main'))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stop.assert_awaited_once()
    assert not ctx._runtimes


@pytest.mark.asyncio
async def test_cleanup_error_preserves_initialization_error(monkeypatch):
    ctx, _, _ = context(monkeypatch)
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', AsyncMock(side_effect=RuntimeError('shm failed')))
    cleanup = AsyncMock(side_effect=RuntimeError('stop failed'))
    monkeypatch.setattr(ctx.runtime_provider, 'release', cleanup)
    with pytest.raises(RuntimeError, match='shm failed'):
        await get_osworld_runtime(ctx, 'main')
    cleanup.assert_awaited_once_with('main', force=True)


@pytest.mark.asyncio
async def test_adapted_runtime_preserves_read_retry_and_file_auth(monkeypatch):
    from nexus.extensions.tasks.os_world.execution import runtime as task_runtime
    from nexus.runtime.providers.ags.base import _BaseAGSRuntime

    ctx, _, _ = context(monkeypatch)
    monkeypatch.setattr(OSWorldTaskRuntime, 'initialize', AsyncMock())
    adapter = await get_osworld_runtime(ctx, 'main')
    failed = SimpleNamespace(status_code=502)
    success = SimpleNamespace(status_code=200)
    request = AsyncMock(side_effect=[failed, success])
    monkeypatch.setattr(_BaseAGSRuntime, '_http_request', request)
    monkeypatch.setattr(task_runtime, '_READ_ONLY_RETRY_BASE_DELAY_SECONDS', 0)
    assert await adapter.read_only_request('GET', 'https://test/screenshot', retry_label='test') is success
    assert request.await_count == 2
    assert request.call_args.kwargs['headers']['X-Access-Token'] == 'test'
    response = SimpleNamespace(content=b'file', raise_for_status=lambda: None)
    fetch = AsyncMock(return_value=response)
    monkeypatch.setattr(task_runtime, 'http_client', SimpleNamespace(post=fetch))
    assert await adapter.get_file('/tmp/file') == b'file'
    assert fetch.call_args.kwargs['headers'] == adapter.handle.headers
    assert await adapter.get_chromium_cdp_connection() == (
        adapter.handle.url(9222),
        adapter.handle.headers,
    )


def test_six_yaml_keep_ags_without_redundant_options():
    root = Path(os_world.__file__).resolve().parent / 'configs'
    configs = list(root.glob('osworld_*.yaml'))
    assert len(configs) == 6
    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config['runtime_provider_name'] == 'ags'
        assert 'runtime_class' not in config['runtime_configs']['main']
        assert 'computer_action_trace' not in config['runner']
        assert 'hold_signal_handlers' not in config['runner']


@pytest.mark.parametrize('harness', ['claude_code', 'codex'])
def test_generic_profile_factory_does_not_import_task(tmp_path, harness):
    code = """
import sys
from nexus.extensions.agents.claude_code.native_bridge.adapters.base import (
    build_execution_profile, execution_profile_registry,
)
assert not any(n.startswith('nexus.extensions.tasks.os_world') for n in sys.modules)
@execution_profile_registry.register('test-policy')
class Profile:
    def __init__(self, *args):
        self.args = args
assert build_execution_profile('test-policy', {}, {}, 'prompt').args[-1] == 'prompt'
try:
    build_execution_profile('unknown', {}, {}, '')
except ValueError:
    pass
else:
    raise AssertionError('unknown profile accepted')
"""
    code = code.replace('agents.claude_code.', f'agents.{harness}.')
    code += f'\nimport nexus.extensions.agents.{harness}.native_bridge.adapters.osworld\n'
    code += "assert not any(n.startswith('nexus.extensions.tasks.os_world') for n in sys.modules)\n"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
