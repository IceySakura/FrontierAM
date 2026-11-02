import asyncio

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from nexus.core.context import ExecutionContext
from nexus.runtime import InvalidPersistentRuntimeError
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.providers.agrt import AGRTRuntimeProvider
from nexus.runtime.providers.agrt import provider as provider_module
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, AGSWindowsAgentArenaRuntime


@pytest.fixture
def provider(monkeypatch):
    provider = AGRTRuntimeProvider(api_key='test-key', template='win-test', poll_interval=0)
    api = AsyncMock(
        side_effect=[
            {'instance_id': 'inst-test', 'status': 'PENDING'},
            {'status': 'PENDING'},
            {'status': 'RUNNING'},
            {'token': 'test-instance-token'},
        ]
    )
    monkeypatch.setattr(provider, '_api', api)
    monkeypatch.setattr(
        provider_module.http_client, 'get', AsyncMock(return_value=Mock(status_code=200, content=b'ok'))
    )
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['raw', 'nexus'])
async def test_acquire_reuses_instance_and_runtime_overrides(provider, mode):
    provider.configure('main', template='configured', timeout=120, interface_mode=mode, request_timeout=15)
    runtime, same = await asyncio.gather(provider.acquire('main', template='override'), provider.acquire('main'))
    assert runtime is same
    expected_type = AGSRawWindowsAgentArenaRuntime if mode == 'raw' else AGSWindowsAgentArenaRuntime
    assert isinstance(runtime, expected_type)
    assert runtime.request_timeout == 15
    assert provider.get_runtime_id('main') == 'inst-test'
    assert runtime.handle.url(5000, '/probe') == 'https://5000-inst-test.agrt.nano.woa.com/probe'
    assert runtime.handle.headers == {'X-Access-Token': 'test-instance-token'}
    assert provider._api.await_args_list[0].kwargs['payload'] == {'template': 'override', 'timeout': '120s'}
    provider._api.side_effect = None
    provider._api.return_value = {}
    await provider.release_all()
    provider._api.assert_awaited_with('POST', '/v1/instances/inst-test/stop', api_key='test-key')
    assert not provider.get_active_runtimes()
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [RuntimeError('service failed'), asyncio.CancelledError()])
async def test_failed_or_cancelled_acquire_stops_instance(provider, failure):
    provider._wait_for_service_ready = AsyncMock(side_effect=failure)
    provider._api.side_effect = [
        {'instance_id': 'inst-test'},
        {'status': 'RUNNING'},
        {'token': 'test-token'},
        {},
    ]
    with pytest.raises(type(failure)):
        await provider.acquire('main')
    provider._api.assert_awaited_with('POST', '/v1/instances/inst-test/stop', api_key='test-key')
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_failed_cleanup_remains_retryable_after_acquire_failure(provider):
    provider._api.side_effect = [
        {'instance_id': 'inst-test'},
        {'status': 'FAILED'},
        RuntimeError('stop unavailable'),
    ]
    with pytest.raises(RuntimeError, match='FAILED'):
        await provider.acquire('main')
    assert provider.get_runtime_id('main') == 'inst-test'
    with pytest.raises(RuntimeError, match='pending cleanup'):
        await provider.acquire('main')
    provider._api.side_effect = None
    provider._api.return_value = {}
    await provider.release_all()
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_failed_stop_is_reported_and_can_be_retried(provider):
    await provider.acquire('main')
    provider._api.side_effect = RuntimeError('stop unavailable')
    with pytest.raises(RuntimeError, match='stop unavailable'):
        await provider.release('main')
    assert provider.get_runtime_id('main') == 'inst-test'
    provider._api.side_effect = None
    provider._api.return_value = {}
    await provider.release('main')
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_close_failure_still_stops_instance(provider):
    runtime = await provider.acquire('main')
    runtime.close = AsyncMock(side_effect=RuntimeError('close failed'))
    provider._api.side_effect = None
    provider._api.return_value = {}
    with pytest.raises(RuntimeError, match='close failed'):
        await provider.release('main')
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_renew_and_token_refresh_use_per_runtime_key(provider):
    runtime = await provider.acquire('main', api_key='runtime-key')
    provider._api.side_effect = [{'expires_at': 'later'}, {'token': 'refreshed-token'}]
    assert await provider.renew('main', 300) == {'expires_at': 'later'}
    assert provider._api.await_args.kwargs == {'api_key': 'runtime-key', 'payload': {'timeout': '300s'}}
    raw = runtime.to_raw()
    await provider.refresh_token('main', 600)
    assert runtime.handle.headers == raw.handle.headers == {'X-Access-Token': 'refreshed-token'}
    assert provider._api.await_args.kwargs == {'api_key': 'runtime-key', 'payload': {'ttl_seconds': 600}}


@pytest.mark.asyncio
async def test_running_timeout_recycles_instance(provider):
    provider.instance_ready_timeout = 0
    provider._api.side_effect = [{'instance_id': 'inst-test'}, {'status': 'PENDING'}, {}]
    with pytest.raises(TimeoutError, match='not RUNNING'):
        await provider.acquire('main')
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [{'api_key': ''}, {'template': ''}, {'interface_mode': 'invalid'}, {'os': 'macos'}],
)
async def test_invalid_config_does_not_create_instance(params, monkeypatch):
    config: dict[str, Any] = {'api_key': 'key', 'template': 'win', **params}
    provider = AGRTRuntimeProvider(**config)
    api = AsyncMock()
    monkeypatch.setattr(provider, '_api', api)
    with pytest.raises(ValueError, match='requires|Unsupported'):
        await provider.acquire('main')
    api.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_plane_uses_bearer_header_and_surfaces_http_errors(monkeypatch):
    request = AsyncMock(return_value=Mock(status_code=403, text='key forbidden'))
    monkeypatch.setattr(provider_module.http_client, 'request', request)
    provider = AGRTRuntimeProvider(domain='agrt.example.test')
    with pytest.raises(RuntimeError, match='HTTP 403 key forbidden'):
        await provider._api('POST', '/v1/instances', api_key='key', payload={'template': 'win'})
    assert request.await_args is not None
    assert request.await_args.args == ('POST', 'https://agrt.example.test/v1/instances')
    assert request.await_args.kwargs['headers'] == {'Authorization': 'Bearer key'}


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['raw', 'nexus'])
async def test_runtime_credential_attaches_without_template_and_release_only_detaches(provider, mode):
    provider.template = None
    provider.configure('main', runtime_credential={'instance_id': 'inst-existing'}, interface_mode=mode)
    provider._api.side_effect = [{'status': 'RUNNING'}, {'token': 'existing-token'}]

    runtime, same = await asyncio.gather(provider.acquire('main'), provider.acquire('main'))

    assert runtime is same
    assert runtime.handle.instance_id == 'inst-existing'
    assert runtime.handle.access_token == 'existing-token'
    assert provider.get_runtime_id('main') == 'inst-existing'
    assert [(c.args[0], c.args[1]) for c in provider._api.await_args_list] == [
        ('GET', '/v1/instances/inst-existing'),
        ('POST', '/v1/instances/inst-existing/token'),
    ]
    await provider.release_all()
    await provider.release('main')
    assert provider._api.await_count == 2
    assert not provider.get_active_runtimes()
    assert not provider._instance_ids
    assert not provider._api_keys
    assert not provider._external_instance_names


@pytest.mark.asyncio
async def test_execution_context_passes_pool_credential_to_agrt_and_releases_lease(provider):
    provider.template = None
    provider._api.side_effect = [{'status': 'RUNNING'}, {'token': 'existing-token'}]
    account = Mock(credentials={'instance_id': 'inst-existing'}, release=AsyncMock(), invalidate=AsyncMock())
    manager = Mock(
        acquire=AsyncMock(return_value=account), release_all=AsyncMock(side_effect=account.release), aclose=AsyncMock()
    )
    context = ExecutionContext(
        runtime_provider=provider,
        runtime_configs={'main': {'runtime_pool': {'name': 'agrt-windows'}}},
    )
    context._account_pool_manager = manager
    async with context:
        runtime = await context.get_runtime()
        assert runtime.handle.instance_id == 'inst-existing'
    account.release.assert_awaited_once()
    account.invalidate.assert_not_awaited()
    assert provider._api.await_count == 2
    assert not provider._instance_ids


@pytest.mark.asyncio
@pytest.mark.parametrize('credential', [{}, {'instance_id': ' '}, 'inst-existing'])
async def test_invalid_runtime_credential_does_not_create_instance(provider, credential):
    with pytest.raises(InvalidPersistentRuntimeError, match='runtime_credential'):
        await provider.acquire('main', runtime_credential=credential)
    provider._api.assert_not_awaited()


@pytest.mark.asyncio
async def test_top_level_instance_id_is_rejected(provider):
    with pytest.raises(ValueError, match='use runtime_credential.instance_id'):
        await provider.acquire('main', instance_id='inst-existing')
    provider._api.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['STOPPED', 'EXPIRED', 'PENDING'])
async def test_non_running_persistent_instance_is_invalid_without_lifecycle_changes(provider, status):
    provider._api.side_effect = [{'status': status}]
    with pytest.raises(InvalidPersistentRuntimeError, match=f'not running: status={status}'):
        await provider.acquire('main', runtime_credential={'instance_id': 'inst-existing'})
    provider._api.assert_awaited_once_with('GET', '/v1/instances/inst-existing', api_key='test-key')
    assert not provider._instance_ids


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [404, 403, 500])
async def test_only_missing_instance_http_error_invalidates_credential(monkeypatch, status):
    request = AsyncMock(return_value=Mock(status_code=status, text='unavailable'))
    monkeypatch.setattr(provider_module.http_client, 'request', request)
    provider = AGRTRuntimeProvider(api_key='test-key')
    with pytest.raises(RuntimeError) as exc:
        await provider.acquire('main', runtime_credential={'instance_id': 'inst-existing'})
    assert isinstance(exc.value, InvalidPersistentRuntimeError) == (status == 404)
    assert request.await_count == 1
    assert not provider._instance_ids


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['token', 'probe'])
@pytest.mark.parametrize('failure', [TimeoutError('temporarily unavailable'), asyncio.CancelledError()])
async def test_persistent_acquire_failure_or_cancellation_never_stops_instance(provider, stage, failure):
    provider._api.side_effect = [{'status': 'RUNNING'}, failure if stage == 'token' else {'token': 'token'}]
    if stage == 'probe':
        provider._wait_for_service_ready = AsyncMock(side_effect=failure)
    with pytest.raises(type(failure)):
        await provider.acquire('main', runtime_credential={'instance_id': 'inst-existing'})
    await provider.release_all()
    assert provider._api.await_count == 2
    assert not provider._instance_ids
    assert not provider._api_keys
    assert not provider._external_instance_names


@pytest.mark.asyncio
async def test_persistent_close_failure_still_detaches_without_stopping_instance(provider):
    provider._api.side_effect = [{'status': 'RUNNING'}, {'token': 'token'}]
    runtime = await provider.acquire('main', runtime_credential={'instance_id': 'inst-existing'})
    runtime.close = AsyncMock(side_effect=RuntimeError('close failed'))
    with pytest.raises(RuntimeError, match='close failed'):
        await provider.release('main')
    assert provider._api.await_count == 2
    assert not provider._instance_ids
    assert not provider._external_instance_names


@pytest.mark.asyncio
async def test_existing_nexus_server_uses_health_rpc_auth_and_detaches(provider):
    provider.template = None
    provider._api.side_effect = [{'status': 'RUNNING'}, {'token': 'token'}, {'token': 'refreshed'}]
    runtime = await provider.acquire(
        'main', runtime_credential={'instance_id': 'inst-existing'}, launch_mode='nexus_server', runtime_port=18888
    )
    assert isinstance(runtime, RemoteRuntime)
    assert runtime.base_url == 'https://18888-inst-existing.agrt.nano.woa.com'
    assert runtime.config.default_headers == {'X-Access-Token': 'token'}
    assert provider_module.http_client.get.await_args.args == (runtime.base_url + '/health',)
    await provider.refresh_token('main')
    assert runtime.config.default_headers == {'X-Access-Token': 'refreshed'}
    runtime.ws_disconnect = AsyncMock()
    await provider.release('main')
    runtime.ws_disconnect.assert_awaited_once()
    assert provider._api.await_count == 3
    assert not provider.get_active_runtimes()
    assert not provider._instance_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params', [{'launch_mode': 'invalid'}, {'launch_mode': 'nexus_server', 'interface_mode': 'raw'}]
)
async def test_invalid_launch_mode_does_not_create_instance(provider, params):
    with pytest.raises(ValueError, match='launch_mode'):
        await provider.acquire('main', **params)
    provider._api.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_merges_environment(provider):
    provider.environment = {'REGION': 'default', 'KEEP': 'yes'}
    await provider.acquire('main', environment={'REGION': 'cn-north'})
    assert provider._api.await_args_list[0].kwargs['payload']['env'] == {
        'REGION': 'cn-north',
        'KEEP': 'yes',
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'params',
    [
        {'worker_venv': True},
        {'worker_venv': {}},
        {'runtime_credential': {'instance_id': 'existing'}, 'environment': {'REGION': 'cn'}},
    ],
)
async def test_invalid_bootstrap_options_fail_before_create(provider, params):
    with pytest.raises(ValueError, match='AGRT'):
        await provider.acquire('main', **params)
    provider._api.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_venv_uses_execution_context_like_gongfeng(provider, monkeypatch):
    provider.launch_mode = 'nexus_server'
    cfg = {'worker_venv': True}
    provider.configure('main', **cfg)
    setup = AsyncMock(side_effect=lambda runtime, config: runtime)
    monkeypatch.setattr(ExecutionContext, '_setup_worker_venv', setup)
    async with ExecutionContext(runtime_provider=provider, runtime_configs={'main': cfg}) as ctx:
        runtime = await ctx.get_runtime()
        assert isinstance(runtime, RemoteRuntime)
        setup.assert_awaited_once_with(runtime, cfg['worker_venv'])
        runtime.drain_trace = AsyncMock()
        runtime.ws_disconnect = AsyncMock()
        provider._api.side_effect = None
        provider._api.return_value = {}


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [{}, {'python_version': '3.12'}, {'modules': ['claude_code']}, 'true', 1, 0, None])
async def test_worker_venv_rejects_non_boolean_before_provisioning(provider, value):
    with pytest.raises(ValueError, match='worker_venv only accepts true or false'):
        await provider.acquire('main', launch_mode='nexus_server', worker_venv=value)
    provider._api.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_worker_venv_skips_setup(provider, monkeypatch):
    provider.launch_mode = 'nexus_server'
    cfg = {'worker_venv': False}
    provider.configure('main', **cfg)
    setup = AsyncMock()
    monkeypatch.setattr(ExecutionContext, '_setup_worker_venv', setup)
    async with ExecutionContext(runtime_provider=provider, runtime_configs={'main': cfg}) as ctx:
        runtime = await ctx.get_runtime()
        setup.assert_not_awaited()
        runtime.drain_trace = AsyncMock()
        runtime.ws_disconnect = AsyncMock()
        provider._api.side_effect = None
        provider._api.return_value = {}


@pytest.mark.asyncio
@pytest.mark.parametrize('os', ['windows', 'linux'])
async def test_nexus_server_does_not_inject_environment(provider, os):
    runtime = await provider.acquire('main', launch_mode='nexus_server', os=os)
    assert isinstance(runtime, RemoteRuntime)
    assert provider_module.http_client.get.await_args.args == (runtime.base_url + '/health',)
    assert 'env' not in provider._api.await_args_list[0].kwargs['payload']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['raw', 'nexus'])
async def test_linux_native_uses_osworld_and_refreshes_token(provider, mode):
    provider.os = 'linux'
    runtime = await provider.acquire('main', interface_mode=mode, template='linux-gpu-osworld', server_port=5001)
    expected_type = AGSRawOSWorldRuntime if mode == 'raw' else AGSOSWorldRuntime
    assert isinstance(runtime, expected_type)
    assert runtime.handle.tool_type == 'osworld'
    assert runtime.server_port == 5001
    assert provider_module.http_client.get.await_args.args == ('https://5001-inst-test.agrt.nano.woa.com/screenshot',)
    assert provider._api.await_args_list[0].kwargs['payload'] == {'template': 'linux-gpu-osworld'}
    provider._api.side_effect = [{'token': 'refreshed-token'}, {}]
    await provider.refresh_token('main')
    assert runtime.handle.headers == {'X-Access-Token': 'refreshed-token'}
    await provider.release('main')
    provider._api.assert_awaited_with('POST', '/v1/instances/inst-test/stop', api_key='test-key')


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['raw', 'nexus'])
async def test_linux_existing_instance_detaches_without_stopping(provider, mode):
    provider._api.side_effect = [{'status': 'RUNNING'}, {'token': 'token'}]
    runtime = await provider.acquire(
        'main', os='linux', interface_mode=mode, runtime_credential={'instance_id': 'inst-existing'}
    )
    assert isinstance(runtime, AGSRawOSWorldRuntime if mode == 'raw' else AGSOSWorldRuntime)
    await provider.release('main')
    assert provider._api.await_count == 2
    assert not provider._instance_ids


@pytest.mark.asyncio
async def test_linux_uses_runtime_platform_override(provider):
    provider.configure('main', os='linux')
    runtime = await provider.acquire('main', launch_mode='nexus_server', runtime_port=18888)
    assert isinstance(runtime, RemoteRuntime)
    assert runtime.base_url == 'https://18888-inst-test.agrt.nano.woa.com'
    assert 'env' not in provider._api.await_args_list[0].kwargs['payload']
