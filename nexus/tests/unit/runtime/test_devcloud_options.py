"""Opt-in DevCloud transport behavior, independently of any benchmark adapter."""

import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexus.runtime.providers.devcloud import async_execute
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider
from nexus.utils.async_http import http_client


@pytest.mark.asyncio
@pytest.mark.parametrize('proxy_url', [None, 'http://proxy.invalid:1080'])
async def test_sync_commands_keep_payload_budget_and_error_contract(monkeypatch, proxy_url):
    response = MagicMock(status_code=503, text='unavailable')
    response.json.return_value = {'error': 'unavailable'}
    post = AsyncMock(return_value=response)
    execute = AsyncMock()
    monkeypatch.setattr(http_client, 'post', post)
    monkeypatch.setattr(async_execute, 'execute', execute)
    runtime = DevCloudRuntime(base_url='http://vm:5000', proxy_url=proxy_url)
    result = await runtime.run_command('echo same', timeout=12)
    # Preserve the existing sync error response, rather than adopting Mac's strict HTTP exception policy.
    assert result.status.value == 'Error'
    assert result.return_code is None
    assert result.stderr == 'unavailable'
    execute.assert_not_awaited()
    call = post.await_args
    assert call.args == ('http://vm:5000/execute',)
    assert call.kwargs['timeout'].total == 1800
    assert call.kwargs['json'] == DevCloudRuntime._build_osworld_execute_payload(
        command='echo same', cwd=None, timeout=12, stdin=None, extra_env=None
    )
    if proxy_url:
        assert call.kwargs['proxy'] == proxy_url
    else:
        assert set(call.kwargs) == {'json', 'timeout'}


@pytest.mark.asyncio
@pytest.mark.parametrize('execute_mode', ['sync', 'async'])
async def test_shared_provider_passes_options_without_selecting_mac_adapter(monkeypatch, execute_mode):
    provider = DevCloudRuntimeProvider(
        sandbox_api_base_url='http://control.invalid', proxy_url='http://proxy.invalid', execute_mode=execute_mode
    )
    monkeypatch.setattr(provider, '_wait_for_osworld_ready', AsyncMock())
    runtime = await provider.acquire('main', runtime_credential={'instance_ip': '10.0.0.1'})
    assert type(runtime) is DevCloudRuntime
    assert runtime.proxy_url == 'http://proxy.invalid'
    assert runtime.execute_mode == execute_mode
    await provider.release('main')


@pytest.mark.asyncio
async def test_direct_and_proxy_runtime_requests_remain_isolated(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    seen = []

    async def handle(request):
        seen.append((request.host, request.path, request.raw_path))
        if request.path == '/execute':
            assert (await request.json())['shell'] is False
            return web.json_response({'returncode': 0, 'output': request.host, 'error': ''})
        if request.path == '/screenshot':
            return web.Response(body=b'legacy-image-bytes')
        if request.path == '/screen_size':
            return web.json_response({'width': 1920, 'height': 1080})
        if request.path == '/file':
            assert (await request.post())['file_path'] == '/tmp/test.txt'
            return web.Response(body=b'downloaded')
        if request.path == '/setup/upload':
            fields = await request.post()
            assert fields['file_path'] == '/tmp/test.txt'
            assert fields['file_data'].file.read() == b'uploaded'
            return web.Response(text='ok')
        raise AssertionError(request.path)

    direct_app, proxy_app = web.Application(), web.Application()
    direct_app.router.add_route('*', '/{path:.*}', handle)
    proxy_app.router.add_route('*', '/{path:.*}', handle)
    monkeypatch.setenv('HTTP_PROXY', 'http://unreachable.invalid:1')
    monkeypatch.setenv('http_proxy', 'http://unreachable.invalid:1')
    monkeypatch.setenv('NO_PROXY', '*')
    monkeypatch.setenv('no_proxy', '*')
    async with TestServer(direct_app) as direct, TestServer(proxy_app) as proxy:
        direct_url = str(direct.make_url('/')).rstrip('/')
        runtimes = [
            DevCloudRuntime(base_url=direct_url),
            DevCloudRuntime(base_url='http://vm.invalid:5000', proxy_url=str(proxy.make_url('/'))),
        ]

        async def exercise(runtime):
            assert (await runtime.run_command('echo probe')).return_code == 0
            assert (await runtime.call_executor('gui_screenshot', {}))['success']
            assert (
                await runtime.call_executor(
                    'download_file',
                    {
                        'files': [{'path': '/tmp/test.txt', 'encoding': 'utf8'}],
                    },
                )
            )['files'][0]['content'] == 'downloaded'
            assert (
                await runtime.call_executor(
                    'upload_file',
                    {
                        'files': [{'path': '/tmp/test.txt', 'content': 'uploaded', 'encoding': 'utf8'}],
                    },
                )
            )['results'][0]['success']

        try:
            await asyncio.gather(*(exercise(runtime) for runtime in runtimes))
        finally:
            await http_client.close()
    assert len(seen) == 10
    for host, path, raw in seen:
        assert raw == ('http://vm.invalid:5000' + path if host == 'vm.invalid' else path)


def test_async_requires_valid_budget_without_changing_sync_defaults():
    assert DevCloudRuntime(base_url='http://vm', request_timeout=0).execute_mode == 'sync'
    for timeout in [0, -1, float('inf'), float('nan')]:
        with pytest.raises(ValueError, match='finite, positive'):
            DevCloudRuntime(base_url='http://vm', request_timeout=timeout, execute_mode='async')
    with pytest.raises(ValueError, match='execute_mode'):
        DevCloudRuntime(base_url='http://vm', execute_mode='unsupported')
