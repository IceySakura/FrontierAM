"""DevCloud tests with mocked HTTP; no real resources are created."""

from __future__ import annotations

import asyncio
import hashlib
import logging

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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


def test_devcloud_auth_headers_use_md5_and_never_include_token() -> None:
    from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider

    provider = DevCloudRuntimeProvider(
        app_id='app-123',
        app_token='secret-token',
        sandbox_api_base_url='http://sandbox.example',
    )

    headers = provider._build_auth_headers(timestamp='1715680000', random_digits='12345678')

    expected = hashlib.md5(b'secret-token171568000012345678').hexdigest()
    assert headers == {
        'APPID': 'app-123',
        'TIMESTP': '1715680000',
        'RANDOM': '12345678',
        'ENCKEY': expected,
    }
    assert 'secret-token' not in ''.join(headers.values())


def test_devcloud_provider_ignores_credentials_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('DEVCLOUD_APP_ID', 'env-app')
    monkeypatch.setenv('DEVCLOUD_APP_TOKEN', 'env-token')

    from nexus.runtime.providers.devcloud.provider import DevCloudRuntimeProvider

    provider = DevCloudRuntimeProvider(
        sandbox_api_base_url='http://sandbox.example',
    )

    assert provider.app_id is None
    assert provider.app_token is None


@pytest.mark.asyncio
async def test_devcloud_provider_requires_credentials_when_creating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DEVCLOUD_APP_ID', raising=False)
    monkeypatch.delenv('DEVCLOUD_APP_TOKEN', raising=False)

    provider = _make_provider(app_id=None, app_token=None)

    with pytest.raises(ValueError, match='app_id.*creating an instance'):
        await provider.acquire('main')


def test_devcloud_registered_in_runtime_provider_registry() -> None:
    import nexus.runtime.providers.devcloud  # noqa: F401

    from nexus.runtime.providers.registry import runtime_provider_registry

    assert 'devcloud' in runtime_provider_registry


@pytest.mark.asyncio
async def test_acquire_rejects_direct_instance_ip() -> None:
    provider = _make_provider()
    provider.configure('main', instance_ip='10.0.0.42')

    with pytest.raises(ValueError, match='use runtime_credential.instance_ip'):
        await provider.acquire('main')


@pytest.mark.asyncio
async def test_acquire_creates_polls_ready_and_returns_runtime() -> None:
    provider = _make_provider()
    calls: list[tuple[str, dict | None]] = []

    async def fake_post(url: str, **kwargs):
        calls.append((url, kwargs.get('json')))
        if url == 'http://sandbox.example/mac/sandboxCreate':
            return _response(200, {'actionCode': 200, 'actionMessage': 'OK', 'data': {'taskId': 'apt-123'}})
        if url == 'http://sandbox.example/mac/sandboxTaskResult':
            return _response(
                200,
                {
                    'actionCode': 200,
                    'actionMessage': 'OK',
                    'data': {
                        'ip': '10.0.0.9',
                        'user': 'bkdevops',
                        'password': 'pw',
                        'status': 'succeeded',
                        'id': 2029232,
                        'taskId': 'apt-123',
                        'assetId': 'asset-1',
                        'os': 'Sonoma',
                        'name': 'mac-1',
                    },
                },
            )
        raise AssertionError(f'unexpected POST {url}')

    with (
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.post',
            new=AsyncMock(side_effect=fake_post),
        ),
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.get',
            new=AsyncMock(return_value=_response(200, {'status': 'ok'}, content=b'ok')),
        ) as provider_get,
    ):
        runtime = await provider.acquire('main')

    assert runtime.base_url == 'http://10.0.0.9:5000'
    assert provider.get_runtime_id('main') == '2029232'
    assert calls[0] == (
        'http://sandbox.example/mac/sandboxCreate',
        {
            'leaseTime': 300,
        },
    )
    assert calls[1] == ('http://sandbox.example/mac/sandboxTaskResult', {'taskId': 'apt-123'})
    assert provider_get.await_args.args[0] == 'http://10.0.0.9:5000/health'
    assert runtime.metadata['password'] == 'pw'
    assert runtime.metadata['taskId'] == 'apt-123'


@pytest.mark.asyncio
async def test_acquire_ready_log_redacts_runtime_password(caplog: pytest.LogCaptureFixture) -> None:
    provider = _make_provider()

    async def fake_post(url: str, **kwargs):
        if url == 'http://sandbox.example/mac/sandboxCreate':
            return _response(200, {'actionCode': 200, 'actionMessage': 'OK', 'data': {'taskId': 'apt-123'}})
        if url == 'http://sandbox.example/mac/sandboxTaskResult':
            return _response(
                200,
                {
                    'actionCode': 200,
                    'actionMessage': 'OK',
                    'data': {
                        'ip': '10.0.0.9',
                        'user': 'bkdevops',
                        'password': 'runtime-password-must-not-be-logged',
                        'status': 'succeeded',
                        'id': 2029232,
                    },
                },
            )
        raise AssertionError(url)

    with (
        caplog.at_level(logging.DEBUG, logger='nexus.runtime.providers.devcloud.provider'),
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.post',
            new=AsyncMock(side_effect=fake_post),
        ),
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.get',
            new=AsyncMock(return_value=_response(200, {'status': 'ok'}, content=b'ok')),
        ),
    ):
        runtime = await provider.acquire('main')

    messages = [record.getMessage() for record in caplog.records]
    assert runtime.metadata['password'] == 'runtime-password-must-not-be-logged'
    assert any("DevCloud 'main': READY" in message for message in messages)
    assert all('runtime-password-must-not-be-logged' not in message for message in messages)
    assert all("'password'" not in message for message in messages)


@pytest.mark.asyncio
async def test_acquire_polls_pending_before_success() -> None:
    provider = _make_provider()
    result_calls = 0

    async def fake_post(url: str, **kwargs):
        nonlocal result_calls
        if url.endswith('/mac/sandboxCreate'):
            return _response(200, {'actionCode': 200, 'data': {'taskId': 'apt-123'}})
        if url.endswith('/mac/sandboxTaskResult'):
            result_calls += 1
            if result_calls == 1:
                return _response(200, {'actionCode': 200, 'data': {'status': 'running'}})
            return _response(200, {'actionCode': 200, 'data': {'status': 'succeeded', 'ip': '10.0.0.9', 'id': 7}})
        raise AssertionError(url)

    with (
        patch('nexus.runtime.providers.devcloud.provider.asyncio.sleep', new=AsyncMock()) as sleep_mock,
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.post',
            new=AsyncMock(side_effect=fake_post),
        ),
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.get',
            new=AsyncMock(return_value=_response(200, content=b'png')),
        ),
    ):
        await provider.acquire('main')

    assert result_calls == 2
    sleep_mock.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_task_failure_raises_runtime_error() -> None:
    provider = _make_provider()

    async def fake_post(url: str, **kwargs):
        if url.endswith('/mac/sandboxCreate'):
            return _response(200, {'actionCode': 200, 'data': {'taskId': 'apt-123'}})
        if url.endswith('/mac/sandboxTaskResult'):
            return _response(200, {'actionCode': 200, 'data': {'status': 'failed', 'actionMessage': 'capacity error'}})
        raise AssertionError(url)

    with patch('nexus.runtime.providers.devcloud.provider.http_client.post', new=AsyncMock(side_effect=fake_post)):
        with pytest.raises(RuntimeError, match='failed'):
            await provider.acquire('main')


@pytest.mark.asyncio
async def test_release_waits_for_exact_instance_deletion() -> None:
    provider = _make_provider()
    provider._vm_ids['main'] = '2029232'
    provider._active_runtimes['main'] = MagicMock()

    with (
        patch(
            'nexus.runtime.providers.devcloud.provider.http_client.post',
            new=AsyncMock(
                side_effect=[
                    _response(200, {'actionCode': 200, 'data': {'taskId': 'delete-1'}}),
                    _response(200, {'actionCode': 200, 'data': {'Status': 'running'}}),
                    _response(200, {'actionCode': 200, 'data': {'Status': 'succeeded'}}),
                ]
            ),
        ) as post_mock,
        patch('nexus.runtime.providers.devcloud.provider.asyncio.sleep', new=AsyncMock()) as sleep_mock,
    ):
        await provider.release('main')

    assert post_mock.await_count == 3
    assert post_mock.await_args_list[0].args[0] == 'http://sandbox.example/mac/sandboxDelete'
    assert post_mock.await_args_list[0].kwargs['json'] == {'id': '2029232'}
    assert post_mock.await_args_list[1].kwargs['json'] == {'taskId': 'delete-1'}
    sleep_mock.assert_awaited_once_with(2.0)
    assert provider.get_active_runtimes() == {}
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_release_without_owned_instance_does_not_delete_by_identity() -> None:
    provider = _make_provider()

    with patch(
        'nexus.runtime.providers.devcloud.provider.http_client.post',
        new=AsyncMock(),
    ) as post_mock:
        await provider.release('main')

    post_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_retries_stalled_confirmation_without_resubmitting_delete() -> None:
    provider = _make_provider(request_timeout=5400, control_request_timeout=0.01)
    provider.TASK_POLL_INTERVAL = 0
    provider._vm_ids['main'] = '-42'
    provider._active_runtimes['main'] = MagicMock()
    queries = 0
    cancelled = asyncio.Event()

    async def fake_post(url, **kwargs):
        nonlocal queries
        assert kwargs['timeout'].total == 0.01
        if url.endswith('/mac/sandboxDelete'):
            return _response(200, {'actionCode': 200, 'data': {'taskId': 'delete-1'}})
        assert kwargs['json'] == {'taskId': 'delete-1'}
        queries += 1
        if queries == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return _response(200, {'actionCode': 200, 'data': {'Status': 'succeeded'}})

    with patch('nexus.runtime.providers.devcloud.provider.http_client.post', side_effect=fake_post) as post:
        await asyncio.wait_for(provider.release('main'), timeout=5)

    assert cancelled.is_set()
    assert queries == 2
    assert sum(c.args[0].endswith('/mac/sandboxDelete') for c in post.await_args_list) == 1
    assert provider.get_runtime_id('main') is None
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('jitter_fraction', [0.0, 0.5, 1.0])
async def test_release_query_backoff_resets_after_response_and_never_resubmits_delete(jitter_fraction: float) -> None:
    import aiohttp

    provider = _make_provider()
    provider._vm_ids['main'] = '-42'
    provider._active_runtimes['main'] = MagicMock()
    with (
        patch.object(
            provider,
            '_call_api',
            side_effect=[
                {'data': {'taskId': 'delete-1'}},
                asyncio.TimeoutError(),
                aiohttp.ServerDisconnectedError(),
                asyncio.TimeoutError(),
                asyncio.TimeoutError(),
                {'data': {'status': 'running'}},
                asyncio.TimeoutError(),
                {'data': {'status': 'succeeded'}},
            ],
        ) as api,
        patch('nexus.runtime.providers.devcloud.provider.asyncio.sleep', new=AsyncMock()) as sleep,
        patch(
            'nexus.runtime.providers.devcloud.provider.random.uniform',
            side_effect=lambda lower, upper: lower + (upper - lower) * jitter_fraction,
        ),
    ):
        await provider.release('main')

    factor = 0.5 + 0.5 * jitter_fraction
    assert [call.args[0] for call in sleep.await_args_list] == [
        2 * factor,
        4 * factor,
        8 * factor,
        8 * factor,
        2,
        2 * factor,
    ]
    assert api.await_args_list[0].args[1:] == ('/mac/sandboxDelete', {'id': '-42'})
    assert all(call.args[1:] == ('/mac/sandboxTaskResult', {'taskId': 'delete-1'}) for call in api.await_args_list[1:])
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('submission_behavior', ['timeout', 'stalled'])
async def test_delete_submission_timeout_preserves_result_without_resubmitting(
    submission_behavior: str, caplog: pytest.LogCaptureFixture
) -> None:
    from nexus.core.context import ExecutionContext

    provider = _make_provider(request_timeout=5400, control_request_timeout=30, release_timeout=0.02)
    provider._vm_ids['main'] = '-42'
    provider._active_runtimes['main'] = MagicMock()
    provider._task_ids['main'] = 'create-1'
    cancelled = asyncio.Event()

    async def fake_api(base_url, path, payload):
        if submission_behavior == 'timeout':
            raise asyncio.TimeoutError()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def run_task():
        async with ExecutionContext(runtime_provider=provider):
            return {'completed': True, 'score': 1}

    with patch.object(provider, '_call_api', side_effect=fake_api) as api, caplog.at_level(logging.INFO):
        assert await asyncio.wait_for(run_task(), timeout=5) == {'completed': True, 'score': 1}
        await provider.release_all()
        await provider.release('main')

    if submission_behavior == 'stalled':
        assert cancelled.is_set()
    api.assert_awaited_once_with('http://sandbox.example', '/mac/sandboxDelete', {'id': '-42'})
    assert 'deletion submission timed out' in caplog.text
    assert 'instance -42' in caplog.text
    assert 'release state unknown' in caplog.text
    assert 'deletion completed' not in caplog.text
    assert provider.get_runtime_id('main') is None
    assert provider.get_active_runtimes() == {}
    assert provider._task_ids == {}
    assert provider._delete_task_ids == {}


@pytest.mark.asyncio
async def test_submission_timeout_does_not_skip_other_instance_cleanup() -> None:
    provider = _make_provider()
    for name, vm_id in [('first', '-41'), ('second', '-42')]:
        provider._vm_ids[name] = vm_id
        provider._active_runtimes[name] = MagicMock()
    with patch.object(
        provider,
        '_call_api',
        side_effect=[
            asyncio.TimeoutError(),
            {'data': {'taskId': 'delete-2'}},
            {'data': {'status': 'succeeded'}},
        ],
    ) as api:
        await provider.release_all()

    assert [call.args[1:] for call in api.await_args_list] == [
        ('/mac/sandboxDelete', {'id': '-41'}),
        ('/mac/sandboxDelete', {'id': '-42'}),
        ('/mac/sandboxTaskResult', {'taskId': 'delete-2'}),
    ]
    assert provider.get_active_runtimes() == {}


@pytest.mark.asyncio
async def test_external_cancellation_during_submission_is_not_swallowed() -> None:
    provider = _make_provider()
    provider._vm_ids['main'] = '-42'
    provider._active_runtimes['main'] = MagicMock()
    submitting = asyncio.Event()

    async def fake_api(*args):
        submitting.set()
        await asyncio.Event().wait()

    with patch.object(provider, '_call_api', side_effect=fake_api) as api:
        task = asyncio.create_task(provider.release('main'))
        try:
            await asyncio.wait_for(submitting.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    api.assert_awaited_once()
    assert provider.get_runtime_id('main') == '-42'
    assert 'main' in provider.get_active_runtimes()


@pytest.mark.asyncio
@pytest.mark.parametrize('query_behavior', ['stalled', 'running', 'backoff'])
async def test_unconfirmed_deletion_does_not_block_result_return(
    query_behavior: str, caplog: pytest.LogCaptureFixture
) -> None:
    from nexus.core.context import ExecutionContext

    provider = _make_provider(request_timeout=5400, release_timeout=0.02)
    if query_behavior != 'backoff':
        provider.TASK_POLL_INTERVAL = 0
    provider._vm_ids['main'] = '-42'
    provider._active_runtimes['main'] = MagicMock()

    async def fake_api(base_url, path, payload):
        if path == '/mac/sandboxDelete':
            return {'data': {'taskId': 'delete-1'}}
        if query_behavior == 'stalled':
            await asyncio.Event().wait()
        if query_behavior == 'backoff':
            raise asyncio.TimeoutError()
        return {'data': {'status': 'running'}}

    async def run_task():
        async with ExecutionContext(runtime_provider=provider):
            return {'completed': True, 'score': 1}

    with patch.object(provider, '_call_api', side_effect=fake_api) as api, caplog.at_level(logging.INFO):
        assert await asyncio.wait_for(run_task(), timeout=5) == {'completed': True, 'score': 1}
        calls = api.await_count
        if query_behavior == 'backoff':
            assert calls == 2  # Deadline interrupts sleep; no extra query or delete is issued.
        await provider.release_all()
        assert api.await_count == calls

    assert 'deletion accepted but confirmation timed out' in caplog.text
    assert 'instance -42, task delete-1' in caplog.text
    assert 'deletion completed' not in caplog.text
    assert provider._delete_task_ids == {'-42': 'delete-1'}
    assert provider.get_active_runtimes() == {}
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_task_poll_budget_interrupts_inflight_query() -> None:
    provider = _make_provider(request_timeout=0.01, control_request_timeout=30)
    cancelled = asyncio.Event()

    async def stuck_query(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with patch.object(provider, '_call_api', side_effect=stuck_query):
        with pytest.raises(TimeoutError, match='did not finish within'):
            await asyncio.wait_for(provider._wait_for_task('create-1'), timeout=5)
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_task_poll_budget_limits_backoff_without_another_request() -> None:
    provider = _make_provider(request_timeout=0.02)
    with patch.object(provider, '_call_api', side_effect=asyncio.TimeoutError()) as api:
        with pytest.raises(TimeoutError, match='did not finish within'):
            await asyncio.wait_for(provider._wait_for_task('delete-1'), timeout=5)
    api.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_query_retries_disconnect_but_propagates_cancellation() -> None:
    import aiohttp

    provider = _make_provider()
    provider.TASK_POLL_INTERVAL = 0
    with patch.object(
        provider,
        '_call_api',
        side_effect=[aiohttp.ServerDisconnectedError(), {'data': {'status': 'succeeded'}}, asyncio.CancelledError()],
    ) as api:
        assert await provider._wait_for_task('delete-1') == {'status': 'succeeded'}
        with pytest.raises(asyncio.CancelledError):
            await provider._wait_for_task('delete-2')
    assert api.await_count == 3


@pytest.mark.asyncio
async def test_external_cancellation_during_backoff_keeps_accepted_delete_state() -> None:
    provider = _make_provider()
    provider._vm_ids['main'] = '-42'
    runtime = MagicMock()
    provider._active_runtimes['main'] = runtime
    sleeping = asyncio.Event()

    async def wait_for_cancellation(delay):
        sleeping.set()
        await asyncio.Event().wait()

    with (
        patch.object(
            provider, '_call_api', side_effect=[{'data': {'taskId': 'delete-1'}}, asyncio.TimeoutError()]
        ) as api,
        patch('nexus.runtime.providers.devcloud.provider.asyncio.sleep', side_effect=wait_for_cancellation),
    ):
        task = asyncio.create_task(provider.release('main'))
        try:
            await asyncio.wait_for(sleeping.wait(), timeout=5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert api.await_count == 2
    assert provider._delete_task_ids == {'-42': 'delete-1'}
    assert provider.get_runtime_id('main') == '-42'
    assert provider.get_active_runtimes()['main'] is runtime


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['tls', 'http', 'api', 'json', 'terminal'])
async def test_task_query_does_not_retry_permanent_failures(failure: str) -> None:
    import ssl

    import aiohttp

    provider = _make_provider()
    responses = {
        'http': _response(403, text='Forbidden'),
        'api': _response(200, {'actionCode': 403, 'actionMessage': 'Forbidden'}),
        'json': _response(200, text='not JSON'),
        'terminal': _response(200, {'actionCode': 200, 'data': {'status': 'failed'}}),
    }
    if failure == 'tls':
        outcome = aiohttp.ClientConnectorSSLError(MagicMock(), ssl.SSLError('TLS failure'))
    else:
        outcome = responses[failure]
        if failure == 'json':
            outcome.json.side_effect = ValueError('invalid JSON')
    with (
        patch('nexus.runtime.providers.devcloud.provider.http_client.post', side_effect=[outcome]) as post,
        patch('nexus.runtime.providers.devcloud.provider.asyncio.sleep', new=AsyncMock()) as sleep,
    ):
        with pytest.raises((aiohttp.ClientSSLError, RuntimeError)):
            await provider._wait_for_task('delete-1')
    post.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.parametrize('field', ['control_request_timeout', 'release_timeout'])
@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_management_timeouts_require_finite_positive_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        _make_provider(**{field: value})


@pytest.mark.asyncio
async def test_cancelled_create_resolves_instance_and_waits_for_cleanup() -> None:
    provider = _make_provider()
    with (
        patch.object(
            provider,
            '_call_api',
            new=AsyncMock(
                side_effect=[
                    {'data': {'taskId': 'create-1'}},
                    {'data': {'taskId': 'delete-1'}},
                ]
            ),
        ) as api,
        patch.object(
            provider,
            '_wait_for_task',
            new=AsyncMock(
                side_effect=[
                    asyncio.CancelledError(),
                    {'status': 'succeeded', 'id': -123, 'ip': '10.0.0.9'},
                    {'status': 'succeeded'},
                ]
            ),
        ) as wait,
    ):
        with pytest.raises(asyncio.CancelledError):
            await provider.acquire('main')
    assert api.await_args_list[1].args == ('http://sandbox.example', '/mac/sandboxDelete', {'id': '-123'})
    assert wait.await_args_list[-1].args == ('delete-1',)
    assert provider.get_runtime_id('main') is None


@pytest.mark.asyncio
async def test_delete_failure_retains_instance_and_is_not_reported_as_released() -> None:
    provider = _make_provider()
    runtime = MagicMock()
    provider._active_runtimes['main'] = runtime
    provider._vm_ids['main'] = '-123'
    with (
        patch.object(provider, '_call_api', new=AsyncMock(return_value={'data': {'taskId': 'delete-1'}})),
        patch.object(provider, '_wait_for_task', new=AsyncMock(side_effect=RuntimeError('delete failed'))),
    ):
        with pytest.raises(RuntimeError, match='delete failed'):
            await provider.release('main')
    assert provider.get_runtime_id('main') == '-123'
    assert provider.get_active_runtimes()['main'] is runtime


@pytest.mark.asyncio
async def test_documented_result_field_names_and_failure_redaction() -> None:
    provider = _make_provider()
    with patch.object(
        provider,
        '_call_api',
        new=AsyncMock(
            side_effect=[
                {'data': {'Status': 'succeeded', 'Ip': '10.0.0.9', 'Id': -123, 'TaskId': 'create-1'}},
                {'data': {'Status': 'failed', 'Password': 'private-password'}},
            ]
        ),
    ):
        result = await provider._wait_for_task('create-1')
        assert result == {'status': 'succeeded', 'ip': '10.0.0.9', 'id': -123, 'taskId': 'create-1'}
        with pytest.raises(RuntimeError, match='failed') as error:
            await provider._wait_for_task('create-2')
        assert 'private-password' not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', [None, 'devcloud', 'osworld'])
async def test_default_runtime_keeps_original_execution_client(kind):
    from nexus.runtime.providers.devcloud.client import DevCloudRuntime

    provider = _make_provider()
    with patch.object(provider, '_wait_for_osworld_ready', new=AsyncMock()):
        runtime = await provider.acquire('main', runtime_kind=kind, runtime_credential={'instance_ip': '10.0.0.42'})
    assert type(runtime) is DevCloudRuntime
    assert runtime.execute_mode == 'sync'
    assert runtime.proxy_url is None
    await provider.release('main')


@pytest.mark.asyncio
async def test_unknown_runtime_kind_fails_before_create():
    provider = _make_provider()
    with patch.object(provider, '_call_api', new=AsyncMock()) as api:
        with pytest.raises(ValueError, match='runtime_kind'):
            await provider.acquire('main', runtime_kind='typo')
    api.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('lease_time', [0, -1, 299, 300])
async def test_lease_validation_precedes_create(lease_time):
    provider = _make_provider(lease_time=lease_time)
    with patch.object(provider, '_call_api', new=AsyncMock(side_effect=RuntimeError('submitted'))) as api:
        if lease_time == 299:
            with pytest.raises(ValueError, match='at least 300'):
                await provider.acquire('main')
            api.assert_not_awaited()
        else:
            with pytest.raises(RuntimeError, match='submitted'):
                await provider.acquire('main')
            api.assert_awaited_once_with('http://sandbox.example', '/mac/sandboxCreate', {'leaseTime': lease_time})


@pytest.mark.asyncio
async def test_legacy_identity_fields_are_ignored_and_only_names_are_logged(caplog):
    legacy = dict.fromkeys(
        ('control_api_base_url', 'project', 'pipeline_id', 'build_id', 'vm_seq_id', 'model'), 'legacy-value'
    )
    with caplog.at_level(logging.WARNING):
        provider = _make_provider(**legacy)
    assert 'obsolete configuration fields' in caplog.text
    assert 'legacy-value' not in caplog.text
    assert not legacy.keys() & provider.config.keys()
    with patch.object(provider, '_call_api', new=AsyncMock(side_effect=RuntimeError('submitted'))) as api:
        with pytest.raises(RuntimeError, match='submitted'):
            await provider.acquire('main', **legacy)
    api.assert_awaited_once_with('http://sandbox.example', '/mac/sandboxCreate', {'leaseTime': 300})
