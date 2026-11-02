import asyncio

from unittest.mock import AsyncMock

import pytest

from nexus.runtime.providers.ags import AGSInstanceHandle, AGSOSWorldRuntime
from nexus.utils.async_http import HTTPResponse


def runtime(timeout=2400):
    return AGSOSWorldRuntime(
        AGSInstanceHandle(instance_id='test', access_token='test', domain='example.invalid'),
        request_timeout=timeout,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(('command_timeout', 'provider_timeout', 'expected'), [(60, 2400, 60), (60, 10, 10)])
async def test_execute_host_deadline(monkeypatch, command_timeout, provider_timeout, expected):
    rt = runtime(provider_timeout)
    response = HTTPResponse(
        status=200,
        headers={},
        text='{"returncode":0,"output":"ok"}',
        content=b'{"returncode":0,"output":"ok"}',
        url='unused',
        encoding='utf-8',
    )
    http = AsyncMock(return_value=response)
    monkeypatch.setattr(rt, '_http_request', http)
    result = await rt.run_command(command='echo ok', timeout=command_timeout)
    assert result.return_code == 0
    assert http.call_args.kwargs['timeout'].total == expected
    assert f'{command_timeout}s bash -lc' in http.call_args.kwargs['json']['command'][2]
    http.assert_awaited_once()
    assert rt.request_timeout == provider_timeout


@pytest.mark.asyncio
async def test_hung_execute_cancels_each_attempt_and_stops_after_three(monkeypatch, caplog):
    rt = runtime()
    calls = []
    cancelled = asyncio.Event()

    async def hung(*args, **kwargs):
        calls.append(kwargs)
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(rt, '_http_request', hung)
    with pytest.raises(TimeoutError, match='completion is unknown'):
        await asyncio.wait_for(rt.run_command(command='potential side effect', timeout=0.02), timeout=1)
    assert cancelled.is_set()
    assert len(calls) == 3
    assert all(call['timeout'].total == 0.02 for call in calls)
    assert all(call['json'] == calls[0]['json'] for call in calls)
    assert 'attempt=3/3' in caplog.text
    assert 'retrying may repeat side effects' in caplog.text
    assert 'potential side effect' not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('failures', [1, 2])
async def test_execute_timeout_recovers_within_three_attempts(monkeypatch, failures):
    rt = runtime()
    response = HTTPResponse(
        status=200,
        headers={},
        text='{"returncode":0,"output":"ok"}',
        content=b'{"returncode":0,"output":"ok"}',
        url='unused',
        encoding='utf-8',
    )
    http = AsyncMock(side_effect=[TimeoutError('read timeout')] * failures + [response])
    monkeypatch.setattr(rt, '_http_request', http)
    result = await rt.run_command(command='echo ok', timeout=60)
    assert result.return_code == 0
    assert result.stdout == 'ok'
    assert http.await_count == failures + 1
    assert all(call.kwargs['timeout'].total == 60 for call in http.call_args_list)
    assert all(call.kwargs['json'] == http.call_args.kwargs['json'] for call in http.call_args_list)


@pytest.mark.asyncio
async def test_other_endpoint_deadline_is_not_retried(monkeypatch):
    rt = runtime()
    http = AsyncMock(side_effect=TimeoutError('read timeout'))
    monkeypatch.setattr(rt, '_http_request', http)
    with pytest.raises(TimeoutError, match='1 attempts'):
        await rt._post_osworld_json('/setup/example', {}, timeout=60)
    http.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_error_is_not_timeout_retry(monkeypatch):
    rt = runtime()
    http = AsyncMock(side_effect=ConnectionResetError('reset'))
    monkeypatch.setattr(rt, '_http_request', http)
    with pytest.raises(ConnectionResetError, match='reset'):
        await rt.run_command(command='echo ok', timeout=60)
    http.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancellation_during_retry_stops_immediately(monkeypatch):
    rt = runtime()
    http = AsyncMock(side_effect=[TimeoutError('first attempt'), asyncio.CancelledError()])
    monkeypatch.setattr(rt, '_http_request', http)
    with pytest.raises(asyncio.CancelledError):
        await rt.run_command(command='echo ok', timeout=60)
    assert http.await_count == 2


@pytest.mark.asyncio
async def test_other_posts_keep_default_timeout(monkeypatch):
    rt = runtime()
    post = AsyncMock(return_value=({}, '', 200))
    monkeypatch.setattr(rt, '_http_post_json', post)
    assert await rt._post_osworld_json('/setup/example', {}) == ({}, '', 200)
    post.assert_awaited_once_with('https://5000-test.example.invalid/setup/example', {})


@pytest.mark.asyncio
async def test_cancellation_is_not_timeout(monkeypatch):
    rt = runtime()
    http = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(rt, '_http_request', http)
    with pytest.raises(asyncio.CancelledError):
        await rt.run_command(command='echo ok', timeout=60)
    http.assert_awaited_once()


@pytest.mark.asyncio
async def test_guest_timeout_still_has_its_original_status(monkeypatch):
    rt = runtime()
    http = AsyncMock(
        return_value=HTTPResponse(
            status=200,
            headers={},
            text='{"returncode":124,"output":"partial"}',
            content=b'{"returncode":124,"output":"partial"}',
            url='unused',
            encoding='utf-8',
        )
    )
    monkeypatch.setattr(rt, '_http_request', http)
    result = await rt.run_command(command='sleep 120', timeout=60)
    assert result.status.value == 'TimeLimitExceeded'
    assert result.return_code == 124
    assert result.stdout == 'partial'
    http.assert_awaited_once()
