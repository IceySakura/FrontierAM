from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from nexus.runner.hooks import run_hook_registry
from nexus.runner.hooks.gui_recording_cos import GuiRecordingCosHook
from nexus.runtime.executors.models.gui import GuiStartRecordingResponse, GuiStopRecordingResponse
from nexus.utils.cos import CosCredentials, CosV5Signer


@pytest.fixture
def recording(monkeypatch, request):
    credentials = CosCredentials.from_shared_payload({
        'secret_id': 'test-cos-id',
        'secret_key': 'test-cos-secret',
        'token': 'test-session-token',
    })
    loader = Mock(return_value=credentials)
    monkeypatch.setattr('nexus.runner.hooks.gui_recording_cos.load_cos_credentials', loader)
    runtime = SimpleNamespace(
        gui_start_recording=AsyncMock(
            return_value=GuiStartRecordingResponse(success=True, output_path='/tmp/test.mp4')
        ),
        gui_stop_recording=AsyncMock(
            return_value=GuiStopRecordingResponse(
                success=True,
                output_path='/tmp/test.mp4',
                size_bytes=3,
                duration_seconds=1,
                width=800,
                height=600,
                fps=10,
            )
        ),
    )
    ctx = SimpleNamespace(run_id='test-run', get_runtime=AsyncMock(return_value=runtime))
    kwargs = {'url_expires_seconds': request.param} if hasattr(request, 'param') else {}
    hook = GuiRecordingCosHook(bucket='demo-1250000000', key_prefix='recordings/test', **kwargs)
    calls = []

    async def upload(**payload):
        calls.append(payload)
        if payload['action'] == 'inspect':
            return {'size_bytes': 3, 'content_md5': 'kAFQmDzST7DWlj99KOF/cg=='}
        return {'etag': '"900150983cd24fb0d6963f7d28e17f72"'}

    monkeypatch.setattr(hook, '_run_upload_script', AsyncMock(side_effect=upload))
    return hook, ctx, runtime, calls, loader


@pytest.mark.asyncio
@pytest.mark.parametrize('recording', [3600, 300, 2592000], indirect=True)
async def test_cos_hook_presigns_on_host_and_uploads_in_sandbox(recording):
    hook, ctx, runtime, calls, loader = recording
    started = int(datetime.now(timezone.utc).timestamp())
    await hook.on_context_ready(ctx)
    await hook.on_run_output(ctx, None)
    await hook.on_context_closing(ctx)
    loader.assert_called_once_with()
    runtime.gui_stop_recording.assert_awaited_once()
    assert [call['action'] for call in calls] == ['inspect', 'upload']
    upload = calls[-1]
    parsed = urlsplit(upload['url'])
    signature = parse_qs(parsed.query)
    assert parsed.scheme == 'https'
    assert parsed.hostname == 'demo-1250000000.cos-internal.ap-guangzhou.tencentcos.cn'
    assert parsed.path == '/' + hook.key
    assert signature['q-header-list'] == ['content-md5;content-type;host']
    assert signature['x-cos-security-token'] == ['test-session-token']
    assert 'test-cos-secret' not in str(upload)
    assert upload['headers'] == {'Content-Type': 'video/mp4', 'Content-MD5': 'kAFQmDzST7DWlj99KOF/cg=='}
    assert upload['method'] == 'PUT'
    assert upload['length'] == 3
    result = hook.get_result()
    assert result['source_deleted'] is True
    assert result['cos']['key'] == hook.key
    assert result['cos']['uri'].startswith('cos://demo-1250000000/recordings/test/')
    download = urlsplit(result['cos']['url'])
    download_query = parse_qs(download.query)
    assert download.scheme == 'https'
    assert download.hostname == 'demo-1250000000.cos.ap-guangzhou.myqcloud.com'
    assert download.path == parsed.path
    assert download_query['q-header-list'] == ['host']
    end = int(download_query['q-sign-time'][0].split(';')[1])
    assert (
        started + hook.url_expires_seconds
        <= end
        <= int(datetime.now(timezone.utc).timestamp()) + hook.url_expires_seconds
    )
    assert datetime.fromisoformat(result['cos']['url_expires_at']).timestamp() == end
    expected = CosV5Signer('test-cos-id', 'test-cos-secret', token='test-session-token').presign_url(
        'GET',
        f'https://{download.netloc}{download.path}',
        expires=hook.url_expires_seconds,
        timestamp=end - hook.url_expires_seconds,
    )
    assert result['cos']['url'] == expected
    assert 'test-cos-secret' not in str(result)
    assert 'artifact' not in result


@pytest.mark.asyncio
async def test_cos_hook_retries_same_object_without_stopping_twice(recording):
    hook, ctx, runtime, calls, _loader = recording
    upload = hook._run_upload_script.side_effect
    attempts = 0

    async def flaky_upload(**payload):
        nonlocal attempts
        result = await upload(**payload)
        if payload['action'] == 'upload':
            attempts += 1
            if attempts == 1:
                raise RuntimeError('connection failed')
        return result

    hook._run_upload_script.side_effect = flaky_upload
    await hook.on_context_ready(ctx)
    with pytest.raises(RuntimeError, match='connection failed'):
        await hook.on_run_output(ctx, None)
    assert 'source_deleted' not in hook.get_result()
    await hook.on_context_closing(ctx)
    runtime.gui_stop_recording.assert_awaited_once()
    uploads = [call for call in calls if call['action'] == 'upload']
    assert len(uploads) == 2
    assert {urlsplit(call['url']).path for call in uploads} == {'/' + hook.key}


@pytest.mark.asyncio
async def test_missing_cos_credentials_do_not_start_recording(recording):
    hook, ctx, _runtime, _calls, loader = recording
    loader.side_effect = ValueError('missing COS credentials')
    with pytest.raises(ValueError, match='missing COS credentials'):
        await hook.on_context_ready(ctx)
    await hook.on_context_closing(ctx)
    ctx.get_runtime.assert_not_awaited()


@pytest.mark.asyncio
async def test_cos_hook_rejects_missing_upload_etag(recording):
    hook, ctx, _runtime, _calls, _loader = recording
    hook._run_upload_script.side_effect = [{'size_bytes': 3, 'content_md5': 'kAFQmDzST7DWlj99KOF/cg=='}, {'etag': None}]
    await hook.on_context_ready(ctx)
    with pytest.raises(RuntimeError, match='no ETag'):
        await hook.on_run_output(ctx, None)
    assert 'cos' not in hook.get_result()
    assert 'source_deleted' not in hook.get_result()


def test_cos_recording_hook_is_discoverable():
    from nexus.core.module_dependencies import resolve_host_module_dependencies, selectors_from_run_input
    from nexus.core.module_meta import discover_all_modules

    selectors = selectors_from_run_input({'runner': {'hooks': {'video': {'name': 'gui_recording_cos'}}}})
    resolved = resolve_host_module_dependencies(selectors, discover_all_modules())
    assert [module.module_path for module in resolved.modules] == ['nexus.runner.hooks.gui_recording_cos']
    assert run_hook_registry.get('gui_recording_cos') is GuiRecordingCosHook


@pytest.mark.parametrize('value', [0, -1, 1.5, True, '3600', None])
def test_cos_hook_rejects_invalid_url_expiry(value):
    with pytest.raises(ValueError, match='url_expires_seconds must be a positive integer'):
        GuiRecordingCosHook(bucket='demo-1250000000', url_expires_seconds=value)
