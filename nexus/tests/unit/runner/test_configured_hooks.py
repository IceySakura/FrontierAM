import asyncio
import uuid

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from nexus.core import TaskRunInput, TaskRunOutput, task_registry
from nexus.core.context import ExecutionContext, require_context
from nexus.runner import single
from nexus.runner.compose import compile_single_file
from nexus.runner.hooks import RunHook, run_hook_registry
from nexus.runner.hooks.composed import ComposedHook
from nexus.runner.hooks.gui_recording import GuiRecordingHook
from nexus.runner.single import HookConfig, RunHookError, RunHooks, RunInput, RunOutput, run
from nexus.runtime.executors.models.gui import GuiStartRecordingResponse, GuiStopRecordingResponse
from nexus.runtime.providers.base import RuntimeProvider
from nexus.utils.artifact.errors import ArtifactPathError
from nexus.utils.artifact.models import ArtifactInfo, UploadSession


@pytest.mark.parametrize('platform', ['waa', 'devcloud_macos'])
def test_recording_example_composes_with_existing_runtime_config(platform, monkeypatch):
    for name in [
        'AGS_SECRET_ID',
        'AGS_SECRET_KEY',
        'AGS_WAA_TOOL_ID',
        'DEVCLOUD_APP_ID',
        'DEVCLOUD_APP_TOKEN',
        'AI_HUB_API_KEY',
    ]:
        monkeypatch.setenv(name, 'example')
    root = Path(__file__).resolve().parents[3]
    config = compile_single_file(root / 'docs/examples/configs' / f'gui_recording_{platform}.yaml')
    hook = config.runner.hooks['recording']
    assert hook.name == 'gui_recording'
    assert hook.required is False
    assert hook.args == {'runtime_name': 'main', 'storage_pool': 'hy-gui', 'fps': 10, 'max_duration_seconds': 3600}
    assert config.runtime_provider_name == ('devcloud' if platform == 'devcloud_macos' else 'ags')
    assert config.runner.timeout == 4200


def register_task(register_test_component, events, *, outcome='success', started=None):
    name = f'hook_task_{uuid.uuid4().hex}'

    @register_test_component(task_registry, name)
    async def task(_input: TaskRunInput) -> TaskRunOutput:
        events.append('task')
        if started:
            started.set()
        if outcome == 'error':
            raise ValueError('task failed')
        if outcome in {'timeout', 'cancel'}:
            await asyncio.Event().wait()
        return TaskRunOutput()

    return name


def register_hook(register_test_component, events, *, fail=None):
    name = f'hook_{uuid.uuid4().hex}'

    @register_test_component(run_hook_registry, name)
    class Hook(RunHook):
        def __init__(self, label):
            self.label = label
            self.count = 0

        async def on_context_ready(self, ctx):
            assert require_context() is ctx
            self.count += 1
            events.append(f'{self.label}:ready')
            if fail == 'ready':
                raise ValueError('ready failed')

        async def on_run_output(self, ctx, output):
            events.append(f'{self.label}:output')
            if fail == 'output':
                raise ValueError('output failed')

        async def on_context_closing(self, ctx):
            assert require_context() is ctx
            events.append(f'{self.label}:closing')
            if fail == 'closing':
                raise ValueError('closing failed')

        def get_result(self):
            if fail == 'get_result':
                raise ValueError('get_result failed')
            return {'count': self.count}

    return name


def test_hook_required_defaults_to_true_and_round_trips():
    assert HookConfig(name='gui_recording').required is True
    config = HookConfig(name='gui_recording', required=False)
    assert HookConfig.model_validate_json(config.model_dump_json()).required is False


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['on_context_ready', 'on_run_output', 'on_context_closing', 'get_result'])
@pytest.mark.parametrize('required', [True, False])
async def test_composed_hook_reraises_only_required_errors(stage, required):
    failure = RuntimeError('hook failed')
    hook = RunHook()
    hook.required = required
    if stage == 'get_result':
        hook.get_result = Mock(side_effect=failure)
    else:
        setattr(hook, stage, AsyncMock(side_effect=failure))
    composed = ComposedHook({'test': hook})
    ctx = ExecutionContext(task_name='test')
    if stage != 'on_context_ready':
        await composed.on_context_ready(ctx)
    output = RunOutput(success=True)
    if stage in {'on_run_output', 'get_result'}:
        callback = composed.on_run_output(ctx, output)
    elif stage == 'on_context_closing':
        await composed.on_run_output(ctx, output)
        callback = composed.on_context_closing(ctx)
    else:
        callback = composed.on_context_ready(ctx)
    if required:
        with pytest.raises(RuntimeError) as exc:
            await callback
        assert exc.value is failure
    else:
        await callback
    if stage == 'on_context_ready' and not required:
        await composed.on_run_output(ctx, output)
    if stage != 'on_context_ready' or not required:
        assert output.hook_errors[f'test.{stage}'] == 'RuntimeError: hook failed'
    assert output.success
    assert output.error is None


@pytest.mark.asyncio
async def test_optional_hook_does_not_swallow_cancellation():
    hook = RunHook()
    hook.required = False
    hook.on_context_ready = AsyncMock(side_effect=asyncio.CancelledError)
    composed = ComposedHook({'test': hook})
    with pytest.raises(asyncio.CancelledError):
        await composed.on_context_ready(ExecutionContext(task_name='test'))


@pytest.mark.asyncio
async def test_configured_hooks_round_trip_order_and_embedding_compatibility(
    register_test_component, monkeypatch, tmp_path
):
    events = []
    cache_context = single.cache_context

    @contextmanager
    def observe_cache(*args, **kwargs):
        events.append('cache:enter')
        with cache_context(*args, **kwargs) as cache:
            yield cache
        events.append('cache:exit')

    monkeypatch.setattr(single, 'cache_context', observe_cache)
    hook_name = register_hook(register_test_component, events)
    config = RunInput.from_dict({
        'task_name': register_task(register_test_component, events),
        'runner': {
            'cache': {'enabled': True, 'file_path': str(tmp_path / 'cache.json')},
            'hooks': {
                'first': {'name': hook_name, 'args': {'label': 'a'}},
                'second': {'name': hook_name, 'args': {'label': 'b'}},
            },
        },
    })
    config = RunInput.model_validate_json(config.model_dump_json())

    async def embedded_output(ctx, output):
        assert output.hook_results == {'first': {'count': 1}, 'second': {'count': 1}}
        events.append('embedded:output')

    hooks = RunHooks(
        on_context_ready=lambda ctx: events.append('embedded:ready'),
        on_run_output=embedded_output,
        on_context_closing=lambda ctx: events.append('embedded:closing'),
    )
    original_callbacks = (hooks.on_context_ready, hooks.on_run_output, hooks.on_context_closing)
    for _ in range(2):
        events.clear()
        result = await run(config, hooks=hooks)
        assert result.success
        assert events == [
            'embedded:ready',
            'a:ready',
            'b:ready',
            'cache:enter',
            'task',
            'cache:exit',
            'a:output',
            'b:output',
            'embedded:output',
            'b:closing',
            'a:closing',
            'embedded:closing',
        ]
        assert (hooks.on_context_ready, hooks.on_run_output, hooks.on_context_closing) == original_callbacks


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['output', 'closing'])
async def test_merged_hooks_preserve_caller_failure_reporting(register_test_component, stage):
    events = []

    def fail(*args):
        raise RuntimeError('embedding failed')

    hooks = RunHooks(**{f'on_run_{stage}' if stage == 'output' else 'on_context_closing': fail})
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={
                'hooks': {
                    'bad': {
                        'name': register_hook(register_test_component, events, fail=stage),
                        'required': False,
                        'args': {'label': 'bad'},
                    }
                }
            },
        ),
        hooks=hooks,
    )
    assert result.success
    assert result.error is None
    assert result.hook_errors[f'bad.on_run_{stage}' if stage == 'output' else 'bad.on_context_closing']
    assert events[-1] == 'bad:closing'
    assert len(hooks.hook_failures) == 1
    assert hooks.hook_failures[0].stage == ('run_output' if stage == 'output' else 'context_closing')
    with pytest.raises(RunHookError, match='embedding failed'):
        hooks.raise_if_failed()


@pytest.mark.asyncio
async def test_embedding_ready_failure_does_not_enter_configured_hooks(register_test_component):
    events = []

    def fail(ctx):
        raise RuntimeError('embedding ready failed')

    hooks = RunHooks(on_context_ready=fail, on_context_closing=lambda ctx: events.append('embedded:closing'))
    with pytest.raises(RuntimeError, match='embedding ready failed'):
        await run(
            RunInput(
                task_name=register_task(register_test_component, events),
                runner={
                    'hooks': {
                        'video': {'name': register_hook(register_test_component, events), 'args': {'label': 'video'}}
                    }
                },
            ),
            hooks=hooks,
        )
    assert events == ['embedded:closing']


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['ready', 'output', 'closing'])
async def test_optional_hook_failure_is_visible_and_other_hooks_continue(register_test_component, failure):
    events = []
    good = register_hook(register_test_component, events)
    bad = register_hook(register_test_component, events, fail=failure)
    config = RunInput(
        task_name=register_task(register_test_component, events),
        runner={
            'hooks': {
                'good': {'name': good, 'args': {'label': 'good'}},
                'bad': {'name': bad, 'required': False, 'args': {'label': 'bad'}},
                'last': {'name': good, 'args': {'label': 'last'}},
            }
        },
    )
    result = await run(config)
    assert result.success
    assert result.error is None
    assert result.hook_errors[f'bad.on_run_{failure}' if failure == 'output' else f'bad.on_context_{failure}']
    assert events[-2:] == ['bad:closing', 'good:closing']
    assert 'last:output' in events
    assert 'task' in events


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error'])
@pytest.mark.parametrize('failure', ['output', 'closing', 'get_result'])
@pytest.mark.parametrize('required', [True, False])
async def test_hook_error_preserves_task_status(register_test_component, outcome, failure, required):
    events = []
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events, outcome=outcome),
            runner={
                'hooks': {
                    'bad': {
                        'name': register_hook(register_test_component, events, fail=failure),
                        'required': required,
                        'args': {'label': 'bad'},
                    }
                }
            },
        )
    )
    assert result.success is (outcome == 'success')
    if outcome == 'success':
        assert result.error is None
    else:
        assert result.error.type == 'ValueError'
        assert result.error.message == 'task failed'
    method = {'output': 'on_run_output', 'closing': 'on_context_closing', 'get_result': 'get_result'}[failure]
    assert result.hook_errors[f'bad.{method}'] == f'ValueError: {failure} failed'


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error'])
async def test_required_closing_failure_does_not_skip_other_cleanup(register_test_component, outcome):
    events = []
    good = register_hook(register_test_component, events)
    bad = register_hook(register_test_component, events, fail='closing')
    hooks = RunHooks(on_context_closing=lambda _ctx: events.append('embedding:closing'))
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events, outcome=outcome),
            runner={
                'hooks': {
                    'first': {'name': good, 'args': {'label': 'first'}},
                    'bad': {'name': bad, 'args': {'label': 'bad'}},
                    'last': {'name': good, 'args': {'label': 'last'}},
                }
            },
        ),
        hooks=hooks,
    )

    assert events[-4:] == ['last:closing', 'bad:closing', 'first:closing', 'embedding:closing']
    assert result.success is (outcome == 'success')
    if outcome == 'error':
        assert result.error.message == 'task failed'
    assert result.hook_errors['bad.on_context_closing'] == 'ValueError: closing failed'
    assert hooks.hook_failures[0].stage == 'context_closing'


@pytest.mark.asyncio
async def test_unknown_hook_is_an_initialization_failure(register_test_component):
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, []),
            runner={'hooks': {'missing': {'name': 'no_such_run_hook'}}},
        )
    )
    assert not result.success
    assert 'no_such_run_hook' in result.error.message


@pytest.fixture
def recording_runtime(monkeypatch):
    events = []
    runtime = SimpleNamespace(
        gui_start_recording=AsyncMock(
            return_value=GuiStartRecordingResponse(success=True, output_path='/tmp/video.mp4')
        ),
        gui_stop_recording=AsyncMock(
            return_value=GuiStopRecordingResponse(
                success=True,
                output_path='/tmp/video.mp4',
                size_bytes=5,
                duration_seconds=1,
                width=1920,
                height=1080,
                fps=10,
            )
        ),
    )

    async def start(**kwargs):
        events.append('start')
        return runtime.gui_start_recording.return_value

    async def stop():
        events.append('stop')
        return runtime.gui_stop_recording.return_value

    async def remote_script(**kwargs):
        events.append(kwargs['action'])
        if kwargs['action'] == 'inspect':
            return {'size_bytes': 5, 'content_md5': 'md5'}
        return {'etag': 'part-etag'}

    remote = AsyncMock(side_effect=remote_script)
    monkeypatch.setattr(GuiRecordingHook, '_run_upload_script', remote)
    runtime.gui_start_recording.side_effect = start
    runtime.gui_stop_recording.side_effect = stop
    monkeypatch.setattr(ExecutionContext, 'get_runtime', AsyncMock(return_value=runtime))

    async def release():
        events.append('release')

    monkeypatch.setattr(single, '_create_runtime_provider', lambda _: Mock(spec=RuntimeProvider, release_all=release))
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.create_upload.return_value = UploadSession(
        artifact_id='artifact-1',
        upload_id='upload-1',
        status='pending',
        method='PUT',
        url='https://cos.example/upload',
        headers={'Content-Type': 'video/mp4'},
    )
    client.get_artifact.return_value = ArtifactInfo(
        artifact_id='artifact-1',
        name='video.mp4',
        content_type='video/mp4',
        size_bytes=5,
        status='ready',
    )
    monkeypatch.setattr('nexus.runner.hooks.gui_recording.ArtifactClient', lambda: client)
    return runtime, events, remote, client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('artifact_folder', 'artifact_name', 'parent_id'),
    [
        (None, None, None),
        ('/', None, None),
        ('/gui-recording/waa', None, 'folder-waa'),
        ('/gui-recording/waa', 'demo.mp4', 'folder-waa'),
    ],
)
async def test_recording_upload_resolves_folder_and_preserves_filename(
    register_test_component, recording_runtime, artifact_folder, artifact_name, parent_id
):
    _, events, _, client = recording_runtime
    client.resolve_resource_id.return_value = parent_id
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={
                'hooks': {
                    'video': {
                        'name': 'gui_recording',
                        'args': {
                            'artifact_name': artifact_name,
                            'artifact_folder': artifact_folder,
                        },
                    }
                }
            },
        )
    )
    assert result.success
    run_id = result.hook_results['video']['run_id']
    if parent_id is not None:
        client.resolve_resource_id.assert_awaited_once_with(artifact_folder, expect='folder')
    else:
        client.resolve_resource_id.assert_not_awaited()
    upload = client.create_upload.call_args.kwargs
    assert upload['filename'] == (artifact_name or f'recording-{run_id}.mp4')
    assert upload['parent_folder_id'] == parent_id
    assert upload['idempotency_key'] == f'gui-recording:{run_id}:main'
    client.create_folder.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_recording_folder_fails_before_recording_or_task(register_test_component, recording_runtime):
    runtime, events, _, client = recording_runtime
    client.resolve_resource_id.side_effect = ArtifactPathError('folder not found')
    with pytest.raises(ArtifactPathError, match='folder not found'):
        await run(
            RunInput(
                task_name=register_task(register_test_component, events),
                runner={'hooks': {'video': {'name': 'gui_recording', 'args': {'artifact_folder': '/missing'}}}},
            )
        )
    assert 'task' not in events
    runtime.gui_start_recording.assert_not_awaited()
    client.create_upload.assert_not_awaited()
    client.create_folder.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'error', 'timeout', 'cancel'])
async def test_recording_uploaded_before_release_on_every_task_exit(
    register_test_component, recording_runtime, outcome
):
    runtime, events, remote, client = recording_runtime
    started = asyncio.Event()
    config = RunInput(
        task_name=register_task(register_test_component, events, outcome=outcome, started=started),
        runner={
            'timeout': 0.01 if outcome == 'timeout' else None,
            'hooks': {
                'video': {'name': 'gui_recording', 'args': {'fps': 12, 'max_duration_seconds': 15}},
            },
        },
    )
    job = asyncio.create_task(run(config))
    if outcome == 'cancel':
        await asyncio.wait_for(started.wait(), timeout=5)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
    else:
        output = await job
        assert output.success is (outcome == 'success')
        assert output.hook_results['video']['artifact']['artifact_id'] == 'artifact-1'
        assert 'local_path' not in output.hook_results['video']
    runtime.gui_start_recording.assert_awaited_once_with(output_path=None, fps=12, max_duration_seconds=15)
    runtime.gui_stop_recording.assert_awaited_once()
    assert events == ['start', 'task', 'stop', 'inspect', 'upload', 'release']
    client.complete_upload.assert_awaited_once_with('upload-1', parts=None)
    assert remote.call_args.kwargs['url'] == 'https://cos.example/upload'
    assert client.create_upload.call_args.kwargs['content_md5'] == 'md5'


@pytest.mark.asyncio
async def test_recording_stops_and_uploads_before_hold_and_early_output(
    register_test_component, recording_runtime, tmp_path, monkeypatch
):
    _, events, _, _ = recording_runtime
    result_file = tmp_path / 'result.json'

    async def hold(*args, **kwargs):
        assert events == ['start', 'task', 'stop', 'inspect', 'upload']
        saved = RunOutput.model_validate_json(result_file.read_text())
        assert saved.hook_results['video']['artifact']['status'] == 'ready'
        assert saved.hook_results['video']['source_deleted'] is True
        events.append('hold')

    monkeypatch.setattr(single, '_hold_sandbox', hold)
    output = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={
                'hold_after_task': 1,
                'early_output_path': str(result_file),
                'hooks': {
                    'video': {'name': 'gui_recording'},
                },
            },
        )
    )
    assert output.success
    assert events[-2:] == ['hold', 'release']


@pytest.mark.asyncio
@pytest.mark.parametrize('required', [True, False])
async def test_failed_start_does_not_stop_an_existing_recording(register_test_component, recording_runtime, required):
    runtime, events, _, client = recording_runtime
    runtime.gui_start_recording.return_value = GuiStartRecordingResponse(success=False, error='Already recording')
    config = RunInput(
        task_name=register_task(register_test_component, events),
        runner={'hooks': {'video': {'name': 'gui_recording', 'required': required}}},
    )
    if required:
        with pytest.raises(RuntimeError, match='Already recording'):
            await run(config)
        assert 'task' not in events
    else:
        output = await run(config)
        assert output.success
        assert output.error is None
        assert output.hook_errors['video.on_context_ready'] == 'RuntimeError: Already recording'
        assert 'task' in events
    runtime.gui_stop_recording.assert_not_awaited()
    client.create_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_upload_failure_retries_without_stopping_twice(register_test_component, recording_runtime):
    runtime, events, remote, client = recording_runtime
    original_script = remote.side_effect
    failed = False

    async def script(**kwargs):
        nonlocal failed
        if kwargs['action'] == 'upload' and not failed:
            failed = True
            raise OSError('upload failed')
        return await original_script(**kwargs)

    remote.side_effect = script
    output = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={'hooks': {'video': {'name': 'gui_recording'}}},
        )
    )
    assert output.success
    assert output.error is None
    assert output.hook_errors['video.on_run_output'] == 'OSError: upload failed'
    assert output.hook_results['video']['artifact']['status'] == 'ready'
    assert output.hook_results['video']['source_deleted'] is True
    assert client.create_upload.call_args_list[0] == client.create_upload.call_args_list[1]
    runtime.gui_stop_recording.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['complete', 'not_ready'])
async def test_recording_retries_confirmation_without_reading_deleted_source(
    register_test_component, recording_runtime, failure
):
    runtime, events, remote, client = recording_runtime
    if failure == 'complete':
        client.complete_upload.side_effect = [RuntimeError('completion failed'), None]
    else:
        ready = client.get_artifact.return_value
        client.get_artifact.side_effect = [replace(ready, status='pending'), ready]
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={'hooks': {'video': {'name': 'gui_recording', 'required': False}}},
        )
    )
    assert result.success
    assert result.hook_errors['video.on_run_output']
    assert result.hook_results['video']['source_deleted'] is True
    assert result.hook_results['video']['artifact']['status'] == 'ready'
    assert [call.kwargs['action'] for call in remote.call_args_list] == ['inspect', 'upload']
    client.create_upload.assert_awaited_once()
    runtime.gui_stop_recording.assert_awaited_once()


@pytest.mark.asyncio
async def test_multipart_upload_reports_each_part_etag(register_test_component, recording_runtime):
    _, events, remote, client = recording_runtime
    client.create_upload.return_value = UploadSession(
        artifact_id='artifact-1',
        upload_id='upload-1',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['https://cos.example/part-1', 'https://cos.example/part-2'],
        part_size=3,
        part_count=2,
    )
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={'hooks': {'video': {'name': 'gui_recording'}}},
        )
    )
    assert result.success
    uploads = [call.kwargs for call in remote.call_args_list if call.kwargs['action'] == 'upload']
    assert [call.kwargs['action'] for call in remote.call_args_list] == ['inspect', 'upload', 'upload']
    assert [(call['offset'], call['length']) for call in uploads] == [(0, 3), (3, 2)]
    assert [call['delete_source'] for call in uploads] == [False, True]
    client.complete_upload.assert_awaited_once_with(
        'upload-1',
        parts=[
            {'part_number': 1, 'etag': 'part-etag'},
            {'part_number': 2, 'etag': 'part-etag'},
        ],
    )


@pytest.mark.asyncio
async def test_ready_artifact_does_not_upload_again(register_test_component, recording_runtime):
    _, events, remote, client = recording_runtime
    client.create_upload.return_value = UploadSession(
        artifact_id='artifact-1',
        upload_id=None,
        status='ready',
        method=None,
        url=None,
        headers={},
    )
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={'hooks': {'video': {'name': 'gui_recording'}}},
        )
    )
    assert result.success
    assert [call.kwargs['action'] for call in remote.call_args_list] == ['inspect', 'upload']
    assert remote.call_args.kwargs == {'action': 'upload', 'path': '/tmp/video.mp4', 'already_uploaded': True}
    client.complete_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_part_etag_never_completes_artifact(register_test_component, recording_runtime):
    _, events, remote, client = recording_runtime
    client.create_upload.return_value = UploadSession(
        artifact_id='artifact-1',
        upload_id='upload-1',
        status='pending',
        method=None,
        url=None,
        headers={},
        part_urls=['https://cos.example/part-1'],
        part_size=5,
        part_count=1,
    )
    original_script = remote.side_effect

    async def script(**kwargs):
        if kwargs['action'] == 'upload':
            return {'etag': None}
        return await original_script(**kwargs)

    remote.side_effect = script
    result = await run(
        RunInput(
            task_name=register_task(register_test_component, events),
            runner={'hooks': {'video': {'name': 'gui_recording', 'args': {'storage_pool': 'test-pool'}}}},
        )
    )
    assert result.success
    assert result.error is None
    assert result.hook_errors['video.on_run_output'] == 'RuntimeError: COS multipart upload returned no ETag'
    assert 'source_deleted' not in result.hook_results['video']
    assert client.create_upload.call_args.kwargs['storage_pool'] == 'test-pool'
    assert result.hook_results['video']['artifact']['artifact_id'] == 'artifact-1'
    client.complete_upload.assert_not_awaited()
