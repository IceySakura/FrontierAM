"""Artifact-service publication and executable guest transport contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tarfile

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.core.context import ExecutionContext
from nexus.extensions.tasks.os_world.data import artifacts
from nexus.utils.artifact import ArtifactInfo, Folder, UploadSession


def run_guest(tmp_path, spec):
    helper = tmp_path / 'transfer.py'
    helper.write_text(artifacts._GUEST_TRANSFER)
    config = tmp_path / 'transfer.json'
    config.write_text(json.dumps(spec))
    result = subprocess.run(
        [sys.executable, '-I', str(helper), str(config)], capture_output=True, text=True, timeout=55
    )
    assert not config.exists()
    return result


def test_guest_snapshot_preserves_empty_files_and_exact_allowlist(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'stdout').write_text('父会话\nchild\n')
    (source / 'stderr').touch()
    (source / 'credentials').write_text('must never be archived')
    archive = tmp_path / 'snapshot.tar.gz'
    result = run_guest(
        tmp_path,
        {
            'action': 'snapshot',
            'archive': str(archive),
            'files': {name: str(source / name) for name in ('stdout', 'stderr', 'exit_code')},
        },
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(result.stdout)
    assert metadata['missing'] == ['exit_code']
    assert metadata['sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        assert bundle.getnames() == ['stdout', 'stderr']
        assert bundle.extractfile('stdout').read().decode() == '父会话\nchild\n'
        assert bundle.getmember('stderr').size == 0


def test_guest_rejects_symlink_and_redacts_exception(tmp_path):
    target = tmp_path / 'secret'
    target.write_text('secret')
    link = tmp_path / 'link'
    link.symlink_to(target)
    result = run_guest(
        tmp_path,
        {'action': 'snapshot', 'archive': str(tmp_path / 'out.tar.gz'), 'files': {'stdout': str(link)}},
    )
    assert result.returncode != 0
    assert set(json.loads(result.stdout)) == {'error_type', 'phase', 'http_status', 'errno', 'verify_code'}
    assert 'secret' not in result.stdout


def test_host_archive_skips_symlinks_and_preserves_empty(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'empty').touch()
    (source / 'sub').mkdir()
    (source / 'sub/log').write_text('full log')
    (source / 'link').symlink_to('/etc/passwd')
    (source / 'outside').symlink_to('/etc', target_is_directory=True)
    target = tmp_path / 'host.tar.gz'
    artifacts._pack_host_files(source, target)
    with tarfile.open(target) as archive:
        assert sorted(archive.getnames()) == ['empty', 'sub/log']


@pytest.mark.asyncio
@pytest.mark.parametrize('multipart', [False, True])
async def test_guest_upload_streams_bytes_without_host_relay(tmp_path, multipart):
    from aiohttp import web

    payload = b'0123456789' * 10000
    archive = tmp_path / 'snapshot.tar.gz'
    archive.write_bytes(payload)
    received = []

    async def put(request):
        if multipart:
            assert 'X-Test' not in request.headers
        else:
            assert request.headers['X-Test'] == 'presigned'
        received.append(await request.read())
        return web.Response(headers={'ETag': '"etag"'})

    app = web.Application()
    app.router.add_put('/part/{number}', put)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        urls = [f'http://127.0.0.1:{port}/part/{n}' for n in range(2 if multipart else 1)]
        import base64

        spec = {
            'action': 'upload',
            'archive': str(archive),
            'digest': {
                'size': len(payload),
                'md5': base64.b64encode(hashlib.md5(payload).digest()).decode(),
                'sha256': hashlib.sha256(payload).hexdigest(),
            },
            'upload': {
                'url': None if multipart else urls[0],
                'headers': {'X-Test': 'presigned'},
                'part_urls': urls if multipart else [],
                'part_size': len(payload) // 2 if multipart else None,
            },
        }
        result = await asyncio.to_thread(run_guest, tmp_path, spec)
        assert result.returncode == 0, result.stdout + result.stderr
        assert b''.join(received) == payload
        assert len(json.loads(result.stdout)['parts']) == (2 if multipart else 0)
        spec['digest']['sha256'] = 'bad'
        result = await asyncio.to_thread(run_guest, tmp_path, spec)
        assert result.returncode != 0
        assert len(received) == (2 if multipart else 1)
    finally:
        await runner.cleanup()


@pytest.fixture
def artifact_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, 'NEXUS_HOME', tmp_path)
    stage = artifacts.RunArtifacts()
    stage.finish(task_id='case', run_id='run', result={'score': 0.75})
    return stage


@pytest.fixture
def client(monkeypatch):
    import nexus.utils.artifact

    mock = AsyncMock()
    mock.__aenter__.return_value = mock
    mock.whoami.return_value = 'tester'
    mock.create_folder.return_value = Folder('folder')
    mock.upload_file.return_value = ArtifactInfo('host', status='ready')
    mock.get_artifact.return_value = ArtifactInfo('guest', status='ready')
    monkeypatch.setattr(nexus.utils.artifact, 'ArtifactClient', lambda **kwargs: mock)
    monkeypatch.setattr(artifacts.asyncio, 'sleep', AsyncMock())
    return mock


@pytest.mark.asyncio
async def test_partial_snapshot_keeps_host_uri_without_claiming_guest_ready(artifact_stage, client):
    artifact_stage.remote.append({'name': 'guest.tar.gz', 'status': 'snapshot_failed'})
    result = await artifact_stage.publish()
    assert result['status'] == 'partial'
    assert result['upload_failed'] is True
    assert len(result['uris']) == 1
    assert result['uris'][0].endswith('/host.tar.gz')
    client.create_upload.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_remote_retry_keeps_host_result_and_folder(artifact_stage, client, monkeypatch):
    entry = {'name': 'guest.tar.gz', 'status': 'staged'}
    artifact_stage.remote.append(entry)
    transfer = AsyncMock(side_effect=[TimeoutError(), None])
    monkeypatch.setattr(artifacts, '_publish_remote', transfer)
    result = await artifact_stage.publish()
    assert result['status'] == 'published'
    assert len(result['uris']) == 2
    assert result['attempts'] == 2
    client.upload_file.assert_awaited_once()
    assert client.create_folder.await_count == 5  # Four date segments, one unique run.
    assert transfer.await_count == 2


@pytest.mark.asyncio
async def test_failure_preserves_partial_uris_and_never_logs_signed_url(artifact_stage, client, monkeypatch, caplog):
    artifact_stage.remote.append({'name': 'guest.tar.gz', 'status': 'staged'})
    monkeypatch.setattr(artifacts, '_publish_remote', AsyncMock(side_effect=OSError('https://signed?secret=token')))
    result = await artifact_stage.publish()
    assert result['status'] == 'upload_failed'
    assert result['attempts'] == 4
    assert len(result['uris']) == 1
    assert 'secret=token' not in caplog.text
    assert json.loads((artifact_stage.files / 'run-result.json').read_text())['result']['score'] == 0.75


@pytest.mark.asyncio
async def test_publish_remote_finalizes_on_host_and_requires_ready(monkeypatch, client):
    entry = {
        'name': 'guest.tar.gz',
        'status': 'staged',
        'size': 10,
        'md5': 'md5',
        'sha256': 'sha',
        'runtime': object(),
        'directory': '/tmp/output',
        'archive': '/tmp/output/log.tar.gz',
    }
    client.create_upload.return_value = UploadSession('guest', 'upload', 'pending', 'PUT', 'https://signed', {})
    transfer = AsyncMock(return_value={'parts': []})
    monkeypatch.setattr(artifacts, '_guest_transfer', transfer)
    await artifacts._publish_remote(client, 'folder', entry, 'run')
    client.complete_upload.assert_awaited_once_with('upload', parts=None)
    assert 'pat' not in json.dumps(transfer.call_args.args[2])
    client.get_artifact.return_value = ArtifactInfo('guest', status='pending')
    with pytest.raises(RuntimeError, match='not ready'):
        await artifacts._publish_remote(client, 'folder', entry, 'run')


@pytest.mark.asyncio
async def test_no_task_publisher_does_not_access_runtime():
    runtime = SimpleNamespace(run_command=AsyncMock(), setup_upload_bytes=AsyncMock())
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}):
        await artifacts.snapshot_sandbox_artifacts(runtime, {'stdout': '/tmp/stdout'}, label='cc')
    runtime.run_command.assert_not_awaited()
    runtime.setup_upload_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_enabled_snapshot_is_best_effort_and_does_not_accept_guest_field_injection(artifact_stage, monkeypatch):
    runtime = SimpleNamespace(run_command=AsyncMock(), setup_upload_bytes=AsyncMock())
    runtime.run_command.return_value = SimpleNamespace(return_code=0, stdout='')
    runtime.setup_upload_bytes.return_value = SimpleNamespace(status_code=200)
    monkeypatch.setattr(
        artifacts,
        '_guest_transfer',
        AsyncMock(return_value={'size': 1, 'md5': 'bad', 'sha256': 'bad', 'status': 'published', 'runtime': 'evil'}),
    )
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}) as ctx:
        ctx.tool_state['osworld_artifact_publisher'] = artifact_stage
        await artifacts.snapshot_sandbox_artifacts(runtime, {'stdout': '/tmp/stdout'}, label='cc')
    assert artifact_stage.remote[0]['status'] == 'snapshot_failed'
    assert artifact_stage.remote[0]['runtime'] is runtime


@pytest.mark.asyncio
async def test_cc_snapshot_precedes_cleanup_and_failure_does_not_skip_cleanup(monkeypatch):
    from nexus.extensions.agents.claude_code.native_bridge.adapters.posix import BasePosixClaudeNativeBridgeAdapter
    from nexus.extensions.tasks.os_world.execution.claude_code import OSWorldClaudeNativeBridgeAdapter

    adapter = object.__new__(OSWorldClaudeNativeBridgeAdapter)
    adapter.runtime = object()
    paths = SimpleNamespace(stdout_log='stdout', stderr_log='stderr', exit_code='exit', pid='pid')
    calls = []

    async def snapshot(*args, **kwargs):
        calls.append('snapshot')
        raise asyncio.CancelledError()

    async def cleanup(*args, **kwargs):
        calls.append(('cleanup', kwargs['terminate']))

    monkeypatch.setattr(artifacts, 'snapshot_sandbox_artifacts', snapshot)
    monkeypatch.setattr(BasePosixClaudeNativeBridgeAdapter, '_cleanup_session', cleanup)
    with pytest.raises(asyncio.CancelledError):
        await adapter._cleanup_session(paths, terminate=True)
    assert calls == ['snapshot', ('cleanup', True)]


@pytest.mark.asyncio
async def test_artifact_cancel_does_not_retry(artifact_stage, client):
    client.whoami.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await artifact_stage.publish()
    assert artifact_stage.reference['attempts'] == 1


@pytest.mark.asyncio
async def test_task_returns_standard_artifact_list(artifact_stage, client, monkeypatch):
    from nexus.extensions.tasks.os_world import task

    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda *args, **kwargs: artifact_stage)
    result = task.OSWorldTaskOutput(completed=True, task_id='case', domain='os', instruction='test', score=1)
    monkeypatch.setattr(task, '_run_os_world', AsyncMock(return_value=result))
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}):
        actual = await task.run_os_world(task.OSWorldTaskInput(task_id='case', run_agent=AsyncMock()))
    assert actual.score == 1
    assert len(actual.artifacts['uris']) == 1
    assert actual.artifacts['uris'][0].startswith('artifact://')
    assert 'artifact' not in actual.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('upload_failed', [False, True])
async def test_run_output_archives_large_details_and_stays_below_one_mib(
    artifact_stage,
    client,
    monkeypatch,
    tmp_path,
    failed,
    upload_failed,
):
    from nexus.core import require_context
    from nexus.extensions.tasks.os_world import task
    from nexus.extensions.tasks.os_world.outputs import OSWorldAgentOutput, OSWorldOutputContractHook
    from nexus.runner.single import ErrorInfo, RunOutput, write_run_output

    blob = '轨迹\n' * 400_000
    counters = {
        'main': {
            'prompt_tokens': 260570,
            'completion_tokens': 1041,
            'reasoning_tokens': 0,
            'llm_call_count': 11,
            'total_tokens': 28804,
            'total_tokens_per_trajectory': [28804],
            'cached_prompt_tokens': 246720,
            'cost': 0.123456,
        },
        'aux': {'prompt_tokens': 15, 'completion_tokens': 7, 'llm_call_count': 1},
    }
    anomalies = {
        'main': {
            'total_requests': 11,
            'total_steps': 10,
            'error_requests': 2,
            'retry_requests': 1,
            'error_steps': 2,
            'retry_steps': 1,
            'request_error_rate': 2 / 11,
            'anomaly_counts': {'HALLUCINATED_TOOL': 2},
            'requests': [{'response': blob}],
        }
    }
    timing = {
        'credited_waste_seconds': 2.125,
        'budget_seconds': 1800,
        'effective_seconds': 53.9,
        'wrap_target': 'agent_phase',
        'termination_reason': None,
    }
    agent = OSWorldAgentOutput.model_validate({
        'exit_status': 'runtime_error_by_env' if failed else 'finished',
        'exit_code': -1 if failed else 0,
        'process_exit_code': -1 if failed else 0,
        'error': 'actual error ' + blob if failed else None,
        'output': blob,
        'stderr': blob,
        'main_trajectory': [{'text': blob}],
        'final_answer': blob,
    })
    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **kwargs: artifact_stage)
    # Restore the real constructor inside the late publisher, which creates a
    # separate immutable archive after task publication.
    factory = type(artifact_stage)
    if upload_failed:
        client.upload_file.side_effect = OSError('service unavailable')
    primary = RuntimeError('original failure')

    async def execute(_):
        ctx = require_context()
        ctx.tool_state['osworld_agent_output'] = agent
        if failed:
            raise primary
        return task.OSWorldTaskOutput(
            completed=True,
            task_id='case',
            domain='os',
            instruction=blob,
            score=1,
            agent_output=agent,
            setup_results=[{'output': blob}],
            evaluator_result={'score': 1, 'observed': blob},
            judge_output={'exit_status': 'finished', 'exit_code': 0, 'result': {'observed': blob}},
        )

    monkeypatch.setattr(task, '_run_os_world', execute)
    ctx = ExecutionContext(task_name='os_world')
    async with ctx:
        inp = task.OSWorldTaskInput(task_id='case', run_agent=AsyncMock())
        if failed:
            with pytest.raises(RuntimeError) as caught:
                await task.run_os_world(inp)
            assert caught.value is primary
            compact_task = ctx.tool_state['osworld_partial_task_output']
            assert json.loads((artifact_stage.files / 'agent-output.json').read_text())['output'] == blob
        else:
            compact_task = await task.run_os_world(inp)
            saved = json.loads((artifact_stage.files / 'run-result.json').read_text())
            assert saved['result']['agent_output']['output'] == blob
            assert saved['result']['judge_output']['result']['observed'] == blob
        original_host = (artifact_stage.root / 'host.tar.gz').read_bytes()
        monkeypatch.setattr(artifacts, 'RunArtifacts', factory)
        raw = RunOutput(
            task_output=compact_task,
            agent_output=agent,
            processor_results={
                'chat': {'messages': [{'content': blob}]},
                '_default_atif_trace': {'atif': {'steps': [{'message': blob}]}},
                'token_counter': counters,
                'bridge_toolcall_anomaly': anomalies,
                'execution_timeout_tracker': timing,
            },
            success=not failed,
            error=ErrorInfo(type='RuntimeError', message='original failure', traceback=blob) if failed else None,
        )
        assert len(raw.model_dump_json().encode()) > 6_669_521
        hook = OSWorldOutputContractHook()
        await hook.on_run_output(ctx, raw)
        await hook.on_context_closing(ctx)
    destination = tmp_path / 'single.json'
    write_run_output(destination, raw)
    data = json.loads(destination.read_text())
    # Local output carries pending bridge payloads intact. The response-size
    # contract applies after bridge persistence consumes these fields.
    pending_atif = data['processor_results'].pop('_default_atif_trace')
    assert pending_atif == {'atif': {'steps': [{'message': blob}]}}
    assert len(json.dumps(data).encode()) < 1_048_576
    assert data['success'] is not failed
    assert data['task_output']['status'] == ('agent_error' if failed else 'success')
    assert data['agent_output']['exit_code'] == (-1 if failed else 0)
    assert 'main_trajectory' not in data['agent_output']
    assert 'output' not in data['task_output']['agent_output']
    stats = data['processor_results']
    assert set(stats) == {'token_counter', 'bridge_toolcall_anomaly', 'execution_timeout_tracker'}
    # These are the original keys read by the platform bridge after run returns.
    assert stats.get('token_counter') == counters
    assert stats['execution_timeout_tracker'] == timing
    assert stats['bridge_toolcall_anomaly'] == {
        'main': {key: value for key, value in anomalies['main'].items() if key != 'requests'}
    }
    ref = data['task_output']['artifacts']['runner_details']
    assert ref['upload_failed'] is upload_failed
    from pathlib import Path

    details = json.loads((Path(ref['local_directory']) / 'files' / 'run-result.json').read_text())['result']
    assert details['processor_results']['chat']['messages'][0]['content'] == blob
    assert details['processor_results']['token_counter'] == counters
    assert details['processor_results']['bridge_toolcall_anomaly'] == anomalies
    assert details['processor_results']['execution_timeout_tracker'] == timing
    assert details['agent_output']['output'] == blob
    assert (artifact_stage.root / 'host.tar.gz').read_bytes() == original_host
    assert 'osworld_output_publisher' not in ctx.tool_state


@pytest.mark.asyncio
async def test_output_archive_disk_failure_is_nonfatal(monkeypatch):
    from nexus.extensions.tasks.os_world.outputs import OSWorldOutputContractHook
    from nexus.runner.single import RunOutput

    publisher = SimpleNamespace(publish_run_output_details=AsyncMock(side_effect=OSError('disk full')))
    ctx = SimpleNamespace(task_name='os_world', tool_state={'osworld_output_publisher': publisher})
    output = RunOutput(
        success=True,
        task_output={
            'status': 'success',
            'score': 0.5,
            'agent_output': {'exit_status': 'finished', 'exit_code': 0, 'main_trajectory': ['large']},
            'artifacts': {'status': 'published', 'uri': 'artifact://test/task', 'upload_failed': False},
        },
        processor_results={'chat': ['large']},
    )
    await OSWorldOutputContractHook().on_run_output(ctx, output)
    assert output.success
    assert output.task_output['score'] == 0.5
    assert output.task_output['status'] == 'success'
    assert output.task_output['artifacts']['upload_failed']
    assert output.task_output['artifacts']['runner_details']['status'] == 'archive_failed'
    assert output.processor_results == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('upload_failed', [False, True])
async def test_runner_failure_keeps_structured_artifacts_and_original_error(
    artifact_stage, client, monkeypatch, upload_failed
):
    from nexus.extensions.tasks.os_world import task
    from nexus.runner.single import RunHooks, RunInput, run

    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **kwargs: artifact_stage)
    monkeypatch.setattr(task, '_run_os_world', AsyncMock(side_effect=ValueError('original evaluator failure')))
    if upload_failed:
        client.upload_file.side_effect = OSError('unavailable')
    result = await run(
        RunInput(
            task_name='os_world',
            task_config={'task_id': 'case'},
            runner={'hooks': {'output_contract': {'name': 'osworld_output_contract'}}},
        ),
        hooks=RunHooks(task_input_overrides={'run_agent': AsyncMock()}),
    )
    assert result.success is False
    assert result.error.type == 'ValueError'
    assert result.error.message == 'original evaluator failure'
    output = result.model_dump(mode='json')['task_output']
    assert output['completed'] is False
    assert output['score'] is None
    assert output['artifacts']['upload_failed'] is upload_failed
    assert bool(output['artifacts']['uris']) is not upload_failed
    assert 'artifact' not in output
    # Platform nests this unchanged under trial_details.run_output_data.
    assert 'artifacts' in {'run_output_data': result.model_dump()}['run_output_data']['task_output']


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['success', 'agent_error', 'judge_error', 'timeout', 'deadline', 'cancel'])
@pytest.mark.parametrize('enabled', [False, True])
async def test_task_partial_result_is_hook_owned_and_preserves_runner_semantics(
    artifact_stage,
    monkeypatch,
    outcome,
    enabled,
):
    from nexus.core import require_context
    from nexus.extensions.tasks.os_world import task
    from nexus.extensions.tasks.os_world.outputs import OSWorldAgentOutput
    from nexus.runner.single import RunHooks, RunInput, run

    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **kwargs: artifact_stage)
    artifact_stage.publish = AsyncMock(return_value=artifact_stage.reference)
    archived = []
    contexts = []

    async def archive(output):
        archived.append(output.model_dump(mode='json'))
        return artifact_stage.reference

    artifact_stage.publish_run_output_details = archive

    async def execute(_):
        ctx = require_context()
        contexts.append(ctx)
        assert 'osworld_partial_task_output' not in ctx.tool_state
        if outcome == 'success':
            return task.OSWorldTaskOutput(completed=True, score=0.5, task_id='case', domain='os', instruction='')
        ctx.tool_state['osworld_output_stage'] = 'judge' if outcome == 'judge_error' else 'agent'
        if outcome == 'judge_error':
            ctx.tool_state['osworld_agent_output'] = OSWorldAgentOutput(exit_status='finished', exit_code=0)
        if outcome == 'cancel':
            raise asyncio.CancelledError()
        if outcome == 'timeout':
            raise TimeoutError('original timeout')
        if outcome == 'deadline':
            await asyncio.sleep(10)
        raise ValueError('original failure')

    async def ready(ctx):
        ctx.tool_state['osworld_partial_task_output'] = {'stale': True}

    monkeypatch.setattr(task, '_run_os_world', execute)
    config = RunInput(
        task_name='os_world',
        task_config={'task_id': 'case'},
        runner={
            **({'hooks': {'output_contract': {'name': 'osworld_output_contract'}}} if enabled else {}),
            **({'timeout': 0.05} if outcome == 'deadline' else {}),
        },
    )
    hooks = RunHooks(on_context_ready=ready, task_input_overrides={'run_agent': AsyncMock()})
    if outcome == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await run(config, hooks=hooks)
        assert not archived
    else:
        result = await run(config, hooks=hooks)
        assert result.success is (outcome == 'success')
        if outcome == 'success':
            data = result.task_output
            assert (data['score'] if isinstance(data, dict) else data.score) == 0.5
            assert result.error is None
        else:
            # A nested TimeoutError is not the runner deadline; only 'deadline' expires runner.timeout.
            expected = {'timeout': 'OSWorldInnerTimeoutError', 'deadline': 'TimeoutError'}.get(outcome, 'ValueError')
            assert result.error.type == expected
            if outcome == 'timeout':
                assert 'timed out before the task deadline' in result.error.message
            elif outcome == 'deadline':
                assert result.error.message == 'Task execution timed out after 0.05 seconds'
            else:
                assert result.error.message == 'original failure'
            if enabled:
                assert result.task_output['status'] == ('judge_error' if outcome == 'judge_error' else 'agent_error')
                assert result.task_output['score'] is None
                # Restore before archival, not just before serializing the final output.
                assert archived[0]['task_output']['status'] == result.task_output['status']
                assert archived[0]['error']['type'] == result.error.type
            else:
                assert result.task_output is None
    if enabled:
        assert 'osworld_partial_task_output' not in contexts[0].tool_state
        assert 'osworld_output_publisher' not in contexts[0].tool_state


@pytest.mark.asyncio
async def test_output_hook_does_not_overwrite_existing_result_or_other_task_state():
    from nexus.extensions.tasks.os_world.outputs import OSWorldOutputContractHook

    hook = OSWorldOutputContractHook()
    ctx = SimpleNamespace(task_name='os_world', tool_state={'osworld_partial_task_output': {'status': 'agent_error'}})
    output = SimpleNamespace(success=False, task_output={'status': 'judge_error', 'score': None}, agent_output=None)
    await hook.on_run_output(ctx, output)
    assert output.task_output['status'] == 'judge_error'
    assert 'osworld_partial_task_output' not in ctx.tool_state
    ctx.task_name = 'other_task'
    ctx.tool_state['osworld_partial_task_output'] = {'status': 'agent_error'}
    output.task_output = None
    await hook.on_run_output(ctx, output)
    await hook.on_context_closing(ctx)
    assert output.task_output is None
    assert 'osworld_partial_task_output' in ctx.tool_state


@pytest.mark.asyncio
@pytest.mark.parametrize('score', [0, 0.5, 1])
async def test_publication_failure_preserves_score(artifact_stage, monkeypatch, score):
    from nexus.extensions.tasks.os_world import task

    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **kwargs: artifact_stage)
    artifact_stage.publish = AsyncMock(side_effect=RuntimeError('upload failed'))
    result = task.OSWorldTaskOutput(completed=True, task_id='case', domain='os', instruction='test', score=score)
    monkeypatch.setattr(task, '_run_os_world', AsyncMock(return_value=result))
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}):
        actual = await task.run_os_world(task.OSWorldTaskInput(task_id='case', run_agent=AsyncMock()))
    assert actual.score == score
    assert actual.error is None
    assert actual.artifacts['upload_failed']
    assert json.loads((artifact_stage.files / 'run-result.json').read_text())['result']['score'] == score
    assert not (artifact_stage.root / 'dataset.yaml').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('primary', [RuntimeError('agent failed'), asyncio.CancelledError()])
async def test_publication_does_not_replace_primary(artifact_stage, monkeypatch, primary):
    from nexus.extensions.tasks.os_world import task

    monkeypatch.setattr(artifacts, 'RunArtifacts', lambda **kwargs: artifact_stage)
    artifact_stage.publish = AsyncMock(side_effect=RuntimeError('upload failed'))
    monkeypatch.setattr(task, '_run_os_world', AsyncMock(side_effect=primary))
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}):
        with pytest.raises(type(primary)) as caught:
            await task.run_os_world(task.OSWorldTaskInput(task_id='case', run_agent=AsyncMock()))
    assert caught.value is primary
    artifact_stage.publish.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failures', [1, 2, 3, 4])
async def test_host_upload_retries_with_same_idempotency_key(artifact_stage, client, failures):
    client.upload_file.side_effect = [TimeoutError()] * failures + [ArtifactInfo('host', status='ready')]
    result = await artifact_stage.publish()
    assert result['attempts'] == min(failures + 1, 4)
    assert result['upload_failed'] is (failures == 4)
    calls = client.upload_file.await_args_list
    assert all(call == calls[0] for call in calls)


@pytest.mark.asyncio
async def test_platform_credentials_are_per_run_not_process_global(artifact_stage, monkeypatch):
    import os

    from nexus.extensions.tasks.os_world import task
    from nexus.runner.single import RunInput

    monkeypatch.setenv('TEST_RAINBOW_ARTIFACT_PAT', 'per-run-secret')
    cfg = RunInput.from_dict({
        'task_name': 'os_world',
        'task_config': {
            'artifact_taihu_pat': '${oc.env:TEST_RAINBOW_ARTIFACT_PAT}',
            'artifact_service_url': 'https://artifact.example.test',
        },
    })
    inp = task.OSWorldTaskInput(**cfg.task_config, run_agent=AsyncMock())
    assert 'per-run-secret' not in repr(inp)
    assert 'artifact_taihu_pat' not in inp.model_dump()
    observed = {}

    def publisher(**kwargs):
        observed.update(kwargs)
        return artifact_stage

    monkeypatch.setattr(artifacts, 'RunArtifacts', publisher)
    artifact_stage.publish = AsyncMock(return_value=artifact_stage.reference)
    result = task.OSWorldTaskOutput(completed=True, task_id='case', domain='os', instruction='test', score=1)
    monkeypatch.setattr(task, '_run_os_world', AsyncMock(return_value=result))
    before = dict(os.environ)
    async with ExecutionContext(task_name='os_world', runtime_configs={}, tool_configs={}):
        await task.run_os_world(inp)
    assert observed == {'taihu_pat': 'per-run-secret', 'service_url': 'https://artifact.example.test'}
    assert os.environ == before
    assert 'per-run-secret' not in (artifact_stage.files / 'run-result.json').read_text()


@pytest.mark.asyncio
async def test_direct_failure_relays_same_snapshot_and_idempotency_key(monkeypatch, client, tmp_path):
    entry = {
        'name': 'guest.tar.gz',
        'status': 'staged',
        'size': 10,
        'md5': 'md5',
        'sha256': 'sha',
        'runtime': object(),
        'directory': '/tmp/out',
        'archive': '/tmp/out/log.tar.gz',
    }
    client.create_upload.return_value = UploadSession('guest', 'upload', 'pending', 'PUT', 'https://signed', {})
    monkeypatch.setattr(artifacts, '_guest_transfer', AsyncMock(side_effect=RuntimeError('TLS rejected')))
    local = tmp_path / 'guest.tar.gz'
    relay = AsyncMock(return_value=local)
    monkeypatch.setattr(artifacts, '_relay_snapshot', relay)
    await artifacts._publish_remote(client, 'folder', entry, 'run')
    assert entry['transfer'] == 'host_relay'
    assert (
        client.upload_file.call_args.kwargs['idempotency_key']
        == client.create_upload.call_args.kwargs['idempotency_key']
    )
    assert client.upload_file.call_args.args == (local,)
    client.complete_upload.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('corrupt', [False, True])
async def test_relay_stream_checks_snapshot_and_removes_partial(tmp_path, corrupt):
    from aiohttp import web

    payload = b'log' * 1024 * 1024

    async def download(request):
        assert (await request.post())['file_path'] == '/tmp/snapshot'
        return web.Response(body=payload)

    app = web.Application()
    app.router.add_post('/file', download)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    try:
        port = site._server.sockets[0].getsockname()[1]
        runtime = SimpleNamespace(
            server_port=5000,
            handle=SimpleNamespace(
                headers={},
                url=lambda *_: f'http://127.0.0.1:{port}/file',
            ),
        )
        dest = tmp_path / 'snapshot.tar.gz'
        entry = {
            'runtime': runtime,
            'archive': '/tmp/snapshot',
            'local_archive': str(dest),
            'size': len(payload),
            'sha256': 'bad' if corrupt else hashlib.sha256(payload).hexdigest(),
        }
        if corrupt:
            with pytest.raises(RuntimeError, match='checksum mismatch'):
                await artifacts._relay_snapshot(entry)
            assert not dest.exists()
        else:
            assert await artifacts._relay_snapshot(entry) == dest
            assert dest.read_bytes() == payload
        assert not dest.with_suffix('.part').exists()
    finally:
        await runner.cleanup()
