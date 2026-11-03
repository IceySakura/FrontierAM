"""Task attribution and owned-grader process tests, without Docker or models."""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from pydantic import ValidationError

from nexus.extensions.tasks.frontieram_offline import grading, task


class SubmissionFailure(ValueError):  # noqa: N818 - matches the frozen controller exception contract.
    pass


@pytest.fixture
def task_fixture(tmp_path, monkeypatch):
    assets = SimpleNamespace(
        provenance=Mock(return_value={'image_id': 'frozen-fixture'}),
        verify_private_data=Mock(),
        controller=SimpleNamespace(SubmissionFailure=SubmissionFailure),
    )
    workspace = SimpleNamespace(
        assets=assets,
        inspect=AsyncMock(return_value={'Id': 'a' * 64}),
        configuration_checks={'network_none': True},
        container='nexus-frontieram-fixture',
        freeze=AsyncMock(return_value={'sha256': 'frozen-submission'}),
    )
    provider_type = type('FrontierAMOfflineProvider', (SimpleNamespace,), {})
    provider = provider_type(configure=Mock(), release=AsyncMock(return_value=True))
    context = SimpleNamespace(runtime_provider=provider, get_runtime=AsyncMock(return_value=workspace))
    monkeypatch.setattr(task, 'require_context', lambda: context)
    run_agent = AsyncMock(
        return_value=SimpleNamespace(
            evaluation_eligible=True,
            exit_status='finished',
            stop_reason='turn_completed',
        )
    )
    grade = AsyncMock(
        return_value={
            'exit_status': 'finished',
            'score': 42.5,
            'valid_runs': 69,
            'total_runs': 69,
        }
    )
    monkeypatch.setattr(task, 'grade', grade)
    value = task.OfflineTaskInput(output_dir=str(tmp_path / 'attempt'), run_agent=run_agent)
    return SimpleNamespace(
        input=value,
        runtime=workspace,
        provider=provider,
        context=context,
        run_agent=run_agent,
        grade=grade,
    )


@pytest.mark.asyncio
async def test_normal_completed_submission_runs_grader_and_writes_result(task_fixture):
    f = task_fixture
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('success', 'finished', 42.5)
    assert (result.valid_runs, result.total_runs) == (69, 69)
    f.run_agent.assert_awaited_once()
    f.runtime.freeze.assert_awaited_once()
    f.grade.assert_awaited_once()
    f.provider.release.assert_awaited_once_with('main', force=True)
    record = json.loads((Path(f.input.output_dir) / 'task_result.json').read_text())
    assert record['score'] == 42.5


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['empty submission', 'symlink in submission', 'submission exceeds artifact limit'])
async def test_normal_agent_with_invalid_source_gets_zero(task_fixture, invalid):
    f = task_fixture
    f.runtime.freeze.side_effect = SubmissionFailure(invalid)
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('invalid_submission', 'finished', 0.0)
    assert result.reason == 'SubmissionFailure'
    f.grade.assert_not_awaited()
    f.provider.release.assert_awaited_once_with('main', force=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('freeze_error', [None, SubmissionFailure('empty submission'), RuntimeError('export failed')])
async def test_agent_interruption_takes_precedence_over_submission_or_freeze_error(task_fixture, freeze_error):
    f = task_fixture
    f.run_agent.return_value = SimpleNamespace(
        evaluation_eligible=False,
        exit_status='inference_error',
        stop_reason='upstream_http',
    )
    f.runtime.freeze.side_effect = freeze_error
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('unmeasured', 'inference_error', None)
    assert result.reason == 'agent_or_inference_interrupted'
    f.grade.assert_not_awaited()
    f.provider.release.assert_awaited_once_with('main', force=True)


@pytest.mark.asyncio
async def test_infrastructure_export_failure_is_unmeasured(task_fixture):
    f = task_fixture
    f.runtime.freeze.side_effect = RuntimeError('synthetic exporter unavailable')
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('unmeasured', 'runtime_error_by_env', None)
    f.grade.assert_not_awaited()
    f.provider.release.assert_awaited_once_with('main', force=True)


@pytest.mark.asyncio
async def test_infrastructure_grader_failure_is_unmeasured(task_fixture):
    f = task_fixture
    f.grade.return_value = {'exit_status': 'runtime_error_by_env', 'score': None, 'reason': 'worker_failed'}
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('unmeasured', 'runtime_error_by_env', None)
    assert result.reason == 'worker_failed'


@pytest.mark.asyncio
async def test_runtime_setup_failure_cannot_become_candidate_zero(task_fixture):
    f = task_fixture
    f.context.get_runtime.side_effect = RuntimeError('synthetic unavailable daemon')
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('unmeasured', 'runtime_error_by_env', None)
    f.run_agent.assert_not_awaited()
    f.grade.assert_not_awaited()
    f.provider.release.assert_awaited_once_with('main', force=True)


@pytest.mark.asyncio
async def test_non_scoring_verification_does_not_create_a_score(task_fixture):
    f = task_fixture
    f.input.evaluate = False
    result = await task.exponential_offline_task(f.input)
    assert result.status == 'verification_completed'
    assert result.score is None
    f.grade.assert_not_awaited()


def test_unknown_task_field_is_rejected_instead_of_silently_ignored(tmp_path):
    with pytest.raises(ValidationError, match='grading_timout'):
        task.OfflineTaskInput.model_validate(
            {'output_dir': str(tmp_path / 'attempt'), 'run_agent': AsyncMock(), 'grading_timout': 1}
        )


@pytest.mark.asyncio
async def test_task_cancellation_releases_its_runtime(task_fixture):
    f = task_fixture
    f.run_agent.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await task.exponential_offline_task(f.input)
    f.provider.release.assert_awaited_once_with('main', force=True)
    f.grade.assert_not_awaited()


class WaitingChild:
    def __init__(self):
        self.pid = 24680
        self.returncode = None
        self.started = asyncio.Event()
        self.exited = asyncio.Event()

    async def wait(self):
        self.started.set()
        await self.exited.wait()
        return self.returncode

    def terminate(self, _pid, _signal):
        self.returncode = -15
        self.exited.set()


@pytest.mark.asyncio
async def test_cancelled_grader_terminates_its_process_group_and_cleans_ledger(tmp_path, monkeypatch):
    child = WaitingChild()
    spawn = AsyncMock(return_value=child)
    cleanup = AsyncMock()
    kill = Mock(side_effect=child.terminate)
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(grading, 'cleanup_owned', cleanup)
    monkeypatch.setattr(grading.os, 'killpg', kill)
    environment = SimpleNamespace(
        assets=SimpleNamespace(source_root=tmp_path / 'frozen'),
        docker=SimpleNamespace(prefix=['/trusted/docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
    )
    pending = asyncio.create_task(grading.grade(environment, tmp_path))
    await child.started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert kill.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    cleanup.assert_awaited_once()
    assert cleanup.await_args is not None
    assert spawn.call_args is not None
    assert cleanup.await_args.args[0]['ledger'] == str(tmp_path / 'grading_owned_resources.jsonl')
    assert spawn.call_args.kwargs['start_new_session'] is True


@pytest.mark.asyncio
async def test_timed_out_grader_is_stopped_before_task_classifies_environment_failure(tmp_path, monkeypatch):
    child = WaitingChild()
    cleanup = AsyncMock()
    kill = Mock(side_effect=child.terminate)
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', AsyncMock(return_value=child))
    monkeypatch.setattr(grading, 'cleanup_owned', cleanup)
    monkeypatch.setattr(grading.os, 'killpg', kill)
    environment = SimpleNamespace(
        assets=SimpleNamespace(source_root=tmp_path / 'frozen'),
        docker=SimpleNamespace(prefix=['/trusted/docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
    )
    with pytest.raises(asyncio.TimeoutError):
        await grading.grade(environment, tmp_path, timeout=0.01)
    assert kill.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_failure_downgrades_the_primary_score_but_preserves_diagnostic(task_fixture):
    f = task_fixture
    f.provider.release.side_effect = RuntimeError('synthetic deletion failure')
    result = await task.exponential_offline_task(f.input)
    assert (result.status, result.exit_status, result.score) == ('unmeasured', 'runtime_error_by_env', None)
    assert result.reason == 'cleanup_failure:RuntimeError'
    assert result.provenance['result_before_cleanup_failure'] == {'exit_status': 'finished', 'score': 42.5}
    assert (Path(f.input.output_dir) / 'cleanup_error.json').exists()
    assert (Path(f.input.output_dir) / 'task_result.json').exists()


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_task_cancellation(task_fixture):
    f = task_fixture
    f.run_agent.side_effect = asyncio.CancelledError()
    f.provider.release.side_effect = RuntimeError('synthetic deletion failure')
    with pytest.raises(asyncio.CancelledError):
        await task.exponential_offline_task(f.input)
    assert (Path(f.input.output_dir) / 'cleanup_error.json').exists()


class ImmediateChild:
    def __init__(self, stdout=b'', returncode=0, stderr=b''):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self.stdout, self.stderr

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_ledger_cleanup_removes_only_matching_owner_full_ids_and_never_base(tmp_path, monkeypatch):
    ledger = tmp_path / 'owned.jsonl'
    records = [
        {'kind': 'container', 'value': 'own-name'},
        {'kind': 'container', 'value': 'foreign-name'},
        {'kind': 'image', 'value': 'own-image-tag'},
        {'kind': 'image', 'value': grading.IMAGE_ID},
        {'kind': 'volume', 'value': 'unknown-kind'},
        {'kind': 'image', 'value': 'alias-for-base'},
    ]
    ledger.write_text(''.join(json.dumps(record) + '\n' for record in records))
    owner = 'local-owner'
    own_container = 'a' * 64
    own_image = 'sha256:' + 'b' * 64
    inspect_results = {
        'own-name': {'Id': own_container, 'Config': {'Labels': {'nexus.frontieram.owner': owner}}},
        'foreign-name': {'Id': 'c' * 64, 'Config': {'Labels': {'nexus.frontieram.owner': 'another-owner'}}},
        'own-image-tag': {'Id': own_image, 'Config': {'Labels': {'nexus.frontieram.owner': owner}}},
        'alias-for-base': {'Id': grading.IMAGE_ID, 'Config': {'Labels': {'nexus.frontieram.owner': owner}}},
    }
    calls = []

    async def spawn(*argv, **_kwargs):
        calls.append(argv)
        if argv[3] == 'inspect':
            return ImmediateChild(json.dumps([inspect_results[argv[-1]]]).encode())
        return ImmediateChild()

    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    await grading.cleanup_owned(
        {
            'ledger': str(ledger),
            'docker_path': '/trusted/docker',
            'docker_host': 'unix:///local/socket',
            'owner': owner,
        }
    )
    deletions = [argv[3:] for argv in calls if argv[3] == 'rm' or argv[3:5] == ('image', 'rm')]
    assert set(deletions) == {('rm', '--force', own_container), ('image', 'rm', own_image)}
    assert all('foreign-name' not in argv[3:] for argv in calls if argv[3] != 'inspect')
    assert all('unknown-kind' not in argv for argv in calls)
    assert all(grading.IMAGE_ID not in argv for argv in calls)


class ExecCalledError(Exception):
    pass


@pytest.mark.parametrize('operation', ['create', 'run'])
def test_docker_shim_adds_owned_label_before_launch(tmp_path, monkeypatch, operation):
    config = tmp_path / 'config.json'
    ledger = tmp_path / 'owned.jsonl'
    config.write_text(json.dumps({'docker_path': '/trusted/docker', 'ledger': str(ledger), 'owner': 'run-token'}))
    execute = Mock(side_effect=ExecCalledError())
    monkeypatch.setattr(grading.os, 'execv', execute)
    with pytest.raises(ExecCalledError):
        grading.docker_command(
            config, ['--host', 'unix:///local/socket', operation, '--name', 'owned', 'fixture-image']
        )
    binary, argv = execute.call_args.args
    assert binary == '/trusted/docker'
    assert argv[:6] == [
        '/trusted/docker',
        '--host',
        'unix:///local/socket',
        operation,
        '--label',
        'nexus.frontieram.owner=run-token',
    ]
    assert json.loads(ledger.read_text()) == {'kind': 'container', 'value': 'owned'}


def test_generated_docker_shim_runs_as_a_standalone_executable(tmp_path, monkeypatch):
    recorded = tmp_path / 'argv'
    fake = tmp_path / 'fake-docker'
    fake.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {recorded}\n')
    fake.chmod(0o755)
    ledger = tmp_path / 'owned.jsonl'
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            {
                'output_dir': str(tmp_path),
                'source_root': '/unused',
                'docker_host': 'unix:///unused',
                'docker_path': str(fake),
                'cpuset': '0',
                'cgroup_parent': None,
                'ledger': str(ledger),
                'owner': 'run-token',
            }
        )
    )

    def build(docker, image, root):
        shim = [str(tmp_path / 'owned-docker'), '--host', 'unix:///unused', 'create', '--name', 'owned', 'img']
        subprocess.run(shim, check=True, timeout=30)
        raise SubmissionFailure('stop after one real shim call')

    controller = SimpleNamespace(
        Docker=Mock(),
        offline_build=build,
        SubmissionFailure=SubmissionFailure,
        save=lambda path, value: path.write_text(json.dumps(value)),
    )
    locked = SimpleNamespace(controller=controller, verify_private_data=Mock(), image_id='fixture')
    monkeypatch.setattr(grading, 'LockedAssets', lambda source_root: locked)
    grading.worker(config)
    assert recorded.read_text().split('\n')[:-1] == [
        '--host',
        'unix:///unused',
        'create',
        '--label',
        'nexus.frontieram.owner=run-token',
        '--name',
        'owned',
        'img',
    ]
    assert json.loads(ledger.read_text()) == {'kind': 'container', 'value': 'owned'}


@pytest.mark.parametrize(
    ('error', 'expected_status', 'expected_score'),
    [
        (SubmissionFailure('empty source'), 'finished', 0.0),
        (RuntimeError('daemon unavailable'), 'runtime_error_by_env', None),
    ],
)
def test_grading_worker_distinguishes_submission_and_environment_failures(
    tmp_path,
    monkeypatch,
    error,
    expected_status,
    expected_score,
):
    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            {
                'output_dir': str(tmp_path),
                'source_root': '/unused',
                'docker_host': 'unix:///unused',
                'cpuset': '0',
                'cgroup_parent': None,
            }
        )
    )
    controller = SimpleNamespace(
        Docker=Mock(),
        offline_build=Mock(side_effect=error),
        SubmissionFailure=SubmissionFailure,
        save=lambda path, value: path.write_text(json.dumps(value)),
    )
    locked = SimpleNamespace(controller=controller, verify_private_data=Mock(), image_id='fixture')
    monkeypatch.setattr(grading, 'LockedAssets', lambda source_root: locked)
    grading.worker(config)
    result = json.loads((tmp_path / 'grading_result.json').read_text())
    assert (result['exit_status'], result['score']) == (expected_status, expected_score)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('inspection_error', 'expected_error'),
    [(b'No such object: already-gone', None), (b'Cannot connect to the Docker daemon', RuntimeError)],
)
async def test_owned_inspection_distinguishes_missing_resource_from_daemon_failure(
    tmp_path, monkeypatch, inspection_error, expected_error
):
    ledger = tmp_path / 'owned.jsonl'
    ledger.write_text(json.dumps({'kind': 'container', 'value': 'owned-name'}) + '\n')
    spawn = AsyncMock(
        side_effect=[ImmediateChild(), ImmediateChild(), ImmediateChild(returncode=1, stderr=inspection_error)]
    )
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    config = {
        'ledger': str(ledger),
        'docker_path': '/trusted/docker',
        'docker_host': 'unix:///local/socket',
        'owner': 'run-token',
    }
    if expected_error:
        with pytest.raises(expected_error, match='owned_resource_inspection_failed'):
            await grading.cleanup_owned(config)
    else:
        await grading.cleanup_owned(config)
    assert spawn.await_count == 3


@pytest.mark.asyncio
async def test_owned_deletion_failure_is_not_silently_reported_as_success(tmp_path, monkeypatch):
    ledger = tmp_path / 'owned.jsonl'
    ledger.write_text(json.dumps({'kind': 'container', 'value': 'owned-name'}) + '\n')
    inspected = {'Id': 'a' * 64, 'Config': {'Labels': {'nexus.frontieram.owner': 'run-token'}}}
    spawn = AsyncMock(
        side_effect=[
            ImmediateChild(),
            ImmediateChild(),
            ImmediateChild(json.dumps([inspected]).encode()),
            ImmediateChild(returncode=1),
        ]
    )
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    with pytest.raises(RuntimeError, match='owned_resource_cleanup_failed'):
        await grading.cleanup_owned(
            {
                'ledger': str(ledger),
                'docker_path': '/trusted/docker',
                'docker_host': 'unix:///local/socket',
                'owner': 'run-token',
            }
        )
    assert spawn.await_count == 4


@pytest.mark.asyncio
async def test_owner_discovery_finds_unrecorded_commit_without_a_ledger(tmp_path, monkeypatch):
    owned_image = 'sha256:' + 'd' * 64
    foreign_image = 'sha256:' + 'e' * 64
    calls = []
    inspected = {
        owned_image: {'Id': owned_image, 'Config': {'Labels': {grading.OWNER_LABEL: 'run-token'}}},
        foreign_image: {'Id': foreign_image, 'Config': {'Labels': {grading.OWNER_LABEL: 'other-run'}}},
    }

    async def spawn(*argv, **_kwargs):
        calls.append(argv[3:])
        if argv[3:5] == ('image', 'ls'):
            return ImmediateChild(f'{owned_image}\n{foreign_image}\n{grading.IMAGE_ID}\n'.encode())
        if argv[3] == 'inspect':
            return ImmediateChild(json.dumps([inspected[argv[-1]]]).encode())
        return ImmediateChild()

    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    await grading.cleanup_owned(
        {
            'ledger': str(tmp_path / 'does-not-exist.jsonl'),
            'docker_path': '/trusted/docker',
            'docker_host': 'unix:///local/socket',
            'owner': 'run-token',
        }
    )
    assert ('image', 'rm', owned_image) in calls
    assert ('image', 'rm', foreign_image) not in calls
    assert all(grading.IMAGE_ID not in argv for argv in calls)
    discovery = [argv for argv in calls if argv[1] == 'ls']
    assert len(discovery) == 2
    assert all(argv[-2:] == ('--filter', 'label=nexus.frontieram.owner=run-token') for argv in discovery)


@pytest.mark.asyncio
async def test_owned_cleanup_attempts_remaining_resources_and_aggregates_failures(tmp_path, monkeypatch):
    ledger = tmp_path / 'owned.jsonl'
    ledger.write_text(
        '\n'.join(json.dumps({'kind': 'container', 'value': value}) for value in ['a', 'b', 'c']) + '\n{"partial":\n'
    )
    calls = []

    async def spawn(*argv, **_kwargs):
        calls.append(argv[3:])
        if argv[3] == 'inspect':
            value = argv[-1]
            if value == 'b':
                return ImmediateChild(returncode=1, stderr=b'Daemon temporarily unavailable')
            info = {'Id': value * 64, 'Config': {'Labels': {grading.OWNER_LABEL: 'run-token'}}}
            return ImmediateChild(json.dumps([info]).encode())
        if argv[3] == 'rm' and argv[-1] == 'a' * 64:
            return ImmediateChild(returncode=1, stderr=b'Deletion failed')
        return ImmediateChild()

    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    with pytest.raises(grading.OwnedCleanupError) as caught:
        await grading.cleanup_owned(
            {
                'ledger': str(ledger),
                'docker_path': '/trusted/docker',
                'docker_host': 'unix:///local/socket',
                'owner': 'run-token',
            }
        )
    assert ('rm', '--force', 'c' * 64) in calls
    assert [error['category'] for error in caught.value.errors] == [
        'owned_resource_ledger_record_invalid',
        'owned_resource_cleanup_failed',
        'owned_resource_inspection_failed',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup_fails', [False, True])
async def test_repeated_grader_cancellation_waits_for_cleanup_and_keeps_cancel_priority(
    tmp_path, monkeypatch, cleanup_fails
):
    child = WaitingChild()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def cleanup(_config):
        cleanup_started.set()
        await finish_cleanup.wait()
        cleanup_finished.set()
        if cleanup_fails:
            raise grading.OwnedCleanupError([{'category': 'owned_resource_cleanup_failed'}])

    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', AsyncMock(return_value=child))
    monkeypatch.setattr(grading.os, 'killpg', Mock(side_effect=child.terminate))
    monkeypatch.setattr(grading, 'cleanup_owned', cleanup)
    environment = SimpleNamespace(
        assets=SimpleNamespace(source_root=tmp_path / 'frozen'),
        docker=SimpleNamespace(prefix=['/trusted/docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
    )
    pending = asyncio.create_task(grading.grade(environment, tmp_path))
    await child.started.wait()
    pending.cancel()
    await cleanup_started.wait()
    for _ in range(3):
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        assert not cleanup_finished.is_set()
    finish_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cleanup_finished.is_set()
    diagnostic = tmp_path / 'grading_cleanup_error.json'
    assert diagnostic.exists() is cleanup_fails
    if cleanup_fails:
        assert json.loads(diagnostic.read_text())['errors'] == [{'category': 'owned_resource_cleanup_failed'}]


@pytest.mark.asyncio
async def test_cancellation_during_process_creation_still_reaps_the_created_worker(tmp_path, monkeypatch):
    child = WaitingChild()
    spawn_started = asyncio.Event()
    finish_spawn = asyncio.Event()

    async def spawn(*_argv, **_kwargs):
        spawn_started.set()
        await finish_spawn.wait()
        return child

    cleanup = AsyncMock()
    kill = Mock(side_effect=child.terminate)
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', spawn)
    monkeypatch.setattr(grading.os, 'killpg', kill)
    monkeypatch.setattr(grading, 'cleanup_owned', cleanup)
    environment = SimpleNamespace(
        assets=SimpleNamespace(source_root=tmp_path / 'frozen'),
        docker=SimpleNamespace(prefix=['/trusted/docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
    )
    pending = asyncio.create_task(grading.grade(environment, tmp_path))
    await spawn_started.wait()
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    finish_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert child.returncode is not None
    assert kill.call_args_list == [call(child.pid, signal.SIGTERM), call(child.pid, signal.SIGKILL)]
    cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_successful_grader_result_cannot_hide_cleanup_failure(tmp_path, monkeypatch):
    child = ImmediateChild()
    (tmp_path / 'grading_result.json').write_text(json.dumps({'exit_status': 'finished', 'score': 88.0}))
    cleanup = AsyncMock(side_effect=RuntimeError('synthetic cleanup failure'))
    monkeypatch.setattr(grading.asyncio, 'create_subprocess_exec', AsyncMock(return_value=child))
    monkeypatch.setattr(grading, 'cleanup_owned', cleanup)
    environment = SimpleNamespace(
        assets=SimpleNamespace(source_root=tmp_path / 'frozen'),
        docker=SimpleNamespace(prefix=['/trusted/docker', '--host', 'unix:///local/socket'], cpuset='0', cgroup_parent=None),
    )
    with pytest.raises(grading.OwnedCleanupError):
        await grading.grade(environment, tmp_path)
    diagnostic = json.loads((tmp_path / 'grading_cleanup_error.json').read_text())
    assert diagnostic['errors'][0]['category'] == 'owned_resource_cleanup_failed'
    assert json.loads((tmp_path / 'grading_result.json').read_text())['score'] == 88.0
