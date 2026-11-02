"""One model sample must cover every required SuperSWE evaluation component."""

import copy
import hashlib
import json
import os

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from nexus.dataset.item import TaskFilesSpec
from nexus.extensions.tasks.swe import superswe
from nexus.extensions.tasks.swe import task as native_task
from nexus.extensions.tasks.swe.t3build_swebuilder_midtrain import dataset_handler
from nexus.extensions.tasks.swe.task import SwebenchTaskInput, SwebenchTaskOutput


PATCH = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n'


async def forbidden_agent(**kwargs):
    raise AssertionError('Supplementary evaluation must not start another agent')


def sample():
    row = {
        'instance_id': 'example__repo-1',
        'repo': 'example/repo',
        'base_commit': 'abc123',
        'dataset_type': 'T3Build_SWEBuilder_Midtrain',
        'image': 'example/image:fixed',
        'problem_statement': 'Fix the behavior',
        'eval_script': 'echo OMNIGRIL_EXIT_CODE=0',
    }
    row[superswe.COMPONENTS_KEY] = [
        {
            'name': 'cli',
            'instance': copy.deepcopy(row),
            'eval_timeout_seconds': 1800,
        }
    ]
    return row


def task_input(**updates):
    return SwebenchTaskInput(instance_id='example__repo-1', instance=sample(), run_agent=forbidden_agent, **updates)


def result(*, resolved=True, completed=True, patch=PATCH, error=None):
    return SwebenchTaskOutput(
        score=1.0 if resolved else 0.0,
        resolved=resolved,
        completed=completed,
        patch_text=patch,
        error=error,
        report_json=json.dumps({'example__repo-1': {'resolved': resolved}}),
    )


def remote_request(monkeypatch, *, row=None, **updates):
    raw = json.dumps(sample() if row is None else row).encode()
    spec = TaskFilesSpec(
        type='zip',
        path='blobs/example__repo-1.zip',
        sha256='a' * 64,
        size=1234,
        remote={'namespace': 'example', 'dataset': 'superswe', 'version': 'v1', 'split': 'test'},
    )
    files = SimpleNamespace(spec=spec, read_bytes=AsyncMock(return_value=raw), download=AsyncMock())
    ctx = SimpleNamespace(task_files=files, run_id='remote-test')
    monkeypatch.setattr(superswe, 'require_context', lambda: ctx)
    values = {
        'instance_id': 'example__repo-1',
        'instance_file': 'instance.json',
        'instance_sha256': hashlib.sha256(raw).hexdigest(),
        'run_agent': forbidden_agent,
    }
    return superswe.SupersweTaskInput(**(values | updates)), files


@pytest.mark.asyncio
async def test_remote_instance_preserves_all_components_and_records_source(monkeypatch):
    request, files = remote_request(monkeypatch)
    native = AsyncMock(side_effect=[result(), result()])
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    output = await superswe.run_superswe_task(request)
    files.read_bytes.assert_awaited_once_with('instance.json')
    files.download.assert_not_called()
    assert request.instance is None
    assert native.await_count == 2
    assert native.await_args_list[1].args[0].patch_text == PATCH
    assert output.completed is True
    assert output.resolved is True
    assert output.extra_fields['superswe_input']['instance_sha256'] == request.instance_sha256
    assert output.extra_fields['superswe_input']['remote']['version'] == 'v1'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'invalid', ['hash', 'missing_hash', 'absolute_path', 'parent_path', 'local', 'latest', 'source']
)
async def test_remote_instance_rejects_unbound_or_ambiguous_input(monkeypatch, invalid):
    request, files = remote_request(monkeypatch)
    if invalid == 'hash':
        request.instance_sha256 = '0' * 64
    elif invalid == 'missing_hash':
        request.instance_sha256 = ''
    elif invalid == 'absolute_path':
        request.instance_file = '/host/instance.json'
    elif invalid == 'parent_path':
        request.instance_file = '../instance.json'
    elif invalid == 'local':
        files.spec.remote = None
    elif invalid == 'latest':
        files.spec.remote.version = 'latest'
    else:
        request.dataset_path = '/host/dataset.jsonl'
    native = AsyncMock()
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    with pytest.raises(ValueError, match='SuperSWE'):
        await superswe.run_superswe_task(request)
    native.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', ['instance_id', 'component'])
async def test_remote_instance_checks_identity_and_component_contract(monkeypatch, invalid):
    row = sample()
    if invalid == 'instance_id':
        row['instance_id'] = 'another-task'
    else:
        row[superswe.COMPONENTS_KEY][0]['instance']['base_commit'] = 'another-commit'
    request, _ = remote_request(monkeypatch, row=row)
    native = AsyncMock()
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    with pytest.raises(ValueError, match='does not match|different'):
        await superswe.run_superswe_task(request)
    native.assert_not_called()


@pytest.mark.asyncio
async def test_injected_wedata_key_is_request_scoped_and_excluded_from_serialization(monkeypatch):
    request, files = remote_request(monkeypatch, wedata_key='test-only-key')
    authenticator = object()
    make_auth = Mock(return_value=authenticator)
    make_files = Mock(return_value=files)
    monkeypatch.setattr(superswe, 'WedataAuthenticator', make_auth)
    monkeypatch.setattr(superswe, 'task_files_from_spec', make_files)
    monkeypatch.setattr(superswe, 'run_swe_task', AsyncMock(side_effect=[result(), result()]))
    monkeypatch.setenv('WEDATA_KEY', 'unchanged-test-value')
    await superswe.run_superswe_task(request)
    assert os.environ['WEDATA_KEY'] == 'unchanged-test-value'
    assert 'wedata_key' not in request.model_dump()
    make_auth.assert_called_once_with(raw_key='test-only-key')
    make_files.assert_called_once_with(files.spec, run_id='remote-test', authenticator=authenticator)


@pytest.mark.asyncio
@pytest.mark.parametrize(('primary', 'secondary'), [(True, True), (False, True), (True, False), (False, False)])
async def test_components_use_same_patch_and_and_results(monkeypatch, primary, secondary):
    calls = []

    async def native(task_request):
        calls.append(task_request)
        return result(resolved=primary if len(calls) == 1 else secondary)

    monkeypatch.setattr(superswe, 'run_swe_task', native)
    output = await superswe.run_superswe_task(task_input())
    assert output.completed is True
    assert output.resolved is (primary and secondary)
    assert output.score == float(primary and secondary)
    assert len(calls) == 2
    assert calls[1].patch_text == PATCH
    assert calls[1].run_oracle is False
    assert calls[1].eval_runtime_name == 'eval_component_cli'
    assert calls[1].eval_runtime_name != calls[0].eval_runtime_name
    assert calls[1].eval_timeout_seconds == 1800
    assert superswe.COMPONENTS_KEY not in calls[0].instance
    assert output.extra_fields['superswe_components']['pending'] == []
    assert json.loads(output.report_json)['example__repo-1']['resolved'] is (primary and secondary)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'primary',
    [
        result(completed=False, resolved=False, error='sandbox unavailable'),
        result(completed=False, resolved=False, patch=''),
    ],
)
async def test_missing_primary_result_never_triggers_new_sampling(monkeypatch, primary):
    native = AsyncMock(return_value=primary)
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    output = await superswe.run_superswe_task(task_input())
    assert native.await_count == 1
    assert output.completed is False
    assert output.resolved is False
    assert output.extra_fields['superswe_components']['pending'] == ['cli']


@pytest.mark.asyncio
@pytest.mark.parametrize('patch', ['', ' \n'])
async def test_empty_candidate_is_graded_by_every_component(monkeypatch, patch):
    native = AsyncMock(side_effect=[result(resolved=False, patch=patch), result(resolved=False, patch=patch)])
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    request = task_input()
    request.instance[superswe.COMPONENTS_KEY][0]['instance']['patch'] = PATCH

    output = await superswe.run_superswe_task(request)

    assert native.await_count == 2
    supplement = native.await_args_list[1].args[0]
    assert supplement.patch_text == patch
    assert supplement.run_oracle is False
    assert output.patch_text == patch
    assert output.completed is True
    assert output.resolved is False
    assert output.error is None
    assert output.extra_fields['superswe_components']['pending'] == []
    assert set(output.extra_fields['superswe_components']['results']) == {'main', 'cli'}


@pytest.mark.asyncio
async def test_oracle_patch_is_reused_for_supplement(monkeypatch):
    native = AsyncMock(side_effect=[result(), result()])
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    await superswe.run_superswe_task(task_input(run_oracle=True))
    assert native.await_args_list[0].args[0].run_oracle is True
    assert native.await_args_list[1].args[0].run_oracle is False
    assert native.await_args_list[1].args[0].patch_text == PATCH


@pytest.mark.asyncio
@pytest.mark.parametrize('secondary', [result(patch='different patch'), RuntimeError('judge crashed')])
async def test_supplement_failure_retains_primary_patch(monkeypatch, secondary):
    monkeypatch.setattr(superswe, 'run_swe_task', AsyncMock(side_effect=[result(), secondary]))
    output = await superswe.run_superswe_task(task_input())
    assert output.completed is False
    assert output.resolved is False
    assert output.score == 0.0
    assert output.patch_text == PATCH
    assert output.error


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['base_commit', 'instance_id', 'repo', 'dataset_type', 'duplicate', 'retry'])
async def test_invalid_component_rejected_before_execution(monkeypatch, change):
    task_request = task_input()
    component = task_request.instance[superswe.COMPONENTS_KEY][0]
    if change == 'duplicate':
        task_request.instance[superswe.COMPONENTS_KEY].append(copy.deepcopy(component))
    elif change == 'retry':
        task_request.eval_max_attempts = 2
    else:
        component['instance'][change] = 'different'
    native = AsyncMock()
    monkeypatch.setattr(superswe, 'run_swe_task', native)
    with pytest.raises(ValueError, match='different|Duplicate|exactly one'):
        await superswe.run_superswe_task(task_request)
    native.assert_not_called()


@pytest.mark.parametrize('patch', [None, '', ' \n'])
def test_empty_patch_has_no_application_evidence(tmp_path, patch):
    log = tmp_path / 'test.log'
    log.write_text('OMNIGRIL_EXIT_CODE=0\n')
    report = dataset_handler.get_eval_report_map(
        {'instance_id': 'x'},
        {'model_patch': patch},
        str(log),
        True,
    )['x']
    assert report['patch_exists'] is False
    assert report['resolved'] is False


@pytest.mark.parametrize('run_log', [None, 'run_instance.log', 'run_instance_after_apply.log'])
def test_nonempty_log_does_not_prove_patch_applied(tmp_path, run_log):
    log = tmp_path / 'test.log'
    log.write_text('OMNIGRIL_EXIT_CODE=0\n')
    if run_log:
        (tmp_path / run_log).write_text('Evaluation started; patch application status was not recorded.\n')
    report = dataset_handler.get_eval_report_map({'instance_id': 'x'}, {'model_patch': PATCH}, str(log), True)['x']
    assert report['patch_successfully_applied'] is None


def test_go_limit_applies_outside_testbed():
    output = dataset_handler.get_eval_script({
        'language': 'go',
        'eval_script': '#!/bin/sh\nset -e\ncd /app\ngo test ./...\n',
    })
    assert output.count('export GOMAXPROCS=4') == 1
    assert 'cd /app' in output


@pytest.mark.asyncio
async def test_eval_initialization_preserves_ignored_dependencies():
    runtime = SimpleNamespace(call_executor=AsyncMock())
    await dataset_handler.eval_runtime_init({'workdir': '/work path'}, runtime)
    commands = [call.args[1]['command'] for call in runtime.call_executor.await_args_list]
    assert commands == ["git config --global --add safe.directory '/work path'", 'git clean -fd']


@pytest.mark.asyncio
@pytest.mark.parametrize('patch', [PATCH, '', ' \n'])
async def test_native_task_retains_patch_on_runtime_failure(monkeypatch, patch):
    provider = SimpleNamespace(request_timeout=1.0, configure=Mock(), release=AsyncMock())
    ctx = SimpleNamespace(
        runtime_provider=provider, _runtimes={}, get_runtime=AsyncMock(side_effect=RuntimeError('unavailable'))
    )
    monkeypatch.setattr(native_task, 'require_context', lambda: ctx)
    monkeypatch.setattr(native_task, 'get_handler', lambda kind: SimpleNamespace(get_repo_language=lambda row: 'go'))
    output = await native_task.run_swe_task(task_input(patch_text=patch))
    assert output.completed is False
    assert output.patch_text == patch
    assert 'unavailable' in output.error
    assert {call.args[0] for call in provider.configure.call_args_list} == {'eval'}


@pytest.mark.asyncio
async def test_native_task_without_a_patch_still_requests_inference(monkeypatch):
    provider = SimpleNamespace(request_timeout=1.0, configure=Mock(), release=AsyncMock())
    ctx = SimpleNamespace(
        runtime_provider=provider, _runtimes={}, get_runtime=AsyncMock(side_effect=RuntimeError('unavailable'))
    )
    monkeypatch.setattr(native_task, 'require_context', lambda: ctx)
    monkeypatch.setattr(native_task, 'get_handler', lambda kind: SimpleNamespace(get_repo_language=lambda row: 'go'))

    output = await native_task.run_swe_task(task_input())

    assert output.completed is False
    assert {call.args[0] for call in provider.configure.call_args_list} == {'main'}
    assert 'unavailable' in output.error


@pytest.mark.parametrize('completed', [True, False])
def test_native_run_single_writes_result_and_preserves_identity(monkeypatch, tmp_path, completed):
    from nexus.cli.commands.run import run_single

    monkeypatch.setattr(
        superswe, 'run_swe_task', AsyncMock(return_value=result(completed=completed, resolved=completed))
    )
    config = tmp_path / 'task.yaml'
    output_path = tmp_path / 'result.json'
    config.write_text(
        yaml.safe_dump({
            'imports': ['nexus.extensions.tasks.swe.superswe'],
            'task_name': 'superswe_task',
            'task_config': {'instance_id': 'example__repo-1', 'instance': sample(), 'run_oracle': True},
            'extra': {'superswe': {'model_key': 'control', 'instance_id': 'example__repo-1', 'run_number': 0}},
        })
    )
    code = run_single(
        Namespace(verbose=False, config=str(config), output=str(output_path), hold=None, id=None, split=None)
    )
    native = json.loads(output_path.read_text())
    assert code == 0
    assert native['task_output']['completed'] is completed
    assert native['extra']['superswe']['instance_id'] == 'example__repo-1'
