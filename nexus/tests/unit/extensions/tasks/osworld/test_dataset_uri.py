"""OSWorld dataset uri regressions."""

from __future__ import annotations

import asyncio
import copy
import json
import threading

import pytest

from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.extensions.tasks.os_world.data import dataset_files as datasets
from nexus.extensions.tasks.os_world.generators import generator_native as generator
from nexus.runner.compose import compile_single, parse_ordered_parts
from nexus.runner.single import RunInput


def record(task_id='task-a', *, split='test'):
    return {
        'id': task_id,
        'body': {'task_name': 'os_world', 'task_config': {'domain': 'chrome', 'task_id': task_id}},
        'metadata': {'domain': 'chrome', 'instruction': 'Do the task'},
        'files': {
            'type': 'zip',
            'path': f'blobs/{task_id}.zip',
            'size': 1234,
            'sha256': 'a' * 64,
            'remote': {'namespace': 'test', 'dataset': 'osworld', 'version': 'v1', 'split': split},
        },
    }


@pytest.fixture
def remote(monkeypatch):
    datasets._load_remote_index.cache_clear()
    datasets._load_manifest_index.cache_clear()
    for key in (
        'OSWORLD_DATASET_MANIFEST',
        'OSWORLD_DATASET_SPLIT',
        'OSWORLD_TEST_META_PATH',
        'OSWORLD_TASK_IDS',
        'OSWORLD_SKIP',
        'OSWORLD_LIMIT',
        'OSWORLD_DOMAIN',
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OSWORLD_DATASET_URI', 'test/osworld@v1')
    monkeypatch.setenv('WEDATA_KEY', '{"id":"fixture","subject":"fixture","key":"test-credential"}')
    state = {
        'records': [record(), record('task-b')],
        'opens': 0,
        'closes': 0,
        'resolved': [],
        'threads': [],
        'error': None,
    }

    class Client:
        def __init__(self, **kwargs):
            state['client_args'] = kwargs

        async def __aenter__(self):
            state['opens'] += 1
            state['threads'].append(threading.get_ident())
            return self

        async def __aexit__(self, *args):
            state['closes'] += 1

        async def resolve_ref(self, uri, *, split):
            state['resolved'].append((uri, split))
            return RemoteDatasetRef(namespace='test', dataset='osworld', version='v1', split=split)

        async def iter_resolved_items(self, ref):
            if state['error']:
                raise state['error']
            for raw in state['records']:
                # Mirrors DatasetClient's parsed records and resolved release.
                await asyncio.sleep(0)
                yield DatasetItem.from_mapping(copy.deepcopy(raw), remote_ref=ref)

    monkeypatch.setattr(datasets, 'DatasetClient', Client)
    yield state
    datasets._load_remote_index.cache_clear()
    datasets._load_manifest_index.cache_clear()


def test_uri_fetches_metadata_once_and_preserves_remote_specs(remote):
    first = datasets.dataset_index()
    assert list(first) == ['task-a', 'task-b']
    assert first['task-a']['metadata'] == record()['metadata']
    assert first['task-a']['files'] == record()['files']
    assert datasets.dataset_task_files('task-b')['size'] == 1234
    assert datasets.dataset_index() == first
    assert remote['opens'] == remote['closes'] == 1
    assert remote['resolved'] == [('test/osworld@v1', 'test')]


async def test_sync_generator_lookup_inside_event_loop_uses_worker(remote):
    caller = threading.get_ident()
    assert len(datasets.dataset_index()) == 2
    assert len(remote['threads']) == 1
    assert remote['threads'][0] != caller
    assert remote['opens'] == remote['closes'] == 1


def test_split_and_server_and_auth_changes_do_not_reuse_other_index(remote, monkeypatch):
    datasets.dataset_index()
    monkeypatch.setenv('OSWORLD_DATASET_SPLIT', 'validation')
    remote['records'] = [record(split='validation')]
    datasets.dataset_index()
    monkeypatch.setenv('TAIJI_DATASET_SERVER_URL', 'http://fixture-server')
    datasets.dataset_index()
    monkeypatch.setenv('WEDATA_KEY', 'different-credential')
    datasets.dataset_index()
    assert remote['opens'] == 4
    assert remote['client_args']['server_url'] == 'http://fixture-server'


def test_latest_resolves_once_and_files_pin_the_release(remote, monkeypatch):
    monkeypatch.setenv('OSWORLD_DATASET_URI', 'test/osworld@latest?split=validation')
    remote['records'] = [record(split='validation')]
    first = datasets.dataset_index()
    assert first['task-a']['files']['remote']['version'] == 'v1'
    assert first['task-a']['files']['remote']['split'] == 'validation'
    datasets.dataset_index()
    assert remote['resolved'] == [('test/osworld@latest', 'validation')]


def test_explicit_file_split_mismatch_is_rejected(remote, monkeypatch):
    monkeypatch.setenv('OSWORLD_DATASET_SPLIT', 'validation')
    with pytest.raises(RuntimeError, match='split=validation, ValueError'):
        datasets.dataset_index()


def test_manifest_override_is_offline(remote, monkeypatch, tmp_path):
    manifest = tmp_path / 'test.jsonl'
    manifest.write_text(json.dumps(record()))
    monkeypatch.setenv('OSWORLD_DATASET_MANIFEST', str(manifest))
    assert list(datasets.dataset_index()) == ['task-a']
    assert remote['opens'] == 0


def test_bad_manifest_does_not_trigger_network_fallback(remote, monkeypatch, tmp_path):
    monkeypatch.setenv('OSWORLD_DATASET_MANIFEST', str(tmp_path / 'missing.jsonl'))
    with pytest.raises(ValueError, match='does not exist'):
        datasets.dataset_index()
    assert remote['opens'] == 0


@pytest.mark.parametrize('defect', ['size', 'domain', 'instruction', 'files', 'duplicate', 'empty'])
def test_invalid_remote_index_is_not_silently_accepted_or_cached(remote, defect):
    if defect == 'size':
        remote['records'][0]['files'].pop('size')
    elif defect in ('domain', 'instruction'):
        remote['records'][0]['metadata'].pop(defect)
    elif defect == 'files':
        remote['records'][0].pop('files')
    elif defect == 'duplicate':
        remote['records'] = [record(), record()]
    else:
        remote['records'] = []
    with pytest.raises(RuntimeError, match='Cannot load OSWorld dataset index'):
        datasets.dataset_index()
    remote['records'] = [record()]
    assert list(datasets.dataset_index()) == ['task-a']
    assert remote['opens'] == remote['closes'] == 2


def test_remote_failure_is_safe_and_never_uses_local_examples(remote, monkeypatch):
    monkeypatch.setenv('OSWORLD_EXAMPLES_ROOT', '/old-local-checkout')
    remote['error'] = RuntimeError('https://host/file?token=SECRET_VALUE')
    with pytest.raises(RuntimeError) as caught:
        datasets.dataset_index()
    assert 'SECRET_VALUE' not in str(caught.value)
    assert caught.value.__suppress_context__
    assert 'no local fallback' in str(caught.value)
    remote['error'] = None
    assert len(datasets.dataset_index()) == 2


@pytest.mark.parametrize(('family', 'harness'), sorted(generator.SUPPORTED))
def test_uri_only_single_and_batch_configs_match(remote, monkeypatch, family, harness, tmp_path):
    settings = {
        'OSWORLD_MODEL_FAMILY': family,
        'OSWORLD_HARNESS': harness,
        'OSWORLD_VALIDATE_EVALUATOR_HOST_DEPS': '0',
        'OSWORLD_AGS_PROVISION_MODE': 'control_plane',
        'OSWORLD_ENABLE_PROXY': 'false',
        'AGS_OSWORLD_TOOL_ID': 'fixture-tool',
        'AGS_SECRET_ID': 'fixture-id',
        'AGS_SECRET_KEY': 'fixture-key',
        'NEXUS_MODEL': 'routing-name',
        'NEXUS_BASE_URL': 'http://fixture',
        'NEXUS_API_KEY': 'key',
        'HUNYUAN_MODEL': 'service',
        'HUNYUAN_BASE_URL': 'http://fixture',
        'OSWORLD_SESSION_NAMESPACE': str(tmp_path),
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    # compile_single calls the synchronous run part from its own asyncio loop.
    single = compile_single(
        parse_ordered_parts([
            '--use',
            f'osworld_native:model_family={family},harness={harness},task_id=task-b',
        ])
    )
    items = list(generator.generate_configs())
    expected = RunInput.from_dict(items[1].config.model_dump())
    assert single.model_dump(exclude={'run_id'}) == expected.model_dump(exclude={'run_id'})
    assert len(items) == 2
    assert remote['opens'] == remote['closes'] == 1
    assert all(not item.config.task_config.get('task_root') for item in items)
    assert 'test-credential' not in json.dumps(items[0].model_dump())


def test_manifest_rejects_mismatched_release_and_duplicates(remote, monkeypatch, tmp_path):
    manifest = tmp_path / 'test.jsonl'
    raw = record()
    raw['files']['remote']['version'] = 'other'
    manifest.write_text(json.dumps(raw))
    monkeypatch.setenv('OSWORLD_DATASET_MANIFEST', str(manifest))
    with pytest.raises(ValueError, match='targets'):
        datasets.dataset_index()
    manifest.write_text('\n'.join(json.dumps(record()) for _ in range(2)))
    with pytest.raises(ValueError, match='Duplicate'):
        datasets.dataset_index()


def test_uri_ignores_stale_examples_root_in_inherited_environment(remote, monkeypatch):
    from nexus.extensions.tasks.os_world.data import task_source

    monkeypatch.setenv('OSWORLD_EXAMPLES_ROOT', '/old-checkout-that-must-not-be-used')
    assert not hasattr(task_source, 'examples_root_str')
    tasks = list(task_source.iter_selected_tasks())
    assert len(tasks) == 2
    assert all(task.task_file is None for task in tasks)
