import copy
import json
import os

from pathlib import Path

import pytest

from nexus.core.context import ExecutionContext
from nexus.dataset import task_files as task_files_module
from nexus.dataset.auth import clear_wedata_key_cache, load_wedata_key
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.tasks.mac_os_world import dataset_source, generator
from nexus.extensions.tasks.mac_os_world.export_platform_dataset import platform_parts
from nexus.extensions.tasks.mac_os_world.task import MacOSWorldTaskInput, load_mac_os_world_task
from nexus.runner.compose import compile_single, parse_ordered_parts
from nexus.runner.compose.merge import deep_merge
from nexus.runner.single import RunInput


async def _unused_agent(**kwargs):
    raise AssertionError('Agent must not be called during task loading')


def _item(task_id='one'):
    return DatasetItem.from_mapping(
        {
            'id': task_id,
            'body': {'task_name': 'mac_os_world', 'task_config': {'task_id': task_id, 'category': 'sys_apps'}},
            'files': {'type': 'zip', 'path': f'blobs/{task_id}.zip', 'size': 200, 'sha256': 'a' * 64},
            'metadata': {'instructions': {'en': f'Open {task_id}', 'zh': f'打开 {task_id}'}},
        },
        remote_ref=RemoteDatasetRef(namespace='example', dataset='macosworld', version='v1', split='test'),
    )


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    clear_wedata_key_cache()
    for key in list(os.environ):
        if key.startswith(('MACOSWORLD_', 'WEDATA_')):
            monkeypatch.delenv(key)
    yield
    clear_wedata_key_cache()


@pytest.mark.asyncio
async def test_pagination_pins_version_and_reads_all_pages(monkeypatch):
    pages = []
    requests = []

    class Client:
        def __init__(self, **kwargs):
            self.io = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def resolve_version(self, parsed):
            assert parsed.version == 'latest'
            return 'v1'

        async def resolve_split(self, parsed, *, version, split):
            assert version == 'v1'
            return split or 'test'

        async def list_items(self, namespace, dataset, version, *, split, cursor, limit):
            pages.append((version, split, cursor))
            ids = [str(n) for n in range(100)] if cursor is None else ['100', '101']
            return {'ids': ids, 'next_cursor': '100' if cursor is None else None}

        async def load_items(self, uri, *, split, item_id):
            requests.append((uri, split, item_id))
            return [_item(item_id)]

    monkeypatch.setattr(dataset_source, 'DatasetClient', Client)
    items = await dataset_source._fetch_items('example/macosworld@latest', None)
    assert len(items) == 102
    assert pages == [('v1', 'test', None), ('v1', 'test', '100')]
    assert {uri for uri, _, _ in requests} == {'example/macosworld@v1'}
    assert {split for _, split, _ in requests} == {'test'}


@pytest.mark.asyncio
async def test_sync_generator_in_async_loop_has_no_checkout_dependency(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MACOSWORLD_DATASET_URI', 'example/macosworld@v1?split=safety')
    monkeypatch.setenv('MACOSWORLD_LANGUAGE', 'zh')
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', '/missing/old-checkout')

    async def fetch(uri, split):
        assert uri.endswith('?split=safety')
        return [_item()]

    monkeypatch.setattr(dataset_source, '_fetch_items', fetch)
    item = next(generator.generate_configs())
    assert item.metadata['instruction'] == '打开 one'
    assert item.config.task_files.remote is not None
    assert 'task_root' not in item.config.task_config
    assert 'task_file' not in item.metadata


def test_missing_dataset_does_not_fall_back_to_existing_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv('MACOSWORLD_TASK_ROOT', str(tmp_path))
    with pytest.raises(ValueError, match='MACOSWORLD_DATASET_URI'):
        list(generator.generate_configs())


def test_failed_dataset_hides_credentials_and_does_not_fall_back(monkeypatch):
    monkeypatch.setenv('MACOSWORLD_DATASET_URI', 'example/macosworld@v1')

    async def fail(*args):
        raise RuntimeError('sensitive-signed-url-and-token')

    monkeypatch.setattr(dataset_source, '_fetch_items', fail)
    with pytest.raises(RuntimeError, match='no local fallback') as error:
        dataset_source.dataset_index()
    assert 'sensitive' not in str(error.value)


@pytest.mark.asyncio
async def test_task_requires_attached_files_before_runtime_creation():
    async with ExecutionContext(task_name='mac_os_world'):
        with pytest.raises(ValueError, match='requires Nexus Dataset task_files'):
            await load_mac_os_world_task(
                MacOSWorldTaskInput(task_id='one', category='sys_apps', run_agent=_unused_agent)
            )


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['missing', 'mismatch', 'legacy_root'])
async def test_task_rejects_invalid_injection(tmp_path, case):
    if case != 'missing':
        (tmp_path / 'task.json').write_text(json.dumps({'id': 'wrong-id'}))
    files = task_files_from_spec({'type': 'directory', 'path': str(tmp_path)})
    params = {'task_id': 'one', 'category': 'sys_apps', 'run_agent': _unused_agent}
    if case == 'legacy_root':
        params['task_root'] = str(tmp_path)
    error_type = RuntimeError if case == 'missing' else ValueError
    async with ExecutionContext(task_name='mac_os_world', task_files=files):
        with pytest.raises(
            error_type,
            match={
                'missing': 'Cannot read',
                'mismatch': 'does not match',
                'legacy_root': 'task_root is no longer supported',
            }[case],
        ):
            await load_mac_os_world_task(MacOSWorldTaskInput(**params))


@pytest.mark.asyncio
@pytest.mark.parametrize('as_json', [False, True])
async def test_config_supplied_wedata_key_refreshes_standard_auth(monkeypatch, as_json):
    key = {'subject': 'example', 'key': 'private-test-key', 'id': 42}
    monkeypatch.setenv('WEDATA_KEY', json.dumps({**key, 'key': 'previous-key'}))
    assert load_wedata_key().key == 'previous-key'
    seen = []

    async def read(files, path, encoding='utf-8'):
        session = await files.create_session()
        seen.append(session.auth)
        assert path == 'task.json'
        return '{"id":"one"}'

    monkeypatch.setattr(task_files_module.RemoteZipTaskFiles, 'read_text', read)
    files = task_files_from_spec(_item().resolved_task_files())
    async with ExecutionContext(task_name='mac_os_world', task_files=files):
        await load_mac_os_world_task(
            MacOSWorldTaskInput(
                task_id='one',
                category='sys_apps',
                wedata_key=json.dumps(key) if as_json else key,
                run_agent=_unused_agent,
            )
        )
    assert seen[0]['key']['key'] == key['key']
    assert json.loads(os.environ['WEDATA_KEY']) == key
    assert 'private-test-key' not in json.dumps(files.model_dump())


def test_platform_export_recombines_without_local_paths_or_model_credentials(monkeypatch):
    monkeypatch.setenv('MACOSWORLD_DATASET_URI', 'example/macosworld@v1')
    monkeypatch.setenv('WEDATA_KEY', 'private-secret')
    monkeypatch.setattr(generator, 'dataset_index', lambda: {'one': _item(), 'two': _item('two')})
    items = list(generator.generate_configs())
    rows, evaluation = platform_parts(items)
    assert len(rows) == 2
    serialized = json.dumps((rows, evaluation))
    assert 'private-secret' not in serialized
    assert 'task_root' not in serialized
    assert 'task_file"' not in serialized
    assert 'GPT_CUA_API_KEY' not in serialized
    for item, row in zip(items, rows, strict=True):
        actual = deep_merge(row['meta_data_']['nexus_run_input'], evaluation['nexus_run_input'])
        original = item.config.model_dump(exclude_unset=True)
        for key in ('processors', 'runner'):
            original.pop(key)
        original['openai_clients'] = {'main': {'name': 'macosworld_openai_responses'}}
        assert actual == original
        RunInput.model_validate(actual)
    changed = copy.deepcopy(items)
    changed[1].config.agent_config['max_iterations'] = 999
    with pytest.raises(ValueError, match='different common settings'):
        platform_parts(changed)


@pytest.mark.parametrize(
    'name',
    [
        'docs/examples/configs/mac_os_world/mac_os_world_gptcua_devcloud.yaml',
        'docs/examples/configs/mac_os_world/mac_os_world_gpt56_luna_devcloud_test.yaml',
    ],
)
def test_all_yaml_configs_compose_with_dataset_identity(name, monkeypatch, tmp_path):
    from nexus.dataset.layout import create_dataset_template

    monkeypatch.setenv('DEVCLOUD_APP_ID', 'test-id')
    monkeypatch.setenv('DEVCLOUD_APP_TOKEN', 'test-token')
    root = create_dataset_template(tmp_path / 'dataset', template='generic', item_id='one')
    # A local authoring Dataset exercises composition without an upstream checkout.
    import yaml

    item_path = root / 'items/one/item.yaml'
    item_path.write_text(
        yaml.safe_dump({
            'id': 'one',
            'body': {'task_name': 'mac_os_world', 'task_config': {'task_id': 'one', 'category': 'sys_apps'}},
            'files': {'type': 'directory', 'path': 'files'},
        })
    )
    repo = Path(__file__).resolve().parents[3]
    config = compile_single(parse_ordered_parts(['--use', str(repo / name), '--data', str(root)]), item_id='one')
    assert config.task_files is not None
    assert config.task_config['task_id'] == 'one'
    assert config.task_config['category'] == 'sys_apps'
    assert 'task_root' not in config.task_config
