import argparse
import json
import tempfile

from pathlib import Path

import pytest
import yaml

from nexus.cli.commands.dataset import (
    _resolve_publish_target,
    _write_reader_task_files,
    dataset_fetch,
    dataset_get,
    dataset_items,
    dataset_namespace,
    dataset_publish,
    dataset_show,
    dataset_versions,
    register_dataset_parser,
)
from nexus.dataset.auth import TaiPatAuthenticator
from nexus.dataset.item import DatasetItem, RemoteDatasetRef, TaskFilesSpec
from nexus.dataset.reader import BaseDatasetReader, DatasetDescriptor, DatasetSplitDescriptor
from nexus.dataset.source import DatasetSourceURI, dataset_source, dataset_source_registry
from nexus.dataset.zip_index import read_zip_index_from_file, write_nexus_zip_directory_to_file


def _publish_args(**kwargs):
    values = {
        'target': None,
        'namespace': None,
        'name': None,
        'version': None,
        'split': None,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_resolve_publish_target_from_full_uri():
    namespace, dataset_name, release_version, split = _resolve_publish_target(
        _publish_args(target='research/manual_terminal_smoke@v1?split=test')
    )

    assert namespace == 'research'
    assert dataset_name == 'manual_terminal_smoke'
    assert release_version == 'v1'
    assert split == 'test'


def test_resolve_publish_target_keeps_flag_compatibility():
    namespace, dataset_name, release_version, split = _resolve_publish_target(
        _publish_args(namespace='research', name='manual_terminal_smoke', version='v1')
    )

    assert namespace == 'research'
    assert dataset_name == 'manual_terminal_smoke'
    assert release_version == 'v1'
    assert split is None


def test_resolve_publish_target_rejects_conflicting_flags():
    with pytest.raises(ValueError, match='conflicts with --namespace'):
        _resolve_publish_target(_publish_args(target='research/manual_terminal_smoke@v1', namespace='nexus-curated'))

    with pytest.raises(ValueError, match='conflicts with --version'):
        _resolve_publish_target(_publish_args(target='research/manual_terminal_smoke@v1', version='v2'))


def test_resolve_publish_target_requires_version():
    with pytest.raises(ValueError, match='namespace is required'):
        _resolve_publish_target(_publish_args())

    with pytest.raises(ValueError, match='must include @version'):
        _resolve_publish_target(_publish_args(target='research/manual_terminal_smoke'))

    with pytest.raises(ValueError, match='version is required'):
        _resolve_publish_target(_publish_args(namespace='research', name='manual_terminal_smoke'))


def test_resolve_publish_target_requires_namespace():
    with pytest.raises(ValueError, match='must include namespace'):
        _resolve_publish_target(_publish_args(target='manual_terminal_smoke@v1'))

    with pytest.raises(ValueError, match='namespace is required'):
        _resolve_publish_target(_publish_args(name='manual_terminal_smoke', version='v1'))


def test_dataset_namespace_parser_accepts_repeated_admins():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_dataset_parser(subparsers)

    args = parser.parse_args(
        [
            'dataset',
            'namespace',
            'update',
            'research',
            '--admin',
            'alice',
            '--admin',
            'bob',
            '--json',
        ]
    )

    assert args.dataset_command == 'namespace'
    assert args.namespace_command == 'update'
    assert args.namespace == 'research'
    assert args.admin == ['alice', 'bob']
    assert args.json is True


def test_dataset_publish_parser_accepts_auth_arguments(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_dataset_parser(subparsers)
    token_file = tmp_path / 'taihu'
    token_file.write_text('file-token\n', encoding='utf-8')

    args = parser.parse_args(
        [
            'dataset',
            'publish',
            str(tmp_path),
            'research/my_dataset@v1',
            '--taihu-token-file',
            str(token_file),
        ]
    )

    assert args.dataset_command == 'publish'
    assert args.taihu_token is None
    assert args.taihu_token_file == token_file


def test_dataset_items_parser_accepts_details() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    register_dataset_parser(subparsers)

    args = parser.parse_args(
        [
            'dataset',
            'items',
            'research/my_dataset@v1',
            '--details',
            '--json',
        ]
    )

    assert args.dataset_command == 'items'
    assert args.details is True
    assert args.json is True


def _full_publish_args(**kwargs):
    values = {
        'path': 'local/dataset',
        'target': 'research/my_dataset@v1',
        'namespace': None,
        'name': None,
        'version': None,
        'split': None,
        'output': None,
        'snapshot': None,
        'with_parquet': None,
        'build_only': True,
        'expires': None,
        'register_asset': False,
        'upload_concurrency': 5,
        'creator': None,
        'admin': [],
        'visibility': 'PRIVATE',
        'description': None,
        'json': True,
        'taihu_token': None,
        'taihu_token_file': None,
    }
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_dataset_publish_forwards_taihu_token_as_authenticator(monkeypatch, capsys):
    captured = {}

    class _FakeBuild:
        namespace = 'research'
        dataset = 'my_dataset'
        snapshot = 'dev-1'
        output_dir = 'out'
        files_root = 'out/files'
        files: list = []
        item_count = 0

    class _FakeResult:
        build = _FakeBuild()
        uploaded = 0

    def _fake_publish_local_dataset(path, **kwargs):
        captured['authenticator'] = kwargs.get('authenticator')
        return 'coro-sentinel'

    def _fake_run_publish(coro):
        assert coro == 'coro-sentinel'
        return _FakeResult()

    monkeypatch.setattr('nexus.dataset.publisher.publish_local_dataset', _fake_publish_local_dataset)
    monkeypatch.setattr('nexus.dataset.publisher.run_publish', _fake_run_publish)

    rc = dataset_publish(_full_publish_args(taihu_token='inline-token'))

    assert rc == 0
    authenticator = captured['authenticator']
    assert isinstance(authenticator, TaiPatAuthenticator)
    assert authenticator.pat == 'inline-token'
    capsys.readouterr()


def test_dataset_publish_without_token_passes_no_authenticator(monkeypatch, capsys):
    captured = {}

    class _FakeBuild:
        namespace = 'research'
        dataset = 'my_dataset'
        snapshot = 'dev-1'
        output_dir = 'out'
        files_root = 'out/files'
        files: list = []
        item_count = 0

    class _FakeResult:
        build = _FakeBuild()
        uploaded = 0

    def _fake_publish_local_dataset(path, **kwargs):
        captured['authenticator'] = kwargs.get('authenticator')
        return 'coro-sentinel'

    monkeypatch.setattr('nexus.dataset.publisher.publish_local_dataset', _fake_publish_local_dataset)
    monkeypatch.setattr('nexus.dataset.publisher.run_publish', lambda coro: _FakeResult())

    rc = dataset_publish(_full_publish_args())

    assert rc == 0
    assert captured['authenticator'] is None
    capsys.readouterr()


@pytest.mark.parametrize('command', ['create', 'update'])
def test_dataset_namespace_writes_permissions(monkeypatch, capsys, command):
    _FakeNamespaceDatasetClient.calls = []
    monkeypatch.setattr('nexus.dataset.client.DatasetClient', _FakeNamespaceDatasetClient)

    rc = dataset_namespace(
        argparse.Namespace(
            namespace_command=command,
            namespace='research',
            admin=['alice', 'bob'],
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        'schema': 'nexus',
        'namespace': 'research',
        'admin_list': ['alice', 'bob'],
        'creator': 'alice',
    }
    assert _FakeNamespaceDatasetClient.calls == [(command, 'research', ['alice', 'bob'])]


def test_dataset_namespace_show_reports_missing_record(monkeypatch, capsys):
    _FakeNamespaceDatasetClient.calls = []
    _FakeNamespaceDatasetClient.permission = None
    monkeypatch.setattr('nexus.dataset.client.DatasetClient', _FakeNamespaceDatasetClient)

    rc = dataset_namespace(argparse.Namespace(namespace_command='show', namespace='missing', json=True))

    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {
        'schema': 'nexus',
        'namespace': 'missing',
        'exists': False,
    }
    assert _FakeNamespaceDatasetClient.calls == [('show', 'missing')]


def test_dataset_inspection_commands_use_registered_source(capsys):
    closed: list[str] = []

    class ExtensionReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                provider='cli-test',
                resolved_reference='cli-test://fixture@resolved',
                name='fixture',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test', item_count=1)],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            assert split == 'test'
            assert limit == 1
            yield 'extension-case'

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            assert split == 'test'
            assert item_id == 'extension-case'
            yield DatasetItem(
                id='extension-case',
                body={'task_name': 'echo', 'task_config': {'message': 'hello'}},
                metadata={'provider': 'extension'},
            )

        async def _close(self, exc_type, exc_val, exc_tb):
            closed.append('closed')

    @dataset_source('cli-test')
    def factory(uri: DatasetSourceURI):
        assert uri.display == 'cli-test://fixture'
        return ExtensionReader(uri)

    common = {'dataset': 'cli-test://fixture', 'taihu_token': None, 'taihu_token_file': None}
    try:
        assert dataset_show(argparse.Namespace(**common, json=True)) == 0
        show_payload = json.loads(capsys.readouterr().out)

        assert dataset_items(argparse.Namespace(**common, split='test', limit=1, json=True)) == 0
        items_payload = json.loads(capsys.readouterr().out)

        assert dataset_get(argparse.Namespace(**common, split='test', item_id='extension-case', output=None)) == 0
        get_payload = json.loads(capsys.readouterr().out)
    finally:
        dataset_source_registry._registry.pop('cli-test', None)

    assert show_payload['provider'] == 'cli-test'
    assert show_payload['resolved_reference'] == 'cli-test://fixture@resolved'
    assert items_payload['items'] == [{'id': 'extension-case'}]
    assert get_payload['id'] == 'extension-case'
    assert get_payload['metadata'] == {'provider': 'extension'}
    assert closed == ['closed', 'closed', 'closed']


def test_dataset_versions_rejects_local_and_extension_readers(tmp_path, caplog):
    local_root = tmp_path / 'dataset'
    local_root.mkdir()
    (local_root / 'dataset.yaml').write_text(
        yaml.safe_dump({'schema_version': 1, 'name': 'fixture', 'splits': {'test': {'items': []}}}),
        encoding='utf-8',
    )
    common = {'json': True, 'taihu_token': None, 'taihu_token_file': None}

    assert dataset_versions(argparse.Namespace(dataset=str(local_root), **common)) == 1
    assert dataset_versions(argparse.Namespace(dataset='reader-test://fixture', **common)) == 1

    messages = [record.getMessage() for record in caplog.records]
    assert sum('dataset_versions_unsupported' in message for message in messages) == 2


def test_dataset_fetch_downloads_complete_zip_cache(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr('nexus.dataset.cache.DEFAULT_DATASET_CACHE_DIR', cache_dir)
    monkeypatch.setattr('nexus.dataset.client.DatasetClient', _FakeDatasetClient)

    rc = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id='case-001',
            limit=None,
            output=None,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['dataset'] == 'research/my_dataset@v1'
    root = cache_dir / 'research' / 'my_dataset' / 'v1'
    assert payload['cache_dir'] == str(root)
    assert payload['output_dir'] == str(root)
    assert payload['splits'] == ['test']
    assert payload['count'] == 1
    assert set(read_zip_index_from_file(root / 'items/test/case-001/files.zip').entries) == {'payload.txt'}

    dataset_yaml = yaml.safe_load((root / 'dataset.yaml').read_text(encoding='utf-8'))
    assert dataset_yaml['splits']['test']['items'] == ['items/test/case-001/item.yaml']

    item_yaml = yaml.safe_load((root / 'items/test/case-001/item.yaml').read_text(encoding='utf-8'))
    assert item_yaml['id'] == 'case-001'
    assert item_yaml['files']['type'] == 'zip'
    assert item_yaml['files']['path'] == 'files.zip'
    assert 'remote' not in item_yaml['files']
    assert 'task_files' not in item_yaml


def test_dataset_fetch_downloads_to_output_dir(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / 'cache'
    output_dir = tmp_path / 'downloaded_dataset'
    monkeypatch.setattr('nexus.dataset.cache.DEFAULT_DATASET_CACHE_DIR', cache_dir)
    monkeypatch.setattr('nexus.dataset.client.DatasetClient', _FakeDatasetClient)

    rc = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id='case-001',
            limit=None,
            output=str(output_dir),
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['cache_dir'] == str(cache_dir / 'research' / 'my_dataset' / 'v1')
    assert payload['output_dir'] == str(output_dir.resolve())
    assert not (cache_dir / 'research' / 'my_dataset' / 'v1' / 'dataset.yaml').exists()
    assert set(read_zip_index_from_file(output_dir / 'items/test/case-001/files.zip').entries) == {'payload.txt'}
    dataset_yaml = yaml.safe_load((output_dir / 'dataset.yaml').read_text(encoding='utf-8'))
    assert dataset_yaml['splits']['test']['items'] == ['items/test/case-001/item.yaml']


def test_dataset_fetch_downloads_all_splits_to_one_root(monkeypatch, tmp_path, capsys):
    cache_dir = tmp_path / 'cache'
    monkeypatch.setattr('nexus.dataset.cache.DEFAULT_DATASET_CACHE_DIR', cache_dir)
    monkeypatch.setattr('nexus.dataset.client.DatasetClient', _FakeMultiSplitDatasetClient)

    rc = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=None,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    root = cache_dir / 'research' / 'my_dataset' / 'v1'
    assert payload['cache_dir'] == str(root)
    assert payload['output_dir'] == str(root)
    assert payload['splits'] == ['train', 'test']
    assert payload['count'] == 2

    dataset_yaml = yaml.safe_load((root / 'dataset.yaml').read_text(encoding='utf-8'))
    assert dataset_yaml['splits']['train']['items'] == ['items/train/train-001/item.yaml']
    assert dataset_yaml['splits']['test']['items'] == ['items/test/test-001/item.yaml']
    assert set(read_zip_index_from_file(root / 'items/train/train-001/files.zip').entries) == {'payload.txt'}
    assert set(read_zip_index_from_file(root / 'items/test/test-001/files.zip').entries) == {'payload.txt'}


def test_dataset_fetch_pins_symbolic_version_before_loading_items(monkeypatch, tmp_path):
    class PinningClient:
        def __init__(self, **kwargs):
            self.resolve_calls = 0
            self.load_versions: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            self.resolve_calls += 1
            assert uri.version == 'latest'
            return 'release-7'

        async def list_splits(self, uri, *, version=None):
            assert uri.version == 'latest'
            assert version == 'release-7'
            return ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            self.load_versions.append(uri.version)
            assert uri.version == 'release-7'
            assert split == 'test'
            return [DatasetItem(id='case-001', body={'task_name': 'echo'})]

    client_holder: dict[str, PinningClient] = {}

    def factory(**kwargs):
        client = PinningClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    output = tmp_path / 'dataset'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@latest',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    assert client_holder['client'].resolve_calls == 1
    assert client_holder['client'].load_versions == ['release-7']
    assert (output / 'items/test/case-001/item.yaml').is_file()


def test_dataset_fetch_source_honors_uri_split(tmp_path):
    seen_splits: list[str | None] = []

    class SplitAwareReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='split-aware',
                name='split-aware',
                default_split='default',
                splits=[DatasetSplitDescriptor(name='default'), DatasetSplitDescriptor(name='selected')],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            yield 'item'

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            seen_splits.append(split)
            assert split == 'selected'
            yield DatasetItem(id='item', body={'task_name': 'echo'})

        async def _close(self, exc_type, exc_val, exc_tb):
            return None

    @dataset_source('split-aware')
    def factory(uri: DatasetSourceURI):
        return SplitAwareReader(uri)

    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='split-aware://fixture?split=selected',
                split=None,
                id=None,
                limit=None,
                output=str(tmp_path / 'dataset'),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('split-aware', None)

    assert result == 0
    assert seen_splits == ['selected']


def test_dataset_fetch_source_rejects_symlink_output(tmp_path):
    class SourceReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='symlink-source',
                name='symlink-source',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test')],
            )

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            yield DatasetItem(id='case-001', body={'task_name': 'echo'})

    @dataset_source('symlink-source')
    def factory(uri: DatasetSourceURI):
        return SourceReader(uri)

    target = tmp_path / 'real-output'
    target.mkdir()
    output = tmp_path / 'output-link'
    output.symlink_to(target, target_is_directory=True)
    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='symlink-source://fixture',
                split=None,
                id=None,
                limit=None,
                output=str(output),
                force=True,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('symlink-source', None)

    assert result == 1
    assert not (target / 'dataset.yaml').exists()


def test_dataset_fetch_source_rejects_ancestor_symlink_output(tmp_path):
    class SourceReader(BaseDatasetReader):
        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference='ancestor-symlink://fixture',
                resolved_reference='ancestor-symlink://fixture',
                provider='ancestor-symlink',
                name='ancestor-symlink',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test')],
            )

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            yield DatasetItem(id='case-001', body={'task_name': 'echo'})

    @dataset_source('ancestor-symlink')
    def factory(uri: DatasetSourceURI):
        return SourceReader()

    outside = tmp_path / 'outside'
    outside.mkdir()
    ancestor = tmp_path / 'ancestor-link'
    ancestor.symlink_to(outside, target_is_directory=True)
    output = ancestor / 'nested'
    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='ancestor-symlink://fixture',
                split=None,
                id=None,
                limit=None,
                output=str(output),
                force=True,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('ancestor-symlink', None)

    assert result == 1
    assert not (outside / 'nested').exists()


def test_dataset_fetch_source_exports_all_descriptor_splits_by_default(tmp_path):
    seen_splits: list[str | None] = []

    class MultiSplitReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='multi-split-source',
                name='multi-split',
                default_split='train',
                splits=[DatasetSplitDescriptor(name='train'), DatasetSplitDescriptor(name='test')],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            yield f'{split}-item'

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            seen_splits.append(split)
            yield DatasetItem(id=f'{split}-item', body={'task_name': 'echo'})

    @dataset_source('multi-split-source')
    def factory(uri: DatasetSourceURI):
        return MultiSplitReader(uri)

    output = tmp_path / 'dataset'
    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='multi-split-source://fixture',
                split=None,
                id=None,
                limit=None,
                output=str(output),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('multi-split-source', None)

    assert result == 0
    assert seen_splits == ['train', 'test']
    dataset_yaml = yaml.safe_load((output / 'dataset.yaml').read_text(encoding='utf-8'))
    assert set(dataset_yaml['splits']) == {'train', 'test'}
    assert (output / 'items/train/train-item/item.yaml').is_file()
    assert (output / 'items/test/test-item/item.yaml').is_file()


def test_dataset_fetch_source_rejects_missing_id_after_searching_all_splits(tmp_path, caplog):
    seen_splits: list[str | None] = []

    class MissingReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='missing-source',
                name='missing-source',
                default_split='train',
                splits=[DatasetSplitDescriptor(name='train'), DatasetSplitDescriptor(name='test')],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            return
            yield  # pragma: no cover

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            seen_splits.append(split)
            from nexus.dataset.reader import DatasetReaderError

            raise DatasetReaderError('dataset_item_not_found', detail=item_id)
            yield  # pragma: no cover

    @dataset_source('missing-source')
    def factory(uri: DatasetSourceURI):
        return MissingReader(uri)

    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='missing-source://fixture',
                split=None,
                id='missing',
                limit=None,
                output=str(tmp_path / 'dataset'),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('missing-source', None)

    assert result == 1
    assert seen_splits == ['train', 'test']
    assert any('dataset_item_not_found' in record.getMessage() for record in caplog.records)


def test_dataset_fetch_source_limit_zero_writes_empty_selected_splits_without_reader_call(tmp_path):
    calls: list[str | None] = []

    class ZeroReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='zero-source',
                name='zero-source',
                default_split='train',
                splits=[DatasetSplitDescriptor(name='train'), DatasetSplitDescriptor(name='test')],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            return
            yield  # pragma: no cover

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            calls.append(split)
            raise AssertionError('limit=0 must not load Reader items')
            yield  # pragma: no cover

    @dataset_source('zero-source')
    def factory(uri: DatasetSourceURI):
        return ZeroReader(uri)

    output = tmp_path / 'dataset'
    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='zero-source://fixture',
                split=None,
                id=None,
                limit=0,
                output=str(output),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
        missing_result = dataset_fetch(
            argparse.Namespace(
                dataset='zero-source://fixture',
                split='missing',
                id=None,
                limit=0,
                output=str(tmp_path / 'missing-split'),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('zero-source', None)

    assert result == 0
    assert missing_result == 1
    assert calls == []
    dataset_yaml = yaml.safe_load((output / 'dataset.yaml').read_text(encoding='utf-8'))
    assert set(dataset_yaml['splits']) == {'train', 'test'}
    assert dataset_yaml['splits']['train']['items'] == []
    assert dataset_yaml['splits']['test']['items'] == []


def test_dataset_fetch_rejects_invalid_limit_before_remote_access(monkeypatch, tmp_path):
    called = False

    class UnexpectedClient:
        def __init__(self, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', UnexpectedClient)
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=-1,
            output=str(tmp_path / 'dataset'),
            force=False,
            json=True,
            taihu_token=None,
            taihu_token_file=None,
        )
    )

    assert result == 1
    assert called is False
    assert not (tmp_path / 'dataset').exists()


def test_dataset_fetch_rejects_unsafe_uri_before_resolving_remote_version(monkeypatch, tmp_path):
    class UnexpectedClient:
        resolved = False

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            self.resolved = True
            raise AssertionError('unsafe URI must be rejected before provider access')

    client_holder: dict[str, UnexpectedClient] = {}

    def factory(**kwargs):
        client = UnexpectedClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@../escape',
            split=None,
            id=None,
            limit=None,
            output=str(tmp_path / 'dataset'),
            json=True,
        )
    )

    assert result == 1
    assert client_holder == {}
    assert not (tmp_path / 'dataset').exists()


def test_dataset_fetch_rejects_missing_explicit_local_path(monkeypatch, tmp_path):
    called = False

    class UnexpectedClient:
        def __init__(self, **kwargs):
            nonlocal called
            called = True

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', UnexpectedClient)
    missing = tmp_path / 'missing-dataset'
    result = dataset_fetch(
        argparse.Namespace(
            dataset=str(missing),
            split=None,
            id=None,
            limit=None,
            output=str(tmp_path / 'output'),
            force=False,
            json=True,
            taihu_token=None,
            taihu_token_file=None,
        )
    )

    assert result == 1
    assert called is False


@pytest.mark.parametrize('unsafe_component', ['../../../outside', '..\\..\\outside'])
def test_dataset_fetch_rejects_unsafe_remote_path_components(monkeypatch, tmp_path, unsafe_component):
    class UnsafeClient:
        def __init__(self, **kwargs):
            self.mode = 'split' if unsafe_component.startswith('../') else 'item'

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return [unsafe_component] if self.mode == 'split' else ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            item_id = unsafe_component if self.mode == 'item' else 'safe-item'
            return [DatasetItem(id=item_id, body={'task_name': 'echo'})]

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', UnsafeClient)
    output = tmp_path / 'output'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 1
    assert not output.exists()
    assert not (tmp_path / 'outside').exists()


def test_dataset_fetch_rejects_symlinked_remote_item_directory(monkeypatch, tmp_path):
    class SymlinkClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            return [DatasetItem(id='case-001', body={'task_name': 'echo'})]

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', SymlinkClient)
    output = tmp_path / 'output'
    escape = tmp_path / 'escape'
    escape.mkdir()
    link = output / 'items' / 'test' / 'case-001'
    link.parent.mkdir(parents=True)
    link.symlink_to(escape, target_is_directory=True)

    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 1
    assert not (escape / 'item.yaml').exists()


def test_dataset_fetch_remote_honors_uri_split_and_sanitizes_item(monkeypatch, tmp_path):
    seen_splits: list[str | None] = []

    class SplitClient:
        def __init__(self, **kwargs):
            self.io = _FakeDatasetIo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            assert uri.split == 'test'
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            raise AssertionError('URI split should avoid listing all splits')

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            seen_splits.append(split)
            return [
                DatasetItem(
                    id='case-001',
                    body={
                        'task_name': 'echo',
                        'task_config': {
                            'task_dir': '/submitter/private',
                            'task_files': {'type': 'directory', 'path': '/submitter/private/files'},
                        },
                    },
                    metadata={'source_path': '/submitter/private', 'nested': {'source_path': '/submitter/private/x'}},
                    files={'type': 'zip', 'path': 'blobs/case.zip'},
                    remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split='test'),
                )
            ]

        async def sign_file_urls(self, uri, files, *, split=None):
            return {files[0]: 'https://signed.example/files.zip'}

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', SplitClient)
    output = tmp_path / 'dataset'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1?split=test',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    assert seen_splits == ['test']
    item_yaml = (output / 'items/test/case-001/item.yaml').read_text(encoding='utf-8')
    assert '/submitter/private' not in item_yaml


def test_dataset_fetch_rejects_remote_item_when_id_is_missing_from_every_split(monkeypatch, tmp_path, caplog):
    class MissingClient:
        def __init__(self, **kwargs):
            self.calls: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['train', 'test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            self.calls.append(str(split))
            raise KeyError(item_id)

    client_holder: dict[str, MissingClient] = {}

    def factory(**kwargs):
        client = MissingClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id='missing',
            limit=None,
            output=str(tmp_path / 'output'),
            json=True,
        )
    )

    assert result == 1
    assert client_holder['client'].calls == ['train', 'test']
    assert any('dataset_item_not_found' in record.getMessage() for record in caplog.records)


def test_dataset_fetch_remote_id_search_skips_missing_split(monkeypatch, tmp_path):
    class SplitClient:
        def __init__(self, **kwargs):
            self.calls: list[str] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['train', 'test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            self.calls.append(str(split))
            if split == 'train':
                raise KeyError(item_id)
            return [DatasetItem(id=item_id, body={'task_name': 'echo'})]

    client_holder: dict[str, SplitClient] = {}

    def factory(**kwargs):
        client = SplitClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    output = tmp_path / 'output'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id='case-001',
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    assert client_holder['client'].calls == ['train', 'test']
    assert (output / 'items/test/case-001/item.yaml').is_file()


def test_dataset_fetch_remote_limit_zero_writes_empty_dataset_without_provider_call(monkeypatch, tmp_path):
    class ZeroClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['train', 'test']

        async def load_items(self, *args, **kwargs):
            raise AssertionError('limit=0 must not call DatasetClient.load_items')

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', ZeroClient)
    output = tmp_path / 'output'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=0,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    dataset_yaml = yaml.safe_load((output / 'dataset.yaml').read_text(encoding='utf-8'))
    assert dataset_yaml['default_split'] == 'train'
    assert set(dataset_yaml['splits']) == {'train', 'test'}
    assert dataset_yaml['splits']['train']['items'] == []
    assert dataset_yaml['splits']['test']['items'] == []

    missing_result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split='missing',
            id=None,
            limit=0,
            output=str(tmp_path / 'missing-split'),
            json=True,
        )
    )
    assert missing_result == 1


def test_dataset_fetch_remote_limit_stops_before_next_split(monkeypatch, tmp_path):
    class LimitedClient:
        def __init__(self, **kwargs):
            self.calls: list[tuple[str, int | None]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['train', 'test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            self.calls.append((str(split), limit))
            if limit == 0:
                raise AssertionError('positive global limit must stop before a zero-limit call')
            return [DatasetItem(id='train-001', body={'task_name': 'echo'})]

    client_holder: dict[str, LimitedClient] = {}

    def factory(**kwargs):
        client = LimitedClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    output = tmp_path / 'output'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=1,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    assert client_holder['client'].calls == [('train', 1)]
    assert (output / 'items/train/train-001/item.yaml').is_file()


def test_dataset_fetch_rejects_malformed_remote_zip_without_identity(monkeypatch, tmp_path):
    class InvalidZipIo:
        async def download_file(self, signed_url, target):
            target.write_bytes(b'not-a-nexus-zip')

    class InvalidZipClient:
        def __init__(self, **kwargs):
            self.io = InvalidZipIo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            return [
                DatasetItem.from_mapping(
                    {
                        'id': 'case-001',
                        'files': {'type': 'zip', 'path': 'blobs/case-001.zip'},
                        'body': {'task_name': 'echo'},
                    },
                    remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split='test'),
                )
            ]

        async def sign_file_urls(self, uri, files, *, split=None):
            return {files[0]: 'https://signed.example/files.zip'}

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', InvalidZipClient)
    output = tmp_path / 'output'
    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 1
    assert not (output / 'items/test/case-001/files.zip').exists()
    assert not (output / 'items/test/case-001/item.yaml').exists()


def test_dataset_fetch_preserves_existing_zip_when_download_identity_fails(monkeypatch, tmp_path):
    class WrongHashIo:
        async def download_file(self, signed_url, target):
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                (source / 'payload.txt').write_text('new payload', encoding='utf-8')
                write_nexus_zip_directory_to_file(source, target)

    class WrongHashClient:
        def __init__(self, **kwargs):
            self.io = WrongHashIo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            return [
                DatasetItem.from_mapping(
                    {
                        'id': 'case-001',
                        'files': {
                            'type': 'zip',
                            'path': 'blobs/case-001.zip',
                            'sha256': '0' * 64,
                        },
                        'body': {'task_name': 'echo'},
                    },
                    remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split='test'),
                )
            ]

        async def sign_file_urls(self, uri, files, *, split=None):
            return {files[0]: 'https://signed.example/files.zip'}

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', WrongHashClient)
    output = tmp_path / 'output'
    existing = output / 'items/test/case-001/files.zip'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'keep this old cache')

    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 1
    assert existing.read_bytes() == b'keep this old cache'
    assert not (output / 'items/test/case-001/item.yaml').exists()


def test_dataset_fetch_does_not_trust_malformed_cached_zip(monkeypatch, tmp_path):
    class ValidIo:
        calls = 0

        async def download_file(self, signed_url, target):
            self.calls += 1
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory)
                (source / 'payload.txt').write_text('cached replacement', encoding='utf-8')
                write_nexus_zip_directory_to_file(source, target)

    class CachedClient:
        def __init__(self, **kwargs):
            self.io = ValidIo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def resolve_version(self, uri):
            return 'v1'

        async def list_splits(self, uri, *, version=None):
            return ['test']

        async def load_items(self, uri, *, split=None, item_id=None, limit=None):
            return [
                DatasetItem.from_mapping(
                    {
                        'id': 'case-001',
                        'files': {'type': 'zip', 'path': 'blobs/case-001.zip'},
                        'body': {'task_name': 'echo'},
                    },
                    remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split='test'),
                )
            ]

        async def sign_file_urls(self, uri, files, *, split=None):
            return {files[0]: 'https://signed.example/files.zip'}

    client_holder: dict[str, CachedClient] = {}

    def factory(**kwargs):
        client = CachedClient(**kwargs)
        client_holder['client'] = client
        return client

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', factory)
    output = tmp_path / 'output'
    cached_zip = output / 'items' / 'test' / 'case-001' / 'files.zip'
    cached_zip.parent.mkdir(parents=True)
    cached_zip.write_bytes(b'corrupt-cache')

    result = dataset_fetch(
        argparse.Namespace(
            dataset='research/my_dataset@v1',
            split=None,
            id=None,
            limit=None,
            output=str(output),
            json=True,
        )
    )

    assert result == 0
    assert client_holder['client'].io.calls == 1
    assert set(read_zip_index_from_file(cached_zip).entries) == {'payload.txt'}


def test_dataset_fetch_source_repackages_remote_files_without_source_identity_comparison(monkeypatch, tmp_path):
    class RemoteFiles:
        async def list(self, pattern='**/*'):
            assert pattern == '**/*'
            return ['payload.txt']

        async def download(self, pattern, directory):
            root = Path(directory)
            (root / 'payload.txt').write_text('remote payload', encoding='utf-8')

    monkeypatch.setattr('nexus.dataset.task_files.task_files_from_spec', lambda *args, **kwargs: RemoteFiles())

    class RemoteReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                resolved_reference=self.uri.display,
                provider='remote-export',
                name='remote-export',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test')],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            yield 'case-001'

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            yield DatasetItem(
                id='case-001',
                body={'task_name': 'echo'},
                files={
                    'type': 'zip',
                    'path': 'remote/case-001.zip',
                    'sha256': '0' * 64,
                    'size': 1,
                    'remote': {
                        'namespace': 'research',
                        'dataset': 'remote-export',
                        'version': 'v1',
                        'split': 'test',
                    },
                },
            )

    @dataset_source('remote-export')
    def factory(uri: DatasetSourceURI):
        return RemoteReader(uri)

    output = tmp_path / 'dataset'
    try:
        result = dataset_fetch(
            argparse.Namespace(
                dataset='remote-export://fixture',
                split=None,
                id=None,
                limit=None,
                output=str(output),
                force=False,
                json=True,
                taihu_token=None,
                taihu_token_file=None,
            )
        )
    finally:
        dataset_source_registry._registry.pop('remote-export', None)

    assert result == 0
    files_zip = output / 'items/test/case-001/files.zip'
    assert set(read_zip_index_from_file(files_zip).entries) == {'payload.txt'}


@pytest.mark.asyncio
async def test_reader_task_files_rejects_empty_local_path(tmp_path):
    with pytest.raises(ValueError, match='non-empty'):
        await _write_reader_task_files(
            TaskFilesSpec(type='zip', path=''),
            tmp_path / 'files.zip',
        )


@pytest.mark.asyncio
async def test_reader_task_files_exports_empty_remote_zip(monkeypatch, tmp_path):
    class EmptyRemoteFiles:
        async def list(self, pattern='**/*'):
            assert pattern == '**/*'
            return []

        async def download(self, pattern, directory):
            raise AssertionError(f'download must not be called for an empty archive: {pattern} {directory}')

    spec = TaskFilesSpec(
        type='zip',
        path='remote/empty.zip',
        remote=RemoteDatasetRef(namespace='research', dataset='fixtures', version='v1', split='test'),
    )
    output = tmp_path / 'empty.zip'

    # Keep the test at the export boundary: remote signing/download is already
    # covered separately, while this verifies that a valid empty archive is
    # represented as an empty portable Nexus ZIP.
    monkeypatch.setattr('nexus.dataset.task_files.task_files_from_spec', lambda *args, **kwargs: EmptyRemoteFiles())
    await _write_reader_task_files(spec, output)

    assert read_zip_index_from_file(output).entries == {}


class _FakeDatasetClient:
    def __init__(self, **kwargs):
        self.io = _FakeDatasetIo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def resolve_version(self, uri):
        assert uri.display == 'research/my_dataset@v1'
        return 'v1'

    async def list_splits(self, uri, *, version=None):
        assert uri.display == 'research/my_dataset@v1'
        assert version == 'v1'
        return ['test']

    async def load_items(self, uri, *, split=None, item_id=None, limit=None):
        assert uri.display == 'research/my_dataset@v1'
        assert split == 'test'
        assert item_id == 'case-001'
        assert limit is None
        return [
            DatasetItem.from_mapping(
                {
                    'id': 'case-001',
                    'files': {
                        'type': 'zip',
                        'path': 'blobs/case-001.zip',
                    },
                    'body': {'task_name': 'echo', 'task_config': {'message': 'hello'}},
                },
                remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split='test'),
            )
        ]

    async def sign_file_urls(self, uri, files, *, split=None):
        assert uri.display == 'research/my_dataset@v1?split=test'
        assert files == ['blobs/case-001.zip']
        assert split == 'test'
        return {'blobs/case-001.zip': 'https://signed.example/files.zip'}


class _FakeNamespaceDatasetClient:
    calls = []
    permission = None

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def create_namespace_permission(self, namespace, admin_list):
        self.calls.append(('create', namespace, admin_list))
        return _namespace_permission(namespace, admin_list)

    async def get_namespace_permission(self, namespace):
        self.calls.append(('show', namespace))
        return self.permission

    async def update_namespace_permission(self, namespace, admin_list):
        self.calls.append(('update', namespace, admin_list))
        return _namespace_permission(namespace, admin_list)


class _FakeDatasetIo:
    async def download_file(self, signed_url, target):
        assert signed_url == 'https://signed.example/files.zip'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / 'payload.txt').write_text('zip-bytes', encoding='utf-8')
            write_nexus_zip_directory_to_file(source, target)


def _namespace_permission(namespace, admin_list):
    return {
        'schema': 'nexus',
        'namespace': namespace,
        'admin_list': admin_list,
        'creator': 'alice',
    }


class _FakeMultiSplitDatasetClient:
    def __init__(self, **kwargs):
        self.io = _FakeDatasetIo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def resolve_version(self, uri):
        assert uri.display == 'research/my_dataset@v1'
        return 'v1'

    async def list_splits(self, uri, *, version=None):
        assert uri.display == 'research/my_dataset@v1'
        assert version == 'v1'
        return ['train', 'test']

    async def load_items(self, uri, *, split=None, item_id=None, limit=None):
        assert uri.display == 'research/my_dataset@v1'
        assert split in {'train', 'test'}
        assert item_id is None
        assert limit is None
        item_id = f'{split}-001'
        return [
            DatasetItem.from_mapping(
                {
                    'id': item_id,
                    'files': {
                        'type': 'zip',
                        'path': f'blobs/{item_id}.zip',
                    },
                    'body': {'task_name': 'echo', 'task_config': {'message': f'hello {split}'}},
                },
                remote_ref=RemoteDatasetRef(namespace='research', dataset='my_dataset', version='v1', split=split),
            )
        ]

    async def sign_file_urls(self, uri, files, *, split=None):
        assert uri.display == f'research/my_dataset@v1?split={split}'
        assert files == [f'blobs/{split}-001.zip']
        return {files[0]: 'https://signed.example/files.zip'}
