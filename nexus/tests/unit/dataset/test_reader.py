from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from nexus.core.module_meta import ModuleMeta, register_objects_from_meta
from nexus.dataset.exceptions import DatasetClientError
from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.reader import (
    BaseDatasetReader,
    DatasetDescriptor,
    DatasetReaderError,
    DatasetSplitDescriptor,
    load_items,
    open_dataset,
)
from nexus.dataset.source import DatasetSourceURI, dataset_source, dataset_source_registry


def _write_dataset(
    root: Path,
    *,
    splits: dict[str, list[tuple[str, str]]],
    default_split: str | None = None,
) -> Path:
    dataset: dict[str, Any] = {
        'schema_version': 1,
        'name': 'reader-fixture',
        'splits': {},
    }
    if default_split is not None:
        dataset['default_split'] = default_split
    for split, entries in splits.items():
        paths: list[str] = []
        for directory, item_id in entries:
            item_path = root / 'items' / split / directory / 'item.yaml'
            item_path.parent.mkdir(parents=True, exist_ok=True)
            item_path.write_text(
                yaml.safe_dump(
                    {
                        'id': item_id,
                        'body': {'task_name': 'echo', 'task_config': {'message': item_id}},
                        'metadata': {'split': split},
                    },
                    sort_keys=False,
                ),
                encoding='utf-8',
            )
            paths.append(item_path.relative_to(root).as_posix())
        dataset['splits'][split] = {'items': paths}
    (root / 'dataset.yaml').parent.mkdir(parents=True, exist_ok=True)
    (root / 'dataset.yaml').write_text(yaml.safe_dump(dataset, sort_keys=False), encoding='utf-8')
    return root


def _assert_code(exc_info: pytest.ExceptionInfo[DatasetReaderError], code: str) -> None:
    assert exc_info.value.code == code
    assert str(exc_info.value).startswith(code)


@pytest.mark.asyncio
async def test_local_reader_describes_and_lists_ids_without_loading_item_bodies(tmp_path, monkeypatch):
    root = _write_dataset(
        tmp_path / 'dataset',
        splits={'train': [('first', 'case-b'), ('second', 'case-a')]},
        default_split='train',
    )

    def fail_from_file(*args, **kwargs):
        raise AssertionError('ID enumeration must not construct DatasetItem bodies')

    monkeypatch.setattr(DatasetItem, 'from_file', fail_from_file)

    async with open_dataset(root) as reader:
        descriptor = await reader.describe()
        item_ids = [item_id async for item_id in reader.iter_item_ids()]

    assert descriptor == DatasetDescriptor(
        reference=str(root),
        resolved_reference=str(root.resolve()),
        provider='local',
        name='reader-fixture',
        default_split='train',
        splits=[DatasetSplitDescriptor(name='train', item_count=2)],
        metadata={'consistency': 'live'},
    )
    assert item_ids == ['case-b', 'case-a']


@pytest.mark.asyncio
async def test_local_reader_contract_split_item_limit_order_and_missing(tmp_path):
    root = _write_dataset(
        tmp_path / 'dataset',
        splits={
            'train': [('first', 'train-a'), ('second', 'train-b')],
            'test': [('only', 'test-a')],
        },
    )

    async with open_dataset(root) as reader:
        with pytest.raises(DatasetReaderError) as required:
            _ = [item async for item in reader.iter_items()]
        _assert_code(required, 'dataset_split_required')

        with pytest.raises(DatasetReaderError) as missing_split:
            _ = [item async for item in reader.iter_items(split='missing')]
        _assert_code(missing_split, 'dataset_split_not_found')

        with pytest.raises(DatasetReaderError) as invalid_limit:
            _ = [item async for item in reader.iter_items(split='train', limit=False)]
        _assert_code(invalid_limit, 'dataset_limit_invalid')

        selected = await load_items(reader, split='train', limit=1)
        assert [item.id for item in selected] == ['train-a']

        selected = await load_items(reader, split='train', item_id='train-b', limit=4)
        assert [item.id for item in selected] == ['train-b']

        with pytest.raises(DatasetReaderError) as missing_item:
            _ = [item async for item in reader.iter_items(split='train', item_id='missing')]
        _assert_code(missing_item, 'dataset_item_not_found')


@pytest.mark.asyncio
async def test_local_reader_limit_zero_still_validates_explicit_split(tmp_path):
    root = _write_dataset(tmp_path / 'dataset', splits={'test': [('only', 'case-a')]})

    async with open_dataset(root) as reader:
        with pytest.raises(DatasetReaderError) as ids_error:
            _ = [item_id async for item_id in reader.iter_item_ids(split='missing', limit=0)]
        _assert_code(ids_error, 'dataset_split_not_found')

        with pytest.raises(DatasetReaderError) as items_error:
            _ = [item async for item in reader.iter_items(split='missing', limit=0)]
        _assert_code(items_error, 'dataset_split_not_found')


@pytest.mark.asyncio
async def test_local_reader_single_split_fallback_and_duplicate_detection(tmp_path):
    root = _write_dataset(
        tmp_path / 'dataset',
        splits={'test': [('first', 'duplicate'), ('second', 'duplicate')]},
    )

    async with open_dataset(root) as reader:
        with pytest.raises(DatasetReaderError) as duplicate_ids:
            _ = [item_id async for item_id in reader.iter_item_ids()]
        _assert_code(duplicate_ids, 'dataset_duplicate_item_id')

        with pytest.raises(DatasetReaderError) as duplicate_items:
            _ = [item async for item in reader.iter_items()]
        _assert_code(duplicate_items, 'dataset_duplicate_item_id')

        with pytest.raises(DatasetReaderError) as duplicate_exact_item:
            _ = [item async for item in reader.iter_items(item_id='duplicate')]
        _assert_code(duplicate_exact_item, 'dataset_duplicate_item_id')


@pytest.mark.asyncio
async def test_reader_and_existing_iterator_fail_after_context_closes(tmp_path):
    root = _write_dataset(
        tmp_path / 'dataset',
        splits={'test': [('first', 'case-a'), ('second', 'case-b')]},
    )

    async with open_dataset(root) as reader:
        iterator = reader.iter_items()
        assert (await anext(iterator)).id == 'case-a'

    with pytest.raises(DatasetReaderError) as describe_closed:
        await reader.describe()
    _assert_code(describe_closed, 'dataset_reader_closed')

    with pytest.raises(DatasetReaderError) as iterator_closed:
        await anext(iterator)
    _assert_code(iterator_closed, 'dataset_reader_closed')


class _FakeTaijiClient:
    def __init__(self) -> None:
        self.resolve_version_calls = 0
        self.refs: list[RemoteDatasetRef] = []

    async def resolve_version(self, parsed) -> str:
        self.resolve_version_calls += 1
        return 'release-7'

    async def list_splits(self, parsed, *, version: str | None = None) -> list[str]:
        assert version == 'release-7'
        return ['test']

    async def resolve_split(self, parsed, *, version: str, split: str | None = None) -> str:
        assert version == 'release-7'
        assert split in {None, 'test'}
        return 'test'

    async def _iter_item_pages(self, ref, *, limit: int | None, page_size: int):
        self.refs.append(ref)
        records = [
            {'id': 'remote-a', 'body': {'task_name': 'echo'}},
            {'id': 'remote-b', 'body': {'task_name': 'echo'}},
        ]
        yield records[:limit], []

    async def iter_resolved_items(self, ref, *, limit: int | None = None, page_size: int = 500):
        self.refs.append(ref)
        ids = ['remote-a', 'remote-b']
        for item_id in ids[:limit]:
            yield DatasetItem.from_mapping(
                {'id': item_id, 'body': {'task_name': 'echo'}},
                remote_ref=ref,
            )


@pytest.mark.asyncio
async def test_taiji_reader_maps_io_not_found_to_reader_error() -> None:
    class MissingItemIo:
        async def get_item(self, *args, **kwargs):
            raise DatasetClientError('HTTP 404: item does not exist')

    class MissingItemClient(_FakeTaijiClient):
        def __init__(self) -> None:
            super().__init__()
            self.io = MissingItemIo()

    client = MissingItemClient()
    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        with pytest.raises(DatasetReaderError) as missing_item:
            _ = [item async for item in reader.iter_items(item_id='missing')]

    _assert_code(missing_item, 'dataset_item_not_found')


@pytest.mark.asyncio
async def test_taiji_reader_rejects_io_item_id_mismatch() -> None:
    class MismatchedItemIo:
        async def get_item(self, *args, **kwargs):
            return {'id': 'other', 'body': {'task_name': 'echo'}}

    class MismatchedItemClient(_FakeTaijiClient):
        def __init__(self) -> None:
            super().__init__()
            self.io = MismatchedItemIo()

    async with open_dataset('research/tasks@latest', dataset_client=MismatchedItemClient()) as reader:
        with pytest.raises(DatasetReaderError) as mismatched_item:
            _ = [item async for item in reader.iter_items(item_id='requested')]

    _assert_code(mismatched_item, 'dataset_item_not_found')


@pytest.mark.asyncio
async def test_taiji_reader_rejects_nested_io_item_id_mismatch() -> None:
    class NestedMismatchedItemIo:
        async def get_item(self, *args, **kwargs):
            # The transport wrapper echoes the requested ID, but the nested
            # record identifies a different item.  The reader must not trust
            # the wrapper and return the wrong body.
            return {
                'id': 'requested',
                'record': {'item_id': 'other', 'body': {'task_name': 'echo'}},
            }

    class NestedMismatchedItemClient(_FakeTaijiClient):
        def __init__(self) -> None:
            super().__init__()
            self.io = NestedMismatchedItemIo()

    async with open_dataset('research/tasks@latest', dataset_client=NestedMismatchedItemClient()) as reader:
        with pytest.raises(DatasetReaderError) as mismatched_item:
            _ = [item async for item in reader.iter_items(item_id='requested')]

    _assert_code(mismatched_item, 'dataset_item_not_found')


class _CloseSensitiveTaijiClient(_FakeTaijiClient):
    def __init__(self) -> None:
        super().__init__()
        self.provider_steps = {'iter_item_ids': 0, 'iter_items': 0}

    async def _iter_item_pages(self, ref, *, limit: int | None, page_size: int):
        self.provider_steps['iter_item_ids'] += 1
        yield [{'id': 'remote-a', 'body': {'task_name': 'echo'}}], []
        self.provider_steps['iter_item_ids'] += 1
        raise RuntimeError('closed reader resumed the Taiji ID provider')

    async def iter_resolved_items(self, ref, *, limit: int | None = None, page_size: int = 500):
        self.provider_steps['iter_items'] += 1
        yield DatasetItem.from_mapping(
            {'id': 'remote-a', 'body': {'task_name': 'echo'}},
            remote_ref=ref,
        )
        self.provider_steps['iter_items'] += 1
        raise RuntimeError('closed reader resumed the Taiji item provider')


@pytest.mark.asyncio
async def test_taiji_reader_pins_symbolic_version_once_and_restarts_iterators():
    client = _FakeTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        first_descriptor = await reader.describe()
        second_descriptor = await reader.describe()
        first_ids = [item_id async for item_id in reader.iter_item_ids(limit=2)]
        second_ids = [item_id async for item_id in reader.iter_item_ids(limit=2)]
        items = [item async for item in reader.iter_items(limit=1)]

    assert first_descriptor is not second_descriptor
    first_descriptor.metadata['mutated'] = True
    assert 'mutated' not in second_descriptor.metadata
    assert first_descriptor.provider == 'taiji'
    assert first_descriptor.reference == 'research/tasks@latest'
    assert first_descriptor.resolved_reference == 'research/tasks@release-7?split=test'
    assert first_ids == second_ids == ['remote-a', 'remote-b']
    assert [item.id for item in items] == ['remote-a']
    assert all(ref.version == 'release-7' and ref.split == 'test' for ref in client.refs)
    assert client.resolve_version_calls == 1


@pytest.mark.asyncio
async def test_taiji_reader_describe_returns_isolated_cached_descriptors():
    client = _FakeTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        first = await reader.describe()
        first.metadata['mutated'] = True
        first.splits[0].name = 'mutated'

        second = await reader.describe()
        assert second.splits[0].name == 'test'
        second.metadata['mutated_again'] = True
        second.splits[0].name = 'mutated_again'

        third = await reader.describe()

    assert 'mutated' not in second.metadata
    assert 'mutated_again' not in third.metadata
    assert third.splits[0].name == 'test'


@pytest.mark.asyncio
async def test_taiji_reader_descriptor_metadata_excludes_credentials() -> None:
    class MetadataClient(_FakeTaijiClient):
        async def show(self, _uri):
            return {
                'safe': 'value',
                'token': 'secret-token',
                'nested': {
                    'api_key': 'secret-key',
                    'signed_url': 'https://signed.example/?sig=secret',
                    'name': 'fixture',
                },
                'download': 'https://signed.example/files.zip?signature=secret',
                'basic_auth_url': 'https://alice:secret@example.test/files.zip',
                'query_sig_url': 'https://example.test/files.zip?sig=secret',
                'cookie': 'session-secret',
                'x-tai-identity': 'identity-secret',
            }

    async with open_dataset('research/tasks@latest', dataset_client=MetadataClient()) as reader:
        descriptor = await reader.describe()

    assert descriptor.metadata == {
        'safe': 'value',
        'nested': {'name': 'fixture'},
        'download': '<redacted>',
        'basic_auth_url': '<redacted>',
        'query_sig_url': '<redacted>',
        'namespace': 'research',
        'consistency': 'versioned',
    }


@pytest.mark.asyncio
async def test_taiji_reader_rejects_unknown_explicit_split() -> None:
    client = _FakeTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        with pytest.raises(DatasetReaderError) as missing_split:
            _ = [item async for item in reader.iter_items(split='missing')]

    _assert_code(missing_split, 'dataset_split_not_found')


@pytest.mark.asyncio
async def test_taiji_reader_limit_zero_still_validates_explicit_split() -> None:
    client = _FakeTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        with pytest.raises(DatasetReaderError) as ids_error:
            _ = [item_id async for item_id in reader.iter_item_ids(split='missing', limit=0)]
        _assert_code(ids_error, 'dataset_split_not_found')

        with pytest.raises(DatasetReaderError) as items_error:
            _ = [item async for item in reader.iter_items(split='missing', limit=0)]
        _assert_code(items_error, 'dataset_split_not_found')


@pytest.mark.asyncio
async def test_taiji_reader_maps_split_catalog_not_found_to_reader_error() -> None:
    class MissingSplitCatalogClient(_FakeTaijiClient):
        async def list_splits(self, _uri, *, version=None):
            assert version == 'release-7'
            raise KeyError('split catalog missing')

    async with open_dataset('research/tasks@latest', dataset_client=MissingSplitCatalogClient()) as reader:
        with pytest.raises(DatasetReaderError) as missing_split:
            _ = [item async for item in reader.iter_items(split='test')]

    _assert_code(missing_split, 'dataset_split_not_found')


@pytest.mark.asyncio
async def test_taiji_reader_preserves_unexpected_split_resolution_errors() -> None:
    class FailingSplitClient(_FakeTaijiClient):
        async def list_splits(self, _uri, *, version=None):
            assert version == 'release-7'
            return []

        async def resolve_split(self, _uri, *, version, split=None):
            assert version == 'release-7'
            assert split == 'test'
            raise ValueError('provider split service unavailable')

    async with open_dataset('research/tasks@latest', dataset_client=FailingSplitClient()) as reader:
        with pytest.raises(ValueError, match='provider split service unavailable') as split_error:
            _ = [item async for item in reader.iter_items(split='test')]

    assert not isinstance(split_error.value, DatasetReaderError)


@pytest.mark.asyncio
async def test_taiji_reader_item_lookup_does_not_bound_provider_scan() -> None:
    client = _FakeTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        items = [item async for item in reader.iter_items(item_id='remote-b', limit=1)]

    assert [item.id for item in items] == ['remote-b']


@pytest.mark.asyncio
async def test_taiji_reader_item_lookup_filters_legacy_provider_results() -> None:
    ref = RemoteDatasetRef(namespace='research', dataset='tasks', version='v7', split='test')

    class LegacyClient:
        async def resolve_version(self, _uri):
            return 'v7'

        async def resolve_split(self, _uri, *, version, split=None):
            assert version == 'v7'
            return split or 'test'

        async def list_splits(self, _uri, *, version=None):
            assert version == 'v7'
            return ['test']

        async def iter_items(self, _uri, *, split=None, item_id=None, limit=None, page_size=500):
            del split, item_id, page_size
            values = ('other', 'target')
            # A legacy provider applies ``limit`` to the collection page even
            # when an item_id filter is supplied. The Reader must scan without
            # that bound to find an exact match after the first result.
            for value in values[:limit] if limit is not None else values:
                yield DatasetItem(id=value, body={'task_name': 'echo'}, remote_ref=ref)

    async with open_dataset('research/tasks@latest', dataset_client=LegacyClient()) as reader:
        items = [item async for item in reader.iter_items(item_id='target', limit=1)]

    assert [item.id for item in items] == ['target']


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['ids', 'items'])
async def test_taiji_reader_enforces_limit_when_provider_over_returns(method: str) -> None:
    class OverReturningClient(_FakeTaijiClient):
        async def _iter_item_pages(self, ref, *, limit, page_size):
            del limit, page_size
            yield (
                [
                    {'id': 'remote-a', 'body': {'task_name': 'echo'}},
                    {'id': 'remote-b', 'body': {'task_name': 'echo'}},
                    {'id': 'remote-c', 'body': {'task_name': 'echo'}},
                ],
                [],
            )

        async def iter_resolved_items(self, ref, *, limit=None, page_size=500):
            del limit, page_size
            for item_id in ('remote-a', 'remote-b', 'remote-c'):
                yield DatasetItem.from_mapping(
                    {'id': item_id, 'body': {'task_name': 'echo'}},
                    remote_ref=ref,
                )

    async with open_dataset('research/tasks@latest', dataset_client=OverReturningClient()) as reader:
        if method == 'ids':
            values = [item_id async for item_id in reader.iter_item_ids(limit=2)]
        else:
            values = [item.id async for item in reader.iter_items(limit=2)]

    assert values == ['remote-a', 'remote-b']


@pytest.mark.asyncio
async def test_taiji_reader_falls_back_to_resolve_split_when_catalog_is_empty() -> None:
    class EmptyCatalogClient(_FakeTaijiClient):
        async def list_splits(self, _uri, *, version=None):
            assert version == 'release-7'
            return []

    async with open_dataset('research/tasks@latest', dataset_client=EmptyCatalogClient()) as reader:
        items = [item async for item in reader.iter_items(split='test', limit=1)]

    assert [item.id for item in items] == ['remote-a']


@pytest.mark.asyncio
async def test_taiji_describe_does_not_retry_provider_assertion_with_symbolic_uri():
    client = _FakeTaijiClient()
    calls = 0

    async def list_splits(_uri, *, version=None):
        nonlocal calls
        calls += 1
        raise AssertionError('provider contract violation')

    client.list_splits = list_splits
    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        with pytest.raises(AssertionError, match='provider contract violation'):
            await reader.describe()

    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('iterator_name', ['iter_item_ids', 'iter_items'])
async def test_taiji_reader_checks_closed_state_before_resuming_provider(iterator_name):
    client = _CloseSensitiveTaijiClient()

    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        iterator = getattr(reader, iterator_name)()
        first = await anext(iterator)
        assert first == 'remote-a' or first.id == 'remote-a'

    with pytest.raises(DatasetReaderError) as iterator_closed:
        await anext(iterator)
    _assert_code(iterator_closed, 'dataset_reader_closed')
    assert client.provider_steps[iterator_name] == 1


@pytest.mark.asyncio
async def test_taiji_reader_closes_partially_consumed_provider_iterator() -> None:
    class ClosableClient(_FakeTaijiClient):
        def __init__(self) -> None:
            super().__init__()
            self.provider_closed = False

        async def iter_resolved_items(self, ref, *, limit=None, page_size=500):
            del ref, limit, page_size
            try:
                yield DatasetItem(id='remote-a', body={'task_name': 'echo'})
                await asyncio.sleep(3600)
            finally:
                self.provider_closed = True

    client = ClosableClient()
    async with open_dataset('research/tasks@latest', dataset_client=client) as reader:
        iterator = reader.iter_items()
        assert (await anext(iterator)).id == 'remote-a'

    assert client.provider_closed is True


class _ExtensionReader(BaseDatasetReader):
    def __init__(self, uri: DatasetSourceURI) -> None:
        super().__init__()
        self.uri = uri
        self.closed = False

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        return DatasetDescriptor(
            reference=self.uri.display,
            provider='example',
            resolved_reference=self.uri.display,
            name='extension',
            default_split='test',
            splits=[DatasetSplitDescriptor(name='test', item_count=1)],
        )

    async def iter_item_ids(self, *, split=None, limit=None) -> AsyncIterator[str]:
        self._require_open()
        yield 'extension-a'

    async def iter_items(self, *, split=None, item_id=None, limit=None) -> AsyncIterator[DatasetItem]:
        self._require_open()
        yield DatasetItem(id='extension-a', body={'task_name': 'echo'})

    async def _close(self, exc_type, exc_val, exc_tb) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_resolver_dispatches_registered_source_factory():
    captured: dict[str, Any] = {}

    @dataset_source('example')
    def factory(uri: DatasetSourceURI):
        captured['uri'] = uri
        return _ExtensionReader(uri)

    try:
        async with open_dataset('ExAmPlE://bucket/path') as reader:
            extension = reader
            assert [item.id async for item in reader.iter_items()] == ['extension-a']
        assert extension.closed is True
    finally:
        dataset_source_registry._registry.pop('example', None)


@pytest.mark.asyncio
async def test_resolver_rejects_unknown_scheme_and_missing_explicit_local_path(tmp_path):
    with pytest.raises(DatasetReaderError) as unknown_scheme:
        async with open_dataset('missing-reader://bucket/path'):
            pass
    _assert_code(unknown_scheme, 'dataset_reader_scheme_not_found')

    with pytest.raises(DatasetReaderError) as missing_local:
        async with open_dataset(tmp_path / 'missing'):
            pass
    _assert_code(missing_local, 'local_dataset_not_found')


@pytest.mark.asyncio
async def test_resolver_rejects_invalid_page_size_before_opening_owned_client(monkeypatch):
    opened = False

    class UnexpectedClient:
        def __init__(self, *, authenticator=None) -> None:
            nonlocal opened
            opened = True

        async def __aenter__(self):
            return self

    monkeypatch.setattr('nexus.dataset.client.DatasetClient', UnexpectedClient)

    with pytest.raises(DatasetReaderError) as invalid_page_size:
        async with open_dataset('research/tasks@latest', options={'page_size': 0}):
            pass

    _assert_code(invalid_page_size, 'dataset_page_size_invalid')
    assert opened is False


def test_module_metadata_registers_lazy_dataset_source_factory(tmp_path):
    module = ModuleMeta(
        name='reader-extension',
        module_path='example.reader',
        fs_path=tmp_path,
        objects={'dataset_sources': {'example-meta': 'example.reader::factory'}},
    )

    try:
        register_objects_from_meta(module)
        entry = dataset_source_registry.get_entry('example-meta')
        assert entry.import_path == 'example.reader::factory'
        assert entry.is_loaded is False
    finally:
        dataset_source_registry._registry.pop('example-meta', None)
