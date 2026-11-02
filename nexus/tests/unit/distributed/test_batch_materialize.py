from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile

from pathlib import Path

import pytest

from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.reader import BaseDatasetReader, DatasetDescriptor, DatasetSplitDescriptor
from nexus.dataset.source import DatasetSourceURI, dataset_source, dataset_source_registry
from nexus.dataset.zip_index import write_nexus_zip_directory_to_file
from nexus.distributed.batch.data_list import DataListError, DataRef, read_indexed_frame, scan_data_list, validate_frame
from nexus.distributed.batch.materialize import (
    MaterializedBatchArtifacts,
    materialize_async_items,
    materialize_compose,
    materialize_generator,
    materialize_items,
)
from nexus.runner.batch.batch_config import BatchItem
from nexus.runner.compose.model import RawRunPart
from nexus.runner.single import RunInput


def _item(
    *,
    item_id: str | None = None,
    message: str = 'hello',
    metadata: dict | None = None,
) -> BatchItem:
    return BatchItem(
        config=RunInput(task_name='echo', task_config={'message': message}),
        run_id=item_id,
        metadata=dict(metadata or {}),
    )


def _canonical_line(item: BatchItem, item_id: str) -> bytes:
    run_input = item.config.model_dump(mode='json', exclude={'run_id'})
    encoded_item_id = json.dumps(item_id, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    encoded_run_input = json.dumps(
        run_input,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return f'{{"version":1,"item_id":{encoded_item_id},"run_input":{encoded_run_input}}}\n'.encode()


def test_materialize_items_writes_canonical_versioned_utf8_records(tmp_path):
    stable = _item(item_id='case-1', message='你好')
    generated = _item(message='second')
    output = tmp_path / 'tasks.jsonl'

    result = materialize_items([stable, generated], output)

    expected = _canonical_line(stable, 'case-1') + _canonical_line(generated, 'item-000000000002')
    assert output.read_bytes() == expected
    assert '你好'.encode() in expected
    assert b'\\u4e00' not in expected
    assert result.path == output
    assert result.total_items == 2
    assert result.size_bytes == len(expected)
    assert result.sha256 == hashlib.sha256(expected).hexdigest()

    records = [json.loads(line) for line in expected.splitlines()]
    assert [record['item_id'] for record in records] == ['case-1', 'item-000000000002']
    assert all(set(record) == {'version', 'item_id', 'run_input'} for record in records)
    assert all(record['version'] == 1 for record in records)
    assert all('run_id' not in record['run_input'] for record in records)
    assert all('metadata' not in record for record in records)


def test_materialize_items_rejects_nonempty_item_metadata(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='task-list v1 does not support item metadata'):
        materialize_items(
            [_item(metadata={'conflict_groups': ['shared-resource']})],
            output,
        )

    assert not output.exists()


def test_materialize_items_v2_preserves_bounded_item_metadata(tmp_path):
    output = tmp_path / 'tasks.jsonl'
    metadata = {'category': 'terminal', 'tags': ['dataset', 'smoke']}

    result = materialize_items(
        [_item(item_id='case-1', metadata=metadata)],
        output,
        task_list_version=2,
    )

    assert result.record_version == 2
    assert json.loads(output.read_text()) == {
        'version': 2,
        'item_id': 'case-1',
        'run_input': _item(item_id='case-1').config.model_dump(mode='json', exclude={'run_id'}),
        'metadata': metadata,
    }


@pytest.mark.parametrize(
    'source_item_id',
    [
        'Mini SWE Agent/Case A',
        '_leading',
        'café',
        'x' * 65,
    ],
)
def test_materialize_items_v2_normalizes_invalid_ids_and_retains_the_source_in_metadata(
    tmp_path,
    source_item_id,
):
    output = tmp_path / 'tasks.jsonl'

    materialize_items(
        [_item(item_id=source_item_id, metadata={'instance_id': 'source-case'})],
        output,
        task_list_version=2,
    )

    record = json.loads(output.read_text())
    assert record['item_id'] != source_item_id
    assert record['metadata'] == {
        'instance_id': 'source-case',
        '_nexus': {'original_run_id': source_item_id},
    }


@pytest.mark.parametrize('source_item_id', ['', '   '])
def test_materialize_items_v2_uses_the_ordinal_for_empty_ids_and_retains_the_source(tmp_path, source_item_id):
    output = tmp_path / 'tasks.jsonl'

    materialize_items(
        [_item(item_id=source_item_id)],
        output,
        task_list_version=2,
    )

    record = json.loads(output.read_text())
    assert record['item_id'] == 'item-000000000001'
    assert record['metadata'] == {'_nexus': {'original_run_id': source_item_id}}


def test_materialize_items_v2_merges_normalization_metadata_without_mutating_the_item(tmp_path):
    output = tmp_path / 'tasks.jsonl'
    item = _item(item_id='Case A', metadata={'_nexus': {'source': 'generator'}})

    materialize_items([item], output, task_list_version=2)

    record = json.loads(output.read_text())
    assert record['metadata']['_nexus'] == {
        'source': 'generator',
        'original_run_id': 'Case A',
    }
    assert item.metadata == {'_nexus': {'source': 'generator'}}


def test_materialize_items_v2_rejects_non_object_nexus_metadata_when_normalization_is_needed(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='metadata _nexus must be an object'):
        materialize_items(
            [_item(item_id='Case A', metadata={'_nexus': 'reserved'})],
            output,
            task_list_version=2,
        )

    assert not output.exists()


def test_materialize_items_v2_rejects_oversized_item_metadata(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='metadata exceeds size limit'):
        materialize_items(
            [_item(metadata={'value': 'x' * (256 * 1024)})],
            output,
            task_list_version=2,
        )

    assert not output.exists()


def test_materialize_generator_preserves_bounded_batch_metadata(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"))]\n'
        'def get_batch_metadata():\n'
        '    return {"group": "preserved", "tags": ["hyway"]}\n'
        'def get_total_count():\n'
        '    raise AssertionError("Cloud materialization must ignore get_total_count")\n'
    )
    output = tmp_path / 'tasks.jsonl'

    result = materialize_generator(generator, output)

    assert result.batch_metadata == {'group': 'preserved', 'tags': ['hyway']}
    assert 'batch_metadata' not in json.loads(output.read_text())

    output_v2 = tmp_path / 'tasks-v2.jsonl'
    result_v2 = materialize_generator(generator, output_v2, task_list_version=2)

    assert result_v2.batch_metadata == result.batch_metadata
    assert 'batch_metadata' not in json.loads(output_v2.read_text())


def test_materialize_generator_rejects_oversized_batch_metadata(tmp_path):
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        '    return [BatchItem(config=RunInput(task_name="echo"))]\n'
        'def get_batch_metadata():\n'
        '    return {"value": "x" * (64 * 1024)}\n'
    )

    with pytest.raises(ValueError, match='batch metadata exceeds size limit'):
        materialize_generator(generator, tmp_path / 'tasks.jsonl', task_list_version=2)


def test_materialize_items_is_byte_deterministic(tmp_path):
    items = [_item(item_id='stable', message='你好'), _item(message='again')]

    first = materialize_items(items, tmp_path / 'first.jsonl')
    second = materialize_items(items, tmp_path / 'second.jsonl')

    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes
    assert first.total_items == second.total_items
    assert first.path.read_bytes() == second.path.read_bytes()


@pytest.mark.parametrize('item_id', ['', '   '])
def test_materialize_items_rejects_empty_source_item_id(tmp_path, item_id):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='empty batch item_id'):
        materialize_items([_item(item_id=item_id)], output)

    assert not output.exists()
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


def test_materialize_items_rejects_duplicate_source_item_ids(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='duplicate batch item_id: duplicate'):
        materialize_items([_item(item_id='duplicate'), _item(item_id='duplicate')], output)

    assert not output.exists()
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


@pytest.mark.parametrize('item_id', ['x' * 65, 'line\nbreak', 'Item', 'item ', 'café', 'cafe\u0301'])
def test_materialize_items_rejects_ids_the_distributed_ledger_cannot_read(tmp_path, item_id):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='invalid batch item_id'):
        materialize_items([_item(item_id=item_id)], output)

    assert not output.exists()
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


@pytest.mark.parametrize(
    ('limits', 'message'),
    [
        ({'max_items': 1}, 'batch task list exceeds item limit'),
        ({'max_line_bytes': 1}, 'batch task list item exceeds line size limit'),
        ({'max_bytes': 1}, 'batch task list exceeds byte limit'),
    ],
)
def test_materialize_items_enforces_generation_limits_without_output(tmp_path, limits, message):
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match=message):
        materialize_items([_item(), _item()], output, **limits)

    assert not output.exists()
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


def test_materialize_items_accepts_exact_line_and_total_byte_limits(tmp_path):
    item = _item(item_id='stable')
    line = _canonical_line(item, 'stable')
    output = tmp_path / 'tasks.jsonl'

    result = materialize_items(
        [item],
        output,
        max_items=1,
        max_line_bytes=len(line),
        max_bytes=len(line),
    )

    assert output.read_bytes() == line
    assert result.size_bytes == len(line)


def test_materialize_items_accepts_unlimited_item_count(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    result = materialize_items(
        [_item(item_id='first'), _item(item_id='second')],
        output,
        max_items=None,
    )

    assert result.total_items == 2


def test_materialize_items_accepts_unlimited_line_size(tmp_path):
    output = tmp_path / 'tasks.jsonl'

    result = materialize_items(
        [_item(item_id='large', message='x' * 1024)],
        output,
        max_line_bytes=None,
    )

    assert result.total_items == 1


def test_materialize_items_preserves_existing_output_when_generation_fails(tmp_path):
    output = tmp_path / 'tasks.jsonl'
    output.write_bytes(b'previous\n')

    def broken_items():
        yield _item()
        raise RuntimeError('generator failed')

    with pytest.raises(RuntimeError, match='generator failed'):
        materialize_items(broken_items(), output)

    assert output.read_bytes() == b'previous\n'
    assert not list(tmp_path.glob(f'.{output.name}.*.tmp'))


def test_materialize_async_items_streams_canonical_records(tmp_path):
    async def items():
        yield _item(item_id='first')
        yield _item(item_id='second')

    output = tmp_path / 'tasks.jsonl'
    result = asyncio.run(materialize_async_items(items(), output, max_items=2))

    assert result.total_items == 2
    assert output.read_bytes() == _canonical_line(_item(item_id='first'), 'first') + _canonical_line(
        _item(item_id='second'), 'second'
    )


def test_materialize_compose_pages_remote_dataset_and_records_resolved_version(tmp_path, monkeypatch):
    class FakeDatasetClient:
        def __init__(self, *, authenticator=None):
            assert authenticator is None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def resolve_version(self, uri):
            assert uri.display == 'research/cases@latest'
            return 'v42'

        async def list_splits(self, uri, *, version=None):
            assert uri.display == 'research/cases@v42'
            assert version == 'v42'
            return ['test']

        async def resolve_split(self, uri, *, version, split=None):
            assert uri.display == 'research/cases@latest'
            assert version == 'v42'
            assert split is None
            return 'test'

        async def iter_resolved_items(self, ref, *, limit=None, page_size=500):
            assert ref.version == 'v42'
            assert limit == 2
            assert page_size == 500
            for item_id in ('case-001', 'case-002'):
                yield DatasetItem(
                    id=item_id,
                    body={'task_name': 'echo', 'task_config': {'message': item_id}},
                    remote_ref=ref,
                )

    import nexus.dataset.client as dataset_client_module

    monkeypatch.setattr(dataset_client_module, 'DatasetClient', FakeDatasetClient)
    parts = [
        RawRunPart(kind='data', value='research/cases@latest', source='--data research/cases@latest'),
        RawRunPart(kind='set', value='runtime_provider_name=local', source='--set runtime_provider_name=local'),
    ]

    result = asyncio.run(
        materialize_compose(
            parts,
            tmp_path / 'tasks.jsonl',
            split='test',
            limit=2,
            max_items=10,
            max_bytes=1024 * 1024,
        )
    )

    records = [json.loads(line) for line in result.path.read_text().splitlines()]
    assert [record['item_id'] for record in records] == ['case-001', 'case-002']
    assert all(record['run_input']['runtime_provider_name'] == 'local' for record in records)
    assert result.resolved_datasets == (
        {
            'input_uri': 'research/cases@latest',
            'resolved_uri': 'research/cases@v42?split=test',
        },
    )


@pytest.mark.parametrize(('selected_id', 'expected_ids'), [(None, ['case-a', 'case-b']), ('case-b', ['case-b'])])
def test_materialize_compose_uses_registered_dataset_reader(tmp_path, selected_id, expected_ids):
    closed: list[bool] = []

    class ExtensionReader(BaseDatasetReader):
        def __init__(self, uri: DatasetSourceURI):
            super().__init__()
            self.uri = uri

        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference=self.uri.display,
                provider='materialize-test',
                resolved_reference='materialize-test://fixture@resolved',
                name='fixture',
                default_split='test',
                splits=[DatasetSplitDescriptor(name='test', item_count=2)],
            )

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            for item_id in ('case-a', 'case-b'):
                yield item_id

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            assert split == 'test'
            assert item_id == selected_id
            assert limit == 2
            for current_id in ('case-a', 'case-b'):
                if item_id is not None and current_id != item_id:
                    continue
                yield DatasetItem(
                    id=current_id,
                    body={'task_name': 'echo', 'task_config': {'source': current_id}},
                    metadata={'provider': 'extension'},
                )

        async def _close(self, exc_type, exc_val, exc_tb):
            closed.append(True)

    @dataset_source('materialize-test')
    def factory(uri: DatasetSourceURI):
        assert uri.display == 'materialize-test://fixture'
        return ExtensionReader(uri)

    output = tmp_path / 'tasks.jsonl'
    parts = [
        RawRunPart(kind='data', value='materialize-test://fixture', source='--data materialize-test://fixture'),
        RawRunPart(kind='set', value='task_config.override=true', source='--set task_config.override=true'),
    ]
    try:
        result = asyncio.run(
            materialize_compose(
                parts,
                output,
                split='test',
                item_id=selected_id,
                limit=2,
                page_size=23,
                task_list_version=2,
            )
        )
    finally:
        dataset_source_registry._registry.pop('materialize-test', None)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record['item_id'] for record in records] == expected_ids
    assert all(record['run_input']['task_config']['override'] is True for record in records)
    assert all(record['metadata']['provider'] == 'extension' for record in records)
    assert result.resolved_datasets == ()
    assert closed == [True]


def test_materialize_compose_replaces_descriptor_split_for_taiji_resolution(tmp_path, monkeypatch):
    class FakeReader(BaseDatasetReader):
        async def describe(self):
            self._require_open()
            return DatasetDescriptor(
                reference='research/cases@latest',
                resolved_reference='research/cases@v42?split=default',
                provider='taiji',
                name='cases',
                default_split='default',
                splits=[DatasetSplitDescriptor(name='default'), DatasetSplitDescriptor(name='selected')],
            )

        async def iter_items(self, *, split=None, item_id=None, limit=None):
            self._require_open()
            assert split == 'selected'
            assert item_id is None
            assert limit == 1
            yield DatasetItem(id='case-001', body={'task_name': 'echo'}, remote_ref=None)

        async def iter_item_ids(self, *, split=None, limit=None):
            self._require_open()
            yield 'case-001'

    monkeypatch.setattr(
        'nexus.distributed.batch.materialize.open_dataset',
        lambda *args, **kwargs: FakeReader(),
    )
    parts = [
        RawRunPart(kind='data', value='research/cases@latest', source='--data research/cases@latest'),
    ]

    result = asyncio.run(
        materialize_compose(
            parts,
            tmp_path / 'tasks.jsonl',
            split='selected',
            limit=1,
            max_items=2,
            max_bytes=1024 * 1024,
        )
    )

    assert result.resolved_datasets == (
        {
            'input_uri': 'research/cases@latest',
            'resolved_uri': 'research/cases@v42?split=selected',
        },
    )


def test_materialize_compose_rejects_remote_dataset_item_metadata(tmp_path, monkeypatch):
    class FakeDatasetClient:
        def __init__(self, *, authenticator=None):
            assert authenticator is None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def resolve_version(self, uri):
            del uri
            return 'v42'

        async def list_splits(self, uri, *, version=None):
            del uri, version
            return ['test']

        async def resolve_split(self, uri, *, version, split=None):
            del uri, version, split
            return 'test'

        async def iter_resolved_items(self, ref, *, limit=None, page_size=500):
            del limit, page_size
            yield DatasetItem(
                id='case-001',
                body={'task_name': 'echo'},
                metadata={'group': 'unsupported'},
                remote_ref=ref,
            )

    import nexus.dataset.client as dataset_client_module

    monkeypatch.setattr(dataset_client_module, 'DatasetClient', FakeDatasetClient)
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='task-list v1 does not support item metadata'):
        asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='research/cases@latest', source='--data research/cases@latest')],
                output,
            )
        )

    assert not output.exists()


def test_materialize_compose_v2_preserves_remote_dataset_item_metadata(tmp_path, monkeypatch):
    metadata = {'category': 'terminal', 'difficulty': 'medium'}

    class FakeDatasetClient:
        def __init__(self, *, authenticator=None):
            assert authenticator is None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def resolve_version(self, uri):
            del uri
            return 'v42'

        async def list_splits(self, uri, *, version=None):
            del uri, version
            return ['test']

        async def resolve_split(self, uri, *, version, split=None):
            del uri, version, split
            return 'test'

        async def iter_resolved_items(self, ref, *, limit=None, page_size=500):
            del limit, page_size
            yield DatasetItem(
                id='case-001',
                body={'task_name': 'echo'},
                metadata=metadata,
                remote_ref=ref,
            )

    import nexus.dataset.client as dataset_client_module

    monkeypatch.setattr(dataset_client_module, 'DatasetClient', FakeDatasetClient)
    output = tmp_path / 'tasks.jsonl'

    result = asyncio.run(
        materialize_compose(
            [RawRunPart(kind='data', value='research/cases@latest', source='--data research/cases@latest')],
            output,
            task_list_version=2,
        )
    )

    assert result.record_version == 2
    assert json.loads(output.read_text())['metadata'] == {
        **metadata,
        'dataset_item_id': 'case-001',
    }


def test_materialize_compose_rejects_local_dataset_item_metadata(tmp_path):
    dataset_root = tmp_path / 'local-dataset'
    item_path = dataset_root / 'items' / 'test' / 'case-local' / 'item.json'
    item_path.parent.mkdir(parents=True)
    item_path.write_text(
        json.dumps({
            'id': 'case-local',
            'body': {'task_name': 'echo'},
            'metadata': {'group': 'unsupported'},
        })
    )

    (dataset_root / 'dataset.yaml').write_text(
        json.dumps({
            'schema_version': 1,
            'name': 'local-dataset',
            'default_split': 'test',
            'splits': {'test': {'items': ['items/test/case-local/item.json']}},
        }),
        encoding='utf-8',
    )
    output = tmp_path / 'tasks.jsonl'

    with pytest.raises(ValueError, match='task-list v1 does not support item metadata'):
        asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value=str(dataset_root), source=f'--data {dataset_root}')],
                output,
            )
        )

    assert not output.exists()


class _MaterializeExtensionReader(BaseDatasetReader):
    def __init__(self, uri: DatasetSourceURI, items: list[DatasetItem], calls: list[str]) -> None:
        super().__init__()
        self.uri = uri
        self.items = items
        self.calls = calls

    async def describe(self):
        self._require_open()
        return DatasetDescriptor(
            reference=self.uri.display,
            provider='materialize-auto',
            resolved_reference=self.uri.display,
            name='fixture',
            default_split='test',
            splits=[DatasetSplitDescriptor(name='test', item_count=len(self.items))],
        )

    async def iter_item_ids(self, *, split=None, limit=None):
        self._require_open()
        for item in self.items[:limit]:
            yield item.id

    async def iter_items(self, *, split=None, item_id=None, limit=None):
        self._require_open()
        self.calls.append('iter')
        assert split in {None, 'test'}
        assert item_id is None
        for item in self.items[:limit]:
            yield item

    async def _close(self, exc_type, exc_val, exc_tb):
        self.calls.append('close')


def _register_materialize_reader(scheme: str, items: list[DatasetItem], calls: list[str]) -> None:
    @dataset_source(scheme)
    def factory(uri: DatasetSourceURI):
        return _MaterializeExtensionReader(uri, items, calls)


def _local_zip(tmp_path, *, content: str = 'zip payload'):
    root = tmp_path / 'zip-source'
    root.mkdir(exist_ok=True)
    (root / 'payload.txt').write_text(content, encoding='utf-8')
    path = tmp_path / 'files.zip'
    write_nexus_zip_directory_to_file(root, path)
    return path


def test_materialize_compose_auto_keeps_no_file_batch_at_v2(tmp_path):
    calls: list[str] = []
    items = [DatasetItem(id='case-a', body={'task_name': 'echo'}, metadata={'kind': 'plain'})]
    _register_materialize_reader('auto-no-files', items, calls)
    try:
        result = asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='auto-no-files://fixture', source='test')],
                tmp_path / 'tasks.jsonl',
                task_list_version=2,
                auto_data_list=True,
                data_list_output_path=tmp_path / 'data_list.bin',
            )
        )
    finally:
        dataset_source_registry._registry.pop('auto-no-files', None)

    assert isinstance(result, MaterializedBatchArtifacts)
    assert result.task_list.record_version == 2
    assert result.data_list is None
    record = json.loads(result.task_list.path.read_text())
    assert record['version'] == 2
    assert record['item_id'] == 'case-a'
    assert record['run_input']['task_name'] == 'echo'
    assert record['metadata'] == {'dataset_item_id': 'case-a', 'kind': 'plain'}
    assert calls == ['iter', 'close']
    assert not (tmp_path / 'data_list.bin').exists()


def test_materialize_compose_auto_rejects_identical_task_and_data_output_paths(tmp_path):
    calls: list[str] = []
    items = [
        DatasetItem(id='case-a', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(_local_zip(tmp_path))})
    ]
    _register_materialize_reader('auto-same-output', items, calls)
    output = tmp_path / 'same.bin'
    try:
        with pytest.raises(ValueError, match='task_list_and_data_list_outputs_must_differ'):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-same-output://fixture', source='test')],
                    output,
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=output,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-same-output', None)

    assert not output.exists()


def test_materialize_compose_auto_rolls_back_existing_outputs_on_publish_failure(tmp_path, monkeypatch):
    import nexus.distributed.batch.materialize as materialize_module

    zip_path = _local_zip(tmp_path)
    item = DatasetItem(id='case-a', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(zip_path)})
    _register_materialize_reader('auto-publish-rollback', [item], [])
    output = tmp_path / 'tasks.jsonl'
    data_output = tmp_path / 'data_list.bin'
    output.write_bytes(b'old task list')
    data_output.write_bytes(b'old data list')
    original_replace = materialize_module.os.replace

    def fail_task_publish(source, destination):
        if Path(destination) == output and Path(source).name == 'tasks.jsonl':
            raise OSError('injected task-list publish failure')
        return original_replace(source, destination)

    monkeypatch.setattr(materialize_module.os, 'replace', fail_task_publish)
    try:
        with pytest.raises(OSError, match='injected task-list publish failure'):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-publish-rollback://fixture', source='test')],
                    output,
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=data_output,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-publish-rollback', None)

    assert output.read_bytes() == b'old task list'
    assert data_output.read_bytes() == b'old data list'
    assert not list(tmp_path.glob('.*.publish-backup-*'))


def test_materialize_compose_auto_writes_mixed_v3_without_local_paths(tmp_path):
    directory = tmp_path / 'directory-files'
    directory.mkdir()
    (directory / 'payload.txt').write_text('directory payload', encoding='utf-8')
    zip_path = _local_zip(tmp_path)
    remote_ref = RemoteDatasetRef(namespace='research', dataset='remote', version='v1', split='test')
    items = [
        DatasetItem(
            id='directory-case',
            body={'task_name': 'echo'},
            files={'type': 'directory', 'path': str(directory)},
        ),
        DatasetItem(id='zip-case', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(zip_path)}),
        DatasetItem(
            id='remote-case',
            body={'task_name': 'echo'},
            files={'type': 'zip', 'path': 'blobs/remote.zip', 'size': 123},
            remote_ref=remote_ref,
        ),
        DatasetItem(id='plain-case', body={'task_name': 'echo'}),
    ]
    calls: list[str] = []
    _register_materialize_reader('auto-mixed', items, calls)
    try:
        result = asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='auto-mixed://fixture', source='test')],
                tmp_path / 'tasks.jsonl',
                limit=4,
                task_list_version=2,
                auto_data_list=True,
                data_list_output_path=tmp_path / 'data_list.bin',
            )
        )
    finally:
        dataset_source_registry._registry.pop('auto-mixed', None)

    assert isinstance(result, MaterializedBatchArtifacts)
    assert result.task_list.record_version == 3
    assert result.data_list is not None
    assert calls == ['iter', 'close']
    task_list_bytes = result.task_list.path.read_bytes()
    assert str(tmp_path).encode() not in task_list_bytes
    records = [json.loads(line) for line in task_list_bytes.splitlines()]
    assert [record['version'] for record in records] == [3, 3, 3, 3]
    assert [record['item_id'] for record in records] == [
        'directory-case',
        'zip-case',
        'remote-case',
        'plain-case',
    ]
    assert records[0]['data_ref'] is not None
    assert 'task_files' not in records[0]['run_input']
    assert records[1]['data_ref'] is not None
    assert 'task_files' not in records[1]['run_input']
    assert records[2]['data_ref'] is None
    assert records[2]['run_input']['task_files']['remote'] == remote_ref.model_dump(mode='json')
    assert records[3]['data_ref'] is None
    assert records[3]['run_input']['task_files'] is None

    index = scan_data_list(
        result.data_list.path,
        max_frame_bytes=1024 * 1024,
        max_bytes=1024 * 1024,
        max_items=4,
    )
    assert [frame.item_id for frame in index.frames] == ['directory-case', 'zip-case']
    for record, expected_content in zip(records, ('directory payload', 'zip payload'), strict=False):
        ref = DataRef.model_validate(record['data_ref'])
        frame = validate_frame(
            read_indexed_frame(
                result.data_list.path,
                ref,
                expected_frame=index.by_offset[ref.offset],
                expected_item_id=record['item_id'],
                max_frame_bytes=1024 * 1024,
            ).encoded,
            expected_ref=ref,
            expected_item_id=record['item_id'],
            max_frame_bytes=1024 * 1024,
        )
        assert frame.zip_bytes is not None
        with zipfile.ZipFile(io.BytesIO(frame.zip_bytes)) as archive:
            assert archive.read('payload.txt').decode() == expected_content


@pytest.mark.parametrize(
    ('files_type', 'prepare', 'error_code'),
    [
        ('zip', 'missing', 'local_task_files_not_found'),
        ('directory', 'file', 'local_task_files_type_mismatch'),
        ('zip', 'directory', 'local_task_files_type_mismatch'),
        ('zip', 'bad-sha', 'local_task_files_identity_mismatch'),
        ('zip', 'bad-size', 'local_task_files_identity_mismatch'),
        ('zip', 'invalid-zip', 'local_task_files_invalid_zip'),
    ],
)
def test_materialize_compose_auto_rejects_invalid_local_task_files(
    tmp_path,
    files_type,
    prepare,
    error_code,
):
    path = tmp_path / 'input'
    declared_sha256 = None
    if prepare == 'file':
        path.write_bytes(b'plain file')
    elif prepare == 'directory':
        path.mkdir()
    elif prepare == 'bad-sha':
        path = _local_zip(tmp_path)
        declared_sha256 = '0' * 64
    elif prepare == 'bad-size':
        path = _local_zip(tmp_path)
    elif prepare == 'invalid-zip':
        path.write_bytes(b'not a ZIP')
    item = DatasetItem(
        id='case-a',
        body={'task_name': 'echo'},
        files={
            'type': files_type,
            'path': str(path),
            'sha256': declared_sha256,
            'size': path.stat().st_size + 1 if prepare == 'bad-size' else None,
        },
    )
    calls: list[str] = []
    _register_materialize_reader('auto-invalid', [item], calls)
    output = tmp_path / 'tasks.jsonl'
    data_output = tmp_path / 'data_list.bin'
    try:
        with pytest.raises(ValueError, match=error_code):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-invalid://fixture', source='test')],
                    output,
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=data_output,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-invalid', None)

    assert not output.exists()
    assert not data_output.exists()
    assert not list(tmp_path.glob('.nexus-materialize-*'))


def test_materialize_compose_auto_enforces_frame_list_and_record_limits(tmp_path):
    zip_path = _local_zip(tmp_path)
    item = DatasetItem(id='case-a', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(zip_path)})
    calls: list[str] = []
    _register_materialize_reader('auto-limits', [item], calls)
    try:
        baseline = asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='auto-limits://fixture', source='test')],
                tmp_path / 'baseline.jsonl',
                task_list_version=2,
                auto_data_list=True,
                data_list_output_path=tmp_path / 'baseline.bin',
            )
        )
        assert baseline.data_list is not None
        record = json.loads(baseline.task_list.path.read_text())
        frame_length = record['data_ref']['length']
        record_size = 24 + len(baseline.task_list.path.read_bytes().rstrip(b'\n')) + frame_length

        with pytest.raises(DataListError) as frame_error:
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-limits://fixture', source='test')],
                    tmp_path / 'frame.jsonl',
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=tmp_path / 'frame.bin',
                    max_data_frame_bytes=frame_length - 1,
                )
            )
        assert frame_error.value.code == 'batch_data_frame_exceeds_byte_limit'

        with pytest.raises(DataListError) as list_error:
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-limits://fixture', source='test')],
                    tmp_path / 'list.jsonl',
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=tmp_path / 'list.bin',
                    max_data_list_bytes=baseline.data_list.size_bytes - 1,
                )
            )
        assert list_error.value.code == 'batch_data_list_exceeds_byte_limit'

        with pytest.raises(ValueError, match='batch_task_record_exceeds_byte_limit'):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-limits://fixture', source='test')],
                    tmp_path / 'record.jsonl',
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=tmp_path / 'record.bin',
                    max_task_record_bytes=record_size - 1,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-limits', None)


def test_materialize_compose_uses_nxbr_canonical_bytes_for_non_ascii_record_limit(tmp_path):
    zip_path = _local_zip(tmp_path)
    item = DatasetItem(
        id='case-unicode',
        body={'task_name': 'echo', 'task_config': {'message': '你好'}},
        files={'type': 'zip', 'path': str(zip_path)},
        metadata={'label': '中文'},
    )
    calls: list[str] = []
    _register_materialize_reader('auto-unicode-limit', [item], calls)
    try:
        baseline = asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='auto-unicode-limit://fixture', source='test')],
                tmp_path / 'unicode-baseline.jsonl',
                task_list_version=2,
                auto_data_list=True,
                data_list_output_path=tmp_path / 'unicode-baseline.bin',
            )
        )
        record = json.loads(baseline.task_list.path.read_text(encoding='utf-8'))
        identity_bytes = json.dumps(
            record,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
        task_list_identity_bytes = baseline.task_list.path.read_bytes().rstrip(b'\n')
        assert len(identity_bytes) > len(task_list_identity_bytes)
        exact_record_size = 24 + len(identity_bytes) + record['data_ref']['length']

        exact = asyncio.run(
            materialize_compose(
                [RawRunPart(kind='data', value='auto-unicode-limit://fixture', source='test')],
                tmp_path / 'unicode-exact.jsonl',
                task_list_version=2,
                auto_data_list=True,
                data_list_output_path=tmp_path / 'unicode-exact.bin',
                max_task_record_bytes=exact_record_size,
            )
        )
        assert exact.task_list.record_version == 3

        with pytest.raises(ValueError, match='batch_task_record_exceeds_byte_limit'):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-unicode-limit://fixture', source='test')],
                    tmp_path / 'unicode-too-large.jsonl',
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=tmp_path / 'unicode-too-large.bin',
                    max_task_record_bytes=exact_record_size - 1,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-unicode-limit', None)


def test_materialize_generator_rejects_local_task_files(tmp_path):
    files = tmp_path / 'files'
    files.mkdir()
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch.batch_config import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        'def generate_configs():\n'
        f'    yield BatchItem(config=RunInput(task_name="echo", task_files={{"type":"directory","path":{str(files)!r}}}))\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='local_generator_task_files_unsupported'):
        materialize_generator(generator, tmp_path / 'tasks.jsonl', task_list_version=2)


def test_materialize_generator_rejects_legacy_harbor_paths(tmp_path):
    task_dir = tmp_path / 'harbor-task'
    generator = tmp_path / 'generator.py'
    generator.write_text(
        'from nexus.runner.batch.batch_config import BatchItem\n'
        'from nexus.runner.single import RunInput\n'
        f'path = {str(task_dir)!r}\n'
        'def generate_configs():\n'
        '    yield BatchItem(\n'
        '        config=RunInput(task_name="harbor_bridge_task", task_config={"task_dir": path}),\n'
        '        metadata={"source_path": path},\n'
        '    )\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='local_generator_harbor_task_files_unsupported'):
        materialize_generator(generator, tmp_path / 'tasks.jsonl', task_list_version=2)

    assert not (tmp_path / 'tasks.jsonl').exists()


@pytest.mark.parametrize('auto_data_list', [False, True])
def test_materialize_compose_rejects_legacy_harbor_paths(tmp_path, auto_data_list):
    task_dir = tmp_path / 'harbor-task'
    item = DatasetItem(
        id='case-a',
        body={
            'task_name': 'harbor_bridge_task',
            'task_config': {'task_dir': str(task_dir)},
        },
        metadata={'source_path': str(task_dir)},
    )
    calls: list[str] = []
    _register_materialize_reader('compose-harbor-legacy', [item], calls)
    output = tmp_path / ('tasks.bin' if auto_data_list else 'tasks.jsonl')
    kwargs = {
        'task_list_version': 2,
        'reject_local_task_files': not auto_data_list,
    }
    if auto_data_list:
        kwargs.update({
            'auto_data_list': True,
            'data_list_output_path': tmp_path / 'data_list.bin',
        })
    try:
        with pytest.raises(ValueError, match='local_compose_harbor_task_files_unsupported'):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='compose-harbor-legacy://fixture', source='test')],
                    output,
                    **kwargs,
                )
            )
    finally:
        dataset_source_registry._registry.pop('compose-harbor-legacy', None)

    assert not output.exists()
    assert not (tmp_path / 'data_list.bin').exists()


def test_materialize_compose_auto_is_deterministic_for_unchanged_zip(tmp_path):
    zip_path = _local_zip(tmp_path)
    item = DatasetItem(id='case-a', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(zip_path)})
    calls: list[str] = []
    _register_materialize_reader('auto-deterministic', [item], calls)
    try:
        results = [
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-deterministic://fixture', source='test')],
                    tmp_path / f'tasks-{ordinal}.jsonl',
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=tmp_path / f'data-{ordinal}.bin',
                )
            )
            for ordinal in range(2)
        ]
    finally:
        dataset_source_registry._registry.pop('auto-deterministic', None)

    assert results[0].task_list.sha256 == results[1].task_list.sha256
    assert results[0].task_list.path.read_bytes() == results[1].task_list.path.read_bytes()
    assert results[0].data_list is not None
    assert results[1].data_list is not None
    assert results[0].data_list.sha256 == results[1].data_list.sha256
    assert results[0].data_list.path.read_bytes() == results[1].data_list.path.read_bytes()


def test_materialize_compose_auto_cancellation_cleans_every_temporary_file(tmp_path):
    zip_path = _local_zip(tmp_path)
    item = DatasetItem(id='case-a', body={'task_name': 'echo'}, files={'type': 'zip', 'path': str(zip_path)})

    class CancellingReader(_MaterializeExtensionReader):
        async def iter_items(self, *, split=None, item_id=None, limit=None):
            yield item
            raise asyncio.CancelledError

    @dataset_source('auto-cancel')
    def factory(uri: DatasetSourceURI):
        return CancellingReader(uri, [item], [])

    output = tmp_path / 'tasks.jsonl'
    data_output = tmp_path / 'data_list.bin'
    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                materialize_compose(
                    [RawRunPart(kind='data', value='auto-cancel://fixture', source='test')],
                    output,
                    task_list_version=2,
                    auto_data_list=True,
                    data_list_output_path=data_output,
                )
            )
    finally:
        dataset_source_registry._registry.pop('auto-cancel', None)

    assert not output.exists()
    assert not data_output.exists()
    assert not list(tmp_path.glob('.nexus-materialize-*'))
