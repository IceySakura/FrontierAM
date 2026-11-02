from __future__ import annotations

import pytest

from nexus.dataset.item import DatasetItem
from nexus.dataset.reader import (
    BaseDatasetReader,
    DatasetDescriptor,
    DatasetSplitDescriptor,
    load_items,
    open_dataset,
)
from nexus.dataset.source import DatasetSourceURI, dataset_source, parse_dataset_source_uri
from nexus.runner.compose import compile_batch, parse_ordered_parts


class _ExampleReader(BaseDatasetReader):
    def __init__(self, uri: DatasetSourceURI) -> None:
        super().__init__()
        self.uri = uri

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        return DatasetDescriptor(
            reference=self.uri.display,
            resolved_reference=self.uri.display,
            provider='unit-source',
            name='items',
            default_split='test',
            splits=[DatasetSplitDescriptor(name='test', item_count=2)],
        )

    async def iter_item_ids(self, *, split: str | None = None, limit: int | None = None):
        self._require_open()
        del split
        values = ['one', 'two']
        if limit is not None:
            values = values[:limit]
        for value in values:
            yield value

    async def iter_items(
        self,
        *,
        split: str | None = None,
        item_id: str | None = None,
        limit: int | None = None,
    ):
        self._require_open()
        del split
        values = [item_id] if item_id else ['one', 'two']
        if limit is not None:
            values = values[:limit]
        for value in values:
            yield DatasetItem(
                id=value,
                body={
                    'task_name': 'identity',
                    'task_config': {'value': value},
                },
            )


@dataset_source('unit-source')
def _example_source_factory(uri: DatasetSourceURI) -> _ExampleReader:
    return _ExampleReader(uri)


def test_parse_dataset_source_uri() -> None:
    parsed = parse_dataset_source_uri('t3-harbor:///tmp/tasks?flat=true&include=a,b')

    assert parsed == DatasetSourceURI(
        scheme='t3-harbor',
        authority='',
        path='/tmp/tasks',
        query={
            'flat': ('true',),
            'include': ('a,b',),
        },
        raw='t3-harbor:///tmp/tasks?flat=true&include=a,b',
    )
    assert parse_dataset_source_uri('namespace/dataset@v1') is None
    assert parse_dataset_source_uri('/tmp/local-dataset') is None
    assert parse_dataset_source_uri('T3-HARBOR:///tmp/my%20tasks') == DatasetSourceURI(
        scheme='t3-harbor',
        authority='',
        path='/tmp/my tasks',
        query={},
        raw='T3-HARBOR:///tmp/my%20tasks',
    )
    with pytest.raises(ValueError, match='split.*non-empty'):
        parse_dataset_source_uri('t3-harbor:///tmp/tasks?split=')
    with pytest.raises(ValueError, match='must not contain a fragment'):
        parse_dataset_source_uri('t3-harbor:///tmp/tasks#invalid')


@pytest.mark.asyncio
async def test_open_dataset_uses_registered_source_factory() -> None:
    reader = open_dataset('unit-source:///tmp/items?split=test')

    async with reader:
        assert isinstance(reader, _ExampleReader)
        assert reader.uri.path == '/tmp/items'
        assert reader.uri.split == 'test'
        assert 'split' not in (reader.uri.query or {})
        assert [item.id for item in await load_items(reader)] == ['one', 'two']

    with pytest.raises(ValueError, match='Unknown dataset source scheme'):
        open_dataset('missing-source:///tmp/items')


def test_run_compose_loads_virtual_dataset_source() -> None:
    items = compile_batch(
        parse_ordered_parts(['--data', 'unit-source:///tmp/items']),
        limit=1,
    )

    assert len(items) == 1
    assert items[0].run_id == 'one'
    assert items[0].config.task_name == 'identity'
    assert items[0].config.task_config == {'value': 'one'}
