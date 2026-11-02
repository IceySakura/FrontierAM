from __future__ import annotations

from pathlib import Path

import pytest

from nexus.dataset.item import DatasetItem, RemoteDatasetRef
from nexus.dataset.layout import create_dataset_template
from nexus.dataset.reader import get_item, list_item_ids, load_items, open_dataset
from nexus.extensions.harbor_bridge.dataset_source import HarborDatasetReader  # noqa: F401


async def _assert_reader_contract(reader, *, provider: str, expected_ids: list[str]) -> None:
    partially_consumed = None
    async with reader:
        descriptor = await reader.describe()
        assert descriptor.provider == provider
        assert descriptor.default_split == 'test'
        assert [split.name for split in descriptor.splits] == ['test']

        assert await list_item_ids(reader) == expected_ids
        assert await list_item_ids(reader, limit=1) == expected_ids[:1]
        assert await list_item_ids(reader, limit=0) == []
        assert [item.id for item in await load_items(reader)] == expected_ids
        assert [item.id for item in await load_items(reader, limit=1)] == expected_ids[:1]
        assert await load_items(reader, limit=0) == []

        with pytest.raises(ValueError, match='non-negative'):
            await list_item_ids(reader, limit=-1)
        with pytest.raises(ValueError, match='non-negative'):
            await load_items(reader, limit=-1)
        with pytest.raises(ValueError, match='non-empty|only the test split'):
            await load_items(reader, split='')

        partially_consumed = reader.iter_item_ids()
        assert await anext(partially_consumed) == expected_ids[0]

    with pytest.raises(RuntimeError, match='inside async with'):
        await reader.describe()
    with pytest.raises(RuntimeError, match='inside async with'):
        await list_item_ids(reader, limit=0)
    assert partially_consumed is not None
    with pytest.raises(RuntimeError, match='inside async with'):
        await anext(partially_consumed)


@pytest.mark.asyncio
async def test_local_dataset_reader_contract(tmp_path: Path) -> None:
    root = create_dataset_template(tmp_path / 'local', template='generic')

    await _assert_reader_contract(
        open_dataset(str(root)),
        provider='local',
        expected_ids=['case-001', 'case-002', 'case-003'],
    )


@pytest.mark.asyncio
async def test_local_reader_rejects_duplicate_ids_for_single_item_lookup(tmp_path: Path) -> None:
    root = tmp_path / 'duplicates'
    (root / 'items').mkdir(parents=True)
    (root / 'dataset.yaml').write_text(
        'default_split: test\nsplits:\n  test:\n    items:\n      - items/one.yaml\n      - items/two.yaml\n',
        encoding='utf-8',
    )
    for name in ('one', 'two'):
        (root / 'items' / f'{name}.yaml').write_text(
            'id: duplicate\nbody:\n  task_name: identity\n',
            encoding='utf-8',
        )

    async with open_dataset(str(root)) as reader:
        with pytest.raises(ValueError, match='Duplicate item id'):
            await get_item(reader, item_id='duplicate')


@pytest.mark.asyncio
async def test_taiji_dataset_reader_contract() -> None:
    ref = RemoteDatasetRef(namespace='research', dataset='cases', version='v7', split='test')
    items = [
        DatasetItem(
            id=item_id,
            body={'task_name': 'identity', 'task_config': {'value': item_id}},
            remote_ref=ref,
        )
        for item_id in ('remote-1', 'remote-2')
    ]

    class FakeDatasetClient:
        async def resolve_version(self, uri):
            assert uri.display == 'research/cases@latest'
            return 'v7'

        async def resolve_split(self, uri, *, version, split=None):
            assert version == 'v7'
            return split or 'test'

        async def show(self, uri):
            assert uri.version == 'v7'
            return {'asset': {'name': 'cases'}, 'overview': {}}

        async def list_splits(self, uri, *, version=None):
            assert version == 'v7'
            return ['test']

        async def iter_item_ids(self, uri, *, split=None, limit=None, page_size=500):
            del uri, split, page_size
            selected = items if limit is None else items[:limit]
            for item in selected:
                yield item.id

        async def iter_items(self, uri, *, split=None, item_id=None, limit=None, page_size=500):
            del uri, split, page_size
            selected = [item for item in items if item_id is None or item.id == item_id]
            if limit is not None:
                selected = selected[:limit]
            for item in selected:
                yield item

    await _assert_reader_contract(
        open_dataset('research/cases@latest', dataset_client=FakeDatasetClient()),
        provider='taiji',
        expected_ids=['remote-1', 'remote-2'],
    )


def _write_harbor_task(root: Path, name: str) -> None:
    task = root / name
    (task / 'environment').mkdir(parents=True)
    (task / 'tests').mkdir()
    (task / 'task.toml').write_text(
        """schema_version = "1.2"

[environment]
docker_image = "task:v1"
workdir = "/workspace"
network_mode = "public"

[verifier]
environment_mode = "shared"
""",
        encoding='utf-8',
    )
    (task / 'instruction.md').write_text(f'Solve {name}.\n', encoding='utf-8')
    (task / 'tests' / 'test.sh').write_text(
        '#!/bin/sh\necho \'{"reward": 1}\' > /logs/verifier/reward.json\n',
        encoding='utf-8',
    )


@pytest.mark.asyncio
async def test_harbor_dataset_reader_contract(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path, 'harbor-1')
    _write_harbor_task(tmp_path, 'harbor-2')

    await _assert_reader_contract(
        open_dataset(f't3-harbor://{tmp_path}?flat=true'),
        provider='t3-harbor',
        expected_ids=['harbor-1', 'harbor-2'],
    )


@pytest.mark.asyncio
async def test_harbor_reader_distinguishes_invalid_and_missing_split(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path, 'harbor-1')

    async with open_dataset(f't3-harbor://{tmp_path}?flat=true') as reader:
        with pytest.raises(ValueError, match='non-empty'):
            _ = [item async for item in reader.iter_items(split='', limit=0)]
        with pytest.raises(ValueError, match='test split'):
            _ = [item async for item in reader.iter_items(split='train', limit=0)]
