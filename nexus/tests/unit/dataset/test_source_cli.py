from __future__ import annotations

import argparse
import json

import pytest


pytest.importorskip('sqlalchemy')

from nexus.cli.commands.dataset import dataset_get, dataset_items, dataset_show, dataset_versions
from nexus.dataset.item import DatasetItem
from nexus.dataset.reader import BaseDatasetReader, DatasetDescriptor, DatasetSplitDescriptor
from nexus.dataset.source import DatasetSourceURI, dataset_source


class _CLIReader(BaseDatasetReader):
    def __init__(self, uri: DatasetSourceURI) -> None:
        super().__init__()
        self.uri = uri

    async def describe(self) -> DatasetDescriptor:
        self._require_open()
        return DatasetDescriptor(
            reference=self.uri.display,
            resolved_reference=self.uri.display,
            provider='cli-source',
            name='tasks',
            default_split='test',
            splits=[DatasetSplitDescriptor(name='test', item_count=2)],
        )

    async def iter_item_ids(self, *, split: str | None = None, limit: int | None = None):
        self._require_open()
        del split
        values = ['alpha', 'beta']
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
        del split, limit
        yield DatasetItem(
            id=item_id or 'alpha',
            body={'task_name': 'identity', 'task_config': {'value': item_id or 'alpha'}},
        )


@dataset_source('cli-source')
def _cli_source_factory(uri: DatasetSourceURI) -> _CLIReader:
    return _CLIReader(uri)


def test_dataset_cli_uses_virtual_source(capsys, caplog) -> None:
    uri = 'cli-source:///tmp/tasks?split=test'

    assert dataset_show(argparse.Namespace(dataset=uri, json=True)) == 0
    assert json.loads(capsys.readouterr().out)['provider'] == 'cli-source'

    assert dataset_versions(argparse.Namespace(dataset=uri, json=True)) == 1
    assert 'does not support version catalogs' in caplog.text

    assert (
        dataset_items(
            argparse.Namespace(
                dataset=uri,
                split=None,
                limit=1,
                json=True,
                keyword=None,
            )
        )
        == 0
    )
    items_payload = json.loads(capsys.readouterr().out)
    assert items_payload['split'] == 'test'
    assert items_payload['items'] == [{'id': 'alpha'}]

    assert (
        dataset_items(
            argparse.Namespace(
                dataset=uri,
                split=None,
                limit=1,
                details=True,
                json=True,
                keyword=None,
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)['items'] == [
        {
            'id': 'alpha',
            'metadata': {},
            'files': None,
        }
    ]

    assert (
        dataset_get(
            argparse.Namespace(
                dataset=uri,
                item_id='beta',
                split=None,
                output=None,
            )
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)['body']['task_config'] == {'value': 'beta'}
