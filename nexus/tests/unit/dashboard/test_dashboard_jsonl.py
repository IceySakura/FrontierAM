import json

from pathlib import Path

import pytest

from aiohttp.test_utils import TestClient, TestServer
from tests.unit.dashboard.test_dashboard_run_results import run_result, trajectory

from nexus.dashboard.server import create_app
from nexus.dashboard.store import ArtifactStore


def encode(value):
    return json.dumps(value, ensure_ascii=False).encode('utf-8')


def test_jsonl_isolates_bad_lines_and_clients_and_preserves_line_numbers(tmp_path):
    path = tmp_path / 'batch.jsonl'
    lines = [
        encode(run_result(trajectory('重复 session'))),
        b'',
        b'{broken',
        encode(run_result({'first': trajectory('重复 session'), 'other': trajectory('client-2'), 'bad': None})),
        encode(run_result(None)),
        b'[]',
        b'\xff',
        encode({'trajectory': trajectory('wrapped'), 'artifact_id': 'exported'}),
        encode(trajectory('native')),
    ]
    path.write_bytes(b'\r\n'.join(lines))
    store = ArtifactStore(path)
    catalog = store.scan()
    assert len(catalog['items']) == 5
    assert len({item['id'] for item in catalog['items']}) == 5
    assert [item['line'] for item in catalog['items']] == [1, 4, 4, 8, 9]
    assert {error['line'] for error in catalog['errors']} == {3, 4, 5, 6, 7}
    assert next(error for error in catalog['errors'] if error['line'] == 4)['client'] == 'bad'
    for item in catalog['items']:
        detail = store.load(item['id'])
        assert detail['trajectory']['session_id'] == item['session_id']
        assert detail['source']['line'] == item['line']
        if item['line'] == 8:
            assert detail['export'] == {'artifact_id': 'exported'}
    assert store.scan() == catalog


def test_jsonl_limit_is_per_record_not_per_file_and_skips_oversized_lines(tmp_path):
    row = encode(run_result(trajectory()))
    path = tmp_path / 'large.jsonl'
    path.write_bytes(row + b'\n' + b'x' * (len(row) * 3) + b'\n' + row + b'\r\n' + row)
    assert path.stat().st_size > len(row)
    store = ArtifactStore(path, max_bytes=len(row))
    result = store.scan()
    assert [item['line'] for item in result['items']] == [1, 3, 4]
    assert len(result['errors']) == 1
    assert result['errors'][0]['line'] == 2
    assert 'read limit' in result['errors'][0]['error']
    for item in result['items']:
        assert store.load(item['id'])['trajectory']['session_id'] == 'sample'
    too_small = ArtifactStore(path, max_bytes=len(row) - 1).scan()
    assert not too_small['items']
    assert len(too_small['errors']) == 4


def test_jsonl_cached_catalog_and_detail_only_seek_and_read_one_line(tmp_path, monkeypatch):
    path = tmp_path / 'batch.jsonl'
    row1 = encode(run_result(trajectory('中文')))
    row2 = encode(run_result(trajectory('target')))
    path.write_bytes(row1 + b'\r\n' + row2 + b'\n')
    store = ArtifactStore(path)
    catalog = store.scan()
    item = catalog['items'][1]
    opened, reads, seeks = [], [], []
    original_open = Path.open

    class TrackedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def seek(self, offset):
            seeks.append(offset)
            return self.stream.seek(offset)

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    def tracked_open(current, *args, **kwargs):
        opened.append(current)
        return TrackedReader(original_open(current, *args, **kwargs))

    monkeypatch.setattr(Path, 'open', tracked_open)
    assert store.scan() == catalog
    assert not opened
    assert store.load(item['id'])['trajectory']['session_id'] == 'target'
    assert opened == [path]
    assert seeks == [len(row1) + 2]
    assert reads == [len(row2) + 1]


@pytest.mark.parametrize('replacement', [b'', b'{unfinished', encode(run_result(trajectory('replacement')))])
def test_modified_or_truncated_jsonl_rejects_stale_offsets(tmp_path, replacement):
    path = tmp_path / 'batch.jsonl'
    path.write_bytes(encode(run_result(trajectory('original'))))
    store = ArtifactStore(path)
    (item,) = store.scan()['items']
    path.write_bytes(replacement)
    with pytest.raises(ValueError, match='refresh'):
        store.load(item['id'])
    current = store.scan()
    if current['items']:
        assert store.load(current['items'][0]['id'])['trajectory']['session_id'] == 'replacement'
    else:
        with pytest.raises(KeyError):
            store.load(item['id'])


def test_jsonl_append_is_only_visible_on_explicit_catalog_refresh(tmp_path):
    path = tmp_path / 'batch.jsonl'
    row = encode(run_result(trajectory()))
    path.write_bytes(row + b'\n')
    store = ArtifactStore(path)
    (first,) = store.scan()['items']
    with path.open('ab') as stream:
        stream.write(row + b'\n{partial')
    with pytest.raises(ValueError, match='refresh'):
        store.load(first['id'])
    catalog = store.scan()
    assert [item['line'] for item in catalog['items']] == [1, 2]
    assert catalog['items'][0]['id'] == first['id']
    assert catalog['errors'][0]['line'] == 3


def test_external_jsonl_symlink_is_not_read_or_loaded(tmp_path):
    root = tmp_path / 'allowed'
    root.mkdir()
    outside = tmp_path / 'outside.jsonl'
    outside.write_bytes(encode(trajectory()))
    path = root / 'batch.jsonl'
    path.symlink_to(outside)
    catalog = ArtifactStore(root).scan()
    assert not catalog['items']
    assert 'within the selected directory' in catalog['errors'][0]['error']
    path.unlink()
    path.write_bytes(encode(trajectory()))
    store = ArtifactStore(root)
    (item,) = store.scan()['items']
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match='within the selected directory'):
        store.load(item['id'])


def test_empty_or_whitespace_jsonl_is_an_empty_catalog(tmp_path):
    path = tmp_path / 'empty.jsonl'
    path.write_bytes(b'\n\r\n  \n')
    assert ArtifactStore(path).scan() == {'items': [], 'errors': []}


@pytest.mark.asyncio
async def test_jsonl_pagination_query_and_detail_use_distinct_record_ids(tmp_path):
    path = tmp_path / 'batch.jsonl'
    path.write_bytes(b'\n'.join(encode(run_result(trajectory(str(index)))) for index in range(5)))
    async with TestClient(TestServer(create_app(path))) as client:
        catalog = await (await client.get('/api/trajectories?offset=2&limit=2')).json()
        assert catalog['total'] == 5
        assert [item['line'] for item in catalog['items']] == [3, 4]
        item = catalog['items'][0]
        response = await client.get('/api/trajectories/' + item['id'])
        assert response.status == 200
        detail = await response.json()
        assert detail['trajectory']['session_id'] == '2'
        assert detail['graph']['segments']
        result = await (await client.get('/api/trajectories?q=4')).json()
        assert [item['session_id'] for item in result['items']] == ['4']
        path.write_bytes(encode(run_result(trajectory('changed'))))
        response = await client.get('/api/trajectories/' + item['id'])
        assert response.status == 422
        assert 'refresh' in await response.text()
