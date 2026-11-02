import asyncio

import pytest

from nexus.dataset.client import DatasetClient


class FakeIo:
    def __init__(self):
        self.calls = []

    async def get_dataset(self, namespace, dataset):
        self.calls.append(('get_dataset', namespace, dataset))
        return {'latest_release': 'v1'}

    async def get_version(self, namespace, dataset, version):
        self.calls.append(('get_version', namespace, dataset, version))
        return {'default_split': 'test', 'splits': {'test': {'ids': ['case-001']}}}

    async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
        self.calls.append(('list_items', namespace, dataset, version, split, limit, cursor))
        return {'ids': ['case-001', 'case-002']}

    async def get_item(self, *args, **kwargs):
        raise AssertionError('list_item_ids must not fetch item bodies')


@pytest.mark.asyncio
async def test_list_item_ids_only_uses_list_endpoint():
    client = DatasetClient()
    client.io = FakeIo()

    ids = await client.list_item_ids('research/my_dataset', limit=1)

    assert ids == ['case-001']
    assert client.io.calls == [
        ('get_dataset', 'research', 'my_dataset'),
        ('get_version', 'research', 'my_dataset', 'v1'),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 1, None),
    ]


class FakeRawItemIo(FakeIo):
    async def get_item(self, namespace, dataset, version, item_id, *, split=None):
        self.calls.append(('get_item', namespace, dataset, version, item_id, split))
        return {'id': item_id, 'record': {'instance_id': item_id, 'repo': 'astropy/astropy'}}


@pytest.mark.asyncio
async def test_get_raw_item_accepts_flat_platform_records():
    client = DatasetClient()
    client.io = FakeRawItemIo()

    item = await client.get_raw_item('research/my_dataset', 'case-001')

    assert item == {'instance_id': 'case-001', 'repo': 'astropy/astropy'}
    assert client.io.calls == [
        ('get_dataset', 'research', 'my_dataset'),
        ('get_version', 'research', 'my_dataset', 'v1'),
        ('get_item', 'research', 'my_dataset', 'v1', 'case-001', 'test'),
    ]


class FakeUc:
    def __init__(self):
        self.calls = []

    async def create_namespace_permission(self, namespace, admin_list):
        self.calls.append(('create', namespace, admin_list))
        return {'namespace': namespace, 'admin_list': admin_list, 'creator': 'alice'}

    async def get_namespace_permission(self, namespace):
        self.calls.append(('get', namespace))
        return {'namespace': namespace, 'admin_list': ['alice'], 'creator': 'alice'}

    async def update_namespace_permission(self, namespace, admin_list):
        self.calls.append(('update', namespace, admin_list))
        return {'namespace': namespace, 'admin_list': admin_list, 'creator': 'alice'}


@pytest.mark.asyncio
async def test_namespace_permission_facade_delegates_to_uc_client():
    client = DatasetClient()
    client.uc = FakeUc()

    created = await client.create_namespace_permission('research', ['alice'])
    current = await client.get_namespace_permission('research')
    updated = await client.update_namespace_permission('research', ['alice', 'bob'])

    assert created['admin_list'] == ['alice']
    assert current['creator'] == 'alice'
    assert updated['admin_list'] == ['alice', 'bob']
    assert client.uc.calls == [
        ('create', 'research', ['alice']),
        ('get', 'research'),
        ('update', 'research', ['alice', 'bob']),
    ]


class FakeUploadIo(FakeIo):
    async def upload_local_file(self, namespace, dataset, snapshot, local_path, **kwargs):
        self.calls.append(('upload_local_file', namespace, dataset, snapshot, local_path, kwargs))
        return {'upload_id': 'upload-id'}


@pytest.mark.asyncio
async def test_upload_file_facade_delegates_to_unified_io_upload():
    client = DatasetClient()
    client.io = FakeUploadIo()

    result = await client.upload_file(
        'research',
        'my_dataset',
        'dev-202607240001',
        '/tmp/large file.parquet',
        part_size=128 * 1024**2,
        expires=3600,
        max_concurrency=3,
    )

    assert result == {'upload_id': 'upload-id'}
    assert client.io.calls == [
        (
            'upload_local_file',
            'research',
            'my_dataset',
            'dev-202607240001',
            '/tmp/large file.parquet',
            {'part_size': 128 * 1024**2, 'expires': 3600, 'max_concurrency': 3},
        )
    ]


class FakePagedIo(FakeIo):
    async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
        self.calls.append(('list_items', namespace, dataset, version, split, limit, cursor))
        pages = {
            None: {
                'items': [
                    {'id': 'case-001', 'body': {'task_name': 'echo'}},
                    {'id': 'case-002', 'body': {'task_name': 'echo'}},
                ],
                'next_cursor': 'page-2',
            },
            'page-2': {
                'items': [
                    {'id': 'case-003', 'body': {'task_name': 'echo'}},
                    {'id': 'case-004', 'body': {'task_name': 'echo'}},
                ],
                'next_cursor': 'page-3',
            },
            'page-3': {'items': [{'id': 'case-005', 'body': {'task_name': 'echo'}}]},
        }
        return pages[cursor]


def test_iter_items_pages_with_bounded_requests_and_total_limit():
    async def collect():
        client = DatasetClient()
        client.io = FakePagedIo()
        items = [item async for item in client.iter_items('research/my_dataset', limit=3, page_size=2)]
        return client, items

    client, items = asyncio.run(collect())

    assert [item.id for item in items] == ['case-001', 'case-002', 'case-003']
    assert all(item.remote_ref is not None and item.remote_ref.version == 'v1' for item in items)
    assert client.io.calls == [
        ('get_dataset', 'research', 'my_dataset'),
        ('get_version', 'research', 'my_dataset', 'v1'),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 2, None),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 1, 'page-2'),
    ]


def test_iter_resolved_items_rejects_repeated_cursor():
    class RepeatingCursorIo(FakePagedIo):
        async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
            return {
                'items': [{'id': f'case-{cursor or "first"}', 'body': {'task_name': 'echo'}}],
                'next_cursor': 'same',
            }

    async def collect():
        client = DatasetClient()
        client.io = RepeatingCursorIo()
        ref = await client.resolve_ref('research/my_dataset')
        return [item async for item in client.iter_resolved_items(ref, page_size=1)]

    with pytest.raises(ValueError, match='repeated cursor'):
        asyncio.run(collect())


class FakePagedIdIo(FakeIo):
    async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
        self.calls.append(('list_items', namespace, dataset, version, split, limit, cursor))
        pages = {
            None: {
                'ids': ['case-001', 'case-002'],
                'pagination': {'has_more': True, 'next_cursor': 'empty-page'},
            },
            'empty-page': {
                'ids': [],
                'pagination': {'has_more': True, 'next_cursor': 'last-page'},
            },
            'last-page': {'ids': ['case-003', 'case-004']},
        }
        return pages[cursor]


@pytest.mark.asyncio
async def test_list_item_ids_pages_in_order_with_global_limit():
    client = DatasetClient()
    client.io = FakePagedIdIo()

    ids = await client.list_item_ids('research/my_dataset', limit=3)

    assert ids == ['case-001', 'case-002', 'case-003']
    assert client.io.calls == [
        ('get_dataset', 'research', 'my_dataset'),
        ('get_version', 'research', 'my_dataset', 'v1'),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 3, None),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 1, 'empty-page'),
        ('list_items', 'research', 'my_dataset', 'v1', 'test', 1, 'last-page'),
    ]


@pytest.mark.asyncio
async def test_list_item_ids_reads_ids_from_nested_records():
    class NestedRecordIo(FakeIo):
        async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
            del namespace, dataset, version, split, limit, cursor
            return {
                'items': [
                    {
                        'record': {
                            'id': 'nested-001',
                            'body': {'task_name': 'echo'},
                        }
                    },
                    {
                        'id': 'outer-002',
                        'record': {
                            'body': {'task_name': 'echo'},
                        },
                    },
                ]
            }

    client = DatasetClient()
    client.io = NestedRecordIo()

    ids = await client.list_item_ids('research/my_dataset')
    items = [item async for item in client.iter_items('research/my_dataset')]

    assert ids == ['nested-001', 'outer-002']
    assert [item.id for item in items] == ids


@pytest.mark.asyncio
async def test_iter_items_rejects_records_without_canonical_ids():
    class MissingIdIo(FakeIo):
        async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
            del namespace, dataset, version, split, limit, cursor
            return {'items': [{'body': {'task_name': 'echo'}}]}

    client = DatasetClient()
    client.io = MissingIdIo()

    with pytest.raises(ValueError, match='missing a canonical item id'):
        _ = [item async for item in client.iter_items('research/my_dataset')]


@pytest.mark.asyncio
async def test_list_item_ids_rejects_repeated_cursor():
    class RepeatingCursorIo(FakeIo):
        async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
            return {'ids': [f'case-{cursor or "first"}'], 'next_cursor': 'same'}

    client = DatasetClient()
    client.io = RepeatingCursorIo()

    with pytest.raises(ValueError, match='repeated cursor'):
        await client.list_item_ids('research/my_dataset')


@pytest.mark.asyncio
async def test_list_item_ids_rejects_missing_cursor_when_more_items_are_reported():
    class MissingCursorIo(FakeIo):
        async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
            return {'ids': ['case-001'], 'has_more': True}

    client = DatasetClient()
    client.io = MissingCursorIo()

    with pytest.raises(ValueError, match='without a next cursor'):
        await client.list_item_ids('research/my_dataset')


class FakeConcurrentIdIo(FakeIo):
    item_ids = [f'case-{index:03d}' for index in range(32)]

    def __init__(self):
        super().__init__()
        self.active_fetches = 0
        self.max_active_fetches = 0

    async def list_items(self, namespace, dataset, version, *, split=None, limit=None, cursor=None):
        self.calls.append(('list_items', namespace, dataset, version, split, limit, cursor))
        if cursor is None:
            return {'ids': self.item_ids[:20], 'next_cursor': 'page-2'}
        return {'ids': self.item_ids[20:]}

    async def get_item(self, namespace, dataset, version, item_id, *, split=None):
        self.active_fetches += 1
        self.max_active_fetches = max(self.max_active_fetches, self.active_fetches)
        try:
            index = int(item_id.rsplit('-', 1)[1])
            await asyncio.sleep((len(self.item_ids) - index) / 10_000)
            return {'id': item_id, 'body': {'task_name': 'echo'}}
        finally:
            self.active_fetches -= 1


@pytest.mark.asyncio
async def test_iter_resolved_items_fetches_id_only_pages_concurrently_in_order():
    client = DatasetClient()
    client.io = FakeConcurrentIdIo()
    ref = await client.resolve_ref('research/my_dataset')

    items = [item async for item in client.iter_resolved_items(ref)]

    assert [item.id for item in items] == client.io.item_ids
    assert client.io.max_active_fetches == 16
    assert [call[-1] for call in client.io.calls if call[0] == 'list_items'] == [None, 'page-2']


class _FakeSession:
    """Minimal stand-in for aiohttp.ClientSession that records close()."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_aenter_closes_session_when_client_construction_fails(monkeypatch, caplog):
    import nexus.dataset.client as client_module

    created_sessions: list[_FakeSession] = []

    def _make_session(*args, **kwargs):
        session = _FakeSession()
        created_sessions.append(session)
        return session

    def _boom(*args, **kwargs):
        raise RuntimeError('io client construction failed')

    monkeypatch.setattr(client_module.aiohttp, 'ClientSession', _make_session)
    monkeypatch.setattr(client_module, 'DatasetIoClient', _boom)

    client = DatasetClient()
    with caplog.at_level('WARNING', logger='nexus.dataset.client'):
        with pytest.raises(RuntimeError, match='io client construction failed'):
            await client.__aenter__()

    # The session must be created, closed, and the half-initialized state cleared.
    assert len(created_sessions) == 1
    assert created_sessions[0].closed is True
    assert client.session is None
    assert client.io is None
    assert client.uc is None
    # The failure must be logged (with the original error message) for triage.
    assert any(
        record.levelname == 'WARNING' and 'io client construction failed' in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_aexit_closes_session_on_normal_path(monkeypatch):
    import nexus.dataset.client as client_module

    created_sessions: list[_FakeSession] = []

    def _make_session(*args, **kwargs):
        session = _FakeSession()
        created_sessions.append(session)
        return session

    # Stub out the sub-clients so construction never touches the environment
    # (auth files, env vars); this test only cares about session lifecycle.
    def _make_stub(*args, **kwargs):
        return object()

    monkeypatch.setattr(client_module.aiohttp, 'ClientSession', _make_session)
    monkeypatch.setattr(client_module, 'DatasetIoClient', _make_stub)
    monkeypatch.setattr(client_module, 'UcMetadataClient', _make_stub)

    async with DatasetClient() as client:
        assert client.session is created_sessions[0]
        assert client.io is not None
        assert client.uc is not None

    assert created_sessions[0].closed is True
