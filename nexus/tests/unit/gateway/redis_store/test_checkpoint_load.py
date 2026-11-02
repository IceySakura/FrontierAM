"""Online restore preserves ownership through failures and cancellation."""

import asyncio
import threading

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import DEFAULT, Mock

import numpy as np
import pytest

from tests.unit.gateway.redis_store.fake_redis import FakeRedis
from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS
from tests.unit.gateway.redis_store.test_checkpoint import DumpFS

from nexus.gateway.trajectory_store_backends.redis import (
    RedisTrajectoryRecordStore,
    _RedisDumpRef,
)
from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import iter_payloads
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.blob_codec import decode_blob
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError
from nexus.gateway.trajectory_store_backends.types import NodePayloadRecord, SegmentDataRecord


class LoadRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.connection_pool = Mock(spec=['get_connection', 'release'])
        self.connection_pool.get_connection.side_effect = lambda: Mock(spec=['send_command', 'disconnect'])
        self.sent_commands = []
        self.before_command = None
        self.after_command = None
        self.closed = False

    def parse_response(self, connection, command_name):
        return self.execute_sent_command(*connection.send_command.call_args.args)

    def execute_sent_command(self, *args):
        self.sent_commands.append(args)
        if self.before_command:
            self.before_command(args)
        result = self.set(args[1], args[2], nx=args[3] == 'NX') if args[0] == 'SET' else self.execute_command(*args)
        if self.after_command:
            self.after_command(args, result)
        return result

    def close(self):
        self.closed = True


class ImportFS(MemoryFS):
    def __init__(self):
        super().__init__()
        self.imports = {}
        self.sources = []
        self.fail_source = None

    def import_file_to_ssd(self, url, source_path):
        assert isinstance(source_path, str)
        if self.imports.get(url, {}).get('state') == 'finished':
            raise RedisClientError('import aborted', code='conflict')
        self.sources.append(source_path)
        self.imports[url] = {'state': 'running', 'error': None}
        if self.fail_source and self.fail_source(source_path):
            self.imports[url] = {'state': 'failed', 'error': 'copy failed'}
            raise RedisClientError('copy failed', code='io_error')
        self.blobs[url] = Path(source_path).read_bytes()
        self.imports[url] = {'state': 'imported', 'error': None}

    def get_ssd_import_status(self, urls):
        return {'items': {url: dict(self.imports.get(url, {'state': 'unknown', 'error': None})) for url in urls}}

    def finalize_ssd_imports(self, urls, keep_urls):
        for url in urls:
            if url in keep_urls:
                if self.imports.get(url, {}).get('state') != 'unknown':
                    self.imports[url] = {'state': 'kept', 'error': None}
            else:
                self.blobs.pop(url, None)
                self.imports[url] = {'state': 'finished', 'error': None}
        return self.get_ssd_import_status(urls)


async def dump_records(tmp_path, records):
    raw = FakeRedis()
    with RedisTrajectoryRecordStore('source', {'redis_client': raw, 'fs_client': DumpFS(raw)}) as source:
        refs = await source.put_many(records)
        dumped, write = source.dump_payloads(refs, dump_dir=str(tmp_path))
        await write
    return dumped


@pytest.fixture
def target():
    with ExitStack() as resources:

        def create(*, fs=None):
            raw = LoadRedis()
            config = {'redis_client': raw, 'key_prefix': 'restored', 'fs_client': fs if fs is not None else ImportFS()}
            store = resources.enter_context(RedisTrajectoryRecordStore('new-run', config))
            return store, raw

        yield create


@pytest.mark.parametrize(
    'invalid',
    [
        'dump_id',
        'kind',
        'entry_id',
        'fields',
        'legacy_path',
        'wrong_prefix',
        'empty_path',
        'parent_path',
        'absolute_path',
        'missing_tokens',
        'missing_blob',
        'old_schema',
    ],
)
async def test_preflight_validates_all_records_before_any_target_write(tmp_path, target, invalid):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dumped = await dump_records(
        tmp_path,
        [
            NodePayloadRecord('t', 1, np.array([1], dtype=np.int32)),
            NodePayloadRecord('t', 2, np.array([2], dtype=np.int32), routed_experts=[3]),
        ],
    )
    ref = dumped[1]
    edits = {'dump_id': 'wrong', 'kind': 'segment', 'entry_id': 999, 'fields': ('topk_values',)}
    if invalid in edits:
        ref = replace(ref, **{invalid: edits[invalid]})
    elif invalid == 'missing_blob':
        next((tmp_path / 'data').iterdir()).unlink()
        metadata = list(iter_payloads(tmp_path, load_blobs=False))
        assert metadata[1]['fields']['routed_experts'].startswith('ckpt://data/')
    elif invalid == 'old_schema':
        path = tmp_path / 'trajectory.parquet'
        table = pq.read_table(path)
        legacy_array = pa.struct([
            ('dtype', pa.string()),
            ('shape', pa.list_(pa.int64())),
            ('source', pa.string()),
            ('integers', pa.large_list(pa.int64())),
            ('unsigned', pa.large_list(pa.uint64())),
            ('floats', pa.large_list(pa.float64())),
            ('booleans', pa.large_list(pa.bool_())),
        ])
        for name in ('token_ids', 'logprobs', 'topk_indices', 'topk_values'):
            values = [
                {'dtype': '<i4', 'shape': [1], 'source': 'ndarray', 'integers': [value]}
                if name == 'token_ids'
                else None
                for value in (1, 2)
            ]
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values, type=legacy_array))
        pq.write_table(table, path)
        with pytest.raises(ValueError, match='checkpoint schema'):
            list(iter_payloads(tmp_path, fields=[], summaries=True))
    else:
        path = tmp_path / 'trajectory.parquet'
        table = pq.read_table(path)
        name = 'token_ids' if invalid == 'missing_tokens' else 'routed_experts'
        index = table.schema.get_field_index(name)
        source = table.column('routed_experts')[1].as_py().removeprefix('ckpt://')
        assert (tmp_path / source).is_file()
        value = {
            'missing_tokens': None,
            'legacy_path': source,
            'wrong_prefix': f'file://{source}',
            'empty_path': 'ckpt://',
            'parent_path': 'ckpt://data/../outside.blob',
            'absolute_path': 'ckpt:///tmp/a.blob',
        }[invalid]
        values = [table.column(index)[0].as_py(), value]
        table = table.set_column(
            index, table.schema.field(index), pa.array(values, type=table.schema.field(index).type)
        )
        pq.write_table(table, path)
        if name == 'routed_experts':
            for load_blobs in (False, True):
                with pytest.raises(ValueError, match='checkpoint blob'):
                    list(iter_payloads(tmp_path, load_blobs=load_blobs))
            assert len(list(iter_payloads(tmp_path, fields=['token_ids']))) == 2
    store, raw = target(fs=ImportFS())
    assert await store.load_payloads([], dump_dir=str(tmp_path / 'absent')) == []
    assert not (tmp_path / 'absent').exists()
    error = FileNotFoundError if invalid == 'missing_blob' else ValueError
    with pytest.raises(error, match='checkpoint|token_ids|blob path|No such file'):
        await store.load_payloads([dumped[0], ref], dump_dir=str(tmp_path))
    assert raw.sent_commands == []
    if invalid in ('legacy_path', 'wrong_prefix', 'empty_path', 'parent_path', 'absolute_path'):
        restored = await store.load_payloads([replace(ref, fields=('token_ids',))], dump_dir=str(tmp_path))
        np.testing.assert_array_equal((await store.get_many(restored))[0]['token_ids'], [2])


async def test_dump_requires_token_ids_only_for_nodes(tmp_path):
    with RedisTrajectoryRecordStore('source', {'redis_client': FakeRedis(), 'fs_client': MemoryFS()}) as source:
        refs = await source.put_many([
            NodePayloadRecord('t', 1, np.array([1], dtype=np.int32)),
            SegmentDataRecord('t', 's', {'reward': 1}),
        ])
        with pytest.raises(ValueError, match='token_ids'):
            source.dump_payloads([refs[0]], dump_dir=str(tmp_path), fields=['topk_values'])
        dumped, write = source.dump_payloads([refs[1]], dump_dir=str(tmp_path), fields=['reward'])
        await write
        assert isinstance(dumped[0], _RedisDumpRef)
        assert dumped[0].kind == 'segment'


@pytest.mark.parametrize('failure', ['redis', 'du_copy'])
async def test_known_failure_preserves_partial_ack_refs_and_cleans_pending(tmp_path, target, failure):
    dumped = await dump_records(
        tmp_path,
        [NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i]) for i in range(3)],
    )
    fs = ImportFS()
    store, raw = target(fs=fs)
    if failure == 'du_copy':
        fs.fail_source = lambda path: decode_blob(Path(path).read_bytes()) == [1]
    else:
        writes = 0

        def fail_second(args):
            nonlocal writes
            if args[0] == 'SET':
                writes += 1
                if writes == 2:
                    raise RedisClientError('known write failure', code='unavailable')

        raw.before_command = fail_second
    with pytest.raises(RedisClientError, match='checkpoint load failed') as caught:
        await store.load_payloads([dumped[0], dumped[1], dumped[0], dumped[2]], dump_dir=str(tmp_path))
    body = caught.value.body
    assert body['partial_refs'][0].record_id == body['partial_refs'][2].record_id
    assert body['partial_refs'][1] is None
    assert body['partial_refs'][3] is not None
    assert body['errors'].keys() == {1}
    assert body['unknown'] == {}
    assert len(raw._kv) == 2
    assert raw._dump_protections == {}
    assert len(fs.blobs) == 2


async def test_unknown_set_is_never_replayed_or_deleted_and_stops_later_records(tmp_path, target):
    dumped = await dump_records(tmp_path, [NodePayloadRecord('t', i, np.array([i], dtype=np.int32)) for i in range(3)])
    store, raw = target()

    def lost_reply(args, result):
        if args[0] == 'SET':
            raise TimeoutError('lost SET reply')

    raw.after_command = lost_reply
    with pytest.raises(RedisClientError) as caught:
        await store.load_payloads(dumped, dump_dir=str(tmp_path))
    assert len([call for call in raw.sent_commands if call[0] == 'SET']) == 1
    assert len(raw._kv) == 1
    assert caught.value.body['partial_refs'] == [None, None, None]
    assert 0 in caught.value.body['unknown']
    assert raw._dump_protections == {}
    raw.connection_pool.release.call_args.args[0].disconnect.assert_called_once()


@pytest.mark.parametrize('with_blob', [False, True])
async def test_cancel_drains_then_reclaims_ack_records_and_imports(tmp_path, with_blob, target):
    dumped = await dump_records(
        tmp_path,
        [
            NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i] if with_blob else None)
            for i in range(2)
        ],
    )
    fs = ImportFS()
    store, raw = target(fs=fs)
    entered, release = threading.Event(), threading.Event()
    writes = 0

    def block(args, result):
        nonlocal writes
        if args[0] == 'SET':
            writes += 1
            if writes == 2:
                entered.set()
                assert release.wait(5)

    raw.after_command = block
    try:
        task = asyncio.create_task(store.load_payloads(dumped, dump_dir=str(tmp_path)))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel('cancel-load')
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert raw._kv == {}
        assert raw._dump_protections == {}
        assert not fs.blobs
        assert bool(list((tmp_path / 'data').glob('*.blob'))) == with_blob
    finally:
        release.set()


async def test_close_waits_for_worker_and_cleanup_without_loop_progress(tmp_path):
    dumped = await dump_records(tmp_path, [NodePayloadRecord('t', 1, np.array([1], dtype=np.int32))])
    raw = LoadRedis()
    store = RedisTrajectoryRecordStore(
        'new-run', {'redis_client': raw, 'key_prefix': 'restored', 'fs_client': ImportFS()}
    )
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def block(args, result):
        if args[0] == 'SET':
            entered.set()
            assert release.wait(5)
            assert not raw.closed

    raw.after_command = block
    task = asyncio.create_task(store.load_payloads(dumped, dump_dir=str(tmp_path)))
    assert await asyncio.to_thread(entered.wait, 3)
    closer = threading.Thread(target=lambda: (store.close(), closed.set()))
    closer.start()
    assert not await asyncio.to_thread(closed.wait, 0.05)
    release.set()
    assert await asyncio.to_thread(closed.wait, 3)
    closer.join()
    assert len(await task) == 1
    assert raw.closed


@pytest.mark.parametrize('fail_close', [False, True])
async def test_concurrent_close_waits_and_rejects_new_load_even_if_client_close_fails(
    tmp_path, target, monkeypatch, fail_close
):
    store, raw = target()
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    original = raw.close

    def close():
        entered.set()
        assert release.wait(5)
        original()
        if fail_close:
            raise OSError('close failed')

    def close_again():
        waiting.set()
        store.close()

    monkeypatch.setattr(raw, 'close', Mock(side_effect=close))
    ref = _RedisDumpRef('not-opened', 0, 'node', None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        owner = pool.submit(store.close)
        try:
            assert entered.wait(2)
            waiter = pool.submit(close_again)
            assert waiting.wait(2)
            assert not waiter.done()
            with pytest.raises(RedisClientError) as caught:
                await store.load_payloads([ref], dump_dir=str(tmp_path))
            assert caught.value.code == 'closed'
        finally:
            release.set()
        if fail_close:
            with pytest.raises(OSError, match='close failed'):
                owner.result(timeout=2)
        else:
            owner.result(timeout=2)
        waiter.result(timeout=2)
    with pytest.raises(RedisClientError) as caught:
        await store.load_payloads([ref], dump_dir=str(tmp_path))
    assert caught.value.code == 'closed'
    store.close()
    assert raw.close.call_count == 1


async def test_lost_commit_reply_retains_blob_without_replay(tmp_path, target):
    from nexus.gateway.trajectory_store_backends.redis_store.client import _RECORD_CAS

    dumped = await dump_records(
        tmp_path, [NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i]) for i in range(2)]
    )
    fs = ImportFS()
    store, raw = target(fs=fs)

    def lost_reply(args, result):
        if args[0] == 'EVAL' and args[1] == _RECORD_CAS:
            raise TimeoutError('lost CAS reply')

    raw.after_command = lost_reply
    with pytest.raises(RedisClientError, match='checkpoint load failed') as caught:
        await store.load_payloads(dumped, dump_dir=str(tmp_path))
    body = caught.value.body
    assert body['partial_refs'] == [None, None]
    assert body['unknown'].keys() == {0}
    assert len([call for call in raw.sent_commands if call[0] == 'EVAL' and call[1] == _RECORD_CAS]) == 1
    assert len(raw._kv) == 1
    assert set(fs.blobs) == set(body['unknown'][0]['blob_urls'].values())
    assert raw._dump_protections == {}


@pytest.mark.parametrize('delete_failure', ['conflict', 'unknown'])
async def test_cancel_preserves_blob_if_conditional_redis_cleanup_is_not_confirmed(tmp_path, delete_failure, target):
    dumped = await dump_records(
        tmp_path, [NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3])]
    )
    fs = ImportFS()
    store, raw = target(fs=fs)
    entered, release = threading.Event(), threading.Event()
    reported = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: reported.append(context))

    from nexus.gateway.trajectory_store_backends.redis_store.client import _RECORD_CAS

    def block(args, result):
        if args[0] == 'EVAL' and args[1] == _RECORD_CAS:
            entered.set()
            assert release.wait(5)

    def interfere(args):
        if args[0] == 'EVAL' and 'nexus:load-delete' in args[1]:
            if delete_failure == 'unknown':
                raise TimeoutError('delete reply unknown')
            raw.set(args[3], b'someone-else')

    raw.after_command, raw.before_command = block, interfere
    try:
        task = asyncio.create_task(store.load_payloads(dumped, dump_dir=str(tmp_path)))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(raw._kv) == 1
        assert raw._dump_protections == {}
        assert len(fs.blobs) == 1
        assert {item['state'] for item in fs.imports.values()} == {'kept'}
        assert reported
        assert reported[0]['unknown']
    finally:
        release.set()
        loop.set_exception_handler(previous)


@pytest.mark.parametrize('invalid', [False, True])
async def test_late_cancellation_delivers_already_terminal_result_or_input_error(
    tmp_path, monkeypatch, invalid, target
):
    from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import load as load_module
    from nexus.gateway.trajectory_store_backends.types import DumpRef

    dumped = await dump_records(tmp_path, [NodePayloadRecord('t', 1, np.array([1], dtype=np.int32))])
    store, _ = target()
    caller = asyncio.current_task()
    loop = asyncio.get_running_loop()
    original = load_module._CheckpointLoader.run

    def cancel_after_terminal(self):
        try:
            return original(self)
        finally:
            loop.call_soon_threadsafe(caller.cancel)

    monkeypatch.setattr(load_module._CheckpointLoader, 'run', cancel_after_terminal)
    if invalid:
        with pytest.raises(TypeError, match='checkpoint references'):
            await store.load_payloads([DumpRef()], dump_dir=str(tmp_path))
    else:
        assert len(await store.load_payloads(dumped, dump_dir=str(tmp_path))) == 1


async def test_shared_source_file_gets_separate_target_ownership_per_entry(tmp_path, target):
    import pyarrow as pa
    import pyarrow.parquet as pq

    dumped = await dump_records(
        tmp_path, [NodePayloadRecord('t', i, np.array([i], dtype=np.int32), routed_experts=[i]) for i in range(2)]
    )
    path = tmp_path / 'trajectory.parquet'
    table = pq.read_table(path)
    index = table.schema.get_field_index('routed_experts')
    source = table.column(index)[0].as_py()
    table = table.set_column(index, 'routed_experts', pa.array([source, source], type=pa.string()))
    pq.write_table(table, path)
    fs = ImportFS()
    store, _ = target(fs=fs)
    refs = await store.load_payloads(dumped, dump_dir=str(tmp_path))
    assert set(refs[0].blob_urls.values()).isdisjoint(refs[1].blob_urls.values())
    assert len(fs.blobs) == 2
    await store.release_many([refs[0]])
    assert (await store.get_many([refs[1]]))[0]['routed_experts'] == [0]
    assert (tmp_path / source.removeprefix('ckpt://')).is_file()


async def test_segment_load_preserves_field_pickle_without_unpickling_values(tmp_path, monkeypatch, target):
    import pickle

    dumped = await dump_records(tmp_path, [SegmentDataRecord('t', 's', {'reward': 2})])
    store, _ = target()

    def forbidden(blob):
        raise AssertionError('load must retain encoded Segment fields')

    with monkeypatch.context() as patch:
        patch.setattr(pickle, 'loads', forbidden)
        refs = await store.load_payloads(dumped, dump_dir=str(tmp_path))
    assert (await store.get_many(refs)) == [{'reward': 2}]


def test_cluster_load_single_send_uses_private_retry_zero_without_changing_shared_client():
    from unittest.mock import MagicMock

    from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient

    raw = MagicMock()
    pipe = raw.pipeline.return_value.__enter__.return_value
    pipe.send_cluster_commands.side_effect = TimeoutError('unknown send')
    with RedisClient(redis_client=raw, fs_client=MemoryFS()) as client:
        retry_calls = tuple(raw.set_retry.call_args_list)
        with pytest.raises(TimeoutError, match='unknown send'):
            client._execute_without_retry('SET', b'key', b'value', 'NX')
        assert pipe.retry._retries == 0
        assert pipe.send_cluster_commands.call_count == 1
        assert pipe.send_cluster_commands.call_args.kwargs['allow_redirections'] is False
        assert tuple(raw.set_retry.call_args_list) == retry_calls
    raw.pipeline.return_value.__exit__.assert_called_once()


async def test_cancel_with_invalid_unhashable_entry_reports_original_preflight_error(tmp_path, monkeypatch, target):
    from nexus.gateway.trajectory_store_backends.redis_store.checkpoint import load as load_module

    dumped = await dump_records(tmp_path, [NodePayloadRecord('t', 1, np.array([1], dtype=np.int32))])
    invalid = replace(dumped[0], entry_id=[])
    store, raw = target()
    entered, release = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    reported, previous = [], loop.get_exception_handler()
    loop.set_exception_handler(lambda loop, context: reported.append(context))
    original = load_module._validate_checkpoint

    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(load_module, '_validate_checkpoint', blocked)
    try:
        task = asyncio.create_task(store.load_payloads([invalid], dump_dir=str(tmp_path)))
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert raw.sent_commands == []
        assert reported
        assert isinstance(reported[0]['exception'], ValueError)
    finally:
        release.set()
        loop.set_exception_handler(previous)


@pytest.mark.parametrize('state', ['writing', 'committed'])
async def test_cancel_hands_protected_import_to_last_dump_owner(tmp_path, target, state):
    from nexus.gateway.trajectory_store_backends.redis_store.client import _RECORD_CAS
    from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import protect_dump_records
    from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import StoredNode

    dumped = await dump_records(
        tmp_path, [NodePayloadRecord('t', 1, np.array([1], dtype=np.int32), routed_experts=[3])]
    )
    fs = ImportFS()
    fs.finalize_ssd_imports = Mock(wraps=fs.finalize_ssd_imports)
    store, raw = target(fs=fs)
    paused, resume, deleted = threading.Event(), threading.Event(), threading.Event()
    record_id = None

    def pause_write(args, result):
        nonlocal record_id
        if (
            state == 'writing'
            and args[0] == 'SET'
            or (state == 'committed' and args[0] == 'EVAL' and args[1] == _RECORD_CAS)
        ):
            key = args[1] if state == 'writing' else args[3]
            record_id = key.decode().rsplit(':', 1)[1]
            paused.set()
            assert resume.wait(5)
        elif args[0] == 'EVAL' and 'nexus:load-delete' in args[1]:
            deleted.set()

    raw.after_command = pause_write
    task = asyncio.create_task(store.load_payloads(dumped, dump_dir=str(tmp_path)))
    try:
        assert await asyncio.to_thread(paused.wait, 3)
        key = store._client._node_key(record_id)
        blob = raw.get(key)
        urls = StoredNode.deserialize(blob).blob_urls
        assert StoredNode.deserialize(blob).state == state
        with protect_dump_records(store._client, [record_id], 'concurrent-dump'):
            task.cancel()
            await asyncio.sleep(0)
            resume.set()
            assert await asyncio.to_thread(deleted.wait, 3)
            await asyncio.wait({task}, timeout=3)
            assert task.done(), 'logical deletion must not wait for the concurrent dump'
            with pytest.raises(asyncio.CancelledError):
                await task
            assert raw.get(key) is None
            assert list(raw._kv.values()) == [blob]
            assert bool(fs.blobs) == (state == 'committed')
            fs.finalize_ssd_imports.assert_not_called()
        assert raw._kv == {}
        assert fs.blobs == {}
        if state == 'writing':
            fs.finalize_ssd_imports.assert_called_once_with(list(urls.values()), [])
            url = next(iter(urls.values()))
            with pytest.raises(RedisClientError, match='aborted'):
                fs.import_file_to_ssd(url, str(next((tmp_path / 'data').iterdir())))
        else:
            fs.finalize_ssd_imports.assert_not_called()
        deletes = [call for call in raw.sent_commands if call[0] == 'EVAL' and 'nexus:load-delete' in call[1]]
        assert len(deletes) == 1
    finally:
        resume.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('state', ['unknown', 'failed'])
def test_deferred_writing_cleanup_requires_confirmed_import_fence(tmp_path, target, state):
    from hashlib import sha1

    from nexus.gateway.trajectory_store_backends.redis import NodeRecord
    from nexus.gateway.trajectory_store_backends.redis_store.checkpoint.load import _DELETE_IF_OWNED
    from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import (
        deferred_record_key,
        dump_owners_key,
        protect_dump_records,
    )

    fs = ImportFS()
    store, raw = target(fs=fs)
    client = store._client
    source = tmp_path / 'source.blob'
    source.write_bytes(b'imported bytes')
    url = fs.allocate_url()
    fs.import_file_to_ssd(url, str(source))
    record_id = 'reserved-load-record'
    key = client._node_key(record_id)
    pending, _ = client._publication._serialize_import_metadata(
        NodeRecord('t', 1, np.array([1], dtype=np.int32)), {'routed_experts': url}
    )
    raw.set(key, pending)
    fs.finalize_ssd_imports = Mock(
        wraps=fs.finalize_ssd_imports,
        side_effect=[{'items': {url: {'state': state, 'error': 'abort not confirmed'}}}, DEFAULT],
    )
    fs.delete = Mock(wraps=fs.delete)
    with protect_dump_records(client, [record_id], 'dump') as lease:
        assert (
            client._execute_without_retry(
                'EVAL',
                _DELETE_IF_OWNED,
                3,
                key,
                dump_owners_key(key),
                deferred_record_key(key),
                sha1(pending).hexdigest(),
            )
            == 2
        )
        with pytest.raises(RedisClientError, match='deferred import cleanup not confirmed') as caught:
            lease.release_records([record_id])
        assert caught.value.code == 'cleanup_failed'
        assert not lease._pending_release_keys
        assert raw.get(deferred_record_key(key)) == pending
        assert fs.blobs == {url: b'imported bytes'}
        fs.delete.assert_not_called()
        assert not any(call[0] == 'UNLINK' for call in raw.sent_commands)
    assert fs.finalize_ssd_imports.call_count == 1
    client.delete_multi_node([record_id], missing_ok=True)
    assert fs.finalize_ssd_imports.call_count == 2
    fs.delete.assert_called_once_with(url)
    assert raw.get(deferred_record_key(key)) is None
    assert fs.blobs == {}
    assert source.read_bytes() == b'imported bytes'
    with pytest.raises(RedisClientError, match='aborted'):
        fs.import_file_to_ssd(url, str(source))
