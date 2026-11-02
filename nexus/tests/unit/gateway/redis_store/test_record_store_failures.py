import asyncio
import threading

from unittest.mock import Mock
from uuid import uuid4

import pytest

from nexus.gateway.trajectory_store_backends.redis import NodeRecord, RedisTrajectoryRecordStore
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.publication import PublicationResult
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


class ScriptedClient:
    """Inject partial writes and failures without depending on network timing."""

    def __init__(self):
        self._publication = self
        self.payloads = {}
        self.fail_create = False
        self.fail_delete = False
        self.fail_after_wait = False
        self.blocked_node_id = None
        self.started = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.events = []

    def close(self):
        self.events.append(('close', threading.get_ident()))

    def put_multi_node(self, payloads, *, nodes=None):
        if self.blocked_node_id is None or payloads[0].node_id == self.blocked_node_id:
            self.started.set()
            assert self.proceed.wait(5)
        self.events.append(('write', threading.get_ident()))
        if self.fail_after_wait:
            raise RuntimeError('injected in-flight failure')
        out = []
        for index, payload in enumerate(payloads):
            if self.fail_create and index == 1:
                raise RedisClientError('injected create failure', code='unavailable')
            node_id = uuid4().hex
            self.payloads[node_id] = payload
            out.append(node_id)
        return out

    def put(self, payloads):
        return [PublicationResult(node_id) for node_id in self.put_multi_node(payloads)]

    def delete(self, refs, *, reference_urls=None):
        self.delete_multi_node(refs, missing_ok=True)

    def delete_multi_node(self, refs, *, missing_ok):
        for index, ref in enumerate(refs):
            if self.fail_delete and index == 1:
                raise RedisClientError('injected delete failure', code='unavailable')
            self.payloads.pop(ref, None)


@pytest.fixture
def client(monkeypatch):
    client = ScriptedClient()
    client.url = 'redis://192.0.2.10:16379/0'

    def connect(**config):
        assert config == {}
        client.events.append(('create', threading.get_ident()))
        return client

    monkeypatch.setattr('nexus.gateway.trajectory_store_backends.redis.RedisClient', connect)
    return client


def test_partial_create_and_delete_are_retryable(client):
    store = RedisTrajectoryRecordStore('run')
    records = [NodeRecord('t', i, [i]) for i in range(3)]

    async def exercise():
        client.fail_create = True
        with pytest.raises(RedisClientError, match='create failure'):
            await store.put_many(records)
        assert len(client.payloads) == 1
        saved_id = next(iter(client.payloads))
        client.fail_create = False
        refs = await store.put_many(records)
        assert len(client.payloads) == 4
        assert all(ref.record_id != saved_id for ref in refs)
        client.fail_delete = True
        with pytest.raises(RedisClientError, match='delete failure'):
            await store.release_many(refs)
        assert len(client.payloads) == 3
        client.fail_delete = False
        await store.release_many(refs)
        assert list(client.payloads) == [saved_id]

    with store:
        asyncio.run(exercise())


def test_publication_reuses_client_and_one_batch_call(client):
    store = RedisTrajectoryRecordStore('run')
    assert not isinstance(store, RedisClient)
    assert store._client is client
    assert [event for event, _ in client.events] == ['create']
    record = NodeRecord('t', 1, [1])
    with store:
        refs = asyncio.run(store.put_many([record, record]))
        refs += asyncio.run(store.put_many([record, record]))
        assert len(set(refs)) == 4
        asyncio.run(store.release_many(refs))
        assert [event for event, _ in client.events] == ['create', 'write', 'write']
    assert [event for event, _ in client.events] == ['create', 'write', 'write', 'close']


def test_write_error_propagates_and_context_closes_client(client, monkeypatch):
    error = ConnectionError('injected failure')
    monkeypatch.setattr(client, 'put_multi_node', Mock(side_effect=error))
    with pytest.raises(ConnectionError) as caught, RedisTrajectoryRecordStore('run') as store:
        asyncio.run(store.put_many([NodeRecord('t', 1, [1])]))
    assert caught.value is error
    assert [event for event, _ in client.events] == ['create', 'close']


@pytest.mark.parametrize('fail', [False, True])
def test_cancel_drains_io_and_keeps_client_usable(client, fail):
    client.proceed.clear()
    client.fail_after_wait = fail
    store = RedisTrajectoryRecordStore('run')

    async def exercise():
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        task = asyncio.create_task(store.put_many([NodeRecord('t', 1, [1])]))
        assert await asyncio.to_thread(client.started.wait, 3)
        for index in range(2):
            task.cancel(f'cancel-{index}')
            await asyncio.sleep(0)
            assert not task.done()
        client.proceed.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        assert cancelled.value.args == ('cancel-0',)
        if fail:
            assert len(errors) == 1
            assert isinstance(errors[0]['exception'], RuntimeError)
        else:
            assert errors == []
        assert [event for event, _ in client.events] == ['create', 'write']
        assert client.events[1][1] != threading.get_ident()
        assert len(client.payloads) == (0 if fail else 1)
        client.fail_after_wait = False
        refs = await store.put_many([NodeRecord('t', 2, [2])])
        await store.release_many(refs)
        assert [event for event, _ in client.events] == ['create', 'write', 'write']

    with store:
        asyncio.run(exercise())
    assert [event for event, _ in client.events] == ['create', 'write', 'write', 'close']


def test_cancelling_one_operation_does_not_block_another(client):
    client.proceed.clear()
    client.blocked_node_id = 1

    async def exercise(store):
        pending = asyncio.create_task(store.put_many([NodeRecord('t', 1, [1])]))
        assert await asyncio.to_thread(client.started.wait, 3)
        pending.cancel()
        try:
            refs = await asyncio.wait_for(store.put_many([NodeRecord('t', 2, [2])]), 3)
            await store.release_many(refs)
            assert not pending.done()
            assert [event for event, _ in client.events] == ['create', 'write']
        finally:
            client.proceed.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert [event for event, _ in client.events] == ['create', 'write', 'write']

    with RedisTrajectoryRecordStore('run') as store:
        asyncio.run(exercise(store))
    assert [event for event, _ in client.events] == ['create', 'write', 'write', 'close']


@pytest.mark.parametrize('code,expected', [('not_found', KeyError), ('unavailable', RedisClientError)])
def test_async_operations_translate_only_missing_records(client, monkeypatch, code, expected):
    error = RedisClientError('injected read failure', code=code)

    def fail(refs):
        raise error

    with RedisTrajectoryRecordStore('run') as store:
        monkeypatch.setattr(store._client, 'get_multi_node', fail, raising=False)
        with pytest.raises(expected, match='injected read failure') as caught:
            asyncio.run(store._run(store._client.get_multi_node, []))
        if code == 'not_found':
            assert caught.value.__cause__ is error
        else:
            assert caught.value is error


@pytest.mark.parametrize('fail', [False, True])
async def test_fs_cancellation_drains_upload_and_keeps_client_usable(monkeypatch, fail):
    pytest.importorskip('zmq')
    from tests.unit.gateway.redis_store.fake_redis import FakeRedis
    from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

    started, release = threading.Event(), threading.Event()
    fs = MemoryFS()
    original_put = fs.put

    def blocked_put(url, blob):
        started.set()
        assert release.wait(5)
        if fail:
            raise RuntimeError('upload failed while draining')
        original_put(url, blob)

    monkeypatch.setattr(fs, 'put', blocked_put)
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    errors = []
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    try:
        with RedisTrajectoryRecordStore('run', {'redis_client': FakeRedis(), 'fs_client': fs}) as store:
            task = asyncio.create_task(store.put_many([NodeRecord('run', 1, [1], routed_experts=b'blob')]))
            try:
                assert await asyncio.to_thread(started.wait, 3)
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert store._client._publication._active_operations == 0
            assert len(errors) == int(fail)
            if fail:
                assert errors[0]['exception'].code == 'publication_failed'
            monkeypatch.setattr(fs, 'put', original_put)
            refs = await store.put_many([NodeRecord('run', 2, [2], routed_experts=b'next')])
            assert (await store.get_many(refs))[0]['routed_experts'] == b'next'
            await store.release_many(refs)
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)
    assert not fs.closed
