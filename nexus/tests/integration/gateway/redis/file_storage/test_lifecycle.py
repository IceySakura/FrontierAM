"""Submission failures and close must wait for admitted network work."""

import asyncio
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest

from nexus.gateway.trajectory_store_backends.redis import NodeRecord, RedisTrajectoryRecordStore, SegmentRecord
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.mark.parametrize('operation', ['put', 'get', 'delete'])
def test_submit_failure_drains_pending_work(client, monkeypatch, operation):
    payload = NodeRecord('run', 1, [1], routed_experts=b'data', multi_modal_items={})
    result = client._publication.put([payload])[0] if operation != 'put' else None
    release, failed = threading.Event(), threading.Event()
    original, count = client._publication._io.submit, 0
    deleted_urls = []
    if operation == 'delete':
        original_delete = client._publication._fs.delete

        def delete(url):
            deleted = original_delete(url)
            deleted_urls.append(url)
            return deleted

        monkeypatch.setattr(client._publication._fs, 'delete', delete)

    def submit(function, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            failed.set()
            raise RuntimeError('submit failed')

        def blocked():
            assert release.wait(5)
            return function(*args, **kwargs)

        return original(blocked)

    monkeypatch.setattr(client._publication._io, 'submit', submit)
    call = {
        'put': lambda: client.put_node(payload),
        'get': lambda: client._publication.get([(result.node_id, NodeRecord)], reference_urls=[result.blob_urls]),
        'delete': lambda: client.delete_multi_node([result.node_id]),
    }[operation]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(call)
        try:
            assert failed.wait(5)
            with pytest.raises(FutureTimeoutError):
                pending.result(timeout=0.05)
        finally:
            release.set()
        with pytest.raises(RedisClientError if operation == 'put' else RuntimeError) as caught:
            pending.result(timeout=5)
    assert 'submit failed' in str(caught.value)
    if operation == 'delete':
        assert not client.has_node(result.node_id)[0]
        assert len(deleted_urls) == 1
        remaining = set(result.blob_urls.values()) - set(deleted_urls)
        assert len(remaining) == 1
        assert client._publication._fs.get(remaining.pop())


def test_close_waits_for_submission_phase(client, monkeypatch):
    started, release, shutdown_called = threading.Event(), threading.Event(), threading.Event()
    publisher = client._publication
    original_submit, original_shutdown = publisher._io.submit, publisher._io.shutdown

    def submit(*args, **kwargs):
        future = original_submit(*args, **kwargs)
        started.set()
        assert release.wait(5)
        return future

    def shutdown(*args, **kwargs):
        shutdown_called.set()
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(publisher._io, 'submit', submit)
    monkeypatch.setattr(publisher._io, 'shutdown', shutdown)
    with ThreadPoolExecutor(max_workers=2) as pool:
        write = pool.submit(client.put_node, NodeRecord('run', 1, [1], routed_experts=b'data'))
        assert started.wait(5)
        closing = pool.submit(client.close)
        try:
            deadline = time.monotonic() + 2
            while not publisher._closing and time.monotonic() < deadline:
                time.sleep(0.01)
            assert publisher._closing
            assert not shutdown_called.is_set()
            assert not closing.done()
            with pytest.raises(RedisClientError) as caught:
                client.put_node(NodeRecord('run', 2, [2]))
            assert caught.value.code == 'closed'
        finally:
            release.set()
        assert isinstance(write.result(timeout=5), str)
        closing.result(timeout=5)
        assert shutdown_called.is_set()


def test_close_waits_for_segment_append_cas(client, redis_backend_config, monkeypatch):
    config = {**redis_backend_config, 'fs_client': client._publication._fs, 'key_prefix': client.key_prefix}
    with RedisTrajectoryRecordStore('run', config) as store:
        segment = asyncio.run(store.put_many([SegmentRecord('run', 's', {'reward': 1})]))[0]
        started, release = threading.Event(), threading.Event()
        original = store._client.compare_and_set_many

        def blocked_cas(updates):
            started.set()
            assert release.wait(5)
            return original(updates)

        monkeypatch.setattr(store._client, 'compare_and_set_many', blocked_cas)
        with ThreadPoolExecutor(max_workers=2) as pool:
            append = pool.submit(store._put_fields_many, {segment: {'advantage': 2}})
            assert started.wait(5)
            closing = pool.submit(store.close)
            try:
                with pytest.raises(FutureTimeoutError):
                    closing.result(timeout=0.1)
            finally:
                release.set()
            append.result(timeout=5)
            closing.result(timeout=5)
