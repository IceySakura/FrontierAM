"""Dump deletion protection works across clients, processes and Cluster slots."""

import multiprocessing

from uuid import uuid4

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.dump_protection import (
    deferred_record_key,
    dump_owners_key,
    protect_dump_records,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import CompareAndSetResult
from tests.unit.gateway.redis_store.payloads import SamplePayload


def _delete_in_process(config, prefix, node_id, entered, finished):
    with RedisClient(**config, key_prefix=prefix) as client:
        entered.set()
        client.delete_node(node_id)
        finished.set()


def test_independent_delete_returns_before_dump_releases_protected_metadata(backend_config):
    context = multiprocessing.get_context('spawn')
    entered, finished = context.Event(), context.Event()
    prefix = 'dump-protection-' + uuid4().hex
    process = context.Process(target=_delete_in_process, args=(backend_config, prefix, 'a', entered, finished))
    try:
        with RedisClient(**backend_config, key_prefix=prefix) as client:
            client.put_node(SamplePayload(text='old'), node_id='a')
            record_key = client._node_key('a')
            key = dump_owners_key(record_key)
            deferred = deferred_record_key(record_key)
            with protect_dump_records(client, ['a'], 'first') as first, protect_dump_records(client, ['a'], 'second'):
                assert client._r.smembers(key) == {b'first', b'second'}
                client.put_node(SamplePayload(text='unrelated'), node_id='b')
                client.put_multi_node([SamplePayload(text='multi')], node_ids=['c'])
                assert client.get_node('c', payload_type=SamplePayload).text == 'multi'
                [original] = client.get_node_blobs(['a'])
                assert client.compare_and_set_many([('a', original, SamplePayload(text='new').serialize())]) == [
                    CompareAndSetResult.UPDATED
                ]
                assert client.get_node('a', payload_type=SamplePayload).text == 'new'
                process.start()
                assert entered.wait(10)
                assert finished.wait(10)
                assert client.get_node_blobs(['a']) == [None]
                assert not client.has_node('a')[0]
                assert client._r.get(deferred) == SamplePayload(text='new').serialize()
                first.release_records(['a'])
                assert client._r.smembers(key) == {b'second'}
                assert client._r.exists(deferred)
            assert finished.wait(10)
            process.join(timeout=10)
            assert process.exitcode == 0
            assert not client.has_node('a')[0]
            assert not client._r.exists(key)
            assert not client._r.exists(deferred)
            assert client.get_node('b', payload_type=SamplePayload).text == 'unrelated'
    finally:
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)


@pytest.mark.parametrize('namespace', ['ordinary', 'namespace-{tenant}', 'namespace-{}-{ignored}'])
def test_protection_and_record_share_cluster_slot(backend_config, namespace):
    from redis.crc import key_slot

    prefix = namespace + '-' + uuid4().hex
    with RedisClient(**backend_config, key_prefix=prefix) as client:
        client.put_multi_node([SamplePayload(text='a'), SamplePayload(text='b')], node_ids=['a', 'b'])
        keys = [dump_owners_key(client._node_key(node_id)) for node_id in ('a', 'b')]
        with protect_dump_records(client, ['b', 'a'], 'export'):
            for node_id, key in zip(('a', 'b'), keys, strict=True):
                assert (
                    key_slot(client._node_key(node_id))
                    == key_slot(key)
                    == key_slot(deferred_record_key(client._node_key(node_id)))
                )
                assert client._r.smembers(key) == {b'export'}
            client.delete_multi_node(['a', 'b'])
            assert all(client._r.exists(deferred_record_key(client._node_key(node_id))) for node_id in ('a', 'b'))
        assert not any(client._r.exists(key) for key in keys)
        assert not any(client._r.exists(deferred_record_key(client._node_key(node_id))) for node_id in ('a', 'b'))


def _hold_protection_in_process(config, prefix, ready, release):
    with RedisClient(**config, key_prefix=prefix) as client:
        with protect_dump_records(client, ['a'], 'child'):
            ready.set()
            assert release.wait(15)


def test_last_owner_in_another_process_reclaims_deferred_metadata(backend_config):
    context = multiprocessing.get_context('spawn')
    ready, release = context.Event(), context.Event()
    prefix = 'dump-owner-' + uuid4().hex
    process = context.Process(target=_hold_protection_in_process, args=(backend_config, prefix, ready, release))
    try:
        with RedisClient(**backend_config, key_prefix=prefix) as client:
            client.put_node(SamplePayload(text='old'), node_id='a')
            deferred = deferred_record_key(client._node_key('a'))
            with protect_dump_records(client, ['a'], 'parent') as lease:
                process.start()
                assert ready.wait(10)
                client.delete_multi_node(['a'])
                lease.release_records(['a'])
                assert client._r.exists(deferred)
                assert not client.has_node('a')[0]
                release.set()
                process.join(timeout=10)
                assert process.exitcode == 0
                assert not client._r.exists(deferred)
    finally:
        release.set()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
