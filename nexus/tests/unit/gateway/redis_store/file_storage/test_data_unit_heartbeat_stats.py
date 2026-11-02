"""Heartbeats read consistent metadata while data requests perform SSD I/O."""

import copy
import time

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Event

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, RouterServer
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


EMPTY_STATS = {
    'objects': 0,
    'used_bytes': 0,
    'resident_bytes': 0,
    'resident_objects': 0,
    'ssd_bytes': 0,
    'ssd_objects': 0,
    'cached_bytes': 0,
    'cached_objects': 0,
}


@pytest.fixture
def service(tmp_path):
    heartbeats = Queue()

    def receive(operation, body, blob):
        assert operation == 'heartbeat'
        assert not blob
        heartbeats.put((time.monotonic(), copy.deepcopy(body)))
        return {}, b''

    controller = RouterServer(Endpoint('127.0.0.1', 0), receive).start()
    unit = DataUnitService('unit', memory_waterline_bytes=12, ssd_offload_dir=tmp_path).start()
    topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint)])
    client = DealerClient(unit.endpoint, timeout=5)
    try:
        unit.configure(topology, controller.endpoint, heartbeat_interval=0.02)
        yield unit, client, heartbeats
    finally:
        client.close()
        unit.stop()
        controller.stop()


def wait_for_heartbeat(heartbeats, *, after=0, stats=None):
    deadline = time.monotonic() + 5
    while True:
        remaining = deadline - time.monotonic()
        assert remaining > 0, 'matching heartbeat was not sent'
        received_at, body = heartbeats.get(timeout=remaining)
        if received_at >= after and (stats is None or body['stats'] == stats):
            return body


def test_initial_and_periodic_heartbeats_include_complete_stats(service):
    unit, client, heartbeats = service
    initial = wait_for_heartbeat(heartbeats)
    assert initial['data_unit_id'] == unit.data_unit_id
    assert initial['endpoint'] == vars(unit.endpoint)
    assert initial['stats'] == EMPTY_STATS

    url = unit._topology.new_url()
    client.request('put', {'url': url}, b'abc')
    expected = {**EMPTY_STATS, 'objects': 1, 'used_bytes': 3, 'resident_bytes': 3, 'resident_objects': 1}
    wait_for_heartbeat(heartbeats, stats=expected)
    assert client.request('stats')[0] == expected


def test_stats_dicts_do_not_modify_published_snapshot_and_stop_publishes_empty_stats(service):
    unit, client, heartbeats = service
    unit._snapshot_stats()['objects'] = 999
    assert unit._snapshot_stats() == EMPTY_STATS
    assert client.request('stats')[0] == EMPTY_STATS
    assert wait_for_heartbeat(heartbeats)['stats'] == EMPTY_STATS

    client.request('put', {'url': unit._topology.new_url()}, b'abc')
    unit.stop()
    assert unit._snapshot_stats() == EMPTY_STATS


@pytest.mark.parametrize('fail_second_eviction', [False, True])
def test_heartbeat_reports_committed_metadata_during_partial_eviction(service, monkeypatch, fail_second_eviction):
    unit, client, heartbeats = service
    for payload in (b'aaa', b'bbb', b'ccc'):
        client.request('put', {'url': unit._topology.new_url()}, payload)
    expected_before = {**EMPTY_STATS, 'objects': 3, 'used_bytes': 9, 'resident_bytes': 9, 'resident_objects': 3}
    wait_for_heartbeat(heartbeats, stats=expected_before)
    entered, release = Event(), Event()
    original_write = unit._write_blob

    def write(url, blob):
        if blob == b'bbb':
            entered.set()
            assert release.wait(timeout=10), 'test did not release the SSD writer'
            if fail_second_eviction:
                raise OSError('injected second eviction failure')
        return original_write(url, blob)

    monkeypatch.setattr(unit, '_write_blob', write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        putting = pool.submit(client.request, 'put', {'url': unit._topology.new_url()}, b'ddd')
        try:
            assert entered.wait(timeout=5)
            blocked_at = time.monotonic()
            expected_during = {
                **expected_before,
                'resident_bytes': 6,
                'resident_objects': 2,
                'ssd_bytes': 3,
                'ssd_objects': 1,
            }
            # Parallel writers can enter the second file before the first
            # victim is committed. Observe that commit explicitly.
            wait_for_heartbeat(heartbeats, after=blocked_at, stats=expected_during)
            blocked_at = time.monotonic()
            for _ in range(2):
                during = wait_for_heartbeat(heartbeats, after=blocked_at)
                assert during['stats'] == expected_during
        finally:
            release.set()
        if fail_second_eviction:
            with pytest.raises(RedisClientError, match='injected second eviction failure'):
                putting.result(timeout=5)
            expected_after = expected_during
        else:
            putting.result(timeout=5)
            expected_after = {
                **expected_before,
                'objects': 4,
                'used_bytes': 12,
                'resident_bytes': 6,
                'resident_objects': 2,
                'ssd_bytes': 6,
                'ssd_objects': 2,
            }
    wait_for_heartbeat(heartbeats, stats=expected_after)
    assert client.request('stats')[0] == expected_after
