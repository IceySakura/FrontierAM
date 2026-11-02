"""Heartbeat reports stay independent of slow controller status requests."""

import json

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage import controller as controller_module
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.controller import ControllerService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


STATS = {
    'objects': 2,
    'used_bytes': 8,
    'resident_bytes': 4,
    'resident_objects': 1,
    'ssd_bytes': 4,
    'ssd_objects': 1,
    'cached_bytes': 0,
    'cached_objects': 0,
}


@pytest.fixture
def services():
    redis = MagicMock()
    redis.set.return_value = True
    registration = DataUnitRegistration('unit-0', Endpoint('127.0.0.1', 17000))
    topology = Topology.create([registration], 'heartbeat-stats')
    redis.get.return_value = json.dumps(topology.dump())
    controller = ControllerService(redis, topology).start()
    try:
        yield SimpleNamespace(redis=redis, topology=topology, controller=controller, registration=registration)
    finally:
        controller.stop()


def _heartbeat(registration, *, stats=STATS):
    return {
        'data_unit_id': registration.data_unit_id,
        'endpoint': {'host': registration.endpoint.host, 'port': registration.endpoint.port},
        'stats': stats,
    }


def test_heartbeat_records_stats_and_receipt_time_without_reading_redis_topology(services, monkeypatch):
    monkeypatch.setattr(controller_module.time, 'time', lambda: 100.0)
    assert services.controller._handle('heartbeat', _heartbeat(services.registration), b'') == ({}, b'')

    key = f'{services.controller._prefix}:heartbeat:unit-0'
    assert services.redis.set.call_args.args[0] == key
    assert json.loads(services.redis.set.call_args.args[1]) == {'received_at': 100.0, 'stats': STATS}
    assert services.redis.set.call_args.kwargs == {'ex': 15}
    services.redis.get.assert_not_called()


@pytest.mark.parametrize(
    'registration',
    [
        DataUnitRegistration('unknown', Endpoint('127.0.0.1', 17000)),
        DataUnitRegistration('unit-0', Endpoint('127.0.0.1', 17001)),
    ],
)
def test_heartbeat_rejects_unregistered_identity_without_storing_report(services, registration):
    services.redis.set.reset_mock()
    with pytest.raises(RedisClientError) as error:
        services.controller._handle('heartbeat', _heartbeat(registration), b'')
    assert error.value.code == 'invalid_registration'
    services.redis.set.assert_not_called()


def test_status_reports_complete_latest_stats_and_missing_reports(services):
    pipe = services.redis.pipeline.return_value.__enter__.return_value
    updated = {**STATS, 'objects': 3, 'used_bytes': 12, 'resident_bytes': 8, 'resident_objects': 2}
    for stats in (STATS, updated):
        services.controller._handle('heartbeat', _heartbeat(services.registration, stats=stats), b'')
        pipe.execute.return_value = [services.redis.set.call_args.args[1]]
        assert services.redis.set.call_args.kwargs == {'ex': 15}
        assert services.controller._handle('status', {}, b'')[0]['units'][0] == {
            'data_unit_id': 'unit-0',
            'alive': True,
            'stats': stats,
        }
    pipe.execute.return_value = [None]
    assert services.controller._handle('status', {}, b'') == (
        {'units': [{'data_unit_id': 'unit-0', 'alive': False, 'stats': None}]},
        b'',
    )
    key = f'{services.controller._prefix}:heartbeat:unit-0'
    assert pipe.get.call_args_list == [call(key)] * 3
    assert all(call.kwargs == {'transaction': False} for call in services.redis.pipeline.call_args_list)
    assert services.controller.topology() == services.topology


def test_heartbeat_returns_while_another_client_status_pipeline_is_blocked(services):
    status_client = DealerClient(services.controller.endpoint, timeout=5)
    heartbeat_client = DealerClient(services.controller.endpoint, timeout=5)
    entered, release = Event(), Event()

    def execute():
        entered.set()
        assert release.wait(timeout=5), 'test did not release the status request'
        return [services.redis.set.call_args.args[1]]

    services.redis.pipeline.return_value.__enter__.return_value.execute.side_effect = execute
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            status = pool.submit(status_client.request, 'status')
            try:
                assert entered.wait(timeout=5)
                heartbeat = pool.submit(heartbeat_client.request, 'heartbeat', _heartbeat(services.registration))
                assert heartbeat.result(timeout=1) == ({}, b'')
                assert not status.done()
                report = json.loads(services.redis.set.call_args.args[1])
                assert report['stats'] == STATS
            finally:
                release.set()
            assert status.result(timeout=5)[0]['units'][0]['stats'] == STATS
    finally:
        release.set()
        status_client.close()
        heartbeat_client.close()
