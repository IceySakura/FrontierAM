"""Direct file ingestion uses paths and preserves independent DU control outcomes."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


@pytest.fixture
def transport():
    topology = Topology.create([
        DataUnitRegistration(f'unit-{index}', Endpoint('127.0.0.1', 17000 + index)) for index in range(2)
    ])

    def request(operation, body):
        return {'items': {url: {'state': 'imported', 'error': None} for url in body['urls']}}, b''

    connections = {unit.data_unit_id: Mock() for unit in topology.data_units}
    for connection in connections.values():
        connection.request.side_effect = request
    client = object.__new__(FSClient)
    client.topology = topology
    client._unit = Mock(side_effect=lambda url: connections[topology.owner(url).data_unit_id])
    urls = {}
    while len(urls) < 2:
        url = client.allocate_url()
        urls[topology.owner(url).data_unit_id] = url
    return SimpleNamespace(client=client, connections=connections, urls=urls)


def test_import_file_to_ssd_transmits_only_source_path_and_waits_for_final_ack(transport, tmp_path):
    url = transport.urls['unit-0']
    path = str(tmp_path / 'shared.blob')
    entered = Event()
    copied = Future()

    def copy(operation, body):
        entered.set()
        return copied.result()

    connection = transport.connections['unit-0']
    connection.request.side_effect = copy
    with ThreadPoolExecutor(max_workers=1) as executor:
        writing = executor.submit(transport.client.import_file_to_ssd, url, path)
        try:
            assert entered.wait(timeout=2), 'path ingestion never reached its DU'
            connection.request.assert_called_once_with('import_file_to_ssd', {'url': url, 'source_path': path})
            assert not writing.done(), 'submission is not a completed import'
        finally:
            copied.set_result(({}, b''))
        assert writing.result(timeout=2) is None


def test_import_file_to_ssd_does_not_replay_a_timed_out_import(transport, tmp_path):
    url = transport.urls['unit-0']
    path = str(tmp_path / 'shared.blob')
    connection = transport.connections['unit-0']
    connection.request.side_effect = RedisClientError('import timed out', code='timeout')
    with pytest.raises(RedisClientError) as error:
        transport.client.import_file_to_ssd(url, path)
    assert error.value.code == 'timeout'
    connection.request.assert_called_once_with('import_file_to_ssd', {'url': url, 'source_path': path})


@pytest.mark.parametrize('operation', ['get_ssd_import_status', 'finalize_ssd_imports'])
def test_import_control_groups_all_urls_before_rpc_and_preserves_items(transport, operation):
    urls = list(transport.urls.values())
    method = getattr(transport.client, operation)
    result = method(urls + [urls[0]], [urls[0]]) if operation == 'finalize_ssd_imports' else method(urls + [urls[0]])
    assert result == {'items': dict.fromkeys(urls, {'state': 'imported', 'error': None})}
    for unit_id, connection in transport.connections.items():
        body = {'urls': [transport.urls[unit_id]]}
        if operation == 'finalize_ssd_imports':
            body['keep_urls'] = [urls[0]] if transport.urls[unit_id] == urls[0] else []
        connection.request.assert_called_once_with(operation, body)


@pytest.mark.parametrize('operation', ['get_ssd_import_status', 'finalize_ssd_imports'])
def test_failed_du_control_keeps_other_ack_and_drains_it_before_return(transport, operation):
    urls = list(transport.urls.values())
    entered = Event()
    delayed = Future()

    def unknown(operation, body):
        raise RedisClientError('outcome unknown', code='timeout')

    def acknowledged(operation, body):
        entered.set()
        return delayed.result()

    transport.connections['unit-0'].request.side_effect = unknown
    transport.connections['unit-1'].request.side_effect = acknowledged
    method = getattr(transport.client, operation)
    arguments = (urls, []) if operation == 'finalize_ssd_imports' else (urls,)
    with ThreadPoolExecutor(max_workers=1) as executor:
        controlling = executor.submit(method, *arguments)
        try:
            assert entered.wait(timeout=2)
            assert not controlling.done()
        finally:
            delayed.set_result(({'items': {transport.urls['unit-1']: {'state': 'finished', 'error': None}}}, b''))
        result = controlling.result(timeout=2)
    assert result['items'][transport.urls['unit-0']] == {'state': 'unknown', 'error': 'outcome unknown'}
    assert result['items'][transport.urls['unit-1']] == {'state': 'finished', 'error': None}
    assert all(connection.request.call_count == 1 for connection in transport.connections.values())


def test_import_control_empty_and_invalid_urls_send_no_rpc(transport):
    assert transport.client.get_ssd_import_status([]) == {'items': {}}
    assert transport.client.finalize_ssd_imports([], []) == {'items': {}}
    with pytest.raises(RedisClientError, match='invalid URL'):
        transport.client.get_ssd_import_status([transport.urls['unit-0'], 'not-a-fs-url'])
    assert all(not connection.request.called for connection in transport.connections.values())


@pytest.mark.parametrize('operation', ['get_ssd_import_status', 'finalize_ssd_imports'])
def test_connection_setup_failure_preserves_other_du_control_ack(transport, monkeypatch, operation):
    original = transport.client._unit
    urls = [transport.urls['unit-0'], transport.urls['unit-1']]

    def connection(url):
        if url == urls[0]:
            raise RedisClientError('connection closed', code='closed')
        return original(url)

    monkeypatch.setattr(transport.client, '_unit', connection)
    method = getattr(transport.client, operation)
    result = method(urls, []) if operation == 'finalize_ssd_imports' else method(urls)
    assert result['items'][urls[0]] == {'state': 'unknown', 'error': 'connection closed'}
    assert result['items'][urls[1]] == {'state': 'imported', 'error': None}
