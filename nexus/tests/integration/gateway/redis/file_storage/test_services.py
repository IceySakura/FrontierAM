"""Essential protocol, routing and real TCP service checks."""

import time

from concurrent.futures import ThreadPoolExecutor

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import (
    DealerClient,
    Message,
    MsgType,
    RouterServer,
)
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import Endpoint
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


def test_blob_lifecycle_and_ownership(services):
    fs = services.fs
    for value in (b'', b'\x00\xff' * 32768):
        url = fs.allocate_url()
        fs.put(url, value)
        assert fs.get(url) == value
        with pytest.raises(RedisClientError, match='immutable'):
            fs.put(url, b'new')
        wrong = next(unit for unit in services.units if unit.data_unit_id != fs.topology.owner(url).data_unit_id)
        connection = DealerClient(wrong.endpoint)
        try:
            with pytest.raises(RedisClientError) as error:
                connection.request(MsgType.GET, {'url': url})
            assert error.value.code == 'wrong_owner'
        finally:
            connection.close()
        assert fs.delete(url)
        assert not fs.delete(url)
        with pytest.raises(RedisClientError) as error:
            fs.get(url)
        assert error.value.code == 'not_found'
    assert all(connection.request('stats')[0]['used_bytes'] == 0 for connection in fs._units.values())


def test_discovery_heartbeat_and_no_replay(services, monkeypatch):
    fs = services.fs
    assert fs._units == {}
    assert len(fs.check_ready()) == 2
    assert fs._units == {}
    assert all(unit['alive'] for unit in fs.status()['units'])
    assert services.redis.get(services.controller._topology_key)
    for unit in services.units:
        key = f'{services.controller._prefix}:heartbeat:{unit.data_unit_id}'
        assert 0 < services.redis.ttl(key) <= 15
    url = fs.allocate_url()
    owner = fs.topology.owner(url).data_unit_id
    fs.put(url, b'data')
    dead = fs._units[owner]
    dead.close()
    assert fs._units[owner] is dead
    assert dead.closed
    fs.refresh_topology()
    assert fs._units[owner] is not dead
    assert not fs._units[owner].closed
    fs._units[owner].close()
    assert fs.get(url) == b'data'
    correct = fs._units[owner]
    wrong_unit = next(unit for unit in services.units if unit.data_unit_id != owner)
    wrong = DealerClient(wrong_unit.endpoint)
    refresh = fs.refresh_topology
    refreshed = []
    try:
        with monkeypatch.context() as patch:
            patch.setitem(fs._units, owner, wrong)
            patch.setattr(fs, 'refresh_topology', lambda: (refreshed.append(True), refresh())[1])
            with pytest.raises(RedisClientError) as error:
                fs.put(url, b'do-not-replay')
            assert error.value.code == 'wrong_owner'
            assert refreshed == [True]
    finally:
        wrong.close()
    assert fs._units[owner] is correct
    services.controller.stop()
    fs._units.pop(owner).close()
    assert fs.get(url) == b'data'
    fs.delete(url)


def test_shared_client_concurrent_replies(services):
    fs = services.fs

    def roundtrip(index):
        url = fs.allocate_url()
        payload = bytes([index]) * 65536
        fs.put(url, payload)
        assert fs.get(url) == payload
        assert fs.delete(url)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(roundtrip, range(32)))


def test_timeout_does_not_consume_next_reply():
    def handler(request):
        if request.body.get('slow'):
            time.sleep(0.2)
        return request.reply(request.body, payload=request.payload)

    server = RouterServer(Endpoint('127.0.0.1', 0), wire_handler=handler).start()
    client = DealerClient(server.endpoint, timeout=0.05)
    try:
        with pytest.raises(RedisClientError) as error:
            client.request('echo', {'slow': True})
        assert error.value.code == 'timeout'
        client.timeout = 2
        reply = client.request(Message.create('echo', {'next': True}, payload=b'next'))
        assert reply.body == {'next': True}
        assert reply.payload == b'next'
    finally:
        client.close()
        server.stop()
