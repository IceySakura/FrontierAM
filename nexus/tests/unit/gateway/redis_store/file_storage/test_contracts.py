"""Essential protocol, routing and real TCP service checks."""

import json

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import (
    Codec,
    Message,
    MsgType,
)
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


def test_topology_and_message_contract():
    units = [DataUnitRegistration(str(i), Endpoint('localhost', 17000 + i)) for i in range(2)]
    topology = Topology.create(units, 'test')
    url = topology.new_url()
    assert len(topology.slots) == 256
    assert topology.owner(url) == Topology.create(units[::-1], 'test').owner(url)
    assert Topology.load(json.loads(json.dumps(topology.dump()))) == topology
    request = Message.create(MsgType.PUT, {'url': url}, sender_id='client', payload=b'\x00\xff')
    restored, blob = Codec.decode_wire(Codec.encode(request))
    assert restored == request
    assert blob == request.payload
    assert request.reply().request_id == request.error_reply('bad', 'bad').request_id == request.request_id
    with pytest.raises(RedisClientError):
        topology.owner(url.replace('test', 'other'))


@pytest.mark.parametrize('payload', ['text'])
def test_codec_rejects_non_bytes_at_encoding_boundary(payload):
    with pytest.raises(TypeError, match='payload must be bytes'):
        Codec.encode(Message.create(MsgType.PUT, payload=payload))


@pytest.mark.parametrize(
    'units',
    [
        [],
        [DataUnitRegistration('unit', Endpoint('localhost', 0))],
        [DataUnitRegistration('unit', Endpoint('0.0.0.0', 17000))],
        [
            DataUnitRegistration('same', Endpoint('localhost', 17000)),
            DataUnitRegistration('same', Endpoint('localhost', 17001)),
        ],
    ],
)
def test_topology_rejects_unusable_membership(units):
    with pytest.raises(ValueError, match='data unit|published endpoints'):
        Topology.create(units)
