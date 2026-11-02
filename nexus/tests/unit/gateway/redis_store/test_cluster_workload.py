from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from tests.unit.gateway.redis_store.file_storage.test_publication import MemoryFS

from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient


@pytest.mark.parametrize('fail', [False, True])
def test_load_primary_uses_workload_slot_and_closes_client(monkeypatch, fail):
    pytest.importorskip('ray')
    key_slot = pytest.importorskip('redis.cluster').key_slot
    from tests.e2e.redis.run_cluster_tests import LoadActor

    workload = LoadActor.__ray_metadata__.modified_class('redis://192.0.2.1:16379/0')
    raw = Mock(spec=['get_node_from_key', 'close'])
    client = RedisClient(redis_client=raw, key_prefix=workload.prefix, fs_client=MemoryFS())
    monkeypatch.setattr(workload, 'client', lambda: client)
    if fail:
        failure = ConnectionError('topology unavailable')
        raw.get_node_from_key.side_effect = failure
        with pytest.raises(ConnectionError) as caught:
            workload.primary()
        assert caught.value is failure
    else:
        raw.get_node_from_key.return_value = SimpleNamespace(host='192.0.2.2', port=16379)
        assert workload.primary() == ['192.0.2.2', 16379]
    raw.get_node_from_key.assert_called_once()
    probe = raw.get_node_from_key.call_args.args[0]
    assert key_slot(probe) == key_slot(client._node_key(uuid4().hex))
    raw.close.assert_called_once_with()
