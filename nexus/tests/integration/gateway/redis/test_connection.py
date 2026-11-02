"""Ordinary Redis URLs route reads and writes on either server topology."""

import asyncio

from uuid import uuid4

import numpy as np
import pytest

from nexus.gateway.trajectory_store_backends.redis import NodeRecord, RedisTrajectoryRecordStore, SegmentRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient


pytestmark = pytest.mark.integration


def test_plain_url_detects_backend_and_routes_batch_operations(backend_config):
    from redis.cluster import RedisCluster

    url = backend_config['url'].replace('redis+cluster://', 'redis://', 1)
    records = [NodeRecord('run', index, [index]) for index in range(32)]
    with RedisClient(url, controller_endpoint=backend_config['controller_endpoint'], key_prefix=uuid4().hex) as client:
        assert isinstance(client._r, RedisCluster) == backend_config['url'].startswith('redis+cluster://')
        ids = client.put_multi_node(records)
        restored = client.get_multi_node([(node_id, NodeRecord) for node_id in ids])
        for actual, expected in zip(restored, records, strict=True):
            assert type(actual) is NodeRecord
            np.testing.assert_equal(vars(actual), vars(expected))
        client.delete_multi_node(ids)
        assert not any(client.has_node(node_id)[0] for node_id in ids)


def test_plain_url_store_append_and_release(backend_config):
    async def run():
        config = {**backend_config, 'url': backend_config['url'].replace('redis+cluster://', 'redis://', 1)}
        with RedisTrajectoryRecordStore(uuid4().hex, config) as store:
            node, segment = await store.put_many([NodeRecord('run', 1, [1]), SegmentRecord('run', 's', {})])
            await store.put_fields_many({segment: {'reward': 3}})
            assert await store.get_many([node], fields=['token_ids']) == [{'token_ids': [1]}]
            assert await store.get_many([segment]) == [{'reward': 3}]
            await store.release_many([node, segment])

    asyncio.run(run())
