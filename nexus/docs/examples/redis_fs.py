"""Run Redis clients with local FS services.

    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python docs/examples/redis_fs.py --redis-url redis://127.0.0.1:16379/0

The example creates its own key prefix and deletes its records. FS services are
local threads; use RayCluster or the standalone CLI for separate processes.
"""

import argparse
import asyncio
import json

from uuid import uuid4

import numpy as np

from nexus.gateway.trajectory_store_backends.redis import NodeRecord, RedisTrajectoryRecordStore, SegmentRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.deploy import LocalCluster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redis-url', default='redis://127.0.0.1:16379/0')
    parser.add_argument('--large-mib', type=int, default=20)
    parser.add_argument('--fs-timeout', type=float, default=120, help='RPC timeout in seconds; 0 disables it')
    args = parser.parse_args()
    if args.large_mib < 0:
        parser.error('--large-mib must be nonnegative')
    config = {'url': args.redis_url, 'key_prefix': 'redis-du-example-' + uuid4().hex}
    inline = NodeRecord('example', 1, [10, 11], logprobs=[-0.1])
    large = NodeRecord(
        'example',
        2,
        [10, 11],
        routed_experts=b'x' * args.large_mib * 1024 * 1024,
        multi_modal_items={'image': [{'pixels': b'example'}]},
    )
    with LocalCluster(args.redis_url, num_data_units=2) as cluster:
        print(
            'FS controller:', cluster.info.controller_endpoint.tcp_address, '| data units:', cluster.info.num_data_units
        )
        client_config = {
            **config,
            'controller_endpoint': cluster.info.controller_endpoint,
            'fs_timeout': None if args.fs_timeout == 0 else args.fs_timeout,
        }
        with RedisClient(**client_config) as client:
            for payload in (inline, large):
                ref = client.put_node(payload)
                restored = client.get_node(ref, payload_type=NodeRecord)
                assert type(restored) is NodeRecord
                np.testing.assert_equal(vars(restored), vars(payload))
                label = 'Inline' if payload is inline else 'FS'
                print(f'{label} client: OK', json.dumps({'node_id': ref}))
                client.delete_node(ref)

        async def records():
            with RedisTrajectoryRecordStore('example', client_config) as store:
                refs = await store.put_many([large, SegmentRecord('example', 'segment-1', {'reward': 1})])
                assert await store.get_many([refs[0]], fields=['token_ids']) == [{'token_ids': [10, 11]}]
                await store.put_fields_many({refs[1]: {'advantage': [1.0]}})
                await store.release_many(refs)

        asyncio.run(records())
        print('Trajectory batch/projection/append/release: OK')


if __name__ == '__main__':
    main()
