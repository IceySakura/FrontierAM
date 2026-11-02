"""Run as a Ray Job using an existing TEST Redis URL reachable from both nodes.

Default: 128 clients, 65 MiB each, two DUs, no GPUs. Only this run's actors and
keys are cleaned up. The supplied Redis service is never stopped or flushed.
"""

import argparse
import json
import time

from uuid import uuid4

import ray

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nexus.cli.commands.redis import parse_memory_bytes
from nexus.gateway.trajectory_store_backends.redis import NodeRecord
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient, connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage import start_ray_cluster
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient
from nexus.gateway.trajectory_store_backends.redis_store.types import DEFAULT_MEMORY_WATERLINE_BYTES


@ray.remote(num_cpus=0.1, num_gpus=0, max_restarts=0)
class Worker:
    def __init__(self, url, controller, prefix):
        assert not ray.get_gpu_ids()
        self.client = RedisClient(url, controller_endpoint=controller, key_prefix=prefix)

    def ready(self):
        return ray.get_runtime_context().get_node_id()

    def put(self, index, size):
        record = NodeRecord('check', index, [index], routed_experts=bytes([index % 251]) * size)
        return self.client.put_node(record)

    def get(self, ref, index, size):
        record = self.client.get_node(ref, payload_type=NodeRecord)
        assert record.token_ids == [index]
        assert record.routed_experts == bytes([index % 251]) * size

    def delete(self, ref):
        self.client.delete_multi_node([ref])

    def close(self):
        self.client.close()


def storage_stats(topology):
    results = []
    for unit in topology.data_units:
        connection = DealerClient(unit.endpoint)
        try:
            results.append(connection.request('stats')[0])
        finally:
            connection.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--redis-url', required=True)
    parser.add_argument('--clients', type=int, default=128)
    parser.add_argument('--mib', type=int, default=65)
    parser.add_argument('--ssd-offload-dir', default=argparse.SUPPRESS)
    parser.add_argument('--memory-waterline-bytes', type=parse_memory_bytes, default=argparse.SUPPRESS)
    parser.add_argument('--memory-safe-line-bytes', type=parse_memory_bytes, default=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.clients < 2 or args.clients % 2 or args.mib <= 0:
        parser.error('clients must be positive and even; mib must be positive')
    ray.init(address='auto')
    nodes = sorted(node['NodeID'] for node in ray.nodes() if node['Alive'] and node['Resources'].get('CPU', 0) > 0)
    if len(nodes) != 2:
        raise RuntimeError('this check requires exactly two live CPU nodes')
    prefix = 'nexus-fs-check-' + uuid4().hex
    workers, report = [], {'clients': args.clients, 'mib': args.mib, 'gpu_request': 0}
    cluster = None
    try:
        storage_options = {
            name: value
            for name, value in vars(args).items()
            if name in ('ssd_offload_dir', 'memory_waterline_bytes', 'memory_safe_line_bytes')
        }
        cluster = start_ray_cluster(
            args.redis_url, data_unit_node_ids=nodes, controller_node_id=nodes[0], **storage_options
        )
        report['storage_options'] = storage_options
        for index in range(args.clients):
            workers.append(
                Worker.options(scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[index % 2], soft=False)).remote(
                    args.redis_url, cluster.info.controller_endpoint, prefix
                )
            )
        placements = ray.get([worker.ready.remote() for worker in workers], timeout=120)
        assert all(placements[i] != placements[(i + 1) % args.clients] for i in range(args.clients))
        size = args.mib * 1024 * 1024
        started = time.monotonic()
        refs = ray.get([worker.put.remote(i, size) for i, worker in enumerate(workers)], timeout=180)
        report['put_seconds'] = time.monotonic() - started
        with cluster.fs_client() as fs:
            stats = storage_stats(fs.topology)
            assert sum(unit['objects'] for unit in stats) == args.clients
            total_used = sum(unit['used_bytes'] for unit in stats)
            assert total_used >= size * args.clients
            effective_waterline = storage_options.get('memory_waterline_bytes', DEFAULT_MEMORY_WATERLINE_BYTES)
            safe_line = storage_options.get('memory_safe_line_bytes')
            effective_safe_line = effective_waterline // 2 if safe_line is None else safe_line
            if total_used > len(stats) * effective_waterline or total_used > args.clients * effective_safe_line:
                assert sum(unit['ssd_objects'] for unit in stats) > 0, 'workload requires SSD offload'
            assert all(unit['resident_bytes'] <= effective_waterline for unit in stats)
            assert all(
                unit['resident_bytes'] + unit['ssd_bytes'] - unit['cached_bytes'] == unit['used_bytes']
                for unit in stats
            )
            assert all(
                unit['resident_objects'] + unit['ssd_objects'] - unit['cached_objects'] == unit['objects']
                for unit in stats
            )
            report['after_put'] = stats
        started = time.monotonic()
        ray.get([workers[(i + 1) % args.clients].get.remote(ref, i, size) for i, ref in enumerate(refs)], timeout=180)
        report['get_with_verification_seconds'] = time.monotonic() - started
        with cluster.fs_client() as fs:
            report['after_get'] = storage_stats(fs.topology)
            assert all(unit['resident_bytes'] <= effective_waterline for unit in report['after_get'])
        ray.get([worker.delete.remote(ref) for worker, ref in zip(workers, refs, strict=True)], timeout=60)
        with cluster.fs_client() as fs:
            assert all(unit['alive'] for unit in fs.status()['units'])
            final_stats = storage_stats(fs.topology)
            assert all(unit['objects'] == 0 and unit['used_bytes'] == 0 for unit in final_stats)
            assert all(
                unit['resident_bytes']
                == unit['ssd_bytes']
                == unit['resident_objects']
                == unit['ssd_objects']
                == unit['cached_bytes']
                == unit['cached_objects']
                == 0
                for unit in final_stats
            )
            report['after_delete'] = final_stats
        redis = connect_redis(args.redis_url)
        try:
            assert not list(redis.scan_iter(match=f'{prefix}:*'))
        finally:
            redis.close()
        report['status'] = 'PASS'
    finally:
        try:
            if workers:
                ray.get([worker.close.remote() for worker in workers], timeout=30)
        finally:
            for worker in workers:
                ray.kill(worker, no_restart=True)
            try:
                if cluster is not None:
                    cluster.stop()
            finally:
                ray.shutdown()
    print('STORAGE_CHECK ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
