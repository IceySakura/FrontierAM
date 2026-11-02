"""Redis KV throughput and one-key Tensor versus pickle benchmark."""

from __future__ import annotations

import argparse
import statistics
import time

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

from tests.stress.redis.payload import BenchmarkPayload

from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient
from nexus.gateway.trajectory_store_backends.redis_store.types import NodeId, Serializable


def _payload(i: int, payload_bytes: int) -> Serializable:
    content = ('x' * payload_bytes) if payload_bytes > 0 else f'msg-{i}'
    return BenchmarkPayload(
        data={'role': 'user', 'content': content},
        text=content[:64],
    )


def _stats(samples: list[float], *, wall_s: float) -> dict[str, float]:
    if not samples:
        return {'ops': 0.0, 'avg_ms': 0.0, 'p50_ms': 0.0, 'qps': 0.0, 'wall_s': 0.0}
    return {
        'ops': float(len(samples)),
        'avg_ms': statistics.mean(samples) * 1000.0,
        'p50_ms': statistics.median(samples) * 1000.0,
        # Aggregate throughput from wall clock (correct under concurrency).
        'qps': len(samples) / wall_s if wall_s > 0 else 0.0,
        'wall_s': wall_s,
    }


def _partition_indices(ops: int, clients: int) -> list[list[int]]:
    """Split ``range(ops)`` across clients (contiguous chunks)."""
    if clients < 1:
        raise ValueError('clients must be >= 1')
    base, rem = divmod(ops, clients)
    out: list[list[int]] = []
    start = 0
    for c in range(clients):
        n = base + (1 if c < rem else 0)
        out.append(list(range(start, start + n)))
        start += n
    return out


def _measure_serial(operation, ops: int):
    samples = []
    results = []
    wall = time.perf_counter()
    for _ in range(ops):
        started = time.perf_counter()
        results.append(operation())
        samples.append(time.perf_counter() - started)
    return results, _stats(samples, wall_s=time.perf_counter() - wall)


def _run_tensor_compare(
    client_factory: Callable[..., RedisClient], key_prefix: str, ops: int, tensor_elements: int, tensor_count: int
):
    """Compare equal numbers of per-Tensor pickle and one-key storage operations."""
    import torch

    tensors = [torch.arange(tensor_elements, dtype=torch.float32) + index for index in range(tensor_count)]
    with client_factory(key_prefix=key_prefix) as client:

        def pickle_put_op():
            return [client.put_node(BenchmarkPayload(data={'tensor': tensor})) for tensor in tensors]

        def pickle_get_op(refs):
            return [client.get_node(ref, payload_type=BenchmarkPayload) for ref in refs]

        def tensor_put_op():
            return [client.put_tensor(tensor) for tensor in tensors]

        def tensor_get_op(refs):
            return [client.get_tensor(ref) for ref in refs]

        pickle_get_op(pickle_put_op())
        tensor_get_op(tensor_put_op())
        pickle_refs, pickle_put = _measure_serial(pickle_put_op, ops)
        _, pickle_get = _measure_serial(lambda: pickle_get_op(pickle_refs.pop()), ops)
        tensor_refs, tensor_put = _measure_serial(tensor_put_op, ops)
        _, tensor_get = _measure_serial(lambda: tensor_get_op(tensor_refs.pop()), ops)

    return {
        'pickle': {'put': pickle_put, 'get': pickle_get},
        'one-key': {'put': tensor_put, 'get': tensor_get},
    }


def _worker_put(
    client_factory: Callable[..., RedisClient],
    key_prefix: str,
    client_id: str,
    indices: list[int],
    payload_bytes: int,
) -> tuple[list[float], list[NodeId]]:
    latencies: list[float] = []
    refs: list[NodeId] = []
    with client_factory(client_id=client_id, key_prefix=key_prefix) as client:
        for i in indices:
            t0 = time.perf_counter()
            ref = client.put_node(_payload(i, payload_bytes))
            latencies.append(time.perf_counter() - t0)
            refs.append(ref)
    return latencies, refs


def _worker_get(
    client_factory: Callable[..., RedisClient],
    key_prefix: str,
    client_id: str,
    refs: list[NodeId],
) -> list[float]:
    latencies: list[float] = []
    with client_factory(client_id=client_id, key_prefix=key_prefix) as client:
        for ref in refs:
            t0 = time.perf_counter()
            got = client.get_node(ref, payload_type=BenchmarkPayload)
            assert got is not None
            latencies.append(time.perf_counter() - t0)
    return latencies


def _run(
    *,
    client_factory: Callable[..., RedisClient],
    key_prefix: str,
    ops: int,
    payload_bytes: int,
    clients: int,
) -> dict[str, dict[str, float]]:
    partitions = _partition_indices(ops, clients)

    put_lat: list[float] = []
    ref_parts: list[list[NodeId]] = [[] for _ in partitions]
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = {
            pool.submit(
                _worker_put,
                client_factory,
                key_prefix,
                f'bench-put-{c}',
                part,
                payload_bytes,
            ): c
            for c, part in enumerate(partitions)
            if part
        }
        for fut in as_completed(futures):
            c = futures[fut]
            latencies, refs = fut.result()
            put_lat.extend(latencies)
            ref_parts[c] = refs
    put_stats = _stats(put_lat, wall_s=time.perf_counter() - wall0)

    get_lat: list[float] = []
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        futures = [
            pool.submit(
                _worker_get,
                client_factory,
                key_prefix,
                f'bench-get-{c}',
                refs,
            )
            for c, refs in enumerate(ref_parts)
            if refs
        ]
        for fut in as_completed(futures):
            get_lat.extend(fut.result())
    get_stats = _stats(get_lat, wall_s=time.perf_counter() - wall0)

    return {'put': put_stats, 'get': get_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description='Nexus Redis put/get micro-benchmark')
    parser.add_argument('--ops', type=int, default=100)
    parser.add_argument('--payload-bytes', type=int, default=256)
    parser.add_argument('--tensor-elements', type=int, default=0)
    parser.add_argument('--tensor-count', type=int, default=1)
    parser.add_argument('--clients', type=int, default=1)
    parser.add_argument('--redis-url', default=None, help='Omit to start an isolated local Redis')
    parser.add_argument('--redis-prefix', default=None)
    args = parser.parse_args()

    if args.clients < 1:
        parser.error('--clients must be >= 1')
    if args.ops < 1:
        parser.error('--ops must be >= 1')
    if args.tensor_elements < 0:
        parser.error('--tensor-elements must be >= 0')
    if args.tensor_count < 1:
        parser.error('--tensor-count must be >= 1')
    if args.tensor_elements and args.clients != 1:
        parser.error('--tensor-elements requires --clients 1')

    from nexus.gateway.trajectory_store_backends.redis_store.file_storage import LocalCluster
    from nexus.gateway.trajectory_store_backends.redis_store.local import start_local_redis

    local_redis = None
    key_prefix = args.redis_prefix or f'nexus-redis-bench-{uuid4().hex[:8]}'
    if args.redis_url:
        target = args.redis_url
    else:
        try:
            local_redis = start_local_redis()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        target = local_redis.url
        print(f'started local redis at {target}')

    try:
        with LocalCluster(target) as cluster:
            with cluster.client(key_prefix=key_prefix) as probe:
                node_id = probe.put_node(BenchmarkPayload(data='probe'))
                probe.get_node(node_id, payload_type=BenchmarkPayload)
            if args.tensor_elements:
                runs = _run_tensor_compare(cluster.client, key_prefix, args.ops, args.tensor_elements, args.tensor_count)
            else:
                results = _run(
                    client_factory=cluster.client,
                    key_prefix=key_prefix,
                    ops=args.ops,
                    payload_bytes=args.payload_bytes,
                    clients=args.clients,
                )
    finally:
        if local_redis is not None:
            local_redis.stop()

    if args.tensor_elements:
        print(
            f'mode=redis ops={args.ops} tensor_elements={args.tensor_elements} '
            f'tensor_count={args.tensor_count} redis_url={target}'
        )
        for variant, results in runs.items():
            for kind, stats in results.items():
                print(
                    f'{variant}.{kind}: qps={stats["qps"]:.1f} avg_ms={stats["avg_ms"]:.3f} '
                    f'p50_ms={stats["p50_ms"]:.3f} wall_s={stats["wall_s"]:.3f}'
                )
        for kind in ('put', 'get'):
            pickle_qps = runs['pickle'][kind]['qps']
            tensor_qps = runs['one-key'][kind]['qps']
            print(f'{kind}_one_key_vs_pickle_qps={(tensor_qps / pickle_qps - 1.0) * 100.0:+.1f}%')
        return

    print(
        f'mode=redis ops={args.ops} clients={args.clients} '
        f'payload_bytes={args.payload_bytes} redis_url={target} prefix={key_prefix}'
    )
    for kind, stats in results.items():
        print(
            f'{kind}: qps={stats["qps"]:.1f} avg_ms={stats["avg_ms"]:.3f} '
            f'p50_ms={stats["p50_ms"]:.3f} wall_s={stats["wall_s"]:.3f}'
        )


if __name__ == '__main__':
    main()
