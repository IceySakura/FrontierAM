from __future__ import annotations

import argparse
import threading
import time

from collections import Counter
from typing import Any

import ray

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tests.unit.gateway.redis_store.payloads import SamplePayload

from nexus.cli.commands.redis import register_redis_cluster_arguments
from nexus.gateway.trajectory_store_backends.redis_store.client import RedisClient, connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.cluster import (
    RedisDeployment,
    call_on,
    cluster_available,
    cluster_healthy,
    connect_ray,
    endpoint_info,
    inspect_node,
    node_map,
    print_json,
    recover_cluster,
    require_service,
    resolve_topology,
    service,
    smoke_node,
    states,
    wait_available,
    wait_healthy,
)


def cluster_url(deployment: RedisDeployment) -> str:
    return endpoint_info(deployment)['redis_url']


def owner(client: Any, key: bytes) -> list[Any]:
    node = client.get_node_from_key(key)
    return [node.host, node.port]


def set_and_wait(client: Any, key: bytes, value: Any, replicas: int, timeout_ms: int) -> list[Any]:
    """Run SET and WAIT on one fixed primary connection.

    WAIT has no key for RedisCluster to route. A direct client guarantees both
    commands use the same connection; MOVED/connection errors fail the whole
    attempt so the caller can refresh topology and retry from the beginning.
    """
    import redis

    target = client.get_node_from_key(key)
    direct = redis.Redis(
        host=target.host,
        port=target.port,
        protocol=2,
        decode_responses=False,
        socket_timeout=0.5,
        socket_connect_timeout=0.5,
        retry_on_timeout=False,
    )
    try:
        pipe = direct.pipeline(transaction=False)
        pipe.set(key, value)
        if replicas:
            pipe.execute_command('WAIT', replicas, timeout_ms)
        return pipe.execute()
    finally:
        direct.close()


@ray.remote(num_cpus=0.1)
class RedisActor:
    def __init__(self, url: str, primary_count: int, key_prefix: str | None = None) -> None:
        self.primary_count = primary_count
        self.client = RedisClient(
            url=url,
            key_prefix=key_prefix or f'nexus-redis-cluster-{int(time.time())}',
        )

    def put_get(self, value: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        ref = self.client.put_node(SamplePayload(data=value))
        got = self.client.get_node(ref, payload_type=SamplePayload)
        actual = got.data
        if actual != value:
            raise RuntimeError(f'payload mismatch: expected={value}, actual={actual}')
        key = self.client._node_key(ref)
        return {
            'node_id': ref,
            'value': actual,
            'latency_ms': round((time.perf_counter() - started) * 1000, 3),
            'primary': owner(self.client._r, key),
        }

    def put_get_multi(self, count: int | None = None) -> dict[str, Any]:
        count = count or max(30, self.primary_count * 20)
        payloads = [SamplePayload(data={'sequence': index}) for index in range(count)]
        refs = self.client.put_multi_node(payloads)
        got = self.client.get_multi_node([(ref, SamplePayload) for ref in refs])
        values = [item.data for item in got]
        expected = [{'sequence': index} for index in range(count)]
        if values != expected:
            raise RuntimeError(f'batch payload mismatch: expected={expected}, actual={values}')
        primaries = sorted({tuple(owner(self.client._r, self.client._node_key(node_id))) for node_id in refs})
        return {'count': count, 'primaries': [list(primary) for primary in primaries]}

    def seed(self) -> dict[str, Any]:
        result = self.put_get({'phase': 'before-failover', 'value': 'seed'})
        key = self.client._node_key(result['node_id'])
        blob = self.client._r.get(key)
        acknowledged = int(set_and_wait(self.client._r, key, blob, 1, 5000)[-1])
        if acknowledged != 1:
            raise RuntimeError(f'seed was not replicated once: {result}')
        result['replicas_acked'] = acknowledged
        return result

    def wait_for_failover(self, seed_id: Any, old_primary: list[Any], timeout_s: float = 45.0) -> dict[str, Any]:
        started = time.monotonic()
        failures = []
        while time.monotonic() - started < timeout_s:
            try:
                old = self.client.get_node(seed_id, payload_type=SamplePayload)
                seed_key = self.client._node_key(seed_id)
                seed_primary = owner(self.client._r, seed_key)
                result = self.put_get({'phase': 'after-failover'})
                result['old_value'] = old.data
                result['seed_primary'] = seed_primary
                result['elapsed_s'] = round(time.monotonic() - started, 3)
                if seed_primary != old_primary:
                    return {'failure_count': len(failures), 'first_failures': failures[:5], 'success': result}
            except Exception as exc:
                failures.append(f'{type(exc).__name__}: {exc}')
            time.sleep(0.2)
        return {'failure_count': len(failures), 'first_failures': failures[:5], 'success': None}


@ray.remote(num_cpus=0.5)
class LoadActor:
    def __init__(self, url: str) -> None:
        self.url = url
        # The hash tag pins all load keys to one shard so recovery time measures
        # the failed shard, not uninterrupted traffic on the other primaries.
        self.prefix = f'nexus-redis-load-{{{int(time.time())}}}'
        self.failure_mark: float | None = None
        self.old_primary: list[Any] | None = None
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def client(self) -> RedisClient:
        return RedisClient(url=self.url, key_prefix=self.prefix)

    def primary(self) -> list[Any]:
        client = self.client()
        try:
            return owner(client._r, client._node_key('probe'))
        finally:
            client.close()

    def worker(self, worker_id: int, deadline: float) -> None:
        client = self.client()
        sequence = 0
        while not self.stop_event.is_set() and time.monotonic() < deadline:
            started = time.monotonic()
            error = None
            primary = None
            try:
                ref = client.put_node(SamplePayload(data={'worker': worker_id, 'sequence': sequence}))
                got = client.get_node(ref, payload_type=SamplePayload)
                if got.data != {'worker': worker_id, 'sequence': sequence}:
                    error = 'payload mismatch'
                primary = owner(client._r, client._node_key(ref))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            ended = time.monotonic()
            with self.lock:
                self.records.append({
                    'end': ended,
                    'latency_ms': (ended - started) * 1000,
                    'error': error,
                    'primary': primary,
                })
            sequence += 1
        client.close()

    def start(self, seconds: float = 16.0, threads: int = 8) -> dict[str, Any]:
        self.old_primary = self.primary()
        deadline = time.monotonic() + seconds
        self.threads = [threading.Thread(target=self.worker, args=(index, deadline)) for index in range(threads)]
        for thread in self.threads:
            thread.start()
        return {'ip': ray.util.get_node_ip_address(), 'threads': threads, 'primary': self.old_primary}

    def mark_failure(self) -> None:
        self.failure_mark = time.monotonic()

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)

    def report(self) -> dict[str, Any]:
        for thread in self.threads:
            thread.join(timeout=20)
        if self.failure_mark is None or not self.records or self.old_primary is None:
            raise RuntimeError('load test did not collect records')
        before = [item for item in self.records if item['end'] < self.failure_mark]
        after = [item for item in self.records if item['end'] >= self.failure_mark]
        success = [item for item in after if not item['error']]
        errors = [item for item in after if item['error']]
        recovered = sorted(
            (item for item in success if item['primary'] and item['primary'] != self.old_primary),
            key=lambda item: item['end'],
        )

        def percentile(items: list[dict[str, Any]], value: float) -> float | None:
            if not items:
                return None
            ordered = sorted(item['latency_ms'] for item in items)
            return round(ordered[min(len(ordered) - 1, int(len(ordered) * value))], 3)

        return {
            'total_ops': len(self.records),
            'pre_success': sum(not item['error'] for item in before),
            'after_success': len(success),
            'after_errors': len(errors),
            'pre_p99_ms': percentile(before, 0.99),
            'after_p99_ms': percentile(success, 0.99),
            'max_latency_ms': round(max(item['latency_ms'] for item in self.records), 3),
            'first_success_new_primary_s': (round(recovered[0]['end'] - self.failure_mark, 3) if recovered else None),
            'new_primary': recovered[0]['primary'] if recovered else None,
            'error_types': sorted({item['error'] for item in errors}),
        }


@ray.remote(num_cpus=0.5)
class SafetyActor:
    def __init__(self, mode: str, url: str) -> None:
        self.url = url
        self.mode = mode
        self.wait_replicas = {'async': 0, 'wait1': 1}[mode]
        self.prefix = f'nexus-redis-rpo-{{{mode}-{int(time.time())}}}'
        self.started = 0.0
        self.failure_mark: float | None = None
        self.stop_event = threading.Event()
        self.records: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def client(self):
        return connect_redis(self.url)

    def primary(self) -> list[Any]:
        client = self.client()
        try:
            return owner(client, f'{self.prefix}:probe'.encode())
        finally:
            client.close()

    def worker(self, worker_id: int) -> None:
        client = self.client()
        sequence = 0
        while not self.stop_event.is_set():
            key = f'{self.prefix}:{worker_id}:{sequence}'.encode()
            started = time.monotonic()
            error = None
            primary = None
            try:
                value = f'{worker_id}:{sequence}'
                result = (
                    set_and_wait(client, key, value, self.wait_replicas, 1000)
                    if self.wait_replicas
                    else [client.set(key, value)]
                )
                if self.wait_replicas and int(result[-1]) < self.wait_replicas:
                    raise TimeoutError(f'WAIT {self.wait_replicas} was not satisfied')
                primary = owner(client, key)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            ended = time.monotonic()
            with self.lock:
                self.records.append({
                    'key': key,
                    'end': ended,
                    'latency_ms': (ended - started) * 1000,
                    'error': error,
                    'primary': primary,
                })
            sequence += 1
        client.close()

    def start(self, threads: int = 4) -> dict[str, Any]:
        self.started = time.monotonic()
        self.threads = [threading.Thread(target=self.worker, args=(index,)) for index in range(threads)]
        for thread in self.threads:
            thread.start()
        return {'ip': ray.util.get_node_ip_address(), 'mode': self.mode, 'primary': self.primary()}

    def mark_failure(self) -> float:
        self.failure_mark = time.monotonic()
        return self.failure_mark - self.started

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)

    def report(self) -> dict[str, Any]:
        self.stop()
        if self.failure_mark is None:
            raise RuntimeError('failure was not marked')
        before = [item for item in self.records if item['end'] <= self.failure_mark and not item['error']]
        after = [item for item in self.records if item['end'] > self.failure_mark]
        success = [item for item in after if not item['error']]
        errors = [item for item in after if item['error']]
        client = self.client()
        missing = []
        try:
            for offset in range(0, len(before), 500):
                batch = before[offset : offset + 500]
                values = client.mget_nonatomic([item['key'] for item in batch])
                missing.extend(item['key'].decode() for item, value in zip(batch, values, strict=True) if value is None)
            new_primary = owner(client, f'{self.prefix}:probe'.encode())
        finally:
            client.close()
        recovered = sorted(
            (item for item in success if item['primary'] and item['primary'] == new_primary),
            key=lambda item: item['end'],
        )
        return {
            'mode': self.mode,
            'pre_acknowledged': len(before),
            'pre_qps': round(len(before) / (self.failure_mark - self.started), 1),
            'post_success': len(success),
            'post_errors': len(errors),
            'max_latency_ms': round(max(item['latency_ms'] for item in self.records), 3),
            'first_success_new_primary_s': (round(recovered[0]['end'] - self.failure_mark, 3) if recovered else None),
            'new_primary': new_primary,
            'missing_acknowledged_keys': len(missing),
            'missing_examples': missing[:10],
            'error_types': sorted({item['error'] for item in errors}),
        }


def wait_service_state(
    nodes: dict[str, dict[str, Any]],
    deployment: RedisDeployment,
    endpoint: list[Any],
    field: str,
    expected: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    latest = {}
    while time.monotonic() < deadline:
        latest = call_on(nodes, endpoint[0], inspect_node, endpoint[1], deployment.remote_dir)
        if latest[field] == expected:
            return latest
        time.sleep(0.2)
    raise RuntimeError(f'{field} on {endpoint[0]}:{endpoint[1]} did not become {expected!r}: {latest}')


def actor(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, actor_class: Any, *args: Any) -> Any:
    strategy = NodeAffinitySchedulingStrategy(node_id=nodes[deployment.client_ip]['NodeID'], soft=False)
    return actor_class.options(scheduling_strategy=strategy).remote(*args)


def stop_actor(handle: Any) -> None:
    try:  # noqa: SIM105 - Preserve the original best-effort actor cleanup.
        ray.get(handle.stop.remote())
    except Exception:
        pass
    ray.kill(handle)


def test_topology_and_distribution(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    current = wait_healthy(nodes, deployment)
    if not cluster_healthy(current, deployment):
        raise RuntimeError(f'expected a fully healthy configured topology: {current}')
    client = actor(nodes, deployment, RedisActor, cluster_url(deployment), deployment.primary_count)
    try:
        report = ray.get(client.put_get_multi.remote())
        print_json('cluster_distribution', report)
        if len(report['primaries']) != deployment.primary_count:
            raise RuntimeError(f'payload keys did not cover all {deployment.primary_count} primaries: {report}')
    finally:
        ray.kill(client)
    print('CLUSTER_TOPOLOGY=PASS')


def test_primary_failover(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    client = actor(nodes, deployment, RedisActor, cluster_url(deployment), deployment.primary_count)
    seed = ray.get(client.seed.remote())
    old_primary = seed['primary']
    print_json('primary_seed', seed)
    try:
        stopped = call_on(nodes, old_primary[0], service, 'stop', True, old_primary[1], deployment.remote_dir)
        print_json('primary_kill', stopped)
        require_service(stopped, 'Primary kill')
        wait_service_state(nodes, deployment, old_primary, 'redis_ping', '')
        wait_available(nodes, deployment)
        recovery = ray.get(client.wait_for_failover.remote(seed['node_id'], old_primary))
        print_json('primary_failover', recovery)
        success = recovery['success']
        if not success or success['seed_primary'] == old_primary or success['old_value'] != seed['value']:
            raise RuntimeError(f'Primary failover validation failed: {recovery}')
        print('PRIMARY_FAILOVER=PASS')
    finally:
        try:
            recover_cluster(nodes, deployment)
        finally:
            ray.kill(client)


def test_replica_failure(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    current = wait_healthy(nodes, deployment)
    target = next([state['ip'], state['port']] for state in current if state['role'] == 'replica')
    client = actor(nodes, deployment, RedisActor, cluster_url(deployment), deployment.primary_count)
    try:
        try:
            stopped = call_on(nodes, target[0], service, 'stop', False, target[1], deployment.remote_dir)
            print_json('replica_stop', stopped)
            require_service(stopped, 'Replica stop')
            wait_service_state(nodes, deployment, target, 'redis_ping', '')
            degraded = states(nodes, deployment)
            if not cluster_available(degraded, deployment):
                raise RuntimeError(f'Cluster became unavailable after replica failure: {degraded}')
            result = ray.get(client.put_get.remote({'test': 'replica-down'}))
            print_json('replica_client', result)
        finally:
            recover_cluster(nodes, deployment)
        print('REPLICA_FAILURE=PASS')
    finally:
        ray.kill(client)


def test_extra_replica_migration(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    current = wait_healthy(nodes, deployment)
    replica_counts = Counter(state['master_id'] for state in current if state['role'] == 'replica')
    target = min(
        (state for state in current if state['role'] == 'primary'),
        key=lambda state: replica_counts[state['node_id']],
    )
    if replica_counts[target['node_id']] != 1:
        raise RuntimeError(f'no single-replica Primary available for migration test: {current}')
    try:
        stopped = call_on(nodes, target['ip'], service, 'stop', True, target['port'], deployment.remote_dir)
        print_json('migration_primary_kill', stopped)
        require_service(stopped, 'Migration-test primary kill')
        wait_service_state(nodes, deployment, [target['ip'], target['port']], 'redis_ping', '')
        wait_available(nodes, deployment)
        deadline = time.monotonic() + 45
        latest = []
        while time.monotonic() < deadline:
            latest = states(nodes, deployment)
            live_primaries = [state for state in latest if state['redis_ping'] == 'PONG' and state['role'] == 'primary']
            live_replicas = [state for state in latest if state['redis_ping'] == 'PONG' and state['role'] == 'replica']
            live_counts = Counter(state['master_id'] for state in live_replicas)
            if len(live_primaries) == deployment.primary_count and all(
                live_counts[state['node_id']] >= 1 for state in live_primaries
            ):
                print_json('extra_replica_migration', latest)
                print('EXTRA_REPLICA_MIGRATION=PASS')
                return
            time.sleep(0.5)
        raise TimeoutError(f'extra Replica did not migrate to the unprotected Primary: {latest}')
    finally:
        recover_cluster(nodes, deployment)


def test_load_failover(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    workload = actor(nodes, deployment, LoadActor, cluster_url(deployment))
    try:
        started = ray.get(workload.start.remote())
        print_json('load_start', started)
        old_primary = started['primary']
        time.sleep(3)
        stopped = call_on(nodes, old_primary[0], service, 'stop', True, old_primary[1], deployment.remote_dir)
        print_json('load_primary_kill', stopped)
        require_service(stopped, 'Load-test primary kill')
        ray.get(workload.mark_failure.remote())
        wait_service_state(nodes, deployment, old_primary, 'redis_ping', '')
        wait_available(nodes, deployment)
        report = ray.get(workload.report.remote())
        print_json('load_report', report)
        if not report['pre_success'] or report['first_success_new_primary_s'] is None:
            raise RuntimeError(f'workload did not run before and recover after failover: {report}')
        print('LOAD_FAILOVER=PASS')
    finally:
        stop_actor(workload)
        recover_cluster(nodes, deployment)


def test_replication_safety(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment, mode: str) -> None:
    workload = actor(nodes, deployment, SafetyActor, mode, cluster_url(deployment))
    old_primary = None
    try:
        started = ray.get(workload.start.remote())
        print_json('safety_start', started)
        old_primary = started['primary']
        time.sleep(3)
        stopped = call_on(nodes, old_primary[0], service, 'stop', True, old_primary[1], deployment.remote_dir)
        print_json('safety_primary_kill', stopped)
        require_service(stopped, 'Safety-test primary kill')
        ray.get(workload.mark_failure.remote())
        wait_service_state(nodes, deployment, old_primary, 'redis_ping', '')
        wait_available(nodes, deployment)
        time.sleep(5)
        report = ray.get(workload.report.remote())
        print_json(f'safety_{mode}', report)
        if not report['pre_acknowledged']:
            raise RuntimeError(f'no acknowledged writes before failure: {report}')
        if report['new_primary'] == old_primary:
            raise RuntimeError(f'Primary did not fail over: {report}')
        if mode == 'wait1' and report['missing_acknowledged_keys']:
            raise RuntimeError(f'WAIT-confirmed writes were lost: {report}')
        if mode == 'async' and report['first_success_new_primary_s'] is None:
            raise RuntimeError(f'writes did not recover on the new primary: {report}')
        print(f'SAFETY_{mode.upper()}=PASS')
    finally:
        stop_actor(workload)
        if old_primary is not None:
            recover_cluster(nodes, deployment)


def run_suite(nodes: dict[str, dict[str, Any]], deployment: RedisDeployment) -> None:
    print_json('suite_start', wait_healthy(nodes, deployment))
    test_topology_and_distribution(nodes, deployment)
    test_replica_failure(nodes, deployment)
    test_primary_failover(nodes, deployment)
    if deployment.extra_replicas:
        test_extra_replica_migration(nodes, deployment)
    test_load_failover(nodes, deployment)
    for mode in ('async', 'wait1'):
        test_replication_safety(nodes, deployment, mode)
    print_json('suite_final', wait_healthy(nodes, deployment))
    print_json('suite_smoke', call_on(nodes, deployment.hosts[-1], smoke_node, deployment.instances[0]))
    print('SUITE=PASS')


def main() -> None:
    parser = argparse.ArgumentParser(description='Test the configured Redis Cluster')
    parser.add_argument('command', choices=('suite', 'test'))
    parser.add_argument(
        'scenario', nargs='?', choices=('topology', 'primary', 'replica', 'migration', 'load', 'async', 'wait1')
    )
    register_redis_cluster_arguments(parser)
    args = parser.parse_args()
    connect_ray()
    deployment = resolve_topology(
        primary_count=args.primary_count,
        replicas_per_primary=args.replicas_per_primary,
        extra_replicas=args.extra_replicas,
        redis_node_ips=args.redis_node_ips,
        redis_client_ip=args.redis_client_ip,
        redis_dir=args.redis_dir,
    )
    nodes = node_map(deployment)
    if args.command == 'suite':
        run_suite(nodes, deployment)
    elif args.scenario == 'topology':
        test_topology_and_distribution(nodes, deployment)
    elif args.scenario == 'primary':
        test_primary_failover(nodes, deployment)
    elif args.scenario == 'replica':
        test_replica_failure(nodes, deployment)
    elif args.scenario == 'migration':
        test_extra_replica_migration(nodes, deployment)
    elif args.scenario == 'load':
        test_load_failover(nodes, deployment)
    elif args.scenario in {'async', 'wait1'}:
        test_replication_safety(nodes, deployment, args.scenario)
    else:
        parser.error('test requires a scenario')


if __name__ == '__main__':
    main()
