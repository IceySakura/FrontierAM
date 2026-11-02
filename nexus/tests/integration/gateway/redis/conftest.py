"""Local RecordStore backends; services are isolated and stopped after testing."""

import os
import shutil
import socket
import subprocess
import time

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.gateway.trajectory_store_backends.redis_store.client import connect_redis
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.controller import ControllerService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.data_unit import DataUnitService
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)


@pytest.fixture(scope='module')
def redis_cluster_url(tmp_path_factory):
    redis = pytest.importorskip('redis')

    binary = os.environ.get('NEXUS_REDIS_SERVER') or shutil.which('redis-server')
    if not binary:
        pytest.skip('redis-server is required for local Cluster tests')
    cli = str(Path(binary).with_name('redis-cli'))
    processes = []
    clients = []
    ports = []
    directory = tmp_path_factory.mktemp('record-redis-cluster')
    try:
        for index in range(3):
            # Reserve a port with room for Redis's port+10000 cluster bus.
            while True:
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                if port < 55000 and port not in ports:
                    with socket.socket() as bus:
                        try:
                            bus.bind(('127.0.0.1', port + 10000))
                        except OSError:
                            continue
                    break
            ports.append(port)
            config = directory / f'{index}.conf'
            config.write_text(
                f'bind 127.0.0.1\nport {port}\ndir {directory}\nsave ""\nappendonly no\n'
                f'cluster-enabled yes\ncluster-config-file nodes-{index}.conf\n'
                'cluster-node-timeout 1000\ncluster-announce-ip 127.0.0.1\n'
            )
            processes.append(
                subprocess.Popen([binary, str(config)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            )
            client = redis.Redis(host='127.0.0.1', port=port)
            clients.append(client)
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
        subprocess.run(
            [
                cli,
                '--cluster',
                'create',
                *(f'127.0.0.1:{p}' for p in ports),
                '--cluster-replicas',
                '0',
                '--cluster-yes',
            ],
            check=True,
            capture_output=True,
            timeout=20,
        )
        deadline = time.monotonic() + 10
        while not all(client.cluster('info')['cluster_state'] == 'ok' for client in clients):
            if time.monotonic() > deadline:
                raise TimeoutError('local Redis Cluster did not become healthy')
            time.sleep(0.05)
        yield f'redis+cluster://127.0.0.1:{ports[0]}/0'
    finally:
        for client in clients:
            client.close()
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=5)


@pytest.fixture(scope='module', params=['redis', 'redis-cluster'])
def redis_backend_config(request):
    if request.param == 'redis-cluster':
        yield {'url': request.getfixturevalue('redis_cluster_url')}
        return
    if request.param == 'redis':
        pytest.importorskip('redis')
        from nexus.gateway.trajectory_store_backends.redis_store.local import LocalRedis, redis_server_available

        if not redis_server_available():
            pytest.skip('redis-server is required for live RecordStore tests')
        with LocalRedis() as server:
            yield {'url': server.url}
        return


@pytest.fixture
def services(redis_backend_config):
    redis = connect_redis(redis_backend_config['url'])
    units, controller, fs = [], None, None
    try:
        for index in range(2):
            unit = DataUnitService(f'unit-{index}', Endpoint('127.0.0.1', 0))
            units.append(unit)
            unit.start()
        topology = Topology.create([DataUnitRegistration(unit.data_unit_id, unit.endpoint) for unit in units])
        controller = ControllerService(redis, topology, Endpoint('127.0.0.1', 0)).start()
        for unit in units:
            unit.configure(topology, controller.endpoint)
        fs = FSClient(controller.endpoint, timeout=2)
        yield SimpleNamespace(fs=fs, controller=controller, units=units, topology=topology, redis=redis)
    finally:
        if fs is not None:
            fs.close()
        for unit in reversed(units):
            unit.stop()
        if controller is not None:
            controller.stop()
        redis.close()


@pytest.fixture
def backend_config(redis_backend_config, services):
    return {**redis_backend_config, 'controller_endpoint': asdict(services.controller.endpoint)}
