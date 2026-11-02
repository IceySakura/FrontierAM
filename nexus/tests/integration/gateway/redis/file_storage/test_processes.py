"""Controller and DU must also work as standalone processes."""

import json
import socket
import subprocess
import sys
import time

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.client import FSClient
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import (
    DataUnitRegistration,
    Endpoint,
    Topology,
)
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


def test_standalone_processes(redis_backend_config, tmp_path):
    reservations = [socket.socket(), socket.socket()]
    try:
        for sock in reservations:
            sock.bind(('127.0.0.1', 0))
        controller, unit = [Endpoint('127.0.0.1', sock.getsockname()[1]) for sock in reservations]
    finally:
        for sock in reservations:
            sock.close()
    config = tmp_path / 'topology.json'
    config.write_text(json.dumps(Topology.create([DataUnitRegistration('unit', unit)]).dump()))
    command = [sys.executable, '-c', 'from nexus.cli.main import main; raise SystemExit(main())', 'redis', 'fs']
    options = ['--topology', str(config), '--controller-port', str(controller.port)]
    processes = []
    fs = None
    with (tmp_path / 'services.log').open('w+') as log:
        try:
            processes.append(
                subprocess.Popen(
                    command + ['controller', *options, '--redis-url', redis_backend_config['url']],
                    stdout=log,
                    stderr=log,
                )
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                log.flush()
                if '"controller"' in (tmp_path / 'services.log').read_text():
                    break
                assert processes[0].poll() is None, (tmp_path / 'services.log').read_text()
                time.sleep(0.05)
            assert '"controller"' in (tmp_path / 'services.log').read_text(), 'Controller did not become ready'
            processes.append(
                subprocess.Popen(command + ['data-unit', *options, '--unit-id', 'unit'], stdout=log, stderr=log)
            )
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                assert all(proc.poll() is None for proc in processes), (tmp_path / 'services.log').read_text()
                try:
                    fs = FSClient(controller, timeout=0.2)
                    break
                except RedisClientError:
                    time.sleep(0.05)
            assert fs is not None, (tmp_path / 'services.log').read_text()
            url = fs.allocate_url()
            fs.put(url, b'\x00\xff' * 32768)
            assert fs.get(url) == b'\x00\xff' * 32768
            assert fs.delete(url)
        finally:
            if fs is not None:
                fs.close()
            for process in reversed(processes):
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
