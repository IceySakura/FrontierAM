"""Own ephemeral Garnet processes; Ray places supervisors and clients use TCP directly."""

from __future__ import annotations

import importlib
import ipaddress
import logging
import re
import shutil
import socket
import subprocess
import tempfile
import time

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


logger = logging.getLogger(__name__)
_STOP_TIMEOUT: float = 120.0
_STARTUP_TIMEOUT: float = 120.0
_TEMP_ROOT: str = '/tmp'


def _normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError('garnet must be a mapping')
    unknown = config.keys() - {'server_command', 'memory'}
    if unknown:
        raise ValueError(
            f'Unsupported garnet configuration keys: {sorted(unknown)}; configure server_command and memory only'
        )
    result = {
        'server_command': ['garnet-server'],
        'memory': '1024g',
        **config,
    }
    command = result['server_command']
    if (
        not isinstance(command, Sequence)
        or isinstance(command, str | bytes)
        or not command
        or any(not isinstance(part, str) or not part or '\0' in part for part in command)
    ):
        raise ValueError('garnet.server_command must be a nonempty list of executable/argument strings')
    if any(part.startswith('-') for part in command):
        raise ValueError('garnet.server_command accepts only the runtime/executable prefix, not server flags')
    result['server_command'] = list(command)
    if not isinstance(result['memory'], str) or not re.fullmatch(r'[1-9]\d*[kKmMgGtT]?', result['memory']):
        raise ValueError('garnet.memory must be a positive size string, for example "1024g"')
    return result


class GarnetServer:
    """Start and stop one preinstalled, in-memory Garnet server.

    Args:
        config: Inner garnet mapping with server_command=['garnet-server'] and
            memory='1024g' defaults. Startup has a fixed 120-second deadline, and
            isolated server directories are created under /tmp.
        host: Local IP address on which to reserve a port and bind Garnet.
        server_id: Endpoint identity; defaults to a fresh UUID.
        node_id: Ray placement identity, or 'local' for standalone use.

    Attributes:
        endpoint: server_id/node_id/host/port mapping, available after start succeeds.
        process: Child Popen handle, retained after stop for exit-status inspection.
        log_path: Startup log in an isolated /tmp directory. Normal stop removes
            the directory; startup failures retain it for diagnosis.
        command: Garnet argument list passed directly to Popen.

    The caller starts and stops each instance once. The environment cleans up
    processes left behind by abnormal worker or task termination.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        host: str,
        *,
        server_id: str | None = None,
        node_id: str = 'local',
    ) -> None:
        self.config = _normalize_config(config)
        ipaddress.ip_address(host)
        self.host = host
        self.port: int | None = None
        self.server_id = server_id or str(uuid4())
        self.node_id = node_id
        self.process: subprocess.Popen[bytes] | None = None
        self.log_path: Path | None = None
        self.directory: Path | None = None
        self.command: list[str] = []
        self._endpoint: dict[str, Any] | None = None
        self._failed = False

    @property
    def endpoint(self) -> dict[str, Any]:
        """Return a copy of the endpoint after a successful start."""
        if self._endpoint is None:
            raise RuntimeError('Garnet has not completed startup; call start() first')
        return dict(self._endpoint)

    def start(self) -> dict[str, Any]:
        """Start and check PING plus INFO garnet_version within the default startup deadline."""
        # NOTE(jerryhrzhou): Redis is optional for metadata imports and is needed only for startup probes.
        redis = importlib.import_module('redis')
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        try:
            directory = Path(tempfile.mkdtemp(prefix='nexus-garnet-', dir=_TEMP_ROOT))
            self.directory = directory
            self.log_path = directory / 'server.log'
            family = socket.AF_INET6 if ipaddress.ip_address(self.host).version == 6 else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as reservation:
                reservation.bind((self.host, 0))
                self.port = reservation.getsockname()[1]
            self.command = [
                *self.config['server_command'],
                '--bind',
                self.host,
                '--port',
                str(self.port),
                '--memory',
                self.config['memory'],
                '--no-obj',
                '--reviv',
            ]
            with self.log_path.open('wb') as log:
                self.process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=directory,
                )
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'Garnet exited with status {self.process.returncode}')
                probe_timeout = min(0.2, (deadline - time.monotonic()) / 8)
                if probe_timeout <= 0:
                    break
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    socket_connect_timeout=probe_timeout,
                    socket_timeout=probe_timeout,
                    retry=redis.retry.Retry(redis.backoff.NoBackoff(), 0),
                    protocol=2,
                )
                try:
                    if client.ping():
                        if not client.info('server').get('garnet_version'):
                            raise RuntimeError('Endpoint is not Garnet: INFO server has no garnet_version')
                        if self.process.poll() is not None:
                            raise RuntimeError('Garnet exited during readiness checks')
                        self._endpoint = {
                            'server_id': self.server_id,
                            'node_id': self.node_id,
                            'host': self.host,
                            'port': self.port,
                        }
                        return self.endpoint
                except (redis.ConnectionError, redis.TimeoutError):
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                finally:
                    client.close()
            raise TimeoutError(f'Garnet startup exceeded {_STARTUP_TIMEOUT} seconds')
        except BaseException as error:
            self._failed = True
            try:
                self.stop()
            except Exception:
                logger.exception('Failed to stop Garnet after startup failure; log: %s', self.log_path)
            if not isinstance(error, Exception):
                raise
            tail = ''
            if self.log_path is not None and self.log_path.exists():
                with self.log_path.open('rb') as log:
                    log.seek(max(0, self.log_path.stat().st_size - 8192))
                    tail = log.read().decode(errors='replace')
            raise RuntimeError(f'{error}; Garnet startup log: {self.log_path}\n{tail}') from error

    def stop(self) -> None:
        """Reap the child and remove its directory unless startup failed."""
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=_STOP_TIMEOUT)
        if not self._failed and self.directory is not None:
            shutil.rmtree(self.directory)


class GarnetServerHandler:
    """Own Ray supervisor handles and stop all children before releasing their actors."""

    def __init__(self, ray: Any) -> None:
        self._ray = ray
        self.actors: list[Any] = []

    def stop(self) -> None:
        """Stop servers and release their Ray actors after normal task completion."""
        self._ray.get([actor.stop.remote() for actor in self.actors], timeout=_STOP_TIMEOUT + 1)
        for actor in self.actors:
            self._ray.kill(actor, no_restart=True)
        self.actors.clear()


def launch_garnet_servers(config: Mapping[str, Any]) -> tuple[dict[str, Any], GarnetServerHandler]:
    """Launch one Garnet supervisor per live machine, including the Ray head.

    Args:
        config: Whole trajectory-store mapping with backend='garnet' and an optional
            garnet mapping as documented by GarnetServer.

    Returns:
        A store mapping with automatically generated server endpoints and an
        owning stop() handler. Inputs accept only server_command and memory;
        generated endpoints are passed internally to clients, not back to this launcher.

    Raises:
        ValueError: Configuration is invalid.
        RuntimeError: Ray is uninitialized, no live machines exist, or startup fails.
            Partial deployments are stopped and node-local startup logs are retained.
    """
    if not isinstance(config, Mapping) or config.get('backend', 'garnet') != 'garnet':
        raise ValueError('Garnet deployment requires a store mapping with backend="garnet"')
    namespace = config.get('namespace')
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('trajectory_store.namespace must be a non-empty string')
    garnet_config = config.get('garnet')
    garnet = _normalize_config({} if garnet_config is None else garnet_config)
    # NOTE(jerryhrzhou): Ray is optional for local process tests and imports of stored metadata.
    ray = importlib.import_module('ray')
    if not ray.is_initialized():
        raise RuntimeError('launch_garnet_servers requires an initialized Ray runtime')
    updated = {**config, 'garnet': garnet}
    nodes: dict[str, str] = {}
    for node in sorted(
        (node for node in ray.nodes() if node['Alive']),
        key=lambda node: (node['NodeManagerAddress'], node['NodeID']),
    ):
        ipaddress.ip_address(node['NodeManagerAddress'])
        nodes.setdefault(node['NodeManagerAddress'], node['NodeID'])
    if not nodes:
        raise RuntimeError('Garnet deployment requires at least one live Ray node')
    affinity = importlib.import_module('ray.util.scheduling_strategies').NodeAffinitySchedulingStrategy
    supervisor = ray.remote(num_cpus=0, num_gpus=0, max_restarts=0)(GarnetServer)
    handler = GarnetServerHandler(ray)
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    try:
        pending = []
        for host, node_id in nodes.items():
            actor = supervisor.options(scheduling_strategy=affinity(node_id=node_id, soft=False)).remote(
                garnet, host, server_id=str(uuid4()), node_id=node_id
            )
            handler.actors.append(actor)
            pending.append(actor.start.remote())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Garnet deployment exceeded the startup deadline while scheduling actors')
        garnet['servers'] = ray.get(pending, timeout=remaining)
    except BaseException:
        handler.stop()
        raise
    return updated, handler
