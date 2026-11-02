"""Lazy direct routing, temporary readiness probes and safe discovery refresh.

Refresh validates immutable membership. It never migrates data or replays an
operation whose remote outcome is unknown. Each contacted DU keeps one cached
I/O thread; lazy creation avoids threads for unused DUs, not for active ones.
Connections share a private ZMQ context with ``io_threads`` transport threads
(the process default has one, which caps receive throughput), and ``get``
returns a zero-copy memoryview that keeps the received frame alive.
"""

from __future__ import annotations

import threading

from concurrent.futures import ThreadPoolExecutor
from typing import cast

import zmq

from nexus.gateway.trajectory_store_backends.redis_store.file_storage.network import DealerClient, MsgType
from nexus.gateway.trajectory_store_backends.redis_store.file_storage.types import BlobClient, Endpoint, Topology
from nexus.gateway.trajectory_store_backends.redis_store.types import RedisClientError


class FSClient(BlobClient):
    def __init__(
        self,
        controller_endpoint: Endpoint,
        *,
        timeout: float | None = 120,
        verify_ready: bool = True,
        io_threads: int = 32,
    ):
        self._context = zmq.Context(io_threads=io_threads)
        self._controller = DealerClient(controller_endpoint, timeout=timeout, context=self._context)
        self._units = {}
        self._dump_routes: dict[str, dict[str, str]] = {}  # dump ID → DU ID → routing URL
        self._lock = threading.RLock()
        self._closed = False
        self._timeout = timeout
        self.topology: Topology | None = None
        self.last_discovery_error: str | None = None
        try:
            hello, _ = self._controller.request(MsgType.HANDSHAKE, {'protocol_version': 1})
            if hello.get('protocol_version') != 1:
                raise RedisClientError('unsupported FS protocol', code='protocol_mismatch')
            self.controller_id = hello['controller_id']
            self.topology = self.refresh_topology()
            if hello['cluster_id'] != self.topology.cluster_id:
                raise RedisClientError('controller cluster identity changed', code='topology_changed')
            if verify_ready:
                self.check_ready()
        except BaseException:
            self.close()
            raise

    def refresh_topology(self) -> Topology:
        """Re-query discovery without accepting slot/address changes during a run."""
        with self._lock:
            if self._closed:
                raise RedisClientError('FS client is closed', code='closed')
            data, _ = self._controller.request(MsgType.DISCOVER)
            topology = Topology.load(data)
            if self.topology is not None and topology != self.topology:
                raise RedisClientError('FS membership changed; migration is not supported', code='topology_changed')
            for unit in topology.data_units:
                connection = self._units.get(unit.data_unit_id)
                if connection is not None and connection.closed:
                    connection.close()
                    self._units[unit.data_unit_id] = self._dealer(unit.endpoint)
            self.topology = topology
            self.last_discovery_error = None
            return topology

    def check_ready(self) -> list[dict]:
        """Probe all DUs with temporary connections, without populating the cache."""
        with self._lock:
            if self._closed:
                raise RedisClientError('FS client is closed', code='closed')
            results = []
            topology = cast(Topology, self.topology)  # Initial discovery completes in __init__.
            for unit in topology.data_units:
                connection = DealerClient(unit.endpoint, timeout=self._timeout, context=self._context)
                try:
                    hello, _ = connection.request(MsgType.HANDSHAKE, {'protocol_version': 1})
                    if (
                        hello.get('protocol_version') != 1
                        or hello.get('cluster_id') != topology.cluster_id
                        or hello.get('data_unit_id') != unit.data_unit_id
                    ):
                        raise RedisClientError(
                            'data unit identity does not match discovery', code='invalid_registration'
                        )
                    results.append({**hello, 'endpoint': vars(unit.endpoint)})
                finally:
                    connection.close()
            return results

    def allocate_url(self) -> str:
        return cast(Topology, self.topology).new_url()

    def _dealer(self, endpoint: Endpoint) -> DealerClient:
        return DealerClient(endpoint, timeout=self._timeout, context=self._context, zero_copy=True)

    def _unit(self, url: str):
        with self._lock:
            if self._closed:
                raise RedisClientError('FS client is closed', code='closed')
            owner = cast(Topology, self.topology).owner(url)
            connection = self._units.get(owner.data_unit_id)
            if connection is None or connection.closed:
                if connection is not None:
                    connection.close()
                connection = self._dealer(owner.endpoint)
                self._units[owner.data_unit_id] = connection
            return connection

    def _request(self, operation: MsgType, url: str, blob: bytes | bytearray | memoryview = b''):
        try:
            return self._unit(url).request(operation, {'url': url}, blob)
        except RedisClientError as exc:
            if exc.code in ('timeout', 'wrong_owner', 'not_ready', 'closed') and not self._closed:
                try:
                    self.refresh_topology()
                except Exception as discovery_error:
                    self.last_discovery_error = str(discovery_error)
                    raise RedisClientError(
                        f'{exc}; discovery refresh failed: {discovery_error}', code=exc.code
                    ) from exc
            # Do not replay even PUT/DELETE: a timeout cannot prove it was not applied.
            raise

    def put(self, url: str, blob: bytes | bytearray | memoryview) -> None:
        self._request(MsgType.PUT, url, blob)

    def get(self, url: str) -> bytes | bytearray | memoryview:
        return self._request(MsgType.GET, url)[1]

    def delete(self, url: str) -> bool:
        return self._request(MsgType.DELETE, url)[0]['deleted']

    def ensure_ssd_copy(self, url: str) -> None:
        self._unit(url).request('ensure_ssd_copy', {'url': url})

    def import_file_to_ssd(self, url: str, source_path: str) -> None:
        """Import a shared file at its owning DU; return after SSD publication."""
        self._unit(url).request('import_file_to_ssd', {'url': url, 'source_path': source_path})

    def _request_import_units(self, operation: str, urls: list[str], keep_urls: list[str] | None = None) -> dict:
        urls_by_unit: dict[str, list[str]] = {}
        for url in dict.fromkeys(urls):
            unit_id = cast(Topology, self.topology).owner(url).data_unit_id
            urls_by_unit.setdefault(unit_id, []).append(url)
        if not urls_by_unit:
            return {'items': {}}
        keep_urls_set = set(keep_urls or ())
        items = {}

        def request_unit(url, body):
            return self._unit(url).request(operation, body)

        with ThreadPoolExecutor(
            max_workers=min(32, len(urls_by_unit)), thread_name_prefix='hykv-fs-import'
        ) as executor:
            futures_by_unit = {}
            for unit_id, unit_urls in urls_by_unit.items():
                body = {'urls': unit_urls}
                if keep_urls is not None:
                    body['keep_urls'] = [url for url in unit_urls if url in keep_urls_set]
                futures_by_unit[unit_id] = executor.submit(request_unit, unit_urls[0], body)
            for unit_id, future in futures_by_unit.items():
                try:
                    result = future.result()[0]
                except (RedisClientError, OSError) as exc:
                    # Preserve acknowledged outcomes from other DUs; never replay an uncertain control request.
                    items.update({url: {'state': 'unknown', 'error': str(exc)} for url in urls_by_unit[unit_id]})
                else:
                    items.update(result['items'])
        return {'items': items}

    def get_ssd_import_status(self, urls: list[str]) -> dict:
        return self._request_import_units('get_ssd_import_status', urls)

    def finalize_ssd_imports(self, urls: list[str], keep_urls: list[str]) -> dict:
        return self._request_import_units('finalize_ssd_imports', urls, keep_urls)

    def _request_dump_units(self, operation, dump_id, requests_by_unit):
        with self._lock:
            unit_routes = dict(self._dump_routes.get(dump_id, {}))
        if not unit_routes:
            return {}
        with ThreadPoolExecutor(max_workers=min(32, len(unit_routes)), thread_name_prefix='hykv-fs-dump') as executor:
            futures_by_unit = {
                unit_id: executor.submit(
                    self._unit(url).request, operation, {'dump_id': dump_id, **requests_by_unit[unit_id]}
                )
                for unit_id, url in unit_routes.items()
                if unit_id in requests_by_unit
            }
            return {unit_id: future.result()[0] for unit_id, future in futures_by_unit.items()}

    def prepare_dump_sources(self, dump_id: str, urls: list[str]) -> dict:
        urls_by_unit: dict[str, list[str]] = {}
        with self._lock:
            unit_routes = self._dump_routes.setdefault(dump_id, {})
            for url in dict.fromkeys(urls):
                unit_id = cast(Topology, self.topology).owner(url).data_unit_id
                urls_by_unit.setdefault(unit_id, []).append(url)
                # Record participants before any RPC so partial failures remain abortable.
                unit_routes[unit_id] = url
        return self._request_dump_units(
            'prepare_dump_sources', dump_id, {key: {'urls': value} for key, value in urls_by_unit.items()}
        )

    def start_dump_copy(self, dump_id: str, dump_dir: str) -> dict:
        with self._lock:
            requests_by_unit = {unit_id: {'dump_dir': dump_dir} for unit_id in self._dump_routes.get(dump_id, {})}
        return self._request_dump_units('start_dump_copy', dump_id, requests_by_unit)

    def get_dump_status(self, dump_id: str) -> dict:
        with self._lock:
            requests_by_unit = dict.fromkeys(self._dump_routes.get(dump_id, {}), {})
        return self._request_dump_units('get_dump_status', dump_id, requests_by_unit)

    def finalize_dump(self, dump_id: str) -> dict:
        with self._lock:
            requests_by_unit = dict.fromkeys(self._dump_routes.get(dump_id, {}), {})
        return self._request_dump_units('finalize_dump', dump_id, requests_by_unit)

    def status(self) -> dict:
        return self._controller.request('status')[0]

    def close(self):
        with self._lock:
            self._closed = True
            units = list(self._units.values())
        for unit in units:
            unit.close()
        self._controller.close()
        if not any(connection.thread_alive for connection in [*units, self._controller]):
            # term() waits for open sockets; skip it if a dealer thread failed to stop.
            self._context.term()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
