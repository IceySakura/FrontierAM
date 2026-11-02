"""Store independently routed fields directly in Garnet over pooled RESP2 connections."""

from __future__ import annotations

import asyncio
import threading
import uuid

from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from .payload_codec import decode_blob, encode_blob
from .types import (
    NodePayloadRecord,
    NodePayloadRef,
    Record,
    RecordRef,
    SegmentDataRef,
    TrajectoryRecordStore,
)
from .utils import _record_fields


_SOCKET_TIMEOUT: float = 120.0
_READ_WORKERS: int = 32
_READ_BATCH_BYTES: int = 64 * 1024**2


@dataclass
class GarnetFieldRef:
    """Locate one field and carry its encoded length after PUT completes."""

    server_id: str
    key: str
    codec: str
    nbytes: int = 0


@dataclass(eq=False)
class GarnetNodePayloadRef(NodePayloadRef):
    """Carry a node's field locations between processes."""

    field_references: dict[str, GarnetFieldRef]


@dataclass(eq=False)
class GarnetSegmentDataRef(SegmentDataRef):
    """Carry the field locations of a segment, including appended fields."""

    field_references: dict[str, GarnetFieldRef]


GarnetRecordRef: TypeAlias = GarnetNodePayloadRef | GarnetSegmentDataRef


class GarnetTrajectoryRecordStore(TrajectoryRecordStore):
    """Read and write fields using endpoints discovered by the Ray launcher.

    The frontend owns ref lifetimes: publications and field appends are unique,
    operations on a ref do not overlap, and clients stay open until work completes.
    I/O errors propagate directly, without cancellation handling or rollback.
    """

    def __init__(self, namespace: str, config: Mapping[str, Any] | None = None) -> None:
        # NOTE(jerryhrzhou): Keep other backends importable without the Garnet extra.
        import redis

        from redis.backoff import NoBackoff
        from redis.retry import Retry

        # NOTE(jerryhrzhou): Compile only when constructing a Garnet client; imports and other backends stay lightweight.
        from .garnet_native import load_native_client_type

        native_client_type = load_native_client_type()

        config = config or {}
        self._key_prefix = f'nexus:garnet:{namespace}'
        servers = config['servers']
        self._workers = len(servers)
        self._clients = {
            server['server_id']: redis.Redis(
                host=server['host'],
                port=server['port'],
                protocol=2,
                decode_responses=False,
                socket_timeout=_SOCKET_TIMEOUT,
                socket_connect_timeout=5,
                max_connections=self._workers,
                retry=Retry(NoBackoff(), 0),
            )
            for server in servers
        }
        self._server_ids = tuple(self._clients)
        self._executor = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix='garnet-field')
        self._native_client_type = native_client_type
        self._read_endpoints = {server['server_id']: (server['host'], server['port']) for server in servers}
        self._read_clients: list[Any] = []
        self._read_local = threading.local()
        self._read_executor = ThreadPoolExecutor(max_workers=_READ_WORKERS, thread_name_prefix='garnet-get')

    def close(self) -> None:
        """Close client resources after the frontend has finished its work."""
        self._executor.shutdown()
        self._read_executor.shutdown()
        for client in self._read_clients:
            client.close()
        self._read_clients.clear()
        for client in self._clients.values():
            client.close()

    def __enter__(self) -> GarnetTrajectoryRecordStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    async def _run_by_server(self, operation: Callable[[str, Any], Any], groups: Mapping[str, Any]) -> list[Any]:
        loop = asyncio.get_running_loop()
        return await asyncio.gather(
            *(loop.run_in_executor(self._executor, operation, server_id, batch) for server_id, batch in groups.items())
        )

    def _field_ref(self, value: Any) -> GarnetFieldRef:
        identifier = uuid.uuid4()
        return GarnetFieldRef(
            self._server_ids[identifier.int % len(self._server_ids)],
            f'{self._key_prefix}:{identifier.hex}',
            'bytes' if isinstance(value, bytes) else 'blob',
        )

    def _write_fields(self, server_id: str, fields: list[tuple[GarnetFieldRef, Any]]) -> None:
        encoded = {}
        for reference, value in fields:
            blob = value if reference.codec == 'bytes' else memoryview(encode_blob(value))
            reference.nbytes = len(blob)
            encoded[reference.key] = blob
        self._clients[server_id].mset(encoded)

    async def put_many(self, records: Sequence[Record]) -> list[RecordRef]:
        """Write one MSET per server and return field locations in record order."""
        references: list[RecordRef] = []
        by_server: dict[str, list[tuple[GarnetFieldRef, Any]]] = defaultdict(list)
        for record in records:
            payload = _record_fields(record)
            fields = {name: self._field_ref(value) for name, value in payload.items()}
            reference_type = GarnetNodePayloadRef if isinstance(record, NodePayloadRecord) else GarnetSegmentDataRef
            references.append(reference_type(fields))
            for name, value in payload.items():
                reference = fields[name]
                by_server[reference.server_id].append((reference, value))
        await self._run_by_server(self._write_fields, by_server)
        return references

    async def get_many(
        self, refs: Sequence[RecordRef], *, fields: Collection[str] | None = None
    ) -> list[dict[str, Any]]:
        """End each MGET batch after crossing 64 MiB of encoded fields, then restore ref order."""
        requested = tuple(fields) if fields is not None else None
        by_server: dict[str, list[tuple[int, str, GarnetFieldRef]]] = defaultdict(list)
        for index, ref in enumerate(refs):
            locations = cast(GarnetRecordRef, ref).field_references
            names = tuple(locations) if requested is None else tuple(name for name in requested if name in locations)
            for name in names:
                reference = locations[name]
                by_server[reference.server_id].append((index, name, reference))

        def batches(
            requests: list[tuple[int, str, GarnetFieldRef]],
        ) -> Iterator[list[tuple[int, str, GarnetFieldRef]]]:
            start = batch_bytes = 0
            for index, (_, _, reference) in enumerate(requests):
                batch_bytes += reference.nbytes
                if batch_bytes > _READ_BATCH_BYTES:
                    yield requests[start : index + 1]
                    start = index + 1
                    batch_bytes = 0
            if start < len(requests):
                yield requests[start:]

        def read(server_id: str, requests: list[tuple[int, str, GarnetFieldRef]]) -> list[tuple[int, str, Any]]:
            if not hasattr(self._read_local, 'clients'):
                self._read_local.clients = {}
            clients = self._read_local.clients
            if server_id not in clients:
                client = self._native_client_type(*self._read_endpoints[server_id])
                clients[server_id] = client
                self._read_clients.append(client)
            values = clients[server_id].mget([reference.key for _, _, reference in requests])
            decoded = []
            for (index, name, reference), value in zip(requests, values, strict=True):
                if value is None:
                    raise KeyError(f'Garnet field missing: server={server_id} key={reference.key}')
                decoded.append((
                    index,
                    name,
                    bytes(memoryview(value)) if reference.codec == 'bytes' else decode_blob(value),
                ))
            return decoded

        payloads: list[dict[str, Any]] = [{} for _ in refs]
        loop = asyncio.get_running_loop()
        tasks = []
        pending = deque((server_id, batches(requests)) for server_id, requests in by_server.items())
        while pending:
            server_id, iterator = pending.popleft()
            batch = next(iterator, None)
            if batch is not None:
                tasks.append(loop.run_in_executor(self._read_executor, read, server_id, batch))
                pending.append((server_id, iterator))
        for batch in await asyncio.gather(*tasks):
            for index, name, value in batch:
                payloads[index][name] = value
        return payloads

    async def put_fields_many(self, updates: Mapping[SegmentDataRef, Mapping[str, Any]]) -> None:
        """Write new fields and attach their locations to the supplied segment refs."""
        by_server: dict[str, list[tuple[GarnetFieldRef, Any]]] = defaultdict(list)
        for ref, payload in updates.items():
            locations = cast(GarnetSegmentDataRef, ref).field_references
            for name, value in payload.items():
                reference = self._field_ref(value)
                locations[name] = reference
                by_server[reference.server_id].append((reference, value))
        await self._run_by_server(self._write_fields, by_server)

    async def release_many(self, refs: Sequence[RecordRef]) -> None:
        """Delete the fields after their final consumer has finished."""
        by_server: dict[str, list[str]] = defaultdict(list)
        for ref in refs:
            for reference in cast(GarnetRecordRef, ref).field_references.values():
                by_server[reference.server_id].append(reference.key)

        def delete(server_id: str, keys: list[str]) -> None:
            for index in range(0, len(keys), 1024):
                self._clients[server_id].delete(*keys[index : index + 1024])

        await self._run_by_server(delete, by_server)
