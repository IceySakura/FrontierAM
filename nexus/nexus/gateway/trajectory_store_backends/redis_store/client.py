"""Redis payload and Tensor storage client.

Stores serialized ``Serializable`` under ``{prefix}:node:{node_id}``.
``node_id`` values are client-minted uuid4 hex strings (``new_node_id``).
Redis Cluster batch operations route keys across slots without transactions.
"""

from __future__ import annotations

import asyncio
import ctypes
import pickle
import struct

from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, TypeVar, cast
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from .types import CompareAndSetResult, NodeId, RedisClientError, Serializable, new_node_id
from .utils import get_ray_head_ip


if TYPE_CHECKING:
    from .file_storage.publication import RedisPublication
    from .file_storage.types import BlobClient, Endpoint


_TENSOR_META_LENGTH = struct.Struct('!I')
T = TypeVar('T', bound=Serializable)

_RECORD_CAS = """
local old = redis.call('GET', KEYS[1])
if not old then return -1 end
if old ~= ARGV[1] then return 0 end
redis.call('SET', KEYS[1], ARGV[2])
return 1
"""


def _require_redis():
    try:
        import redis
    except ImportError as exc:  # pragma: no cover
        raise ImportError('redis package is required for backend="redis". Install with: uv sync --extra redis') from exc
    return redis


def parse_sentinel_url(url: str) -> tuple[str, list[tuple[str, int]], int]:
    parsed = urlparse(url)
    if parsed.scheme != 'redis+sentinel' or not parsed.hostname:
        raise ValueError('expected redis+sentinel://<master-name>/<db>?sentinels=<host:port,...>')
    values = parse_qs(parsed.query).get('sentinels', [])
    if len(values) != 1:
        raise ValueError('Sentinel URL requires one sentinels query parameter')
    sentinels = []
    for endpoint in values[0].split(','):
        host, port = endpoint.rsplit(':', 1)
        sentinels.append((host, int(port)))
    return parsed.hostname, sentinels, int(parsed.path.lstrip('/') or 0)


def parse_cluster_url(url: str) -> list[tuple[str, int]]:
    """Parse Cluster startup nodes; Redis Cluster supports database zero only."""
    parsed = urlparse(url)
    if parsed.scheme != 'redis+cluster' or not parsed.hostname:
        raise ValueError('expected redis+cluster://<host>:<port>/0?nodes=<host:port,...>')
    if int(parsed.path.lstrip('/') or 0) != 0:
        raise ValueError('Redis Cluster supports database 0 only')
    endpoints = [parsed.netloc, *parse_qs(parsed.query).get('nodes', [''])[0].split(',')]
    nodes = []
    for endpoint in endpoints:
        if not endpoint:
            continue
        host, port = endpoint.rsplit(':', 1)
        node = (host, int(port))
        if node not in nodes:
            nodes.append(node)
    return nodes


def connect_redis(url: str, **kwargs: Any) -> Any:
    """Create a RESP2 client, discovering Cluster routing for ordinary Redis URLs."""
    redis = _require_redis()
    kwargs.setdefault('decode_responses', False)
    kwargs.setdefault('protocol', 2)
    if url.startswith('redis+sentinel://'):
        from redis.sentinel import Sentinel

        master_name, sentinels, db = parse_sentinel_url(url)
        sentinel = Sentinel(sentinels, sentinel_kwargs={'protocol': kwargs['protocol']}, protocol=kwargs['protocol'])
        return sentinel.master_for(master_name, db=db, **kwargs)
    if url.startswith('redis+cluster://'):
        from redis.cluster import ClusterNode, RedisCluster

        db = int(kwargs.pop('db', 0))
        if db != 0:
            raise ValueError('Redis Cluster supports database 0 only')
        startup_nodes = [ClusterNode(host, port) for host, port in parse_cluster_url(url)]
        return RedisCluster(startup_nodes=startup_nodes, **kwargs)
    with ExitStack() as cleanup:
        client = redis.Redis.from_url(url, **kwargs)
        cleanup.callback(client.close)
        if client.info('cluster')['cluster_enabled']:
            from redis.cluster import RedisCluster

            return RedisCluster.from_url(url, **kwargs)
        cleanup.pop_all()
        return client


def _validate_node_id(node_id: NodeId) -> None:
    if not isinstance(node_id, str):
        raise TypeError('node_id must be a string')


def _allocate_node_ids(count: int, node_ids: Sequence[NodeId] | None) -> list[NodeId]:
    if isinstance(node_ids, str | bytes | bytearray):
        raise TypeError('node_ids must be a sequence of IDs, not a string or bytes')
    ids = [new_node_id() for _ in range(count)] if node_ids is None else list(node_ids)
    if len(ids) != count or any(not isinstance(node_id, str) or not node_id for node_id in ids):
        raise ValueError('node_ids must contain one nonempty string per payload')
    if len(set(ids)) != len(ids):
        raise ValueError('node_ids must be unique within a batch')
    return ids


def _decode_payload(node_id: NodeId, payload_type: type[T], blob: bytes | None) -> T:
    if blob is None:
        raise RedisClientError(f'payload not found for node_id={node_id}', code='not_found')
    obj = payload_type.deserialize(blob)
    if not isinstance(obj, payload_type):
        raise RedisClientError(f'invalid payload for node_id={node_id}: expected {payload_type!r}, got {type(obj)!r}')
    return obj


class RedisClient:
    """Redis client storing Serializable payloads and Tensor values."""

    backend = 'redis'
    _publication: RedisPublication

    def __init__(
        self,
        url: str | None = None,
        *,
        key_prefix: str = 'nexus:trajectory',
        client_id: str | None = None,
        redis_client: Any | None = None,
        controller_endpoint: Endpoint | dict[str, Any] | None = None,
        fs_client: BlobClient | None = None,
        fs_timeout: float | None = 120,
        io_workers: int = 32,
        fs_io_threads: int = 32,
        **_ignored: Any,
    ) -> None:
        """``io_workers`` bounds in-flight Redis/FS calls; ``fs_io_threads`` sizes the FS ZMQ context."""
        if io_workers < 2 or (controller_endpoint is not None and fs_client is not None):
            raise ValueError('FS needs at least two I/O workers and one configuration source')
        self.url = url
        self.key_prefix = key_prefix.rstrip(':')
        self.client_id = client_id or f'redis-client-{uuid4().hex[:8]}'
        if redis_client is not None:
            self._r = redis_client
        else:
            if self.url is None:
                self.url = f'redis://{get_ray_head_ip()}:16379/0'
            self._r = connect_redis(self.url)
        owns_fs = fs_client is None
        try:
            from .file_storage.publication import RedisPublication

            if fs_client is None:
                from .file_storage.client import FSClient
                from .file_storage.types import Endpoint as FSEndpoint

                if controller_endpoint is None or isinstance(controller_endpoint, dict):
                    endpoint_options: dict[str, Any] = {'port': 17000}
                    endpoint_options.update(cast('dict[str, Any]', controller_endpoint or {}))
                    if 'host' not in endpoint_options:
                        endpoint_options['host'] = get_ray_head_ip()
                    endpoint = FSEndpoint(**endpoint_options)
                elif isinstance(controller_endpoint, FSEndpoint):
                    endpoint = controller_endpoint
                else:
                    raise TypeError('controller_endpoint must be an Endpoint or mapping')
                fs_client = FSClient(endpoint, timeout=fs_timeout, io_threads=fs_io_threads)
            self._publication = RedisPublication(self, fs_client, owns_fs=owns_fs, io_workers=io_workers)
        except BaseException:
            self._close_redis()
            raise

    def _node_key(self, node_id: NodeId) -> bytes:
        return f'{self.key_prefix}:node:{node_id}'.encode()

    def _execute_without_retry(self, *args: Any) -> Any:
        """Send one restore mutation without replaying a possibly applied command."""
        if callable(getattr(self._r, 'mget_nonatomic', None)):
            from redis.backoff import NoBackoff
            from redis.retry import Retry

            with self._r.pipeline(transaction=False) as pipe:
                # This attribute belongs to this pipeline, not the shared client.
                pipe.retry = Retry(NoBackoff(), 0)
                pipe.execute_command(*args)
                return pipe.send_cluster_commands(pipe._execution_strategy.command_queue, allow_redirections=False)[0]
        from redis.exceptions import ConnectionError as RedisConnectionError
        from redis.exceptions import TimeoutError as RedisTimeoutError

        pool = self._r.connection_pool
        connection = pool.get_connection()
        try:
            connection.send_command(*args)
            return self._r.parse_response(connection, args[0])
        except (ConnectionError, TimeoutError, RedisConnectionError, RedisTimeoutError):
            connection.disconnect()
            raise
        finally:
            pool.release(connection)

    def close(self) -> None:
        self._publication.close()

    def _close_redis(self) -> None:
        close = getattr(self._r, 'close', None)
        if callable(close):
            close()

    def __enter__(self) -> RedisClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def put_tensor(self, tensor: Any) -> NodeId:
        """Store Tensor metadata and CPU storage in one Redis value."""
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'tensor must be torch.Tensor, got {type(tensor).__name__}')

        device = str(tensor.device) if tensor.device.type != 'cpu' else None
        cpu_tensor = tensor.detach().cpu().contiguous()
        tensor_meta = (tuple(cpu_tensor.shape), cpu_tensor.dtype)
        if device is not None:
            tensor_meta += (device,)
        meta = pickle.dumps(tensor_meta, protocol=pickle.HIGHEST_PROTOCOL)

        nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
        record = bytearray(_TENSOR_META_LENGTH.size + len(meta) + nbytes)
        _TENSOR_META_LENGTH.pack_into(record, 0, len(meta))
        payload_offset = _TENSOR_META_LENGTH.size + len(meta)
        record[_TENSOR_META_LENGTH.size : payload_offset] = meta
        if nbytes:
            # Append the contiguous CPU Tensor storage after the variable-length metadata.
            buffer_type = ctypes.c_ubyte * nbytes
            record[payload_offset:] = memoryview(buffer_type.from_address(cpu_tensor.data_ptr()))

        tensor_id = new_node_id()
        self._r.set(f'{self.key_prefix}:tensor:{tensor_id}'.encode(), record)
        return tensor_id

    def get_tensor(self, tensor_id: NodeId) -> Any:
        """Restore a Tensor from one variable-length Redis record."""
        _validate_node_id(tensor_id)

        import torch

        record = self._r.get(f'{self.key_prefix}:tensor:{tensor_id}'.encode())
        if record is None:
            raise RedisClientError(f'Tensor data not found for tensor_id={tensor_id}', code='not_found')

        owner = bytearray(record)
        (meta_length,) = _TENSOR_META_LENGTH.unpack_from(owner)
        payload_offset = _TENSOR_META_LENGTH.size + meta_length
        tensor_meta = pickle.loads(owner[_TENSOR_META_LENGTH.size : payload_offset])
        shape, dtype = tensor_meta[:2]
        device = tensor_meta[2] if len(tensor_meta) == 3 else None
        if len(owner) == payload_offset:
            tensor = torch.empty(shape, dtype=dtype)
        else:
            tensor = torch.frombuffer(memoryview(owner)[payload_offset:], dtype=dtype).reshape(shape)
        return tensor.to(device) if device is not None else tensor

    def put_node(self, payload: Serializable, *, node_id: NodeId | None = None, **kwargs: Any) -> NodeId:
        return self.put_multi_node([payload], node_ids=None if node_id is None else [node_id], **kwargs)[0]

    def put_multi_node(
        self, payloads: Sequence[Serializable], *, node_ids: Sequence[NodeId] | None = None, **kwargs: Any
    ) -> list[NodeId]:
        from .file_storage.publication import publication_errors

        results = self._publication.put(payloads, node_ids=node_ids, overwrite=kwargs.get('overwrite'))
        errors = publication_errors(results)
        if errors:
            raise RedisClientError(
                f'record publication failed: {errors}',
                code='publication_failed',
                body={
                    'partial_ids': [result.node_id if result.error is None else None for result in results],
                    'errors': errors,
                },
            )
        return [result.node_id for result in results]

    def get_node(self, node_id: NodeId, *, payload_type: type[T], **kwargs: Any) -> T:
        """Read a payload using the caller-selected class's decoder."""
        _validate_node_id(node_id)
        if not isinstance(payload_type, type) or not issubclass(payload_type, Serializable):
            raise TypeError('payload_type must be Serializable or a subclass')
        return self._publication.get([(node_id, payload_type)], fields=kwargs.get('_restore_fields'))[0]

    def get_multi_node(
        self,
        requests: Sequence[tuple[NodeId, type[T]]],
        **kwargs: Any,
    ) -> list[T]:
        """Read mixed payload types in request order with one batch fetch."""
        if not requests:
            return []
        for node_id, payload_type in requests:
            _validate_node_id(node_id)
            if not isinstance(payload_type, type) or not issubclass(payload_type, Serializable):
                raise TypeError('payload_type must be Serializable or a subclass')
        return self._publication.get(requests, fields=kwargs.get('_restore_fields'))

    def has_node(self, node_id: NodeId) -> tuple[bool, int | None]:
        exists = bool(self._r.exists(self._node_key(node_id)))
        return (True, 0) if exists else (False, None)

    def delete_node(self, node_id: NodeId) -> None:
        self._publication.delete([node_id], missing_ok=False)

    def get_node_blobs(self, node_ids: Sequence[NodeId]) -> list[bytes | None]:
        """Read raw values in input order, retaining None for missing keys."""
        if not node_ids:
            return []
        for node_id in node_ids:
            _validate_node_id(node_id)
        keys = [self._node_key(node_id) for node_id in node_ids]
        cluster_mget = getattr(self._r, 'mget_nonatomic', None)
        return cluster_mget(keys) if callable(cluster_mget) else self._r.mget(keys)

    def compare_and_set_many(self, updates: Sequence[tuple[NodeId, bytes, bytes]]) -> list[CompareAndSetResult]:
        """Compare raw bytes and replace atomically per key, without retries or rollback."""
        if not updates:
            return []
        for node_id, _, _ in updates:
            _validate_node_id(node_id)
        with self._r.pipeline(transaction=False) as pipe:
            for node_id, expected, replacement in updates:
                pipe.execute_command('EVAL', _RECORD_CAS, 1, self._node_key(node_id), expected, replacement)
            results = self._publication.execute_pipeline(pipe)
            return [CompareAndSetResult(result) for result in results]

    def delete_multi_node(self, node_ids: Sequence[NodeId], *, missing_ok: bool = True) -> None:
        self._publication.delete(node_ids, missing_ok=missing_ok)


class AsyncRedisClient(RedisClient):
    """Async facade that runs the synchronous Redis client in worker threads."""

    async def async_put_tensor(self, tensor: Any) -> NodeId:
        return await asyncio.to_thread(self.put_tensor, tensor)

    async def async_get_tensor(self, tensor_id: NodeId) -> Any:
        return await asyncio.to_thread(self.get_tensor, tensor_id)

    async def async_put_node(self, payload: Serializable, **kwargs: Any) -> NodeId:
        return await asyncio.to_thread(self.put_node, payload, **kwargs)

    async def async_get_node(self, node_id: NodeId, *, payload_type: type[T], **kwargs: Any) -> T:
        return cast(T, await asyncio.to_thread(self.get_node, node_id, payload_type=payload_type, **kwargs))
