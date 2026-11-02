from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import struct
import time

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nexus.dataset.item import TaskFilesSpec
from nexus.distributed.batch.data_list import DataListError, DataListFrame, DataRef, IndexedFramePayload, validate_frame
from nexus.distributed.submission.input import InputNormalizationError, normalize_run_input
from nexus.distributed.utils.identity import canonical_json, canonical_sha256
from nexus.runner.trace.tdsql.config import build_db_config


logger = logging.getLogger(__name__)


BATCH_TASK_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS batch_task_records (
    parent_execution_id VARCHAR(64) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    record_version INT NOT NULL,
    record_sha256 CHAR(64) NOT NULL,
    record_data LONGBLOB NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (parent_execution_id, item_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 shardkey=parent_execution_id
""".strip()

_INSERT_RECORD_SQL = """
INSERT IGNORE INTO batch_task_records
    (parent_execution_id, item_id, record_version, record_sha256, record_data)
VALUES (%s, %s, %s, %s, %s)
""".strip()

NXBR_MAGIC = b'NXBR'
NXBR_VERSION = 3
NXBR_HEADER = struct.Struct('<4sHHQQ')
DEFAULT_MAX_TASK_RECORD_BYTES = 256 * 1024 * 1024
DEFAULT_TASK_STORE_WRITE_CHUNK_BYTES = 64 * 1024 * 1024
# Upper bound on how many primary keys a single delete pass scans before applying
# the byte budget. Keeps the ``item_id IN (...)`` statement small even when rows
# are tiny (e.g. remote-only records without frames).
DEFAULT_TASK_STORE_DELETE_SCAN_LIMIT = 1000


@dataclass(frozen=True, slots=True)
class DecodedBatchTaskRecord:
    identity: dict[str, Any]
    run_input: dict[str, Any]
    frame: DataListFrame | None


@dataclass(frozen=True, slots=True)
class BatchTaskRecord:
    parent_execution_id: str
    item_id: str
    record_version: int
    record_sha256: str
    record_data: bytes

    @classmethod
    def from_identity(cls, *, parent_execution_id: str, identity: Mapping[str, Any]) -> BatchTaskRecord:
        item_id = identity.get('item_id')
        version = identity.get('version')
        if not isinstance(parent_execution_id, str) or not parent_execution_id:
            raise ValueError('batch_task_record_parent_required')
        if not isinstance(item_id, str) or not item_id:
            raise ValueError('batch_task_record_item_required')
        if isinstance(version, bool) or version not in {1, 2}:
            raise ValueError('batch_task_record_version_invalid')
        payload = dict(identity)
        return cls(
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            record_version=version,
            record_sha256=canonical_sha256(payload),
            record_data=canonical_json(payload).encode('utf-8'),
        )

    @classmethod
    def from_v3_identity(
        cls,
        *,
        parent_execution_id: str,
        identity: Mapping[str, Any],
        frame: IndexedFramePayload | None,
        max_record_bytes: int,
        max_frame_bytes: int = DEFAULT_MAX_TASK_RECORD_BYTES,
    ) -> BatchTaskRecord:
        _positive_bytes_limit(max_record_bytes, code='batch_task_record_limit_invalid')
        _positive_bytes_limit(max_frame_bytes, code='batch_data_frame_limit_invalid')
        payload = dict(identity)
        item_id = payload.get('item_id')
        if not isinstance(parent_execution_id, str) or not parent_execution_id:
            raise ValueError('batch_task_record_parent_required')
        if not isinstance(item_id, str) or not item_id:
            raise ValueError('batch_task_record_item_required')
        if (
            payload.get('version') != NXBR_VERSION
            or set(payload) != {'version', 'item_id', 'run_input', 'metadata', 'data_ref'}
            or not isinstance(payload.get('run_input'), dict)
            or not isinstance(payload.get('metadata'), dict)
        ):
            raise ValueError('batch_task_record_invalid')
        try:
            validate_batch_task_files_classification(
                run_input=payload['run_input'],
                has_data_ref=payload['data_ref'] is not None,
            )
        except ValueError as exc:
            raise ValueError('batch_task_record_invalid') from exc
        try:
            data_ref = DataRef.model_validate(payload['data_ref']) if payload['data_ref'] is not None else None
            if (data_ref is None) != (frame is None):
                raise ValueError('frame presence does not match data_ref')
            if data_ref is not None:
                assert frame is not None
                if (
                    data_ref != frame.descriptor.ref
                    or item_id != frame.descriptor.item_id
                    or len(frame.encoded) != frame.descriptor.length
                    or frame.descriptor.length > max_frame_bytes
                ):
                    raise ValueError('frame identity does not match data_ref')
        except (DataListError, ValueError) as exc:
            raise ValueError('batch_task_record_invalid') from exc

        identity_bytes = canonical_json(payload).encode('utf-8')
        frame_bytes = frame.encoded if frame is not None else b''
        record_size = NXBR_HEADER.size + len(identity_bytes) + len(frame_bytes)
        if record_size > max_record_bytes:
            raise ValueError('batch_task_record_exceeds_byte_limit')
        record_data = (
            NXBR_HEADER.pack(NXBR_MAGIC, NXBR_VERSION, 0, len(identity_bytes), len(frame_bytes))
            + identity_bytes
            + frame_bytes
        )
        return cls(
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            record_version=NXBR_VERSION,
            record_sha256=canonical_sha256(payload),
            record_data=record_data,
        )


class BatchTaskStore(Protocol):
    backend_name: str

    async def put_records(self, records: Sequence[BatchTaskRecord]) -> None: ...

    async def read_record(self, *, parent_execution_id: str, item_id: str) -> BatchTaskRecord: ...

    async def delete_records(self, *, parent_execution_id: str) -> None: ...

    async def close(self) -> None: ...


def decode_task_record(
    record: BatchTaskRecord,
    *,
    expected_sha256: str,
    max_record_bytes: int = DEFAULT_MAX_TASK_RECORD_BYTES,
    max_frame_bytes: int = DEFAULT_MAX_TASK_RECORD_BYTES,
) -> DecodedBatchTaskRecord:
    _positive_bytes_limit(max_record_bytes, code='batch_task_record_limit_invalid')
    _positive_bytes_limit(max_frame_bytes, code='batch_data_frame_limit_invalid')
    if record.record_sha256 != expected_sha256:
        raise ValueError('batch_task_record_identity_mismatch')
    if len(record.record_data) > max_record_bytes:
        raise ValueError('batch_task_record_exceeds_byte_limit')
    if record.record_version == NXBR_VERSION:
        return _decode_v3_task_record(
            record,
            expected_sha256=expected_sha256,
            max_frame_bytes=max_frame_bytes,
        )
    if record.record_version not in {1, 2}:
        raise ValueError('batch_task_record_invalid')
    try:
        payload = json.loads(record.record_data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('batch_task_record_invalid') from exc
    if not isinstance(payload, dict) or canonical_sha256(payload) != expected_sha256:
        raise ValueError('batch_task_record_identity_mismatch')
    if payload.get('version') != record.record_version or payload.get('item_id') != record.item_id:
        raise ValueError('batch_task_record_identity_mismatch')
    run_input = payload.get('run_input')
    if not isinstance(run_input, dict):
        raise ValueError('batch_task_record_invalid')
    try:
        normalized_run_input = normalize_run_input(run_input)
    except InputNormalizationError as exc:
        raise ValueError('batch_task_record_invalid') from exc
    return DecodedBatchTaskRecord(identity=payload, run_input=normalized_run_input, frame=None)


def decode_run_input(record: BatchTaskRecord, *, expected_sha256: str) -> dict[str, Any]:
    return decode_task_record(record, expected_sha256=expected_sha256).run_input


def _decode_v3_task_record(
    record: BatchTaskRecord,
    *,
    expected_sha256: str,
    max_frame_bytes: int,
) -> DecodedBatchTaskRecord:
    if len(record.record_data) < NXBR_HEADER.size:
        raise ValueError('batch_task_record_invalid')
    magic, version, flags, identity_size, frame_size = NXBR_HEADER.unpack_from(record.record_data)
    if magic != NXBR_MAGIC or version != NXBR_VERSION or flags != 0 or record.record_version != NXBR_VERSION:
        raise ValueError('batch_task_record_invalid')
    expected_size = NXBR_HEADER.size + identity_size + frame_size
    if expected_size != len(record.record_data):
        raise ValueError('batch_task_record_invalid')
    identity_end = NXBR_HEADER.size + identity_size
    identity_bytes = record.record_data[NXBR_HEADER.size : identity_end]
    frame_bytes = record.record_data[identity_end:]
    try:
        payload = json.loads(identity_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('batch_task_record_invalid') from exc
    if not isinstance(payload, dict) or canonical_json(payload).encode('utf-8') != identity_bytes:
        raise ValueError('batch_task_record_invalid')
    if canonical_sha256(payload) != expected_sha256:
        raise ValueError('batch_task_record_identity_mismatch')
    if (
        set(payload) != {'version', 'item_id', 'run_input', 'metadata', 'data_ref'}
        or payload.get('version') != NXBR_VERSION
        or payload.get('item_id') != record.item_id
        or not isinstance(payload.get('run_input'), dict)
        or not isinstance(payload.get('metadata'), dict)
    ):
        raise ValueError('batch_task_record_identity_mismatch')
    try:
        normalized_run_input = normalize_run_input(payload['run_input'])
        validate_batch_task_files_classification(
            run_input=normalized_run_input,
            has_data_ref=payload['data_ref'] is not None,
        )
        data_ref = DataRef.model_validate(payload['data_ref']) if payload['data_ref'] is not None else None
        if (data_ref is None) != (frame_size == 0):
            raise ValueError('frame presence does not match data_ref')
        frame = (
            validate_frame(
                frame_bytes,
                expected_ref=data_ref,
                expected_item_id=record.item_id,
                max_frame_bytes=max_frame_bytes,
            )
            if data_ref is not None
            else None
        )
    except (DataListError, InputNormalizationError, ValueError) as exc:
        raise ValueError('batch_task_record_invalid') from exc
    return DecodedBatchTaskRecord(identity=payload, run_input=normalized_run_input, frame=frame)


def validate_batch_task_files_classification(*, run_input: dict[str, Any], has_data_ref: bool) -> None:
    """Reject local paths in cloud records and require frames for local files."""

    raw_task_files = run_input.get('task_files')
    if raw_task_files is None:
        return
    try:
        task_files = TaskFilesSpec.model_validate(raw_task_files)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid task_files specification') from exc
    if has_data_ref:
        raise ValueError('task_files must be omitted when data_ref is present')
    if not task_files.is_remote or task_files.type != 'zip':
        raise ValueError('local task_files require a data-list frame')


def _positive_bytes_limit(value: int, *, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(code)
    return value


class TDSQLBatchTaskStore:
    table_statement = BATCH_TASK_TABLE_STATEMENT

    def __init__(
        self,
        *,
        host: str,
        user: str,
        database: str,
        password: str = '',
        port: int = 3306,
        backend_name: str = 'tdsql',
        write_chunk_bytes: int = DEFAULT_TASK_STORE_WRITE_CHUNK_BYTES,
        read_only: bool = False,
    ) -> None:
        self._db_config = build_db_config(host=host, user=user, password=password, db=database, port=port)
        self.backend_name = backend_name
        self._write_chunk_bytes = _positive_bytes_limit(
            write_chunk_bytes,
            code='batch_task_store_write_chunk_bytes_invalid',
        )
        self._connection: Any | None = None
        self._lock = asyncio.Lock()
        self._schema_ready = False
        self._read_only = read_only

    async def put_records(self, records: Sequence[BatchTaskRecord]) -> None:
        if not records:
            return
        parent_execution_id = records[0].parent_execution_id
        if any(record.parent_execution_id != parent_execution_id for record in records):
            raise ValueError('batch_task_records_must_share_parent')

        for chunk in _chunk_records_by_data_bytes(records, max_bytes=self._write_chunk_bytes):
            await self._put_record_chunk(chunk)

    async def _put_record_chunk(self, records: Sequence[BatchTaskRecord]) -> None:
        parent_execution_id = records[0].parent_execution_id

        async def operation(connection: Any) -> None:
            async with connection.cursor() as cursor:
                rows = [
                    (
                        record.parent_execution_id,
                        record.item_id,
                        record.record_version,
                        record.record_sha256,
                        record.record_data,
                    )
                    for record in records
                ]
                await cursor.executemany(_INSERT_RECORD_SQL, rows)
                placeholders = ', '.join(['%s'] * len(records))
                await cursor.execute(
                    'SELECT item_id, record_version, record_sha256 FROM batch_task_records '
                    f'WHERE parent_execution_id = %s AND item_id IN ({placeholders})',
                    (parent_execution_id, *(record.item_id for record in records)),
                )
                fetched = cursor.fetchall()
                if inspect.isawaitable(fetched):
                    fetched = await fetched
            persisted = {str(item_id): (int(version), str(sha256)) for item_id, version, sha256 in fetched}
            expected = {record.item_id: (record.record_version, record.record_sha256) for record in records}
            if persisted != expected:
                raise ValueError('batch_task_record_identity_mismatch')

        await self._write(operation)

    async def read_record(
        self, *, parent_execution_id: str, item_id: str, max_bytes: int | None = None
    ) -> BatchTaskRecord:
        if max_bytes is not None:
            _positive_bytes_limit(max_bytes, code='batch_task_record_limit_invalid')

        async def operation(connection: Any) -> tuple[Any, ...] | None:
            async with connection.cursor() as cursor:
                payload = (
                    'record_data'
                    if max_bytes is None
                    else 'CASE WHEN OCTET_LENGTH(record_data) <= %s THEN record_data ELSE NULL END'
                )
                await cursor.execute(
                    f'SELECT record_version, record_sha256, {payload} FROM batch_task_records '
                    'WHERE parent_execution_id = %s AND item_id = %s',
                    (parent_execution_id, item_id) if max_bytes is None else (max_bytes, parent_execution_id, item_id),
                )
                fetched = cursor.fetchone()
                if inspect.isawaitable(fetched):
                    fetched = await fetched
                return fetched

        row = await self._read(operation)
        if row is None:
            raise KeyError('batch_task_record_not_found')
        record_data = row[2]
        if record_data is None:
            raise ValueError('batch_task_record_exceeds_byte_limit')
        if isinstance(record_data, memoryview):
            record_data = record_data.tobytes()
        elif isinstance(record_data, str):
            record_data = record_data.encode('utf-8')
        elif not isinstance(record_data, bytes):
            record_data = bytes(record_data)
        return BatchTaskRecord(
            parent_execution_id=parent_execution_id,
            item_id=item_id,
            record_version=int(row[0]),
            record_sha256=str(row[1]),
            record_data=record_data,
        )

    async def delete_records(self, *, parent_execution_id: str) -> None:
        if not isinstance(parent_execution_id, str) or not parent_execution_id:
            raise ValueError('batch_task_record_parent_required')
        # Delete in byte-bounded batches, each in its own transaction. A single
        # parent can hold multiple GiB of LONGBLOB task records, and a one-shot
        # ``DELETE ... WHERE parent_execution_id = ?`` emits the full row images
        # into the binlog. TXSQL aborts any transaction whose binlog exceeds its
        # threshold (~1.5 GiB), so chunking keeps every transaction well under it
        # while remaining idempotent across retries.
        #
        # Observability contract (kept intentionally quiet for a hot path):
        #   * one INFO on successful completion summarising rounds/items/bytes;
        #   * one ERROR on failure carrying the same accumulated context + exc_info
        #     (then re-raise so the caller keeps its existing error handling);
        #   * per-chunk DEBUG for deep drill-down (off by default);
        #   * WARN when a single record exceeds the byte budget (fires at most
        #     once per oversized record, not per round).
        started_at = time.monotonic()
        rounds = 0
        total_items = 0
        total_bytes = 0
        try:
            while True:
                deleted, chunk_bytes = await self._delete_record_chunk(parent_execution_id, round_no=rounds + 1)
                if deleted == 0:
                    break
                rounds += 1
                total_items += deleted
                total_bytes += chunk_bytes
        except Exception:
            logger.error(
                'batch_task_store_delete_failed parent=%s rounds_committed=%d '
                'items_committed=%d bytes_committed=%d chunk_bytes=%d elapsed_ms=%d',
                parent_execution_id,
                rounds,
                total_items,
                total_bytes,
                self._write_chunk_bytes,
                int((time.monotonic() - started_at) * 1000),
                exc_info=True,
            )
            raise
        logger.info(
            'batch_task_store_delete_done parent=%s rounds=%d items=%d bytes=%d chunk_bytes=%d elapsed_ms=%d',
            parent_execution_id,
            rounds,
            total_items,
            total_bytes,
            self._write_chunk_bytes,
            int((time.monotonic() - started_at) * 1000),
        )

    async def _delete_record_chunk(self, parent_execution_id: str, *, round_no: int = 0) -> tuple[int, int]:
        deleted = 0
        chunk_bytes = 0
        oversized: tuple[str, int] | None = None
        started_at = time.monotonic()

        async def operation(connection: Any) -> None:
            nonlocal deleted, chunk_bytes, oversized
            async with connection.cursor() as cursor:
                await cursor.execute(
                    'SELECT item_id, LENGTH(record_data) FROM batch_task_records '
                    'WHERE parent_execution_id = %s ORDER BY item_id '
                    f'LIMIT {DEFAULT_TASK_STORE_DELETE_SCAN_LIMIT}',
                    (parent_execution_id,),
                )
                fetched = cursor.fetchall()
                if inspect.isawaitable(fetched):
                    fetched = await fetched
                item_ids: list[str] = []
                batch_bytes = 0
                for item_id, length in fetched:
                    size = int(length or 0)
                    # Always take at least one row so an oversized record still makes progress.
                    if item_ids and batch_bytes + size > self._write_chunk_bytes:
                        break
                    item_ids.append(str(item_id))
                    batch_bytes += size
                    if len(item_ids) == 1 and size > self._write_chunk_bytes:
                        # Recorded now, logged after commit so the WARN never fires
                        # for a transaction that ends up rolled back.
                        oversized = (str(item_id), size)
                if not item_ids:
                    return
                placeholders = ', '.join(['%s'] * len(item_ids))
                await cursor.execute(
                    f'DELETE FROM batch_task_records WHERE parent_execution_id = %s AND item_id IN ({placeholders})',
                    (parent_execution_id, *item_ids),
                )
                deleted = len(item_ids)
                chunk_bytes = batch_bytes

        await self._write(operation)

        if oversized is not None:
            item_id, size = oversized
            logger.warning(
                'batch_task_store_delete_oversized_record parent=%s item_id=%s bytes=%d chunk_bytes=%d round=%d',
                parent_execution_id,
                item_id,
                size,
                self._write_chunk_bytes,
                round_no,
            )
        if deleted > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                'batch_task_store_delete_chunk parent=%s round=%d deleted=%d bytes=%d elapsed_ms=%d',
                parent_execution_id,
                round_no,
                deleted,
                chunk_bytes,
                int((time.monotonic() - started_at) * 1000),
            )
        return deleted, chunk_bytes

    async def close(self) -> None:
        async with self._lock:
            self._close_connection()

    async def _write(self, operation) -> None:
        if self._read_only:
            raise ValueError('batch_task_store_is_read_only')
        async with self._lock:
            for attempt in range(2):
                connection = await self._ensure_connection()
                try:
                    await operation(connection)
                    await connection.commit()
                    return
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        await connection.rollback()
                    if attempt == 0 and _is_recoverable_error(exc):
                        self._close_connection()
                        continue
                    raise

    async def _read(self, operation):
        async with self._lock:
            for attempt in range(2):
                connection = await self._ensure_connection()
                try:
                    result = await operation(connection)
                    await connection.rollback()
                    return result
                except asyncio.CancelledError:
                    self._close_connection()
                    raise
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        await connection.rollback()
                    if attempt == 0 and _is_recoverable_error(exc):
                        self._close_connection()
                        continue
                    raise
        raise RuntimeError('unreachable')

    async def _ensure_connection(self):
        if self._connection is None:
            from nexus.runner.trace.tdsql._driver import connect

            self._connection = await connect(**self._db_config)
            self._schema_ready = False
        if not self._schema_ready and not self._read_only:
            async with self._connection.cursor() as cursor:
                await cursor.execute(self.table_statement)
            await self._connection.commit()
            self._schema_ready = True
        return self._connection

    def _close_connection(self) -> None:
        connection = self._connection
        self._connection = None
        self._schema_ready = False
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()


class MySQLBatchTaskStore(TDSQLBatchTaskStore):
    """The same task-record protocol on an ordinary MySQL control database."""

    table_statement = BATCH_TASK_TABLE_STATEMENT.removesuffix(' shardkey=parent_execution_id').replace(
        'item_id VARCHAR(64) NOT NULL', 'item_id VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL'
    )


def create_batch_task_store(config: Any, *, read_only: bool = False) -> BatchTaskStore | None:
    backend_name = config.batch_task_store_backend
    if backend_name == 'control-db':
        from sqlalchemy.engine import make_url

        url = make_url(config.read_database_url())
        if url.get_backend_name() != 'mysql' or not url.host or not url.username or not url.database:
            raise ValueError('control-db task storage requires a MySQL control database')
        if set(url.query) - {'charset'}:
            raise ValueError('control-db task storage does not support these database URL query options')
        return MySQLBatchTaskStore(
            host=url.host,
            user=url.username,
            password=url.password or '',
            port=url.port or 3306,
            database=url.database,
            backend_name=backend_name,
            write_chunk_bytes=config.batch_task_store_write_chunk_bytes,
            read_only=read_only,
        )
    backend = config.run_trace_sinks.get(backend_name)
    if backend is None:
        return None
    if backend.sink_class != 'nexus.runner.trace.tdsql::TDSQLTraceSink':
        raise ValueError(f'batch task store backend must use TDSQLTraceSink: {backend_name}')
    values = dict(backend.config)
    for field_name, credential_ref in backend.secret_config.items():
        values[field_name] = config.read_run_trace_credential(credential_ref)
    host = values.get('host')
    user = values.get('user')
    port = values.get('port', 3306)
    password = values.get('password', '')
    if not isinstance(host, str) or not isinstance(user, str):
        raise ValueError(f'batch task store backend connection is invalid: {backend_name}')
    if isinstance(port, bool) or not isinstance(port, int) or not isinstance(password, str):
        raise ValueError(f'batch task store backend connection is invalid: {backend_name}')
    return TDSQLBatchTaskStore(
        host=host,
        port=port,
        user=user,
        password=password,
        database=config.batch_task_store_database,
        backend_name=backend_name,
        write_chunk_bytes=config.batch_task_store_write_chunk_bytes,
        read_only=read_only,
    )


def _chunk_records_by_data_bytes(
    records: Sequence[BatchTaskRecord],
    *,
    max_bytes: int,
) -> Iterator[tuple[BatchTaskRecord, ...]]:
    current: list[BatchTaskRecord] = []
    current_bytes = 0
    for record in records:
        size_bytes = len(record.record_data)
        if current and current_bytes + size_bytes > max_bytes:
            yield tuple(current)
            current = []
            current_bytes = 0
        current.append(record)
        current_bytes += size_bytes
        if current_bytes >= max_bytes:
            yield tuple(current)
            current = []
            current_bytes = 0
    if current:
        yield tuple(current)


def _is_recoverable_error(exc: BaseException) -> bool:
    from nexus.runner.trace.tdsql._driver import InterfaceError, OperationalError

    return isinstance(exc, (InterfaceError, OperationalError))


__all__ = [
    'BATCH_TASK_TABLE_STATEMENT',
    'BatchTaskRecord',
    'BatchTaskStore',
    'DecodedBatchTaskRecord',
    'NXBR_HEADER',
    'MySQLBatchTaskStore',
    'TDSQLBatchTaskStore',
    'create_batch_task_store',
    'decode_run_input',
    'decode_task_record',
    'validate_batch_task_files_classification',
]
