from __future__ import annotations

import hashlib
import io
import json
import logging
import struct
import zipfile

from pathlib import Path

import pytest

from nexus.dataset.zip_index import write_nexus_zip_from_directory
from nexus.distributed.config import DistributedConfig


def _zip_bytes(tmp_path: Path) -> bytes:
    root = tmp_path / 'task-files'
    root.mkdir()
    (root / 'input.txt').write_bytes(b'payload')
    return write_nexus_zip_from_directory(root)


def _frame_bytes(tmp_path: Path, item_id: str = 'item-1') -> tuple[bytes, dict[str, object]]:
    archive = _zip_bytes(tmp_path)
    metadata = json.dumps(
        {
            'item_id': item_id,
            'task_files': {
                'sha256': hashlib.sha256(archive).hexdigest(),
                'size_bytes': len(archive),
                'type': 'zip',
            },
            'version': 1,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    frame = struct.pack('<QQQ', 24 + len(metadata) + len(archive), len(metadata), len(archive)) + metadata + archive
    return frame, {
        'version': 1,
        'offset': 32,
        'length': len(frame),
        'sha256': hashlib.sha256(frame).hexdigest(),
    }


def _indexed_frame(encoded: bytes, raw_ref: dict[str, object], *, item_id: str = 'item-1'):
    from nexus.distributed.batch.data_list import DataRef, IndexedFramePayload, validate_frame

    ref = DataRef.model_validate(raw_ref)
    descriptor = validate_frame(encoded, expected_ref=ref, expected_item_id=item_id)
    return IndexedFramePayload(encoded=encoded, descriptor=descriptor)


def test_batch_task_record_uses_canonical_identity_and_sha256():
    from nexus.distributed.batch.task_store import BatchTaskRecord, decode_run_input

    identity = {
        'version': 2,
        'item_id': 'item-1',
        'run_input': {'task_name': 'identity', 'task_config': {'marker': 'task-store'}},
        'metadata': {'suite': 'smoke'},
    }

    record = BatchTaskRecord.from_identity(parent_execution_id='exe-parent', identity=identity)

    assert record.record_data == (
        b'{"item_id":"item-1","metadata":{"suite":"smoke"},"run_input":'
        b'{"task_config":{"marker":"task-store"},"task_name":"identity"},"version":2}'
    )
    assert len(record.record_sha256) == 64
    assert decode_run_input(record, expected_sha256=record.record_sha256)['task_config'] == {'marker': 'task-store'}


def test_decode_run_input_rejects_reference_hash_mismatch():
    from nexus.distributed.batch.task_store import BatchTaskRecord, decode_run_input

    record = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={'version': 1, 'item_id': 'item-1', 'run_input': {'task_name': 'identity'}},
    )

    with pytest.raises(ValueError, match='batch_task_record_identity_mismatch'):
        decode_run_input(record, expected_sha256='0' * 64)


def test_v3_batch_task_record_has_golden_nxbr_envelope_and_decodes_zip(tmp_path: Path):
    from nexus.distributed.batch.task_store import BatchTaskRecord, decode_task_record

    frame, data_ref = _frame_bytes(tmp_path)
    identity = {
        'version': 3,
        'item_id': 'item-1',
        'run_input': {'task_name': 'identity', 'task_config': {'message': '你好'}},
        'metadata': {'suite': 'smoke'},
        'data_ref': data_ref,
    }
    identity_bytes = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()
    expected = struct.pack('<4sHHQQ', b'NXBR', 3, 0, len(identity_bytes), len(frame)) + identity_bytes + frame

    record = BatchTaskRecord.from_v3_identity(
        parent_execution_id='exe-parent',
        identity=identity,
        frame=_indexed_frame(frame, data_ref),
        max_record_bytes=len(expected),
    )
    decoded = decode_task_record(
        record,
        expected_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        max_record_bytes=len(expected),
    )

    assert record.record_version == 3
    assert record.record_sha256 == hashlib.sha256(identity_bytes).hexdigest()
    assert record.record_data == expected
    assert decoded.identity == identity
    assert decoded.run_input['task_config'] == {'message': '你好'}
    assert decoded.frame is not None
    assert decoded.frame.item_id == 'item-1'
    with zipfile.ZipFile(io.BytesIO(decoded.frame.zip_bytes)) as archive:
        assert archive.read('input.txt') == b'payload'


def test_v3_batch_task_record_reuses_indexed_frame_validation(tmp_path: Path, monkeypatch):
    from nexus.distributed.batch import task_store as task_store_module
    from nexus.distributed.batch.data_list import DataRef, IndexedFramePayload, validate_frame
    from nexus.distributed.batch.task_store import BatchTaskRecord

    encoded, raw_ref = _frame_bytes(tmp_path)
    data_ref = DataRef.model_validate(raw_ref)
    descriptor = validate_frame(encoded, expected_ref=data_ref, expected_item_id='item-1')
    payload = IndexedFramePayload(encoded=encoded, descriptor=descriptor)
    identity = {
        'version': 3,
        'item_id': 'item-1',
        'run_input': {'task_name': 'identity'},
        'metadata': {},
        'data_ref': raw_ref,
    }

    def unexpected_validation(*_args, **_kwargs):
        raise AssertionError('indexed frame must not be validated again')

    monkeypatch.setattr(task_store_module, 'validate_frame', unexpected_validation)

    record = BatchTaskRecord.from_v3_identity(
        parent_execution_id='exe-parent',
        identity=identity,
        frame=payload,
        max_record_bytes=4096,
    )

    assert record.record_data.endswith(encoded)


def test_v3_batch_task_record_supports_null_data_ref_without_frame():
    from nexus.distributed.batch.task_store import BatchTaskRecord, decode_task_record

    identity = {
        'version': 3,
        'item_id': 'item-remote',
        'run_input': {'task_name': 'identity'},
        'metadata': {},
        'data_ref': None,
    }
    record = BatchTaskRecord.from_v3_identity(
        parent_execution_id='exe-parent',
        identity=identity,
        frame=None,
        max_record_bytes=1024,
    )

    decoded = decode_task_record(record, expected_sha256=record.record_sha256, max_record_bytes=1024)

    assert struct.unpack_from('<4sHHQQ', record.record_data) == (b'NXBR', 3, 0, len(record.record_data) - 24, 0)
    assert decoded.frame is None
    assert decoded.run_input['task_name'] == 'identity'


def test_v3_batch_task_record_rejects_local_task_files_without_data_ref():
    from nexus.distributed.batch.task_store import BatchTaskRecord

    identity = {
        'version': 3,
        'item_id': 'item-local-path',
        'run_input': {
            'task_name': 'identity',
            'task_files': {'type': 'directory', 'path': '/etc'},
        },
        'metadata': {},
        'data_ref': None,
    }

    with pytest.raises(ValueError, match='batch_task_record_invalid'):
        BatchTaskRecord.from_v3_identity(
            parent_execution_id='exe-parent',
            identity=identity,
            frame=None,
            max_record_bytes=4096,
        )


def test_v3_batch_task_record_decode_rejects_malicious_local_task_files_without_data_ref():
    from nexus.distributed.batch.task_store import NXBR_HEADER, BatchTaskRecord, decode_task_record
    from nexus.distributed.utils.identity import canonical_json, canonical_sha256

    identity = {
        'version': 3,
        'item_id': 'item-local-path',
        'run_input': {
            'task_name': 'identity',
            'task_files': {'type': 'directory', 'path': '/etc'},
        },
        'metadata': {},
        'data_ref': None,
    }
    identity_bytes = canonical_json(identity).encode()
    record_sha256 = canonical_sha256(identity)
    malicious = BatchTaskRecord(
        parent_execution_id='exe-parent',
        item_id='item-local-path',
        record_version=3,
        record_sha256=record_sha256,
        record_data=NXBR_HEADER.pack(b'NXBR', 3, 0, len(identity_bytes), 0) + identity_bytes,
    )

    with pytest.raises(ValueError, match='batch_task_record_invalid'):
        decode_task_record(malicious, expected_sha256=record_sha256, max_record_bytes=4096)


def test_v3_batch_task_record_rejects_task_files_when_frame_is_present(tmp_path: Path):
    from nexus.distributed.batch.task_store import BatchTaskRecord

    frame, data_ref = _frame_bytes(tmp_path)
    identity = {
        'version': 3,
        'item_id': 'item-1',
        'run_input': {
            'task_name': 'identity',
            'task_files': {
                'type': 'zip',
                'path': 'blobs/remote.zip',
                'remote': {'namespace': 'ns', 'dataset': 'ds', 'version': 'v1'},
            },
        },
        'metadata': {},
        'data_ref': data_ref,
    }

    with pytest.raises(ValueError, match='batch_task_record_invalid'):
        BatchTaskRecord.from_v3_identity(
            parent_execution_id='exe-parent',
            identity=identity,
            frame=_indexed_frame(frame, data_ref),
            max_record_bytes=4096,
        )


def test_v3_task_list_parser_rejects_local_task_files_without_data_ref():
    from nexus.distributed.errors import DomainValidationError
    from nexus.distributed.repositories.batch import _parse_task_list_record

    line = json.dumps(
        {
            'version': 3,
            'item_id': 'item-local-path',
            'run_input': {
                'task_name': 'identity',
                'task_files': {'type': 'directory', 'path': '/etc'},
            },
            'metadata': {},
            'data_ref': None,
        },
        separators=(',', ':'),
    ).encode()

    with pytest.raises(DomainValidationError, match='batch_task_list_data_ref_invalid'):
        _parse_task_list_record(line, offset=0)


def test_v3_batch_task_record_enforces_complete_record_limit(tmp_path: Path):
    from nexus.distributed.batch.task_store import BatchTaskRecord

    frame, data_ref = _frame_bytes(tmp_path)
    identity = {
        'version': 3,
        'item_id': 'item-1',
        'run_input': {'task_name': 'identity'},
        'metadata': {},
        'data_ref': data_ref,
    }
    identity_size = len(json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode())
    record_size = 24 + identity_size + len(frame)

    BatchTaskRecord.from_v3_identity(
        parent_execution_id='exe-parent',
        identity=identity,
        frame=_indexed_frame(frame, data_ref),
        max_record_bytes=record_size,
    )
    with pytest.raises(ValueError, match='batch_task_record_exceeds_byte_limit'):
        BatchTaskRecord.from_v3_identity(
            parent_execution_id='exe-parent',
            identity=identity,
            frame=_indexed_frame(frame, data_ref),
            max_record_bytes=record_size - 1,
        )


@pytest.mark.parametrize(
    'mutation',
    ['missing_frame', 'orphan_frame', 'wrong_item', 'bad_magic', 'bad_flags', 'trailing', 'noncanonical_identity'],
)
def test_v3_batch_task_record_rejects_malformed_envelope_or_frame(tmp_path: Path, mutation):
    from nexus.distributed.batch.task_store import BatchTaskRecord, decode_task_record

    frame, data_ref = _frame_bytes(tmp_path)
    identity = {
        'version': 3,
        'item_id': 'item-1',
        'run_input': {'task_name': 'identity'},
        'metadata': {},
        'data_ref': data_ref,
    }
    if mutation == 'missing_frame':
        with pytest.raises(ValueError, match='batch_task_record_invalid'):
            BatchTaskRecord.from_v3_identity(
                parent_execution_id='exe-parent', identity=identity, frame=None, max_record_bytes=4096
            )
        return
    if mutation == 'orphan_frame':
        with pytest.raises(ValueError, match='batch_task_record_invalid'):
            BatchTaskRecord.from_v3_identity(
                parent_execution_id='exe-parent',
                identity={**identity, 'data_ref': None},
                frame=_indexed_frame(frame, data_ref),
                max_record_bytes=4096,
            )
        return
    if mutation == 'wrong_item':
        with pytest.raises(ValueError, match='batch_task_record_invalid'):
            BatchTaskRecord.from_v3_identity(
                parent_execution_id='exe-parent',
                identity={**identity, 'item_id': 'item-other'},
                frame=_indexed_frame(frame, data_ref),
                max_record_bytes=4096,
            )
        return

    valid = BatchTaskRecord.from_v3_identity(
        parent_execution_id='exe-parent',
        identity=identity,
        frame=_indexed_frame(frame, data_ref),
        max_record_bytes=4096,
    )
    payload = bytearray(valid.record_data)
    if mutation == 'bad_magic':
        payload[:4] = b'FAIL'
    elif mutation == 'bad_flags':
        payload[6:8] = (1).to_bytes(2, 'little')
    elif mutation == 'trailing':
        payload.extend(b'x')
    else:
        identity_bytes = json.dumps(identity, ensure_ascii=True, sort_keys=False, separators=(',', ':')).encode()
        payload = bytearray(struct.pack('<4sHHQQ', b'NXBR', 3, 0, len(identity_bytes), len(frame)))
        payload.extend(identity_bytes)
        payload.extend(frame)
    malformed = BatchTaskRecord(
        parent_execution_id=valid.parent_execution_id,
        item_id=valid.item_id,
        record_version=valid.record_version,
        record_sha256=valid.record_sha256,
        record_data=bytes(payload),
    )

    with pytest.raises(ValueError, match='batch_task_record_invalid|batch_task_record_identity_mismatch'):
        decode_task_record(malformed, expected_sha256=valid.record_sha256, max_record_bytes=4096)


def test_tdsql_batch_task_store_uses_isolated_nexus_runner_table():
    from nexus.distributed.batch.task_store import BATCH_TASK_TABLE_STATEMENT

    assert 'CREATE TABLE IF NOT EXISTS batch_task_records' in BATCH_TASK_TABLE_STATEMENT
    assert 'record_data LONGBLOB NOT NULL' in BATCH_TASK_TABLE_STATEMENT
    assert 'PRIMARY KEY (parent_execution_id, item_id)' in BATCH_TASK_TABLE_STATEMENT
    assert 'shardkey=parent_execution_id' in BATCH_TASK_TABLE_STATEMENT


async def test_tdsql_batch_task_store_idempotently_writes_and_reads_records(monkeypatch):
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    class Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def executemany(self, _sql, rows):
            rows = list(rows)
            self.connection.executemany_sizes.append(sum(len(row[4]) for row in rows))
            for parent_id, item_id, version, sha256, data in rows:
                self.connection.rows.setdefault((parent_id, item_id), (version, sha256, data))

        async def execute(self, sql, params=None):
            self.connection.statements.append(sql)
            if sql.startswith('DELETE FROM batch_task_records'):
                parent_id, *item_ids = params
                targets = set(item_ids)
                for key in [key for key in self.connection.rows if key[0] == parent_id and key[1] in targets]:
                    del self.connection.rows[key]
            elif sql.startswith('SELECT item_id, LENGTH(record_data)'):
                parent_id = params[0]
                matching = sorted(key[1] for key in self.connection.rows if key[0] == parent_id)
                self.result = [(item_id, len(self.connection.rows[(parent_id, item_id)][2])) for item_id in matching]
            elif sql.startswith('SELECT item_id'):
                parent_id, *item_ids = params
                self.result = [
                    (item_id, *self.connection.rows[(parent_id, item_id)][:2])
                    for item_id in item_ids
                    if (parent_id, item_id) in self.connection.rows
                ]
            elif sql.startswith('SELECT record_version'):
                self.result = self.connection.rows.get(tuple(params))

        def fetchall(self):
            return self.result

        def fetchone(self):
            return self.result

    class Connection:
        def __init__(self):
            self.rows = {}
            self.statements = []
            self.executemany_sizes = []
            self.commits = 0
            self.closed = False

        def cursor(self):
            return Cursor(self)

        async def commit(self):
            self.commits += 1

        async def rollback(self):
            return None

        def close(self):
            self.closed = True

    connection = Connection()

    async def connect(**kwargs):
        assert kwargs['db'] == 'nexus_runner'
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    first = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={'version': 1, 'item_id': 'item-1', 'run_input': {'task_name': 'identity'}},
    )
    second = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={'version': 1, 'item_id': 'item-2', 'run_input': {'task_name': 'identity'}},
    )
    oversized = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={
            'version': 1,
            'item_id': 'item-3',
            'run_input': {'task_name': 'identity', 'task_config': {'payload': 'x' * 256}},
        },
    )
    store = TDSQLBatchTaskStore(
        host='mysql.example',
        user='writer',
        database='nexus_runner',
        write_chunk_bytes=len(first.record_data) + len(second.record_data),
    )

    await store.put_records([first, second, oversized])
    loaded = await store.read_record(parent_execution_id=first.parent_execution_id, item_id=first.item_id)
    await store.delete_records(parent_execution_id=first.parent_execution_id)
    await store.close()

    assert loaded == first
    assert connection.executemany_sizes == [
        len(first.record_data) + len(second.record_data),
        len(oversized.record_data),
    ]
    assert sum(statement.startswith('CREATE TABLE') for statement in connection.statements) == 1
    assert not connection.rows
    assert connection.closed


async def test_tdsql_batch_task_store_refreshes_snapshot_between_reads(monkeypatch):
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    committed_rows = {}

    class Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def execute(self, sql, params=None):
            if sql.startswith('SELECT record_version'):
                if self.connection.snapshot is None:
                    self.connection.snapshot = dict(committed_rows)
                self.result = self.connection.snapshot.get(tuple(params))

        def fetchone(self):
            return self.result

    class Connection:
        def __init__(self):
            self.snapshot = None

        def cursor(self):
            return Cursor(self)

        async def commit(self):
            self.snapshot = None

        async def rollback(self):
            self.snapshot = None

        def close(self):
            return None

    connection = Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    store = TDSQLBatchTaskStore(host='mysql.example', user='reader', database='nexus_runner')
    first = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={'version': 1, 'item_id': 'item-1', 'run_input': {'task_name': 'identity'}},
    )
    second = BatchTaskRecord.from_identity(
        parent_execution_id='exe-parent',
        identity={'version': 1, 'item_id': 'item-2', 'run_input': {'task_name': 'identity'}},
    )
    committed_rows[(first.parent_execution_id, first.item_id)] = (
        first.record_version,
        first.record_sha256,
        first.record_data,
    )

    assert await store.read_record(parent_execution_id=first.parent_execution_id, item_id=first.item_id) == first
    committed_rows[(second.parent_execution_id, second.item_id)] = (
        second.record_version,
        second.record_sha256,
        second.record_data,
    )

    assert await store.read_record(parent_execution_id=second.parent_execution_id, item_id=second.item_id) == second


async def test_tdsql_batch_task_store_deletes_records_in_binlog_safe_batches(monkeypatch):
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    class Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def executemany(self, _sql, rows):
            for parent_id, item_id, version, sha256, data in rows:
                self.connection.rows.setdefault((parent_id, item_id), (version, sha256, data))

        async def execute(self, sql, params=None):
            if sql.startswith('DELETE FROM batch_task_records'):
                parent_id, *item_ids = params
                batch_bytes = sum(len(self.connection.rows[(parent_id, item_id)][2]) for item_id in item_ids)
                self.connection.delete_batches.append((list(item_ids), batch_bytes))
                for item_id in item_ids:
                    self.connection.rows.pop((parent_id, item_id), None)
            elif sql.startswith('SELECT item_id, LENGTH(record_data)'):
                parent_id = params[0]
                matching = sorted(key[1] for key in self.connection.rows if key[0] == parent_id)
                self.result = [(item_id, len(self.connection.rows[(parent_id, item_id)][2])) for item_id in matching]
            elif sql.startswith('SELECT item_id'):
                parent_id, *item_ids = params
                self.result = [
                    (item_id, *self.connection.rows[(parent_id, item_id)][:2])
                    for item_id in item_ids
                    if (parent_id, item_id) in self.connection.rows
                ]

        def fetchall(self):
            return self.result

    class Connection:
        def __init__(self):
            self.rows = {}
            self.delete_batches = []

        def cursor(self):
            return Cursor(self)

        async def commit(self):
            return None

        async def rollback(self):
            return None

        def close(self):
            return None

    connection = Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    records = [
        BatchTaskRecord.from_identity(
            parent_execution_id='exe-parent',
            identity={
                'version': 1,
                'item_id': f'item-{index:03d}',
                'run_input': {'task_name': 'identity', 'task_config': {'payload': 'x' * 400}},
            },
        )
        for index in range(5)
    ]
    record_size = len(records[0].record_data)
    store = TDSQLBatchTaskStore(
        host='mysql.example',
        user='writer',
        database='nexus_runner',
        write_chunk_bytes=record_size * 2,
    )
    await store.put_records(records)

    await store.delete_records(parent_execution_id='exe-parent')

    assert not connection.rows
    # Five equally sized records with a two-record byte budget => three delete transactions.
    assert [len(item_ids) for item_ids, _ in connection.delete_batches] == [2, 2, 1]
    # No single delete transaction exceeds the chunk budget that bounds the emitted binlog.
    assert all(batch_bytes <= store._write_chunk_bytes for _, batch_bytes in connection.delete_batches)


async def test_tdsql_batch_task_store_delete_progresses_when_record_exceeds_chunk(monkeypatch):
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    class Cursor:
        def __init__(self, connection):
            self.connection = connection
            self.result = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def executemany(self, _sql, rows):
            for parent_id, item_id, version, sha256, data in rows:
                self.connection.rows.setdefault((parent_id, item_id), (version, sha256, data))

        async def execute(self, sql, params=None):
            if sql.startswith('DELETE FROM batch_task_records'):
                parent_id, *item_ids = params
                self.connection.delete_batches.append(list(item_ids))
                for item_id in item_ids:
                    self.connection.rows.pop((parent_id, item_id), None)
            elif sql.startswith('SELECT item_id, LENGTH(record_data)'):
                parent_id = params[0]
                matching = sorted(key[1] for key in self.connection.rows if key[0] == parent_id)
                self.result = [(item_id, len(self.connection.rows[(parent_id, item_id)][2])) for item_id in matching]
            elif sql.startswith('SELECT item_id'):
                parent_id, *item_ids = params
                self.result = [
                    (item_id, *self.connection.rows[(parent_id, item_id)][:2])
                    for item_id in item_ids
                    if (parent_id, item_id) in self.connection.rows
                ]

        def fetchall(self):
            return self.result

    class Connection:
        def __init__(self):
            self.rows = {}
            self.delete_batches = []

        def cursor(self):
            return Cursor(self)

        async def commit(self):
            return None

        async def rollback(self):
            return None

        def close(self):
            return None

    connection = Connection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    records = [
        BatchTaskRecord.from_identity(
            parent_execution_id='exe-parent',
            identity={'version': 1, 'item_id': f'item-{index}', 'run_input': {'task_name': 'identity'}},
        )
        for index in range(2)
    ]
    # Byte budget smaller than any single record must still delete one row per pass.
    store = TDSQLBatchTaskStore(host='mysql.example', user='writer', database='nexus_runner', write_chunk_bytes=1)
    await store.put_records(records)

    await store.delete_records(parent_execution_id='exe-parent')

    assert not connection.rows
    assert connection.delete_batches == [['item-0'], ['item-1']]


def test_batch_task_store_factory_reuses_tdsql_backend_with_isolated_database():
    from nexus.distributed.batch.task_store import TDSQLBatchTaskStore, create_batch_task_store

    config = DistributedConfig(
        run_trace_sinks={
            'tdsql': {
                'sinkClass': 'nexus.runner.trace.tdsql::TDSQLTraceSink',
                'sourceClass': 'nexus.runner.trace.tdsql::TDSQLTraceSource',
                'config': {'host': 'mysql.example', 'port': 3307, 'user': 'writer', 'db': 'trace_db'},
                'secretConfig': {'password': 'tdsql-password'},
            }
        },
        secrets={'run_trace_credentials': {'tdsql-password': 'secret'}},
    )

    store = create_batch_task_store(config)

    assert isinstance(store, TDSQLBatchTaskStore)
    assert store._db_config['host'] == 'mysql.example'
    assert store._db_config['port'] == 3307
    assert store._db_config['db'] == 'nexus_runner'
    assert store._db_config['password'] == 'secret'
    assert store._write_chunk_bytes == 64 * 1024**2


class _InMemoryTaskStoreConnection:
    """Minimal asyncmy-style connection used by the delete-side logging tests."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], tuple[int, str, bytes]] = {}
        self.delete_batches: list[list[str]] = []

    def cursor(self):
        return _InMemoryTaskStoreCursor(self)

    async def commit(self):
        return None

    async def rollback(self):
        return None

    def close(self):
        return None


class _InMemoryTaskStoreCursor:
    def __init__(self, connection: _InMemoryTaskStoreConnection) -> None:
        self.connection = connection
        self.result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def executemany(self, _sql, rows):
        for parent_id, item_id, version, sha256, data in rows:
            self.connection.rows.setdefault((parent_id, item_id), (version, sha256, data))

    async def execute(self, sql, params=None):
        if sql.startswith('DELETE FROM batch_task_records'):
            parent_id, *item_ids = params
            self.connection.delete_batches.append(list(item_ids))
            for item_id in item_ids:
                self.connection.rows.pop((parent_id, item_id), None)
        elif sql.startswith('SELECT item_id, LENGTH(record_data)'):
            parent_id = params[0]
            matching = sorted(key[1] for key in self.connection.rows if key[0] == parent_id)
            self.result = [(item_id, len(self.connection.rows[(parent_id, item_id)][2])) for item_id in matching]
        elif sql.startswith('SELECT item_id'):
            parent_id, *item_ids = params
            self.result = [
                (item_id, *self.connection.rows[(parent_id, item_id)][:2])
                for item_id in item_ids
                if (parent_id, item_id) in self.connection.rows
            ]

    def fetchall(self):
        return self.result


async def test_delete_records_emits_single_info_summary(monkeypatch, caplog):
    """One successful delete_records call should emit exactly one INFO summary
    (per-chunk output stays DEBUG-only to keep hot-path logs quiet)."""
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    connection = _InMemoryTaskStoreConnection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    records = [
        BatchTaskRecord.from_identity(
            parent_execution_id='exe-log-summary',
            identity={
                'version': 1,
                'item_id': f'item-{index:03d}',
                'run_input': {'task_name': 'identity', 'task_config': {'payload': 'x' * 400}},
            },
        )
        for index in range(5)
    ]
    record_size = len(records[0].record_data)
    store = TDSQLBatchTaskStore(
        host='mysql.example', user='writer', database='nexus_runner', write_chunk_bytes=record_size * 2
    )
    await store.put_records(records)

    caplog.set_level(logging.INFO, logger='nexus.distributed.batch.task_store')
    await store.delete_records(parent_execution_id='exe-log-summary')

    done_records = [rec for rec in caplog.records if rec.getMessage().startswith('batch_task_store_delete_done')]
    assert len(done_records) == 1, 'delete_records should emit exactly one INFO summary'
    message = done_records[0].getMessage()
    assert 'parent=exe-log-summary' in message
    # Five records + two-record byte budget => three chunks.
    assert 'rounds=3' in message
    assert 'items=5' in message
    # Per-chunk INFO must stay silent; DEBUG-level lines are filtered out here.
    assert not [rec for rec in caplog.records if rec.getMessage().startswith('batch_task_store_delete_chunk')]


async def test_delete_records_logs_error_and_reraises_on_failure(monkeypatch, caplog):
    """Any failure inside _delete_record_chunk must emit one ERROR carrying the
    partial progress + parent id + exc_info, then re-raise for the caller."""
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    connection = _InMemoryTaskStoreConnection()
    call_count = {'value': 0}
    original_execute = _InMemoryTaskStoreCursor.execute

    async def flaky_execute(self, sql, params=None):
        if sql.startswith('DELETE FROM batch_task_records'):
            call_count['value'] += 1
            if call_count['value'] == 2:
                raise RuntimeError('simulated txsql binlog abort')
        return await original_execute(self, sql, params)

    monkeypatch.setattr(_InMemoryTaskStoreCursor, 'execute', flaky_execute)

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    records = [
        BatchTaskRecord.from_identity(
            parent_execution_id='exe-log-error',
            identity={
                'version': 1,
                'item_id': f'item-{index:03d}',
                'run_input': {'task_name': 'identity', 'task_config': {'payload': 'x' * 400}},
            },
        )
        for index in range(4)
    ]
    record_size = len(records[0].record_data)
    store = TDSQLBatchTaskStore(
        host='mysql.example', user='writer', database='nexus_runner', write_chunk_bytes=record_size * 2
    )
    await store.put_records(records)

    caplog.set_level(logging.INFO, logger='nexus.distributed.batch.task_store')
    with pytest.raises(RuntimeError, match='simulated txsql binlog abort'):
        await store.delete_records(parent_execution_id='exe-log-error')

    failed_records = [rec for rec in caplog.records if rec.getMessage().startswith('batch_task_store_delete_failed')]
    assert len(failed_records) == 1
    message = failed_records[0].getMessage()
    assert 'parent=exe-log-error' in message
    # First chunk committed before the second failed => rounds_committed=1, items_committed=2.
    assert 'rounds_committed=1' in message
    assert 'items_committed=2' in message
    assert failed_records[0].exc_info is not None
    # No success summary should be emitted when the call fails.
    assert not [rec for rec in caplog.records if rec.getMessage().startswith('batch_task_store_delete_done')]


async def test_delete_records_warns_on_oversized_record(monkeypatch, caplog):
    """A single record whose size exceeds the byte budget must still make
    progress and emit a WARN identifying the offending item exactly once."""
    from nexus.distributed.batch.task_store import BatchTaskRecord, TDSQLBatchTaskStore

    connection = _InMemoryTaskStoreConnection()

    async def connect(**_kwargs):
        return connection

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', connect)
    records = [
        BatchTaskRecord.from_identity(
            parent_execution_id='exe-log-oversized',
            identity={
                'version': 1,
                'item_id': f'item-{index:03d}',
                'run_input': {'task_name': 'identity', 'task_config': {'payload': 'x' * 400}},
            },
        )
        for index in range(2)
    ]
    # Byte budget below any single record so the bounded-scan fallback triggers.
    store = TDSQLBatchTaskStore(host='mysql.example', user='writer', database='nexus_runner', write_chunk_bytes=1)
    await store.put_records(records)

    caplog.set_level(logging.WARNING, logger='nexus.distributed.batch.task_store')
    await store.delete_records(parent_execution_id='exe-log-oversized')

    warns = [rec for rec in caplog.records if rec.getMessage().startswith('batch_task_store_delete_oversized_record')]
    # Each oversized record triggers exactly one WARN as its own chunk.
    assert len(warns) == 2
    assert all('parent=exe-log-oversized' in rec.getMessage() for rec in warns)
    assert not connection.rows


@pytest.mark.parametrize('read_only', [False, True])
def test_control_database_task_store_does_not_require_a_registered_trace_backend(read_only):
    from nexus.distributed.batch.task_store import MySQLBatchTaskStore, create_batch_task_store
    from nexus.distributed.config.views import instance_manager_config_view

    config = DistributedConfig(
        database_url='mysql+asyncmy://task-user:task-secret@mysql.test:3307/runner_dev?charset=utf8mb4',
        batch_task_store_backend='control-db',
    )
    assert config.run_trace_sinks == {}
    store = create_batch_task_store(instance_manager_config_view(config), read_only=read_only)
    assert store._read_only is read_only
    assert isinstance(store, MySQLBatchTaskStore)
    assert store.backend_name == 'control-db'
    assert store._db_config['db'] == 'runner_dev'
    assert store._db_config['port'] == 3307
    assert 'shardkey' not in store.table_statement
    assert 'VARCHAR(256) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin' in store.table_statement
    assert 'PRIMARY KEY' in store.table_statement


def test_control_database_task_store_rejects_non_mysql_connection():
    from nexus.distributed.batch.task_store import create_batch_task_store

    with pytest.raises(ValueError, match='requires a MySQL'):
        create_batch_task_store(
            DistributedConfig(database_url='sqlite+aiosqlite://', batch_task_store_backend='control-db')
        )
