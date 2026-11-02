"""Durable output handling for Cloud trace event exports."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import mmap
import os
import re
import tempfile
import uuid

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, BinaryIO, NamedTuple, TypeVar, cast
from urllib.parse import urlsplit, urlunsplit

from nexus.distributed.export.types import _JSONL_FIELDS, _LOCAL_TERMINAL_STATUSES, ExportError
from nexus.distributed.trace.models import TraceStreamKey
from nexus.utils.common import to_thread_complete


logger = logging.getLogger(__name__)


_EVENT_COLUMNS = (
    'id',
    'run_id',
    'run_span_id',
    'parent_execution_id',
    'record_id',
    'event_type',
    'time_unix_nano',
    'trace_id',
    'span_id',
    'parent_span_id',
    'event_data',
    'created_at',
)
_EVENT_COLUMN_SET = frozenset(_EVENT_COLUMNS)


# Write-time index cap. Each _IndexEntry costs ~280 B in practice (strings come
# from parsed JSON, so nothing is shared), i.e. ~280 MB at the cap. Past it the
# index is dropped during writing and validation/filtering use the parse-per-line
# paths, so memory stays bounded instead of being spent and then not used.
_MAX_INDEX_ENTRIES = 1_000_000
_READ_BLOCK = 1 << 20

# The parse-free check reads each line's head; it relies on run_id being the first
# serialized key (make_jsonl_record builds the dict in _JSONL_FIELDS order).
assert _JSONL_FIELDS[0] == 'run_id'


def _has_artifacts(record: Mapping[str, Any]) -> bool:
    # Zero and false are valid outputs; only absent or empty top-level payloads are omitted.
    return any(
        value is not None and value != '' and value != [] and value != {}
        for value in (record.get(name) for name in ('task_output', 'agent_output', 'processor_results'))
    )


class _IndexEntry(NamedTuple):
    """Per-record bookkeeping captured while writing; enables parse-free validation."""

    run_id: Any
    status: Any
    retry_count: Any
    offset: int
    length: int  # bytes, including the trailing b'\n'
    has_artifacts: bool


def _check_record_semantics(record: Mapping[str, Any], line_number: int) -> str | None:
    """Mirror _validate_jsonl_file's per-line checks; returns the first error, if any.

    Called while the record is still in memory so validation costs nothing extra.
    Errors are raised later (at close_and_validate) to preserve the established
    close-time failure contract.
    """
    if not isinstance(record, dict):
        return f'JSONL line {line_number} is not an object'
    if tuple(record) not in (_JSONL_FIELDS, (*_JSONL_FIELDS, 'extra')):
        return f'unexpected JSONL fields at line {line_number}'
    if 'extra' in record and not isinstance(record['extra'], dict):
        return f'invalid extra at line {line_number}'
    run_id = record['run_id']
    if not isinstance(run_id, str) or not run_id:
        return f'invalid run_id at line {line_number}'
    if record['status'] not in _LOCAL_TERMINAL_STATUSES:
        return f'invalid status at line {line_number}'
    retry_count = record['retry_count']
    if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
        return f'invalid retry_count at line {line_number}'
    return None


def _expected_line_head(run_id: Any) -> bytes:
    # Mirrors iter_json_bytes(sort_keys=False, separators=(', ', ': ')) byte for byte.
    return b'{"run_id": ' + json.dumps(run_id, ensure_ascii=False).encode('utf-8', 'backslashreplace')


def _verify_integrity(
    path: Path,
    index: Sequence[_IndexEntry],
    expected_hexdigest: str,
    expected_count: int,
    require_sorted: bool,
) -> None:
    """Check the finished file against write-time bookkeeping, without JSON parsing.

    What this proves: the bytes on disk are exactly the bytes the serializer
    emitted (sha256 + size), each indexed line starts with its own
    ``{"run_id": <id>`` and ends with ``}\\n``, and the file holds exactly one
    newline per record. What it does not prove: that the serializer's output
    parses as JSON -- that is delegated to iter_json_bytes; use strict validation
    to re-parse every line.
    """
    if len(index) != expected_count:
        raise ExportError(f'JSONL record count mismatch: expected {expected_count}, got {len(index)}')
    previous_run_id: Any = None
    with path.open('rb') as stream:
        expected_offset = 0
        for i, entry in enumerate(index, start=1):
            if entry.offset != expected_offset:
                raise ExportError(f'JSONL index is not contiguous at line {i}')
            expected_offset += entry.length
            expected_head = _expected_line_head(entry.run_id)
            if entry.length < len(expected_head) + 2:
                raise ExportError(f'JSONL line {i} does not match the written record')
            stream.seek(entry.offset)
            if stream.read(len(expected_head)) != expected_head:
                raise ExportError(f'JSONL line {i} does not match the written record')
            stream.seek(entry.offset + entry.length - 2)
            if stream.read(2) != b'}\n':
                raise ExportError(f'JSONL line {i} does not match the written record')
            if require_sorted and previous_run_id is not None and entry.run_id <= previous_run_id:
                raise ExportError(f'JSONL run_id order is not strictly increasing at line {i}')
            previous_run_id = entry.run_id
        digest = hashlib.sha256()
        size = 0
        newlines = 0
        stream.seek(0)
        for block in iter(lambda: stream.read(_READ_BLOCK), b''):
            digest.update(block)
            size += len(block)
            newlines += block.count(b'\n')
    if size != expected_offset:
        raise ExportError('JSONL file size mismatch')
    if newlines != expected_count:
        raise ExportError(f'JSONL record count mismatch: expected {expected_count} lines, got {newlines}')
    if digest.hexdigest() != expected_hexdigest:
        raise ExportError('JSONL file content does not match the written bytes')


_SOURCE_KEYS = (
    'control_url',
    'execution_id',
    'spec_type',
    'db_host',
    'db_port',
    'db_name',
    'table',
    'storage_scope',
    'content',
    'schema_version',
)
_CHECKPOINT_KEYS = frozenset({
    'version',
    'export_id',
    'started_at',
    'last_committed_at',
    'source',
    'output_path',
    'committed_bytes',
    'committed_records',
    'committed_sha256',
    'streams',
})
_STREAM_CHECKPOINT_KEYS = frozenset({'run_id', 'run_span_id', 'last_record_id'})
_SUPPORTED_SPEC_TYPES = frozenset({'nexus_batch', 'nexus_single', 'nexus_live'})
_SHA256_PATTERN = re.compile(r'[0-9a-f]{64}')
_READ_CHUNK_SIZE = 1024 * 1024
_T = TypeVar('_T')


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportError(f'{label} must be a non-empty string')
    return value


def _plain_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExportError(f'{label} must be an integer greater than or equal to {minimum}')
    return value


def _normalize_control_url(value: object) -> str:
    raw = _nonempty_string(value, 'source control_url')
    parsed = urlsplit(raw)
    if parsed.username is not None or parsed.password is not None:
        raise ExportError('source control_url must not contain credentials')
    normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip('/'), '', ''))
    return _nonempty_string(normalized, 'source control_url')


def _canonical_source(source: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _SOURCE_KEYS if key not in source]
    if missing:
        raise ExportError(f'source is missing required field: {missing[0]}')
    canonical: dict[str, Any] = {
        'control_url': _normalize_control_url(source['control_url']),
        'execution_id': _nonempty_string(source['execution_id'], 'source execution_id'),
        'spec_type': _nonempty_string(source['spec_type'], 'source spec_type'),
        'db_host': _nonempty_string(source['db_host'], 'source db_host'),
        'db_port': _plain_integer(source['db_port'], 'source db_port', minimum=1),
        'db_name': _nonempty_string(source['db_name'], 'source db_name'),
        'table': _nonempty_string(source['table'], 'source table'),
        'storage_scope': _nonempty_string(source['storage_scope'], 'source storage_scope'),
        'content': _nonempty_string(source['content'], 'source content'),
        'schema_version': _plain_integer(source['schema_version'], 'source schema_version', minimum=1),
    }
    if 'trace_backend' in source:
        # Optional, and rebuilt field by field above -- so leaving it out here dropped it
        # silently, and the identity comparison on resume never saw it. That is the whole
        # point of recording it: two backends can publish one table name on different
        # schemas, and resuming across them would splice both shapes into one file.
        #
        # Absent rather than null when unnamed, so a checkpoint written before the option
        # existed still compares equal to a run that does not name one.
        canonical['trace_backend'] = _nonempty_string(source['trace_backend'], 'source trace_backend')
    if canonical['db_port'] > 65535:
        raise ExportError('source db_port must be at most 65535')
    if canonical['spec_type'] not in _SUPPORTED_SPEC_TYPES:
        raise ExportError('source spec_type is unsupported')
    if canonical['content'] != 'traces':
        raise ExportError('source content must be traces')
    if canonical['schema_version'] != 1:
        raise ExportError('source schema_version is unsupported')
    try:
        TraceStreamKey(canonical['storage_scope'], canonical['table'], 'source-validation', 'source-validation')
    except ValueError as exc:
        raise ExportError('source trace storage identity is invalid') from exc
    return canonical


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExportError('value is not valid UTF-8 JSON') from exc


def _json_chunks(value: Mapping[str, Any]) -> Iterator[bytes]:
    """Encode one record without accumulating a second full-page payload."""
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    try:
        for part in encoder.iterencode(value):
            for offset in range(0, len(part), 16 * 1024):
                yield part[offset : offset + 16 * 1024].encode('utf-8')
        yield b'\n'
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ExportError('value is not valid UTF-8 JSON') from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f'invalid JSON constant: {value}')


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _write_all(stream: BinaryIO, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(payload):
        count = stream.write(view[written:])
        if count is None or count <= 0:
            raise OSError('short write while appending event data')
        written += count


class JsonlEventWriter:
    """Serialize raw trace pages with a durable recovery checkpoint."""

    def __init__(
        self,
        output: Path,
        *,
        source: Mapping[str, Any],
        resume: bool = False,
        follow: bool = True,
    ) -> None:
        if resume and not follow:
            raise ExportError('snapshot exports cannot be resumed')
        self._output = Path(output).expanduser().resolve()
        self._checkpoint = self._output.with_name(f'{self._output.name}.checkpoint.json')
        self._manifest = self._output.with_name(f'{self._output.name}.manifest.json')
        self._lock_path = self._output.with_name(f'{self._output.name}.lock')
        self._partial = self._output.with_name(f'{self._output.name}.partial')
        self._source = _canonical_source(source)
        self._resume = resume
        self._follow = follow
        self._io_lock = asyncio.Lock()
        self._lock_descriptor: int | None = None
        self._stream: BinaryIO | None = None
        self._opened = False
        self._closed = False
        self._failed = False
        self._export_id = ''
        self._started_at = ''
        self._last_committed_at = ''
        self._committed_records = 0
        self._committed_bytes = 0
        self._hasher = hashlib.sha256()
        self._cursors: dict[TraceStreamKey, int] = {}
        self._observed: dict[str, Any] = {}

    @property
    def output(self) -> Path:
        return self._output

    @property
    def checkpoint(self) -> Path:
        return self._checkpoint

    @property
    def manifest(self) -> Path:
        return self._manifest

    @property
    def committed_records(self) -> int:
        return self._committed_records

    @property
    def committed_bytes(self) -> int:
        return self._committed_bytes

    @property
    def stream_count(self) -> int:
        return len(self._cursors)

    def cursor_for(self, stream: TraceStreamKey) -> int:
        return self._cursors.get(stream, 0)

    async def open(self) -> None:
        await self._thread_operation(self._open_sync, 'open event output')

    async def commit_page(
        self,
        stream: TraceStreamKey,
        *,
        expected_cursor: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        frozen_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ExportError('event row must be an object')
            frozen_rows.append(dict(row))
        await self._thread_operation(
            lambda: self._commit_page_sync(stream, expected_cursor, frozen_rows),
            'commit event page',
        )

    async def update_observation(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise ExportError('manifest observation must be an object')
        frozen = dict(values)
        _json_bytes(frozen)
        await self._thread_operation(lambda: self._update_observation_sync(frozen), 'update event manifest')

    async def close(self, *, reason: str) -> None:
        if self._closed:
            return
        self._validate_close_reason(reason)
        await self._thread_operation(lambda: self._close_sync(reason), 'close event output')

    async def _thread_operation(self, operation: Callable[[], _T], label: str) -> _T:
        async with self._io_lock:
            task = asyncio.create_task(asyncio.to_thread(operation))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                    continue
                except ExportError:
                    raise
                except Exception as exc:
                    raise ExportError(f'failed to {label}: {type(exc).__name__}') from exc
            if cancelled:
                with suppress(BaseException):
                    task.result()
                raise asyncio.CancelledError
            try:
                return task.result()
            except ExportError:
                raise
            except Exception as exc:
                raise ExportError(f'failed to {label}: {type(exc).__name__}') from exc

    def _open_sync(self) -> None:
        if self._opened:
            raise ExportError('event output is already open')
        if self._closed:
            raise ExportError('event output is closed')
        self._output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._acquire_lock()
            if self._resume:
                self._resume_sync()
            elif self._follow:
                self._create_follow_sync()
            else:
                self._create_snapshot_sync()
            self._opened = True
        except BaseException:
            with suppress(BaseException):
                self._close_handles()
            raise

    def _acquire_lock(self) -> None:
        descriptor = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise ExportError(f'event output is locked: {self._output}') from exc
        self._lock_descriptor = descriptor

    def _create_follow_sync(self) -> None:
        self._reject_existing(self._output, self._checkpoint, self._manifest, self._partial)
        descriptor = os.open(self._output, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, 'r+b')
        self._stream.flush()
        os.fsync(self._stream.fileno())
        _fsync_directory(self._output.parent)
        now = _utc_now()
        self._export_id = f'export_{uuid.uuid4().hex}'
        self._started_at = now
        self._last_committed_at = now
        try:
            _atomic_write_json(self._checkpoint, self._checkpoint_value())
        except OSError as exc:
            raise ExportError('failed to publish initial checkpoint') from exc
        try:
            self._write_manifest('following')
        except OSError as exc:
            raise ExportError('failed to publish initial manifest') from exc

    def _create_snapshot_sync(self) -> None:
        self._reject_existing(self._output, self._checkpoint, self._manifest, self._partial)
        descriptor = os.open(self._partial, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        self._stream = os.fdopen(descriptor, 'r+b')
        self._stream.flush()
        os.fsync(self._stream.fileno())
        _fsync_directory(self._partial.parent)
        now = _utc_now()
        self._export_id = f'export_{uuid.uuid4().hex}'
        self._started_at = now
        self._last_committed_at = now

    @staticmethod
    def _reject_existing(*paths: Path) -> None:
        for path in paths:
            if path.exists():
                raise ExportError(f'refusing to overwrite existing export file: {path}')

    def _resume_sync(self) -> None:
        if not self._checkpoint.exists():
            raise ExportError('resume requires an existing checkpoint')
        if not self._output.exists():
            raise ExportError('resume checkpoint has no data file')
        checkpoint = self._load_checkpoint()
        recovered_hasher, recovered_cursors = self._validate_committed_prefix(checkpoint)
        stream = self._output.open('r+b')
        try:
            os.fchmod(stream.fileno(), 0o600)
            stream.truncate(checkpoint['committed_bytes'])
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(checkpoint['committed_bytes'])
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        self._export_id = checkpoint['export_id']
        self._started_at = checkpoint['started_at']
        self._last_committed_at = checkpoint['last_committed_at']
        self._committed_bytes = checkpoint['committed_bytes']
        self._committed_records = checkpoint['committed_records']
        self._hasher = recovered_hasher
        self._cursors = recovered_cursors
        self._observed = {}
        try:
            self._write_manifest('following')
        except OSError as exc:
            raise ExportError('failed to rebuild manifest') from exc

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            value = json.loads(self._checkpoint.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExportError('checkpoint is unreadable') from exc
        if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
            raise ExportError('checkpoint has an invalid schema')
        if value['version'] != 1 or isinstance(value['version'], bool):
            raise ExportError('checkpoint version is unsupported')
        if value['source'] != self._source:
            raise ExportError('checkpoint source does not match requested source')
        if value['output_path'] != str(self._output):
            raise ExportError('checkpoint output path does not match requested output')
        _nonempty_string(value['export_id'], 'checkpoint export_id')
        _nonempty_string(value['started_at'], 'checkpoint started_at')
        _nonempty_string(value['last_committed_at'], 'checkpoint last_committed_at')
        _plain_integer(value['committed_bytes'], 'checkpoint committed_bytes')
        _plain_integer(value['committed_records'], 'checkpoint committed_records')
        digest = value['committed_sha256']
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ExportError('checkpoint hash is invalid')
        self._checkpoint_cursors(value['streams'])
        return value

    def _checkpoint_cursors(self, value: object) -> dict[TraceStreamKey, int]:
        if not isinstance(value, list):
            raise ExportError('checkpoint cursor map is invalid')
        cursors: dict[TraceStreamKey, int] = {}
        for item in value:
            if not isinstance(item, Mapping) or set(item) != _STREAM_CHECKPOINT_KEYS:
                raise ExportError('checkpoint cursor map is invalid')
            item_mapping = cast(Mapping[str, Any], item)
            run_id = _nonempty_string(item_mapping['run_id'], 'checkpoint stream run_id')
            run_span_id = _nonempty_string(item_mapping['run_span_id'], 'checkpoint stream run_span_id')
            last_record_id = _plain_integer(item_mapping['last_record_id'], 'checkpoint stream cursor', minimum=1)
            try:
                stream = TraceStreamKey(self._source['storage_scope'], self._source['table'], run_id, run_span_id)
            except ValueError as exc:
                raise ExportError('checkpoint cursor map is invalid') from exc
            if stream in cursors:
                raise ExportError('checkpoint cursor map contains duplicate streams')
            self._validate_stream_source(stream)
            cursors[stream] = last_record_id
        return cursors

    def _validate_committed_prefix(self, checkpoint: Mapping[str, Any]) -> tuple[Any, dict[TraceStreamKey, int]]:
        committed_bytes = checkpoint['committed_bytes']
        try:
            size = self._output.stat().st_size
        except OSError as exc:
            raise ExportError('event data file is unreadable') from exc
        if size < committed_bytes:
            raise ExportError('event data file is shorter than its committed prefix')
        expected_cursors = self._checkpoint_cursors(checkpoint['streams'])
        recovered_cursors: dict[TraceStreamKey, int] = {}
        recovered_records = 0
        hasher = hashlib.sha256()
        pending = b''
        remaining = committed_bytes
        try:
            with self._output.open('rb') as stream:
                while remaining:
                    chunk = stream.read(min(_READ_CHUNK_SIZE, remaining))
                    if not chunk:
                        raise ExportError('event data file is shorter than its committed prefix')
                    remaining -= len(chunk)
                    hasher.update(chunk)
                    parts = (pending + chunk).split(b'\n')
                    pending = parts.pop()
                    for line in parts:
                        if not line:
                            raise ExportError('committed event data contains an empty line')
                        row = self._decode_recovery_row(line)
                        stream_key = TraceStreamKey(
                            self._source['storage_scope'], self._source['table'], row['run_id'], row['run_span_id']
                        )
                        cursor = recovered_cursors.get(stream_key, 0)
                        self._validate_page(stream_key, cursor, [row])
                        recovered_cursors[stream_key] = row['record_id']
                        recovered_records += 1
        except OSError as exc:
            raise ExportError('event data file is unreadable') from exc
        if pending:
            raise ExportError('committed event prefix ends with an incomplete line')
        if hasher.hexdigest() != checkpoint['committed_sha256']:
            raise ExportError('committed event prefix hash does not match checkpoint')
        if recovered_records != checkpoint['committed_records']:
            raise ExportError('committed event record count does not match checkpoint')
        if recovered_cursors != expected_cursors:
            raise ExportError('committed event cursors do not match checkpoint')
        return hasher, recovered_cursors

    @staticmethod
    def _decode_recovery_row(line: bytes) -> dict[str, Any]:
        try:
            value = json.loads(line.decode('utf-8'))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ExportError('committed event prefix contains invalid JSON') from exc
        if not isinstance(value, dict):
            raise ExportError('committed event prefix contains a non-object row')
        return value

    def _commit_page_sync(
        self,
        stream: TraceStreamKey,
        expected_cursor: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._require_writable()
        if self._failed:
            raise ExportError('event writer stopped after an I/O failure')
        current_cursor = self._cursors.get(stream, 0)
        _plain_integer(expected_cursor, 'expected cursor')
        if expected_cursor != current_cursor:
            raise ExportError('expected cursor does not match committed cursor')
        self._validate_page(stream, expected_cursor, rows)
        if not rows:
            return
        assert self._stream is not None
        candidate_hasher = self._hasher.copy()
        candidate_bytes = self._committed_bytes
        candidate_records = self._committed_records + len(rows)
        candidate_cursors = {**self._cursors, stream: rows[-1]['record_id']}
        candidate_time = _utc_now()
        try:
            for row in rows:
                for chunk in _json_chunks(row):
                    _write_all(self._stream, chunk)
                    candidate_hasher.update(chunk)
                    candidate_bytes += len(chunk)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            if self._follow:
                checkpoint = self._checkpoint_value(
                    committed_bytes=candidate_bytes,
                    committed_records=candidate_records,
                    hasher=candidate_hasher,
                    cursors=candidate_cursors,
                    last_committed_at=candidate_time,
                )
                try:
                    _atomic_write_json(self._checkpoint, checkpoint)
                except OSError as exc:
                    raise ExportError('failed to publish checkpoint') from exc
            self._hasher = candidate_hasher
            self._committed_bytes = candidate_bytes
            self._committed_records = candidate_records
            self._cursors = candidate_cursors
            self._last_committed_at = candidate_time
            if self._follow:
                try:
                    self._write_manifest('following')
                except OSError as exc:
                    raise ExportError('failed to publish manifest after checkpoint') from exc
        except BaseException:
            self._failed = True
            raise

    def _validate_page(
        self,
        stream: TraceStreamKey,
        expected_cursor: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._validate_stream_source(stream)
        cursor = expected_cursor
        for row in rows:
            if set(row) != _EVENT_COLUMN_SET:
                raise ExportError('event row columns do not match the raw event schema')
            _plain_integer(row['id'], 'event id', minimum=1)
            record_id = _plain_integer(row['record_id'], 'event record_id', minimum=1)
            _plain_integer(row['time_unix_nano'], 'event time_unix_nano')
            if record_id <= cursor:
                raise ExportError('event record_id must increase above the expected cursor')
            if row['run_id'] != stream.run_id or row['run_span_id'] != stream.run_span_id:
                raise ExportError('event row identity does not match its stream')
            for field in ('run_id', 'run_span_id', 'event_type', 'trace_id', 'span_id'):
                _nonempty_string(row[field], f'event {field}')
            for field in ('parent_execution_id', 'parent_span_id', 'created_at'):
                if row[field] is not None and not isinstance(row[field], str):
                    raise ExportError(f'event {field} must be a string or null')
            event_data = row['event_data']
            if not isinstance(event_data, str):
                raise ExportError('event event_data must be a JSON object string')
            try:
                parsed_event_data = json.loads(event_data, parse_constant=_reject_json_constant)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ExportError('event event_data is invalid JSON') from exc
            if not isinstance(parsed_event_data, dict):
                raise ExportError('event event_data must encode a JSON object')
            if self._source['spec_type'] == 'nexus_batch':
                if row['parent_execution_id'] != self._source['execution_id']:
                    raise ExportError('event row parent does not match the Batch execution')
            elif row['run_id'] != self._source['execution_id']:
                raise ExportError('event row run_id does not match the requested execution')
            for _ in _json_chunks(row):
                pass
            cursor = record_id

    def _validate_stream_source(self, stream: TraceStreamKey) -> None:
        if not isinstance(stream, TraceStreamKey):
            raise ExportError('event stream identity is invalid')
        if stream.storage_scope != self._source['storage_scope'] or stream.table != self._source['table']:
            raise ExportError('event stream does not belong to the requested source')
        if self._source['spec_type'] != 'nexus_batch' and stream.run_id != self._source['execution_id']:
            raise ExportError('event stream does not belong to the requested execution')

    def _checkpoint_value(
        self,
        *,
        committed_bytes: int | None = None,
        committed_records: int | None = None,
        hasher: Any | None = None,
        cursors: Mapping[TraceStreamKey, int] | None = None,
        last_committed_at: str | None = None,
    ) -> dict[str, Any]:
        selected_cursors = self._cursors if cursors is None else cursors
        return {
            'version': 1,
            'export_id': self._export_id,
            'started_at': self._started_at,
            'last_committed_at': self._last_committed_at if last_committed_at is None else last_committed_at,
            'source': self._source,
            'output_path': str(self._output),
            'committed_bytes': self._committed_bytes if committed_bytes is None else committed_bytes,
            'committed_records': self._committed_records if committed_records is None else committed_records,
            'committed_sha256': (self._hasher if hasher is None else hasher).hexdigest(),
            'streams': [
                {'run_id': stream.run_id, 'run_span_id': stream.run_span_id, 'last_record_id': cursor}
                for stream, cursor in sorted(selected_cursors.items(), key=lambda item: item[0].as_tuple())
            ],
        }

    def _manifest_value(self, status: str) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'export_id': self._export_id,
            'source': self._source,
            'mode': 'follow' if self._follow else 'snapshot',
            'status': status,
            'started_at': self._started_at,
            'last_committed_at': self._last_committed_at,
            'committed_records': self._committed_records,
            'committed_bytes': self._committed_bytes,
            'stream_count': len(self._cursors),
            'observed': self._observed,
        }

    def _write_manifest(self, status: str) -> None:
        _atomic_write_json(self._manifest, self._manifest_value(status))

    def _update_observation_sync(self, values: Mapping[str, Any]) -> None:
        self._require_writable()
        if self._failed:
            raise ExportError('event writer stopped after an I/O failure')
        self._observed.update(values)
        if self._follow:
            try:
                self._write_manifest('following')
            except OSError as exc:
                self._failed = True
                raise ExportError('failed to publish manifest observation') from exc

    def _validate_close_reason(self, reason: str) -> None:
        allowed = {'interrupted', 'error'} if self._follow else {'snapshot', 'interrupted', 'error'}
        if reason not in allowed:
            raise ExportError(f'invalid close reason for export mode: {reason}')

    def _close_sync(self, reason: str) -> None:
        if self._closed:
            return
        if not self._opened:
            self._closed = True
            self._close_handles()
            return
        error: BaseException | None = None
        try:
            assert self._stream is not None
            self._stream.flush()
            os.fsync(self._stream.fileno())
            if self._follow:
                status = 'error' if self._failed or reason == 'error' else 'interrupted'
                try:
                    self._write_manifest(status)
                except OSError as exc:
                    error = ExportError('failed to publish final manifest')
                    error.__cause__ = exc
            elif reason == 'snapshot' and not self._failed:
                self._stream.close()
                self._stream = None
                try:
                    os.replace(self._partial, self._output)
                    _fsync_directory(self._output.parent)
                except OSError as exc:
                    error = ExportError('failed to publish snapshot data')
                    error.__cause__ = exc
                if error is None:
                    try:
                        self._write_manifest('snapshot')
                    except OSError as exc:
                        error = ExportError('failed to publish snapshot manifest')
                        error.__cause__ = exc
        except BaseException as exc:
            error = exc
        finally:
            self._closed = True
            try:
                self._close_handles()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def _require_writable(self) -> None:
        if not self._opened:
            raise ExportError('event output is not open')
        if self._closed:
            raise ExportError('event output is closed')

    def _close_handles(self) -> None:
        error: BaseException | None = None
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                error = exc
        lock_descriptor = self._lock_descriptor
        self._lock_descriptor = None
        if lock_descriptor is not None:
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            except BaseException as exc:
                if error is None:
                    error = exc
            try:
                os.close(lock_descriptor)
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error


__all__ = ['JsonlEventWriter']


class ResultSpillFile:
    """A sibling temp file holding one prefetched response body.

    The depth-2 results pipeline keeps the next record's HTTP body on disk
    instead of in RAM, so at most one record is materialized at a time. Files
    live next to the JSONL partial (never /tmp: it is frequently tmpfs, which
    would silently put the payload back into memory) and are removed as soon
    as they are consumed or the export aborts.
    """

    def __init__(self, output: Path, seq: int) -> None:
        self.path = output.with_name(f'{output.name}.prefetch-{seq}')
        self.missing = False
        self.prepared: Any = None
        self.scan_fallback = False

    async def download(self, api: Any, execution_id: str, scan_factory: Any | None = None) -> None:
        """Stream the record's RunOutput body from `api` into this file.

        With ``scan`` (方案 D), the body is validated/parsed incrementally during
        the download; ``self.prepared`` carries the scan result and
        ``self.scan_fallback`` marks bodies needing the legacy full-parse path.
        """
        descriptor = -1
        scan = scan_factory() if scan_factory else None
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, 'wb') as fileobj:
                descriptor = -1
                length = await api.get_result_to_file(execution_id, fileobj, scan=scan)
                if scan is not None:
                    if length is None:
                        scan.abort()  # 404: nothing scanned, keep legacy missing-row semantics
                        self.prepared = None
                    else:
                        self.prepared = scan.finish()
                        self.scan_fallback = scan.fallback
                        if self.scan_fallback:
                            self.prepared = None  # legacy orjson path handles this body
                else:
                    self.prepared = None
        except ExportError:
            if scan is not None:
                with suppress(Exception):
                    scan.abort()
            raise
        except OSError as exc:
            if scan is not None:
                with suppress(Exception):
                    scan.abort()
            self.discard()
            raise ExportError(f'cannot write prefetch output {self.path}: {exc}') from exc
        except BaseException:
            # Cancellation and other control-flow exceptions: abort the scan
            # thread so it cannot linger blocked on its queue.
            if scan is not None:
                with suppress(Exception):
                    scan.abort()
            raise
        finally:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
        self.missing = length is None

    def load_raw(self) -> bytes | None:
        """Raw response body, or None when the server answered 404."""
        if self.missing:
            return None
        return self.path.read_bytes()

    @contextmanager
    def open_view(self) -> Iterator[bytes | memoryview | None]:
        """Expose the spilled body without copying it into a Python ``bytes``.

        ``orjson`` accepts a memoryview directly. The stdlib fallback copies only
        when it is actually needed, preserving compatibility for lone surrogates
        and other inputs rejected by orjson.
        """
        if self.missing:
            yield None
            return
        with self.path.open('rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            if size == 0:
                yield b''
                return
            mapped = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ)
            view = memoryview(mapped)
            try:
                yield view
            finally:
                view.release()
                mapped.close()

    def discard(self) -> None:
        with suppress(OSError):
            self.path.unlink(missing_ok=True)


class ResultSpillPool:
    """Hands out sequentially numbered spill files for one export run."""

    def __init__(self, output: Path) -> None:
        self._output = output
        self._seq = 0
        self._live: list[ResultSpillFile] = []

    def new_file(self) -> ResultSpillFile:
        self._seq += 1
        file = ResultSpillFile(self._output, self._seq)
        self._live.append(file)
        return file

    def cleanup(self) -> None:
        """Discard every unconsumed file (idempotent; consumed files are gone)."""
        for file in self._live:
            file.discard()
        self._live.clear()

    @staticmethod
    def sweep(output: Path) -> None:
        """Remove prefetch leftovers from a previous crashed run of `output`."""
        for leftover in output.parent.glob(f'{output.name}.prefetch-*'):
            with suppress(OSError):
                leftover.unlink()


class AtomicJsonlOutput:
    def __init__(self, output: Path, *, force: bool) -> None:
        self.output = output.expanduser().resolve(strict=False)
        self.partial = self.output.with_name(f'{self.output.name}.partial')
        self.force = force
        self._stream: BinaryIO | None = None
        self._validated = False
        self._index: list[_IndexEntry] = []
        self._index_ready = False
        self._digest = hashlib.sha256()
        self._offset = 0
        self._records = 0
        self._first_error: str | None = None
        self._lock_fd: int | None = None

    @property
    def index_available(self) -> bool:
        """Whether parse-free validation/filtering can be used for this output."""
        return self._index_ready

    @property
    def lock_path(self) -> Path:
        return self.partial.with_name(f'{self.partial.name}.lock')

    def _acquire_partial_lock(self, path: Path) -> int:
        """Take an exclusive advisory lock on `path`; raise if another writer is live.

        The lock file records the owning process so the conflict error can point at
        it. The kernel releases the lock when the process dies, which is what lets a
        retry after a crash take over the stale partial without manual cleanup.
        """
        import fcntl
        import socket

        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            holder = {}
            with suppress(OSError, json.JSONDecodeError):
                holder = json.loads(path.read_text(encoding='utf-8'))
            raise ExportError(
                f'another export is actively writing {self.output} '
                f'(pid {holder.get("pid")} on {holder.get("host", "unknown host")}, '
                f'started {holder.get("started_at", "unknown time")}). '
                'Use a different --output, or terminate that process.'
            ) from exc
        payload = json.dumps(
            {
                'pid': os.getpid(),
                'host': socket.gethostname(),
                'started_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            },
            ensure_ascii=False,
        )
        os.truncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload.encode('utf-8'))
        return fd

    def _release_partial_lock(self) -> None:
        lock_fd = self._lock_fd
        if lock_fd is not None:
            # Remove only the pathname for the inode we actually locked. Do
            # this while holding the lock; closing first could let a new writer
            # acquire the inode and then have its lock file unlinked by us.
            with suppress(OSError):
                lock_stat = os.stat(self.lock_path)
                fd_stat = os.fstat(lock_fd)
                if (lock_stat.st_dev, lock_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino):
                    self.lock_path.unlink(missing_ok=True)
            with suppress(OSError):
                os.close(lock_fd)
            self._lock_fd = None

    async def open(self) -> None:
        if self._stream is not None:
            raise ExportError('JSONL output is already open')
        self._stream = await asyncio.to_thread(self._open_exclusive)
        self._index_ready = True

    def _open_exclusive(self) -> BinaryIO:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.force:
            raise ExportError(f'output already exists; pass --force to replace it: {self.output}')
        # Acquire the lock before removing stale data. In particular, --force
        # must not unlink an active writer's lock and then race it on the same
        # partial file.
        lock_fd = self._acquire_partial_lock(self.lock_path)
        self._lock_fd = lock_fd
        try:
            if self.force:
                self.partial.unlink(missing_ok=True)
            # Leftover prefetch files from a crashed run are safe to remove now
            # that this writer owns the output lock.
            ResultSpillPool.sweep(self.output)
            stale_size = self.partial.stat().st_size if self.partial.exists() else 0
            stream = self.partial.open('wb')
        except OSError as exc:
            self._release_partial_lock()
            raise ExportError(f'cannot open partial output {self.partial}: {exc}') from exc
        if stale_size:
            # A dead writer left bytes behind; we hold the lock, so reclaiming is safe.
            logger.warning(
                'reclaiming stale partial %s (%d bytes from a previous interrupted export)',
                self.partial,
                stale_size,
            )
        return stream

    async def write_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        if self._stream is None:
            raise ExportError('JSONL output is not open')
        try:
            await to_thread_complete(self._write_records, records)
        except asyncio.CancelledError:
            # Interruption (Ctrl-C): stop holding the lock but keep the partial for
            # diagnosis; a retry takes over the stale partial automatically because
            # to_thread_complete has already waited for the worker thread.
            self._release_partial_lock()
            raise

    def _write_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        from nexus.utils.json_stream import iter_json_bytes

        assert self._stream is not None
        try:
            for record in records:
                record = dict(record)
                self._records += 1
                if self._first_error is None:
                    self._first_error = _check_record_semantics(record, self._records)
                length = 0
                for chunk in iter_json_bytes(record, sort_keys=False, allow_nan=False, separators=(', ', ': ')):
                    self._stream.write(chunk)
                    self._digest.update(chunk)
                    length += len(chunk)
                self._stream.write(b'\n')
                self._digest.update(b'\n')
                length += 1
                self._account_row(
                    length,
                    run_id=record.get('run_id'),
                    status=record.get('status'),
                    retry_count=record.get('retry_count'),
                    has_artifacts=_has_artifacts(record),
                )
        except (TypeError, ValueError) as exc:
            raise ExportError('JSONL record is not serializable') from exc

    async def write_row_chunks(
        self,
        chunks: Iterator[bytes],
        *,
        run_id: Any,
        status: Any,
        retry_count: Any,
        has_artifacts: bool,
    ) -> None:
        """Write one pre-serialized row from an iterable of byte chunks.

        Streaming twin of ``write_records`` for rows assembled without a
        materialized dict (the transcode path): the chunk iterable — including
        any file reads it performs — is consumed in the worker thread, and
        digest/offset/index accounting is identical to ``write_records``. Field
        set and order are guaranteed by construction at the call site, so the
        deferred semantic check does not apply.
        """
        if self._stream is None:
            raise ExportError('JSONL output is not open')
        try:
            await to_thread_complete(
                self._write_row_chunks,
                chunks,
                run_id=run_id,
                status=status,
                retry_count=retry_count,
                has_artifacts=has_artifacts,
            )
        except asyncio.CancelledError:
            # Interruption (Ctrl-C): stop holding the lock but keep the partial for
            # diagnosis; a retry takes over the stale partial automatically because
            # to_thread_complete has already waited for the worker thread.
            self._release_partial_lock()
            raise

    def _write_row_chunks(
        self,
        chunks: Iterator[bytes],
        *,
        run_id: Any,
        status: Any,
        retry_count: Any,
        has_artifacts: bool,
    ) -> None:
        assert self._stream is not None
        self._records += 1
        length = 0
        try:
            for chunk in chunks:
                self._stream.write(chunk)
                self._digest.update(chunk)
                length += len(chunk)
            self._stream.write(b'\n')
            self._digest.update(b'\n')
            length += 1
        except (TypeError, ValueError) as exc:
            raise ExportError('JSONL record is not serializable') from exc
        self._account_row(length, run_id=run_id, status=status, retry_count=retry_count, has_artifacts=has_artifacts)

    def _account_row(self, length: int, *, run_id: Any, status: Any, retry_count: Any, has_artifacts: bool) -> None:
        if self._index_ready:
            if len(self._index) >= _MAX_INDEX_ENTRIES:
                logger.warning(
                    'JSONL export exceeds %d records; dropping the write-time index and '
                    'falling back to parse-per-line validation and filtering',
                    _MAX_INDEX_ENTRIES,
                )
                self._index = []
                self._index_ready = False
            else:
                self._index.append(
                    _IndexEntry(
                        run_id=run_id,
                        status=status,
                        retry_count=retry_count,
                        offset=self._offset,
                        length=length,
                        has_artifacts=has_artifacts,
                    )
                )
        self._offset += length

    async def close_and_validate(self, *, expected_count: int, require_sorted: bool, strict: bool = False) -> None:
        if self._stream is None:
            raise ExportError('JSONL output is not open')
        stream = self._stream
        self._stream = None
        await asyncio.to_thread(_flush_fsync_close, stream)
        if strict or not self._index_ready:
            await asyncio.to_thread(_validate_jsonl_file, self.partial, expected_count, require_sorted)
        else:
            if self._first_error is not None:
                # Deferred from write time to keep the established close-time contract.
                raise ExportError(self._first_error)
            await asyncio.to_thread(
                _verify_integrity,
                self.partial,
                self._index,
                self._digest.hexdigest(),
                expected_count,
                require_sorted,
            )
        self._validated = True

    async def filter_records(self, keep: Callable[[Mapping[str, Any]], bool]) -> int:
        """Filter completed output, preserving retained lines exactly as written.

        Legacy parse-per-line path: `keep` receives each fully parsed record.
        Rewriting the partial invalidates the write-time index — afterwards only
        publish()/abort() (or another legacy filter) may run on this output.
        """
        if not self._validated:
            raise ExportError('JSONL output has not been validated')
        # Invalidate before touching the file: if filtering fails half-way the
        # index must not be trusted either.
        self._index = []
        self._index_ready = False
        filtering = asyncio.create_task(asyncio.to_thread(self._filter_records, keep))
        try:
            return await asyncio.shield(filtering)
        except asyncio.CancelledError:
            # Match write_records: wait before aborting so the worker can finish cleanup.
            await filtering
            raise

    async def filter_records_by_index(self, keep: Callable[[_IndexEntry], bool]) -> int:
        """Filter completed output using write-time bookkeeping; no JSON parsing.

        `keep` receives each _IndexEntry (run_id/status/retry_count/has_artifacts/...).
        Retained lines are copied byte-for-byte; the replacement file is re-verified
        against the retained index and its own sha256.
        """
        if not self._validated:
            raise ExportError('JSONL output has not been validated')
        if not self._index_ready:
            raise ExportError('JSONL index is unavailable; use the legacy filter_records path')
        filtering = asyncio.create_task(asyncio.to_thread(self._filter_records_by_index, keep))
        try:
            return await asyncio.shield(filtering)
        except asyncio.CancelledError:
            await filtering
            raise

    def _filter_records_by_index(self, keep: Callable[[_IndexEntry], bool]) -> int:
        temporary = None
        try:
            retained_digest = hashlib.sha256()
            retained: list[_IndexEntry] = []
            with (
                self.partial.open('rb') as source,
                tempfile.NamedTemporaryFile(
                    mode='wb', prefix=f'.{self.partial.name}.', suffix='.tmp', dir=self.partial.parent, delete=False
                ) as target,
            ):
                temporary = Path(target.name)
                retained_count = 0
                next_offset = 0
                source_offset = 0
                for i, entry in enumerate(self._index):
                    # The partial is always written (and re-written here) with no gaps,
                    # so every entry must start exactly where the previous one ended.
                    if entry.offset != source_offset:
                        raise ExportError(f'JSONL index is not contiguous at line {i + 1}')
                    source_offset += entry.length
                    keep_it = keep(entry)  # evaluated exactly once per record
                    source.seek(entry.offset)
                    remaining = entry.length
                    copied = 0
                    while remaining > 0:
                        block = source.read(min(remaining, _READ_BLOCK))
                        if not block:
                            raise ExportError(f'JSONL line {i + 1} is shorter than indexed')
                        if keep_it:
                            target.write(block)
                            retained_digest.update(block)
                            copied += len(block)
                        remaining -= len(block)
                    if keep_it:
                        # Rewrite the offset/length to the retained file's layout so the
                        # post-filter integrity check can seek correctly.
                        retained.append(entry._replace(offset=next_offset, length=copied))
                        retained_count += 1
                        next_offset += copied
                target.flush()
                os.fsync(target.fileno())
            expected_hex = retained_digest.hexdigest()
            _verify_integrity(temporary, retained, expected_hex, retained_count, require_sorted=False)
            os.replace(temporary, self.partial)
            _result_fsync_directory(self.partial.parent)
            self._index = retained
            self._digest = retained_digest  # hashes exactly the retained bytes, in order
            self._offset = next_offset
            self._records = retained_count
            return retained_count
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _filter_records(self, keep: Callable[[Mapping[str, Any]], bool]) -> int:
        temporary = None
        try:
            with (
                self.partial.open('rb') as source,
                tempfile.NamedTemporaryFile(
                    mode='wb', prefix=f'.{self.partial.name}.', suffix='.tmp', dir=self.partial.parent, delete=False
                ) as target,
            ):
                temporary = Path(target.name)
                retained = 0
                for line in source:
                    if keep(json.loads(line)):
                        target.write(line)
                        retained += 1
                target.flush()
                os.fsync(target.fileno())
            _validate_jsonl_file(temporary, retained, require_sorted=False)
            os.replace(temporary, self.partial)
            _result_fsync_directory(self.partial.parent)
            return retained
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def publish(self) -> None:
        if not self._validated:
            raise ExportError('JSONL output has not been validated')
        await asyncio.to_thread(self._publish_sync)
        self._release_partial_lock()

    def _publish_sync(self) -> None:
        if self.output.exists() and not self.force:
            raise ExportError(f'output appeared during export: {self.output}')
        os.replace(self.partial, self.output)
        _result_fsync_directory(self.output.parent)

    async def abort(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            await asyncio.to_thread(_flush_fsync_close, stream)
        # Give up the lock but keep the partial file for diagnosis; a retry can
        # then take it over without manual cleanup. The write-time bookkeeping is
        # void after this point, so drop it to prevent post-abort misuse.
        self._index = []
        self._index_ready = False
        self._release_partial_lock()
        ResultSpillPool.sweep(self.output)


class AtomicJsonOutput:
    def __init__(self, output: Path, *, force: bool) -> None:
        self.output = output.expanduser().resolve(strict=False)
        self.partial = self.output.with_name(f'{self.output.name}.partial')
        self.force = force
        self._prepared = False

    async def prepare(self, value: Mapping[str, Any]) -> None:
        if self._prepared:
            raise ExportError('JSON output is already prepared')
        await asyncio.to_thread(self._prepare_sync, value)
        self._prepared = True

    def _prepare_sync(self, value: Mapping[str, Any]) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.force:
            raise ExportError(f'output already exists; pass --force to replace it: {self.output}')
        try:
            content = json.dumps(
                dict(value),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        except (TypeError, ValueError) as exc:
            raise ExportError('Batch metadata is not JSON serializable') from exc
        if self.force:
            self.partial.unlink(missing_ok=True)
        try:
            with self.partial.open('x', encoding='utf-8') as stream:
                stream.write(f'{content}\n')
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise ExportError(f'partial output already exists: {self.partial}') from exc

    async def publish(self) -> None:
        if not self._prepared:
            raise ExportError('JSON output has not been prepared')
        await asyncio.to_thread(self._publish_sync)

    def _publish_sync(self) -> None:
        if self.output.exists() and not self.force:
            raise ExportError(f'output appeared during export: {self.output}')
        os.replace(self.partial, self.output)
        _result_fsync_directory(self.output.parent)

    async def abort(self) -> None:
        return None


def _flush_fsync_close(stream: IO[Any]) -> None:
    try:
        stream.flush()
        os.fsync(stream.fileno())
    finally:
        stream.close()


def _validate_jsonl_file(path: Path, expected_count: int, require_sorted: bool) -> None:
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 0:
        raise ExportError('invalid expected JSONL record count')
    count = 0
    previous_run_id: str | None = None
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExportError(f'invalid JSONL at line {line_number}') from exc
            if not isinstance(record, dict):
                raise ExportError(f'JSONL line {line_number} is not an object')
            if tuple(record) not in (_JSONL_FIELDS, (*_JSONL_FIELDS, 'extra')):
                raise ExportError(f'unexpected JSONL fields at line {line_number}')
            if 'extra' in record and not isinstance(record['extra'], dict):
                raise ExportError(f'invalid extra at line {line_number}')
            run_id = record['run_id']
            if not isinstance(run_id, str) or not run_id:
                raise ExportError(f'invalid run_id at line {line_number}')
            if require_sorted and previous_run_id is not None and run_id <= previous_run_id:
                raise ExportError(f'JSONL run_id order is not strictly increasing at line {line_number}')
            previous_run_id = run_id
            if record['status'] not in _LOCAL_TERMINAL_STATUSES:
                raise ExportError(f'invalid status at line {line_number}')
            retry_count = record['retry_count']
            if isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count < 0:
                raise ExportError(f'invalid retry_count at line {line_number}')
            count += 1
    if count != expected_count:
        raise ExportError(f'JSONL record count mismatch: expected {expected_count}, got {count}')


def _result_fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _unlink_and_fsync(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _result_fsync_directory(path.parent)
