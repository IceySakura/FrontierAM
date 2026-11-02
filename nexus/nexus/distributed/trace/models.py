"""Cloud trace identities and transport models."""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nexus.distributed.errors import DistributedDomainError
from nexus.runner.trace.tdsql.schema import PRINCIPAL_TRACE_TABLE_PATTERN, TraceSchemaDialect


_PRINCIPAL_STORAGE_SCOPE = 'principal_v1'
_LEGACY_BATCH_STORAGE_SCOPE = 'batch_principal_v1'


@dataclass(frozen=True, slots=True)
class TraceStreamKey:
    """Stable identity for one persisted trace event stream."""

    storage_scope: str
    table: str
    run_id: str
    run_span_id: str

    def __post_init__(self) -> None:
        storage_scope = self.storage_scope
        if storage_scope == _LEGACY_BATCH_STORAGE_SCOPE:
            storage_scope = _PRINCIPAL_STORAGE_SCOPE
            object.__setattr__(self, 'storage_scope', storage_scope)
        if storage_scope != _PRINCIPAL_STORAGE_SCOPE:
            raise ValueError(f'unsupported trace storage scope: {storage_scope!r}')
        if not isinstance(self.table, str) or PRINCIPAL_TRACE_TABLE_PATTERN.fullmatch(self.table) is None:
            raise ValueError(f'invalid trace table: {self.table!r}')
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError('run id must not be empty')
        if not isinstance(self.run_span_id, str) or not self.run_span_id:
            raise ValueError('run span id must not be empty')

    def as_tuple(self) -> tuple[str, str, str, str]:
        """Return the delimiter-free stream identity."""
        return (self.storage_scope, self.table, self.run_id, self.run_span_id)


Identity = Annotated[str, Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.:-]+$')]
BackendName = Annotated[str, Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')]
TraceReadMode = Literal['raw', 'preview']
# The schema module's own alias, so a new dialect cannot be accepted by a reader config
# while the writer that has to produce it knows nothing about the name.
TraceDialect = TraceSchemaDialect


class TraceSourceIdentity(BaseModel):
    """Stable dataset identity stored alongside its writer registration."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    source_id: Identity
    source_revision: Identity
    storage_scope: Literal['principal_v1'] = 'principal_v1'


class TraceReaderConfig(TraceSourceIdentity):
    backend: BackendName
    adapter: Literal['tdsql_principal_v1'] = 'tdsql_principal_v1'
    environment_ids: tuple[Identity, ...] = Field(min_length=1, max_length=100)
    host: str = Field(min_length=1, max_length=255)
    port: int = Field(default=3306, ge=1, le=65535)
    database: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_]+$')
    credential_ref: Identity
    # Boundless tables are range-partitioned on parent_created_at and every read must carry
    # that key, so the reader has to know which schema the dataset uses.
    dialect: TraceDialect = 'tdsql'
    display_name: str | None = Field(default=None, max_length=128)
    legacy_from: datetime | None = None
    legacy_until: datetime | None = None

    @model_validator(mode='after')
    def validate_legacy_window(self) -> TraceReaderConfig:
        if (self.legacy_from is None) != (self.legacy_until is None):
            raise ValueError('legacy trace mapping requires both time bounds')
        if self.legacy_from is not None and self.legacy_until is not None:
            if self.legacy_from.tzinfo is None or self.legacy_until.tzinfo is None:
                raise ValueError('legacy trace mapping requires UTC offsets')
            if self.legacy_from >= self.legacy_until:
                raise ValueError('legacy trace mapping time bounds are reversed')
        return self


class TraceReadConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    enabled: bool = False
    backend_allowlist: tuple[BackendName, ...] = ()
    readers: tuple[TraceReaderConfig, ...] = Field(default=(), max_length=100)
    cursor_current_key_id: Identity = 'default'
    cursor_ttl_seconds: int = Field(default=86400, ge=1, le=604800)
    max_concurrent_queries: int = Field(default=8, ge=1, le=128)
    max_principal_queries: int = Field(default=2, ge=1, le=16)
    queue_timeout_seconds: float = Field(default=0.1, gt=0, le=1)
    query_timeout_seconds: float = Field(default=5, gt=0, le=30)
    request_timeout_seconds: float = Field(default=10, gt=0, le=60)
    page_target_bytes: int = Field(default=2 * 1024 * 1024, ge=1024, le=8 * 1024 * 1024)
    max_event_bytes: int = Field(default=8 * 1024 * 1024, ge=1024, le=8 * 1024 * 1024)
    max_assignments: int = Field(default=1000, ge=1, le=1000)
    assignment_page_size: int = Field(default=100, ge=1, le=100)

    @model_validator(mode='after')
    def validate_budgets(self) -> TraceReadConfig:
        if self.page_target_bytes > self.max_event_bytes:
            raise ValueError('trace page target exceeds the event hard limit')
        if self.query_timeout_seconds + 3 > self.request_timeout_seconds:
            raise ValueError('trace request budget must reserve three seconds for query cleanup')
        identities = [
            (r.backend, r.source_id, r.source_revision, env) for r in self.readers for env in r.environment_ids
        ]
        if len(identities) != len(set(identities)):
            raise ValueError('duplicate trace reader mapping')
        return self


class TraceReadError(DistributedDomainError):
    def __init__(self, code: str, status: int, *, retryable: bool = False, detail: str | None = None) -> None:
        super().__init__(code, retryable=retryable, detail=detail)
        self.status = status


@dataclass(frozen=True, slots=True)
class AuthorizedTraceTarget:
    """Internal result of a fresh, full execution:read authorization check."""

    principal_id: str
    execution_id: str
    owner_principal_id: str
    environment_id: str
    execution_status: str
    current_assignment_id: str | None
    driver_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AssignmentTraceMetadata:
    assignment_id: str
    attempt_id: str
    attempt_no: int
    status: str
    created_at: datetime
    runtime_info: dict[str, Any] | None


@dataclass(slots=True)
class TraceReadDiagnostics:
    execution_id: str | None = None
    principal_id: str | None = None
    owner_principal_id: str | None = None
    backend: str | None = None
    mode: TraceReadMode | None = None
    omitted_records: int = 0
    omitted_bytes: int = 0
    queries: int = 0
    query_seconds: float = 0
    encoding_seconds: float = 0


RAW_EVENT_FIELDS = (
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


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')


def encode_raw_event(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the raw payload and encode database integers without browser precision loss."""
    if set(record) != set(RAW_EVENT_FIELDS):
        raise ValueError('invalid raw event fields')
    result = dict(record)
    for name in ('id', 'record_id', 'time_unix_nano'):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('invalid raw event integer')
        result[name] = str(value)
    for name in set(RAW_EVENT_FIELDS) - {'id', 'record_id', 'time_unix_nano'}:
        if record[name] is not None and not isinstance(record[name], str):
            raise ValueError('invalid raw event string')
    if record['event_type'] not in {'start', 'update', 'end'}:
        raise ValueError('invalid raw event type')
    return result


def encode_preview_event(record: dict[str, Any]) -> dict[str, Any]:
    """Expose a checked database omission without changing the original column values."""
    if set(record) != {*RAW_EVENT_FIELDS, 'event_data_bytes', 'event_data_omitted'}:
        raise ValueError('invalid preview event fields')
    size = record['event_data_bytes']
    omitted = record['event_data_omitted']
    if type(size) is not int or size < 0 or type(omitted) is not bool:
        raise ValueError('invalid preview event metadata')
    if (omitted and record['event_data'] is not None) or (not omitted and not isinstance(record['event_data'], str)):
        raise ValueError('invalid preview event body')
    result = encode_raw_event({name: record[name] for name in RAW_EVENT_FIELDS})
    return {**result, 'event_data_bytes': size, 'event_data_omitted': omitted}
