from __future__ import annotations

import asyncio
import json
import re

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from nexus.distributed.config import ZhiyanLogQueryConfig


SOURCE_NONE = 'none'
SOURCE_INSTANCE_MANAGER = 'instance_manager'
SOURCE_ZHIYAN = 'zhiyan'

AVAILABILITY_NOT_STARTED = 'not_started'
AVAILABILITY_READY = 'ready'
AVAILABILITY_INGEST_PENDING = 'ingest_pending'
AVAILABILITY_UNAVAILABLE = 'unavailable'

ERROR_MANAGER_NOT_STARTED = 'manager_log_not_started'
ERROR_MANAGER_UNAVAILABLE = 'manager_log_unavailable'
ERROR_ZHIYAN_NOT_CONFIGURED = 'zhiyan_log_not_configured'
ERROR_ZHIYAN_INGEST_PENDING = 'zhiyan_log_ingest_pending'
ERROR_ZHIYAN_QUERY_FAILED = 'zhiyan_log_query_failed'
ERROR_ZHIYAN_QUERY_LIMITED = 'zhiyan_log_query_limited'

_LINE_RE = re.compile(r'^(?P<timestamp>\S+)\s+(?P<level>[A-Za-z]+)\s+(?P<message>.*)$')
_PIPE_LINE_RE = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\|\s+'
    r'(?P<level>[A-Za-z]+)\s+\|\s+(?P<logger>[^|]*)\s+\|\s+(?P<message>.*)$'
)
_SAFE_FIELDS = frozenset({'component', 'manager_id', 'assignment_id', 'attempt_id', 'execution_id', 'run_id'})
_MAX_MESSAGE_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ExecutionLogLine:
    timestamp: datetime | None
    level: str
    message: str
    logger: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    exception: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ExecutionLogPayload:
    execution_id: str
    execution_status: str
    source: str
    availability: str
    partial: bool
    lines: list[ExecutionLogLine]
    fetched_at: datetime
    retry_after_ms: int | None = None
    error_code: str | None = None


class ZhiyanLogError(RuntimeError):
    def __init__(self, error_code: str, *, status: int | None = None) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.status = status


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value) / (1000 if value > 10_000_000_000 else 1), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _line_from_record(record: Any) -> ExecutionLogLine | None:
    if not isinstance(record, dict):
        return None
    timestamp = _parse_timestamp(
        record.get('timestamp', record.get('time', record.get('ts', record.get('created_at'))))
    )
    if timestamp is None:
        return None
    message = record.get('message', record.get('msg', record.get('log', '')))
    if not isinstance(message, str):
        message = str(message) if message is not None else ''
    message = message[:_MAX_MESSAGE_BYTES]
    level = record.get('level', 'INFO')
    level = level.upper() if isinstance(level, str) else 'INFO'
    logger = record.get('logger')
    logger = logger[:_MAX_MESSAGE_BYTES] if isinstance(logger, str) else None
    fields: dict[str, str] = {}
    nested = record.get('fields')
    if isinstance(nested, dict):
        for key in _SAFE_FIELDS:
            value = nested.get(key)
            if isinstance(value, (str, int, float, bool)):
                fields[key] = str(value)
    for key in _SAFE_FIELDS:
        if key in fields:
            continue
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)):
            fields[key] = str(value)
    exception = None
    raw_exception = record.get('exception')
    if isinstance(raw_exception, dict):
        safe_exception = {
            key: value[:_MAX_MESSAGE_BYTES]
            for key in ('type', 'message', 'traceback')
            if isinstance((value := raw_exception.get(key)), str)
        }
        exception = safe_exception or None
    return ExecutionLogLine(
        timestamp=timestamp,
        level=level[:16],
        message=message,
        logger=logger,
        fields=fields,
        exception=exception,
    )


def parse_manager_log_text(
    text: str, *, max_bytes: int = 1024 * 1024, max_lines: int = 1000
) -> tuple[list[ExecutionLogLine], bool]:
    encoded = text.encode('utf-8', errors='replace')
    partial = len(encoded) > max_bytes
    bounded = encoded[:max_bytes].decode('utf-8', errors='replace')
    if partial and '\n' in bounded:
        bounded = bounded[: bounded.rfind('\n')]
    result: list[ExecutionLogLine] = []
    for raw in bounded.splitlines():
        if len(result) >= max_lines:
            partial = True
            break
        try:
            record = json.loads(raw)
        except (TypeError, ValueError):
            record = None
        if isinstance(record, dict):
            line = _line_from_record(record)
            if line is not None:
                result.append(line)
                continue
        pipe_match = _PIPE_LINE_RE.match(raw)
        if pipe_match:
            result.append(
                ExecutionLogLine(
                    timestamp=_parse_timestamp(pipe_match.group('timestamp')),
                    level=pipe_match.group('level').upper()[:16],
                    message=pipe_match.group('message')[:_MAX_MESSAGE_BYTES],
                    logger=pipe_match.group('logger').strip() or None,
                )
            )
            continue
        match = _LINE_RE.match(raw)
        if match:
            result.append(
                ExecutionLogLine(
                    timestamp=_parse_timestamp(match.group('timestamp')),
                    level=match.group('level').upper()[:16],
                    message=match.group('message')[:_MAX_MESSAGE_BYTES],
                )
            )
        else:
            result.append(ExecutionLogLine(timestamp=None, level='INFO', message=raw[:_MAX_MESSAGE_BYTES]))
    return result, partial


def normalize_zhiyan_records(records: Iterable[Any], *, max_lines: int) -> tuple[list[ExecutionLogLine], bool]:
    normalized = [line for record in records if (line := _line_from_record(record)) is not None]
    normalized.sort(
        key=lambda line: (line.timestamp is None, line.timestamp or datetime.max.replace(tzinfo=timezone.utc))
    )
    partial = len(normalized) > max_lines
    return normalized[-max_lines:] if partial else normalized, partial


def _epoch_millis(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


async def read_remote_log(
    session: aiohttp.ClientSession,
    url: str,
    *,
    headers: dict[str, str],
    lines: int,
    max_bytes: int = 1024 * 1024,
    timeout: float = 5,
    allow_redirects: bool = True,
) -> tuple[list[ExecutionLogLine], bool]:
    async with session.get(
        url,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=timeout),
        allow_redirects=allow_redirects,
    ) as response:
        if response.status >= 400 or (not allow_redirects and response.status != 200):
            raise aiohttp.ClientError(f'log upstream status {response.status}')
        # read(n) may return before all network chunks have arrived.
        try:
            body = await response.content.readexactly(max_bytes + 1)
        except asyncio.IncompleteReadError as exc:
            body = exc.partial
    return parse_manager_log_text(body.decode('utf-8', errors='replace'), max_bytes=max_bytes, max_lines=lines)


class ZhiyanLogClient:
    def __init__(self, *, session: aiohttp.ClientSession, config: ZhiyanLogQueryConfig, token: str) -> None:
        self._session = session
        self._config = config
        self._token = token

    async def query(
        self,
        *,
        log_name: str,
        filters: list[dict[str, str]],
        start_time: datetime,
        end_time: datetime,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not filters:
            raise ValueError('Log query requires a filter')
        body = {
            'startTime': _epoch_millis(start_time),
            'endTime': _epoch_millis(end_time),
            'dataflowName': log_name,
            'storageType': self._config.storage_type,
            'filter': filters,
            'orderBy': 'desc',
            'page': 1,
            'limit': limit,
        }
        headers = {
            'Content-Type': 'application/json',
            'x-api-version': 'v2',
            'token': self._token,
            'projectname': self._config.project_name,
            'staffname': self._config.staff_name,
        }
        url = f'{self._config.endpoint.rstrip("/")}/log/v2/search/query'
        timeout = aiohttp.ClientTimeout(total=10, connect=3, sock_read=7)
        try:
            async with self._session.post(
                url,
                json=body,
                headers=headers,
                timeout=timeout,
                auto_decompress=True,
            ) as response:
                if response.status == 429:
                    raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_LIMITED, status=response.status)
                if response.status >= 400:
                    raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_FAILED, status=response.status)
                try:
                    payload = await response.json()
                except (TypeError, ValueError, aiohttp.ContentTypeError) as exc:
                    raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_FAILED, status=response.status) from exc
        except ZhiyanLogError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_FAILED) from exc
        except aiohttp.ClientError as exc:
            raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_FAILED) from exc

        if not isinstance(payload, dict):
            raise ZhiyanLogError(ERROR_ZHIYAN_QUERY_FAILED)
        code = payload.get('code', 0)
        if code not in (0, '0', None):
            error_code = (
                ERROR_ZHIYAN_QUERY_LIMITED if str(code) in {'429', 'rate_limit', '限流'} else ERROR_ZHIYAN_QUERY_FAILED
            )
            raise ZhiyanLogError(error_code)
        data = payload.get('data', payload)
        records = data.get('records', data.get('items', data.get('list', []))) if isinstance(data, dict) else data
        return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


__all__ = [
    'AVAILABILITY_INGEST_PENDING',
    'AVAILABILITY_NOT_STARTED',
    'AVAILABILITY_READY',
    'AVAILABILITY_UNAVAILABLE',
    'ERROR_MANAGER_NOT_STARTED',
    'ERROR_MANAGER_UNAVAILABLE',
    'ERROR_ZHIYAN_INGEST_PENDING',
    'ERROR_ZHIYAN_NOT_CONFIGURED',
    'ERROR_ZHIYAN_QUERY_FAILED',
    'ERROR_ZHIYAN_QUERY_LIMITED',
    'ExecutionLogLine',
    'ExecutionLogPayload',
    'SOURCE_INSTANCE_MANAGER',
    'SOURCE_NONE',
    'SOURCE_ZHIYAN',
    'ZhiyanLogClient',
    'ZhiyanLogError',
    'normalize_zhiyan_records',
    'parse_manager_log_text',
    'read_remote_log',
]
