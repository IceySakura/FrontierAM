from __future__ import annotations

import asyncio
import logging
import re
import sys

from collections.abc import AsyncIterator
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from nexus.distributed.observability.context import current_distributed_execution_context
from nexus.utils.log import setup_logging
from nexus.utils.log.formatters import JsonLineFormatter, TrimFormatter


DEFAULT_TAIL_MAX_BYTES = 4 * 1024 * 1024
_SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9_.-]+')
_TOP_LEVEL_EXTRA_FIELDS = {
    'allocation_id',
    'artifact_id',
    'execution_id',
    'assignment_id',
    'attempt_id',
    'batch_id',
    'environment_id',
    'manager_id',
    'manager_session_id',
    'scheduler_owner_id',
    'command_id',
}
_TEXT_LOG_TIMESTAMP_RE = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| ')
_DATE_ONLY_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def parse_log_level(value: str | int | None, *, default: int = logging.INFO) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    normalized = value.strip().upper()
    if not normalized:
        return default
    if normalized.isdigit():
        return int(normalized)
    return int(getattr(logging, normalized, default))


def safe_log_name(value: str) -> str:
    return _SAFE_NAME_RE.sub('-', value).strip('-') or 'default'


def distributed_log_path(log_dir: str | Path, component: str, identity: str | None = None) -> Path:
    name = safe_log_name(component if identity is None else f'{component}-{identity}')
    return Path(log_dir) / f'{name}.log'


def distributed_json_log_path(log_dir: str | Path, component: str, identity: str | None = None) -> Path:
    return distributed_log_path(log_dir, component, identity).with_suffix('.jsonl')


def setup_distributed_logging(
    *,
    log_dir: str,
    log_level: str | int,
    component: str,
    identity: str | None = None,
    trim_threshold: tuple[int, int] = (200, 200),
) -> Path | None:
    level = parse_log_level(log_level)
    setup_logging(log_level=level, trim_threshold=trim_threshold)
    if not log_dir:
        return None

    path = distributed_log_path(log_dir, component, identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    json_handler = TimedRotatingFileHandler(
        distributed_json_log_path(log_dir, component, identity),
        when='midnight',
        backupCount=7,
        encoding='utf-8',
    )
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(
        JsonLineFormatter(
            component=component,
            identity=identity,
            context_provider=current_distributed_execution_context,
            top_level_extra_fields=_TOP_LEVEL_EXTRA_FIELDS,
        )
    )
    root_logger.handlers.insert(0, json_handler)
    handler = TimedRotatingFileHandler(path, when='midnight', backupCount=7, encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(TrimFormatter(trim_threshold))
    root_logger.addHandler(handler)
    logging.getLogger(__name__).info('Distributed log file enabled: %s', path)
    return path


def tail_log_file(
    path: str | Path,
    *,
    lines: int = 200,
    max_bytes: int = DEFAULT_TAIL_MAX_BYTES,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    if lines <= 0:
        return ''
    log_path = Path(path)
    with log_path.open('rb') as handle:
        handle.seek(0, 2)
        end = handle.tell()
        size = min(end, max_bytes)
        handle.seek(end - size)
        data = handle.read(size)
    text = data.decode('utf-8', errors='replace')
    if start_time or end_time:
        return _filter_text_log_by_time_range(text, lines=lines, start_time=start_time, end_time=end_time)
    tail = text.splitlines(keepends=True)[-lines:]
    return ''.join(tail)


async def async_tail_log_file(
    path: str | Path,
    *,
    lines: int = 200,
    max_bytes: int = DEFAULT_TAIL_MAX_BYTES,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    return await asyncio.to_thread(
        tail_log_file,
        path,
        lines=lines,
        max_bytes=max_bytes,
        start_time=start_time,
        end_time=end_time,
    )


def _log_file_size(path: Path) -> int:
    return path.stat().st_size


def _read_log_chunk(path: Path, position: int, current_size: int) -> bytes:
    with path.open('rb') as handle:
        handle.seek(position)
        return handle.read(current_size - position)


async def follow_log_file(
    path: str | Path,
    *,
    lines: int = 200,
    start_time: str | None = None,
    end_time: str | None = None,
    poll_interval: float = 1.0,
) -> AsyncIterator[bytes]:
    log_path = Path(path)
    initial = await async_tail_log_file(log_path, lines=lines, start_time=start_time, end_time=end_time)
    if initial:
        yield initial.encode('utf-8')
    position = await asyncio.to_thread(_log_file_size, log_path)
    while True:
        try:
            current_size = await asyncio.to_thread(_log_file_size, log_path)
            if current_size < position:
                position = 0
            if current_size > position:
                chunk = await asyncio.to_thread(_read_log_chunk, log_path, position, current_size)
                position = current_size
                if chunk:
                    yield chunk
        except FileNotFoundError:
            position = 0
        await asyncio.sleep(poll_interval)


def print_log_tail(
    text: str,
    *,
    stream=None,
) -> None:
    target = stream or sys.stdout
    target.write(text)
    if text and not text.endswith('\n'):
        target.write('\n')
    target.flush()


def _filter_text_log_by_time_range(
    text: str,
    *,
    lines: int,
    start_time: str | None,
    end_time: str | None,
) -> str:
    start = _parse_log_time_bound(start_time, is_end=False)
    end = _parse_log_time_bound(end_time, is_end=True)
    if start is not None and end is not None and start > end:
        raise ValueError('start_time must be before or equal to end_time')

    filtered_records = []
    for timestamp, record_lines in _text_log_records(text):
        if timestamp is None:
            continue
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        filtered_records.append(record_lines)
    return ''.join(''.join(record_lines) for record_lines in filtered_records[-lines:])


def _text_log_records(text: str) -> list[tuple[datetime | None, list[str]]]:
    records: list[tuple[datetime | None, list[str]]] = []
    current_timestamp: datetime | None = None
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        timestamp = _text_log_line_timestamp(line)
        if timestamp is not None:
            if current_lines:
                records.append((current_timestamp, current_lines))
            current_timestamp = timestamp
            current_lines = [line]
        elif current_lines:
            current_lines.append(line)
        else:
            current_lines = [line]
    if current_lines:
        records.append((current_timestamp, current_lines))
    return records


def _text_log_line_timestamp(line: str) -> datetime | None:
    match = _TEXT_LOG_TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')


def _parse_log_time_bound(value: str | None, *, is_end: bool) -> datetime | None:
    if value is None or value.strip() == '':
        return None
    raw = value.strip()
    if _DATE_ONLY_RE.fullmatch(raw):
        parsed = datetime.strptime(raw, '%Y-%m-%d')
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999) if is_end else parsed
    normalized = raw.replace('Z', '+00:00') if raw.endswith('Z') else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(raw, '%Y-%m-%d %H:%M:%S')
        except ValueError as exc:
            bound = 'end_time' if is_end else 'start_time'
            raise ValueError(f'invalid {bound}') from exc
    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed
