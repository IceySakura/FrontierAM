"""Local-file backend for raw runner trace journals."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.runner.trace.trace import RunTraceContext, RunTraceSink, RunTraceSource
from nexus.trace.records import SpanEventRecord
from nexus.utils.platform_metrics import measure_operation


RUN_TRACE_FORMAT = 'span_event_record.jsonl'


def _artifact_segment(value: str, *, field_name: str) -> str:
    if not value or value in {'.', '..'}:
        raise ValueError(f'invalid {field_name}: {value!r}')
    if len(value) > 128:
        raise ValueError(f'invalid {field_name}: value is longer than 128 characters')
    if any(character in value for character in ('/', '\\', '\x00')):
        raise ValueError(f'invalid {field_name}: path separators and NUL are not allowed')
    return value


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.tmp')
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    os.replace(tmp_path, path)


def _append_records(path: Path, records: list[SpanEventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        for record in records:
            file.write(json.dumps(record.model_dump(mode='json'), ensure_ascii=False, separators=(',', ':')))
            file.write('\n')


def _read_records_after(
    path: Path,
    cursor: int,
    *,
    root: Path | None = None,
    limit: int | None = None,
) -> list[SpanEventRecord]:
    _validate_path(path, root=root)
    result: list[SpanEventRecord] = []
    with path.open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            record = SpanEventRecord.model_validate_json(line)
            if record.record_id > cursor:
                result.append(record)
                if limit is not None and len(result) >= limit:
                    break
    return result


def _read_result(path: Path, run_span_id: str, *, root: Path | None = None) -> dict[str, Any]:
    _validate_path(path, root=root)
    output: dict[str, Any] | None = None
    with path.open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            record = SpanEventRecord.model_validate_json(line)
            if record.span_id != run_span_id:
                continue
            candidate = _run_output_from_record(record)
            if candidate is not None:
                output = candidate
    if output is None:
        raise FileNotFoundError('run result artifact is not available')
    return output


def _validate_path(path: Path, *, root: Path | None) -> None:
    if root is None:
        return
    resolved_root = root.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f'artifact path escapes root: {path}') from exc


@dataclass(slots=True)
class RunTraceLocalSink(RunTraceSink):
    root: Path
    execution_id: str
    assignment_id: str
    attempt_id: str
    batch_size: int = 100
    flush_interval: float = 0.2
    queue_maxsize: int = 10_000
    run_dir: Path = field(init=False)
    trace_events_path: Path = field(init=False)
    manifest_path: Path = field(init=False)
    _queue: asyncio.Queue[SpanEventRecord | None] = field(init=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False)
    _context: RunTraceContext | None = field(default=None, init=False)
    _trace_event_cursor: int = field(default=0, init=False)
    _flushed_trace_event_cursor: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _status: str = field(default='running', init=False)
    _current_turn_id: str | None = field(default=None, init=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.queue_maxsize < 1:
            raise ValueError('queue_maxsize must be at least 1')
        self.root = Path(self.root)
        execution_id = _artifact_segment(self.execution_id, field_name='execution_id')
        attempt_id = _artifact_segment(self.attempt_id, field_name='attempt_id')
        assignment_id = _artifact_segment(self.assignment_id, field_name='assignment_id')
        self.run_dir = self.root / execution_id / attempt_id / assignment_id
        _validate_path(self.run_dir, root=self.root)
        self.trace_events_path = self.run_dir / 'trace_events.jsonl'
        self.manifest_path = self.run_dir / 'manifest.json'
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)

    async def start(self, context: RunTraceContext) -> None:
        if self._task is not None:
            raise RuntimeError('run trace sink is already started')
        self._context = context
        try:
            _validate_path(self.run_dir, root=self.root)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            await self._write_manifest()
            self._task = asyncio.create_task(self._writer_loop(), name=f'run-trace:{self.assignment_id}')
        except Exception:
            self._status = 'artifact_failed'
            self._closed = True
            raise

    async def on_event(self, context: RunTraceContext, event: SpanEventRecord) -> None:
        if not isinstance(event, SpanEventRecord):
            raise TypeError(f'expected SpanEventRecord, got {type(event).__name__}')
        if self._closed:
            raise RuntimeError('run trace sink is closed')
        if self._task is None:
            raise RuntimeError('run trace sink is not started')
        self._raise_writer_failure()
        self._context = context
        self._trace_event_cursor = max(self._trace_event_cursor, event.record_id)
        self._capture_run_metadata(event)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            self._status = 'artifact_failed'
            raise RuntimeError(f'run trace writer queue is full for {self.assignment_id}') from exc

    async def close(self, context: RunTraceContext | None = None) -> None:
        if context is not None:
            self._context = context
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(),
                name=f'run-trace-close:{self.assignment_id}',
            )
        await asyncio.shield(self._close_task)

    async def _close_impl(self) -> None:
        if self._closed:
            return
        self._closed = True
        writer_error: Exception | None = None
        try:
            if self._task is not None:
                if not self._task.done():
                    sentinel_task = asyncio.create_task(self._queue.put(None))
                    done, _ = await asyncio.wait(
                        {sentinel_task, self._task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if self._task in done:
                        sentinel_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await sentinel_task
                    else:
                        await sentinel_task
                await self._task
        except Exception as exc:
            writer_error = exc
            self._status = 'artifact_failed'
        try:
            await self._write_manifest()
        except Exception:
            if writer_error is None:
                raise
        if writer_error is not None:
            raise writer_error

    def metadata(self) -> dict[str, Any]:
        context = self._context
        payload: dict[str, Any] = {
            'version': 2,
            'format': RUN_TRACE_FORMAT,
            'execution_id': self.execution_id,
            'assignment_id': self.assignment_id,
            'attempt_id': self.attempt_id,
            'trace_events_path': str(self.trace_events_path),
            'manifest_path': str(self.manifest_path),
            'trace_event_cursor': self._trace_event_cursor,
            'flushed_trace_event_cursor': self._flushed_trace_event_cursor,
            'status': self._status,
        }
        if context is not None:
            payload['run_id'] = context.run_id
            payload['run_span_id'] = context.run_span_id
            payload['run_kind'] = context.run_kind
            if context.metadata:
                payload['metadata'] = dict(context.metadata)
        if self._current_turn_id is not None:
            payload['current_turn_id'] = self._current_turn_id
        return payload

    async def _writer_loop(self) -> None:
        batch: list[SpanEventRecord] = []
        stopping = False
        while not stopping:
            item = await self._queue.get()
            if item is None:
                stopping = True
            else:
                batch.append(item)

            deadline = asyncio.get_running_loop().time() + self.flush_interval
            while not stopping and len(batch) < self.batch_size:
                timeout = max(0.0, deadline - asyncio.get_running_loop().time())
                if timeout == 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    stopping = True
                    break
                batch.append(item)

            if batch:
                to_write = batch
                batch = []
                with measure_operation('trace_write', family='persistence', source='local_trace'):
                    await asyncio.to_thread(_append_records, self.trace_events_path, to_write)
                self._flushed_trace_event_cursor = max(self._flushed_trace_event_cursor, to_write[-1].record_id)

    async def _write_manifest(self) -> None:
        await asyncio.to_thread(_atomic_write_json, self.manifest_path, self.metadata())

    def _capture_run_metadata(self, record: SpanEventRecord) -> None:
        if self._context is None or record.span_id != self._context.run_span_id or record.event_type != 'update':
            return
        for item in record.event_data.get('events') or []:
            if item.get('name') != 'run.event':
                continue
            attributes = item.get('attributes') or {}
            event_data = attributes.get('event_data') or {}
            if not isinstance(event_data, dict):
                continue
            turn_id = event_data.get('turn_id')
            if isinstance(turn_id, str):
                self._current_turn_id = turn_id
            event_type = attributes.get('event_type')
            if event_type == 'run.status':
                status = event_data.get('status')
                if isinstance(status, str):
                    self._status = 'succeeded' if status == 'closed' else status
            elif event_type == 'run.output' and self._status == 'running':
                output = event_data.get('output')
                if isinstance(output, dict) and output.get('success') is False:
                    self._status = 'failed'

    def _raise_writer_failure(self) -> None:
        task = self._task
        if task is None or not task.done():
            return
        if task.cancelled():
            raise RuntimeError('run trace writer task was cancelled')
        error = task.exception()
        if error is not None:
            raise error


@dataclass(slots=True)
class RunTraceLocalSource(RunTraceSource):
    root: str | Path | None = None

    def __post_init__(self) -> None:
        if self.root is not None:
            self.root = Path(self.root)

    async def read_events_after(
        self,
        metadata: Mapping[str, Any],
        cursor: int,
        *,
        limit: int | None = None,
    ) -> list[SpanEventRecord]:
        if cursor < 0:
            raise ValueError('cursor must not be negative')
        if limit is not None and limit < 1:
            raise ValueError('limit must be at least 1')
        trace_events_path = metadata.get('trace_events_path')
        if not isinstance(trace_events_path, str):
            raise FileNotFoundError('run trace artifact is not available')
        return await asyncio.to_thread(
            _read_records_after,
            Path(trace_events_path),
            cursor,
            root=self._root_path(),
            limit=limit,
        )

    async def read_result(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        trace_events_path = metadata.get('trace_events_path')
        run_span_id = metadata.get('run_span_id')
        if not isinstance(trace_events_path, str) or not isinstance(run_span_id, str):
            raise FileNotFoundError('run result artifact is not available')
        return await asyncio.to_thread(
            _read_result,
            Path(trace_events_path),
            run_span_id,
            root=self._root_path(),
        )

    def _root_path(self) -> Path | None:
        if self.root is None:
            return None
        if isinstance(self.root, Path):
            return self.root
        return Path(self.root)


def _run_output_from_record(record: SpanEventRecord) -> dict[str, Any] | None:
    if record.event_type != 'update':
        return None
    for item in record.event_data.get('events') or []:
        if item.get('name') != 'run.event':
            continue
        attributes = item.get('attributes') or {}
        if attributes.get('event_type') != 'run.output':
            continue
        event_data = attributes.get('event_data') or {}
        if not isinstance(event_data, dict):
            continue
        output = event_data.get('output')
        if isinstance(output, dict):
            return output
    return None
