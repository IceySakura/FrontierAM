"""Stateless, authorized trace pagination with bounded query admission."""

from __future__ import annotations

import asyncio
import time

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, TypeVar

from nexus.distributed.trace.cursor import TraceCursorCodec, decode_stream_id, stream_id, target_scope
from nexus.distributed.trace.models import (
    AssignmentTraceMetadata,
    AuthorizedTraceTarget,
    TraceReadConfig,
    TraceReadDiagnostics,
    TraceReadError,
    TraceReadMode,
    encode_preview_event,
    encode_raw_event,
    json_bytes,
)
from nexus.distributed.trace.sources import ResolvedTraceSource, TraceSourceResolver
from nexus.runner.trace.tdsql.sink_source import (
    RawTraceCleanupError,
    RawTraceEventTooLargeError,
    RawTraceRecordInvalidError,
)


if TYPE_CHECKING:
    from nexus.distributed.config import DistributedConfig


_T = TypeVar('_T')
MetadataLoader = Callable[[], Awaitable[list[AssignmentTraceMetadata]]]


class TraceReadAdmission:
    def __init__(self, config: TraceReadConfig) -> None:
        self.config = config
        self._condition = asyncio.Condition()
        self._active = 0
        self._principals: dict[str, int] = {}
        self._waiters = 0

    @asynccontextmanager
    async def acquire(self, principal: str) -> AsyncIterator[None]:
        deadline = time.monotonic() + self.config.queue_timeout_seconds
        async with self._condition:
            if self._waiters >= self.config.max_concurrent_queries * 8:
                raise TraceReadError('trace_read_busy', 429, retryable=True)
            self._waiters += 1
            try:
                while (
                    self._active >= self.config.max_concurrent_queries
                    or self._principals.get(principal, 0) >= self.config.max_principal_queries
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TraceReadError('trace_read_busy', 429, retryable=True)
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except asyncio.TimeoutError:
                        raise TraceReadError('trace_read_busy', 429, retryable=True) from None
                self._active += 1
                self._principals[principal] = self._principals.get(principal, 0) + 1
            finally:
                self._waiters -= 1
        try:
            yield
        finally:
            cleanup = asyncio.create_task(self._release(principal))
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError

    async def _release(self, principal: str) -> None:
        async with self._condition:
            self._active -= 1
            self._principals[principal] -= 1
            if not self._principals[principal]:
                del self._principals[principal]
            self._condition.notify_all()


class TraceReadService:
    def __init__(self, config: DistributedConfig) -> None:
        self.config = config.trace_read
        self.resolver = TraceSourceResolver(config)
        self.codec = TraceCursorCodec(
            {key: secret.get_secret_value() for key, secret in config.secrets.trace_cursor_keys.items()},
            current_key_id=self.config.cursor_current_key_id,
            ttl_seconds=self.config.cursor_ttl_seconds,
        )
        self.admission = TraceReadAdmission(self.config)
        self._cleanup_failed = False

    @staticmethod
    def _scope(target: AuthorizedTraceTarget, source: ResolvedTraceSource, span: str | None = None) -> dict[str, str]:
        reader = source.reader
        scope = {
            **target_scope(target),
            'backend': reader.backend,
            'source_id': reader.source_id,
            'source_revision': reader.source_revision,
        }
        if source.parent_created_at is not None:
            # Paging reads resolve from the cursor alone, which must therefore carry the
            # Boundless partition key so later pages stay pruned to one partition.
            scope['parent_created_at'] = source.parent_created_at
        if span is not None:
            scope['span'] = span
        return scope

    @staticmethod
    def _base(target: AuthorizedTraceTarget) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'execution_id': target.execution_id,
            'execution_status': target.execution_status,
            'observed_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        }

    async def _query(
        self,
        target: AuthorizedTraceTarget,
        operation: Callable[[], Awaitable[_T]],
        *,
        diagnostics: TraceReadDiagnostics | None = None,
    ) -> _T:
        async with self.admission.acquire(target.principal_id):
            if self._cleanup_failed:
                raise TraceReadError('trace_source_misconfigured', 503)
            started = time.monotonic()
            initial_encoding = diagnostics.encoding_seconds if diagnostics is not None else 0
            try:
                return await asyncio.wait_for(operation(), timeout=self.config.query_timeout_seconds)
            except TraceReadError:
                raise
            except RawTraceCleanupError:
                self._cleanup_failed = True
                raise TraceReadError('trace_source_misconfigured', 503) from None
            except RawTraceEventTooLargeError as error:
                raise TraceReadError(
                    'trace_event_too_large', 413, detail=f'record_id={error.record_id} size_bytes={error.size_bytes}'
                ) from None
            except (RawTraceRecordInvalidError, ValueError, UnicodeError):
                raise TraceReadError('trace_record_invalid', 502) from None
            except Exception as error:
                code = error.args[0] if error.args else None
                if isinstance(error, (ImportError, FileNotFoundError)) or code in (
                    1044,
                    1045,
                    1049,
                    1054,
                    1064,
                    1142,
                    1146,
                ):
                    raise TraceReadError('trace_source_misconfigured', 503) from None
                raise TraceReadError('trace_source_unavailable', 503, retryable=True) from None
            finally:
                if diagnostics is not None:
                    diagnostics.queries += 1
                    diagnostics.query_seconds += max(
                        0, time.monotonic() - started - (diagnostics.encoding_seconds - initial_encoding)
                    )

    async def _resolve(
        self, target: AuthorizedTraceTarget, load_metadata: MetadataLoader, *, backend: str | None = None
    ):
        try:
            assignments = await load_metadata()
        except TraceReadError:
            raise
        except Exception:
            raise TraceReadError('trace_metadata_scan_incomplete', 503, retryable=True) from None
        if len(assignments) > self.config.max_assignments:
            raise TraceReadError('trace_metadata_scan_incomplete', 503, retryable=True)
        try:
            return self.resolver.resolve(target, assignments, backend=backend)
        except TraceReadError:
            raise
        except Exception:
            raise TraceReadError('trace_source_unresolved', 409) from None

    async def list_streams(
        self,
        target: AuthorizedTraceTarget,
        *,
        load_metadata: MetadataLoader,
        backend: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
        diagnostics: TraceReadDiagnostics | None = None,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise TraceReadError('invalid_request', 400)
        decoded = self.codec.decode(cursor, purpose='discovery', target=target) if cursor is not None else None
        if decoded is not None:
            if backend is not None and backend != decoded['backend']:
                raise TraceReadError('invalid_trace_cursor', 400)
            backend = decoded['backend']
        resolution = await self._resolve(target, load_metadata, backend=backend)
        base = {
            **self._base(target),
            'availability': resolution.availability,
            'items': [],
            'next_cursor': None,
            'scan_complete': False,
        }
        if resolution.availability == 'waiting_source':
            if decoded is not None:
                raise TraceReadError('trace_source_unresolved', 409)
            return base
        if backend is None and len(resolution.sources) > 1:
            return {
                **base,
                'availability': 'select_backend',
                'backends': [
                    {'name': source.reader.backend, 'display_name': source.reader.display_name or source.reader.backend}
                    for source in resolution.sources
                ],
            }
        source = next((s for s in resolution.sources if backend is None or s.reader.backend == backend), None)
        if source is None:
            raise TraceReadError('trace_source_unresolved', 409)
        scope = self._scope(target, source)
        if diagnostics is not None:
            diagnostics.backend = source.reader.backend
        if decoded is not None and any(
            decoded.get(key) != scope.get(key) for key in ('source_id', 'source_revision', 'parent_created_at')
        ):
            raise TraceReadError('trace_source_changed', 409)
        reader = self.resolver.create_source(source)
        spans = await self._query(
            target,
            lambda: reader.list_raw_streams(
                source.scope(target.execution_id),
                after=decoded['position'] if decoded is not None else None,
                limit=limit + 1,
            ),
            diagnostics=diagnostics,
        )
        items = []
        for span in spans[:limit]:
            attempt = source.attempts.get(span)
            items.append({
                'stream_id': stream_id(
                    backend=source.reader.backend,
                    source_id=source.reader.source_id,
                    source_revision=source.reader.source_revision,
                    run_id=target.execution_id,
                    run_span_id=span,
                ),
                'run_id': target.execution_id,
                'run_span_id': span,
                'attempt_id': attempt[0] if attempt is not None else None,
                'attempt_no': attempt[1] if attempt is not None else None,
                'is_current': attempt[2] if attempt is not None else None,
            })
        has_more = len(spans) > limit
        return {
            **base,
            'backend': source.reader.backend,
            'items': items,
            'scan_complete': not has_more,
            'next_cursor': self.codec.issue(scope, purpose='discovery', position=spans[limit - 1])
            if has_more
            else None,
        }

    async def read_events(
        self,
        target: AuthorizedTraceTarget,
        stream: str,
        *,
        load_metadata: MetadataLoader,
        cursor: str | None = None,
        limit: int = 200,
        mode: TraceReadMode = 'raw',
        diagnostics: TraceReadDiagnostics | None = None,
    ) -> dict[str, Any]:
        if type(limit) is not int or not 1 <= limit <= 500 or mode not in ('raw', 'preview'):
            raise TraceReadError('invalid_request', 400)
        if diagnostics is not None:
            diagnostics.mode = mode
        identity = decode_stream_id(stream, execution_id=target.execution_id)
        decoded = self.codec.decode(cursor, purpose='events', target=target) if cursor is not None else None
        if decoded is not None:
            if any(decoded[key] != identity[key] for key in ('backend', 'source_id', 'source_revision', 'span')):
                raise TraceReadError('invalid_trace_cursor', 400)
            source = self.resolver.from_cursor(target, decoded)
        else:
            resolution = await self._resolve(target, load_metadata, backend=identity['backend'])
            source = next((s for s in resolution.sources if s.reader.backend == identity['backend']), None)
            if source is None:
                raise TraceReadError('trace_source_unresolved', 409)
            if (source.reader.source_id, source.reader.source_revision) != (
                identity['source_id'],
                identity['source_revision'],
            ):
                raise TraceReadError('trace_source_changed', 409)
        if diagnostics is not None:
            diagnostics.backend = source.reader.backend
        span = identity['span']
        position = int(decoded['position']) if decoded is not None else 0
        if position > 2**63 - 1:
            raise TraceReadError('invalid_trace_cursor', 400)
        scope = self._scope(target, source, span)
        payload = {
            **self._base(target),
            'mode': mode,
            'stream_id': stream,
            'items': [],
            'next_cursor': self.codec.issue(scope, purpose='events', position=str(position)),
            'has_more': False,
        }
        items: list[dict[str, Any]] = []
        encoded_bytes = 0

        def encode_candidate(record: dict[str, Any]) -> bool:
            nonlocal encoded_bytes
            try:
                item = encode_preview_event(record) if mode == 'preview' else encode_raw_event(record)
                item_bytes = len(json_bytes(item))
            except (ValueError, TypeError, UnicodeError):
                raise RawTraceRecordInvalidError('raw event cannot be encoded') from None
            if item_bytes > self.config.max_event_bytes:
                if mode == 'raw':
                    raise RawTraceEventTooLargeError(record['record_id'], item_bytes)
                item['event_data'] = None
                item['event_data_omitted'] = True
                item_bytes = len(json_bytes(item))
                if item_bytes > self.config.max_event_bytes:
                    raise RawTraceEventTooLargeError(record['record_id'], item_bytes)
            token = self.codec.issue(scope, purpose='events', position=item['record_id'])
            envelope_bytes = len(json_bytes({**payload, 'next_cursor': token}))
            total = envelope_bytes + encoded_bytes + item_bytes + len(items)
            if items and total > self.config.page_target_bytes:
                return False
            items.append(item)
            if diagnostics is not None and item.get('event_data_omitted'):
                diagnostics.omitted_records += 1
                diagnostics.omitted_bytes += item['event_data_bytes']
            encoded_bytes += item_bytes
            payload['next_cursor'] = token
            return True

        def accept(record: dict[str, Any]) -> bool:
            started = time.monotonic()
            try:
                return encode_candidate(record)
            finally:
                if diagnostics is not None:
                    diagnostics.encoding_seconds += time.monotonic() - started

        reader = self.resolver.create_source(source)
        metadata = source.scope(target.execution_id, span)
        page = await self._query(
            target,
            lambda: reader.read_raw_event_page(
                metadata,
                position,
                limit=limit,
                max_event_bytes=self.config.max_event_bytes,
                **({'omit_oversized_event_data': True} if mode == 'preview' else {}),
                accept_record=accept,
            ),
            diagnostics=diagnostics,
        )
        if not page.items and not await self._query(
            target, lambda: reader.raw_stream_exists(metadata), diagnostics=diagnostics
        ):
            raise TraceReadError('trace_stream_not_found', 404)
        return {**payload, 'items': items, 'has_more': page.has_more}
