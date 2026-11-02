"""SQLite-backed local history store for the live runner."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from nexus.runner.live.models import CreateRunRequest, RunInfo, RunSummary, request_cache_key


_ACTIVE_STATUSES = ('starting', 'idle', 'running', 'cached', 'killing')
_TERMINAL_STATUSES = {'closed', 'failed', 'killed'}
logger = logging.getLogger(__name__)


TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS live_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    cache_key TEXT,
    current_turn_id TEXT,
    runtime_base_url TEXT,
    runtimes_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    output_json TEXT,
    trace_metadata_json TEXT,
    events_cursor INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT
);
"""

INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_live_runs_updated_created ON live_runs(updated_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_runs_status_updated_created ON live_runs(status, updated_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_runs_cache_key_updated_created ON live_runs(cache_key, updated_at DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_live_runs_closed_at ON live_runs(closed_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _json_load(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


def _safe_request_cache_key(request: dict[str, Any], *, run_id: str) -> str | None:
    try:
        return request_cache_key(request)
    except ValueError:
        logger.warning('Ignoring unparsable live cache key for historical run %s', run_id, exc_info=True)
        return None


class SQLiteLiveRunStore:
    """Local SQLite store for live run control metadata and history lookup."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self._conn: aiosqlite.Connection | None = None

    async def start(self) -> None:
        await self._ensure_open()

    async def close(self) -> None:
        async with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                await conn.close()

    async def _ensure_open(self) -> None:
        if self._conn is not None:
            return
        async with self._lock:
            if self._conn is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self.path)
            conn.row_factory = sqlite3.Row
            await conn.execute('PRAGMA foreign_keys=ON')
            await conn.execute('PRAGMA journal_mode=WAL')
            await conn.execute('PRAGMA busy_timeout=5000')
            await conn.executescript(TABLE_SCHEMA)
            await self._migrate_schema_locked(conn)
            await conn.executescript(INDEX_SCHEMA)
            self._conn = conn
            await self._mark_unfinished_runs_failed_locked(conn)
            await conn.commit()

    async def _migrate_schema_locked(self, conn: aiosqlite.Connection) -> None:
        async with conn.execute('PRAGMA table_info(live_runs)') as cursor:
            columns = {row['name'] for row in await cursor.fetchall()}
        if 'cache_key' not in columns:
            await conn.execute('ALTER TABLE live_runs ADD COLUMN cache_key TEXT')
        if 'trace_metadata_json' not in columns:
            await conn.execute('ALTER TABLE live_runs ADD COLUMN trace_metadata_json TEXT')
        await conn.execute('DROP TABLE IF EXISTS live_events')

        async with conn.execute('SELECT run_id, request_json FROM live_runs WHERE cache_key IS NULL') as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            request = _json_load(row['request_json'], {})
            cache_key = _safe_request_cache_key(request, run_id=row['run_id'])
            if cache_key is not None:
                await conn.execute('UPDATE live_runs SET cache_key = ? WHERE run_id = ?', (cache_key, row['run_id']))

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError('SQLite live run store is not open')
        return self._conn

    async def has_run(self, run_id: str) -> bool:
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute('SELECT 1 FROM live_runs WHERE run_id = ?', (run_id,)) as cursor:
                row = await cursor.fetchone()
            return row is not None

    async def create_run(self, run_id: str, request: CreateRunRequest) -> None:
        now = _now()
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            await conn.execute(
                """
                INSERT INTO live_runs (
                    run_id, status, request_json, cache_key, runtimes_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    'starting',
                    request.model_dump_json(),
                    _safe_request_cache_key(request.model_dump(mode='json'), run_id=run_id),
                    '{}',
                    now,
                    now,
                ),
            )
            await conn.commit()

    async def delete_run(self, run_id: str) -> None:
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            await conn.execute('DELETE FROM live_runs WHERE run_id = ?', (run_id,))
            await conn.commit()

    async def save_run_info(self, info: RunInfo, *, trace_metadata: Mapping[str, Any] | None = None) -> None:
        now = _now()
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            await self._save_run_info_locked(conn, info, now=now, trace_metadata=trace_metadata)
            await conn.commit()

    async def _save_run_info_locked(
        self,
        conn: aiosqlite.Connection,
        info: RunInfo,
        *,
        now: str,
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        closed_at = now if info.status in _TERMINAL_STATUSES else None
        await conn.execute(
            """
            UPDATE live_runs
            SET status = ?,
                cache_key = COALESCE(?, cache_key),
                current_turn_id = ?,
                runtime_base_url = ?,
                runtimes_json = ?,
                error = ?,
                output_json = ?,
                trace_metadata_json = COALESCE(?, trace_metadata_json),
                events_cursor = ?,
                updated_at = ?,
                closed_at = COALESCE(closed_at, ?)
            WHERE run_id = ?
            """,
            (
                info.status,
                info.cache_key,
                info.current_turn_id,
                info.runtime_base_url,
                _json_dump({name: runtime.model_dump(mode='json') for name, runtime in info.runtimes.items()}),
                info.error,
                _json_dump(info.output) if info.output is not None else None,
                _json_dump(dict(trace_metadata)) if trace_metadata is not None else None,
                info.events_cursor,
                now,
                closed_at,
                info.run_id,
            ),
        )

    async def get_run_info(self, run_id: str) -> RunInfo | None:
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute('SELECT * FROM live_runs WHERE run_id = ?', (run_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_info(row)

    async def get_run_request(self, run_id: str) -> dict[str, Any] | None:
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute('SELECT request_json FROM live_runs WHERE run_id = ?', (run_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return _json_load(row['request_json'], {})

    async def get_trace_metadata(self, run_id: str) -> dict[str, Any] | None:
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute('SELECT trace_metadata_json FROM live_runs WHERE run_id = ?', (run_id,)) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        metadata = _json_load(row['trace_metadata_json'], None)
        return metadata if isinstance(metadata, dict) else None

    async def list_run_infos(
        self,
        *,
        limit: int = 100,
        statuses: Sequence[str] | None = None,
        cache_key: str | None = None,
        query: str | None = None,
    ) -> list[RunInfo]:
        where_clause, params = self._list_where_clause(statuses=statuses, cache_key=cache_key, query=query)
        params.append(limit)
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(
                f"""
                SELECT * FROM live_runs
                {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                params,
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._row_to_info(row) for row in rows]

    async def list_run_summaries(
        self,
        *,
        limit: int = 100,
        statuses: Sequence[str] | None = None,
        cache_key: str | None = None,
        query: str | None = None,
    ) -> list[RunSummary]:
        where_clause, params = self._list_where_clause(statuses=statuses, cache_key=cache_key, query=query)
        params.append(limit)
        await self._ensure_open()
        async with self._lock:
            conn = self._require_conn()
            async with conn.execute(
                f"""
                SELECT run_id, status, events_cursor, cache_key, current_turn_id, runtime_base_url, error
                FROM live_runs
                {where_clause}
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                params,
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._row_to_summary(row) for row in rows]

    def _list_where_clause(
        self,
        *,
        statuses: Sequence[str] | None,
        cache_key: str | None,
        query: str | None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        status_values = list(statuses or [])
        if status_values:
            placeholders = ','.join('?' for _ in status_values)
            clauses.append(f'status IN ({placeholders})')
            params.extend(status_values)
        if cache_key:
            clauses.append('cache_key = ?')
            params.append(cache_key)
        if query:
            clauses.append('(LOWER(run_id) LIKE ? OR LOWER(COALESCE(cache_key, "")) LIKE ?)')
            query_pattern = f'%{query.lower()}%'
            params.extend([query_pattern, query_pattern])
        return (f'WHERE {" AND ".join(clauses)}' if clauses else ''), params

    def _row_to_info(self, row: sqlite3.Row) -> RunInfo:
        request = _json_load(row['request_json'], {})
        return RunInfo(
            run_id=row['run_id'],
            status=row['status'],
            events_cursor=row['events_cursor'],
            cache_key=row['cache_key'] or _safe_request_cache_key(request, run_id=row['run_id']),
            current_turn_id=row['current_turn_id'],
            runtime_base_url=row['runtime_base_url'],
            runtimes=_json_load(row['runtimes_json'], {}),
            error=row['error'],
            output=_json_load(row['output_json'], None),
        )

    def _row_to_summary(self, row: sqlite3.Row) -> RunSummary:
        return RunSummary(
            run_id=row['run_id'],
            status=row['status'],
            events_cursor=row['events_cursor'],
            cache_key=row['cache_key'],
            current_turn_id=row['current_turn_id'],
            runtime_base_url=row['runtime_base_url'],
            error=row['error'],
        )

    async def _mark_unfinished_runs_failed_locked(self, conn: aiosqlite.Connection) -> None:
        now = _now()
        placeholders = ','.join('?' for _ in _ACTIVE_STATUSES)
        await conn.execute(
            f"""
            UPDATE live_runs
            SET status = 'failed',
                error = COALESCE(error, 'Live server restarted before this run completed'),
                updated_at = ?,
                closed_at = COALESCE(closed_at, ?)
            WHERE status IN ({placeholders})
            """,
            (now, now, *_ACTIVE_STATUSES),
        )


__all__ = ['SQLiteLiveRunStore']
