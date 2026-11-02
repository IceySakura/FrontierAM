"""Bounded SQL aggregates over logical tasks; no task payloads are read here."""

from __future__ import annotations

import asyncio
import json

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import LargeBinary, case, cast, exists, func, literal, literal_column, or_, select, union_all
from sqlalchemy.orm import aliased

from nexus.distributed.config.observability import BusinessStatisticsConfig
from nexus.distributed.persistence.orm import BatchItemAttemptRow, BatchItemRow, ExecutionRow
from nexus.distributed.utils.clock import utcnow


TERMINAL = ('succeeded', 'failed', 'lost', 'expired', 'killed', 'cancelled')
CATEGORIES = ('succeeded', 'failed', 'cancelled', 'unfinished')
BEIJING = timezone(timedelta(hours=8))


class StatisticsQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    start_at: datetime
    end_at: datetime
    basis: Literal['finished', 'created', 'started'] = 'finished'
    environment_id: str | None = Field(default=None, min_length=1, max_length=64)
    task_kind: Literal['all', 'single', 'live', 'batch'] = 'all'
    task_name: str | None = Field(default=None, min_length=1, max_length=1024)

    @model_validator(mode='after')
    def validate_window(self) -> StatisticsQuery:
        if self.start_at.tzinfo is None or self.end_at.tzinfo is None:
            raise ValueError('statistics timestamps require an explicit timezone')
        if not timedelta(0) < self.end_at - self.start_at <= timedelta(days=31):
            raise ValueError('statistics window must be positive and at most 31 days')
        if any(ord(char) < 32 for char in (self.environment_id or '') + (self.task_name or '')):
            raise ValueError('statistics filters contain control characters')
        return self

    @property
    def start(self) -> datetime:
        return self.start_at.astimezone(timezone.utc).replace(tzinfo=None)

    @property
    def end(self) -> datetime:
        return self.end_at.astimezone(timezone.utc).replace(tzinfo=None)

    def filters(self) -> dict:
        return {**self.model_dump(exclude={'start_at', 'end_at'}), 'start_at': iso(self.start), 'end_at': iso(self.end)}


class StatisticsLimitError(RuntimeError):
    pass


def iso(value: datetime) -> str:
    return value.isoformat() + 'Z'


def category(status: str) -> str:
    if status == 'succeeded':
        return 'succeeded'
    if status in ('failed', 'lost', 'expired'):
        return 'failed'
    if status in ('killed', 'cancelled'):
        return 'cancelled'
    return 'unfinished'


def counters(raw: dict[str, int]) -> dict:
    result = dict.fromkeys(CATEGORIES, 0)
    for status, count in raw.items():
        result[category(status)] += count
    denominator = result['succeeded'] + result['failed']
    return {
        'total': sum(raw.values()),
        **result,
        'status_counts': dict(sorted(raw.items())),
        'success_numerator': result['succeeded'],
        'success_denominator': denominator,
        'success_rate': result['succeeded'] / denominator if denominator else None,
    }


@dataclass
class Source:
    statement: Any
    event_at: Any
    expected_time: Any
    name: Any
    batch_records: Any = None
    coverage_records: Any = None
    estimated_time: Any = None


def sources(query: StatisticsQuery) -> list[Source]:
    result = []
    if query.task_kind != 'batch':
        row = ExecutionRow
        stamp = getattr(row, {'created': 'created_at', 'started': 'started_at', 'finished': 'finished_at'}[query.basis])
        kinds = ['nexus_single', 'nexus_live'] if query.task_kind == 'all' else ['nexus_' + query.task_kind]
        statement = select(stamp.label('event_at'), row.status.label('status'), row.task_name.label('task_name')).where(
            row.parent_execution_id.is_(None),
            row.spec_type.in_(kinds),
        )
        if query.environment_id:
            statement = statement.where(row.environment_id == query.environment_id)
        if query.basis == 'finished':
            expected = row.status.in_(TERMINAL)
            statement = statement.where(expected)
        elif query.basis == 'started':
            expected = row.status.in_(('running', 'waiting_input', 'stopping', 'succeeded'))
        else:
            expected = literal(True)
        result.append(Source(statement, stamp, expected, row.task_name))
    if query.task_kind in ('all', 'batch'):
        item = BatchItemRow
        parent = aliased(ExecutionRow)
        stamp = getattr(
            item, {'created': 'created_at', 'started': 'first_started_at', 'finished': 'finished_at'}[query.basis]
        )
        statement = select(stamp.label('event_at'), item.status.label('status'), item.task_name.label('task_name'))
        if query.basis == 'finished':
            expected = item.status.in_(TERMINAL)
            statement = statement.where(expected)
        elif query.basis == 'started':
            attempt = BatchItemAttemptRow
            expected = or_(
                item.status == 'succeeded',
                item.child_execution_status.in_(('running', 'waiting_input', 'stopping')),
                exists(
                    select(literal(1)).where(
                        attempt.parent_execution_id == item.parent_execution_id,
                        attempt.item_id == item.item_id,
                        attempt.started_at.is_not(None),
                    )
                ),
            )
        else:
            expected = literal(True)
        records = statement
        statement = statement.join(parent, parent.execution_id == item.parent_execution_id)
        if query.environment_id:
            statement = statement.where(parent.environment_id == query.environment_id)
        result.append(
            Source(
                statement,
                stamp,
                expected,
                item.task_name,
                records,
                estimated_time=item.finished_at_estimated if query.basis == 'finished' else None,
            )
        )
    return result


def cohort(query: StatisticsQuery):
    statements = []
    for source in sources(query):
        statement = source.statement.where(source.event_at >= query.start, source.event_at < query.end)
        if query.task_name is not None:
            statement = statement.where(cast(source.name, LargeBinary) == query.task_name.encode('utf-8'))
        statements.append(statement)
    return union_all(*statements).subquery('logical_tasks')


async def execute(session, statement, timeout: float):
    statement = statement.prefix_with(f'/*+ MAX_EXECUTION_TIME({max(1, int(timeout * 1000))}) */', dialect='mysql')
    marker = session.info.get('business_statistics_query_marker')
    if marker:
        statement = statement.prefix_with(f'/* nexus_statistics:{marker} */', dialect='mysql')
    return await asyncio.wait_for(session.execute(statement), timeout)


async def load_statistics(session, query: StatisticsQuery, config: BusinessStatisticsConfig) -> dict:
    started = utcnow()
    timeout = config.query_timeout_seconds
    mysql = session.get_bind().dialect.name == 'mysql'
    raw = defaultdict(int)
    hourly_raw = defaultdict(lambda: defaultdict(int))
    name_raw = defaultdict(lambda: defaultdict(int))
    estimated_time = 0
    selected_sources = sources(query)
    for source in selected_sources:
        if source.batch_records is None or query.basis == 'created':
            continue
        # Enumerate only Batch parents through a loose index scan. Looking up a parent for
        # every historical item is expensive, especially when most timestamps are NULL.
        parent_keys = select(BatchItemRow.parent_execution_id).distinct().subquery('batch_parents')
        parents = select(parent_keys.c.parent_execution_id).join(
            ExecutionRow, ExecutionRow.execution_id == parent_keys.c.parent_execution_id
        )
        if query.environment_id:
            parents = parents.where(ExecutionRow.environment_id == query.environment_id)
        parent_ids = (await execute(session, parents.limit(10001), timeout)).scalars().all()
        if len(parent_ids) <= 10000:
            source.coverage_records = source.batch_records.where(BatchItemRow.parent_execution_id.in_(parent_ids))
    # Aggregate each source before combining it. A raw UNION would materialize millions of task rows.
    for source in selected_sources:
        statement = source.statement.where(source.event_at >= query.start, source.event_at < query.end)
        if query.task_name is not None:
            statement = statement.where(cast(source.name, LargeBinary) == query.task_name.encode('utf-8'))
        status_column = source.statement.selected_columns.status
        hour = (
            func.date_format(source.event_at, '%Y-%m-%dT%H:00:00')
            if mysql
            else func.strftime('%Y-%m-%dT%H:00:00', source.event_at)
        )
        rows = (
            await execute(
                session,
                statement.with_only_columns(
                    hour,
                    status_column,
                    source.name.is_not(None),
                    func.count(),
                    func.sum(case((source.estimated_time, 1), else_=0))
                    if source.estimated_time is not None
                    else literal(0),
                ).group_by(hour, status_column, source.name.is_not(None)),
                timeout,
            )
        ).all()
        known_names = 0
        for stamp, status, known_name, count, estimated_count in rows:
            estimated_time += int(estimated_count or 0)
            raw[str(status)] += int(count)
            hourly_raw[datetime.fromisoformat(stamp)][status] += int(count)
            if known_name:
                known_names += int(count)
            else:
                name_raw[None][status] += int(count)
        if not known_names:
            continue
        name_groups = [source.name, status_column]
        if mysql:
            # Binary TEXT collation still pads spaces and truncates long sort keys. Hash only
            # those names: ordinary short names retain exact grouping without per-row SHA work.
            needs_digest = or_(
                func.octet_length(source.name) > literal_column('@@max_sort_length'),
                func.octet_length(source.name) != func.octet_length(func.rtrim(source.name)),
            )
            name_groups.append(case((needs_digest, func.sha2(source.name, 256)), else_=None))
        names = (
            await execute(
                session,
                statement
                .where(source.name.is_not(None))
                .with_only_columns(source.name, status_column, func.count())
                .group_by(*name_groups)
                .limit(config.max_name_groups * 16 + 1),
                timeout,
            )
        ).all()
        for name, status, count in names:
            name_raw[name][status] += int(count)
        if len(name_raw) > config.max_name_groups or len(names) > config.max_name_groups * 16:
            raise StatisticsLimitError('statistics_too_many_task_names')

    missing_time = 0
    excluded_unknown_names = 0
    for source in selected_sources:
        # This count is deliberately unbounded in time: a NULL timestamp cannot be placed inside a window.
        coverage_records = source.coverage_records if source.coverage_records is not None else source.statement
        missing = coverage_records.where(source.event_at.is_(None), source.expected_time)
        if query.task_name is not None:
            missing = missing.where(
                or_(cast(source.name, LargeBinary) == query.task_name.encode('utf-8'), source.name.is_(None))
            )
            unknown = source.statement.where(
                source.event_at >= query.start, source.event_at < query.end, source.name.is_(None)
            )
            excluded_unknown_names += int(
                (await execute(session, select(func.count()).select_from(unknown.subquery()), timeout)).scalar_one()
            )
        # created_at is NOT NULL on both source tables.
        if query.basis == 'created':
            continue
        if mysql and source.batch_records is not None:
            # MySQL estimates NULL as selective even for unbackfilled columns. Probe the
            # narrow timestamp index first; avoid millions of secondary-to-primary lookups.
            many_nulls = select(literal(1)).select_from(BatchItemRow).where(source.event_at.is_(None))
            many_nulls = (
                many_nulls
                .with_hint(BatchItemRow, f'FORCE INDEX (idx_batch_stats_{query.basis})', dialect_name='mysql')
                .offset(10000)
                .limit(1)
            )
            if (await execute(session, many_nulls, timeout)).first() is not None:
                missing = missing.with_hint(BatchItemRow, 'FORCE INDEX (PRIMARY)', dialect_name='mysql')
        missing_time += int(
            (
                await execute(session, missing.with_only_columns(func.count(), maintain_column_froms=True), timeout)
            ).scalar_one()
        )

    hourly, daily = time_buckets(query, hourly_raw)
    summary = counters(raw)
    named = [{'task_name': name, **counters(counts)} for name, counts in name_raw.items()]
    named.sort(key=lambda row: (-row['total'], row['task_name'] is None, row['task_name'] or ''))
    for item in named:
        item['share'] = item['total'] / summary['total'] if summary['total'] else None
    top = [item for item in named if item['task_name'] is not None][:10]
    other_raw = defaultdict(int)
    for item in named:
        if item['task_name'] is not None and item not in top:
            for status, count in item['status_counts'].items():
                other_raw[status] += count
    unknown = next(
        (item for item in named if item['task_name'] is None), {'task_name': None, **counters({}), 'share': None}
    )
    count_complete = not missing_time and not excluded_unknown_names and not estimated_time
    for bucket in hourly + daily:
        bucket['count_complete'] = count_complete
    assert summary['total'] == sum(item['total'] for item in hourly) == sum(item['total'] for item in named)
    value = {
        'definition_version': 1,
        'source': 'business_database',
        'data_status': 'ready' if count_complete else 'partial',
        'filters': query.filters(),
        'timezone': 'Asia/Shanghai',
        'generated_at': iso(started),
        'summary': summary,
        'hourly': hourly,
        'daily': daily,
        'peaks': {'hourly': peak(hourly), 'daily': peak(daily)},
        'task_names': {'items': named, 'top': top, 'other': counters(dict(other_raw)), 'unknown': unknown},
        'coverage': {
            'count_complete': count_complete,
            'estimated_time_records': estimated_time,
            'estimated_time_source': 'updated_at' if estimated_time else None,
            'missing_time_records': missing_time,
            'missing_time_scope': 'Selected task types and environment across retained history; time membership is unknown.',
            'unknown_name_records': unknown['total'],
            'excluded_unknown_name_records': excluded_unknown_names,
            'name_coverage_ratio': (summary['total'] - unknown['total']) / summary['total']
            if summary['total']
            else None,
            'retention_coverage': 'Source records currently retained in the business database; deleted history cannot be reconstructed.',
        },
    }
    if len(json.dumps(value, ensure_ascii=False).encode()) > config.max_response_bytes:
        raise StatisticsLimitError('statistics_response_too_large')
    return value


def time_buckets(query: StatisticsQuery, raw: dict) -> tuple[list[dict], list[dict]]:
    stamp = query.start.replace(minute=0, second=0, microsecond=0)
    hourly = []
    daily_raw = defaultdict(lambda: defaultdict(int))
    while stamp < query.end:
        begin, end = max(stamp, query.start), min(stamp + timedelta(hours=1), query.end)
        counts = raw.get(stamp, {})
        hourly.append({
            'start_at': iso(begin),
            'end_at': iso(end),
            'partial_period': begin != stamp or end != stamp + timedelta(hours=1),
            **counters(counts),
        })
        date = stamp.replace(tzinfo=timezone.utc).astimezone(BEIJING).date()
        for status, count in counts.items():
            daily_raw[date][status] += count
        daily_raw[date]  # Retain a known empty day.
        stamp += timedelta(hours=1)
    daily = []
    for date, counts in sorted(daily_raw.items()):
        start = datetime.combine(date, datetime.min.time()) - timedelta(hours=8)
        end = start + timedelta(days=1)
        daily.append({
            'date': date.isoformat(),
            'start_at': iso(max(start, query.start)),
            'end_at': iso(min(end, query.end)),
            'partial_period': start < query.start or end > query.end,
            **counters(dict(counts)),
        })
    return hourly, daily


def peak(buckets: list[dict]) -> dict:
    maximum = max((item['total'] for item in buckets), default=0)
    winners = [item for item in buckets if item['total'] == maximum] if maximum else []
    first = winners[0] if winners else None
    return {
        'count': maximum,
        'start_at': first['start_at'] if first else None,
        'end_at': first['end_at'] if first else None,
        'partial_period': first['partial_period'] if first else False,
        'tied_periods': len(winners),
    }
