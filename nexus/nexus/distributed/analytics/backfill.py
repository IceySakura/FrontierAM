"""Explicit, restartable backfill. Inputs are read before short compare-and-set writes."""

from __future__ import annotations

import asyncio
import base64
import json

from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import Text, and_, case, cast, func, or_, select, tuple_, update

from nexus.distributed.analytics.fields import task_name_from_input, task_name_from_spec
from nexus.distributed.analytics.query import TERMINAL, execute
from nexus.distributed.batch.task_store import TDSQLBatchTaskStore, create_batch_task_store, decode_task_record
from nexus.distributed.config.store import load_distributed_config
from nexus.distributed.persistence.db import create_engine, create_session_factory
from nexus.distributed.persistence.orm import BatchItemAttemptRow, BatchItemRow, ExecutionRow
from nexus.distributed.repositories.batch import _parse_task_list_record
from nexus.distributed.utils.identity import canonical_sha256


MAX_PAGE_BYTES = 16 * 1024 * 1024


def encode_cursor(phase, after, environment_id, parent_execution_id):
    data = json.dumps([1, phase, after, environment_id, parent_execution_id]).encode()
    return base64.urlsafe_b64encode(data).decode()


def decode_cursor(cursor, environment_id, parent_execution_id):
    if not cursor:
        return 'executions', None
    try:
        if len(cursor) > 2048:
            raise ValueError
        version, phase, after, environment, parent = json.loads(base64.urlsafe_b64decode(cursor))
        if (
            version != 1
            or phase not in {'executions', 'items'}
            or (environment, parent) != (environment_id, parent_execution_id)
        ):
            raise ValueError
        if after is not None and (
            not isinstance(after, list)
            or len(after) != (1 if phase == 'executions' else 2)
            or any(not isinstance(key, str) or not key or len(key) > 64 for key in after)
        ):
            raise ValueError
        return phase, after
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid_statistics_backfill_cursor') from exc


def _local_name(path, item, max_bytes):
    if not 0 < item.task_length <= max_bytes:
        raise ValueError('task_list_record_exceeds_backfill_budget')
    with Path(path).open('rb') as file:
        file.seek(item.task_offset)
        line = file.read(item.task_length)
    # Existing records may have been accepted under a different metadata limit.
    record = _parse_task_list_record(line, offset=item.task_offset, max_metadata_bytes=max_bytes)
    if record.item_id != item.item_id or canonical_sha256(record.identity()) != item.spec_sha256:
        raise ValueError('batch_task_record_identity_mismatch')
    return task_name_from_input(record.run_input)


async def backfill_page(
    factory,
    *,
    cursor=None,
    environment_id=None,
    parent_execution_id=None,
    page_size=100,
    apply=False,
    task_store=None,
    task_lists=None,
    max_bytes=MAX_PAGE_BYTES,
):
    if not 1 <= page_size <= 1000 or not 1024 <= max_bytes <= MAX_PAGE_BYTES:
        raise ValueError('invalid_statistics_backfill_budget')
    phase, after = decode_cursor(cursor, environment_id, parent_execution_id)
    report: dict[str, Any] = {
        'phase': phase,
        'scanned': 0,
        'updated': 0,
        'would_update': 0,
        'unresolved_names': 0,
        'unresolved_times': 0,
        'read_errors': {},
        'complete': False,
    }
    if phase == 'executions':
        row = ExecutionRow
        statement = select(
            row.execution_id, row.spec_sha256, row.spec_type, func.length(cast(row.spec_json, Text)).label('size')
        ).where(
            row.task_name.is_(None),
            row.spec_type.in_(('nexus_single', 'nexus_live')),
        )
        if environment_id:
            statement = statement.where(row.environment_id == environment_id)
        if parent_execution_id:
            statement = statement.where(
                or_(row.execution_id == parent_execution_id, row.parent_execution_id == parent_execution_id)
            )
        if after:
            statement = statement.where(row.execution_id > after[0])
        async with factory() as session:
            rows = (await execute(session, statement.order_by(row.execution_id).limit(page_size), 5)).all()
        if not rows:
            report['cursor'] = encode_cursor('items', None, environment_id, parent_execution_id)
            return report
        selected, consumed = [], 0
        for item in rows:
            if selected and consumed + (item.size or 0) > max_bytes:
                break
            report['scanned'] += 1
            after = [item.execution_id]
            if (item.size or 0) > max_bytes:
                report['unresolved_names'] += 1
                report['read_errors']['input_exceeds_byte_budget'] = (
                    report['read_errors'].get('input_exceeds_byte_budget', 0) + 1
                )
                continue
            selected.append(item)
            consumed += item.size or 0
        async with factory() as session:
            payloads = (
                dict(
                    (
                        await execute(
                            session,
                            select(row.execution_id, row.spec_json).where(
                                row.execution_id.in_([item.execution_id for item in selected])
                            ),
                            5,
                        )
                    ).all()
                )
                if selected
                else {}
            )
        changes = []
        for item in selected:
            name = task_name_from_spec(item.spec_type, payloads.get(item.execution_id))
            if name is None:
                report['unresolved_names'] += 1
                continue
            changes.append(
                update(row)
                .where(
                    row.execution_id == item.execution_id, row.task_name.is_(None), row.spec_sha256 == item.spec_sha256
                )
                .values(task_name=name)
            )
    else:
        row = BatchItemRow
        statement = select(
            row.parent_execution_id,
            row.item_id,
            row.current_child_execution_id,
            row.status,
            row.status_version,
            row.task_name,
            row.first_started_at,
            row.finished_at,
            row.finished_at_estimated,
            row.task_offset,
            row.task_length,
            row.spec_sha256,
        ).join(ExecutionRow, ExecutionRow.execution_id == row.parent_execution_id)
        if environment_id:
            statement = statement.where(ExecutionRow.environment_id == environment_id)
        if parent_execution_id:
            statement = statement.where(row.parent_execution_id == parent_execution_id)
        if after:
            statement = statement.where(
                or_(
                    row.parent_execution_id > after[0],
                    and_(row.parent_execution_id == after[0], row.item_id > after[1]),
                )
            )
        async with factory() as session:
            rows = (
                await execute(session, statement.order_by(row.parent_execution_id, row.item_id).limit(page_size), 5)
            ).all()
        if not rows:
            report.update(cursor=None, complete=True)
            return report
        keys = [(item.parent_execution_id, item.item_id) for item in rows]
        child = ExecutionRow
        async with factory() as session:
            children = {
                item.execution_id: item
                for item in (
                    await execute(
                        session,
                        select(
                            child.execution_id,
                            child.task_name,
                            child.started_at,
                            child.finished_at,
                            child.status,
                            case(
                                (func.length(cast(child.spec_json, Text)) <= max_bytes // len(rows), child.spec_json),
                                else_=None,
                            ).label('spec'),
                            child.spec_type,
                        ).where(
                            child.execution_id.in_([
                                item.current_child_execution_id for item in rows if item.current_child_execution_id
                            ])
                        ),
                        5,
                    )
                ).all()
            }
            attempt = BatchItemAttemptRow
            starts = {
                (parent, item): stamp
                for parent, item, stamp in (
                    await execute(
                        session,
                        select(
                            attempt.parent_execution_id,
                            attempt.item_id,
                            func.min(attempt.started_at),
                        )
                        .where(tuple_(attempt.parent_execution_id, attempt.item_id).in_(keys))
                        .group_by(attempt.parent_execution_id, attempt.item_id),
                        5,
                    )
                ).all()
            }
        changes = []
        errors = Counter()
        remaining_bytes = max_bytes
        task_store_available = isinstance(task_store, TDSQLBatchTaskStore)
        for item in rows:
            report['scanned'] += 1
            after = [item.parent_execution_id, item.item_id]
            values = {}
            current = children.get(item.current_child_execution_id)
            if item.task_name is None:
                name = (current.task_name or task_name_from_spec(current.spec_type, current.spec)) if current else None
                if name is None and remaining_bytes > 0 and task_store_available:
                    assert isinstance(task_store, TDSQLBatchTaskStore)
                    try:
                        record = await asyncio.wait_for(
                            task_store.read_record(
                                parent_execution_id=item.parent_execution_id,
                                item_id=item.item_id,
                                max_bytes=remaining_bytes,
                            ),
                            2,
                        )
                        remaining_bytes -= len(record.record_data)
                        decoded = decode_task_record(
                            record,
                            expected_sha256=item.spec_sha256,
                            max_record_bytes=max_bytes,
                            max_frame_bytes=max_bytes,
                        )
                        name = task_name_from_input(decoded.run_input)
                    except Exception as exc:
                        errors[type(exc).__name__] += 1
                        if not isinstance(exc, (KeyError, ValueError)):
                            task_store_available = False
                            await task_store.close()
                local_path = (task_lists or {}).get(item.parent_execution_id)
                if name is None and local_path and remaining_bytes > 0:
                    try:
                        name = await asyncio.to_thread(_local_name, local_path, item, remaining_bytes)
                        remaining_bytes -= item.task_length
                    except (ValueError, OSError) as exc:
                        errors[type(exc).__name__] += 1
                if name is None:
                    report['unresolved_names'] += 1
                else:
                    values['task_name'] = name
            known_starts = [
                stamp
                for stamp in (
                    item.first_started_at,
                    starts.get((item.parent_execution_id, item.item_id)),
                    current.started_at if current else None,
                )
                if stamp is not None
            ]
            if known_starts and min(known_starts) != item.first_started_at:
                values['first_started_at'] = min(known_starts)
            if item.status in TERMINAL and (item.finished_at is None or item.finished_at_estimated):
                if current and current.status == item.status and current.finished_at is not None:
                    values['finished_at'] = current.finished_at
                    values['finished_at_estimated'] = False
                else:
                    report['unresolved_times'] += 1
            if values:
                conditions = [
                    row.parent_execution_id == item.parent_execution_id,
                    row.item_id == item.item_id,
                    row.status_version == item.status_version,
                ]
                # Separate backfill workers and live writes must not overwrite each other's completed fields.
                if 'task_name' in values:
                    conditions.append(row.task_name.is_(None))
                if 'finished_at' in values:
                    conditions.extend([
                        row.finished_at == item.finished_at,
                        row.finished_at_estimated == item.finished_at_estimated,
                    ])
                if 'first_started_at' in values:
                    conditions.append(row.first_started_at == item.first_started_at)
                changes.append(update(row).where(*conditions).values(**values))
        report['read_errors'] = dict(errors)
    report['would_update'] = len(changes)
    if apply and changes:
        async with factory() as session:
            for statement in changes:
                result = await asyncio.wait_for(session.execute(statement), 5)
                report['updated'] += result.rowcount
            await session.commit()
    report['cursor'] = encode_cursor(phase, after, environment_id, parent_execution_id)
    return report


async def run_backfill(config, args):
    engine = create_engine(config.model_copy(update={'db_pool_size': 1, 'db_max_overflow': 0, 'db_pool_timeout': 1}))
    factory = create_session_factory(engine)
    task_store = None
    try:
        if args.with_task_store:
            effective = await load_distributed_config(
                factory, database_url=config.read_database_url(), role='reconciler', require_stored_config=False
            )
            task_store = create_batch_task_store(effective, read_only=True)
        task_lists = {}
        for mapping in args.task_list:
            parent, separator, path = mapping.partition('=')
            if not separator or not parent or not path:
                raise ValueError('task-list requires PARENT_EXECUTION_ID=LOCAL_PATH')
            task_lists[parent] = Path(path)
        cursor = args.cursor
        for _ in range(args.max_pages):
            report = await asyncio.wait_for(
                backfill_page(
                    factory,
                    cursor=cursor,
                    environment_id=args.environment_id,
                    parent_execution_id=args.parent_execution_id,
                    page_size=args.batch_size,
                    apply=args.apply,
                    task_store=task_store,
                    task_lists=task_lists,
                ),
                30,
            )
            print(json.dumps({'applied': args.apply, **report}), flush=True)
            cursor = report['cursor']
            if report['complete']:
                break
            await asyncio.sleep(args.pause_seconds)
    finally:
        if task_store is not None:
            await task_store.close()
        await engine.dispose()
