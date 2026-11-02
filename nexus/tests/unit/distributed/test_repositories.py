from __future__ import annotations

from collections import Counter
from datetime import timedelta

import pytest

from sqlalchemy import event, inspect, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainConflictError, ResourceGoneError, TemporaryUnavailableError
from nexus.distributed.execution.models import (
    AssignmentStatus,
    CommandStatus,
    ExecutionMode,
    ExecutionStatus,
    RouteStatus,
)
from nexus.distributed.persistence.db import TransactionRetryError
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    AssignmentRow,
    EnvironmentAllocationRow,
    EnvironmentRow,
    ExecutionCommandRow,
    ExecutionCredentialRow,
    ExecutionEventRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
    SchemaVersionRow,
    ShardLeaseRow,
    create_schema,
)
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.gateway import GatewayRepository
from nexus.distributed.repositories.manager import ManagerRepository
from nexus.distributed.repositories.reconciler import ReconcilerRepository
from nexus.distributed.repositories.scheduler import SchedulerRepository
from nexus.distributed.repositories.shard_lease import ShardLeaseFenceStatus, ShardLeaseRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import stable_shard


@pytest.fixture
async def session_factory():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def _register_manager(
    session_factory,
    *,
    manager_id: str,
    session_id: str = 'session-1',
    modes: list[str] | None = None,
    spec_types: list[str] | None = None,
    instance_modes: list[str] | None = None,
    capacity_total: int = 1,
    environment_id: str = 'predefined-default',
    pool_id: str = 'default',
    workload: str = 'execution',
    labels: dict | None = None,
    draining: bool = False,
) -> None:
    async with session_factory() as session:
        repo = ManagerRepository(session)
        replaced = await repo.register_session(
            manager_id=manager_id,
            session_id=session_id,
            endpoint_url=f'http://{manager_id}.internal:9001',
            supported_modes=modes or [ExecutionMode.JOB],
            supported_spec_types=spec_types or ['nexus_single'],
            supported_instance_modes=instance_modes or ['asyncio'],
            capacity_total=capacity_total,
            environment_id=environment_id,
            pool_id=pool_id,
            workload=workload,
            labels=labels or {},
            draining=draining,
            force_replace=False,
            now=utcnow(),
        )
        await session.commit()
    assert replaced is True


async def _create_execution(
    session_factory,
    *,
    execution_id: str,
    mode: str = ExecutionMode.JOB,
    spec_type: str = 'nexus_single',
    route_status: str = RouteStatus.NONE,
    max_attempts: int = 1,
    placement: dict | None = None,
    slot_cost: int = 1,
    trace_carrier: dict[str, str] | None = None,
    owner_principal_id: str = 'legacy',
):
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        execution = await repo.create_execution(
            CreateExecution(
                execution_id=execution_id,
                mode=mode,
                spec_type=spec_type,
                instance_mode='asyncio',
                spec={'run_input': {'task_name': 'identity'}},
                driver_config={},
                placement=placement or {},
                route_status=route_status,
                slot_cost=slot_cost,
                max_attempts=max_attempts,
                trace_carrier=trace_carrier,
                owner_principal_id=owner_principal_id,
            ),
            now=utcnow(),
        )
        await session.commit()
        return execution


async def test_execution_repository_can_require_aggregate_level_idempotency_retry(session_factory):
    first_request = CreateExecution(
        execution_id='execution-winner',
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
        instance_mode='asyncio',
        spec={'run_input': {'task_name': 'identity'}},
        idempotency_scope='submission:single:test',
        idempotency_key='request-1',
        owner_principal_id='owner',
    )
    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_execution(first_request, now=utcnow())
        await session.commit()

    second_request = CreateExecution(
        execution_id='execution-loser',
        mode=first_request.mode,
        spec_type=first_request.spec_type,
        instance_mode=first_request.instance_mode,
        spec=first_request.spec,
        idempotency_scope=first_request.idempotency_scope,
        idempotency_key=first_request.idempotency_key,
        owner_principal_id=first_request.owner_principal_id,
    )
    async with session_factory() as session:
        with pytest.raises(TransactionRetryError, match='execution_idempotency_winner_changed'):
            await ExecutionRepository(session, shard_count=8).create_execution(
                second_request,
                now=utcnow(),
                reuse_existing=False,
            )


async def test_execution_repository_persists_internal_trace_carrier(session_factory):
    trace_carrier = {
        'traceparent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'tracestate': 'vendor=value',
    }

    execution = await _create_execution(
        session_factory,
        execution_id='execution-with-trace-carrier',
        trace_carrier=trace_carrier,
    )

    assert not hasattr(execution, 'trace_carrier')
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
    assert row is not None
    assert row.trace_carrier_json == trace_carrier


async def test_execution_repository_captures_current_trace_for_batch_create(
    session_factory, monkeypatch, otel_hyway_instrumentor
):
    trace_carrier = {
        'traceparent': '00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01',
        'tracestate': 'nexus=validation,idempotency-key=parent-request',
    }
    monkeypatch.setattr('nexus.distributed.repositories.execution.inject_carrier', lambda: trace_carrier)
    requests = [
        CreateExecution(
            execution_id=f'batch-child-{index}',
            mode=ExecutionMode.JOB,
            spec_type='nexus_single',
            instance_mode='asyncio',
            spec={
                'run_input': {'task_name': 'identity'},
                'batch': {'parent_execution_id': 'parent', 'item_id': f'item-{index}', 'attempt_no': 0},
            },
            idempotency_scope='batch:parent',
            idempotency_key=f'item-{index}',
        )
        for index in range(2)
    ]

    async with session_factory() as session:
        await ExecutionRepository(session, shard_count=8).create_executions_batch(requests, now=utcnow())
        await session.commit()
        rows = [await session.get(ExecutionRow, request.execution_id) for request in requests]

    for index, row in enumerate(rows):
        assert row is not None
        assert row.trace_carrier_json == {
            **trace_carrier,
            'tracestate': f'hyway_instance_id=item-{index},' + trace_carrier['tracestate'],
        }


async def test_execution_log_context_reads_timing_without_changing_public_record(session_factory):
    execution = await _create_execution(session_factory, execution_id='log-context-execution')
    started_at = utcnow()
    finished_at = started_at + timedelta(seconds=5)
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.started_at = started_at
        row.finished_at = finished_at
        row.status = ExecutionStatus.SUCCEEDED
        row.updated_at = finished_at
        await session.commit()

    async with session_factory() as session:
        context = await ExecutionRepository(session, shard_count=8).get_log_context(execution.execution_id)

    assert context.execution.execution_id == execution.execution_id
    assert context.execution.status == ExecutionStatus.SUCCEEDED
    assert context.started_at == started_at
    assert context.finished_at == finished_at
    assert not hasattr(context.execution, 'started_at')


async def test_scheduler_creates_assignment_carrier_from_execution_carrier(session_factory, monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nexus.utils.otel.propagation import inject_carrier

    monkeypatch.setenv('NEXUS_OTEL_ENABLED', 'true')
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, 'get_tracer', provider.get_tracer)
    tracer = provider.get_tracer('tests.distributed.scheduler')
    try:
        await _register_manager(session_factory, manager_id='trace-manager')
        with tracer.start_as_current_span('execution.enqueue') as enqueue_span:
            execution_carrier = inject_carrier()
        execution = await _create_execution(
            session_factory,
            execution_id='scheduled-with-trace-carrier',
            trace_carrier=execution_carrier,
        )

        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
            await session.commit()
            assert assignment is not None
            row = await session.get(AssignmentRow, assignment.assignment_id)

        assert row is not None
        assert row.trace_carrier_json is not None
        schedule_span = next(
            span for span in exporter.get_finished_spans() if span.name == 'distributed.scheduler.assign'
        )
        assert schedule_span.parent is not None
        assert schedule_span.parent.span_id == enqueue_span.get_span_context().span_id
        assert row.trace_carrier_json['traceparent'].split('-')[2] == f'{schedule_span.context.span_id:016x}'
    finally:
        provider.shutdown()


async def test_execution_and_manager_repository_return_identical_record_snapshots(session_factory):
    await _register_manager(session_factory, manager_id='record-manager')
    execution = await _create_execution(session_factory, execution_id='record-execution')
    now = utcnow()
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        await session.commit()
    assert assignment is not None

    async with session_factory() as session:
        execution_snapshot = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        pending = await ManagerRepository(session).list_pending_assignments(
            'record-manager',
            'session-1',
        )

    assert len(pending) == 1
    manager_execution, manager_assignment, _trace_carrier = pending[0]
    assert manager_execution == execution_snapshot
    assert manager_assignment == execution_snapshot.current_assignment
    assert manager_assignment == assignment


async def test_manager_pending_assignment_sort_uses_narrow_candidate_projection(session_factory):
    await _register_manager(session_factory, manager_id='pending-query-manager')
    execution = await _create_execution(session_factory, execution_id='pending-query-execution')
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()
    assert assignment is not None

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            pending = await ManagerRepository(session).list_pending_assignments(
                assignment.manager_id,
                assignment.manager_session_id,
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert len(pending) == 1
    ordered = next(
        statement.lower() for statement in statements if 'order by assignments.created_at' in statement.lower()
    )
    assert 'assignments.runtime_info_json' not in ordered
    assert 'assignments.trace_carrier_json' not in ordered
    assert 'executions.spec_json' not in ordered
    assert 'executions.driver_config_json' not in ordered
    assert 'executions.result_json' not in ordered


async def test_manager_ack_lock_queries_omit_unneeded_payload_columns(session_factory):
    await _register_manager(session_factory, manager_id='ack-query-manager')
    execution = await _create_execution(session_factory, execution_id='ack-query-execution')
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()
    assert assignment is not None

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            acknowledged = await ManagerRepository(session).ack_assignments(
                [assignment.assignment_id],
                assignment.manager_id,
                assignment.manager_session_id,
                now=utcnow(),
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert len(acknowledged) == 1
    execution_select = next(
        statement.lower() for statement in statements if 'order by executions.execution_id' in statement.lower()
    )
    assignment_select = next(
        statement.lower()
        for statement in statements
        if 'order by assignments.execution_id, assignments.assignment_id' in statement.lower()
    )
    assert 'executions.spec_json' not in execution_select
    assert 'executions.driver_config_json' not in execution_select
    assert 'executions.result_json' not in execution_select
    assert 'assignments.trace_carrier_json' not in assignment_select
    assert 'assignments.endpoint_error' not in assignment_select


async def test_scheduler_round_robins_queued_work_across_owners(session_factory):
    await _register_manager(session_factory, manager_id='im-fair', capacity_total=4)
    await _create_execution(session_factory, execution_id='owner-a-1', owner_principal_id='owner-a')
    await _create_execution(session_factory, execution_id='owner-a-2', owner_principal_id='owner-a')
    await _create_execution(session_factory, execution_id='owner-a-3', owner_principal_id='owner-a')
    await _create_execution(session_factory, execution_id='owner-b-1', owner_principal_id='owner-b')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        )
        assignments = await scheduler.claim_and_assign_batch(
            shard_ids=list(range(8)),
            now=utcnow(),
            scan_limit=4,
            max_assignments=2,
        )
        owners = {
            (await session.get(ExecutionRow, assignment.execution_id)).owner_principal_id for assignment in assignments
        }

    assert owners == {'owner-a', 'owner-b'}
    assert scheduler.last_claim_stats.scanned == 4
    assert scheduler.last_claim_stats.claimed == 4
    assert scheduler.last_claim_stats.assigned == 2
    assert scheduler.last_claim_stats.lock_contention == 0
    assert scheduler.last_claim_stats.no_capacity == 0
    assert scheduler.last_claim_stats.candidate_queries == 2


async def test_scheduler_fair_candidate_claim_uses_constant_execution_query_count(session_factory):
    await _register_manager(session_factory, manager_id='fair-query-manager')
    now = utcnow()
    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'fair-query-manager')
        assert manager is not None
        manager.supported_spec_versions_json = {'nexus_single': [1]}
        repository = ExecutionRepository(session, shard_count=8)
        for index in range(25):
            await repository.create_execution(
                CreateExecution(
                    execution_id=f'fair-query-{index:02d}',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'spec_version': 2} if index == 0 else {},
                    owner_principal_id=f'owner-{index:02d}',
                ),
                now=now,
            )
        await session.commit()

    execution_selects = []
    engine = session_factory.kw['bind']

    def record_execution_select(_conn, _cursor, statement, _parameters, _context, _executemany):
        normalized = statement.lstrip().lower()
        if normalized.startswith('select') and 'executions' in normalized:
            execution_selects.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_execution_select)
    try:
        async with session_factory() as session:
            scheduler = SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            )
            candidate_ids = await scheduler._fair_execution_candidate_ids(
                shard_ids=list(range(8)),
                now=now,
                scan_limit=25,
            )
            claimed = await scheduler._lock_fair_executions(
                candidate_ids=candidate_ids,
                shard_ids=list(range(8)),
                now=now,
            )
            managers = await scheduler._healthy_managers(now)
            assert len(managers) == 1
            assert scheduler._manager_can_run(claimed[0], managers[0], {}) is False
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_execution_select)

    assert len(claimed) == 25
    assert len(execution_selects) == 2
    assert 'case ' not in execution_selects[1].lower()
    assert {
        'driver_config_json',
        'error_json',
        'result_json',
        'spec_json',
    } <= inspect(claimed[0]).unloaded


async def test_scheduler_healthy_manager_scan_omits_runtime_snapshots(session_factory):
    await _register_manager(session_factory, manager_id='narrow-manager-scan')
    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            managers = await SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
            )._healthy_managers(utcnow())
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert len(managers) == 1
    statement = next(
        statement.lower()
        for statement in statements
        if 'from instance_managers' in statement.lower() and 'instance_manager_liveness' in statement.lower()
    )
    assert 'instance_managers.local_instances_json' not in statement
    assert 'instance_managers.metadata_json' not in statement
    assert 'instance_managers.capability_snapshot_json' not in statement


async def test_scheduler_ranked_candidates_round_robin_owners_and_order_within_owner(session_factory):
    now = utcnow()
    requests = (
        ('owner-a-low', 'owner-a', 0),
        ('owner-a-high', 'owner-a', 10),
        ('owner-b-only', 'owner-b', 0),
        ('owner-c-first', 'owner-c', 5),
        ('owner-c-second', 'owner-c', 0),
    )
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for execution_id, owner_id, priority in requests:
            await repository.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id=owner_id,
                    priority=priority,
                ),
                now=now,
            )
        await session.commit()

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        )
        candidate_ids = await scheduler._fair_execution_candidate_ids(
            shard_ids=list(range(8)),
            now=now,
            scan_limit=len(requests),
        )
        claimed = await scheduler._lock_fair_executions(
            candidate_ids=candidate_ids,
            shard_ids=list(range(8)),
            now=now,
        )

    assert [row.execution_id for row in claimed] == [
        'owner-a-high',
        'owner-b-only',
        'owner-c-first',
        'owner-a-low',
        'owner-c-second',
    ]


async def test_scheduler_ranked_candidate_lock_revalidates_execution_state(session_factory, monkeypatch):
    now = utcnow()
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for execution_id in ('candidate-stale', 'candidate-current'):
            await repository.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    owner_principal_id=execution_id,
                ),
                now=now,
            )
        await session.commit()

    async with session_factory() as session:
        original_scalars = session.scalars
        scalar_calls = 0

        async def complete_candidate_before_lock(statement, *args, **kwargs):
            nonlocal scalar_calls
            scalar_calls += 1
            result = await original_scalars(statement, *args, **kwargs)
            if scalar_calls == 1:
                stale = await session.get(ExecutionRow, 'candidate-stale')
                assert stale is not None
                stale.status = ExecutionStatus.SUCCEEDED
                stale.finished_at = now
                await session.flush()
            return result

        monkeypatch.setattr(session, 'scalars', complete_candidate_before_lock)
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        )
        candidate_ids = await scheduler._fair_execution_candidate_ids(
            shard_ids=list(range(8)),
            now=now,
            scan_limit=2,
        )
        claimed = await scheduler._lock_fair_executions(
            candidate_ids=candidate_ids,
            shard_ids=list(range(8)),
            now=now,
        )

    assert [row.execution_id for row in claimed] == ['candidate-current']


async def test_scheduler_owner_quota_does_not_block_other_owner(session_factory):
    await _register_manager(session_factory, manager_id='im-owner-quota', capacity_total=3)
    await _create_execution(session_factory, execution_id='quota-a-1', owner_principal_id='owner-a')
    await _create_execution(session_factory, execution_id='quota-a-2', owner_principal_id='owner-a')
    await _create_execution(session_factory, execution_id='quota-b-1', owner_principal_id='owner-b')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            default_user_quota={'running_executions': 1, 'reserved_slots': 1},
        )
        assignments = await scheduler.claim_and_assign_batch(
            shard_ids=list(range(8)),
            now=utcnow(),
            scan_limit=3,
            max_assignments=3,
        )
        owners = [
            (await session.get(ExecutionRow, assignment.execution_id)).owner_principal_id for assignment in assignments
        ]
        blocked = await session.get(ExecutionRow, 'quota-a-2')

    assert sorted(owners) == ['owner-a', 'owner-b']
    assert blocked.status == ExecutionStatus.QUEUED
    assert blocked.last_schedule_error == 'running_execution_quota_exceeded'


async def test_manager_register_session_stores_environment_metadata(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-env',
        environment_id='predefined-east',
        pool_id='interactive',
        labels={'region': 'east', 'disk': 'ssd'},
        draining=True,
        capacity_total=3,
    )

    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'im-env')

    assert manager is not None
    assert manager.environment_id == 'predefined-east'
    assert manager.pool_id == 'interactive'
    assert manager.workload == 'execution'
    assert manager.labels_json == {'region': 'east', 'disk': 'ssd'}
    assert manager.draining is True


@pytest.mark.parametrize(
    ('field', 'value', 'error'),
    [
        ('environment_id', 'e' * 65, 'invalid_environment_id'),
        ('pool_id', 'p' * 65, 'invalid_pool_id'),
    ],
)
async def test_manager_register_rejects_identity_longer_than_database_contract(
    session_factory,
    field,
    value,
    error,
):
    kwargs = {field: value}

    with pytest.raises(ValueError, match=error):
        await _register_manager(session_factory, manager_id=f'im-invalid-{field}', **kwargs)


async def test_execution_repository_rejects_environment_scope_longer_than_database_contract(session_factory):
    async with session_factory() as session:
        with pytest.raises(ValueError, match='invalid_pool_id'):
            await ExecutionRepository(session, shard_count=8).create_execution(
                CreateExecution(
                    execution_id='execution-invalid-pool',
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={'run_input': {'task_name': 'identity'}},
                    environment_id='environment-a',
                    pool_id='p' * 65,
                ),
                now=utcnow(),
            )


async def _assign_execution(session_factory, execution_id: str):
    execution = await _create_execution(session_factory, execution_id=execution_id)
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()
    assert assignment is not None
    return execution, assignment


async def test_scheduler_claim_filters_manager_capabilities_and_reserves_capacity(session_factory, monkeypatch):
    await _register_manager(
        session_factory, manager_id='live-only', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    await _register_manager(
        session_factory, manager_id='job-manager', modes=[ExecutionMode.JOB], spec_types=['nexus_single']
    )
    execution = await _create_execution(session_factory, execution_id='job-1')

    async with session_factory() as session:
        repo = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await repo.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'job-manager'
    assert assignment.status == AssignmentStatus.PENDING

    await _create_execution(session_factory, execution_id='job-2')
    async with session_factory() as session:
        repo = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        second = await repo.claim_and_assign(shard_ids=[stable_shard('job-2', 8)], now=utcnow())
        await session.commit()

    assert second is None

    async def stale_reserved_slots(_self, _manager_ids):
        return {}

    monkeypatch.setattr(SchedulerRepository, '_reserved_slots_by_manager', stale_reserved_slots)
    await _create_execution(session_factory, execution_id='job-stale-capacity')
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        stale = await scheduler.claim_and_assign(shard_ids=[stable_shard('job-stale-capacity', 8)], now=utcnow())
        await session.commit()

    assert stale is None


async def test_scheduler_placement_target_manager_is_hard_constraint(session_factory):
    await _register_manager(session_factory, manager_id='im-a', capacity_total=1)
    await _register_manager(session_factory, manager_id='im-b', capacity_total=1)
    execution = await _create_execution(
        session_factory,
        execution_id='target-placement',
        placement={'target_manager_id': 'im-b'},
    )

    async with session_factory() as session:
        repo = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await repo.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'im-b'


async def test_scheduler_placement_preferred_manager_falls_back_when_draining(session_factory):
    await _register_manager(session_factory, manager_id='im-a', draining=True, capacity_total=1)
    await _register_manager(session_factory, manager_id='im-b', capacity_total=1)
    execution = await _create_execution(
        session_factory,
        execution_id='preferred-placement',
        placement={'preferred_manager_id': 'im-a', 'environment': 'predefined-default', 'pool': 'default'},
    )

    async with session_factory() as session:
        repo = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await repo.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'im-b'


async def test_scheduler_placement_filters_environment_pool_and_labels(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-a',
        environment_id='predefined-west',
        pool_id='default',
        labels={'region': 'west', 'gpu': 'false'},
    )
    await _register_manager(
        session_factory,
        manager_id='im-b',
        environment_id='predefined-east',
        pool_id='interactive',
        labels={'region': 'east', 'gpu': 'true'},
    )
    execution = await _create_execution(
        session_factory,
        execution_id='label-placement',
        placement={
            'environment': 'predefined-east',
            'pool': 'interactive',
            'labels': {'region': 'east', 'gpu': 'true'},
        },
    )

    async with session_factory() as session:
        repo = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await repo.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'im-b'


async def test_scheduler_claim_batch_assigns_up_to_capacity_without_overselling(session_factory):
    await _register_manager(session_factory, manager_id='im-1', capacity_total=2)
    await _create_execution(session_factory, execution_id='batch-job-1')
    await _create_execution(session_factory, execution_id='batch-job-2')
    await _create_execution(session_factory, execution_id='batch-job-3')
    now = utcnow()

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            no_capacity_backoff=timedelta(seconds=7),
        )
        assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=now)
        await session.commit()

    assert len(assignments) == 2
    assert {assignment.execution_id for assignment in assignments} == {'batch-job-1', 'batch-job-2'}
    assert {assignment.manager_id for assignment in assignments} == {'im-1'}

    async with session_factory() as session:
        pending_count = len(
            (
                await session.scalars(
                    select(AssignmentRow).where(
                        AssignmentRow.manager_id == 'im-1',
                        AssignmentRow.status == AssignmentStatus.PENDING,
                    )
                )
            ).all()
        )
        blocked = await session.get(ExecutionRow, 'batch-job-3')

    assert pending_count == 2
    assert blocked is not None
    assert blocked.status == ExecutionStatus.QUEUED
    assert blocked.current_assignment_id is None
    assert blocked.last_schedule_error == 'no_capacity'
    assert blocked.next_schedule_at == now + timedelta(seconds=7)


async def test_scheduler_balances_batch_assignments_across_equal_capacity_managers(session_factory):
    await _register_manager(session_factory, manager_id='im-a', capacity_total=50)
    await _register_manager(session_factory, manager_id='im-b', capacity_total=50)
    for index in range(10):
        await _create_execution(session_factory, execution_id=f'balance-equal-{index:02d}')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
        await session.commit()

    assert len(assignments) == 10
    assert sorted(Counter(assignment.manager_id for assignment in assignments).values()) == [5, 5]


async def test_scheduler_batches_assignment_sql_instead_of_querying_per_execution(session_factory):
    await _register_manager(session_factory, manager_id='im-batch-a', capacity_total=50)
    await _register_manager(session_factory, manager_id='im-batch-b', capacity_total=50)
    for index in range(30):
        await _create_execution(session_factory, execution_id=f'batched-sql-{index:02d}')

    statements = 0
    engine = session_factory.kw['bind']

    def count_statement(_conn, _cursor, _statement, _parameters, _context, _executemany):
        nonlocal statements
        statements += 1

    event.listen(engine.sync_engine, 'before_cursor_execute', count_statement)
    try:
        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', count_statement)

    assert len(assignments) == 30
    assert statements <= 20


async def test_scheduler_bulk_locks_manager_rows_for_large_assignment_batch(session_factory):
    for index in range(64):
        await _register_manager(session_factory, manager_id=f'im-bulk-lock-{index:02d}', capacity_total=2)
    for index in range(128):
        await _create_execution(session_factory, execution_id=f'bulk-lock-job-{index:03d}')

    statements = 0
    engine = session_factory.kw['bind']

    def count_statement(_conn, _cursor, _statement, _parameters, _context, _executemany):
        nonlocal statements
        statements += 1

    event.listen(engine.sync_engine, 'before_cursor_execute', count_statement)
    try:
        async with session_factory() as session:
            scheduler = SchedulerRepository(
                session,
                shard_count=8,
                manager_healthy_ttl=timedelta(seconds=15),
                scan_limit=128,
                max_assignments=128,
                manager_lock_limit=64,
            )
            assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', count_statement)

    assert len(assignments) == 128
    assert len({assignment.manager_id for assignment in assignments}) == 64
    assert statements <= 25


async def test_scheduler_limits_manager_rows_locked_by_one_assignment_batch(session_factory):
    for index in range(10):
        await _register_manager(session_factory, manager_id=f'im-lock-{index:02d}', capacity_total=10)
    for index in range(30):
        await _create_execution(session_factory, execution_id=f'bounded-manager-lock-{index:02d}')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            manager_lock_limit=3,
        )
        assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
        await session.commit()

    assert len(assignments) == 30
    assert len({assignment.manager_id for assignment in assignments}) == 3


async def test_scheduler_scales_manager_locks_to_owned_shard_share(session_factory):
    for index in range(16):
        await _register_manager(session_factory, manager_id=f'im-shard-share-{index:02d}', capacity_total=10)

    execution_ids: list[str] = []
    candidate = 0
    while len(execution_ids) < 16:
        execution_id = f'shard-share-job-{candidate:03d}'
        candidate += 1
        if stable_shard(execution_id, 8) in {0, 1}:
            execution_ids.append(execution_id)
            await _create_execution(session_factory, execution_id=execution_id)

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            manager_lock_limit=64,
        )
        assignments = await scheduler.claim_and_assign_batch(shard_ids=[0, 1], now=utcnow())
        await session.commit()

    assert len(assignments) == len(execution_ids)
    assert len({assignment.manager_id for assignment in assignments}) == 4


async def test_scheduler_partitions_manager_lock_candidates_by_owned_shards(session_factory):
    for index in range(8):
        await _register_manager(session_factory, manager_id=f'im-partition-{index:02d}', capacity_total=10)
    execution = await _create_execution(session_factory, execution_id='partition-job')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            manager_lock_limit=8,
        )
        managers = await scheduler._healthy_managers(utcnow())
        reserved_slots = await scheduler._reserved_slots_by_manager([manager.manager_id for manager in managers])
        current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        manager_home_shards = scheduler._manager_home_shards(managers)
        left = scheduler._batch_manager_lock_candidates(
            [current],
            managers,
            reserved_slots,
            owned_shard_ids={0, 1, 2, 3},
            manager_home_shards=manager_home_shards,
        )
        right = scheduler._batch_manager_lock_candidates(
            [current],
            managers,
            reserved_slots,
            owned_shard_ids={4, 5, 6, 7},
            manager_home_shards=manager_home_shards,
        )

    assert set(manager_home_shards.values()) == set(range(8))
    assert len(left) == 4
    assert len(right) == 4
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == set(manager_home_shards)


async def test_scheduler_falls_back_to_global_managers_when_partition_has_no_capacity(session_factory):
    for index in range(8):
        await _register_manager(session_factory, manager_id=f'im-partition-fallback-{index:02d}', capacity_total=1)
    execution = await _create_execution(session_factory, execution_id='partition-fallback-job')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            manager_lock_limit=8,
        )
        managers = await scheduler._healthy_managers(utcnow())
        manager_home_shards = scheduler._manager_home_shards(managers)
        owned_shard = manager_home_shards[managers[0].manager_id]
        reserved_slots = {
            manager.manager_id: manager.capacity_total if manager_home_shards[manager.manager_id] == owned_shard else 0
            for manager in managers
        }
        current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        selected = scheduler._batch_manager_lock_candidates(
            [current],
            managers,
            reserved_slots,
            owned_shard_ids={owned_shard},
            manager_home_shards=manager_home_shards,
        )

    assert selected
    assert all(manager_home_shards[manager_id] != owned_shard for manager_id in selected)


async def test_scheduler_lazily_ranks_only_manager_candidates_needed_for_lock_budget(session_factory, monkeypatch):
    manager_count = 128
    execution_count = 64
    for index in range(manager_count):
        await _register_manager(session_factory, manager_id=f'im-lazy-{index:03d}', capacity_total=10)
    for index in range(execution_count):
        await _create_execution(session_factory, execution_id=f'lazy-job-{index:03d}')

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            manager_lock_limit=8,
        )
        managers = await scheduler._healthy_managers(utcnow())
        executions = list(
            await session.scalars(select(ExecutionRow).where(ExecutionRow.execution_id.like('lazy-job-%')))
        )
        manager_home_shards = scheduler._manager_home_shards(managers)
        score_calls = 0

        def deterministic_score(execution, manager, _reserved_slots):
            nonlocal score_calls
            score_calls += 1
            execution_index = int(execution.execution_id.rsplit('-', 1)[1])
            manager_index = int(manager.manager_id.rsplit('-', 1)[1])
            return (int(execution_index % manager_count != manager_index), manager_index, 0, manager.manager_id)

        monkeypatch.setattr(scheduler, '_manager_load_score', deterministic_score)
        selected = scheduler._batch_manager_lock_candidates(
            executions,
            managers,
            {manager.manager_id: 0 for manager in managers},
            owned_shard_ids=set(range(8)),
            manager_home_shards=manager_home_shards,
        )

    assert len(selected) == 8
    assert score_calls <= 8 * manager_count


async def test_scheduler_retries_manager_lock_contention_before_no_capacity_backoff(
    session_factory,
    monkeypatch,
):
    await _register_manager(session_factory, manager_id='im-contended', capacity_total=10)
    execution = await _create_execution(session_factory, execution_id='manager-lock-contended')
    now = utcnow()

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            no_capacity_backoff=timedelta(seconds=7),
        )

        async def lock_no_managers(*_args, **_kwargs):
            return []

        monkeypatch.setattr(scheduler, '_lock_manager_candidates', lock_no_managers)
        claimed = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=now)
        current = await session.get(ExecutionRow, execution.execution_id)
        await session.commit()

    assert claimed is None
    assert current is not None
    assert current.status == ExecutionStatus.QUEUED
    assert current.last_schedule_error == 'manager_lock_contended'
    retry_delay = current.next_schedule_at - now
    assert timedelta(milliseconds=250) <= retry_delay <= timedelta(milliseconds=750)


async def test_scheduler_clears_previous_schedule_error_after_assignment(session_factory):
    await _register_manager(session_factory, manager_id='im-recovered-contention', capacity_total=1)
    execution = await _create_execution(session_factory, execution_id='recovered-manager-lock-contention')
    async with session_factory() as session:
        current = await session.get(ExecutionRow, execution.execution_id)
        assert current is not None
        current.last_schedule_error = 'manager_lock_contended'
        await session.commit()

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        current = await session.get(ExecutionRow, execution.execution_id)
        await session.commit()

    assert assignment is not None
    assert current is not None
    assert current.status == ExecutionStatus.ASSIGNED
    assert current.last_schedule_error is None


async def test_scheduler_balances_batch_assignments_by_capacity_ratio(session_factory):
    await _register_manager(session_factory, manager_id='im-large', capacity_total=100)
    await _register_manager(session_factory, manager_id='im-small', capacity_total=50)
    for index in range(30):
        await _create_execution(session_factory, execution_id=f'balance-weighted-{index:02d}')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
        await session.commit()

    assert len(assignments) == 30
    assert Counter(assignment.manager_id for assignment in assignments) == {
        'im-large': 20,
        'im-small': 10,
    }


async def test_scheduler_balances_high_slot_cost_assignments_until_capacity_is_full(session_factory):
    await _register_manager(session_factory, manager_id='im-a', capacity_total=50)
    await _register_manager(session_factory, manager_id='im-b', capacity_total=50)
    for index in range(12):
        await _create_execution(session_factory, execution_id=f'balance-capacity-{index:02d}', slot_cost=10)
    now = utcnow()

    async with session_factory() as session:
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            no_capacity_backoff=timedelta(seconds=7),
        )
        assignments = await scheduler.claim_and_assign_batch(shard_ids=list(range(8)), now=now)
        queued = (
            await session.scalars(
                select(ExecutionRow).where(
                    ExecutionRow.execution_id.in_(['balance-capacity-10', 'balance-capacity-11'])
                )
            )
        ).all()
        await session.commit()

    assert len(assignments) == 10
    assert Counter(assignment.manager_id for assignment in assignments) == {'im-a': 5, 'im-b': 5}
    assert {execution.status for execution in queued} == {ExecutionStatus.QUEUED}
    assert {execution.last_schedule_error for execution in queued} == {'no_capacity'}
    assert {execution.next_schedule_at for execution in queued} == {now + timedelta(seconds=7)}


async def test_scheduler_reservation_releases_capacity_after_terminal_assignment(session_factory):
    await _register_manager(session_factory, manager_id='im-1', capacity_total=1)
    first, first_assignment = await _assign_execution(session_factory, 'capacity-first')
    await _create_execution(session_factory, execution_id='capacity-second')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        blocked = await scheduler.claim_and_assign(shard_ids=[stable_shard('capacity-second', 8)], now=utcnow())
        await session.commit()

    assert blocked is None

    for terminal_status in (AssignmentStatus.REVOKED, AssignmentStatus.RELEASED, AssignmentStatus.LOST):
        async with session_factory() as session:
            first_row = await session.get(AssignmentRow, first_assignment.assignment_id)
            second_row = await session.get(ExecutionRow, 'capacity-second')
            assert first_row is not None
            assert second_row is not None
            first_row.status = terminal_status
            first_row.updated_at = utcnow()
            second_row.status = ExecutionStatus.QUEUED
            second_row.next_schedule_at = utcnow()
            second_row.current_assignment_id = None
            second_row.current_attempt_id = None
            await session.commit()

        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            claimed = await scheduler.claim_and_assign(shard_ids=[stable_shard('capacity-second', 8)], now=utcnow())
            await session.commit()

        assert claimed is not None
        assert claimed.execution_id == 'capacity-second'

        async with session_factory() as session:
            claimed_row = await session.get(AssignmentRow, claimed.assignment_id)
            second_row = await session.get(ExecutionRow, 'capacity-second')
            assert claimed_row is not None
            assert second_row is not None
            claimed_row.status = AssignmentStatus.REVOKED
            claimed_row.updated_at = utcnow()
            second_row.status = ExecutionStatus.QUEUED
            second_row.next_schedule_at = utcnow()
            second_row.current_assignment_id = None
            second_row.current_attempt_id = None
            await session.commit()

    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(first.execution_id)
    assert current.current_assignment.status in {
        AssignmentStatus.REVOKED,
        AssignmentStatus.RELEASED,
        AssignmentStatus.LOST,
    }


async def test_scheduler_orders_by_schedule_time_priority_and_creation(session_factory):
    await _register_manager(session_factory, manager_id='im-1', capacity_total=3)
    now = utcnow()
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        low = await repo.create_execution(
            CreateExecution(
                execution_id='order-low',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                priority=1,
            ),
            now=now,
        )
        high = await repo.create_execution(
            CreateExecution(
                execution_id='order-high',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                priority=10,
            ),
            now=now,
        )
        future = await repo.create_execution(
            CreateExecution(
                execution_id='order-future',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                priority=100,
            ),
            now=now,
        )
        future_row = await session.get(ExecutionRow, future.execution_id)
        assert future_row is not None
        future_row.next_schedule_at = now + timedelta(seconds=30)
        await session.commit()

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        first = await scheduler.claim_and_assign(shard_ids=list(range(8)), now=now)
        second = await scheduler.claim_and_assign(shard_ids=list(range(8)), now=now)
        third = await scheduler.claim_and_assign(shard_ids=list(range(8)), now=now)
        await session.commit()

    assert first is not None
    assert second is not None
    assert first.execution_id == high.execution_id
    assert second.execution_id == low.execution_id
    assert third is None


async def test_manager_session_fencing_rejects_healthy_replacement_and_invalidates_old_heartbeat(session_factory):
    await _register_manager(session_factory, manager_id='im-1', session_id='old-session')

    async with session_factory() as session:
        repo = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        replaced = await repo.register_session(
            manager_id='im-1',
            session_id='new-session',
            endpoint_url='http://im-1-new.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            environment_id='predefined-default',
            pool_id='default',
            workload='execution',
            force_replace=False,
            now=utcnow(),
        )
        await session.commit()

    assert replaced is False

    async with session_factory() as session:
        repo = ManagerRepository(session, healthy_ttl=timedelta(seconds=15))
        replaced = await repo.register_session(
            manager_id='im-1',
            session_id='new-session',
            endpoint_url='http://im-1-new.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=True,
            now=utcnow(),
        )
        old_heartbeat = await repo.heartbeat(
            manager_id='im-1',
            session_id='old-session',
            capacity_total=1,
            observed_capacity_used=0,
            local_instances={},
            now=utcnow(),
        )
        new_heartbeat = await repo.heartbeat(
            manager_id='im-1',
            session_id='new-session',
            capacity_total=1,
            observed_capacity_used=0,
            local_instances={},
            now=utcnow(),
        )
        await session.commit()

    assert replaced is True
    assert old_heartbeat is False
    assert new_heartbeat is True


async def test_manager_heartbeat_rejects_session_displaced_only_in_registration_row(session_factory):
    await _register_manager(session_factory, manager_id='im-session-race', session_id='old-session')
    displaced_at = utcnow()

    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'im-session-race')
        liveness = await session.get(InstanceManagerLivenessRow, 'im-session-race')
        assert manager is not None
        assert liveness is not None
        manager.session_id = 'new-session'
        previous_heartbeat = liveness.last_heartbeat_at
        await session.commit()

    async with session_factory() as session:
        accepted = await ManagerRepository(session).heartbeat(
            manager_id='im-session-race',
            session_id='old-session',
            capacity_total=1,
            observed_capacity_used=0,
            now=displaced_at,
        )
        await session.commit()

    async with session_factory() as session:
        liveness = await session.get(InstanceManagerLivenessRow, 'im-session-race')
        assert liveness is not None
        assert liveness.last_heartbeat_at == previous_heartbeat
    assert accepted is False


async def test_manager_heartbeat_preserves_registration_capability_snapshot(session_factory):
    registered_at = utcnow()
    async with session_factory() as session:
        repo = ManagerRepository(session)
        assert await repo.register_session(
            manager_id='im-capability',
            session_id='session-capability',
            endpoint_url='http://im-capability.internal:9001',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=2,
            force_replace=False,
            labels={'registered': 'true'},
            bootstrap_version=1,
            artifact_ref_version=1,
            supported_spec_versions={'nexus_single': [1]},
            owner_principal_id='principal-a',
            environment_id='environment-a',
            pool_id='execution-a',
            allocation_id='allocation-a',
            workload='execution',
            allocation_spec_hash='a' * 64,
            capability_snapshot={'runtime': 'python-3.13', 'features': ['run-result-v1']},
            metadata={'endpoint_kind': 'pod_ip', 'pod_uid': 'pod-uid-a'},
            now=registered_at,
        )
        assert await repo.heartbeat(
            manager_id='im-capability',
            session_id='session-capability',
            capacity_total=3,
            observed_capacity_used=1,
            local_instances={'assignment-1': {'execution_id': 'execution-1'}},
            now=registered_at + timedelta(seconds=1),
        )
        await session.commit()

    async with session_factory() as session:
        row = await session.get(InstanceManagerRow, 'im-capability')
        liveness = await session.get(InstanceManagerLivenessRow, 'im-capability')

    assert row is not None
    assert row.endpoint_url == 'http://im-capability.internal:9001'
    assert row.labels_json == {'registered': 'true'}
    assert row.bootstrap_version == 1
    assert row.artifact_ref_version == 1
    assert row.supported_spec_versions_json == {'nexus_single': [1]}
    assert row.supported_modes_json == [ExecutionMode.JOB]
    assert row.supported_spec_types_json == ['nexus_single']
    assert row.supported_instance_modes_json == ['asyncio']
    assert row.owner_principal_id == 'principal-a'
    assert row.environment_id == 'environment-a'
    assert row.pool_id == 'execution-a'
    assert row.allocation_id == 'allocation-a'
    assert row.workload == 'execution'
    assert row.allocation_spec_hash == 'a' * 64
    assert row.capability_snapshot_json == {'runtime': 'python-3.13', 'features': ['run-result-v1']}
    assert row.metadata_json == {'endpoint_kind': 'pod_ip', 'pod_uid': 'pod-uid-a'}
    assert row.capacity_total == 2
    assert row.observed_capacity_used == 0
    assert row.local_instances_json == {}
    assert row.last_heartbeat_at == registered_at
    assert liveness is not None
    assert liveness.session_id == 'session-capability'
    assert liveness.capacity_total == 3
    assert liveness.observed_capacity_used == 1
    assert liveness.last_heartbeat_at == registered_at + timedelta(seconds=1)
    assert liveness.fenced_at is None


async def test_scheduler_requires_ready_allocation_and_copies_allocation_identity(session_factory):
    now = utcnow()
    async with session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='environment-dynamic',
                name='dynamic',
                allocation_type='dynamic',
                provider_type='kubernetes',
                provider_config_ref='cluster-a',
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            EnvironmentAllocationRow(
                allocation_id='allocation-dynamic',
                top_level_execution_id='dynamic-job',
                owner_principal_id='principal-a',
                environment_id='environment-dynamic',
                pool_id='execution',
                workload='execution',
                status='provisioning',
                desired_manager_count=1,
                ready_manager_count=0,
                manager_count=1,
                cpu_millicores=1000,
                memory_bytes=1024,
                slots=1,
                active_batches=0,
                allocation_spec_json={},
                allocation_spec_hash='b' * 64,
                config_version=1,
                status_version=1,
                last_progress_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        execution = ExecutionRow(
            execution_id='dynamic-job',
            owner_principal_id='principal-a',
            created_by='principal-a',
            environment_id='environment-dynamic',
            pool_id='execution',
            allocation_id='allocation-dynamic',
            workload='execution',
            allocation_spec_hash='b' * 64,
            version=1,
            shard_id=0,
            mode=ExecutionMode.JOB,
            spec_type='nexus_single',
            instance_mode='asyncio',
            spec_json={},
            driver_config_json={},
            placement_json={},
            status=ExecutionStatus.QUEUED,
            route_status=RouteStatus.NONE,
            slot_cost=1,
            next_attempt_seq=1,
            attempt_count=0,
            max_attempts=1,
            next_event_seq=1,
            priority=0,
            next_schedule_at=now,
            queue_ttl_seconds=30,
            created_at=now,
            updated_at=now,
        )
        session.add(execution)
        await ManagerRepository(session).register_session(
            manager_id='allocation-dynamic-0',
            session_id='session-dynamic',
            endpoint_url='http://allocation-dynamic-0',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            owner_principal_id='principal-a',
            environment_id='environment-dynamic',
            pool_id='execution',
            allocation_id='allocation-dynamic',
            workload='execution',
            allocation_spec_hash='b' * 64,
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assert await scheduler.claim_and_assign(shard_ids=[0], now=now) is None
        execution = await session.get(ExecutionRow, 'dynamic-job')
        assert execution is not None
        assert execution.first_schedulable_at is None
        assert execution.queue_expires_at is None
        allocation = await session.get(EnvironmentAllocationRow, 'allocation-dynamic')
        allocation.status = 'ready'
        await session.commit()

    schedulable_at = now + timedelta(seconds=1)
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        locked_entities: list[type] = []

        def record_locked_entities(orm_execute_state):
            statement = orm_execute_state.statement
            if getattr(statement, '_for_update_arg', None) is None:
                return
            locked_entities.extend(
                description['entity']
                for description in statement.column_descriptions
                if description.get('entity') is not None
            )

        event.listen(session.sync_session, 'do_orm_execute', record_locked_entities)
        try:
            assignment = await scheduler.claim_and_assign(shard_ids=[0], now=schedulable_at)
            await session.commit()
        finally:
            event.remove(session.sync_session, 'do_orm_execute', record_locked_entities)

    assert assignment is not None
    assert ExecutionRow in locked_entities
    assert EnvironmentAllocationRow not in locked_entities
    assert assignment.environment_id == 'environment-dynamic'
    assert assignment.pool_id == 'execution'
    assert assignment.allocation_id == 'allocation-dynamic'
    assert assignment.workload == 'execution'
    assert assignment.allocation_spec_hash == 'b' * 64
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, 'dynamic-job')
    assert execution is not None
    assert execution.first_schedulable_at == schedulable_at
    assert execution.queue_expires_at == schedulable_at + timedelta(seconds=30)


async def test_scheduler_claim_is_fenced_by_current_shard_lease_owner(session_factory):
    await _register_manager(session_factory, manager_id='lease-fence-manager')
    execution = await _create_execution(session_factory, execution_id='lease-fenced-execution')
    now = utcnow()
    async with session_factory() as session:
        session.add(
            ShardLeaseRow(
                shard_group='default',
                shard_id=execution.shard_id,
                shard_count=8,
                owner_id='scheduler-b',
                lease_expires_at=now + timedelta(minutes=1),
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        repository = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            lease_owner_id='scheduler-a',
            lease_shard_group='default',
        )
        assert await repository.claim_and_assign(shard_ids=[execution.shard_id], now=now) is None
        await session.commit()

    async with session_factory() as session:
        current = await session.get(ExecutionRow, execution.execution_id)
        lease = await session.get(ShardLeaseRow, ('default', execution.shard_id))
        assert current is not None
        assert current.status == ExecutionStatus.QUEUED
        assert lease is not None
        lease.owner_id = 'scheduler-a'
        lease.lease_expires_at = now + timedelta(minutes=1)
        await session.commit()

    async with session_factory() as session:
        repository = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            lease_owner_id='scheduler-a',
            lease_shard_group='default',
        )
        assignment = await repository.claim_and_assign(shard_ids=[execution.shard_id], now=now)
        await session.commit()

    assert assignment is not None


async def test_scheduler_predefined_allocation_matches_environment_pool_and_execution_workload(session_factory):
    now = utcnow()
    async with session_factory() as session:
        session.add(
            EnvironmentRow(
                environment_id='environment-predefined',
                name='predefined',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits_json={},
                default_timeouts_json={},
                config_version=1,
                status='enabled',
                created_by='admin',
                updated_by='admin',
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            EnvironmentAllocationRow(
                allocation_id='allocation-predefined',
                top_level_execution_id='predefined-job',
                owner_principal_id='principal-a',
                environment_id='environment-predefined',
                pool_id='interactive',
                workload='execution',
                status='ready',
                desired_manager_count=0,
                ready_manager_count=0,
                manager_count=0,
                cpu_millicores=0,
                memory_bytes=0,
                slots=1,
                active_batches=0,
                allocation_spec_json={},
                allocation_spec_hash='c' * 64,
                config_version=1,
                status_version=1,
                last_progress_at=now,
                created_at=now,
                ready_at=now,
                updated_at=now,
            )
        )
        session.add(
            ExecutionRow(
                execution_id='predefined-job',
                owner_principal_id='principal-a',
                created_by='principal-a',
                environment_id='environment-predefined',
                pool_id='interactive',
                allocation_id='allocation-predefined',
                workload='execution',
                allocation_spec_hash='c' * 64,
                version=1,
                shard_id=0,
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec_json={},
                driver_config_json={},
                placement_json={},
                status=ExecutionStatus.QUEUED,
                route_status=RouteStatus.NONE,
                slot_cost=1,
                next_attempt_seq=1,
                attempt_count=0,
                max_attempts=1,
                next_event_seq=1,
                priority=0,
                next_schedule_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        await ManagerRepository(session).register_session(
            manager_id='predefined-manager',
            session_id='predefined-session',
            endpoint_url='http://predefined-manager',
            supported_modes=[ExecutionMode.JOB],
            supported_spec_types=['nexus_single'],
            supported_instance_modes=['asyncio'],
            capacity_total=1,
            force_replace=False,
            environment_id='environment-predefined',
            pool_id='interactive',
            workload='execution',
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[0], now=now)
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'predefined-manager'
    assert assignment.allocation_id == 'allocation-predefined'


async def test_manager_ack_and_mark_methods_reject_stale_assignments(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'stale-manager-op')

    async with session_factory() as session:
        manager = ManagerRepository(session)
        with pytest.raises(ValueError, match='assignment_session_mismatch'):
            await manager.ack_assignment(assignment.assignment_id, 'im-1', 'wrong-session', now=utcnow())
        await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        with pytest.raises(ValueError, match='assignment_not_pending'):
            await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())

        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        execution_row.current_assignment_id = 'other-assignment'
        with pytest.raises(ValueError, match='execution_assignment_mismatch'):
            await manager.mark_running(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info={},
                now=utcnow(),
            )
        await session.rollback()


async def test_manager_bulk_acks_pending_assignments_without_per_assignment_queries(session_factory):
    await _register_manager(session_factory, manager_id='im-bulk-ack', capacity_total=10)
    for index in range(10):
        await _create_execution(session_factory, execution_id=f'bulk-ack-{index:02d}')

    async with session_factory() as session:
        assignments = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            scan_limit=10,
            max_assignments=10,
        ).claim_and_assign_batch(shard_ids=list(range(8)), now=utcnow())
        await session.commit()

    assignment_ids = [assignment.assignment_id for assignment in assignments]
    statements = 0
    engine = session_factory.kw['bind']

    def count_statement(_conn, _cursor, _statement, _parameters, _context, _executemany):
        nonlocal statements
        statements += 1

    event.listen(engine.sync_engine, 'before_cursor_execute', count_statement)
    try:
        async with session_factory() as session:
            acknowledged = await ManagerRepository(session).ack_assignments(
                assignment_ids,
                'im-bulk-ack',
                assignments[0].manager_session_id,
                now=utcnow(),
            )
            await session.commit()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', count_statement)

    assert [assignment.assignment_id for assignment in acknowledged] == assignment_ids
    assert statements <= 10
    async with session_factory() as session:
        assignment_statuses = list(
            await session.scalars(select(AssignmentRow.status).where(AssignmentRow.assignment_id.in_(assignment_ids)))
        )
        execution_rows = list(
            await session.scalars(
                select(ExecutionRow).where(ExecutionRow.execution_id.in_([item.execution_id for item in assignments]))
            )
        )
        events = list(
            await session.scalars(
                select(ExecutionEventRow).where(
                    ExecutionEventRow.execution_id.in_([item.execution_id for item in assignments]),
                    ExecutionEventRow.source == 'manager',
                )
            )
        )

    assert assignment_statuses == [AssignmentStatus.ACTIVE] * 10
    assert {row.status for row in execution_rows} == {ExecutionStatus.STARTING}
    assert len(events) == 20


async def test_manager_cannot_complete_cloud_execution_without_ready_result_artifact(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'missing-ready-result')

    async with session_factory() as session:
        manager = ManagerRepository(session)
        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        execution_row.environment_id = 'env-cloud-result'
        await manager.ack_assignment(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            now=utcnow(),
        )
        with pytest.raises(TemporaryUnavailableError) as exc_info:
            await manager.complete_execution(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                result={'ok': True},
                now=utcnow(),
            )
        assert exc_info.value.code == 'execution_result_artifact_not_ready'
        await session.rollback()

    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        assert row.status != ExecutionStatus.SUCCEEDED


async def test_manager_complete_merges_final_runtime_info_atomically(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'complete-with-runtime-info')
    final_runtime_info = {
        '_artifacts': {
            'single': {
                'version': 4,
                'run_id': execution.execution_id,
            }
        }
    }

    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.complete_execution(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            result={'success': True},
            runtime_info=final_runtime_info,
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
        execution_row = await session.get(ExecutionRow, execution.execution_id)

    assert assignment_row is not None
    assert execution_row is not None
    assert assignment_row.runtime_info_json == final_runtime_info
    assert assignment_row.status == AssignmentStatus.RELEASED
    assert execution_row.status == ExecutionStatus.SUCCEEDED


async def test_manager_mark_running_preserves_runtime_info_on_empty_update(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    _, assignment = await _assign_execution(session_factory, 'runtime-info-preserve')

    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url='http://im-1.internal/executions/asn-1',
            endpoint_ready=True,
            runtime_info={'main': {'name': 'main', 'proxy_url': '/v1/executions/runtime-info-preserve/runtimes/main'}},
            now=utcnow(),
        )
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url='http://im-1.internal/executions/asn-1',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        row = await session.get(AssignmentRow, assignment.assignment_id)

    assert row is not None
    assert row.runtime_info_json == {
        'runtimes': {'main': {'name': 'main', 'proxy_url': '/v1/executions/runtime-info-preserve/runtimes/main'}}
    }


async def test_manager_runtime_info_keeps_runtimes_input_and_artifacts_separate(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    _, assignment = await _assign_execution(session_factory, 'runtime-info-structured')

    runtime = {'name': 'main', 'debug_gui_url': '/debug/gui'}
    artifact = {'events_path': '/tmp/events.jsonl'}
    input_payload = {'turn_id': 'turn-1', 'request': {'hint': 'choose'}}
    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url='http://im-1.internal/executions/asn-1',
            endpoint_ready=True,
            runtime_info={'main': runtime, '_artifacts': {'live': artifact}},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            payload=input_payload,
            now=utcnow(),
        )
        await manager.update_runtime_info(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            runtime_info={
                '_artifacts': {
                    'live': {**artifact, 'events_cursor': 9, 'current_turn_id': 'turn-1'},
                }
            },
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        row = await session.get(AssignmentRow, assignment.assignment_id)
        execution = await session.get(ExecutionRow, assignment.execution_id)

    assert row is not None
    assert execution is not None
    assert execution.status == ExecutionStatus.WAITING_INPUT
    assert row.runtime_info_json == {
        'runtimes': {'main': runtime},
        'input': {'turn_id': 'turn-1'},
        '_artifacts': {'live': {**artifact, 'events_cursor': 9, 'current_turn_id': 'turn-1'}},
    }


async def test_manager_allows_runtime_info_update_while_execution_is_stopping(session_factory):
    await _register_manager(session_factory, manager_id='im-1', modes=[ExecutionMode.LIVE], spec_types=['nexus_live'])
    execution = await _create_execution(
        session_factory,
        execution_id='stopping-runtime-info',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None

        manager = ManagerRepository(session)
        await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url='http://im-1.internal/executions/stopping-runtime-info',
            endpoint_ready=True,
            runtime_info={'_artifacts': {'live': {'events_cursor': 5}}},
            now=utcnow(),
        )

        repo = ExecutionRepository(session, shard_count=8)
        stopped = await repo.request_stop(execution.execution_id, now=utcnow())
        assert stopped.status == ExecutionStatus.STOPPING

        await manager.update_runtime_info(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            runtime_info={'_artifacts': {'live': {'events_cursor': 7, 'status': 'killed'}}},
            now=utcnow(),
        )
        await session.commit()

    async with session_factory() as session:
        row = await session.get(AssignmentRow, assignment.assignment_id)

    assert row is not None
    assert row.runtime_info_json['_artifacts']['live'] == {'events_cursor': 7, 'status': 'killed'}


async def test_stop_queued_and_pending_execution_does_not_create_command(session_factory):
    queued = await _create_execution(session_factory, execution_id='queued-stop')

    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        stopped = await repo.request_stop(queued.execution_id, now=utcnow())
        await session.commit()

    assert stopped.status == ExecutionStatus.KILLED

    await _register_manager(session_factory, manager_id='im-1')
    pending = await _create_execution(session_factory, execution_id='pending-stop')
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[pending.shard_id], now=utcnow())
        repo = ExecutionRepository(session, shard_count=8)
        stopped = await repo.request_stop(pending.execution_id, now=utcnow())
        commands = await repo.list_commands(pending.execution_id)
        await session.commit()

    assert assignment is not None
    assert stopped.status == ExecutionStatus.KILLED
    assert stopped.current_assignment.status == AssignmentStatus.REVOKED
    assert commands == []


async def test_stop_running_execution_creates_bound_stop_command(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution = await _create_execution(session_factory, execution_id='running-stop')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        manager = ManagerRepository(session)
        assert assignment is not None
        await manager.ack_assignment(assignment.assignment_id, 'im-1', assignment.manager_session_id, now=utcnow())
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url='http://im-1.internal:9001/v1/internal/executions/running-stop',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        repo = ExecutionRepository(session, shard_count=8)
        stopped = await repo.request_stop(execution.execution_id, now=utcnow())
        stopped_again = await repo.request_stop(execution.execution_id, now=utcnow())
        commands = await repo.list_commands(execution.execution_id)
        await session.commit()

    assert stopped.status == ExecutionStatus.STOPPING
    assert stopped_again.status == ExecutionStatus.STOPPING
    assert len(commands) == 1
    assert commands[0].type == 'stop.request'
    assert commands[0].status == CommandStatus.PENDING
    assert commands[0].assignment_id == assignment.assignment_id
    assert commands[0].attempt_id == assignment.attempt_id


async def test_bulk_stop_preserves_queued_pending_and_running_semantics(session_factory):
    await _register_manager(session_factory, manager_id='im-1', capacity_total=2)
    queued = await _create_execution(session_factory, execution_id='bulk-stop-queued')
    pending = await _create_execution(session_factory, execution_id='bulk-stop-pending')
    running = await _create_execution(session_factory, execution_id='bulk-stop-running')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        pending_assignment = await scheduler.claim_and_assign(shard_ids=[pending.shard_id], now=utcnow())
        running_assignment = await scheduler.claim_and_assign(shard_ids=[running.shard_id], now=utcnow())
        assert pending_assignment is not None
        assert running_assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            running_assignment.assignment_id,
            'im-1',
            running_assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.mark_running(
            running_assignment.assignment_id,
            'im-1',
            running_assignment.manager_session_id,
            endpoint_url='http://im-1.internal/bulk-stop-running',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )

        changed = await ExecutionRepository(session, shard_count=8).request_stop_many(
            [running.execution_id, queued.execution_id, pending.execution_id],
            stop_reason='user_stop',
            now=utcnow(),
        )
        await session.commit()

    assert changed == 3
    async with session_factory() as session:
        queued_row = await session.get(ExecutionRow, queued.execution_id)
        pending_row = await session.get(ExecutionRow, pending.execution_id)
        running_row = await session.get(ExecutionRow, running.execution_id)
        pending_assignment_row = await session.get(AssignmentRow, pending_assignment.assignment_id)
        commands = await ExecutionRepository(session, shard_count=8).list_commands(running.execution_id)

    assert queued_row is not None
    assert queued_row.status == ExecutionStatus.KILLED
    assert pending_row is not None
    assert pending_row.status == ExecutionStatus.KILLED
    assert pending_assignment_row is not None
    assert pending_assignment_row.status == AssignmentStatus.REVOKED
    assert running_row is not None
    assert running_row.status == ExecutionStatus.STOPPING
    assert len(commands) == 1
    assert commands[0].type == 'stop.request'
    assert commands[0].assignment_id == running_assignment.assignment_id


async def test_bulk_stop_lock_queries_omit_unneeded_payload_columns(session_factory):
    execution = await _create_execution(session_factory, execution_id='bulk-stop-narrow-projection')
    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            changed = await ExecutionRepository(session, shard_count=8).request_stop_many(
                [execution.execution_id],
                now=utcnow(),
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert changed == 1
    ordered = next(
        statement.lower() for statement in statements if 'order by executions.execution_id' in statement.lower()
    )
    assert 'executions.spec_json' not in ordered
    assert 'executions.driver_config_json' not in ordered
    assert 'executions.result_json' not in ordered
    assert 'executions.error_json' not in ordered


async def test_reconciler_lock_queries_omit_assignment_and_manager_snapshots(session_factory):
    await _register_manager(session_factory, manager_id='reconciler-query-manager')
    execution = await _create_execution(session_factory, execution_id='reconciler-query-execution')
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        await session.commit()
    assert assignment is not None

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            reconciler = ReconcilerRepository(session)
            pairs = await reconciler._lock_assignment_candidates([
                (execution.execution_id, assignment.assignment_id),
            ])
            managers = await reconciler._lock_managers([assignment.manager_id])
            await session.rollback()
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert len(pairs) == 1
    assert list(managers) == [assignment.manager_id]
    assignment_select = next(
        statement.lower()
        for statement in statements
        if 'order by assignments.execution_id, assignments.assignment_id' in statement.lower()
    )
    manager_select = next(
        statement.lower() for statement in statements if 'order by instance_managers.manager_id' in statement.lower()
    )
    assert 'assignments.runtime_info_json' not in assignment_select
    assert 'assignments.trace_carrier_json' not in assignment_select
    assert 'assignments.endpoint_error' not in assignment_select
    assert 'instance_managers.local_instances_json' not in manager_select
    assert 'instance_managers.metadata_json' not in manager_select
    assert 'instance_managers.capability_snapshot_json' not in manager_select


async def test_manager_mark_running_does_not_revive_execution_stopped_by_another_session(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'stale-manager-running-after-stop')

    async with session_factory() as manager_session:
        manager = ManagerRepository(manager_session)
        await manager.ack_assignment(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager_session.commit()
        cached_assignment = await manager_session.get(AssignmentRow, assignment.assignment_id)
        cached_execution = await manager_session.get(ExecutionRow, execution.execution_id)
        assert cached_assignment is not None
        assert cached_assignment.status == AssignmentStatus.ACTIVE
        assert cached_execution is not None
        assert cached_execution.status == ExecutionStatus.STARTING

        async with session_factory() as stop_session:
            stopped = await ExecutionRepository(stop_session, shard_count=8).request_stop(
                execution.execution_id,
                now=utcnow(),
            )
            await stop_session.commit()

        assert stopped.status == ExecutionStatus.STOPPING
        with pytest.raises(DomainConflictError) as exc_info:
            await manager.mark_running(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                endpoint_url=None,
                endpoint_ready=False,
                runtime_info={},
                now=utcnow(),
            )
        assert exc_info.value.code == 'execution_not_active'
        assert exc_info.value.detail == ExecutionStatus.STOPPING
        await manager_session.rollback()

    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
    assert row is not None
    assert row.status == ExecutionStatus.STOPPING


async def test_request_stop_refreshes_assignment_state_loaded_before_manager_ack(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'stop-races-manager-ack')

    async with session_factory() as stop_session:
        stale_execution = await stop_session.get(ExecutionRow, execution.execution_id)
        stale_assignment = await stop_session.get(AssignmentRow, assignment.assignment_id)
        assert stale_execution is not None
        assert stale_execution.status == ExecutionStatus.ASSIGNED
        assert stale_assignment is not None
        assert stale_assignment.status == AssignmentStatus.PENDING

        async with session_factory() as manager_session:
            await ManagerRepository(manager_session).ack_assignment(
                assignment.assignment_id,
                'im-1',
                assignment.manager_session_id,
                now=utcnow(),
            )
            await manager_session.commit()

        stopped = await ExecutionRepository(stop_session, shard_count=8).request_stop(
            execution.execution_id,
            now=utcnow(),
        )
        commands = await ExecutionRepository(stop_session, shard_count=8).list_commands(execution.execution_id)
        await stop_session.commit()

    assert stopped.status == ExecutionStatus.STOPPING
    assert stopped.current_assignment is not None
    assert stopped.current_assignment.status == AssignmentStatus.ACTIVE
    assert len(commands) == 1
    assert commands[0].type == 'stop.request'


async def test_request_stop_terminalizes_active_execution_with_revoked_assignment(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution, assignment = await _assign_execution(session_factory, 'stop-orphaned-active-execution')

    async with session_factory() as session:
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.mark_running(
            assignment.assignment_id,
            'im-1',
            assignment.manager_session_id,
            endpoint_url=None,
            endpoint_ready=False,
            runtime_info={},
            now=utcnow(),
        )
        assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
        assert assignment_row is not None
        assignment_row.status = AssignmentStatus.REVOKED
        await session.commit()

    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        stopped = await repository.request_stop(execution.execution_id, now=utcnow())
        commands = await repository.list_commands(execution.execution_id)
        await session.commit()

    assert stopped.status == ExecutionStatus.KILLED
    assert stopped.current_assignment is not None
    assert stopped.current_assignment.status == AssignmentStatus.REVOKED
    assert commands == []


async def test_reconciler_queue_ttl_starts_at_first_schedulable_time_for_allocated_execution(session_factory):
    now = utcnow()
    execution = await _create_execution(session_factory, execution_id='allocation-queue-ttl')
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        row.allocation_id = 'allocation-not-ready'
        row.queue_ttl_seconds = 1
        row.created_at = now - timedelta(minutes=5)
        row.updated_at = row.created_at
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        assert await reconciler.expire_queued(now=now) == 0
        row = await session.get(ExecutionRow, execution.execution_id)
        row.first_schedulable_at = now - timedelta(seconds=2)
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        assert await reconciler.expire_queued(now=now) == 1
        await session.commit()


async def test_reconciler_queue_deadline_scan_is_not_blocked_by_older_long_ttl_rows(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repository = ExecutionRepository(session, shard_count=8)
        for execution_id, created_at, queue_ttl_seconds in (
            ('long-ttl-oldest', now - timedelta(minutes=10), 3600),
            ('long-ttl-older', now - timedelta(minutes=9), 3600),
            ('short-ttl-expired-first', now - timedelta(seconds=10), 1),
            ('short-ttl-expired-second', now - timedelta(seconds=5), 1),
        ):
            await repository.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                    queue_ttl_seconds=queue_ttl_seconds,
                ),
                now=created_at,
            )
        await session.commit()

    async with session_factory() as session:
        changed = await ReconcilerRepository(session, batch_size=1).expire_queued(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        expired_first = await session.get(ExecutionRow, 'short-ttl-expired-first')
        expired_second = await session.get(ExecutionRow, 'short-ttl-expired-second')
        oldest = await session.get(ExecutionRow, 'long-ttl-oldest')
        older = await session.get(ExecutionRow, 'long-ttl-older')
    assert expired_first is not None
    assert expired_first.status == ExecutionStatus.EXPIRED
    assert expired_second is not None
    assert expired_second.status == ExecutionStatus.QUEUED
    assert oldest is not None
    assert oldest.status == ExecutionStatus.QUEUED
    assert older is not None
    assert older.status == ExecutionStatus.QUEUED

    async with session_factory() as session:
        changed = await ReconcilerRepository(session, batch_size=1).expire_queued(now=now)
        await session.commit()
    assert changed == 1
    async with session_factory() as session:
        expired_second = await session.get(ExecutionRow, 'short-ttl-expired-second')
    assert expired_second is not None
    assert expired_second.status == ExecutionStatus.EXPIRED


async def test_reconciler_only_processes_owned_execution_shards(session_factory):
    now = utcnow()
    owned = await _create_execution(
        session_factory,
        execution_id='reconciler-owned-shard',
    )
    other = await _create_execution(
        session_factory,
        execution_id='reconciler-other-shard',
    )
    if owned.shard_id == other.shard_id:
        other = await _create_execution(
            session_factory,
            execution_id='reconciler-other-shard-2',
        )
    assert owned.shard_id != other.shard_id

    async with session_factory() as session:
        for execution_id in (owned.execution_id, other.execution_id):
            row = await session.get(ExecutionRow, execution_id)
            assert row is not None
            row.queue_expires_at = now - timedelta(seconds=1)
        await session.commit()

    async with session_factory() as session:
        changed = await ReconcilerRepository(
            session,
            shard_ids=[owned.shard_id],
            shard_count=8,
        ).expire_queued(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        owned_row = await session.get(ExecutionRow, owned.execution_id)
        other_row = await session.get(ExecutionRow, other.execution_id)
    assert owned_row is not None
    assert owned_row.status == ExecutionStatus.EXPIRED
    assert other_row is not None
    assert other_row.status == ExecutionStatus.QUEUED


async def test_reconciler_rolls_back_when_final_shard_fence_is_lost(session_factory):
    now = utcnow()
    execution = await _create_execution(
        session_factory,
        execution_id='reconciler-final-fence',
    )
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.queue_expires_at = now - timedelta(seconds=1)
        session.add(
            ShardLeaseRow(
                shard_group='execution-reconciler',
                shard_id=execution.shard_id,
                shard_count=8,
                owner_id='reconciler-b',
                lease_expires_at=now + timedelta(minutes=1),
                updated_at=now,
            )
        )
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(
            session,
            shard_ids=[execution.shard_id],
            shard_count=8,
            lease_owner_id='reconciler-a',
        )
        assert await reconciler.expire_queued(now=now) == 1
        assert not await reconciler.finalize_shard_fence(now=now)

    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
    assert row is not None
    assert row.status == ExecutionStatus.QUEUED


async def test_reconciler_rolls_back_without_marking_lease_lost_when_final_fence_is_contended(
    session_factory,
    monkeypatch,
):
    now = utcnow()
    execution = await _create_execution(
        session_factory,
        execution_id='reconciler-contended-fence',
    )
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.queue_expires_at = now - timedelta(seconds=1)
        await session.commit()

    async def contended_fence(self, *, owner_id, shard_group, shard_ids, now):
        del self, owner_id, shard_group, shard_ids, now
        return ShardLeaseFenceStatus.CONTENDED

    monkeypatch.setattr(ShardLeaseRepository, 'fence', contended_fence)
    async with session_factory() as session:
        reconciler = ReconcilerRepository(
            session,
            shard_ids=[execution.shard_id],
            shard_count=8,
            lease_owner_id='reconciler-a',
        )
        assert await reconciler.expire_queued(now=now) == 1
        assert not await reconciler.finalize_shard_fence(now=now)
        assert not reconciler.lease_fence_lost

    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
    assert row is not None
    assert row.status == ExecutionStatus.QUEUED


async def test_execution_deadlines_are_materialized_once_at_state_boundaries(session_factory):
    await _register_manager(session_factory, manager_id='deadline-manager')
    created_at = utcnow()
    started_at = created_at + timedelta(seconds=5)
    stopped_at = started_at + timedelta(seconds=10)
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='execution-deadlines',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                queue_ttl_seconds=30,
                execution_ttl_seconds=60,
                stop_timeout_seconds=20,
            ),
            now=created_at,
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        assert row.queue_expires_at == created_at + timedelta(seconds=30)
        assert row.execution_expires_at is None
        assert row.stop_deadline_at is None

        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=30),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=created_at + timedelta(seconds=1))
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=created_at + timedelta(seconds=2),
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url=None,
            endpoint_ready=False,
            runtime_info={},
            now=started_at,
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        assert row.execution_expires_at == started_at + timedelta(seconds=60)
        row.execution_ttl_seconds = 9999

        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={'request_id': 'input-1'},
            now=started_at + timedelta(seconds=5),
        )
        assert row.execution_expires_at == started_at + timedelta(seconds=60)

        await ExecutionRepository(session, shard_count=8).request_stop(execution.execution_id, now=stopped_at)
        assert row.stop_deadline_at == stopped_at + timedelta(seconds=20)
        row.stop_timeout_seconds = 9999
        row.updated_at = stopped_at + timedelta(minutes=10)
        assert row.stop_deadline_at == stopped_at + timedelta(seconds=20)
        await session.commit()


async def test_reconciler_stopping_deadline_is_not_extended_by_later_updates(session_factory):
    await _register_manager(session_factory, manager_id='stop-deadline-manager')
    started_at = utcnow()
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='stop-deadline',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                stop_timeout_seconds=5,
            ),
            now=started_at,
        )
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=30),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=started_at)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=started_at,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url=None,
            endpoint_ready=False,
            runtime_info={},
            now=started_at,
        )
        await ExecutionRepository(session, shard_count=8).request_stop(execution.execution_id, now=started_at)
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.updated_at = started_at + timedelta(seconds=4)
        await session.commit()

    async with session_factory() as session:
        changed = await ReconcilerRepository(session).reconcile_stopping_timeouts(now=started_at + timedelta(seconds=6))
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
    assert row is not None
    assert row.status == ExecutionStatus.KILLED


async def test_stop_terminal_execution_is_rejected_but_stopping_and_killed_are_idempotent(session_factory):
    queued = await _create_execution(session_factory, execution_id='terminal-stop')
    async with session_factory() as session:
        row = await session.get(ExecutionRow, queued.execution_id)
        assert row is not None
        row.status = ExecutionStatus.SUCCEEDED
        row.finished_at = utcnow()
        repo = ExecutionRepository(session, shard_count=8)
        with pytest.raises(DomainConflictError) as exc_info:
            await repo.request_stop(queued.execution_id, now=utcnow())
        assert exc_info.value.code == 'execution_not_stoppable'
        assert exc_info.value.detail == 'cannot stop execution in status succeeded'
        row.status = ExecutionStatus.KILLED
        stopped = await repo.request_stop(queued.execution_id, now=utcnow())
        await session.commit()

    assert stopped.status == ExecutionStatus.KILLED


async def test_stop_rejection_does_not_dirty_live_route_status(session_factory):
    live = await _create_execution(
        session_factory,
        execution_id='terminal-live-stop',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        row = await session.get(ExecutionRow, live.execution_id)
        assert row is not None
        row.status = ExecutionStatus.SUCCEEDED
        row.finished_at = utcnow()
        repo = ExecutionRepository(session, shard_count=8)
        with pytest.raises(DomainConflictError) as exc_info:
            await repo.request_stop(live.execution_id, now=utcnow())
        assert exc_info.value.code == 'execution_not_stoppable'
        assert exc_info.value.detail == 'cannot stop execution in status succeeded'
        await session.commit()

    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(live.execution_id)

    assert current.status == ExecutionStatus.SUCCEEDED
    assert current.route_status == RouteStatus.ACTIVE


async def test_reconciler_marks_stale_stopping_execution_killed(session_factory):
    await _register_manager(
        session_factory, manager_id='im-stop', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    stop_now = utcnow()
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='stopping-timeout',
                mode=ExecutionMode.LIVE,
                spec_type='nexus_live',
                instance_mode='asyncio',
                spec={},
                route_status=RouteStatus.ACTIVE,
                stop_timeout_seconds=3,
            ),
            now=stop_now,
        )
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=stop_now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(assignment.assignment_id, 'im-stop', assignment.manager_session_id, now=stop_now)
        await manager.mark_running(
            assignment.assignment_id,
            'im-stop',
            assignment.manager_session_id,
            endpoint_url='http://im-stop.internal:9001/v1/internal/executions/stopping-timeout',
            endpoint_ready=True,
            runtime_info={},
            now=stop_now,
        )
        await ExecutionRepository(session, shard_count=8).request_stop(execution.execution_id, now=stop_now)
        command = await session.scalar(
            select(ExecutionCommandRow).where(
                ExecutionCommandRow.execution_id == execution.execution_id,
                ExecutionCommandRow.type == 'stop.request',
            )
        )
        assert command is not None
        command.status = CommandStatus.DONE
        command.completed_at = stop_now
        command.updated_at = stop_now
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.reconcile_stopping_timeouts(now=stop_now + timedelta(seconds=4))
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution('stopping-timeout')

    assert current.status == ExecutionStatus.KILLED
    assert current.route_status == RouteStatus.CLOSED
    assert current.current_assignment.status == AssignmentStatus.RELEASED


async def test_reconciler_revalidates_manager_stop_completion_after_timeout_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-stop-race')
    stop_now = utcnow()
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='stopping-timeout-race',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                stop_timeout_seconds=1,
            ),
            now=stop_now,
        )
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=stop_now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=stop_now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-stop-race.internal',
            endpoint_ready=True,
            runtime_info={},
            now=stop_now,
        )
        await ExecutionRepository(session, shard_count=8).request_stop(execution.execution_id, now=stop_now)

        reconciler = ReconcilerRepository(session)
        lock_candidates = reconciler._lock_execution_candidates

        async def complete_stop_before_lock(candidates):
            await manager.mark_killed(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                now=stop_now + timedelta(seconds=2),
            )
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_execution_candidates', complete_stop_before_lock)
        changed = await reconciler.reconcile_stopping_timeouts(now=stop_now + timedelta(seconds=2))
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.KILLED
    assert current.current_assignment.status == AssignmentStatus.RELEASED


async def test_reconciler_skips_deferred_batch_parent_stopping_timeout(session_factory):
    stop_now = utcnow()
    async with session_factory() as session:
        execution = await ExecutionRepository(session, shard_count=8).create_execution(
            CreateExecution(
                execution_id='batch-stopping-timeout',
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={},
                route_status=RouteStatus.NONE,
                stop_timeout_seconds=3,
            ),
            now=stop_now,
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.STOPPING
        row.updated_at = stop_now
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.reconcile_stopping_timeouts(now=stop_now + timedelta(seconds=4))
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution('batch-stopping-timeout')

    assert current.status == ExecutionStatus.STOPPING


async def test_gateway_command_binds_current_assignment_and_rejects_stale_route(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='live-1',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        manager = ManagerRepository(session)
        assert assignment is not None
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/live-1',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={'prompt': 'choose'},
            now=utcnow(),
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        command = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Alpha'},
            idempotency_key='input-1',
            now=utcnow(),
        )
        await session.commit()

    assert command.assignment_id == assignment.assignment_id
    assert command.attempt_id == assignment.attempt_id

    async with session_factory() as session:
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        duplicate = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Ignored'},
            idempotency_key='input-1',
            now=utcnow(),
        )
        await session.commit()

    assert duplicate.command_id == command.command_id
    assert duplicate.payload == {'text': 'Alpha'}

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        await reconciler.reconcile_manager_dead(now=utcnow())
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        with pytest.raises(
            (TemporaryUnavailableError, DomainConflictError, ResourceGoneError),
            match='assignment_unavailable|route_not_active|execution_not_ready|execution_closed',
        ):
            await gateway.submit_command(
                execution.execution_id,
                command_type='input.submit',
                payload={'text': 'Beta'},
                idempotency_key='input-2',
                now=utcnow(),
            )
        await session.commit()


async def test_gateway_rejects_command_when_live_execution_not_waiting_for_input(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='not-waiting',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/not-waiting',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        with pytest.raises(ValueError, match='execution_not_ready'):
            await gateway.submit_command(
                execution.execution_id,
                command_type='input.submit',
                payload={'text': 'Alpha'},
                idempotency_key=None,
                now=utcnow(),
            )
        await session.commit()


async def test_manager_does_not_claim_command_for_stale_current_assignment(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='stale-command',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        first = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert first is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(first.assignment_id, first.manager_id, first.manager_session_id, now=utcnow())
        await manager.mark_running(
            first.assignment_id,
            first.manager_id,
            first.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/stale-command',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            first.assignment_id,
            first.manager_id,
            first.manager_session_id,
            payload={'prompt': 'choose'},
            now=utcnow(),
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        command = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Alpha'},
            idempotency_key='input-stale',
            now=utcnow(),
        )

        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        second = AssignmentRow(
            assignment_id='asn-new-current',
            execution_id=execution.execution_id,
            attempt_id='stale-command-attempt-new',
            attempt_seq=2,
            manager_id=first.manager_id,
            manager_session_id=first.manager_session_id,
            status=AssignmentStatus.ACTIVE,
            slot_cost=1,
            endpoint_ready=True,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/asn-new-current',
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(second)
        execution_row.current_assignment_id = second.assignment_id
        execution_row.current_attempt_id = second.attempt_id
        execution_row.updated_at = utcnow()

        claimed = await manager.claim_commands(first.manager_id, first.manager_session_id, now=utcnow())
        command_row = await session.get(ExecutionCommandRow, command.command_id)
        await session.commit()

    assert claimed == []
    assert command_row is not None
    assert command_row.status == CommandStatus.CANCELLED


async def test_manager_claim_commands_rejects_stale_attempt_and_session(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='stale-attempt',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/stale-attempt',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={},
            now=utcnow(),
        )
        stale_attempt = ExecutionCommandRow(
            command_id='cmd-stale-attempt',
            execution_id=execution.execution_id,
            assignment_id=assignment.assignment_id,
            attempt_id='wrong-attempt',
            type='input.submit',
            status=CommandStatus.PENDING,
            payload_json={},
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(stale_attempt)
        claimed = await manager.claim_commands(assignment.manager_id, assignment.manager_session_id, now=utcnow())
        row = await session.get(ExecutionCommandRow, 'cmd-stale-attempt')
        await session.commit()

    assert claimed == []
    assert row is not None
    assert row.status == CommandStatus.CANCELLED


async def test_manager_claim_commands_prefilters_current_assignments_without_session_get(session_factory, monkeypatch):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='claim-batch-prefilter',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/claim-batch-prefilter',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={},
            now=utcnow(),
        )
        session.add(
            ExecutionCommandRow(
                command_id='cmd-current-prefilter',
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                type='input.submit',
                status=CommandStatus.PENDING,
                payload_json={},
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        await session.commit()

    get_calls = 0
    original_get = AsyncSession.get

    async def counting_get(self, *args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        return await original_get(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, 'get', counting_get)

    async with session_factory() as session:
        manager = ManagerRepository(session)
        claimed = await manager.claim_commands(assignment.manager_id, assignment.manager_session_id, now=utcnow())
        await session.commit()

    assert [command.command_id for command in claimed] == ['cmd-current-prefilter']
    assert get_calls == 0


async def test_manager_command_sort_uses_narrow_candidate_projection(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-command-projection',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='command-projection',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=utcnow(),
        )
        session.add(
            ExecutionCommandRow(
                command_id='cmd-narrow-projection',
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                type='input.submit',
                status=CommandStatus.PENDING,
                payload_json={'text': 'hello'},
                created_at=utcnow(),
                updated_at=utcnow(),
            )
        )
        await session.commit()

    statements: list[str] = []
    engine = session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with session_factory() as session:
            claimed = await ManagerRepository(session).claim_commands(
                assignment.manager_id,
                assignment.manager_session_id,
                now=utcnow(),
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert [command.command_id for command in claimed] == ['cmd-narrow-projection']
    ordered = next(
        statement.lower() for statement in statements if 'order by execution_commands.created_at' in statement.lower()
    )
    assert 'execution_commands.payload_json' not in ordered
    assert 'execution_commands.error_json' not in ordered
    assert 'assignments.runtime_info_json' not in ordered
    assert 'executions.spec_json' not in ordered
    assert 'executions.driver_config_json' not in ordered


async def test_gateway_idempotency_key_must_match_current_assignment_attempt(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
        capacity_total=2,
    )
    execution = await _create_execution(
        session_factory,
        execution_id='idempotency-current-attempt',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
        max_attempts=2,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        first = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert first is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(first.assignment_id, first.manager_id, first.manager_session_id, now=utcnow())
        await manager.mark_running(
            first.assignment_id,
            first.manager_id,
            first.manager_session_id,
            endpoint_url='http://im-live/v1/internal/executions/first',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            first.assignment_id, first.manager_id, first.manager_session_id, payload={}, now=utcnow()
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        first_command = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Alpha'},
            idempotency_key='turn-1',
            now=utcnow(),
        )
        first_row = await session.get(AssignmentRow, first.assignment_id)
        assert first_row is not None
        first_row.status = AssignmentStatus.LOST
        first_row.finished_at = utcnow()
        second = AssignmentRow(
            assignment_id='asn-current-idempotency',
            execution_id=execution.execution_id,
            attempt_id='idempotency-current-attempt-attempt-2',
            attempt_seq=2,
            manager_id=first.manager_id,
            manager_session_id=first.manager_session_id,
            status=AssignmentStatus.ACTIVE,
            slot_cost=1,
            endpoint_url='http://im-live/v1/internal/executions/second',
            endpoint_ready=True,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(second)
        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        execution_row.current_assignment_id = second.assignment_id
        execution_row.current_attempt_id = second.attempt_id
        execution_row.status = ExecutionStatus.WAITING_INPUT
        execution_row.updated_at = utcnow()

        with pytest.raises(ValueError, match='idempotency_key_conflict'):
            await gateway.submit_command(
                execution.execution_id,
                command_type='input.submit',
                payload={'text': 'Beta'},
                idempotency_key='turn-1',
                now=utcnow(),
            )
        duplicate = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Alpha duplicate'},
            idempotency_key='turn-2',
            now=utcnow(),
        )
        same_current = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Ignored'},
            idempotency_key='turn-2',
            now=utcnow(),
        )
        await session.commit()

    assert first_command.assignment_id == first.assignment_id
    assert duplicate.assignment_id == second.assignment_id
    assert duplicate.attempt_id == second.attempt_id
    assert same_current.command_id == duplicate.command_id
    assert same_current.payload == {'text': 'Alpha duplicate'}


async def test_activate_cached_live_route_is_idempotent_and_rejects_closed_route(session_factory):
    execution = await _create_execution(
        session_factory,
        execution_id='cached-live',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.CACHED,
    )

    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        activated = await repo.activate_route(execution.execution_id, now=utcnow())
        activated_again = await repo.activate_route(execution.execution_id, now=utcnow())
        await repo.request_stop(execution.execution_id, now=utcnow())
        with pytest.raises(ResourceGoneError) as exc_info:
            await repo.activate_route(execution.execution_id, now=utcnow())
        assert exc_info.value.code == 'execution_terminal'
        await session.commit()

    assert activated.route_status == RouteStatus.ACTIVE
    assert activated_again.route_status == RouteStatus.ACTIVE


async def test_gateway_route_requires_endpoint_ready(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='endpoint-live',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        manager = ManagerRepository(session)
        assert assignment is not None
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/endpoint-live',
            endpoint_ready=False,
            runtime_info={},
            now=utcnow(),
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        with pytest.raises(TemporaryUnavailableError, match='endpoint_not_ready'):
            await gateway.resolve_route(execution.execution_id, now=utcnow())
        await session.commit()


async def test_gateway_route_rejects_manager_session_mismatch(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='session-mismatch-route',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        manager = ManagerRepository(session)
        assert assignment is not None
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live.internal:9001/v1/internal/executions/session-mismatch-route',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, payload={}, now=utcnow()
        )
        manager_row = await session.get(AssignmentRow, assignment.assignment_id)
        assert manager_row is not None
        im = await session.get(
            __import__('nexus.distributed.persistence.orm', fromlist=['InstanceManagerRow']).InstanceManagerRow,
            'im-live',
        )
        assert im is not None
        im.session_id = 'replacement-session'
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        with pytest.raises(TemporaryUnavailableError, match='assignment_unavailable'):
            await gateway.resolve_route(execution.execution_id, now=utcnow())
        await session.commit()


async def test_scheduler_renews_only_owned_shard_leases(session_factory):
    owner_now = utcnow()
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=4, manager_healthy_ttl=timedelta(seconds=15))
        owned = await scheduler.renew_shard_leases(
            owner_id='scheduler-a',
            shard_group='default',
            shard_ids=[0, 1],
            lease_ttl=timedelta(seconds=20),
            now=owner_now,
        )
        await session.commit()

    assert owned == [0, 1]

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=4, manager_healthy_ttl=timedelta(seconds=15))
        stolen = await scheduler.renew_shard_leases(
            owner_id='scheduler-b',
            shard_group='default',
            shard_ids=[0, 1],
            lease_ttl=timedelta(seconds=20),
            now=owner_now + timedelta(seconds=1),
        )
        await session.commit()

    assert stolen == []

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=4, manager_healthy_ttl=timedelta(seconds=15))
        stolen_after_expiry = await scheduler.renew_shard_leases(
            owner_id='scheduler-b',
            shard_group='default',
            shard_ids=[0, 1],
            lease_ttl=timedelta(seconds=20),
            now=owner_now + timedelta(seconds=21),
        )
        await session.commit()

    assert stolen_after_expiry == [0, 1]


async def test_scheduler_lease_renew_acquire_uses_batch_statements():
    engine = create_async_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    statements = []

    def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        if 'shard_leases' in statement:
            statements.append(statement.lower())

    event.listen(engine.sync_engine, 'before_cursor_execute', before_cursor_execute)
    try:
        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            owned = await scheduler.renew_shard_leases(
                owner_id='scheduler-a',
                shard_group='default',
                shard_ids=list(range(8)),
                lease_ttl=timedelta(seconds=20),
                now=utcnow(),
            )
            await session.commit()

        assert owned == list(range(8))
        insert_count = sum(1 for statement in statements if statement.lstrip().startswith('insert'))
        update_count = sum(1 for statement in statements if statement.lstrip().startswith('update'))
        select_count = sum(1 for statement in statements if statement.lstrip().startswith('select'))
        assert insert_count == 1
        assert update_count == 2
        assert select_count == 3

        statements.clear()
        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            renewed = await scheduler.renew_shard_leases(
                owner_id='scheduler-a',
                shard_group='default',
                shard_ids=list(range(8)),
                lease_ttl=timedelta(seconds=20),
                now=utcnow(),
            )
            await session.commit()

        assert renewed == list(range(8))
        assert not any(statement.lstrip().startswith('insert') for statement in statements)
        assert sum(1 for statement in statements if statement.lstrip().startswith('update')) == 1
        assert sum(1 for statement in statements if statement.lstrip().startswith('select')) == 3

        async with session_factory() as session:
            rows = (await session.scalars(select(ShardLeaseRow).order_by(ShardLeaseRow.shard_id))).all()
        assert [row.shard_id for row in rows] == list(range(8))
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', before_cursor_execute)
        await engine.dispose()


async def test_shard_lease_initialization_creates_expired_full_topology(session_factory):
    now = utcnow()
    async with session_factory() as session:
        owned = await ShardLeaseRepository(session, shard_count=4).renew(
            owner_id='scheduler-a',
            shard_group='execution-scheduler',
            shard_ids=[3],
            lease_ttl=timedelta(seconds=20),
            now=now,
        )
        await session.commit()

    async with session_factory() as session:
        rows = list(
            await session.scalars(
                select(ShardLeaseRow)
                .where(ShardLeaseRow.shard_group == 'execution-scheduler')
                .order_by(ShardLeaseRow.shard_id)
            )
        )

    assert owned == [3]
    assert [row.shard_id for row in rows] == [0, 1, 2, 3]
    assert all(row.owner_id == 'topology-unowned' and row.lease_expires_at <= now for row in rows[:3])
    assert rows[3].owner_id == 'scheduler-a'
    assert rows[3].lease_expires_at > now


async def test_scheduler_skips_unhealthy_and_instance_mode_mismatch_with_backoff(session_factory):
    await _register_manager(session_factory, manager_id='stale-manager')
    await _register_manager(session_factory, manager_id='wrong-instance', instance_modes=['process'])
    execution = await _create_execution(session_factory, execution_id='needs-asyncio')
    now = utcnow()

    async with session_factory() as session:
        stale_manager = await session.get(InstanceManagerRow, 'stale-manager')
        stale_liveness = await session.get(InstanceManagerLivenessRow, 'stale-manager')
        assert stale_manager is not None
        assert stale_liveness is not None
        stale_manager.last_heartbeat_at = now - timedelta(seconds=120)
        stale_liveness.last_heartbeat_at = now - timedelta(seconds=120)
        scheduler = SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
            no_capacity_backoff=timedelta(seconds=7),
        )
        claimed = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=now)
        current = await session.get(ExecutionRow, execution.execution_id)
        await session.commit()

    assert claimed is None
    assert current is not None
    assert current.status == ExecutionStatus.QUEUED
    assert current.last_schedule_error == 'no_capacity'
    assert current.next_schedule_at == now + timedelta(seconds=7)


async def test_scheduler_accepts_legacy_manager_heartbeat_during_liveness_rollout(session_factory):
    await _register_manager(session_factory, manager_id='legacy-heartbeat-manager')
    execution = await _create_execution(session_factory, execution_id='legacy-heartbeat-execution')
    now = utcnow()

    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'legacy-heartbeat-manager')
        liveness = await session.get(InstanceManagerLivenessRow, 'legacy-heartbeat-manager')
        assert manager is not None
        assert liveness is not None
        manager.last_heartbeat_at = now
        liveness.last_heartbeat_at = now - timedelta(minutes=5)
        await session.commit()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        await session.commit()

    assert assignment is not None
    assert assignment.manager_id == 'legacy-heartbeat-manager'


async def test_scheduler_does_not_revive_fenced_manager_from_legacy_heartbeat(session_factory):
    await _register_manager(session_factory, manager_id='fenced-heartbeat-manager')
    execution = await _create_execution(session_factory, execution_id='fenced-heartbeat-execution')
    now = utcnow()

    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'fenced-heartbeat-manager')
        liveness = await session.get(InstanceManagerLivenessRow, 'fenced-heartbeat-manager')
        assert manager is not None
        assert liveness is not None
        manager.last_heartbeat_at = now
        liveness.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.fenced_at = now - timedelta(seconds=1)
        await session.commit()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        await session.commit()

    assert assignment is None


async def test_schema_creation_records_current_schema_version(session_factory):
    async with session_factory() as session:
        row = await session.get(SchemaVersionRow, 1)
    assert row is not None


async def test_schema_declares_phase_one_indexes(session_factory):
    bind = session_factory.kw['bind']
    async with bind.connect() as conn:
        indexes = await conn.run_sync(
            lambda sync_conn: {
                table: {index['name']: tuple(index['column_names']) for index in inspect(sync_conn).get_indexes(table)}
                for table in (
                    'executions',
                    'assignments',
                    'execution_commands',
                    'instance_managers',
                    'shard_leases',
                )
            }
        )

    assert 'idx_executions_sched' in indexes['executions']
    assert indexes['executions']['idx_executions_queue_deadline'] == (
        'status',
        'shard_id',
        'queue_expires_at',
        'execution_id',
    )
    assert indexes['executions']['idx_executions_cleanup'] == (
        'status',
        'shard_id',
        'cleanup_done_at',
        'execution_id',
    )
    assert 'ix_executions_current_assignment_id' not in indexes['executions']
    assert 'idx_assignments_manager_status' in indexes['assignments']
    assert 'idx_execution_commands_assignment_pending' in indexes['execution_commands']
    assert 'ix_instance_managers_last_heartbeat_at' not in indexes['instance_managers']
    assert 'idx_shard_leases_expires' in indexes['shard_leases']


async def test_reconciler_requeues_lost_execution_until_retry_exhausted(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution = await _create_execution(session_factory, execution_id='retry-me', max_attempts=2)

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        changed = await reconciler.reconcile_manager_dead(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        current = await repo.get_execution(execution.execution_id)

    assert current.status == ExecutionStatus.QUEUED
    assert current.current_assignment.status == AssignmentStatus.LOST

    await _register_manager(session_factory, manager_id='im-1', session_id='session-2')

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        second_assignment = await scheduler.claim_and_assign(shard_ids=[current.shard_id], now=utcnow())
        assert second_assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            second_assignment.assignment_id,
            second_assignment.manager_id,
            second_assignment.manager_session_id,
            now=utcnow(),
        )
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        changed = await reconciler.reconcile_manager_dead(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        exhausted = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)

    assert exhausted.status == ExecutionStatus.LOST
    assert exhausted.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_hot_path_queries_do_not_load_execution_payloads(session_factory):
    await _register_manager(session_factory, manager_id='reconciler-projection-manager')
    assigned = await _create_execution(session_factory, execution_id='reconciler-projection-assigned')
    for execution_id in (
        'reconciler-projection-queued',
        'reconciler-projection-running',
        'reconciler-projection-stopping',
    ):
        await _create_execution(session_factory, execution_id=execution_id)

    now = utcnow()
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[assigned.shard_id], now=now)
        assert assignment is not None

        queued = await session.get(ExecutionRow, 'reconciler-projection-queued')
        running = await session.get(ExecutionRow, 'reconciler-projection-running')
        stopping = await session.get(ExecutionRow, 'reconciler-projection-stopping')
        assert queued is not None
        assert running is not None
        assert stopping is not None
        queued.queue_ttl_seconds = 60
        queued.first_schedulable_at = now
        queued.queue_expires_at = None
        running.status = ExecutionStatus.RUNNING
        running.started_at = now
        running.execution_ttl_seconds = 60
        running.execution_expires_at = None
        stopping.status = ExecutionStatus.STOPPING
        stopping.stop_timeout_seconds = 60
        stopping.stop_deadline_at = None
        await session.commit()

    execution_selects: list[str] = []
    engine = session_factory.kw['bind']

    def record_execution_select(_conn, _cursor, statement, _parameters, _context, _executemany):
        normalized = ' '.join(statement.lower().split())
        if normalized.startswith('select') and ' from executions' in normalized:
            execution_selects.append(normalized)

    event.listen(engine.sync_engine, 'before_cursor_execute', record_execution_select)
    try:
        async with session_factory() as session:
            reconciler = ReconcilerRepository(session, batch_size=10)
            await reconciler._backfill_queue_deadlines()
            await reconciler._backfill_execution_deadlines()
            await reconciler._backfill_stop_deadlines(now=now)
            await reconciler._lock_execution_candidates(['reconciler-projection-queued'])
            await reconciler._lock_assignment_candidates([
                (assigned.execution_id, assignment.assignment_id),
            ])
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', record_execution_select)

    assert len(execution_selects) == 5
    payload_columns = {
        'driver_config_json',
        'error_json',
        'placement_json',
        'result_json',
        'spec_json',
        'trace_carrier_json',
    }
    for statement in execution_selects:
        projection = statement.split(' from executions', 1)[0]
        for column in payload_columns:
            assert column not in projection


async def test_reconciler_preserves_assignment_with_fresh_legacy_manager_heartbeat(session_factory):
    await _register_manager(session_factory, manager_id='legacy-reconciler-manager')
    execution = await _create_execution(session_factory, execution_id='legacy-reconciler-execution')
    assigned_at = utcnow()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=assigned_at)
        assert assignment is not None
        await session.commit()

    reconcile_at = assigned_at + timedelta(minutes=5)
    async with session_factory() as session:
        manager = await session.get(InstanceManagerRow, 'legacy-reconciler-manager')
        liveness = await session.get(InstanceManagerLivenessRow, 'legacy-reconciler-manager')
        assert manager is not None
        assert liveness is not None
        manager.last_heartbeat_at = reconcile_at
        liveness.last_heartbeat_at = assigned_at
        await session.commit()

    async with session_factory() as session:
        changed = await ReconcilerRepository(
            session,
            manager_dead_ttl=timedelta(seconds=30),
        ).reconcile_manager_dead(now=reconcile_at)
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.ASSIGNED
    assert current.current_assignment.status == AssignmentStatus.PENDING


async def test_reconciler_revalidates_manager_completion_after_candidate_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-completion-race')
    execution = await _create_execution(session_factory, execution_id='completion-race', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-completion-race.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert liveness is not None
        assert manager_row is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)

        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=30))
        lock_candidates = reconciler._lock_assignment_candidates

        async def complete_before_lock(candidates):
            await manager.complete_execution(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                result={'success': True},
                now=now,
            )
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_assignment_candidates', complete_before_lock)
        changed = await reconciler.reconcile_manager_dead(now=now)
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.SUCCEEDED
    assert current.current_assignment.status == AssignmentStatus.RELEASED


async def test_reconciler_revalidates_manager_heartbeat_after_candidate_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-heartbeat-race')
    execution = await _create_execution(session_factory, execution_id='heartbeat-race', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert liveness is not None
        assert manager_row is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)

        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=30))
        lock_candidates = reconciler._lock_assignment_candidates

        async def heartbeat_before_lock(candidates):
            await ManagerRepository(session).heartbeat(
                manager_id=assignment.manager_id,
                session_id=assignment.manager_session_id,
                capacity_total=1,
                observed_capacity_used=1,
                local_instances={},
                now=now,
            )
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_assignment_candidates', heartbeat_before_lock)
        changed = await reconciler.reconcile_manager_dead(now=now)
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.ASSIGNED
    assert current.current_assignment.status == AssignmentStatus.PENDING


async def test_reconciler_revalidates_manager_failure_after_candidate_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-failure-race')
    execution = await _create_execution(session_factory, execution_id='failure-race', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-failure-race.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert liveness is not None
        assert manager_row is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)

        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=30))
        lock_candidates = reconciler._lock_assignment_candidates

        async def fail_before_lock(candidates):
            await manager.fail_execution(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                error={'type': 'ExpectedFailure'},
                now=now,
            )
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_assignment_candidates', fail_before_lock)
        changed = await reconciler.reconcile_manager_dead(now=now)
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.FAILED
    assert current.current_assignment.status == AssignmentStatus.RELEASED


async def test_reconciler_revalidates_restored_manager_session_after_candidate_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-session-race')
    execution = await _create_execution(session_factory, execution_id='session-race', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert manager_row is not None
        manager_row.session_id = 'replacement-session'

        reconciler = ReconcilerRepository(session)
        lock_candidates = reconciler._lock_assignment_candidates

        async def restore_session_before_lock(candidates):
            manager_row.session_id = assignment.manager_session_id
            await session.flush()
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_assignment_candidates', restore_session_before_lock)
        changed = await reconciler.reconcile_session_mismatch(now=now)
        await session.commit()

    assert changed == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.ASSIGNED
    assert current.current_assignment.status == AssignmentStatus.PENDING


async def test_reconciler_repairs_nonactive_current_assignment_before_start_and_is_idempotent(session_factory):
    await _register_manager(session_factory, manager_id='im-repair-before-start')
    execution = await _create_execution(session_factory, execution_id='repair-before-start', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
        assert assignment_row is not None
        assignment_row.status = AssignmentStatus.REVOKED
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.repair_active_assignments(now=now)
        changed_again = await reconciler.repair_active_assignments(now=now)
        await session.commit()

    assert changed == 1
    assert changed_again == 0
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.QUEUED
    assert current.current_assignment.status == AssignmentStatus.REVOKED


async def test_reconciler_repairs_started_execution_with_nonactive_assignment_as_lost(session_factory):
    await _register_manager(session_factory, manager_id='im-repair-started')
    execution = await _create_execution(session_factory, execution_id='repair-started', max_attempts=2)
    now = utcnow()

    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-repair-started.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
        assert assignment_row is not None
        assignment_row.status = AssignmentStatus.LOST
        await session.commit()

    async with session_factory() as session:
        changed = await ReconcilerRepository(session).repair_active_assignments(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.LOST
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_cancels_unfinished_commands_when_assignment_is_lost(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='lost-cancels-commands',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
        max_attempts=2,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        session.add_all([
            ExecutionCommandRow(
                command_id='cmd-lost-pending',
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                type='input.submit',
                status=CommandStatus.PENDING,
                payload_json={},
                created_at=utcnow(),
                updated_at=utcnow(),
            ),
            ExecutionCommandRow(
                command_id='cmd-lost-claimed',
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                type='input.submit',
                status=CommandStatus.CLAIMED,
                payload_json={},
                created_at=utcnow(),
                updated_at=utcnow(),
            ),
            ExecutionCommandRow(
                command_id='cmd-lost-done',
                execution_id=execution.execution_id,
                assignment_id=assignment.assignment_id,
                attempt_id=assignment.attempt_id,
                type='input.submit',
                status=CommandStatus.DONE,
                payload_json={},
                created_at=utcnow(),
                updated_at=utcnow(),
            ),
        ])
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        changed = await reconciler.reconcile_manager_dead(now=utcnow())
        pending = await session.get(ExecutionCommandRow, 'cmd-lost-pending')
        claimed = await session.get(ExecutionCommandRow, 'cmd-lost-claimed')
        done = await session.get(ExecutionCommandRow, 'cmd-lost-done')
        await session.commit()

    assert changed == 1
    assert pending is not None
    assert pending.status == CommandStatus.CANCELLED
    assert claimed is not None
    assert claimed.status == CommandStatus.CANCELLED
    assert done is not None
    assert done.status == CommandStatus.DONE


async def test_reconciler_writes_control_plane_events_for_lost_retry(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution = await _create_execution(session_factory, execution_id='retry-events', max_attempts=2)
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        await reconciler.reconcile_manager_dead(now=utcnow())
        events = await ExecutionRepository(session, shard_count=8).list_control_events(execution.execution_id)
        await session.commit()

    event_types = [event.type for event in events]
    assert event_types[-3:] == [
        'assignment.status.changed',
        'execution.status.changed',
        'reconciler.retry',
    ]


async def test_reconciler_session_mismatch_loses_assignment_and_preserves_live_route_when_retrying(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='live-session-mismatch',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
        max_attempts=2,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        row = await session.get(
            __import__('nexus.distributed.persistence.orm', fromlist=['InstanceManagerRow']).InstanceManagerRow,
            'im-live',
        )
        assert row is not None
        row.session_id = 'new-session'
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.reconcile_session_mismatch(now=utcnow())
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.QUEUED
    assert current.route_status == RouteStatus.ACTIVE
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_running_live_session_loss_is_terminal(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live-running',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='live-running-session-loss',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
        max_attempts=2,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=utcnow(),
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live-running.internal/executions/live-running-session-loss',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        manager_row = await session.get(InstanceManagerRow, 'im-live-running')
        assert manager_row is not None
        manager_row.session_id = 'replacement-session'
        changed = await ReconcilerRepository(session).reconcile_session_mismatch(now=utcnow())
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.LOST
    assert current.route_status == RouteStatus.CLOSED
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_running_job_manager_loss_is_terminal(session_factory):
    await _register_manager(session_factory, manager_id='im-job-running')
    execution = await _create_execution(
        session_factory,
        execution_id='job-running-manager-loss',
        max_attempts=2,
    )
    now = utcnow()
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-job-running.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert liveness is not None
        assert manager_row is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)
        changed = await ReconcilerRepository(
            session,
            manager_dead_ttl=timedelta(seconds=30),
        ).reconcile_manager_dead(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.LOST
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_running_batch_manager_loss_requeues_durable_parent(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-batch-running',
        spec_types=['nexus_batch'],
        workload='orchestration',
    )
    execution = await _create_execution(
        session_factory,
        execution_id='batch-running-manager-loss',
        spec_type='nexus_batch',
        max_attempts=2,
    )
    now = utcnow()
    async with session_factory() as session:
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-batch-running.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        liveness = await session.get(InstanceManagerLivenessRow, assignment.manager_id)
        manager_row = await session.get(InstanceManagerRow, assignment.manager_id)
        assert liveness is not None
        assert manager_row is not None
        manager_row.last_heartbeat_at = now - timedelta(minutes=5)
        liveness.last_heartbeat_at = now - timedelta(minutes=5)
        changed = await ReconcilerRepository(
            session,
            manager_dead_ttl=timedelta(seconds=30),
        ).reconcile_manager_dead(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.QUEUED
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_pending_timeout_respects_attempt_budget(session_factory):
    await _register_manager(session_factory, manager_id='im-pending-timeout')
    execution = await _create_execution(
        session_factory,
        execution_id='pending-timeout-budget',
        max_attempts=2,
    )
    timeout_now = utcnow()
    for expected_status in (ExecutionStatus.QUEUED, ExecutionStatus.LOST):
        async with session_factory() as session:
            scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
            assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=timeout_now)
            assert assignment is not None
            assignment_row = await session.get(AssignmentRow, assignment.assignment_id)
            assert assignment_row is not None
            assignment_row.created_at = timeout_now - timedelta(seconds=60)
            changed = await ReconcilerRepository(
                session,
                assign_ack_timeout=timedelta(seconds=30),
            ).reconcile_pending_timeouts(now=timeout_now)
            await session.commit()

        assert changed == 1
        async with session_factory() as session:
            current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
        assert current.status == expected_status
        assert current.current_assignment.status == AssignmentStatus.REVOKED


async def test_reconciler_closes_live_route_when_retry_exhausted(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='live-retry-exhausted',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
        max_attempts=1,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        reconciler = ReconcilerRepository(session, manager_dead_ttl=timedelta(seconds=0))
        await reconciler.reconcile_manager_dead(now=utcnow())
        await session.commit()

    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.LOST
    assert current.route_status == RouteStatus.CLOSED


async def test_reconciler_pending_timeout_revokes_and_consumes_attempt(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution = await _create_execution(session_factory, execution_id='pending-timeout', max_attempts=2)

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        reconciler = ReconcilerRepository(session, assign_ack_timeout=timedelta(seconds=0))
        changed = await reconciler.reconcile_pending_timeouts(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)

    assert current.status == ExecutionStatus.QUEUED
    assert current.attempt_count == 1
    assert current.current_assignment.status == AssignmentStatus.REVOKED


async def test_reconciler_starting_timeout_marks_assignment_lost_and_retries(session_factory):
    await _register_manager(session_factory, manager_id='im-1')
    execution = await _create_execution(session_factory, execution_id='starting-timeout', max_attempts=2)

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=utcnow(),
        )
        reconciler = ReconcilerRepository(session, start_timeout=timedelta(seconds=0))
        changed = await reconciler.reconcile_starting_timeouts(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)

    assert current.status == ExecutionStatus.QUEUED
    assert current.attempt_count == 1
    assert current.current_assignment.status == AssignmentStatus.LOST


async def test_reconciler_expires_queued_execution_and_requests_ttl_stop(session_factory):
    now = utcnow()
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        queued = await repo.create_execution(
            CreateExecution(
                execution_id='queue-ttl',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
                queue_ttl_seconds=1,
            ),
            now=now - timedelta(seconds=5),
        )
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.expire_queued(now=now)
        queue_stats = reconciler.last_stats
        await session.commit()

    assert changed == 1
    assert queue_stats.scanned == 1
    assert queue_stats.changed == 1
    assert queue_stats.overdue_lag_seconds >= 4
    async with session_factory() as session:
        expired = await ExecutionRepository(session, shard_count=8).get_execution(queued.execution_id)
    assert expired.status == ExecutionStatus.EXPIRED

    await _register_manager(
        session_factory, manager_id='im-live', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    live = await _create_execution(
        session_factory,
        execution_id='execution-ttl',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        execution_row = await session.get(ExecutionRow, live.execution_id)
        assert execution_row is not None
        execution_row.execution_ttl_seconds = 1
        ttl_now = utcnow()
        running_at = ttl_now - timedelta(seconds=5)
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[live.shard_id], now=ttl_now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=ttl_now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live/v1/internal/executions/execution-ttl',
            endpoint_ready=True,
            runtime_info={},
            now=running_at,
        )
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.request_ttl_stops(now=ttl_now)
        execution_stats = reconciler.last_stats
        commands = await ExecutionRepository(session, shard_count=8).list_commands(live.execution_id)
        await session.commit()

    assert changed == 1
    assert execution_stats.scanned == 1
    assert execution_stats.changed == 1
    assert execution_stats.overdue_lag_seconds >= 4
    assert len(commands) == 1
    assert commands[0].type == 'stop.request'
    async with session_factory() as session:
        ttl_execution = await session.get(ExecutionRow, live.execution_id)
    assert ttl_execution.stop_reason == 'execution_ttl'


async def test_reconciler_revalidates_manager_success_after_ttl_scan(session_factory, monkeypatch):
    await _register_manager(session_factory, manager_id='im-ttl-race')
    execution = await _create_execution(session_factory, execution_id='ttl-success-race')
    now = utcnow()

    async with session_factory() as session:
        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        execution_row.execution_ttl_seconds = 1
        assignment = await SchedulerRepository(
            session,
            shard_count=8,
            manager_healthy_ttl=timedelta(seconds=15),
        ).claim_and_assign(shard_ids=[execution.shard_id], now=now)
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            now=now,
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-ttl-race.internal',
            endpoint_ready=True,
            runtime_info={},
            now=now,
        )
        execution_row = await session.get(ExecutionRow, execution.execution_id)
        assert execution_row is not None
        execution_row.started_at = now - timedelta(seconds=5)

        reconciler = ReconcilerRepository(session)
        lock_candidates = reconciler._lock_execution_candidates

        async def complete_before_lock(candidates):
            await manager.complete_execution(
                assignment.assignment_id,
                assignment.manager_id,
                assignment.manager_session_id,
                result={'success': True},
                now=now,
            )
            return await lock_candidates(candidates)

        monkeypatch.setattr(reconciler, '_lock_execution_candidates', complete_before_lock)
        changed = await reconciler.request_ttl_stops(now=now)
        commands = await ExecutionRepository(session, shard_count=8).list_commands(execution.execution_id)
        await session.commit()

    assert changed == 0
    assert commands == []
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.status == ExecutionStatus.SUCCEEDED


async def test_reconciler_cleanup_terminal_releases_active_assignment_and_closes_live_route(session_factory):
    await _register_manager(
        session_factory, manager_id='im-live', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    execution = await _create_execution(
        session_factory,
        execution_id='cleanup-live',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.FAILED
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.cleanup_terminal(now=utcnow())
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        current = await ExecutionRepository(session, shard_count=8).get_execution(execution.execution_id)
    assert current.route_status == RouteStatus.CLOSED
    assert current.current_assignment.status == AssignmentStatus.RELEASED

    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        assert row.cleanup_done_at is not None
        reconciler = ReconcilerRepository(session)
        changed_again = await reconciler.cleanup_terminal(now=utcnow())
        await session.commit()

    assert changed_again == 0


async def test_reconciler_cleanup_terminal_preserves_result_artifact_reference(session_factory):
    execution = await _create_execution(
        session_factory,
        execution_id='cleanup-result-reference',
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
    )
    now = utcnow()
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.environment_id = 'env-result-retention'
        row.status = ExecutionStatus.SUCCEEDED
        row.result_artifact_id = 'artifact-result-retention'
        row.finished_at = now
        session.add(
            ArtifactRow(
                artifact_id='artifact-result-retention',
                owner_principal_id='legacy',
                environment_id='env-result-retention',
                owner_tenant=None,
                kind='run_result',
                producer_key='run-result:cleanup-result-reference:attempt-1',
                uri='memory://artifact-result-retention',
                sha256='a' * 64,
                size_bytes=2,
                provider_etag=None,
                status='ready',
                created_by='test',
                expires_at=now - timedelta(seconds=1),
                created_at=now,
                updated_at=now,
            )
        )
        session.add(
            ArtifactRefRow(
                artifact_id='artifact-result-retention',
                resource_type='execution',
                resource_id=execution.execution_id,
                created_at=now,
            )
        )
        changed = await ReconcilerRepository(session).cleanup_terminal(now=now)
        await session.commit()

    assert changed == 1
    async with session_factory() as session:
        reference = await session.get(
            ArtifactRefRow,
            ('artifact-result-retention', 'execution', execution.execution_id),
        )
        assert reference is not None


async def test_reconciler_cleanup_terminal_removes_top_level_execution_credentials(session_factory):
    execution = await _create_execution(
        session_factory,
        execution_id='cleanup-execution-credential',
        mode=ExecutionMode.JOB,
        spec_type='nexus_single',
    )
    now = utcnow()
    async with session_factory() as session:
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.SUCCEEDED
        row.finished_at = now
        session.add(
            ExecutionCredentialRow(
                execution_id=execution.execution_id,
                credential_kind='wedata',
                payload_json={'subject': 'alice', 'key': 'secret', 'id': 'epoch-1'},
                version=1,
                expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )
        assert await ReconcilerRepository(session).cleanup_terminal(now=now) == 1
        await session.commit()

    async with session_factory() as session:
        credential = await session.get(ExecutionCredentialRow, (execution.execution_id, 'wedata'))
        assert credential is None


async def test_reconciler_cleanup_terminal_batches_current_assignment_lookup(session_factory, monkeypatch):
    await _register_manager(
        session_factory, manager_id='im-live', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    execution = await _create_execution(
        session_factory,
        execution_id='cleanup-live-batch',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        row = await session.get(ExecutionRow, execution.execution_id)
        assert row is not None
        row.status = ExecutionStatus.FAILED
        await session.commit()

    get_calls = 0
    original_get = AsyncSession.get

    async def counting_get(self, *args, **kwargs):
        nonlocal get_calls
        get_calls += 1
        return await original_get(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, 'get', counting_get)

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.cleanup_terminal(now=utcnow())
        await session.commit()

    assert changed == 1
    assert get_calls == 0


async def test_reconciler_cleanup_terminal_fast_path_uses_single_set_based_update(session_factory):
    execution_ids = [f'cleanup-fast-{index}' for index in range(5)]
    for execution_id in execution_ids:
        await _create_execution(session_factory, execution_id=execution_id)
    async with session_factory() as session:
        for execution_id in execution_ids:
            row = await session.get(ExecutionRow, execution_id)
            assert row is not None
            row.status = ExecutionStatus.SUCCEEDED
        await session.commit()

    # Fast-path rows (no assignment to release, no live route to close) must be stamped by a single
    # set-based ``UPDATE ... WHERE execution_id IN (...)`` rather than one single-row UPDATE per row,
    # which is what caused the slow-SQL warnings under bulk terminal cleanup.
    cleanup_updates: list[str] = []
    engine = session_factory.kw['bind']
    sync_engine = engine.sync_engine

    def _capture(conn, cursor, statement, parameters, context, executemany):  # noqa: ARG001
        normalized = ' '.join(statement.split())
        if 'UPDATE executions' in normalized and 'cleanup_done_at' in normalized:
            cleanup_updates.append(normalized)

    event.listen(sync_engine, 'before_cursor_execute', _capture)
    try:
        async with session_factory() as session:
            reconciler = ReconcilerRepository(session)
            changed = await reconciler.cleanup_terminal(now=utcnow())
            await session.commit()
    finally:
        event.remove(sync_engine, 'before_cursor_execute', _capture)

    assert changed == 5
    assert len(cleanup_updates) == 1
    assert 'execution_id IN (' in cleanup_updates[0]
    assert 'WHERE executions.execution_id = ' not in cleanup_updates[0]

    async with session_factory() as session:
        for execution_id in execution_ids:
            row = await session.get(ExecutionRow, execution_id)
            assert row is not None
            assert row.cleanup_done_at is not None

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        assert await reconciler.cleanup_terminal(now=utcnow()) == 0


async def test_reconciler_cleanup_terminal_mixes_batched_fast_path_with_per_row_slow_path(session_factory):
    await _register_manager(
        session_factory, manager_id='im-live', modes=[ExecutionMode.LIVE], spec_types=['nexus_live']
    )
    fast_ids = [f'cleanup-mixed-fast-{index}' for index in range(3)]
    for execution_id in fast_ids:
        await _create_execution(session_factory, execution_id=execution_id)
    live = await _create_execution(
        session_factory,
        execution_id='cleanup-mixed-live',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )
    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[live.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        for execution_id in fast_ids:
            row = await session.get(ExecutionRow, execution_id)
            assert row is not None
            row.status = ExecutionStatus.SUCCEEDED
        live_row = await session.get(ExecutionRow, live.execution_id)
        assert live_row is not None
        live_row.status = ExecutionStatus.FAILED
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session)
        changed = await reconciler.cleanup_terminal(now=utcnow())
        await session.commit()

    assert changed == 4
    async with session_factory() as session:
        for execution_id in fast_ids:
            row = await session.get(ExecutionRow, execution_id)
            assert row is not None
            assert row.cleanup_done_at is not None
        current = await ExecutionRepository(session, shard_count=8).get_execution(live.execution_id)
        assert current.cleanup_done_at is not None
        assert current.route_status == RouteStatus.CLOSED
        assert current.current_assignment.status == AssignmentStatus.RELEASED


async def test_reconciler_batch_size_limits_timeout_and_cleanup_scans(session_factory):
    await _register_manager(session_factory, manager_id='im-batch', capacity_total=2)
    first = await _create_execution(session_factory, execution_id='pending-batch-1', max_attempts=2)
    second = await _create_execution(session_factory, execution_id='pending-batch-2', max_attempts=2)

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        first_assignment = await scheduler.claim_and_assign(shard_ids=[first.shard_id], now=utcnow())
        second_assignment = await scheduler.claim_and_assign(shard_ids=[second.shard_id], now=utcnow())
        assert first_assignment is not None
        assert second_assignment is not None
        reconciler = ReconcilerRepository(session, assign_ack_timeout=timedelta(seconds=0), batch_size=1)
        changed = await reconciler.reconcile_pending_timeouts(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        first_row = await session.get(AssignmentRow, first_assignment.assignment_id)
        second_row = await session.get(AssignmentRow, second_assignment.assignment_id)

    assert {first_row.status, second_row.status} == {AssignmentStatus.REVOKED, AssignmentStatus.PENDING}

    now = utcnow()
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        for execution_id in ('terminal-batch-1', 'terminal-batch-2'):
            row = await repo.create_execution(
                CreateExecution(
                    execution_id=execution_id,
                    mode=ExecutionMode.JOB,
                    spec_type='nexus_single',
                    instance_mode='asyncio',
                    spec={},
                ),
                now=now,
            )
            execution_row = await session.get(ExecutionRow, row.execution_id)
            assert execution_row is not None
            execution_row.status = ExecutionStatus.FAILED
        await session.commit()

    async with session_factory() as session:
        reconciler = ReconcilerRepository(session, batch_size=1)
        changed = await reconciler.cleanup_terminal(now=utcnow())
        await session.commit()

    assert changed == 1

    async with session_factory() as session:
        cleaned = (
            await session.scalars(
                select(ExecutionRow).where(
                    ExecutionRow.execution_id.in_(['terminal-batch-1', 'terminal-batch-2']),
                    ExecutionRow.cleanup_done_at.is_not(None),
                )
            )
        ).all()

    assert len(cleaned) == 1


async def test_control_plane_events_are_persisted_with_monotonic_sequence(session_factory):
    async with session_factory() as session:
        repo = ExecutionRepository(session, shard_count=8)
        execution = await repo.create_execution(
            CreateExecution(
                execution_id='evented',
                mode=ExecutionMode.JOB,
                spec_type='nexus_single',
                instance_mode='asyncio',
                spec={},
            ),
            now=utcnow(),
        )
        await repo.append_control_event(
            execution.execution_id,
            event_type='custom.event',
            source='test',
            payload={'ok': True},
            event_key='custom-key',
            now=utcnow(),
        )
        duplicate = await repo.append_control_event(
            execution.execution_id,
            event_type='custom.event',
            source='test',
            payload={'ignored': True},
            event_key='custom-key',
            now=utcnow(),
        )
        events = await repo.list_control_events(execution.execution_id, after_seq=0)
        await session.commit()

    assert [event.seq for event in events] == [1, 2]
    assert events[0].type == 'execution.created'
    assert events[1].payload == {'ok': True}
    assert duplicate.seq == 2

    async with session_factory() as session:
        raw_count = len((await session.scalars(select(ExecutionEventRow))).all())
    assert raw_count == 2


async def test_control_plane_events_cover_assignment_command_and_manager_transitions(session_factory):
    await _register_manager(
        session_factory,
        manager_id='im-live',
        modes=[ExecutionMode.LIVE],
        spec_types=['nexus_live'],
    )
    execution = await _create_execution(
        session_factory,
        execution_id='control-plane-flow',
        mode=ExecutionMode.LIVE,
        spec_type='nexus_live',
        route_status=RouteStatus.ACTIVE,
    )

    async with session_factory() as session:
        scheduler = SchedulerRepository(session, shard_count=8, manager_healthy_ttl=timedelta(seconds=15))
        assignment = await scheduler.claim_and_assign(shard_ids=[execution.shard_id], now=utcnow())
        assert assignment is not None
        manager = ManagerRepository(session)
        await manager.ack_assignment(
            assignment.assignment_id, assignment.manager_id, assignment.manager_session_id, now=utcnow()
        )
        await manager.mark_running(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            endpoint_url='http://im-live/v1/internal/executions/control-plane-flow',
            endpoint_ready=True,
            runtime_info={},
            now=utcnow(),
        )
        await manager.mark_waiting_input(
            assignment.assignment_id,
            assignment.manager_id,
            assignment.manager_session_id,
            payload={},
            now=utcnow(),
        )
        gateway = GatewayRepository(session, manager_healthy_ttl=timedelta(seconds=15))
        command = await gateway.submit_command(
            execution.execution_id,
            command_type='input.submit',
            payload={'text': 'Alpha'},
            idempotency_key='turn-1',
            now=utcnow(),
        )
        claimed = await manager.claim_commands(assignment.manager_id, assignment.manager_session_id, now=utcnow())
        await manager.complete_command(claimed[0].command_id, now=utcnow())
        events = await ExecutionRepository(session, shard_count=8).list_control_events(execution.execution_id)
        await session.commit()

    assert command.command_id == claimed[0].command_id
    event_types = [event.type for event in events]
    assert event_types == [
        'execution.created',
        'assignment.status.changed',
        'scheduler.assigned',
        'assignment.status.changed',
        'execution.status.changed',
        'execution.status.changed',
        'execution.status.changed',
        'command.status.changed',
        'command.status.changed',
        'command.status.changed',
    ]
