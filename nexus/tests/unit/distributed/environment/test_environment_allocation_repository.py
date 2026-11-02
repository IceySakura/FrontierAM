from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from sqlalchemy import delete, event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.errors import DomainConflictError
from nexus.distributed.persistence.orm import (
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    EnvironmentRow,
    create_schema,
)
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.allocation import (
    AllocationRepository,
    AllocationResources,
    CreateEnvironmentAllocation,
)
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.quota import QuotaExceededError, QuotaRepository
from nexus.distributed.utils.clock import utcnow


@pytest.fixture
async def allocation_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "allocation.db"}')
    await create_schema(engine)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _seed_environment(
    session,
    *,
    principal_id: str = 'principal-1',
    environment_limits: dict | None = None,
    grant_limits: dict | None = None,
):
    now = utcnow()
    await AccessRepository(session).upsert_user(
        principal_id=principal_id,
        issuer='tai',
        tenant='tenant',
        subject=principal_id,
        roles={'user'},
        display_name=None,
        actor='admin',
        now=now,
    )
    environments = EnvironmentRepository(session)
    try:
        await environments.create_environment(
            environment_id='env-kind',
            name='kind',
            allocation_type='dynamic',
            provider_type='kubernetes',
            provider_config_ref='k8s/default',
            resource_limits=environment_limits or {},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
    except ValueError as exc:
        if str(exc) != 'environment_already_exists':
            raise
    await environments.upsert_grant(
        principal_id=principal_id,
        environment_id='env-kind',
        limits=grant_limits or {},
        enabled=True,
        actor='admin',
        now=now,
    )


async def test_environment_and_grant_precreate_zero_resource_usage(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)

        environment_usage = await session.get(EnvironmentResourceUsageRow, 'environment:env-kind')
        grant_usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-kind:principal-1')

        assert environment_usage is not None
        assert grant_usage is not None
        for usage in (environment_usage, grant_usage):
            assert usage.manager_count == 0
            assert usage.cpu_millicores == 0
            assert usage.memory_bytes == 0
            assert usage.slots == 0
            assert usage.active_batches == 0


async def test_upsert_grant_reuses_orphaned_zero_resource_usage(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        await session.execute(
            delete(EnvironmentGrantRow).where(
                EnvironmentGrantRow.principal_id == 'principal-1',
                EnvironmentGrantRow.environment_id == 'env-kind',
            )
        )
        await session.flush()

        grant = await EnvironmentRepository(session).upsert_grant(
            principal_id='principal-1',
            environment_id='env-kind',
            limits={'slots': 2},
            enabled=True,
            actor='admin',
            now=utcnow(),
        )

        assert grant.version == 1
        usage = await session.get(EnvironmentResourceUsageRow, 'grant:env-kind:principal-1')
        assert usage is not None
        assert usage.slots == 0


def _allocation_request(
    allocation_id: str,
    *,
    principal_id: str = 'principal-1',
    managers: int = 2,
    allocation_spec: dict | None = None,
) -> CreateEnvironmentAllocation:
    return CreateEnvironmentAllocation(
        allocation_id=allocation_id,
        top_level_execution_id=f'execution-{allocation_id}',
        owner_principal_id=principal_id,
        environment_id='env-kind',
        pool_id=f'pool-{allocation_id}',
        workload='execution',
        desired_manager_count=managers,
        resources=AllocationResources(
            manager_count=managers,
            cpu_millicores=managers * 2_000,
            memory_bytes=managers * 4_096,
            slots=managers * 2,
            active_batches=0,
        ),
        allocation_spec=allocation_spec or {'image': 'nexus:test', 'manager_count': managers},
        allocation_spec_hash=(allocation_id[-1] * 64),
    )


async def test_allocation_repository_extracts_immutable_zhiyan_log_binding(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request(
                'allocation-zhiyan-binding',
                allocation_spec={
                    'allocation_type': 'dynamic',
                    'provider_type': 'tkex',
                    'provider_config': {'zhiyan_log': {'log_name': 'manager-log-flow'}},
                    'credential_ref': 'must-not-be-returned',
                },
            ),
            now=utcnow(),
        )

        binding = await AllocationRepository(session).zhiyan_log_binding(allocation.allocation_id)

    assert binding is not None
    assert binding.log_name == 'manager-log-flow'
    assert binding.provider_type == 'tkex'
    assert not hasattr(binding, 'credential_ref')


async def test_allocation_repository_returns_no_zhiyan_binding_for_legacy_snapshot(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-legacy-binding'),
            now=utcnow(),
        )
        assert await AllocationRepository(session).zhiyan_log_binding(allocation.allocation_id) is None


async def test_allocation_repository_binds_predefined_zhiyan_log_from_deployment_default(
    allocation_session_factory,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request(
                'allocation-predefined-binding',
                allocation_spec={
                    'allocation_type': 'predefined',
                    'provider_type': 'predefined',
                    'provider_config_ref': None,
                },
            ),
            now=utcnow(),
        )

        # Without the deployment-wide dataflow, predefined allocations have no binding.
        assert await AllocationRepository(session).zhiyan_log_binding(allocation.allocation_id) is None

        binding = await AllocationRepository(session).zhiyan_log_binding(
            allocation.allocation_id,
            predefined_log_name=' dev_nexus_manager_log ',
        )

    assert binding is not None
    assert binding.log_name == 'dev_nexus_manager_log'
    assert binding.provider_type == 'predefined'


async def test_allocation_repository_ignores_predefined_log_name_for_tkex_allocation(
    allocation_session_factory,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request(
                'allocation-tkex-over-default',
                allocation_spec={
                    'allocation_type': 'dynamic',
                    'provider_type': 'tkex',
                    'provider_config': {'zhiyan_log': {'log_name': 'manager-log-flow'}},
                },
            ),
            now=utcnow(),
        )

        binding = await AllocationRepository(session).zhiyan_log_binding(
            allocation.allocation_id,
            predefined_log_name='should-be-ignored',
        )

    assert binding is not None
    assert binding.log_name == 'manager-log-flow'
    assert binding.provider_type == 'tkex'


async def test_allocation_repository_rejects_pool_identity_longer_than_database_contract(
    allocation_session_factory,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)

        with pytest.raises(ValueError, match='invalid_pool_id'):
            await AllocationRepository(session).create_allocation(
                replace(_allocation_request('allocation-long-pool'), pool_id='p' * 65),
                now=utcnow(),
            )


async def test_allocation_transition_rejects_a_stale_operation_claim(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        allocation = await repository.create_allocation(_allocation_request('allocation-a'), now=utcnow())
        assert allocation.status == 'requested'
        assert allocation.config_version == 1

        first_claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller-a',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert await repository.transition(
            first_claim,
            status='provisioning',
            provider_resource_ref={'kind': 'StatefulSet', 'name': 'allocation-a'},
            now=utcnow(),
        )
        assert not await repository.transition(first_claim, status='ready', ready_manager_count=2, now=utcnow())

        second_claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller-b',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert second_claim.operation_id != first_claim.operation_id
        assert await repository.transition(second_claim, status='ready', ready_manager_count=2, now=utcnow())
        ready = await repository.get_allocation(allocation.allocation_id)
        assert ready.status == 'ready'
        assert ready.ready_manager_count == 2
        assert ready.provider_resource_ref == {'kind': 'StatefulSet', 'name': 'allocation-a'}


async def test_allocation_transition_rejects_an_expired_operation_lease(allocation_session_factory):
    now = utcnow()
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        allocation = await repository.create_allocation(_allocation_request('allocation-e'), now=now)
        claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller-a',
            operation_ttl=timedelta(seconds=30),
            now=now,
        )

        assert not await repository.transition(
            claim,
            status='provisioning',
            now=now + timedelta(seconds=31),
        )
        replacement = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller-b',
            operation_ttl=timedelta(seconds=30),
            now=now + timedelta(seconds=31),
        )
        assert replacement.operation_id != claim.operation_id


async def test_allocation_claim_renewal_extends_only_the_current_fenced_operation(allocation_session_factory):
    now = utcnow()
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        allocation = await repository.create_allocation(_allocation_request('allocation-r'), now=now)
        claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller-a',
            operation_ttl=timedelta(seconds=30),
            now=now,
        )

        assert await repository.renew_claim(
            claim,
            operation_ttl=timedelta(seconds=30),
            now=now + timedelta(seconds=20),
        )
        assert not await repository.renew_claim(
            replace(claim, operation_id='stale-operation'),
            operation_ttl=timedelta(seconds=30),
            now=now + timedelta(seconds=21),
        )
        with pytest.raises(ValueError, match='allocation_operation_leased'):
            await repository.claim_allocation(
                allocation.allocation_id,
                operation_owner='controller-b',
                operation_ttl=timedelta(seconds=30),
                now=now + timedelta(seconds=35),
            )
        assert await repository.transition(
            claim,
            status='provisioning',
            provider_resource_ref={'kind': 'StatefulSet', 'name': allocation.allocation_id},
            now=now + timedelta(seconds=35),
        )


async def test_allocation_transitions_use_immutable_allocation_type_snapshot(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        request = _allocation_request('allocation-snapshot')
        request = replace(
            request,
            allocation_spec={**request.allocation_spec, 'allocation_type': 'dynamic'},
        )
        allocation = await repository.create_allocation(request, now=utcnow())
        claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert await repository.transition(claim, status='provisioning', now=utcnow())

        environment = await session.get(EnvironmentRow, 'env-kind')
        assert environment is not None
        environment.allocation_type = 'predefined'
        await session.flush()

        claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        with pytest.raises(ValueError, match='provider_resource_ref_required'):
            await repository.transition(claim, status='ready', ready_manager_count=2, now=utcnow())
        assert await repository.transition(
            claim,
            status='ready',
            ready_manager_count=2,
            provider_resource_ref={'kind': 'StatefulSet', 'name': allocation.allocation_id},
            now=utcnow(),
        )

        for status in ('draining', 'deleting', 'released'):
            claim = await repository.claim_allocation(
                allocation.allocation_id,
                operation_owner='controller',
                operation_ttl=timedelta(seconds=30),
                now=utcnow(),
            )
            assert await repository.transition(claim, status=status, now=utcnow())

        released = await repository.get_allocation(allocation.allocation_id)
        assert released.status == 'released'


async def test_legacy_allocation_transition_rejects_changed_environment_without_type_snapshot(
    allocation_session_factory,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        allocation = await repository.create_allocation(_allocation_request('allocation-legacy'), now=utcnow())
        environment = await session.get(EnvironmentRow, 'env-kind')
        assert environment is not None
        environment.allocation_type = 'predefined'
        environment.config_version += 1
        await session.flush()
        claim = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )

        with pytest.raises(ValueError, match='allocation_type_snapshot_missing'):
            await repository.transition(claim, status='draining', now=utcnow())


async def test_admin_force_cleanup_revokes_lease_and_lists_draining_allocation(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        repository = AllocationRepository(session)
        allocation = await repository.create_allocation(_allocation_request('allocation-force'), now=utcnow())
        await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='stale-controller',
            operation_ttl=timedelta(hours=1),
            now=utcnow(),
        )

        cleaned = await repository.force_cleanup(allocation.allocation_id, now=utcnow())

        assert cleaned.status == 'draining'
        assert cleaned.ready_manager_count == 0
        assert cleaned.last_error_code == 'admin_force_cleanup'
        replacement = await repository.claim_allocation(
            allocation.allocation_id,
            operation_owner='environment-controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert replacement.status == 'draining'
        listed = await repository.list_allocations(
            owner_principal_id='principal-1',
            environment_id='env-kind',
            status='draining',
        )
        assert [item.allocation_id for item in listed] == ['allocation-force']


async def test_compute_usage_is_reserved_at_environment_and_grant_scopes(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(
            session,
            environment_limits={
                'manager_count': 5,
                'cpu_millicores': 20_000,
                'memory_bytes': 100_000,
                'slots': 20,
                'active_batches': 2,
            },
            grant_limits={
                'manager_count': 3,
                'cpu_millicores': 10_000,
                'memory_bytes': 50_000,
                'slots': 10,
                'active_batches': 1,
            },
        )
        allocations = AllocationRepository(session)
        first = await allocations.create_allocation(_allocation_request('allocation-1'), now=utcnow())
        quota = QuotaRepository(session, default_limits={})
        environment_usage, grant_usage = await quota.reserve_allocation(first.allocation_id, now=utcnow())
        assert environment_usage.manager_count == 2
        assert grant_usage.manager_count == 2
        assert grant_usage.slots == 4

        second = await allocations.create_allocation(_allocation_request('allocation-2'), now=utcnow())
        with pytest.raises(QuotaExceededError, match='grant_manager_count_limit_exceeded'):
            await quota.reserve_allocation(second.allocation_id, now=utcnow())

        persisted_environment = await session.get(EnvironmentResourceUsageRow, 'environment:env-kind')
        persisted_grant = await session.get(EnvironmentResourceUsageRow, 'grant:env-kind:principal-1')
        assert persisted_environment.manager_count == 2
        assert persisted_grant.manager_count == 2

        claim = await allocations.claim_allocation(
            first.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert await allocations.transition(claim, status='draining', now=utcnow())
        claim = await allocations.claim_allocation(
            first.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert await allocations.transition(claim, status='deleting', now=utcnow())
        claim = await allocations.claim_allocation(
            first.allocation_id,
            operation_owner='controller',
            operation_ttl=timedelta(seconds=30),
            now=utcnow(),
        )
        assert await allocations.transition(claim, status='released', now=utcnow())

        environment_usage, grant_usage = await quota.release_allocation(first.allocation_id, now=utcnow())
        assert environment_usage.manager_count == 0
        assert grant_usage.manager_count == 0
        assert await quota.release_allocation(first.allocation_id, now=utcnow()) is None
        assert (await quota.get_compute_usage(environment_id='env-kind')).manager_count == 0
        assert (await quota.get_compute_usage(environment_id='env-kind', principal_id='principal-1')).manager_count == 0


async def test_allocation_claim_sort_uses_narrow_candidate_projection(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-narrow-claim'),
            now=utcnow(),
        )
        await session.commit()

    statements: list[str] = []
    engine = allocation_session_factory.kw['bind']

    def capture_sql(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
    try:
        async with allocation_session_factory() as session:
            claims = await AllocationRepository(session).claim_allocations(
                operation_owner='narrow-claim-controller',
                operation_ttl=timedelta(minutes=2),
                statuses=None,
                limit=1,
                now=utcnow(),
            )
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

    assert len(claims) == 1
    assert claims[0].allocation_id == allocation.allocation_id
    ordered = next(
        statement.lower()
        for statement in statements
        if 'order by distributed_environment_allocations.updated_at' in statement.lower()
    )
    assert 'distributed_environment_allocations.allocation_spec_json' not in ordered
    assert 'distributed_environment_allocations.provider_resource_ref_json' not in ordered
    assert 'distributed_environment_allocations.last_error_message' not in ordered


async def test_repeated_allocation_reservation_does_not_lock_shared_usage(
    allocation_session_factory,
    monkeypatch,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-idempotent-reserve'),
            now=utcnow(),
        )
        quota = QuotaRepository(session, default_limits={})
        await quota.reserve_allocation(allocation.allocation_id, now=utcnow())

        async def fail_if_shared_usage_is_locked(**_kwargs):
            raise AssertionError('idempotent reservation must not lock shared usage')

        monkeypatch.setattr(quota, '_lock_compute_usage_pair', fail_if_shared_usage_is_locked)

        assert await quota.reserve_allocation(allocation.allocation_id, now=utcnow()) is None


async def test_allocation_reservation_locks_precreated_usage_without_upsert(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-precreated-usage'),
            now=utcnow(),
        )
        statements: list[str] = []

        def capture_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement)

        engine = session.bind
        assert engine is not None
        event.listen(engine.sync_engine, 'before_cursor_execute', capture_sql)
        try:
            await QuotaRepository(session, default_limits={}).reserve_allocation(
                allocation.allocation_id,
                now=utcnow(),
            )
        finally:
            event.remove(engine.sync_engine, 'before_cursor_execute', capture_sql)

        usage_inserts = [
            statement
            for statement in statements
            if statement.lstrip().upper().startswith('INSERT INTO DISTRIBUTED_ENVIRONMENT_RESOURCE_USAGE')
        ]
        assert usage_inserts == []
        usage_selects = [
            statement
            for statement in statements
            if statement.lstrip().upper().startswith('SELECT')
            and 'FROM distributed_environment_resource_usage' in statement
        ]
        assert len(usage_selects) == 1


async def test_environment_limit_update_rejects_current_usage(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(
            session,
            environment_limits={'manager_count': 5},
            grant_limits={'manager_count': 5},
        )
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-environment-limit'),
            now=utcnow(),
        )
        await QuotaRepository(session, default_limits={}).reserve_allocation(
            allocation.allocation_id,
            now=utcnow(),
        )

        with pytest.raises(DomainConflictError, match='environment_manager_count_limit_below_usage'):
            await EnvironmentRepository(session).update_environment(
                'env-kind',
                resource_limits={'manager_count': 1},
                actor='admin',
                now=utcnow(),
            )


async def test_grant_limit_update_rejects_current_usage(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(
            session,
            environment_limits={'manager_count': 5},
            grant_limits={'manager_count': 5},
        )
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-grant-limit'),
            now=utcnow(),
        )
        await QuotaRepository(session, default_limits={}).reserve_allocation(
            allocation.allocation_id,
            now=utcnow(),
        )

        with pytest.raises(DomainConflictError, match='grant_manager_count_limit_below_usage'):
            await EnvironmentRepository(session).upsert_grant(
                principal_id='principal-1',
                environment_id='env-kind',
                limits={'manager_count': 1},
                enabled=True,
                actor='admin',
                now=utcnow(),
            )


async def test_environment_and_grant_can_be_disabled_with_current_usage(allocation_session_factory):
    async with allocation_session_factory() as session:
        await _seed_environment(
            session,
            environment_limits={'manager_count': 5},
            grant_limits={'manager_count': 5},
        )
        allocation = await AllocationRepository(session).create_allocation(
            _allocation_request('allocation-disable-access'),
            now=utcnow(),
        )
        await QuotaRepository(session, default_limits={}).reserve_allocation(
            allocation.allocation_id,
            now=utcnow(),
        )

        environment = await EnvironmentRepository(session).update_environment(
            'env-kind',
            status='disabled',
            actor='admin',
            now=utcnow(),
        )
        grant = await EnvironmentRepository(session).upsert_grant(
            principal_id='principal-1',
            environment_id='env-kind',
            limits={'manager_count': 5},
            enabled=False,
            actor='admin',
            now=utcnow(),
        )

        assert environment.status == 'disabled'
        assert grant.enabled is False
        assert (await session.get(EnvironmentResourceUsageRow, 'environment:env-kind')).manager_count == 2
        assert (await session.get(EnvironmentResourceUsageRow, 'grant:env-kind:principal-1')).manager_count == 2


async def test_repeated_allocation_release_does_not_lock_shared_usage(
    allocation_session_factory,
    monkeypatch,
):
    async with allocation_session_factory() as session:
        await _seed_environment(session)
        allocations = AllocationRepository(session)
        allocation = await allocations.create_allocation(
            _allocation_request('allocation-idempotent-release'),
            now=utcnow(),
        )
        quota = QuotaRepository(session, default_limits={})
        await quota.reserve_allocation(allocation.allocation_id, now=utcnow())
        for status in ('draining', 'deleting', 'released'):
            claim = await allocations.claim_allocation(
                allocation.allocation_id,
                operation_owner='controller',
                operation_ttl=timedelta(seconds=30),
                now=utcnow(),
            )
            assert await allocations.transition(claim, status=status, now=utcnow())
        await quota.release_allocation(allocation.allocation_id, now=utcnow())

        async def fail_if_shared_usage_is_locked(**_kwargs):
            raise AssertionError('idempotent release must not lock shared usage')

        monkeypatch.setattr(quota, '_lock_compute_usage_pair', fail_if_shared_usage_is_locked)

        assert await quota.release_allocation(allocation.allocation_id, now=utcnow()) is None
