from __future__ import annotations

from datetime import timedelta

import pytest

from tests.stress.distributed.run_reconciler_capacity import (
    CORE_OPERATIONS,
    SHARD_COUNT,
    TOPOLOGIES,
    _completed_processes,
    _heartbeat_config,
    allocation_fixture_counts,
    batch_fixture_counts,
    capacity_fixture_prefix,
    core_fixture_counts,
    core_heartbeat_target_count,
    heartbeat_fixture_target_count,
    scenario_loop_names,
    validate_heartbeat_coverage,
    validate_recovery_config,
    validate_timing_config,
    validate_topology,
)


def test_reconciler_capacity_topologies_cover_every_shard_once() -> None:
    assert TOPOLOGIES == ((1, 64), (4, 16), (16, 4), (32, 2))
    assert all(replicas * ownership_limit == SHARD_COUNT for replicas, ownership_limit in TOPOLOGIES)


def test_reconciler_capacity_mixed_scenario_runs_real_controller_paths() -> None:
    assert scenario_loop_names('core') == ('core',)
    assert scenario_loop_names('allocation') == ('environment',)
    assert scenario_loop_names('batch') == ('batch',)
    assert scenario_loop_names('mixed') == ('core', 'environment', 'batch')


def test_reconciler_capacity_counts_ordinary_and_batch_allocations() -> None:
    counts = allocation_fixture_counts(
        ordinary_count=10_000,
        batch_count=200,
        batch_execution_managers=50,
    )

    assert counts == {
        'top_level_executions': 10_200,
        'allocations': 10_400,
        'managers': 20_200,
        'ordinary_allocations': 10_000,
        'batch_orchestration_allocations': 200,
        'batch_execution_allocations': 200,
    }


def test_reconciler_capacity_core_fixture_covers_every_operation_evenly() -> None:
    counts = core_fixture_counts(items_per_operation=10_000)

    assert tuple(counts) == CORE_OPERATIONS
    assert set(counts.values()) == {10_000}
    assert sum(counts.values()) == 90_000


def test_reconciler_capacity_heartbeats_every_non_stale_core_manager() -> None:
    assert core_heartbeat_target_count() == 2 * SHARD_COUNT


def test_reconciler_capacity_mixed_heartbeat_targets_are_additive() -> None:
    assert (
        heartbeat_fixture_target_count(
            scenario='mixed',
            ordinary_count=40,
            allocation_batch_count=8,
            batch_execution_managers=4,
            heartbeat_manager_limit=200,
            batch_parent_count=8,
        )
        == 128 + 80 + 8
    )


def test_reconciler_capacity_heartbeat_limit_is_explicitly_partial() -> None:
    eligible = heartbeat_fixture_target_count(
        scenario='allocation',
        ordinary_count=100,
        allocation_batch_count=10,
        batch_execution_managers=4,
        heartbeat_manager_limit=0,
        batch_parent_count=0,
    )

    assert eligible == 150
    with pytest.raises(ValueError, match='partial Manager heartbeat coverage'):
        validate_heartbeat_coverage(eligible_managers=eligible, target_count=100, allow_partial=False)
    validate_heartbeat_coverage(eligible_managers=eligible, target_count=100, allow_partial=True)


def test_reconciler_capacity_fixture_identity_does_not_change_with_topology() -> None:
    assert capacity_fixture_prefix('core') == 'reconciler-capacity-core'
    assert capacity_fixture_prefix('allocation') == 'reconciler-capacity-allocation'
    assert capacity_fixture_prefix('batch') == 'reconciler-capacity-batch'
    assert capacity_fixture_prefix('mixed') == 'reconciler-capacity-mixed'


def test_reconciler_capacity_batch_fixture_includes_one_healthy_manager_per_parent() -> None:
    assert batch_fixture_counts(parent_count=1_000, children_per_parent=100) == {
        'parents': 1_000,
        'children': 100_000,
        'managers': 1_000,
    }


def test_reconciler_capacity_rejects_partial_or_overlapping_shard_topology() -> None:
    validate_topology(replicas=16, ownership_limit=4)

    with pytest.raises(ValueError, match='exactly cover'):
        validate_topology(replicas=16, ownership_limit=3)


def test_reconciler_capacity_validates_worker_restart_target() -> None:
    validate_recovery_config(replicas=16, worker_index=15, restart_after_seconds=1)
    validate_recovery_config(replicas=16, worker_index=0, restart_after_seconds=0)

    with pytest.raises(ValueError, match='worker index'):
        validate_recovery_config(replicas=16, worker_index=16, restart_after_seconds=1)
    with pytest.raises(ValueError, match='non-negative'):
        validate_recovery_config(replicas=16, worker_index=0, restart_after_seconds=-1)


def test_reconciler_capacity_rejects_any_process_that_exits_during_measurement() -> None:
    class Process:
        def __init__(self, name: str, pid: int, exitcode: int | None) -> None:
            self.name = name
            self.pid = pid
            self.exitcode = exitcode

    assert _completed_processes(
        [
            Process('running', 1, None),
            Process('clean-early-exit', 2, 0),
            Process('failed', 3, 7),
        ]
    ) == [('clean-early-exit', 2, 0), ('failed', 3, 7)]


def test_reconciler_capacity_validates_heartbeat_and_lease_timing() -> None:
    validate_timing_config(
        manager_heartbeat_interval=5,
        manager_healthy_ttl=15,
        manager_self_fence_timeout=30,
        shard_lease_ttl=20,
        shard_lease_renew_interval=5,
        environment_operation_ttl=120,
        environment_lease_renew_interval=10,
        batch_claim_ttl=600,
        batch_claim_renew_interval=30,
    )

    invalid = (
        ({'manager_heartbeat_interval': 15}, '(?i)heartbeat interval'),
        ({'manager_healthy_ttl': 30}, '(?i)healthy TTL'),
        ({'shard_lease_renew_interval': 20}, '(?i)shard lease renew interval'),
        ({'environment_lease_renew_interval': 120}, '(?i)environment lease renew interval'),
        ({'batch_claim_renew_interval': 600}, '(?i)batch claim renew interval'),
    )
    defaults = {
        'manager_heartbeat_interval': 5,
        'manager_healthy_ttl': 15,
        'manager_self_fence_timeout': 30,
        'shard_lease_ttl': 20,
        'shard_lease_renew_interval': 5,
        'environment_operation_ttl': 120,
        'environment_lease_renew_interval': 10,
        'batch_claim_ttl': 600,
        'batch_claim_renew_interval': 30,
    }
    for override, message in invalid:
        with pytest.raises(ValueError, match=message):
            validate_timing_config(**(defaults | override))


def test_reconciler_capacity_heartbeat_config_keeps_dead_ttl_above_healthy_ttl() -> None:
    config = _heartbeat_config(
        database_url='mysql+asyncmy://benchmark.invalid/capacity',
        pool_size=4,
        max_overflow=4,
        healthy_ttl_seconds=15,
        self_fence_timeout_seconds=30,
    )

    assert config.manager_healthy_ttl == timedelta(seconds=15)
    assert config.manager_dead_ttl > config.manager_healthy_ttl
    assert config.manager_self_fence_timeout > config.manager_healthy_ttl
    assert config.manager_dead_ttl > config.manager_self_fence_timeout


def test_reconciler_capacity_heartbeat_config_expands_dead_ttl_for_long_self_fence() -> None:
    config = _heartbeat_config(
        database_url='mysql+asyncmy://benchmark.invalid/capacity',
        pool_size=4,
        max_overflow=4,
        healthy_ttl_seconds=300,
        self_fence_timeout_seconds=600,
    )

    assert config.manager_dead_ttl > timedelta(seconds=600)
