from __future__ import annotations

from datetime import datetime

import pytest

from nexus.distributed.execution.phase import (
    AllocationPhaseState,
    BatchPhaseState,
    ExecutionPhaseSnapshot,
    PhaseProjectionError,
    project_execution_phase,
)


def _snapshot(**updates) -> ExecutionPhaseSnapshot:
    values = {
        'status': 'queued',
        'spec_type': 'nexus_single',
        'allocations': (AllocationPhaseState(status='ready'),),
    }
    values.update(updates)
    return ExecutionPhaseSnapshot(**values)


@pytest.mark.parametrize(
    ('snapshot', 'phase'),
    [
        (
            _snapshot(allocations=(AllocationPhaseState(status='requested'),)),
            'waiting_resources',
        ),
        (
            _snapshot(allocations=(AllocationPhaseState(status='requested', dependency_build_status='requested'),)),
            'building_dependencies',
        ),
        (
            _snapshot(allocations=(AllocationPhaseState(status='requested', dependency_build_status='failed'),)),
            'dependency_build_failed',
        ),
        (
            _snapshot(
                allocations=(
                    AllocationPhaseState(
                        status='requested',
                        dependency_build_status='failed',
                        dependency_build_next_retry_at=datetime(2026, 8, 9),
                    ),
                )
            ),
            'building_dependencies',
        ),
        (
            _snapshot(
                spec_type='nexus_batch',
                allocations=(
                    AllocationPhaseState(status='ready', dependency_build_status='ready'),
                    AllocationPhaseState(status='requested', dependency_build_status='building'),
                ),
                batch=BatchPhaseState(
                    generation_status='complete',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=1,
                    total_items=1,
                    dependency_status='building',
                ),
            ),
            'building_dependencies',
        ),
        (
            _snapshot(
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='pending',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=0,
                    total_items=0,
                ),
            ),
            'generating',
        ),
        (
            _snapshot(
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='publishing',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=0,
                    total_items=10,
                ),
            ),
            'generating',
        ),
        (
            _snapshot(
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='materializing',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=5,
                    total_items=10,
                ),
            ),
            'materializing',
        ),
        (
            _snapshot(
                status='running',
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='complete',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=10,
                    total_items=10,
                ),
            ),
            'running_children',
        ),
        (_snapshot(status='queued', policy_deferred=True), 'waiting_policy'),
        (_snapshot(status='running', policy_deferred=True), 'running'),
        (
            _snapshot(
                status='queued',
                allocations=(AllocationPhaseState(status='requested'),),
                policy_deferred=True,
            ),
            'waiting_resources',
        ),
        (
            _snapshot(
                status='stopping',
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='complete',
                    control_state='stopping',
                    submission_closed=True,
                    materialized_count=10,
                    total_items=10,
                ),
            ),
            'finalizing',
        ),
        (_snapshot(status='running', result_artifact_status='uploading'), 'uploading_result'),
        (_snapshot(status='running', result_artifact_status='ready'), 'uploading_result'),
        (_snapshot(status='succeeded', allocations=(AllocationPhaseState(status='draining'),)), 'releasing'),
        (_snapshot(status='failed', allocations=(AllocationPhaseState(status='released'),)), 'failed'),
        (_snapshot(status='expired', allocations=()), 'expired'),
        (_snapshot(status='running'), 'running'),
    ],
)
def test_execution_phase_projects_persisted_state(snapshot: ExecutionPhaseSnapshot, phase: str):
    assert project_execution_phase(snapshot) == phase


@pytest.mark.parametrize(
    ('snapshot', 'message'),
    [
        (
            _snapshot(allocations=(AllocationPhaseState(status='mystery'),)),
            'unknown_allocation_status',
        ),
        (
            _snapshot(
                allocations=(
                    AllocationPhaseState(status='requested', dependency_build_status='ready'),
                    AllocationPhaseState(status='requested', dependency_build_status='failed'),
                )
            ),
            'conflicting_dependency_build_status',
        ),
        (
            _snapshot(spec_type='nexus_batch', batch=None),
            'batch_state_missing',
        ),
        (
            _snapshot(
                status='running',
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='complete',
                    control_state='finalized',
                    submission_closed=True,
                    materialized_count=1,
                    total_items=1,
                ),
            ),
            'non_terminal_execution_has_finalized_batch',
        ),
        (
            _snapshot(
                spec_type='nexus_batch',
                batch=BatchPhaseState(
                    generation_status='complete',
                    control_state='active',
                    submission_closed=False,
                    materialized_count=1,
                    total_items=2,
                ),
            ),
            'complete_generation_count_mismatch',
        ),
        (
            _snapshot(result_artifact_status='failed'),
            'unknown_result_artifact_status',
        ),
    ],
)
def test_execution_phase_rejects_unknown_or_contradictory_state(
    snapshot: ExecutionPhaseSnapshot,
    message: str,
):
    with pytest.raises(PhaseProjectionError, match=message):
        project_execution_phase(snapshot)
