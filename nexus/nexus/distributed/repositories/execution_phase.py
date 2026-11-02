from __future__ import annotations

import logging

from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.artifacts.run_result import run_result_producer_key
from nexus.distributed.execution.models import (
    ExecutionRecord,
    ExecutionStatus,
)
from nexus.distributed.execution.phase import (
    AllocationPhaseState,
    BatchPhaseState,
    ExecutionPhaseSnapshot,
    PhaseProjectionError,
    project_execution_phase,
)
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
)


logger = logging.getLogger(__name__)


class ExecutionPhaseRepository:
    """Read persisted phase inputs in batches and apply the shared projector."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def project(self, execution: ExecutionRecord) -> str:
        return (await self.project_many((execution,)))[execution.execution_id]

    async def project_many(self, executions: Sequence[ExecutionRecord]) -> dict[str, str]:
        if not executions:
            return {}

        execution_by_id = {execution.execution_id: execution for execution in executions}
        execution_records = tuple(execution_by_id.values())
        execution_ids = tuple(execution_by_id)
        allocations = await self._allocation_states(execution_ids)
        batch_ids = tuple(execution.execution_id for execution in executions if execution.spec_type == 'nexus_batch')
        batches = await self._batch_states(batch_ids) if batch_ids else {}
        policy_deferred_ids = await self._policy_deferred_ids(execution_records)
        artifact_statuses = await self._result_artifact_statuses(execution_records)

        phases: dict[str, str] = {}
        for execution_id, execution in execution_by_id.items():
            snapshot = ExecutionPhaseSnapshot(
                status=execution.status,
                spec_type=execution.spec_type,
                allocations=tuple(allocations.get(execution_id, ())),
                batch=batches.get(execution_id),
                result_artifact_status=artifact_statuses.get(execution_id),
                policy_deferred=execution_id in policy_deferred_ids,
            )
            try:
                phases[execution_id] = project_execution_phase(snapshot)
            except PhaseProjectionError as exc:
                logger.error(
                    'Execution phase projection failed: execution_id=%s error=%s snapshot=%r',
                    execution_id,
                    exc,
                    snapshot,
                )
                phases[execution_id] = 'unknown'
        return phases

    async def _allocation_states(
        self,
        execution_ids: tuple[str, ...],
    ) -> dict[str, list[AllocationPhaseState]]:
        rows = (
            await self.session.execute(
                select(
                    EnvironmentAllocationRow.top_level_execution_id,
                    EnvironmentAllocationRow.status,
                    DependencyBuildRow.status,
                    DependencyBuildRow.next_retry_at,
                )
                .outerjoin(
                    DependencyBuildRow,
                    DependencyBuildRow.dependency_build_id == EnvironmentAllocationRow.dependency_build_id,
                )
                .where(EnvironmentAllocationRow.top_level_execution_id.in_(execution_ids))
                .order_by(
                    EnvironmentAllocationRow.top_level_execution_id,
                    EnvironmentAllocationRow.workload,
                    EnvironmentAllocationRow.allocation_id,
                )
            )
        ).all()
        states: dict[str, list[AllocationPhaseState]] = defaultdict(list)
        for execution_id, allocation_status, dependency_build_status, dependency_build_next_retry_at in rows:
            states[str(execution_id)].append(
                AllocationPhaseState(
                    status=str(allocation_status),
                    dependency_build_status=(
                        str(dependency_build_status) if dependency_build_status is not None else None
                    ),
                    dependency_build_next_retry_at=dependency_build_next_retry_at,
                )
            )
        return states

    async def _batch_states(self, execution_ids: tuple[str, ...]) -> dict[str, BatchPhaseState]:
        rows = (
            await self.session.execute(
                select(
                    BatchRunRow.parent_execution_id,
                    BatchRunRow.generation_status,
                    BatchRunRow.control_state,
                    BatchRunRow.submission_closed,
                    BatchRunRow.materialized_count,
                    BatchRunRow.total_items,
                    BatchRunRow.dependency_status,
                ).where(BatchRunRow.parent_execution_id.in_(execution_ids))
            )
        ).all()
        return {
            row.parent_execution_id: BatchPhaseState(
                generation_status=row.generation_status,
                control_state=row.control_state,
                submission_closed=row.submission_closed,
                materialized_count=row.materialized_count,
                total_items=row.total_items,
                dependency_status=row.dependency_status,
            )
            for row in rows
        }

    async def _policy_deferred_ids(
        self,
        executions: tuple[ExecutionRecord, ...],
    ) -> set[str]:
        queued_ids = tuple(
            execution.execution_id
            for execution in executions
            if execution.status == ExecutionStatus.QUEUED and execution.spec_type != 'nexus_batch'
        )
        if not queued_ids:
            return set()
        rows = (
            await self.session.execute(
                select(ExecutionRow.execution_id).where(
                    ExecutionRow.execution_id.in_(queued_ids),
                    ExecutionRow.last_schedule_error.startswith('policy_deferred:', autoescape=True),
                )
            )
        ).scalars()
        return {str(execution_id) for execution_id in rows}

    async def _result_artifact_statuses(
        self,
        executions: tuple[ExecutionRecord, ...],
    ) -> dict[str, str]:
        direct_ids = {
            execution.result_artifact_id for execution in executions if execution.result_artifact_id is not None
        }
        producer_keys = {
            execution.execution_id: run_result_producer_key(
                execution.execution_id,
                execution.current_attempt_id,
            )
            for execution in executions
            if execution.current_attempt_id is not None and execution.result_artifact_id is None
        }
        if not direct_ids and not producer_keys:
            return {}

        rows = []
        columns = (ArtifactRow.artifact_id, ArtifactRow.producer_key, ArtifactRow.kind, ArtifactRow.status)
        if direct_ids:
            rows.extend(
                (await self.session.execute(select(*columns).where(ArtifactRow.artifact_id.in_(direct_ids)))).all()
            )
        if producer_keys:
            # Match the full existing unique index. A producer-key-only lookup scans artifact history.
            owner_keys = [
                (execution.owner_principal_id, 'run_result', producer_keys[execution.execution_id])
                for execution in executions
                if execution.execution_id in producer_keys and execution.owner_principal_id is not None
            ]
            if owner_keys:
                rows.extend(
                    (
                        await self.session.execute(
                            select(*columns).where(
                                tuple_(ArtifactRow.owner_principal_id, ArtifactRow.kind, ArtifactRow.producer_key).in_(
                                    owner_keys
                                )
                            )
                        )
                    ).all()
                )
        by_id = {row.artifact_id: row for row in rows}
        by_producer_key = {
            row.producer_key: row for row in rows if row.kind == 'run_result' and row.producer_key is not None
        }

        statuses: dict[str, str] = {}
        for execution in executions:
            if execution.result_artifact_id is not None:
                artifact = by_id.get(execution.result_artifact_id)
                if artifact is None:
                    statuses[execution.execution_id] = 'missing'
                    continue
            else:
                artifact = by_producer_key.get(producer_keys.get(execution.execution_id))
            if artifact is None:
                continue
            statuses[execution.execution_id] = (
                artifact.status if artifact.kind == 'run_result' else f'invalid_kind:{artifact.kind}'
            )
        return statuses


__all__ = ['ExecutionPhaseRepository']
