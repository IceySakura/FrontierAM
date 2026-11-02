from __future__ import annotations

from typing import Any

from sqlalchemy import select

from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.persistence.orm import DependencyBuildRow


def allocation_payload(
    allocation: Any,
    *,
    admin: bool,
    dependency_build: DependencyBuildRow | None = None,
) -> dict[str, Any]:
    payload = to_jsonable(allocation)
    payload['degraded'] = payload.get('status') == 'ready' and int(payload.get('ready_manager_count') or 0) < int(
        payload.get('desired_manager_count') or 0
    )
    if dependency_build is not None:
        build_payload = {
            'dependency_build_id': dependency_build.dependency_build_id,
            'status': dependency_build.status,
            'attempt': dependency_build.attempt,
            'next_retry_at': dependency_build.next_retry_at,
            'started_at': dependency_build.started_at,
            'finished_at': dependency_build.finished_at,
            'updated_at': dependency_build.updated_at,
        }
        if admin:
            build_payload.update({
                'provider_job_name': dependency_build.provider_job_name,
                'error_code': dependency_build.error_code,
                'error_message': dependency_build.error_message,
            })
        payload['dependency_build'] = to_jsonable(build_payload)
    if not admin:
        for field in ('provider_resource_ref', 'last_error_code', 'last_error_message'):
            payload.pop(field, None)
    return payload


async def dependency_builds_by_id(session: Any, allocations: list[Any]) -> dict[str, DependencyBuildRow]:
    build_ids = {allocation.dependency_build_id for allocation in allocations if allocation.dependency_build_id}
    if not build_ids:
        return {}
    rows = (
        await session.scalars(select(DependencyBuildRow).where(DependencyBuildRow.dependency_build_id.in_(build_ids)))
    ).all()
    return {row.dependency_build_id: row for row in rows}


__all__ = ['allocation_payload', 'dependency_builds_by_id']
