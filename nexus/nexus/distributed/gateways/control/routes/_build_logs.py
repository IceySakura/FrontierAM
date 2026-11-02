from __future__ import annotations

import asyncio
import ipaddress
import json

from datetime import timedelta

from aiohttp import web
from sqlalchemy import or_, select

from nexus.distributed.access.auth import auth_context
from nexus.distributed.config.views import environment_config_view
from nexus.distributed.dependency_build.logs import BUILD_LOG_PORT, BUILD_LOG_TAIL_BYTES, build_log_token
from nexus.distributed.environment.providers import WorkloadClientResolver
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, HTTP_CLIENT_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.execution_logs import (
    ZhiyanLogClient,
    ZhiyanLogError,
    normalize_zhiyan_records,
    read_remote_log,
)
from nexus.distributed.gateways.control.routes._shared import query_int
from nexus.distributed.persistence.orm import BatchRunRow, DependencyBuildRow, EnvironmentAllocationRow
from nexus.distributed.repositories.execution import ExecutionRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.provider_io import redact_provider_io


def _build_log_payload(request, build, *, source, availability='unavailable', phase=None, lines=None, **kwargs):
    payload = {
        'dependency_build_id': build.dependency_build_id,
        'attempt': build.attempt,
        'status': build.status,
        'phase': phase or build.status,
        'source': source,
        'availability': availability,
        'lines': lines or [],
        'line_count': len(lines or []),
        'partial': False,
        'fetched_at': utcnow(),
        'retry_after_ms': None,
        'error_code': None,
        **kwargs,
    }
    if execution_id := request.match_info.get('execution_id'):
        payload['execution_id'] = execution_id
    return to_jsonable(payload)


async def _read_live_build_logs(request, build, client, *, lines):
    pods = await client.list_job_pods(build.provider_job_name)
    phase = 'waiting_for_pod'
    pod = pods[0] if pods else None
    if pod:
        status = pod.get('status') or {}
        scheduled = next((c for c in status.get('conditions', []) if c.get('type') == 'PodScheduled'), {})
        builder = next((c for c in status.get('containerStatuses', []) if c.get('name') == 'builder'), {})
        state = builder.get('state') or {}
        if scheduled.get('status') == 'False':
            phase = 'waiting_for_scheduling'
        elif 'waiting' in state:
            reason = state['waiting'].get('reason')
            phase = 'image_pull_failed' if reason in {'ErrImagePull', 'ImagePullBackOff'} else 'starting'
        elif 'terminated' in state:
            return _build_log_payload(
                request, build, source='builder_file', phase='finalizing', error_code='build_log_expired'
            )
        elif status.get('podIP'):
            return await _fetch_builder_logs(request, build, pod, lines=lines)
        else:
            phase = 'preparing'
    return _build_log_payload(request, build, source='builder_file', availability='not_started', phase=phase)


async def _fetch_builder_logs(request, build, pod, *, lines):
    address = ipaddress.ip_address(pod['status']['podIP'])
    host = f'[{address}]' if address.version == 6 else str(address)
    kid = pod['metadata']['annotations']['nexus.io/build-log-key-id']
    token = build_log_token(
        request.app[CONFIG_KEY].artifact_ticket_keyring()[kid], build.dependency_build_id, build.attempt
    )
    parsed, partial = await read_remote_log(
        request.app[HTTP_CLIENT_KEY],
        f'http://{host}:{BUILD_LOG_PORT}/v1/internal/logs?lines={lines}',
        headers={'Authorization': f'Bearer {token}'},
        lines=lines,
        max_bytes=BUILD_LOG_TAIL_BYTES,
        timeout=3,
        allow_redirects=False,
    )
    return _build_log_payload(
        request, build, source='builder_file', availability='ready', lines=parsed, partial=partial
    )


async def _read_zhiyan_build_logs(request, build, *, lines):
    config = request.app[CONFIG_KEY]
    settings = config.zhiyan_log_query
    binding = (build.provider_config_snapshot_json or {}).get('builder_zhiyan_log') or {}
    if not settings.enabled or build.provider_type != 'tkex' or not binding.get('log_name'):
        return _build_log_payload(request, build, source='builder_file', error_code='build_log_expired')
    try:
        token = config.read_zhiyan_log_credential(settings.credential_ref)
    except ValueError:
        return _build_log_payload(request, build, source='zhiyan', error_code='zhiyan_log_not_configured')
    now = utcnow()
    terminal_at = build.finished_at or build.updated_at
    end = min(now, terminal_at + timedelta(seconds=settings.ingest_grace_seconds))
    start = max(build.created_at - timedelta(seconds=60), end - timedelta(seconds=settings.query_window_seconds))
    try:
        records = await ZhiyanLogClient(session=request.app[HTTP_CLIENT_KEY], config=settings, token=token).query(
            log_name=binding['log_name'],
            filters=[
                {'key': 'dependency_build_id', 'operator': 'eq', 'value': build.dependency_build_id},
                {'key': 'build_attempt', 'operator': 'eq', 'value': str(build.attempt)},
            ],
            start_time=start,
            end_time=end,
            limit=min(lines, settings.max_lines),
        )
    except ZhiyanLogError as exc:
        return _build_log_payload(request, build, source='zhiyan', error_code=exc.error_code)
    normalized, partial = normalize_zhiyan_records(records, max_lines=lines)
    pending = not normalized and (now - terminal_at).total_seconds() < settings.ingest_grace_seconds
    return _build_log_payload(
        request,
        build,
        source='zhiyan',
        availability='ready' if normalized else 'ingest_pending' if pending else 'unavailable',
        lines=normalized,
        partial=partial or start > build.created_at - timedelta(seconds=60),
        error_code='zhiyan_log_ingest_pending' if pending else None,
        retry_after_ms=2000 if pending else None,
    )


def _apply_log_byte_budget(item, budget):
    # Keep newest records and account for JSON escaping, including multibyte text.
    kept = []
    budget -= 2
    for line in reversed(item['lines']):
        line = redact_provider_io(line)
        size = len(json.dumps(line).encode()) + 2
        if size > budget:
            item['partial'] = True
            if kept or budget < 1024:
                break
            line = {'message': line['message'][: budget // 12 - 64] + ' [truncated]'}
            size = len(json.dumps(line).encode()) + 2
        kept.append(line)
        budget -= size
    item['lines'] = list(reversed(kept))
    item['line_count'] = len(kept)


async def get_execution_build_logs(request: web.Request) -> web.Response:
    from nexus.distributed.gateways.control.routes.executions import _get_authorized_execution

    lines = query_int(request, 'lines', 200, min_value=1, max_value=1000)
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            ExecutionRepository(session, shard_count=config.shard_count),
            scope='execution:read',
        )
        allocation_build_ids = select(EnvironmentAllocationRow.dependency_build_id).where(
            or_(
                EnvironmentAllocationRow.top_level_execution_id == execution.execution_id,
                EnvironmentAllocationRow.allocation_id == execution.allocation_id,
            ),
            EnvironmentAllocationRow.owner_principal_id == execution.owner_principal_id,
            EnvironmentAllocationRow.environment_id == execution.environment_id,
        )
        batch = await session.get(BatchRunRow, execution.execution_id)
        child_build_id = batch.child_dependency_build_id if batch else None
        builds = list(
            await session.scalars(
                select(DependencyBuildRow)
                .where(
                    or_(
                        DependencyBuildRow.dependency_build_id.in_(allocation_build_ids),
                        DependencyBuildRow.dependency_build_id == child_build_id,
                    ),
                    DependencyBuildRow.owner_principal_id == execution.owner_principal_id,
                    DependencyBuildRow.environment_id == execution.environment_id,
                )
                .order_by(DependencyBuildRow.created_at.desc())
                .limit(10)
            )
        )
    return await build_logs_response(request, builds, lines=lines)


async def build_logs_response(request, builds, *, lines):
    resolver = None
    budget = BUILD_LOG_TAIL_BYTES // max(len(builds), 1)
    items = []
    try:
        for build in builds:
            if build.status not in {'requested', 'building'}:
                item = await _read_zhiyan_build_logs(request, build, lines=lines)
            elif not build.provider_job_name or not build.provider_type:
                item = _build_log_payload(
                    request,
                    build,
                    source='builder_file',
                    availability='not_started',
                    phase='waiting_for_job',
                )
            else:
                try:
                    if resolver is None:
                        resolver = WorkloadClientResolver(config=environment_config_view(request.app[CONFIG_KEY]))
                    resolved = resolver.resolve(
                        provider_type=build.provider_type,
                        provider_config_ref=build.provider_config_ref,
                        snapshot=build.provider_config_snapshot_json,
                    )
                    item = await asyncio.wait_for(
                        _read_live_build_logs(request, build, resolved.client, lines=lines), timeout=5
                    )
                except Exception:
                    item = _build_log_payload(request, build, source='builder_file', error_code='build_log_unavailable')
            _apply_log_byte_budget(item, budget)
            items.append(item)
    finally:
        if resolver is not None:
            await resolver.close()
    return web.json_response({'items': items}, headers={'Cache-Control': 'no-store'})
