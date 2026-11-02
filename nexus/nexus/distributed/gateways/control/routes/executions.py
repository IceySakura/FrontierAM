from __future__ import annotations

from typing import Any

from aiohttp import web
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from nexus.distributed.access.auth import AuthContext, auth_context
from nexus.distributed.execution.models import ExecutionMode, ExecutionRecord, RouteStatus
from nexus.distributed.gateways.common.keys import (
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.gateways.common.platform_metrics import bind_metric_environment
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.gateways.control.routes._allocations import allocation_payload as _allocation_payload
from nexus.distributed.gateways.control.routes._allocations import dependency_builds_by_id as _dependency_builds_by_id
from nexus.distributed.gateways.control.routes._build_logs import get_execution_build_logs
from nexus.distributed.gateways.control.routes._shared import is_truthy_query as _is_truthy_query
from nexus.distributed.gateways.control.routes._shared import query_int as _query_int
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.persistence.orm import RequestBudgetLeaseRow
from nexus.distributed.policies.request_rate import (
    configured_rate_per_minute,
    consume_request_rate_snapshot,
    request_budget_key,
    resolve_request_selection,
)
from nexus.distributed.policies.scheduler import consume_snapshot, resolve_selection, selection_keys
from nexus.distributed.repositories.allocation import AllocationRepository
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.batch import BatchRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.execution_credentials import ExecutionCredentialRepository
from nexus.distributed.repositories.execution_phase import ExecutionPhaseRepository
from nexus.distributed.repositories.policy import PolicyRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import new_id


class CreateExecutionPayload(BaseModel):
    execution_id: str | None = None
    mode: str = ExecutionMode.JOB
    spec_type: str
    instance_mode: str = 'asyncio'
    spec: dict[str, Any] = Field(default_factory=dict)
    driver_config: dict[str, Any] = Field(default_factory=dict)
    placement: dict[str, Any] = Field(default_factory=dict)
    route_status: str = RouteStatus.NONE
    slot_cost: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=1, ge=1)
    priority: int = 0
    queue_ttl_seconds: int | None = None
    execution_ttl_seconds: int | None = None
    stop_timeout_seconds: int | None = 60

    @field_validator('execution_id')
    @classmethod
    def validate_execution_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or value in {'.', '..'} or len(value) > 64:
            raise ValueError('invalid execution_id')
        if any(character in value for character in ('/', '\\', '\x00')):
            raise ValueError('invalid execution_id')
        return value


def _assignment_summary(assignment: dict[str, Any] | None) -> dict[str, Any] | None:
    if assignment is None:
        return None
    return {
        'assignment_id': assignment.get('assignment_id'),
        'execution_id': assignment.get('execution_id'),
        'attempt_id': assignment.get('attempt_id'),
        'attempt_seq': assignment.get('attempt_seq'),
        'manager_id': assignment.get('manager_id'),
        'manager_session_id': assignment.get('manager_session_id'),
        'status': assignment.get('status'),
        'slot_cost': assignment.get('slot_cost'),
        'endpoint_ready': assignment.get('endpoint_ready'),
        'created_at': assignment.get('created_at'),
        'updated_at': assignment.get('updated_at'),
    }


def _execution_summary(item: dict[str, Any]) -> dict[str, Any]:
    summary = {
        'execution_id': item.get('execution_id'),
        'shard_id': item.get('shard_id'),
        'mode': item.get('mode'),
        'spec_type': item.get('spec_type'),
        'instance_mode': item.get('instance_mode'),
        'status': item.get('status'),
        'version': item.get('version'),
        'phase': item.get('phase'),
        'route_status': item.get('route_status'),
        'slot_cost': item.get('slot_cost'),
        'current_assignment_id': item.get('current_assignment_id'),
        'current_attempt_id': item.get('current_attempt_id'),
        'attempt_count': item.get('attempt_count'),
        'max_attempts': item.get('max_attempts'),
        'next_schedule_at': item.get('next_schedule_at'),
        'stop_timeout_seconds': item.get('stop_timeout_seconds'),
        'created_at': item.get('created_at'),
        'updated_at': item.get('updated_at'),
        'owner_principal_id': item.get('owner_principal_id'),
        'parent_execution_id': item.get('parent_execution_id'),
        'environment_id': item.get('environment_id'),
        'pool_id': item.get('pool_id'),
        'allocation_id': item.get('allocation_id'),
        'workload': item.get('workload'),
        'submission_sha256': item.get('submission_sha256'),
        'allocation_spec_hash': item.get('allocation_spec_hash'),
        'result_artifact_id': item.get('result_artifact_id'),
        'first_schedulable_at': item.get('first_schedulable_at'),
        'stop_reason': item.get('stop_reason'),
        'queue_expires_at': item.get('queue_expires_at'),
        'execution_expires_at': item.get('execution_expires_at'),
        'stop_deadline_at': item.get('stop_deadline_at'),
        'cleanup_done_at': item.get('cleanup_done_at'),
        'current_assignment': _assignment_summary(item.get('current_assignment')),
    }
    if item.get('spec_type') == 'nexus_batch':
        summary['outcome'] = item.get('outcome')
    return summary


def _execution_owner_filter(request: web.Request, context: AuthContext) -> str | None:
    owner_principal_id = request.query.get('owner') or None
    if owner_principal_id is not None:
        request.app[AUTHORIZATION_KEY].require_shared_owner_access(context, owner_principal_id)
    return owner_principal_id


def _execution_lookup_filters(request: web.Request, context: AuthContext) -> dict[str, Any]:
    return {
        'owner_principal_ids': request.app[AUTHORIZATION_KEY].accessible_owner_ids(context),
        'environment_id': None,
    }


async def _get_authorized_execution(
    request: web.Request,
    session: Any,
    context: AuthContext,
    repo: ExecutionRepository,
    *,
    scope: str,
    summary_only: bool = False,
):
    lookup = repo.get_summary if summary_only else repo.get_execution
    execution = await lookup(
        request.match_info['execution_id'],
        **_execution_lookup_filters(request, context),
    )
    if execution.owner_principal_id is None:
        raise web.HTTPNotFound(text='resource_not_found')
    request.app[AUTHORIZATION_KEY].require_execution_access(
        context,
        owner_principal_id=execution.owner_principal_id,
        environment_id=execution.environment_id,
    )
    if not context.is_admin and context.auth_type != 'disabled':
        if execution.environment_id is None:
            raise web.HTTPNotFound(text='resource_not_found')
        await EnvironmentRepository(session).require_grant_scope(
            execution.owner_principal_id,
            execution.environment_id,
            required_scope=scope,
        )
    bind_metric_environment(request, execution.environment_id)
    return execution


async def _execution_list_environment_access(
    request: web.Request,
    session: Any,
    context: AuthContext,
) -> tuple[str | None, frozenset[str] | None, frozenset[tuple[str, str]] | None]:
    requested = request.query.get('environment_id') or None
    if context.is_admin or context.auth_type == 'disabled':
        return requested, None, None
    owner_principal_ids = request.app[AUTHORIZATION_KEY].accessible_owner_ids(context) or frozenset()
    allowed_pairs = await EnvironmentRepository(session).list_granted_owner_environment_pairs(
        owner_principal_ids,
        required_scope='execution:read',
    )
    if context.credential_environment_id is not None:
        allowed_pairs = frozenset(pair for pair in allowed_pairs if pair[1] == context.credential_environment_id)
    if requested is not None:
        allowed_pairs = frozenset(pair for pair in allowed_pairs if pair[1] == requested)
    return requested, None, allowed_pairs


async def create_execution(request: web.Request) -> web.Response:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_admin(context)
    payload = await _read_model(request, CreateExecutionPayload)
    config = request.app[CONFIG_KEY]
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        execution = await repo.create_execution(
            CreateExecution(
                execution_id=payload.execution_id or new_id('exe'),
                mode=payload.mode,
                spec_type=payload.spec_type,
                instance_mode=payload.instance_mode,
                spec=payload.spec,
                driver_config=payload.driver_config,
                placement=payload.placement,
                route_status=payload.route_status,
                slot_cost=payload.slot_cost,
                max_attempts=payload.max_attempts,
                priority=payload.priority,
                queue_ttl_seconds=payload.queue_ttl_seconds,
                execution_ttl_seconds=payload.execution_ttl_seconds,
                stop_timeout_seconds=payload.stop_timeout_seconds,
                owner_principal_id=context.principal_id,
                owner_tenant=context.tenant,
                created_by=context.principal_id,
            ),
            now=utcnow(),
        )
        await session.commit()
    return web.json_response(to_jsonable(execution), status=201)


async def get_execution(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    summary_only = _is_truthy_query(request.query.get('summary'))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            repo,
            scope='execution:read',
            summary_only=summary_only,
        )
        phase = await ExecutionPhaseRepository(session).project(execution)
        credential_kinds = await ExecutionCredentialRepository(session).list_kinds(
            execution.execution_id,
            now=utcnow(),
        )
        outcomes = await BatchRepository(session, config=config).batch_outcomes(
            [execution.execution_id] if execution.spec_type == 'nexus_batch' else []
        )
    payload = to_jsonable(execution)
    if not summary_only and execution.parent_execution_id:
        from nexus.distributed.persistence.orm import BatchItemAttemptRow

        async with request.app[SESSION_FACTORY_KEY]() as session:
            attempt = (
                (
                    await session.execute(
                        select(
                            BatchItemAttemptRow.parent_execution_id,
                            BatchItemAttemptRow.item_id,
                            BatchItemAttemptRow.attempt_no,
                        ).where(BatchItemAttemptRow.child_execution_id == execution.execution_id)
                    )
                )
                .mappings()
                .first()
            )
        payload['batch_context'] = dict(attempt) if attempt else {'parent_execution_id': execution.parent_execution_id}
    if summary_only:
        payload = _execution_summary(payload)
    if execution.spec_type == 'nexus_batch':
        payload['outcome'] = outcomes.get(execution.execution_id)
    payload['phase'] = phase
    payload['credential_kinds'] = list(credential_kinds)
    return web.json_response(payload)


def _effective_policy_target(execution: ExecutionRecord, policy_key: str) -> tuple[Any, str, str]:
    requested = execution.placement.get(policy_key)
    effective_spec_type = execution.spec_type
    environment_id = execution.environment_id or execution.placement.get('environment', '')
    if execution.spec_type == 'nexus_batch':
        batch_cfg = execution.spec.get('batch') if isinstance(execution.spec, dict) else None
        child_template = batch_cfg.get('child_template') if isinstance(batch_cfg, dict) else None
        if isinstance(child_template, dict):
            requested = child_template.get(policy_key)
            environment_id = child_template.get('environment_id') or environment_id
        effective_spec_type = 'nexus_single'
    return requested, effective_spec_type, str(environment_id)


async def get_scheduler_policy(request: web.Request) -> web.Response:
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
        requested, effective_spec_type, environment_id = _effective_policy_target(execution, 'scheduler_policy')
        selection = resolve_selection(
            config.policy_executor,
            requested,
            environment_id=environment_id,
            owner_principal_id=execution.owner_principal_id,
            spec_type=effective_spec_type,
        )
        if selection is None:
            return web.json_response({'enabled': False})
        keys = selection_keys(selection, config.policy_executor)
        rows = await PolicyRepository(session, shard_count=config.shard_count).read_scopes(keys)
        now = utcnow()
        effect = consume_snapshot(selection, config.policy_executor, rows, now=now)
        payload = {
            'enabled': True,
            'allow': effect.allow,
            'reason_code': effect.reason,
            'binding': selection.binding,
            'mode': config.policy_executor.bindings[selection.binding].mode,
            'scopes': [
                {
                    'scope_id': scope,
                    'snapshot': rows[key].snapshot_json if key in rows else None,
                    'valid_until': rows[key].valid_until if key in rows else None,
                    'last_error': rows[key].last_error if key in rows else None,
                    'next_evaluate_at': rows[key].next_evaluate_at if key in rows else None,
                }
                for scope, key in zip(selection.scopes, keys, strict=True)
            ],
            'next_schedule_at': None if execution.spec_type == 'nexus_batch' else execution.next_schedule_at,
        }
    return web.json_response(to_jsonable(payload))


async def get_request_policy(request: web.Request) -> web.Response:
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
        requested, effective_spec_type, environment_id = _effective_policy_target(execution, 'request_policy')
        selection = resolve_request_selection(
            config.policy_executor,
            requested,
            environment_id=environment_id,
            owner_principal_id=execution.owner_principal_id,
            spec_type=effective_spec_type,
        )
        if selection is None:
            return web.json_response({'enabled': False})
        key = request_budget_key(selection)
        rows = await PolicyRepository(session, shard_count=config.shard_count).read_scopes([key])
        scope_row = rows.get(key)
        now = utcnow()
        reading = consume_request_rate_snapshot(selection, config.policy_executor, scope_row, now=now)
        binding = config.policy_executor.bindings[selection.binding]
        configured_quota_qpm = configured_rate_per_minute(binding, selection.scope)
        snapshot_age_seconds: float | None = None
        if scope_row is not None and isinstance(scope_row.snapshot_json, dict):
            generated_raw = scope_row.snapshot_json.get('generated_at')
            if isinstance(generated_raw, str):
                from datetime import datetime as _dt

                try:
                    snapshot_age_seconds = max(0.0, (now - _dt.fromisoformat(generated_raw)).total_seconds())
                except ValueError:
                    snapshot_age_seconds = None
        lease_rows = (
            await session.scalars(
                select(RequestBudgetLeaseRow)
                .where(RequestBudgetLeaseRow.budget_key == key)
                .order_by(RequestBudgetLeaseRow.manager_id, RequestBudgetLeaseRow.manager_session_id)
            )
        ).all()
        managers = [
            {
                'manager_id': row.manager_id,
                'manager_session_id': row.manager_session_id,
                'binding': row.binding_id,
                'scope': row.scope_id,
                'observed_mode': row.mode,
                'observed_config_revision': row.config_revision,
                'lease_id': row.lease_id if row.valid_until and row.valid_until > now else None,
                'lease_version': row.lease_version if row.valid_until and row.valid_until > now else None,
                'granted_qpm': row.granted_qpm if row.valid_until and row.valid_until > now else 0.0,
                'valid_until': row.valid_until if row.valid_until and row.valid_until > now else None,
                'demand_active': row.demand_expires_at > now,
                'reason_code': row.reason_code,
                'metrics': row.last_result_json.get('metrics') if isinstance(row.last_result_json, dict) else None,
            }
            for row in lease_rows
        ]
        payload = {
            'enabled': True,
            'binding': selection.binding,
            'scope': selection.scope,
            'budget_key': key,
            'mode': reading.mode,
            'config_revision': reading.config_revision,
            'configured_quota_qpm': configured_quota_qpm,
            'effective_rate_per_minute': reading.rate_per_minute,
            'allow': reading.allow,
            'ready': reading.ready,
            'reason_code': reading.reason_code,
            'snapshot_version': reading.snapshot_version,
            'snapshot_age_seconds': snapshot_age_seconds,
            'valid_until': reading.valid_until,
            'managers': managers,
        }
    return web.json_response(to_jsonable(payload))


async def list_executions(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    legacy_offset = 'offset' in request.query
    offset = _query_int(request, 'offset', 0, min_value=0) if legacy_offset else 0
    cursor = request.query.get('cursor')
    if legacy_offset and cursor is not None:
        raise web.HTTPBadRequest(text='cursor_and_offset_are_mutually_exclusive')
    status = request.query.get('status')
    mode = request.query.get('mode')
    spec_type = request.query.get('spec_type')
    q = request.query.get('q')
    top_level_only = _is_truthy_query(request.query.get('top_level'))
    owner_principal_id = _execution_owner_filter(request, context)
    owner_principal_ids = request.app[AUTHORIZATION_KEY].accessible_owner_ids(context)
    summary = _is_truthy_query(request.query.get('summary'))
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        environment_id, environment_ids, owner_environment_pairs = await _execution_list_environment_access(
            request,
            session,
            context,
        )
        total = (
            await repo.count_executions(
                status=status,
                mode=mode,
                spec_type=spec_type,
                q=q,
                top_level_only=top_level_only,
                owner_principal_id=owner_principal_id,
                owner_principal_ids=owner_principal_ids,
                owner_environment_pairs=owner_environment_pairs,
                environment_id=environment_id,
                environment_ids=environment_ids,
            )
            if legacy_offset or _is_truthy_query(request.query.get('include_total'))
            else None
        )
        executions = await repo.list_executions(
            summary_only=summary,
            status=status,
            mode=mode,
            spec_type=spec_type,
            q=q,
            top_level_only=top_level_only,
            owner_principal_id=owner_principal_id,
            owner_principal_ids=owner_principal_ids,
            owner_environment_pairs=owner_environment_pairs,
            environment_id=environment_id,
            environment_ids=environment_ids,
            limit=limit if legacy_offset else limit + 1,
            offset=offset,
            cursor=cursor,
        )
        has_more = not legacy_offset and len(executions) > limit
        executions = executions[:limit]
        phases = await ExecutionPhaseRepository(session).project_many(executions)
        outcomes = await BatchRepository(session, config=config).batch_outcomes([
            execution.execution_id for execution in executions if execution.spec_type == 'nexus_batch'
        ])
    items = to_jsonable(executions)
    for item in items:
        item['phase'] = phases[item['execution_id']]
        if item['spec_type'] == 'nexus_batch':
            item['outcome'] = outcomes.get(item['execution_id'])
    if summary:
        items = [_execution_summary(item) for item in items]
    payload: dict[str, Any] = {'items': items, 'limit': limit}
    if legacy_offset:
        payload.update({'total': total, 'offset': offset})
    else:
        payload['next_cursor'] = repo.page_cursor(executions[-1]) if has_more and executions else None
        if total is not None:
            payload['total'] = total
    return web.json_response(payload)


async def activate_execution(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:command')
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        await _get_authorized_execution(
            request,
            session,
            context,
            repo,
            scope='execution:command',
        )
        execution = await repo.activate_route(request.match_info['execution_id'], now=now)
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=execution.owner_principal_id,
            operation='execution.activate',
            resource_type='execution',
            resource_id=execution.execution_id,
            metadata={'route_status': execution.route_status},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(execution))


async def stop_execution(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:stop')
    now = utcnow()
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        await _get_authorized_execution(
            request,
            session,
            context,
            repo,
            scope='execution:stop',
        )
        execution = await repo.request_stop(request.match_info['execution_id'], now=now)
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=execution.owner_principal_id,
            operation='execution.stop',
            resource_type='execution',
            resource_id=execution.execution_id,
            metadata={'stop_reason': execution.stop_reason},
            now=now,
        )
        await session.commit()
    return web.json_response(to_jsonable(execution))


async def list_control_events(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    cursor = _query_int(request, 'cursor', 0, min_value=0)
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        repo = ExecutionRepository(session, shard_count=config.shard_count)
        await _get_authorized_execution(
            request,
            session,
            context,
            repo,
            scope='execution:read',
        )
        events = await repo.list_control_events(request.match_info['execution_id'], after_seq=cursor, limit=limit)
    return web.json_response({'items': to_jsonable(events)})


async def get_batch(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    parent_execution_id = request.match_info['execution_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution_repo = ExecutionRepository(session, shard_count=config.shard_count)
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            execution_repo,
            scope='execution:read',
        )
        if execution.spec_type != 'nexus_batch':
            raise web.HTTPNotFound(text='batch_not_found')
        summary = await BatchRepository(session, config=config).batch_summary(
            parent_execution_id,
            include_status_counts=_is_truthy_query(request.query.get('include_status_counts')),
        )
    return web.json_response(to_jsonable(summary))


async def list_batch_items(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    parent_execution_id = request.match_info['execution_id']
    limit = _query_int(request, 'limit', 100, min_value=1, max_value=500)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            ExecutionRepository(session, shard_count=config.shard_count),
            scope='execution:read',
        )
        if execution.spec_type != 'nexus_batch':
            raise web.HTTPNotFound(text='batch_not_found')
        items = await BatchRepository(session, config=config).list_items(
            parent_execution_id,
            status=request.query.get('status'),
            after_item_id=request.query.get('cursor'),
            limit=limit + 1,
            item_id_prefix=request.query.get('item_id_prefix'),
            child_execution_id=request.query.get('child_execution_id'),
            error_code=request.query.get('error_code'),
        )
    has_more = len(items) > limit
    items = items[:limit]
    cursor = items[-1]['item_id'] if has_more and items else None
    return web.json_response({'items': to_jsonable(items), 'next_cursor': cursor})


async def get_batch_item(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    parent_execution_id = request.match_info['execution_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            ExecutionRepository(session, shard_count=config.shard_count),
            scope='execution:read',
        )
        if execution.spec_type != 'nexus_batch':
            raise web.HTTPNotFound(text='batch_not_found')
        item = await BatchRepository(session, config=config).get_item(
            parent_execution_id,
            request.match_info['item_id'],
        )
    return web.json_response(to_jsonable(item))


async def get_batch_item_attempt(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    try:
        attempt_no = int(request.match_info['attempt_no'])
    except ValueError as exc:
        raise web.HTTPBadRequest(text='invalid attempt_no') from exc
    if attempt_no < 0:
        raise web.HTTPBadRequest(text='invalid attempt_no')
    parent_execution_id = request.match_info['execution_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _get_authorized_execution(
            request,
            session,
            context,
            ExecutionRepository(session, shard_count=config.shard_count),
            scope='execution:read',
        )
        if execution.spec_type != 'nexus_batch':
            raise web.HTTPNotFound(text='batch_not_found')
        attempt = await BatchRepository(session, config=config).get_attempt(
            parent_execution_id,
            request.match_info['item_id'],
            attempt_no,
        )
    return web.json_response(to_jsonable(attempt))


async def list_execution_allocations(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, 'execution:read')
    execution_id = request.match_info['execution_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _get_authorized_execution(
            request,
            session,
            context,
            ExecutionRepository(session, shard_count=config.shard_count),
            scope='execution:read',
        )
        allocations = await AllocationRepository(session).list_allocations(
            top_level_execution_id=execution_id,
            limit=10,
        )
        dependency_builds = await _dependency_builds_by_id(session, allocations)
    return web.json_response({
        'items': [
            _allocation_payload(
                allocation,
                admin=context.is_admin,
                dependency_build=(
                    dependency_builds.get(allocation.dependency_build_id)
                    if allocation.dependency_build_id is not None
                    else None
                ),
            )
            for allocation in allocations
        ]
    })


def register_execution_routes(app: web.Application) -> None:
    app.router.add_post('/v1/executions', create_execution)
    app.router.add_get('/v1/executions', list_executions)
    app.router.add_get('/v1/executions/{execution_id}/scheduler-policy', get_scheduler_policy)
    app.router.add_get('/v1/executions/{execution_id}/request-policy', get_request_policy)
    app.router.add_get('/v1/executions/{execution_id}/control-events', list_control_events)
    app.router.add_get('/v1/executions/{execution_id}/batch', get_batch)
    app.router.add_get('/v1/executions/{execution_id}/batch/items', list_batch_items)
    app.router.add_get('/v1/executions/{execution_id}/batch/items/{item_id}', get_batch_item)
    app.router.add_get(
        '/v1/executions/{execution_id}/batch/items/{item_id}/attempts/{attempt_no}',
        get_batch_item_attempt,
    )
    app.router.add_get('/v1/executions/{execution_id}/allocations', list_execution_allocations)
    app.router.add_get('/v1/executions/{execution_id}/build-logs', get_execution_build_logs)
    app.router.add_get('/v1/executions/{execution_id}', get_execution)
    app.router.add_post('/v1/executions/{execution_id}/activate', activate_execution)
    app.router.add_post('/v1/executions/{execution_id}/stop', stop_execution)


__all__ = ['register_execution_routes']
