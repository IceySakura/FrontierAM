from __future__ import annotations

from typing import Any
from urllib.parse import quote

from aiohttp import web

from nexus.distributed.access.auth import auth_context
from nexus.distributed.gateways.common.keys import AUTHORIZATION_KEY, CONFIG_KEY, SESSION_FACTORY_KEY
from nexus.distributed.gateways.common.platform_metrics import bind_metric_environment
from nexus.distributed.gateways.control.routes._shared import read_model as _read_model
from nexus.distributed.persistence.db import run_idempotent_transaction as _run_idempotent_transaction
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.submission import SubmissionRepository
from nexus.distributed.submission.specs import BatchSubmission, LiveSubmission, SingleSubmission
from nexus.distributed.submission.workload_image import workload_image_repository
from nexus.distributed.utils.clock import utcnow


async def create_submission(request: web.Request) -> web.Response:
    submission_type = request.match_info['submission_type']
    model = {'single': SingleSubmission, 'live': LiveSubmission, 'batch': BatchSubmission}.get(submission_type)
    if model is None:
        raise web.HTTPNotFound()
    submission = await _read_model(request, model)
    context = auth_context(request)
    policy = request.app[AUTHORIZATION_KEY]
    policy.require_scope(context, f'submission:{submission_type}')
    policy.require_environment(context, submission.environment_id)
    owner = context.principal_id
    now = utcnow()

    async def create(session: Any) -> Any:
        result = await SubmissionRepository(session, config=request.app[CONFIG_KEY]).create(
            submission,
            actor_principal_id=context.principal_id,
            owner_principal_id=owner,
            owner_tenant=context.tenant,
            now=now,
        )
        if not result.reused:
            await AuditRepository(session).append(
                actor_principal_id=context.principal_id,
                target_owner_principal_id=owner,
                operation=f'submission.{submission_type}.create',
                resource_type='execution',
                resource_id=result.execution.execution_id,
                metadata={
                    'environment_id': submission.environment_id,
                    'submission_sha256': result.execution.submission_sha256,
                },
                now=now,
            )
        return result

    result = await _run_idempotent_transaction(
        request.app[SESSION_FACTORY_KEY],
        create,
        conflict_markers=('uq_executions_idempotency', 'executions.idempotency_scope'),
    )
    bind_metric_environment(request, result.execution.environment_id)
    control_path = f'/v1/executions/{result.execution.execution_id}'
    config = request.app[CONFIG_KEY]
    control_url = f'{config.control_public_base_url.rstrip("/")}{control_path}'
    application_base = config.application_public_base_url.rstrip('/')
    application_path = f'{application_base}{control_path}'
    payload = {
        'execution_id': result.execution.execution_id,
        'status': result.execution.status,
        'phase': result.phase,
        'reused': result.reused,
        'control_url': control_url,
        'result_url': f'{application_path}/result',
    }
    payload.update(
        _submission_web_ui_urls(
            console_base_url=config.console_public_base_url,
            execution_id=result.execution.execution_id,
            spec_type=result.execution.spec_type,
        )
    )
    if result.execution.spec_type == 'nexus_live':
        payload['events_url'] = f'{application_path}/live/events'
        payload['command_url'] = f'{application_path}/commands'
    elif result.execution.spec_type == 'nexus_batch':
        payload['batch_url'] = f'{control_url}/batch'
    if result.batch_id is not None:
        payload['batch_id'] = result.batch_id
    if result.credential_kinds:
        payload['credential_kinds'] = list(result.credential_kinds)
    return web.json_response(payload, status=202)


def _submission_web_ui_urls(*, console_base_url: str, execution_id: str, spec_type: str) -> dict[str, str]:
    base_url = console_base_url.rstrip('/')
    encoded_execution_id = quote(execution_id, safe='')
    task_path = (
        f'/v1/console/batches/{encoded_execution_id}'
        if spec_type == 'nexus_batch'
        else f'/v1/console/executions/{encoded_execution_id}'
    )
    environment_path = '/v1/console/my-environments'
    return {
        'task_web_ui': f'{base_url}{task_path}',
        'environment_web_ui': f'{base_url}{environment_path}',
        'tracedb_web_ui': f'{base_url}{environment_path}',
    }


async def get_capabilities(request: web.Request) -> web.Response:
    context = auth_context(request)
    config = request.app[CONFIG_KEY]
    try:
        image_repository = workload_image_repository(
            config.environment_base_image,
            require_digest=config.production_mode,
        )
    except ValueError:
        image_repository = None
    return web.json_response({
        'submissions': ['single', 'live', 'batch'],
        'admin': context.is_admin,
        'environment_providers': config.environment_enabled_providers,
        'workload_image': {
            'enabled': image_repository is not None,
            'repository': image_repository,
            'base_image': config.environment_base_image,
        },
        'durable_run_trace': False,
        'automatic_environment_scale_down': False,
    })


def register_submission_routes(app: web.Application) -> None:
    app.router.add_get('/v1/capabilities', get_capabilities)
    app.router.add_post('/v1/submissions/{submission_type}', create_submission)


__all__ = ['register_submission_routes']
