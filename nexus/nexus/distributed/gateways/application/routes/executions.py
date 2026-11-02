from __future__ import annotations

import asyncio
import json
import logging
import time

from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, TypeVar

import aiohttp

from aiohttp import web
from pydantic import BaseModel, Field, ValidationError

from nexus.distributed.access.auth import (
    auth_context,
)
from nexus.distributed.artifacts.admission import TransferAdmissionBusyError
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import (
    DomainConflictError,
    ResourceGoneError,
    ResourceNotFoundError,
    TemporaryUnavailableError,
)
from nexus.distributed.execution.models import ExecutionMode
from nexus.distributed.execution.trace_backends import (
    is_live_result_pointer,
    is_run_result_pointer,
    live_trace_metadata,
    single_trace_metadata,
)
from nexus.distributed.gateways.application.routes._cors import add_cors_headers as _add_cors_headers
from nexus.distributed.gateways.application.routes._result_pipeline import (
    RESULT_PIPELINE_KEY,
    ResultPipeline,
    result_pipeline_context,
)
from nexus.distributed.gateways.application.routes.artifact_shared import (
    _artifact_download_deadline,
    _artifact_transfer_reservation,
    _validate_run_result_size,
)
from nexus.distributed.gateways.common.errors import gateway_error_response
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_DOWNLOAD_ADMISSION_KEY,
    ARTIFACT_SPOOL_KEY,
    ARTIFACT_STORE_KEY,
    AUTHORIZATION_KEY,
    CONFIG_KEY,
    HTTP_CLIENT_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.gateways.common.platform_metrics import bind_metric_environment
from nexus.distributed.gateways.common.proxy import (
    proxy_request_headers as _proxy_headers,
)
from nexus.distributed.gateways.common.proxy import (
    proxy_stream as _forward_stream_proxy,
)
from nexus.distributed.gateways.common.serde import to_jsonable
from nexus.distributed.instance_manager.liveness import effective_manager_liveness
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    AssignmentRow,
    ExecutionRow,
    InstanceManagerLivenessRow,
    InstanceManagerRow,
)
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.gateway import GatewayRepository
from nexus.distributed.utils.clock import utcnow
from nexus.utils.common import await_complete
from nexus.utils.platform_metrics import measure_operation, measured, record


STREAM_CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=300)
ModelT = TypeVar('ModelT', bound=BaseModel)
logger = logging.getLogger(__name__)


class SubmitCommandPayload(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


async def _read_model(request: web.Request, model: type[ModelT]) -> ModelT:
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f'invalid JSON body: {exc}') from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise web.HTTPUnprocessableEntity(text=exc.json()) from exc


async def submit_command(request: web.Request) -> web.Response:
    payload = await _read_model(request, SubmitCommandPayload)
    config = request.app[CONFIG_KEY]
    manager_endpoint_url: str | None = None
    context = auth_context(request)
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _authorize_execution(request, session, scope='execution:command')
        repo = GatewayRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        command = await repo.submit_command(
            request.match_info['execution_id'],
            command_type=payload.type,
            payload=payload.payload,
            idempotency_key=payload.idempotency_key,
            created_by=context.principal_id,
            now=utcnow(),
        )
        await AuditRepository(session).append(
            actor_principal_id=context.principal_id,
            target_owner_principal_id=execution.owner_principal_id,
            operation='execution.command',
            resource_type='execution_command',
            resource_id=command.command_id,
            metadata={'type': command.type},
            now=utcnow(),
        )
        if command.assignment_id is not None:
            assignment = await session.get(AssignmentRow, command.assignment_id)
            if assignment is not None:
                manager = await session.get(InstanceManagerRow, assignment.manager_id)
                if manager is not None:
                    manager_endpoint_url = manager.endpoint_url
        await session.commit()
    if manager_endpoint_url is not None:
        await _wakeup_manager(manager_endpoint_url, config, request.app[HTTP_CLIENT_KEY])
    return web.json_response(to_jsonable(command), status=202)


async def _wakeup_manager(endpoint_url: str, config: DistributedConfig, http_client: aiohttp.ClientSession) -> None:
    headers = {}
    internal_token = config.read_internal_token()
    if internal_token:
        headers['X-Nexus-Internal-Token'] = internal_token
        if config.auth_mode == 'disabled':
            headers['Authorization'] = f'Bearer {internal_token}'
    try:
        async with http_client.post(
            f'{endpoint_url.rstrip("/")}/v1/internal/wakeup',
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=2),
        ):
            pass
    except Exception:
        return


async def stream_events(request: web.Request) -> web.StreamResponse:
    execution_id = request.match_info['execution_id']
    config = request.app[CONFIG_KEY]
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _authorize_execution(request, session, scope='execution:read')
        if execution.mode != ExecutionMode.LIVE:
            raise ResourceNotFoundError('events_not_supported')
        repo = GatewayRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        try:
            _, assignment, _ = await repo.resolve_route(execution_id, now=utcnow())
            artifact_events = False
            endpoint_url = assignment.endpoint_url
        except ResourceGoneError:
            try:
                assignment = await _terminal_artifact_assignment(session, execution, artifact_kind='live')
            except ResourceNotFoundError as artifact_error:
                raise ResourceGoneError('trace_events_not_retained') from artifact_error
            artifact_events = True
            try:
                endpoint_url = await _artifact_endpoint_url(session, assignment, config=config)
            except TemporaryUnavailableError as endpoint_error:
                raise ResourceGoneError('trace_events_not_retained') from endpoint_error
    if endpoint_url is None:
        raise TemporaryUnavailableError('endpoint_not_ready')
    trace_events_path = '/artifacts/events' if config.auth_mode == 'disabled' else '/run-trace/events'
    upstream_path = trace_events_path if artifact_events else '/events'
    upstream_url = f'{endpoint_url.rstrip("/")}{upstream_path}'
    if request.query_string:
        upstream_url = f'{upstream_url}?{request.query_string}'
    return await _proxy_stream(request, upstream_url)


async def read_live_result(request: web.Request) -> web.StreamResponse:
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _authorize_execution(request, session, scope='execution:read')
        if execution.mode != ExecutionMode.LIVE:
            raise ResourceNotFoundError('events_not_supported')
        if execution.result_artifact_id is not None:
            stored_object, ref = await _terminal_run_result_object(session, execution)
        else:
            stored_object = None
            ref = None
        if ref is None and not is_live_result_pointer(execution.result_json):
            return web.json_response(to_jsonable(execution.result_json or {}))
    if stored_object is None or ref is None:
        raise ResourceNotFoundError('artifact_not_found')
    return await _read_result_response(request, stored_object=stored_object, ref=ref)


async def read_execution_result(request: web.Request) -> web.StreamResponse:
    async with request.app[SESSION_FACTORY_KEY]() as session:
        execution = await _authorize_execution(request, session, scope='execution:read')
        if execution.result_artifact_id is not None:
            stored_object, ref = await _terminal_run_result_object(session, execution)
        else:
            stored_object = None
            ref = None
        if ref is None and not _is_external_run_result_pointer(execution.result_json):
            return web.json_response(to_jsonable(execution.result_json or {}))
    if stored_object is None or ref is None:
        raise ResourceNotFoundError('artifact_not_found')
    return await _read_result_response(request, stored_object=stored_object, ref=ref)


@contextmanager
def _result_stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    started = time.perf_counter()
    try:
        with measure_operation(name, family='persistence'):
            yield
    finally:
        timings[name] = time.perf_counter() - started


@measured('result_read', family='persistence')
async def _read_result_response(request: web.Request, *, stored_object: Any, ref: ArtifactRef) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    pipeline = request.app[RESULT_PIPELINE_KEY]
    timings: dict[str, float] = {}
    try:
        async with _artifact_download_deadline(config):
            async with AsyncExitStack() as leases:
                with _result_stage('result_queue', timings):
                    await leases.enter_async_context(pipeline.admit())
                path = await leases.enter_async_context(
                    _download_result_file(request, stored_object=stored_object, ref=ref, timings=timings)
                )
                chunks = await leases.enter_async_context(pipeline.stream(path, timings=timings))
                with _result_stage('result_send', timings):
                    return await _send_result_response(request, chunks)
    finally:
        logger.info('Result pipeline: %s', json.dumps({'artifact_id': ref.artifact_id, 'seconds': timings}))


async def _send_result_response(request: web.Request, chunks: AsyncIterator[bytes]) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    response = web.StreamResponse(headers={'Content-Type': 'application/json; charset=utf-8'})
    _add_cors_headers(response, request)
    try:
        await response.prepare(request)
        if request.method != 'HEAD':
            async for chunk in chunks:
                await asyncio.wait_for(response.write(chunk), timeout=config.artifact_download_write_idle_timeout)
        await asyncio.wait_for(response.write_eof(), timeout=config.artifact_download_write_idle_timeout)
    except BaseException:
        response.force_close()
        if request.transport is not None:
            request.transport.close()
        raise
    finally:
        chunk = b''
    return response


async def _terminal_run_result_object(session: Any, execution: ExecutionRow) -> tuple[Any, ArtifactRef]:
    artifact = await session.get(ArtifactRow, execution.result_artifact_id)
    if (
        artifact is None
        or artifact.status != 'ready'
        or artifact.kind != 'run_result'
        or artifact.owner_principal_id != execution.owner_principal_id
        or artifact.environment_id != execution.environment_id
        or (
            execution.current_attempt_id is not None
            and artifact.producer_key != f'run-result:{execution.execution_id}:{execution.current_attempt_id}'
        )
    ):
        expected = artifact is not None and artifact.status in {'expired', 'deleted', 'deleting'}
        record(
            'persistence_operations',
            stage='result_reference',
            operation='read',
            outcome='expected_unavailable' if expected else 'error',
        )
        if artifact is not None and artifact.status == 'ready':
            record('data_integrity_errors', source='result_reference', reason='invalid_result_reference')
        raise ResourceNotFoundError('artifact_not_found')
    ref = ArtifactRef(
        artifact_id=artifact.artifact_id,
        kind='run_result',
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
    )
    owner_principal_id = execution.owner_principal_id
    environment_id = execution.environment_id
    assert owner_principal_id is not None
    assert environment_id is not None
    stored_object = await ArtifactRepository(session).resolve_stored_object(
        ref,
        owner_principal_id=owner_principal_id,
        environment_id=environment_id,
    )
    record('persistence_operations', stage='result_reference', operation='read', outcome='success')
    return stored_object, ref


@asynccontextmanager
async def _download_result_file(
    request: web.Request, *, stored_object: Any, ref: ArtifactRef, timings: dict[str, float]
) -> AsyncIterator[Path]:
    config = request.app[CONFIG_KEY]
    _validate_run_result_size(config, ref.kind, ref.size_bytes)
    capacity = request.app[ARTIFACT_SPOOL_KEY]
    leases = AsyncExitStack()

    async def populate(path: Path) -> None:
        async with _artifact_transfer_reservation(
            request.app[ARTIFACT_DOWNLOAD_ADMISSION_KEY], direction='download', kind=ref.kind, size_bytes=0
        ):
            try:
                await request.app[ARTIFACT_STORE_KEY].get(
                    stored_object, path, expected_size_bytes=ref.size_bytes, expected_sha256=ref.sha256
                )
            except ValueError as exc:
                if str(exc) not in {'artifact_remote_size_mismatch', 'artifact_remote_sha256_mismatch'}:
                    raise
                raise DomainConflictError('artifact_content_identity_mismatch') from exc

    try:
        with _result_stage('result_download', timings):
            try:
                path = await leases.enter_async_context(
                    capacity.open_cached(sha256=ref.sha256, size_bytes=ref.size_bytes, populate=populate)
                )
            except TransferAdmissionBusyError as exc:
                raise TemporaryUnavailableError('artifact_download_busy') from exc
        yield path
    finally:
        await await_complete(leases.aclose())


async def read_execution_info(request: web.Request) -> web.Response:
    config = request.app[CONFIG_KEY]
    execution_id = request.match_info['execution_id']
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await _authorize_execution(request, session, scope='execution:read')
        repo = GatewayRepository(session, manager_healthy_ttl=config.manager_healthy_ttl)
        _, assignment, _ = await repo.resolve_route(execution_id, now=utcnow())
    if assignment.endpoint_url is None:
        raise TemporaryUnavailableError('endpoint_not_ready')
    upstream_url = f'{assignment.endpoint_url.rstrip("/")}/info'
    return web.json_response(await _upstream_json(request, upstream_url))


async def _terminal_artifact_assignment(
    session: Any,
    execution: ExecutionRow,
    *,
    artifact_kind: str | None = None,
) -> AssignmentRow:
    if execution.current_assignment_id is None:
        raise ResourceNotFoundError('artifact_not_found')
    assignment = await session.get(AssignmentRow, execution.current_assignment_id)
    if assignment is None:
        raise ResourceNotFoundError('artifact_not_found')
    if artifact_kind is not None and not _has_run_trace_metadata(
        assignment.runtime_info_json,
        artifact_kind=artifact_kind,
    ):
        raise ResourceNotFoundError('artifact_not_found')
    return assignment


async def _artifact_endpoint_url(session: Any, assignment: AssignmentRow, *, config: DistributedConfig) -> str:
    manager = await session.get(InstanceManagerRow, assignment.manager_id)
    if manager is None or manager.session_id != assignment.manager_session_id:
        raise TemporaryUnavailableError('endpoint_not_ready')
    liveness = await session.get(InstanceManagerLivenessRow, manager.manager_id)
    effective = effective_manager_liveness(manager, liveness)
    if not effective.active or utcnow() - effective.last_heartbeat_at > config.manager_healthy_ttl:
        raise TemporaryUnavailableError('endpoint_not_ready')
    if assignment.endpoint_url:
        return assignment.endpoint_url
    return f'{manager.endpoint_url.rstrip("/")}/v1/internal/executions/{assignment.assignment_id}'


def _is_external_run_result_pointer(value: Any) -> bool:
    return is_live_result_pointer(value) or is_run_result_pointer(value)


def _has_run_trace_metadata(runtime_info: dict[str, Any] | None, *, artifact_kind: str | None = None) -> bool:
    if artifact_kind == 'live':
        return live_trace_metadata(runtime_info) is not None
    if artifact_kind == 'single':
        return single_trace_metadata(runtime_info) is not None
    return live_trace_metadata(runtime_info) is not None or single_trace_metadata(runtime_info) is not None


async def _authorize_execution(request: web.Request, session: Any, *, scope: str) -> ExecutionRow:
    context = auth_context(request)
    request.app[AUTHORIZATION_KEY].require_scope(context, scope)
    execution = await session.get(ExecutionRow, request.match_info['execution_id'])
    if execution is None:
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


async def _proxy_stream(request: web.Request, upstream_url: str) -> web.StreamResponse:
    config = request.app[CONFIG_KEY]
    session = request.app[HTTP_CLIENT_KEY]
    try:
        return await _forward_stream_proxy(
            request,
            upstream_url,
            session=session,
            headers=_proxy_headers(request.headers, config),
            timeout=STREAM_CLIENT_TIMEOUT,
            response_hook=_add_cors_headers,
        )
    except aiohttp.ClientError as exc:
        logger.warning('Runtime stream proxy unavailable: %s', type(exc).__name__)
        return gateway_error_response(TemporaryUnavailableError('executor_unavailable'))


async def _upstream_json(request: web.Request, upstream_url: str) -> dict[str, Any]:
    config = request.app[CONFIG_KEY]
    session = request.app[HTTP_CLIENT_KEY]
    try:
        async with session.get(
            upstream_url,
            headers=_proxy_headers(request.headers, config),
            allow_redirects=False,
        ) as upstream:
            text = await upstream.text()
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = {'body': text}
            if upstream.status >= 400:
                raise TemporaryUnavailableError('artifact_unavailable')
            if not isinstance(payload, dict):
                raise TemporaryUnavailableError('artifact_unavailable')
            return payload
    except aiohttp.ClientError as exc:
        raise TemporaryUnavailableError('artifact_unavailable') from exc


def register_execution_routes(app: web.Application) -> None:
    config = app[CONFIG_KEY]
    app[RESULT_PIPELINE_KEY] = ResultPipeline(config)
    app.cleanup_ctx.append(result_pipeline_context)
    logger.info(
        'Result pipeline limits: %s',
        json.dumps(
            {
                name: getattr(config, name)
                for name in type(config).model_fields
                if name.startswith('artifact_run_result_')
            },
            default=str,
        ),
    )
    app.router.add_get('/v1/executions/{execution_id}/info', read_execution_info)
    app.router.add_get('/v1/executions/{execution_id}/live/events', stream_events)
    app.router.add_get('/v1/executions/{execution_id}/live/result', read_live_result)
    app.router.add_get('/v1/executions/{execution_id}/result', read_execution_result)
    app.router.add_post('/v1/executions/{execution_id}/commands', submit_command)


__all__ = ['register_execution_routes']
