from __future__ import annotations

import asyncio
import hmac

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web

from nexus.distributed.artifacts.admission import (
    TransferAdmission,
    TransferAdmissionBusyError,
)
from nexus.distributed.artifacts.file_io import FileSizeExceededError, write_chunks
from nexus.distributed.artifacts.leases import PeriodicLease
from nexus.distributed.artifacts.tickets import (
    ARTIFACT_CONTENT_TICKET_HEADER,
    ArtifactContentTicketClaims,
)
from nexus.distributed.config import DistributedConfig
from nexus.distributed.errors import (
    DomainConflictError,
    DomainValidationError,
    ResourceNotFoundError,
)
from nexus.distributed.gateways.common.keys import (
    ARTIFACT_CONTENT_TICKET_SIGNER_KEY,
    CONFIG_KEY,
    SESSION_FACTORY_KEY,
)
from nexus.distributed.persistence.orm import (
    ArtifactRefRow,
    ArtifactRow,
    AssignmentRow,
    BatchRunRow,
    DependencyBuildRow,
    EnvironmentAllocationRow,
    ExecutionRow,
)
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.identity import canonical_sha256
from nexus.utils.artifact.files import FileIdentity


def _internal_artifact_idempotency_scope(*parts: object) -> str:
    return canonical_sha256(['internal-artifact-upload-scope', 1, *parts])


@dataclass(frozen=True, slots=True)
class InternalArtifactScope:
    execution: ExecutionRow
    assignment: AssignmentRow | None
    execution_id: str
    assignment_id: str | None
    attempt_id: str | None
    owner_principal_id: str
    environment_id: str
    allocation_id: str
    credential_allocation_id: str


@dataclass(frozen=True, slots=True)
class DependencyBuildArtifactScope:
    build: DependencyBuildRow
    owner_principal_id: str
    environment_id: str


def _artifact_upload_failure_code(exc: BaseException) -> str:
    detail = str(exc)
    if detail == 'artifact_content_identity_mismatch':
        return detail
    if isinstance(exc, web.HTTPRequestEntityTooLarge):
        return 'artifact_content_size_mismatch'
    return type(exc).__name__


@asynccontextmanager
async def _artifact_upload_deadline(config: DistributedConfig):
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError('artifact upload deadline requires an asyncio task')
    expired = False

    def cancel_for_deadline() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(
        config.artifact_upload_deadline.total_seconds(),
        cancel_for_deadline,
    )
    try:
        yield
    except asyncio.CancelledError as exc:
        if not expired:
            raise
        raise web.HTTPGatewayTimeout(text='artifact_upload_deadline_exceeded') from exc
    finally:
        handle.cancel()


@asynccontextmanager
async def _artifact_download_deadline(config: DistributedConfig):
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError('artifact download deadline requires an asyncio task')
    expired = False

    def cancel_for_deadline() -> None:
        nonlocal expired
        expired = True
        task.cancel()

    handle = asyncio.get_running_loop().call_later(
        config.artifact_download_deadline.total_seconds(),
        cancel_for_deadline,
    )
    try:
        yield
    except asyncio.CancelledError as exc:
        if not expired:
            raise
        raise web.HTTPGatewayTimeout(text='artifact_download_deadline_exceeded') from exc
    finally:
        handle.cancel()


async def _write_upload_body(
    request: web.Request,
    destination: Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    max_size_bytes: int,
) -> FileIdentity:
    chunks = request.content.iter_chunked(1024 * 1024).__aiter__()

    async def receive_chunks():
        while True:
            try:
                chunk = await asyncio.wait_for(
                    anext(chunks),
                    timeout=request.app[CONFIG_KEY].artifact_upload_body_idle_timeout,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                raise web.HTTPRequestTimeout(text='artifact_upload_body_idle_timeout') from exc
            yield chunk

    try:
        identity = await write_chunks(
            destination, receive_chunks(), max_size_bytes=min(expected_size_bytes, max_size_bytes), compute_md5=True
        )
    except FileSizeExceededError as exc:
        raise web.HTTPRequestEntityTooLarge(max_size=exc.limit, actual_size=exc.actual) from exc
    if identity.size_bytes != expected_size_bytes or identity.sha256 != expected_sha256:
        raise DomainConflictError('artifact_content_identity_mismatch')
    return identity


def _receiving_upload_lease(
    request: web.Request,
    *,
    upload_id: str,
    operation_id: str,
) -> PeriodicLease:
    config = request.app[CONFIG_KEY]

    async def renew() -> bool:
        async with request.app[SESSION_FACTORY_KEY]() as session:
            renewed = await ArtifactRepository(session).renew_receiving(
                upload_id,
                operation_id=operation_id,
                lease_ttl=config.artifact_upload_lease_ttl,
                now=utcnow(),
            )
            await session.commit()
            return renewed

    return PeriodicLease(
        renew=renew,
        interval=config.artifact_upload_lease_renew_interval.total_seconds(),
        lost_code='artifact_upload_lease_lost',
    )


async def _release_receiving(
    request: web.Request,
    *,
    upload_id: str,
    operation_id: str,
    failure_code: str,
) -> None:
    async with request.app[SESSION_FACTORY_KEY]() as session:
        await ArtifactRepository(session).release_receiving(
            upload_id,
            operation_id=operation_id,
            failure_code=failure_code,
            now=utcnow(),
        )
        await session.commit()


@asynccontextmanager
async def _artifact_transfer_reservation(
    admission: TransferAdmission,
    *,
    direction: str,
    kind: str,
    size_bytes: int,
):
    try:
        async with admission.reserve(kind, size_bytes=size_bytes):
            yield
    except TransferAdmissionBusyError as exc:
        raise web.HTTPServiceUnavailable(
            text=f'artifact_{direction}_busy:{exc}',
            headers={'Retry-After': '1'},
        ) from exc


def _validate_run_result_size(config: DistributedConfig, kind: str, size_bytes: int) -> None:
    if kind == 'run_result' and size_bytes > config.artifact_run_result_max_size_bytes:
        raise web.HTTPRequestEntityTooLarge(
            max_size=config.artifact_run_result_max_size_bytes,
            actual_size=size_bytes,
        )


def _internal_artifact_kind(value: str) -> str:
    normalized = value.replace('_', '-').lower()
    result = {'run-result': 'run_result', 'task-list': 'task_list', 'wheelhouse': 'wheelhouse'}.get(normalized)
    if result is None:
        raise web.HTTPNotFound(text='artifact_kind_not_supported')
    return result


async def _resolve_dependency_build_artifact_scope(
    request: web.Request,
    session: Any,
    *,
    operation: str,
    artifact_id: str | None = None,
) -> DependencyBuildArtifactScope:
    claims = _internal_artifact_claims(request)
    if claims is None or claims.dependency_build_id is None:
        raise web.HTTPForbidden(text='dependency_build_content_ticket_required')
    if operation not in claims.operations:
        raise web.HTTPForbidden(text='artifact_content_operation_denied')
    build = await session.get(DependencyBuildRow, claims.dependency_build_id)
    if (
        build is None
        or build.environment_id != claims.environment_id
        or build.attempt != claims.dependency_build_attempt
    ):
        raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
    if build.status not in {'requested', 'building', 'ready'}:
        raise web.HTTPForbidden(text='artifact_content_scope_released')
    if artifact_id is not None:
        if artifact_id not in claims.artifact_ids:
            raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
        reference = await session.get(
            ArtifactRefRow,
            {
                'artifact_id': artifact_id,
                'resource_type': 'dependency_build',
                'resource_id': build.dependency_build_id,
            },
        )
        if reference is None:
            raise web.HTTPNotFound(text='artifact_not_found')
    return DependencyBuildArtifactScope(
        build=build,
        owner_principal_id=build.owner_principal_id,
        environment_id=build.environment_id,
    )


async def _internal_producer_key(session: Any, *, scope: InternalArtifactScope, kind: str) -> str:
    if kind == 'run_result':
        if scope.execution.spec_type not in {'nexus_single', 'nexus_live', 'nexus_batch'}:
            raise DomainValidationError('artifact_kind_not_allowed_for_execution')
        return f'run-result:{scope.execution_id}:{_required_scope_value(scope.attempt_id, "attempt_id")}'
    if scope.execution.spec_type != 'nexus_batch':
        raise DomainValidationError('artifact_kind_not_allowed_for_execution')
    batch = await session.get(BatchRunRow, scope.execution_id)
    if batch is None or batch.owner_principal_id != scope.owner_principal_id:
        raise ResourceNotFoundError('batch_not_found')
    return f'batch-task-list:{batch.parent_execution_id}:{batch.generation_epoch}'


async def _resolve_internal_artifact_scope(
    request: web.Request,
    session: Any,
    *,
    operation: str,
    artifact_id: str | None = None,
) -> InternalArtifactScope:
    claims = _internal_artifact_claims(request)
    if claims is not None and operation not in claims.operations:
        raise web.HTTPForbidden(text='artifact_content_operation_denied')
    execution_id = _scope_header_or_claim(request, 'X-Nexus-Execution-Id', claims, 'execution_id')
    assignment_id = _scope_header_or_claim(request, 'X-Nexus-Assignment-Id', claims, 'assignment_id')
    attempt_id = _scope_header_or_claim(request, 'X-Nexus-Attempt-Id', claims, 'attempt_id')
    if not execution_id:
        raise web.HTTPBadRequest(text='artifact_execution_scope_required')
    execution = await session.get(ExecutionRow, execution_id)
    if (
        execution is None
        or not execution.owner_principal_id
        or not execution.environment_id
        or not execution.allocation_id
    ):
        raise web.HTTPNotFound(text='artifact_execution_scope_not_found')
    credential_allocation_id = str(execution.allocation_id)
    if claims is not None:
        if claims.environment_id != execution.environment_id:
            raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
        claim_allocation_id = claims.allocation_id
        if claim_allocation_id is None:
            raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
        allocation = await session.get(EnvironmentAllocationRow, claim_allocation_id)
        if allocation is None or allocation.status in {'deleting', 'released'}:
            raise web.HTTPForbidden(text='artifact_content_scope_released')
        if claims.top_level_execution_id is None:
            if claims.execution_id != execution.execution_id or claims.allocation_id != execution.allocation_id:
                raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
        elif (
            allocation.top_level_execution_id != claims.top_level_execution_id
            or allocation.environment_id != claims.environment_id
            or allocation.owner_principal_id != execution.owner_principal_id
            or claims.execution_id != claims.top_level_execution_id
            or (
                execution.execution_id != claims.top_level_execution_id
                and execution.allocation_id != claims.allocation_id
            )
        ):
            raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
        credential_allocation_id = claim_allocation_id
    if (
        artifact_id is not None
        and claims is not None
        and artifact_id not in claims.artifact_ids
        and (
            claims.top_level_execution_id is None
            or not await _artifact_is_linked_to_scope(
                session,
                artifact_id=artifact_id,
                execution_id=execution_id,
                allocation_id=credential_allocation_id,
            )
        )
    ):
        raise web.HTTPForbidden(text='artifact_content_scope_mismatch')

    assignment: AssignmentRow | None = None
    if assignment_id or attempt_id or operation != 'artifact:read':
        if not assignment_id or not attempt_id:
            raise web.HTTPBadRequest(text='artifact_attempt_scope_required')
        assignment = await session.get(AssignmentRow, assignment_id)
        if (
            assignment is None
            or execution.current_assignment_id != assignment_id
            or execution.current_attempt_id != attempt_id
            or assignment.execution_id != execution_id
            or assignment.attempt_id != attempt_id
            or assignment.owner_principal_id != execution.owner_principal_id
            or assignment.environment_id != execution.environment_id
            or assignment.allocation_id != execution.allocation_id
        ):
            raise web.HTTPForbidden(text='artifact_attempt_scope_mismatch')
        if claims is not None and (
            claims.assignment_id not in {None, assignment_id} or claims.attempt_id not in {None, attempt_id}
        ):
            raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
    elif claims is None:
        raise web.HTTPBadRequest(text='artifact_attempt_scope_required')

    return InternalArtifactScope(
        execution=execution,
        assignment=assignment,
        execution_id=execution_id,
        assignment_id=assignment_id,
        attempt_id=attempt_id,
        owner_principal_id=execution.owner_principal_id,
        environment_id=execution.environment_id,
        allocation_id=execution.allocation_id,
        credential_allocation_id=credential_allocation_id,
    )


async def _artifact_is_linked_to_scope(
    session: Any,
    *,
    artifact_id: str,
    execution_id: str,
    allocation_id: str,
) -> bool:
    for resource_type, resource_id in (('execution', execution_id), ('allocation', allocation_id)):
        reference = await session.get(
            ArtifactRefRow,
            {
                'artifact_id': artifact_id,
                'resource_type': resource_type,
                'resource_id': resource_id,
            },
        )
        if reference is not None:
            return True
    return False


def _internal_artifact_claims(request: web.Request) -> ArtifactContentTicketClaims | None:
    raw_ticket = request.headers.get(ARTIFACT_CONTENT_TICKET_HEADER)
    if raw_ticket:
        try:
            return request.app[ARTIFACT_CONTENT_TICKET_SIGNER_KEY].verify(raw_ticket, now=utcnow())
        except ValueError as exc:
            raise web.HTTPUnauthorized(text=str(exc)) from exc
    config = request.app[CONFIG_KEY]
    try:
        internal_token = config.read_internal_token()
    except ValueError as exc:
        raise web.HTTPServiceUnavailable(text='internal_auth_unavailable') from exc
    supplied_token = request.headers.get('X-Nexus-Internal-Token')
    if not internal_token or supplied_token is None or not hmac.compare_digest(supplied_token, internal_token):
        raise web.HTTPUnauthorized(text='internal_auth_required')
    return None


def _scope_header_or_claim(
    request: web.Request,
    header: str,
    claims: ArtifactContentTicketClaims | None,
    claim_name: str,
) -> str | None:
    value = request.headers.get(header)
    claim_value = getattr(claims, claim_name, None) if claims is not None else None
    allocation_scoped_execution = (
        claims is not None and claim_name == 'execution_id' and claims.top_level_execution_id is not None
    )
    if value and claim_value and value != claim_value and not allocation_scoped_execution:
        raise web.HTTPForbidden(text='artifact_content_scope_mismatch')
    return value or claim_value


def _required_scope_value(value: str | None, field: str) -> str:
    if not value:
        raise DomainValidationError(f'{field}_required')
    return value


def _validate_upload_claims(claims: Any, upload: Any, artifact: ArtifactRow) -> None:
    if (
        claims.artifact_id != upload.artifact_id
        or claims.owner_principal_id != upload.owner_principal_id
        or claims.environment_id != artifact.environment_id
        or claims.expected_kind != upload.expected_kind
        or claims.expected_size_bytes != upload.expected_size_bytes
        or claims.expected_sha256 != upload.expected_sha256
    ):
        raise web.HTTPUnauthorized(text='invalid_upload_ticket')


def _artifact_ref_payload(artifact: Any) -> dict[str, Any]:
    if artifact.status != 'ready':
        raise DomainConflictError('artifact_not_ready')
    return {
        'artifact_id': artifact.artifact_id,
        'sha256': artifact.sha256,
        'size_bytes': artifact.size_bytes,
        'kind': artifact.kind,
    }
