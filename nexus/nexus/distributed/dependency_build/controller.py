from __future__ import annotations

import asyncio
import logging

from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import async_sessionmaker

from nexus.distributed.artifacts.tickets import ArtifactContentTicketSigner
from nexus.distributed.config.views import DependencyBuildConfigView
from nexus.distributed.dependency_build.models import DependencyBuildStatus
from nexus.distributed.dependency_build.rendering import DependencyBuildJobRenderer, RenderedDependencyBuild
from nexus.distributed.environment.kubernetes import (
    WorkloadConflictError,
    WorkloadNotFoundError,
    WorkloadPermanentError,
)
from nexus.distributed.environment.ports import DependencyBuildWorkloadClient, WorkloadClientResolverPort
from nexus.distributed.observability import ControlLoopWork
from nexus.distributed.ports import ReconcileController
from nexus.distributed.repositories.dependency_build import (
    DependencyBuildClaim,
    DependencyBuildRecord,
    DependencyBuildRepository,
)
from nexus.distributed.utils.clock import utcnow
from nexus.distributed.utils.leases import LeaseLostError, PeriodicLease
from nexus.utils.provider_io import redact_provider_io


logger = logging.getLogger(__name__)
ResultT = TypeVar('ResultT')
_DIAGNOSTIC_POD_LIMIT = 2
_DIAGNOSTIC_LOG_TAIL_LINES = 50
_DIAGNOSTIC_LOG_LIMIT_BYTES = 256 * 1024
_DIAGNOSTIC_REQUEST_TIMEOUT_SECONDS = 5.0
_DIAGNOSTIC_MESSAGE_MARKER = '; provider_diagnostics: '
_FAILURE_MESSAGE_LIMIT = 2048
_PERMANENT_BUILD_FAILURES = {
    'dependency_build_key_mismatch',
    'dependency_nexus_wheel_invalid',
    'dependency_source_package_invalid',
}


class DependencyBuildFailureError(RuntimeError):
    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


class DependencyBuildController(ReconcileController):
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker,
        config: DependencyBuildConfigView,
        client_resolver: WorkloadClientResolverPort,
        owner_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.config = config
        self.owner_id = owner_id or f'{config.environment_controller_owner_id}:dependency-build'
        self.renderer = DependencyBuildJobRenderer(config=config)
        self.client_resolver = client_resolver
        self._signer = ArtifactContentTicketSigner(
            keys=config.artifact_ticket_keyring(),
            current_key_id=config.artifact_upload_ticket_current_key_id,
            ttl=config.environment_artifact_content_ticket_ttl,
        )
        self._last_stats = ControlLoopWork()

    @property
    def last_stats(self) -> ControlLoopWork:
        return self._last_stats

    async def reconcile_once(self) -> int:
        changed = 0
        processed_build_ids: set[str] = set()
        for _ in range(self.config.dependency_build_batch_size):
            async with self.session_factory() as session:
                claim = await DependencyBuildRepository(session).claim_next(
                    operation_owner=self.owner_id,
                    lease_ttl=self.config.dependency_build_lease_ttl,
                    now=utcnow(),
                    exclude_dependency_build_ids=processed_build_ids,
                )
                await session.commit()
            if claim is None:
                break
            processed_build_ids.add(claim.record.dependency_build_id)
            changed += 1
            try:
                async with self._claim_lease(claim) as lease:
                    await self._reconcile(claim, lease=lease)
            except LeaseLostError:
                logger.warning(
                    'Dependency build lease lost; stale owner stopped: owner=%s build_id=%s operation_id=%s',
                    self.owner_id,
                    claim.record.dependency_build_id,
                    claim.operation_id,
                )
            except WorkloadPermanentError as exc:
                if claim.record.status == DependencyBuildStatus.READY:
                    await self._release_if_owned(claim, retry_after=self.config.dependency_build_retry)
                else:
                    await self._fail_if_owned(claim, exc, retry=False)
            except Exception:
                logger.exception(
                    'Dependency build reconcile failed: owner=%s build_id=%s operation_id=%s',
                    self.owner_id,
                    claim.record.dependency_build_id,
                    claim.operation_id,
                )
                await self._release_if_owned(claim, retry_after=self.config.dependency_build_retry)
        self._last_stats = ControlLoopWork(
            scanned=len(processed_build_ids),
            claimed=len(processed_build_ids),
            changed=changed,
        )
        return changed

    async def close(self) -> None:
        pass

    async def _reconcile(self, claim: DependencyBuildClaim, *, lease: PeriodicLease) -> None:
        record = claim.record
        client = self._client_for(record)
        if record.status == DependencyBuildStatus.READY:
            await self._cleanup_ready(claim, client, lease=lease)
            return
        if record.status in {DependencyBuildStatus.REQUESTED, DependencyBuildStatus.FAILED}:
            attempt = record.attempt + 1
            provider_job_name = self.renderer.job_name(record.dependency_build_id, attempt)
            async with self.session_factory() as session:
                record = await DependencyBuildRepository(session).begin_attempt(
                    record.dependency_build_id,
                    operation_id=claim.operation_id,
                    attempt=attempt,
                    provider_job_name=provider_job_name,
                    now=utcnow(),
                    deadline_seconds=self.config.dependency_builder_active_deadline_seconds,
                )
                await session.commit()
            claim = replace(claim, record=record)
            await lease.renew_now()

        token = self._signer.issue_dependency_build(
            dependency_build_id=record.dependency_build_id,
            attempt=record.attempt,
            environment_id=record.environment_id,
            operations=['artifact:read', 'wheelhouse:write', 'dependency_build:report_failure'],
            artifact_ids=[
                record.build_spec.source_package.artifact_id,
                record.build_spec.nexus_wheel.artifact_id,
                record.build_spec.uv_lock.artifact_id,
            ],
            now=_ticket_epoch(utcnow(), ttl=self.config.environment_artifact_content_ticket_ttl),
        )
        rendered = self.renderer.render(record, token=token)
        await self._ensure_secret(client, rendered, record, lease=lease)
        job_name = record.provider_job_name
        if not job_name:
            await self._fail(claim, RuntimeError('dependency_build_job_name_missing'), retry=True, lease=lease)
            return
        try:
            job = await self._provider_call(lease, lambda: client.get_job(job_name))
        except WorkloadNotFoundError:
            if record.wheelhouse is not None and record.metadata is not None:
                await self._finalize_success(claim, client, rendered, record, lease=lease)
                return
            try:
                job = await self._ensure_job(client, rendered, record, lease=lease)
            except WorkloadNotFoundError:
                await self._release(claim, retry_after=self.config.dependency_build_retry, lease=lease)
                return
        _validate_owned(job, record)
        outcome = _job_outcome(job)
        has_collector = any(c.get('name') == 'zhiyan-agent' for c in job['spec']['template']['spec']['containers'])
        if has_collector:
            pods = await self._provider_call(lease, lambda: client.list_job_pods(job_name))
            builder_outcome = _builder_outcome(pods, now=utcnow())
            if builder_outcome is not None:
                outcome = builder_outcome
        if outcome == 'active':
            await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)
            return
        if outcome == 'failed':
            record = await self._stage_failure_diagnostics(claim, client, job, record, lease=lease)
            with suppress(WorkloadNotFoundError):
                await self._provider_call(lease, lambda: client.delete_job(job_name))
            with suppress(WorkloadNotFoundError):
                await self._provider_call(lease, lambda: client.delete_secret(rendered.secret_name))
            error = DependencyBuildFailureError(
                record.error_code or 'RuntimeError',
                record.error_message or _job_failure_message(job),
            )
            await self._fail(
                claim,
                error,
                retry=(
                    record.attempt < self.config.dependency_build_max_attempts
                    and record.error_code not in _PERMANENT_BUILD_FAILURES
                ),
                lease=lease,
            )
            return
        if record.wheelhouse is None or record.metadata is None:
            await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)
            return
        if has_collector:
            async with self.session_factory() as session:
                now = utcnow()
                await DependencyBuildRepository(session).mark_ready(
                    record.dependency_build_id,
                    operation_id=claim.operation_id,
                    wheelhouse=record.wheelhouse,
                    metadata=record.metadata,
                    now=now,
                    cleanup_at=now + timedelta(seconds=30),
                )
                await session.commit()
            return
        try:
            await self._provider_call(lease, lambda: client.delete_job(job_name))
        except WorkloadNotFoundError:
            await self._finalize_success(claim, client, rendered, record, lease=lease)
        except Exception:
            logger.exception(
                'Failed to delete completed dependency build Job; retrying cleanup: build_id=%s job=%s',
                record.dependency_build_id,
                job_name,
            )
            await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)
        else:
            await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)

    async def _cleanup_ready(
        self, claim: DependencyBuildClaim, client: DependencyBuildWorkloadClient, *, lease: PeriodicLease
    ) -> None:
        record = claim.record
        job_name = record.provider_job_name
        if job_name:
            try:
                job = await self._provider_call(lease, lambda: client.get_job(job_name))
            except WorkloadNotFoundError:
                pass
            else:
                _validate_owned(job, record)
                with suppress(WorkloadNotFoundError):
                    await self._provider_call(lease, lambda: client.delete_job(job_name))
            try:
                await self._provider_call(lease, lambda: client.get_job(job_name))
            except WorkloadNotFoundError:
                pass
            else:
                await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)
                return
        secret_name = self.renderer.secret_name(record.dependency_build_id, record.attempt)
        with suppress(WorkloadNotFoundError):
            await self._provider_call(lease, lambda: client.delete_secret(secret_name))
        async with self.session_factory() as session:
            await DependencyBuildRepository(session).release_claim(
                record.dependency_build_id,
                operation_id=claim.operation_id,
                now=utcnow(),
            )
            await session.commit()

    async def _stage_failure_diagnostics(
        self,
        claim: DependencyBuildClaim,
        client: DependencyBuildWorkloadClient,
        job: dict[str, Any],
        record: DependencyBuildRecord,
        *,
        lease: PeriodicLease,
    ) -> DependencyBuildRecord:
        base_message = record.error_message or _job_failure_message(job)
        diagnostics = await self._collect_failure_diagnostics(client, record.provider_job_name or '')
        error_message = _failure_message(base_message, diagnostics)
        error_code = record.error_code or 'RuntimeError'
        await lease.renew_now()
        async with self.session_factory() as session:
            staged = await DependencyBuildRepository(session).stage_failure_diagnostics(
                record.dependency_build_id,
                operation_id=claim.operation_id,
                attempt=record.attempt,
                error_code=error_code,
                error_message=error_message,
                now=utcnow(),
            )
            await session.commit()
        return staged

    async def _collect_failure_diagnostics(
        self,
        client: DependencyBuildWorkloadClient,
        job_name: str,
    ) -> str:
        summaries: list[str] = []
        try:
            await asyncio.wait_for(
                self._append_failure_diagnostics(
                    summaries,
                    client,
                    job_name,
                ),
                timeout=_DIAGNOSTIC_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception('Could not fully collect failed dependency build diagnostics: job=%s', job_name)
        return '; '.join(summaries)

    async def _append_failure_diagnostics(
        self,
        summaries: list[str],
        client: DependencyBuildWorkloadClient,
        job_name: str,
    ) -> None:
        try:
            job_events = await client.list_resource_events(job_name)
        except Exception:
            logger.exception('Could not list failed dependency build Job events: job=%s', job_name)
        else:
            summaries.extend(_event_summaries(job_events))
        pods = await client.list_job_pods(job_name)
        for pod in pods[:_DIAGNOSTIC_POD_LIMIT]:
            pod_name = str((pod.get('metadata') or {}).get('name') or '')
            summaries.extend(_pod_status_summary(pod))
            if not pod_name:
                continue
            try:
                events = await client.list_resource_events(pod_name)
            except Exception:
                logger.exception('Could not list failed dependency build Pod events: pod=%s', pod_name)
            else:
                summaries.extend(_event_summaries(events))
            try:
                log_tail = await client.read_pod_log(
                    pod_name,
                    container='builder',
                    tail_lines=_DIAGNOSTIC_LOG_TAIL_LINES,
                    limit_bytes=_DIAGNOSTIC_LOG_LIMIT_BYTES,
                )
            except Exception:
                logger.exception('Could not read failed dependency build Pod log: pod=%s', pod_name)
            else:
                if compact_log := _compact(log_tail, limit=768):
                    summaries.append(f'log_tail={compact_log}')

    async def _finalize_success(
        self,
        claim: DependencyBuildClaim,
        client: DependencyBuildWorkloadClient,
        rendered: RenderedDependencyBuild,
        record: DependencyBuildRecord,
        *,
        lease: PeriodicLease,
    ) -> None:
        try:
            await self._provider_call(lease, lambda: client.delete_secret(rendered.secret_name))
        except WorkloadNotFoundError:
            pass
        except Exception:
            logger.exception(
                'Failed to delete completed dependency build Secret; retrying cleanup: build_id=%s secret=%s',
                record.dependency_build_id,
                rendered.secret_name,
            )
            await self._release(claim, retry_after=self.config.dependency_build_poll_interval, lease=lease)
            return
        if record.wheelhouse is None or record.metadata is None:
            raise RuntimeError('dependency_build_staged_output_missing')
        async with self.session_factory() as session:
            await DependencyBuildRepository(session).mark_ready(
                record.dependency_build_id,
                operation_id=claim.operation_id,
                wheelhouse=record.wheelhouse,
                metadata=record.metadata,
                now=utcnow(),
            )
            await session.commit()

    async def _release(
        self,
        claim: DependencyBuildClaim,
        *,
        retry_after: timedelta,
        lease: PeriodicLease,
    ) -> None:
        await lease.renew_now()
        now = utcnow()
        async with self.session_factory() as session:
            await DependencyBuildRepository(session).release_claim(
                claim.record.dependency_build_id,
                operation_id=claim.operation_id,
                next_retry_at=now + retry_after,
                now=now,
            )
            await session.commit()

    async def _fail(
        self,
        claim: DependencyBuildClaim,
        error: Exception,
        *,
        retry: bool,
        lease: PeriodicLease,
    ) -> None:
        await lease.renew_now()
        now = utcnow()
        error_code = str(getattr(error, 'error_code', type(error).__name__))
        async with self.session_factory() as session:
            await DependencyBuildRepository(session).mark_failed(
                claim.record.dependency_build_id,
                operation_id=claim.operation_id,
                error_code=error_code,
                error_message=str(error),
                retry_at=now + self.config.dependency_build_retry if retry else None,
                now=now,
            )
            await session.commit()

    async def _ensure_secret(
        self,
        client: DependencyBuildWorkloadClient,
        rendered: RenderedDependencyBuild,
        record: DependencyBuildRecord,
        *,
        lease: PeriodicLease,
    ) -> None:
        try:
            existing = await self._provider_call(lease, lambda: client.get_secret(rendered.secret_name))
        except WorkloadNotFoundError:
            try:
                existing = await self._provider_call(lease, lambda: client.create_secret(rendered.secret))
            except WorkloadConflictError:
                existing = await self._provider_call(lease, lambda: client.get_secret(rendered.secret_name))
        _validate_owned(existing, record)
        current_hash = ((existing.get('metadata') or {}).get('annotations') or {}).get('nexus.io/credential-sha256')
        desired_hash = rendered.secret['metadata']['annotations']['nexus.io/credential-sha256']
        if current_hash != desired_hash:
            existing = await self._provider_call(
                lease,
                lambda: client.patch_secret(rendered.secret_name, rendered.secret),
            )
            _validate_owned(existing, record)

    async def _ensure_job(
        self,
        client: DependencyBuildWorkloadClient,
        rendered: RenderedDependencyBuild,
        record: DependencyBuildRecord,
        *,
        lease: PeriodicLease,
    ) -> dict[str, Any]:
        try:
            existing = await self._provider_call(lease, lambda: client.get_job(rendered.job_name))
        except WorkloadNotFoundError:
            try:
                existing = await self._provider_call(lease, lambda: client.create_job(rendered.job))
            except WorkloadConflictError:
                existing = await self._provider_call(lease, lambda: client.get_job(rendered.job_name))
        _validate_owned(existing, record)
        return existing

    def _claim_lease(self, claim: DependencyBuildClaim) -> PeriodicLease:
        async def renew() -> bool:
            try:
                async with self.session_factory() as session:
                    await DependencyBuildRepository(session).renew(
                        claim.record.dependency_build_id,
                        operation_id=claim.operation_id,
                        lease_ttl=self.config.dependency_build_lease_ttl,
                        now=utcnow(),
                    )
                    await session.commit()
                return True
            except ValueError as exc:
                if str(exc) == 'dependency_build_lease_lost':
                    return False
                raise

        return PeriodicLease(
            renew=renew,
            interval=self.config.dependency_build_lease_renew_interval.total_seconds(),
            lost_code='dependency_build_lease_lost',
        )

    async def _provider_call(
        self,
        lease: PeriodicLease,
        operation: Callable[[], Awaitable[ResultT]],
    ) -> ResultT:
        await lease.renew_now()
        result = await operation()
        await lease.renew_now()
        return result

    async def _release_if_owned(self, claim: DependencyBuildClaim, *, retry_after: timedelta) -> None:
        try:
            async with self._claim_lease(claim) as lease:
                await self._release(claim, retry_after=retry_after, lease=lease)
        except LeaseLostError:
            return

    async def _fail_if_owned(self, claim: DependencyBuildClaim, error: Exception, *, retry: bool) -> None:
        try:
            async with self._claim_lease(claim) as lease:
                await self._fail(claim, error, retry=retry, lease=lease)
        except LeaseLostError:
            return

    def _client_for(self, record: DependencyBuildRecord) -> DependencyBuildWorkloadClient:
        if (
            record.provider_type is None
            or record.provider_config_ref is None
            or record.provider_config_snapshot is None
        ):
            raise WorkloadPermanentError('dependency_build_provider_snapshot_missing')
        try:
            resolved = self.client_resolver.resolve(
                provider_type=record.provider_type,
                provider_config_ref=record.provider_config_ref,
                snapshot=record.provider_config_snapshot,
            )
        except ValueError as exc:
            raise WorkloadPermanentError(str(exc)) from exc
        return resolved.client


def _validate_owned(resource: dict[str, Any], record: DependencyBuildRecord) -> None:
    annotations = (resource.get('metadata') or {}).get('annotations') or {}
    if annotations.get('nexus.io/managed-by') != 'nexus':
        raise WorkloadPermanentError('dependency_build_resource_not_owned')
    if annotations.get('nexus.io/dependency-build-id') != record.dependency_build_id:
        raise WorkloadPermanentError('dependency_build_resource_id_conflict')
    if annotations.get('nexus.io/dependency-build-key') != record.build_key:
        raise WorkloadPermanentError('dependency_build_resource_key_conflict')
    if annotations.get('nexus.io/dependency-build-attempt') != str(record.attempt):
        raise WorkloadPermanentError('dependency_build_resource_attempt_conflict')


def _pod_status_summary(pod: dict[str, Any]) -> list[str]:
    metadata = pod.get('metadata') or {}
    status = pod.get('status') or {}
    summary: list[str] = []
    if pod_name := _compact(metadata.get('name'), limit=128):
        summary.append(f'pod={pod_name}')
    if phase := _compact(status.get('phase'), limit=64):
        summary.append(f'phase={phase}')
    if reason := _compact(status.get('reason'), limit=128):
        summary.append(f'pod_reason={reason}')
    if message := _compact(status.get('message'), limit=256):
        summary.append(f'pod_message={message}')

    container_statuses: list[dict[str, Any]] = []
    for field in ('initContainerStatuses', 'containerStatuses'):
        values = status.get(field)
        if isinstance(values, list):
            container_statuses.extend(item for item in values if isinstance(item, dict))
    for container_status in container_statuses:
        if container_name := _compact(container_status.get('name'), limit=64):
            summary.append(f'container={container_name}')
        state = container_status.get('state') or {}
        waiting = state.get('waiting') if isinstance(state, dict) else None
        if isinstance(waiting, dict):
            if waiting_reason := _compact(waiting.get('reason'), limit=128):
                summary.append(f'waiting_reason={waiting_reason}')
            if waiting_message := _compact(waiting.get('message'), limit=256):
                summary.append(f'waiting_message={waiting_message}')
        terminated = state.get('terminated') if isinstance(state, dict) else None
        if isinstance(terminated, dict):
            exit_code = terminated.get('exitCode')
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                summary.append(f'exit_code={exit_code}')
            if termination_reason := _compact(terminated.get('reason'), limit=128):
                summary.append(f'termination_reason={termination_reason}')
            if termination_message := _compact(terminated.get('message'), limit=256):
                summary.append(f'termination_message={termination_message}')
    return summary


def _event_summaries(events: list[dict[str, Any]]) -> list[str]:
    warnings = [event for event in events if event.get('type') == 'Warning']
    selected = (warnings or events)[-3:]
    summaries: list[str] = []
    for event in selected:
        event_type = _compact(event.get('type'), limit=32) or 'Unknown'
        reason = _compact(event.get('reason'), limit=128) or 'Unknown'
        value = f'event={event_type}/{reason}'
        count = event.get('count')
        if isinstance(count, int) and not isinstance(count, bool):
            value += f' count={count}'
        if message := _compact(event.get('message'), limit=256):
            value += f' message={message}'
        summaries.append(value)
    return summaries


def _compact(value: Any, *, limit: int) -> str:
    if value is None:
        return ''
    redacted = redact_provider_io(str(value))
    return ' '.join(str(redacted).split())[:limit]


def _failure_message(base_message: str, diagnostics: str) -> str:
    if not diagnostics:
        return _compact(base_message, limit=_FAILURE_MESSAGE_LIMIT)
    safe_diagnostics = _compact(diagnostics, limit=_FAILURE_MESSAGE_LIMIT // 2)
    base_message = base_message.partition(_DIAGNOSTIC_MESSAGE_MARKER)[0]
    base_limit = _FAILURE_MESSAGE_LIMIT - len(_DIAGNOSTIC_MESSAGE_MARKER) - len(safe_diagnostics)
    safe_base = _compact(base_message, limit=base_limit)
    return f'{safe_base}{_DIAGNOSTIC_MESSAGE_MARKER}{safe_diagnostics}'


def _builder_outcome(pods: list[dict[str, Any]], *, now: datetime) -> str | None:
    # Successful builds can unblock scheduling; failed attempts retain their upload grace.
    if len(pods) != 1:
        return None
    statuses = (pods[0].get('status') or {}).get('containerStatuses', [])
    builder = next((item for item in statuses if item.get('name') == 'builder'), {})
    terminated = (builder.get('state') or {}).get('terminated') or {}
    try:
        finished = datetime.fromisoformat(terminated['finishedAt'].replace('Z', '+00:00'))
    except (KeyError, ValueError, TypeError):
        return None
    if finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    if terminated.get('exitCode') == 0:
        return 'succeeded'
    collector = next((item for item in statuses if item.get('name') == 'zhiyan-agent'), {})
    if 'terminated' not in (collector.get('state') or {}) and (current - finished).total_seconds() < 30:
        return 'active'
    return 'succeeded' if terminated.get('exitCode') == 0 else 'failed'


def _job_outcome(job: dict[str, Any]) -> str:
    status = job.get('status') or {}
    conditions = status.get('conditions') or []
    uncounted = status.get('uncountedTerminatedPods') or {}
    uncounted_succeeded = uncounted.get('succeeded') if isinstance(uncounted, dict) else None
    if (
        int(status.get('succeeded') or 0) > 0
        or any(item.get('type') == 'Complete' and str(item.get('status')).lower() == 'true' for item in conditions)
        or isinstance(uncounted_succeeded, list)
        and bool(uncounted_succeeded)
    ):
        return 'succeeded'
    uncounted_failed = uncounted.get('failed') if isinstance(uncounted, dict) else None
    no_job_retries = (job.get('spec') or {}).get('backoffLimit') == 0
    if (
        int(status.get('failed') or 0) > 0
        or any(item.get('type') == 'Failed' and str(item.get('status')).lower() == 'true' for item in conditions)
        or no_job_retries
        and isinstance(uncounted_failed, list)
        and bool(uncounted_failed)
    ):
        return 'failed'
    return 'active'


def _job_failure_message(job: dict[str, Any]) -> str:
    for condition in (job.get('status') or {}).get('conditions') or []:
        if condition.get('type') == 'Failed':
            return str(condition.get('message') or condition.get('reason') or 'dependency build job failed')[:2048]
    return 'dependency build job failed'


def _ticket_epoch(now: datetime, *, ttl: timedelta) -> datetime:
    rotation_seconds = max(1, int(ttl.total_seconds()) // 2)
    aware = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    bucket = int(aware.timestamp()) // rotation_seconds * rotation_seconds
    return datetime.fromtimestamp(bucket, tz=timezone.utc).replace(tzinfo=None)


__all__ = [
    'DependencyBuildController',
    'DependencyBuildWorkloadClient',
]
