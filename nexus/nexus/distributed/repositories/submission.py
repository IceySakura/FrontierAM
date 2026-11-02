from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.core.module_dependencies import ModuleDependencyError, selectors_from_run_input
from nexus.distributed.access.run_trace_backends import RunTraceBackendAccessPolicy
from nexus.distributed.config import DistributedConfig
from nexus.distributed.dependency_build.wheelhouse import WheelhouseBuildSpec
from nexus.distributed.environment.limits import (
    SourcePackageLimitsSpec,
    effective_artifact_size_limit,
    effective_environment_resource_limits,
    source_package_limits,
    validate_timeout_defaults,
)
from nexus.distributed.environment.models import (
    EnvironmentAllocationType,
    EnvironmentRecord,
)
from nexus.distributed.environment.provider_config import require_provider_type_enabled
from nexus.distributed.errors import DomainConflictError, DomainValidationError
from nexus.distributed.execution.models import (
    ExecutionMode,
    ExecutionRecord,
    RouteStatus,
)
from nexus.distributed.persistence.orm import BatchRunRow, EnvironmentAllocationRow, ExecutionRow
from nexus.distributed.policies.request_rate import resolve_request_selection
from nexus.distributed.policies.scheduler import resolve_selection
from nexus.distributed.repositories.allocation import (
    AllocationRepository,
    AllocationResources,
    CreateEnvironmentAllocation,
)
from nexus.distributed.repositories.artifact import ArtifactRepository
from nexus.distributed.repositories.dependency_build import DependencyBuildRecord, DependencyBuildRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.repositories.execution import CreateExecution, ExecutionRepository
from nexus.distributed.repositories.execution_credentials import ExecutionCredentialRepository
from nexus.distributed.repositories.execution_phase import ExecutionPhaseRepository
from nexus.distributed.repositories.provider_config import ProviderConfigRepository
from nexus.distributed.repositories.quota import QuotaRepository
from nexus.distributed.submission.resources import NormalizedResourceGroup, normalize_resource_group
from nexus.distributed.submission.specs import (
    BatchSubmission,
    ImageCode,
    LiveSubmission,
    LocalTaskListGeneration,
    RemoteMaterializeGeneration,
    SingleSubmission,
    SourcePackageCode,
    Submission,
    artifact_refs,
    resolve_source_package_extras,
)
from nexus.distributed.submission.workload_image import resolve_workload_image
from nexus.distributed.utils.identity import (
    canonical_sha256,
    new_id,
    validate_environment_id,
    validate_pool_id,
    validate_submission_idempotency_key,
)
from nexus.utils.platform_metrics import bind_metric_attributes


_GIB = 1024**3
_DEFAULT_WORKSPACE_BYTES = 4 * _GIB
_TOP_LEVEL_MAX_ATTEMPTS = 2
_TIMEOUT_FIELDS = (
    'provisioning_timeout_seconds',
    'queue_ttl_seconds',
    'execution_ttl_seconds',
    'stop_timeout_seconds',
)


@dataclass(frozen=True)
class SubmissionResult:
    execution: ExecutionRecord
    allocations: tuple[Any, ...]
    reused: bool
    phase: str
    batch_id: str | None = None
    credential_kinds: tuple[str, ...] = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.execution, name)


class SubmissionRepository:
    def __init__(self, session: AsyncSession, *, config: DistributedConfig) -> None:
        self.session = session
        self.config = config

    async def create(
        self,
        submission: Submission,
        *,
        actor_principal_id: str,
        owner_principal_id: str,
        owner_tenant: str | None,
        now: datetime,
    ) -> SubmissionResult:
        validate_submission_idempotency_key(submission.idempotency_key)
        validate_environment_id(submission.environment_id)
        if submission.pool is not None:
            validate_pool_id(submission.pool)
        submit_kind = _submission_kind(submission)
        submission_sha256 = _submission_sha256(submission)
        idempotency_scope = _idempotency_scope(
            owner_principal_id=owner_principal_id,
            environment_id=submission.environment_id,
            submit_kind=submit_kind,
        )
        environment, grant = await EnvironmentRepository(self.session).require_submission_access(
            owner_principal_id,
            submission.environment_id,
            required_scope=f'submission:{submit_kind}',
        )
        bind_metric_attributes(environment=environment.environment_id)
        existing = await self.session.scalar(
            select(ExecutionRow).where(
                ExecutionRow.idempotency_scope == idempotency_scope,
                ExecutionRow.idempotency_key == submission.idempotency_key,
            )
        )
        if existing is not None:
            if (
                existing.owner_principal_id != owner_principal_id
                or existing.environment_id != submission.environment_id
                or existing.spec_type != f'nexus_{submit_kind}'
                or existing.submission_sha256 != submission_sha256
            ):
                raise DomainConflictError('submission_idempotency_conflict')
            return await self._result(existing, reused=True, now=now)

        policy_selection = resolve_selection(
            self.config.policy_executor,
            submission.scheduler_policy.model_dump() if submission.scheduler_policy else None,
            environment_id=submission.environment_id,
            owner_principal_id=owner_principal_id,
            spec_type='nexus_single' if isinstance(submission, BatchSubmission) else f'nexus_{submit_kind}',
        )
        request_policy_selection = resolve_request_selection(
            self.config.policy_executor,
            submission.request_policy.model_dump() if submission.request_policy else None,
            environment_id=submission.environment_id,
            owner_principal_id=owner_principal_id,
            spec_type='nexus_single' if isinstance(submission, BatchSubmission) else f'nexus_{submit_kind}',
        )
        submission = submission.model_copy(
            update={
                'scheduler_policy': policy_selection,
                'request_policy': request_policy_selection,
            }
        )
        requested_trace_backends = (
            submission.run_trace_backends.sinks if submission.run_trace_backends is not None else None
        )
        resolved_trace_backends = await RunTraceBackendAccessPolicy(
            self.session,
            config=self.config,
        ).resolve_for_submission(owner_principal_id, requested_trace_backends)
        driver_config = (
            {}
            if resolved_trace_backends.sinks is None
            else {
                'run_trace_backends': {
                    'sinks': list(resolved_trace_backends.sinks),
                    'append_artifact_fallback': True,
                }
            }
        )

        _validate_environment_shape(submission, environment)
        submission = _resolve_submission_cpu_memory_ratio(
            submission, auto_increase_cpu=self.config.submission_auto_increase_cpu
        )
        if submission.resources is not None and submission.resources.managers.liveness is not None:
            if environment.allocation_type != EnvironmentAllocationType.DYNAMIC:
                raise DomainValidationError('manager_liveness_requires_dynamic_allocation')
            try:
                submission.resources.managers.liveness.validate_platform(self.config)
            except ValueError as exc:
                raise DomainValidationError(str(exc)) from exc
        require_provider_type_enabled(environment.provider_type, self.config.environment_enabled_providers)
        provider_config = await ProviderConfigRepository(self.session).validate_environment(
            allocation_type=environment.allocation_type,
            provider_type=environment.provider_type,
            provider_config_ref=environment.provider_config_ref,
            workload_namespace=self.config.environment_namespace,
        )
        provider_config_snapshot = dict(provider_config.config) if provider_config is not None else None
        effective_limits = effective_environment_resource_limits(
            environment.resource_limits,
            grant.limits,
        )
        validate_timeout_defaults(effective_limits, environment.default_timeouts)

        artifact_repository = ArtifactRepository(self.session)
        max_artifact_size = effective_artifact_size_limit(
            effective_limits,
            global_max_size_bytes=self.config.artifact_upload_max_size_bytes,
        )
        package_limits = SourcePackageLimitsSpec.from_package_limits(
            source_package_limits(
                effective_limits,
                global_artifact_max_size_bytes=self.config.artifact_upload_max_size_bytes,
            )
        )
        for ref in artifact_refs(submission):
            if ref.size_bytes > max_artifact_size:
                raise DomainValidationError('artifact_exceeds_environment_size_limit')
            await artifact_repository.resolve_ready_ref(
                ref,
                owner_principal_id=owner_principal_id,
                environment_id=submission.environment_id,
            )

        dependency_build = await self._ensure_dependency_build(
            submission,
            owner_principal_id=owner_principal_id,
            environment_id=environment.environment_id,
            provider_type=environment.provider_type,
            provider_config_ref=environment.provider_config_ref,
            provider_config_snapshot=provider_config_snapshot,
            artifact_repository=artifact_repository,
            source_package_limits=package_limits,
            now=now,
        )
        workload_image = _submission_workload_image(
            submission,
            base_image=self.config.environment_base_image,
            require_digest=self.config.production_mode,
        )

        timeouts = _resolve_top_level_timeouts(submission, environment=environment, grant_limits=grant.limits)
        if isinstance(submission, BatchSubmission):
            return await self._create_batch(
                submission,
                environment=environment,
                grant_limits=grant.limits,
                effective_limits=effective_limits,
                max_artifact_size_bytes=max_artifact_size,
                source_package_limits=package_limits,
                timeouts=timeouts,
                actor_principal_id=actor_principal_id,
                owner_principal_id=owner_principal_id,
                owner_tenant=owner_tenant,
                submission_sha256=submission_sha256,
                idempotency_scope=idempotency_scope,
                artifact_repository=artifact_repository,
                dependency_build=dependency_build,
                workload_image=workload_image,
                provider_config_snapshot=provider_config_snapshot,
                driver_config=driver_config,
                now=now,
            )
        execution_id = new_id('exe')
        allocation_id = new_id('alc')
        predefined = environment.allocation_type == EnvironmentAllocationType.PREDEFINED
        pool_id = submission.pool or f'pool-{allocation_id}'
        resources = AllocationResources(slots=1) if predefined else _single_resources(submission)
        allocation_spec = _allocation_spec(
            execution_id=execution_id,
            environment=environment,
            resource_limits=effective_limits,
            source_package_limits=package_limits,
            allocation_id=allocation_id,
            pool_id=pool_id,
            workload='execution',
            resources=resources,
            code=submission.code.model_dump(mode='json'),
            host_env=[item.model_dump(mode='json') for item in submission.host_env],
            provisioning_timeout_seconds=timeouts['provisioning_timeout_seconds'],
            dependency_build=dependency_build,
            workload_image=workload_image,
            provider_config_snapshot=provider_config_snapshot,
        )
        if submission.resources is not None and submission.resources.managers.liveness is not None:
            allocation_spec['manager_liveness'] = submission.resources.managers.liveness.snapshot()
        if submission.request_policy is not None:
            allocation_spec['request_policy'] = submission.request_policy.model_dump(mode='json')
        allocation_spec_hash = canonical_sha256(allocation_spec)
        spec = _single_execution_spec(submission, timeouts=timeouts)
        mode, route_status = _execution_mode(submission)
        quota_repository = QuotaRepository(self.session, default_limits=self.config.default_user_quota)
        await quota_repository.admit_submission(owner_principal_id, now=now)
        execution = await ExecutionRepository(self.session, shard_count=self.config.shard_count).create_execution(
            CreateExecution(
                execution_id=execution_id,
                mode=mode,
                spec_type=f'nexus_{submit_kind}',
                instance_mode='asyncio',
                spec=spec,
                driver_config=driver_config,
                placement=_placement(
                    environment_id=environment.environment_id,
                    pool_id=pool_id,
                    allocation_id=allocation_id,
                    workload='execution',
                    scheduler_policy=submission.scheduler_policy.model_dump() if submission.scheduler_policy else None,
                    request_policy=submission.request_policy.model_dump() if submission.request_policy else None,
                ),
                route_status=route_status,
                slot_cost=1,
                max_attempts=_TOP_LEVEL_MAX_ATTEMPTS,
                priority=0,
                queue_ttl_seconds=timeouts['queue_ttl_seconds'],
                execution_ttl_seconds=timeouts['execution_ttl_seconds'],
                stop_timeout_seconds=timeouts['stop_timeout_seconds'],
                idempotency_scope=idempotency_scope,
                idempotency_key=submission.idempotency_key,
                owner_principal_id=owner_principal_id,
                owner_tenant=owner_tenant,
                created_by=actor_principal_id,
                environment_id=environment.environment_id,
                pool_id=pool_id,
                allocation_id=allocation_id,
                workload='execution',
                submission_sha256=submission_sha256,
                allocation_spec_hash=allocation_spec_hash,
                first_schedulable_at=now if predefined else None,
            ),
            now=now,
            reuse_existing=False,
        )
        credential_kinds = await _store_submission_credentials(
            self.session,
            execution_id=execution_id,
            submission=submission,
            timeouts=timeouts,
            now=now,
        )
        allocation = await AllocationRepository(self.session).create_allocation(
            CreateEnvironmentAllocation(
                allocation_id=allocation_id,
                top_level_execution_id=execution_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                pool_id=pool_id,
                workload='execution',
                desired_manager_count=resources.manager_count,
                resources=resources,
                allocation_spec=allocation_spec,
                allocation_spec_hash=allocation_spec_hash,
                dependency_build_id=(dependency_build.dependency_build_id if dependency_build else None),
            ),
            now=now,
        )
        if predefined:
            allocation = await _mark_predefined_ready(
                AllocationRepository(self.session),
                allocation.allocation_id,
                now=now,
            )
        await quota_repository.reserve_allocation(
            allocation.allocation_id,
            now=now,
        )
        for ref in artifact_refs(submission):
            await artifact_repository.add_ref(
                ref.artifact_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                resource_type='execution',
                resource_id=execution_id,
                now=now,
            )
            await artifact_repository.add_ref(
                ref.artifact_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                resource_type='allocation',
                resource_id=allocation_id,
                now=now,
            )
        phase = await ExecutionPhaseRepository(self.session).project(execution)
        return SubmissionResult(
            execution=execution,
            allocations=(allocation,),
            reused=False,
            phase=phase,
            credential_kinds=credential_kinds,
        )

    async def _create_batch(
        self,
        submission: BatchSubmission,
        *,
        environment: EnvironmentRecord,
        grant_limits: dict[str, Any],
        effective_limits: dict[str, int],
        max_artifact_size_bytes: int,
        source_package_limits: SourcePackageLimitsSpec,
        timeouts: dict[str, int | None],
        actor_principal_id: str,
        owner_principal_id: str,
        owner_tenant: str | None,
        submission_sha256: str,
        idempotency_scope: str,
        artifact_repository: ArtifactRepository,
        dependency_build: DependencyBuildRecord | None,
        workload_image: str,
        provider_config_snapshot: dict[str, Any] | None,
        driver_config: dict[str, Any],
        now: datetime,
    ) -> SubmissionResult:
        predefined = environment.allocation_type == EnvironmentAllocationType.PREDEFINED
        if predefined:
            max_inflight = _predefined_batch_max_inflight(
                submission,
                effective_limits=effective_limits,
            )
        else:
            if submission.resources is None or submission.batch.max_inflight is None:
                raise DomainValidationError('dynamic_environment_requires_resources')
            max_inflight = submission.batch.max_inflight
        batch_values = submission.batch.model_dump(mode='json')
        batch_values['max_inflight'] = max_inflight
        batch_values['submit_batch_size'] = min(self.config.batch_submit_batch_size, max_inflight)
        batch_values['child_queue_ttl_seconds'] = _resolve_timeout(
            'child_queue_ttl_seconds',
            submission.batch.child_queue_ttl_seconds,
            defaults=environment.default_timeouts,
            environment_limits=environment.resource_limits,
            grant_limits=grant_limits,
        )
        batch_values['timeout_per_run'] = _resolve_timeout(
            'timeout_per_run',
            submission.batch.timeout_per_run,
            defaults=environment.default_timeouts,
            environment_limits=environment.resource_limits,
            grant_limits=grant_limits,
        )
        generation = submission.generation.model_dump(mode='json')
        generation.pop('timeout_seconds', None)
        generation_timeout = _resolve_timeout(
            'generation_timeout_seconds',
            submission.generation.timeout_seconds,
            defaults=environment.default_timeouts,
            environment_limits=environment.resource_limits,
            grant_limits=grant_limits,
        )
        generation_limits = _generation_limits(
            effective_limits,
            timeout_seconds=generation_timeout,
            max_artifact_size_bytes=max_artifact_size_bytes,
            max_data_frame_bytes=self.config.batch_data_frame_max_size_bytes,
            max_task_record_bytes=self.config.batch_task_record_max_size_bytes,
            max_data_list_bytes=self.config.batch_data_list_max_size_bytes,
        )
        _validate_generation_admission(submission, generation_limits)
        task_list_manifest = _task_list_manifest(submission)
        if task_list_manifest is not None:
            generation['task_list_manifest'] = task_list_manifest

        execution_id = new_id('exe')
        parent_allocation_id = new_id('alc')
        child_allocation_id = new_id('alc')
        if predefined:
            parent_pool_id = 'orchestration'
            child_pool_id = submission.pool or 'execution'
            parent_resources = AllocationResources(slots=1, active_batches=1)
            child_resources = AllocationResources(slots=max_inflight)
        else:
            resources = submission.resources
            if resources is None:
                raise DomainValidationError('dynamic_environment_requires_resources')
            normalized_resources = resources.normalized()
            parent_pool_id = f'pool-{parent_allocation_id}'
            child_pool_id = f'pool-{child_allocation_id}'
            parent_resources = _allocation_resources(
                normalized_resources.batch_manager,
                active_batches=1,
            )
            child_resources = _allocation_resources(normalized_resources.managers)
        parent_spec = _allocation_spec(
            execution_id=execution_id,
            environment=environment,
            resource_limits=effective_limits,
            source_package_limits=source_package_limits,
            allocation_id=parent_allocation_id,
            pool_id=parent_pool_id,
            workload='orchestration',
            resources=parent_resources,
            code=submission.code.model_dump(mode='json'),
            host_env=[item.model_dump(mode='json') for item in submission.host_env],
            provisioning_timeout_seconds=timeouts['provisioning_timeout_seconds'],
            workspace_size_bytes=generation_limits.get('max_workspace_bytes'),
            dependency_build=dependency_build,
            workload_image=workload_image,
            provider_config_snapshot=provider_config_snapshot,
        )
        child_spec = _allocation_spec(
            execution_id=execution_id,
            environment=environment,
            resource_limits=effective_limits,
            source_package_limits=source_package_limits,
            allocation_id=child_allocation_id,
            pool_id=child_pool_id,
            workload='execution',
            resources=child_resources,
            code=submission.code.model_dump(mode='json'),
            host_env=[item.model_dump(mode='json') for item in submission.host_env],
            provisioning_timeout_seconds=timeouts['provisioning_timeout_seconds'],
            dependency_build=dependency_build,
            workload_image=workload_image,
            provider_config_snapshot=provider_config_snapshot,
        )
        parent_spec_hash = canonical_sha256(parent_spec)
        if submission.resources is not None and submission.resources.managers.liveness is not None:
            child_spec['manager_liveness'] = submission.resources.managers.liveness.snapshot()
        if submission.request_policy is not None:
            child_spec['request_policy'] = submission.request_policy.model_dump(mode='json')
        child_spec_hash = canonical_sha256(child_spec)
        batch_values.update({
            'child_environment_id': environment.environment_id,
            'child_pool_id': child_pool_id,
            'child_allocation_id': child_allocation_id,
            'child_allocation_spec_hash': child_spec_hash,
            'child_template': {
                'mode': ExecutionMode.JOB,
                'spec_type': 'nexus_single',
                'instance_mode': 'asyncio',
                'workload': 'execution',
                'environment_id': environment.environment_id,
                'pool_id': child_pool_id,
                'allocation_id': child_allocation_id,
                'allocation_spec_hash': child_spec_hash,
                'driver_config': driver_config,
                'slot_cost': 1,
                'max_attempts': _TOP_LEVEL_MAX_ATTEMPTS,
                'queue_ttl_seconds': batch_values['child_queue_ttl_seconds'],
                'execution_ttl_seconds': batch_values['timeout_per_run'],
                'stop_timeout_seconds': timeouts['stop_timeout_seconds'],
                **(
                    {'scheduler_policy': submission.scheduler_policy.model_dump()}
                    if submission.scheduler_policy
                    else {}
                ),
                **({'request_policy': submission.request_policy.model_dump()} if submission.request_policy else {}),
                'spec_template': {
                    'spec_version': 1,
                    'code': submission.code.model_dump(mode='json'),
                },
            },
        })
        quota_repository = QuotaRepository(self.session, default_limits=self.config.default_user_quota)
        await quota_repository.admit_submission(owner_principal_id, now=now)
        execution = await ExecutionRepository(self.session, shard_count=self.config.shard_count).create_execution(
            CreateExecution(
                execution_id=execution_id,
                mode=ExecutionMode.JOB,
                spec_type='nexus_batch',
                instance_mode='asyncio',
                spec={
                    'spec_version': 1,
                    'code': submission.code.model_dump(mode='json'),
                    'generation': generation,
                    'generation_limits': generation_limits,
                    'batch': batch_values,
                    'timeouts': timeouts,
                },
                driver_config=driver_config,
                placement=_placement(
                    environment_id=environment.environment_id,
                    pool_id=parent_pool_id,
                    allocation_id=parent_allocation_id,
                    workload='orchestration',
                ),
                route_status=RouteStatus.NONE,
                slot_cost=1,
                max_attempts=_TOP_LEVEL_MAX_ATTEMPTS,
                priority=0,
                queue_ttl_seconds=timeouts['queue_ttl_seconds'],
                execution_ttl_seconds=timeouts['execution_ttl_seconds'],
                stop_timeout_seconds=timeouts['stop_timeout_seconds'],
                idempotency_scope=idempotency_scope,
                idempotency_key=submission.idempotency_key,
                owner_principal_id=owner_principal_id,
                owner_tenant=owner_tenant,
                created_by=actor_principal_id,
                environment_id=environment.environment_id,
                pool_id=parent_pool_id,
                allocation_id=parent_allocation_id,
                workload='orchestration',
                submission_sha256=submission_sha256,
                allocation_spec_hash=parent_spec_hash,
                first_schedulable_at=now if predefined else None,
            ),
            now=now,
            reuse_existing=False,
        )
        credential_kinds = await _store_submission_credentials(
            self.session,
            execution_id=execution_id,
            submission=submission,
            timeouts=timeouts,
            now=now,
        )
        allocation_repository = AllocationRepository(self.session)
        parent_allocation = await allocation_repository.create_allocation(
            CreateEnvironmentAllocation(
                allocation_id=parent_allocation_id,
                top_level_execution_id=execution_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                pool_id=parent_pool_id,
                workload='orchestration',
                desired_manager_count=parent_resources.manager_count,
                resources=parent_resources,
                allocation_spec=parent_spec,
                allocation_spec_hash=parent_spec_hash,
                dependency_build_id=(dependency_build.dependency_build_id if dependency_build else None),
            ),
            now=now,
        )
        child_allocation = await allocation_repository.create_allocation(
            CreateEnvironmentAllocation(
                allocation_id=child_allocation_id,
                top_level_execution_id=execution_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                pool_id=child_pool_id,
                workload='execution',
                desired_manager_count=child_resources.manager_count,
                resources=child_resources,
                allocation_spec=child_spec,
                allocation_spec_hash=child_spec_hash,
                dependency_build_id=(dependency_build.dependency_build_id if dependency_build else None),
            ),
            now=now,
        )
        if predefined:
            parent_allocation = await _mark_predefined_ready(
                allocation_repository,
                parent_allocation.allocation_id,
                now=now,
            )
            child_allocation = await _mark_predefined_ready(
                allocation_repository,
                child_allocation.allocation_id,
                now=now,
            )
        await quota_repository.reserve_allocation(parent_allocation.allocation_id, now=now)
        await quota_repository.reserve_allocation(child_allocation.allocation_id, now=now)
        for ref in artifact_refs(submission):
            await artifact_repository.add_ref(
                ref.artifact_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment.environment_id,
                resource_type='execution',
                resource_id=execution_id,
                now=now,
            )
            for allocation_id in (parent_allocation_id, child_allocation_id):
                await artifact_repository.add_ref(
                    ref.artifact_id,
                    owner_principal_id=owner_principal_id,
                    environment_id=environment.environment_id,
                    resource_type='allocation',
                    resource_id=allocation_id,
                    now=now,
                )
        await self._ensure_batch_run(
            execution_id,
            shard_id=execution.shard_id,
            submission=submission,
            child_template=batch_values['child_template'],
            dependency_build=dependency_build,
            owner_principal_id=owner_principal_id,
            owner_tenant=owner_tenant,
            now=now,
        )
        phase = await ExecutionPhaseRepository(self.session).project(execution)
        return SubmissionResult(
            execution=execution,
            allocations=(parent_allocation, child_allocation),
            reused=False,
            phase=phase,
            batch_id=execution_id,
            credential_kinds=credential_kinds,
        )

    async def _ensure_dependency_build(
        self,
        submission: Submission,
        *,
        owner_principal_id: str,
        environment_id: str,
        provider_type: str,
        provider_config_ref: str | None,
        provider_config_snapshot: dict[str, Any] | None,
        artifact_repository: ArtifactRepository,
        source_package_limits: SourcePackageLimitsSpec,
        now: datetime,
    ) -> DependencyBuildRecord | None:
        if not isinstance(submission.code, SourcePackageCode):
            return None
        if provider_config_ref is None or provider_config_snapshot is None:
            raise DomainValidationError('source_package_requires_dynamic_provider_config')
        try:
            module_selectors = (
                list(selectors_from_run_input(submission.run_input))
                if isinstance(submission, (SingleSubmission, LiveSubmission))
                else []
            )
        except ModuleDependencyError as exc:
            raise DomainValidationError(exc.code, detail=exc.detail) from exc
        build, _ = await DependencyBuildRepository(self.session).get_or_create(
            WheelhouseBuildSpec(
                owner_principal_id=owner_principal_id,
                environment_id=environment_id,
                source_package=submission.code.source_package,
                nexus_wheel=submission.code.nexus_wheel,
                uv_lock=submission.code.uv_lock,
                source_package_limits=source_package_limits,
                extras=resolve_source_package_extras(submission.code.extras),
                module_selectors=module_selectors,
                workload_image=self.config.environment_base_image,
                builder_image=self.config.environment_builder_image,
            ),
            provider_type=provider_type,
            provider_config_ref=provider_config_ref,
            provider_config_snapshot=provider_config_snapshot,
            now=now,
        )
        for ref in (
            submission.code.source_package,
            submission.code.nexus_wheel,
            submission.code.uv_lock,
        ):
            await artifact_repository.add_ref(
                ref.artifact_id,
                owner_principal_id=owner_principal_id,
                environment_id=environment_id,
                resource_type='dependency_build',
                resource_id=build.dependency_build_id,
                now=now,
            )
        return build

    async def _ensure_batch_run(
        self,
        parent_execution_id: str,
        *,
        shard_id: int,
        submission: BatchSubmission,
        child_template: dict[str, Any],
        dependency_build: DependencyBuildRecord | None,
        owner_principal_id: str,
        owner_tenant: str | None,
        now: datetime,
    ) -> str:
        existing = await self.session.get(BatchRunRow, parent_execution_id)
        task_list_ref = None
        generation_status = 'pending'
        manifest = _task_list_manifest(submission)
        if isinstance(submission.generation, LocalTaskListGeneration):
            task_list_ref = submission.generation.task_list.ref.model_dump(mode='json')
            generation_status = 'materializing'
        if existing is not None:
            if (
                existing.owner_principal_id != owner_principal_id
                or existing.generation_mode != submission.generation.mode
                or existing.task_list_ref_json != task_list_ref
                or existing.shard_id != shard_id
            ):
                raise DomainConflictError('batch_submission_idempotency_conflict')
            return existing.generation_status

        values: dict[str, Any] = {
            'parent_execution_id': parent_execution_id,
            'shard_id': shard_id,
            'generation_mode': submission.generation.mode,
            'source_parent_execution_id': None,
            'source_summary_version': None,
            'generation_status': generation_status,
            'generation_epoch': 1,
            'materialized_cursor': None,
            'materialized_count': 0,
            'task_list_ref_json': task_list_ref,
            'dependency_status': 'pending' if isinstance(submission.code, SourcePackageCode) else 'not_required',
            'module_dependencies_json': None,
            'child_dependency_build_id': (
                dependency_build.dependency_build_id if dependency_build is not None else None
            ),
            'child_template_json': child_template,
            'owner_principal_id': owner_principal_id,
            'owner_tenant': owner_tenant,
            'total_items': manifest['item_count'] if manifest is not None else 0,
            'submitted_items': 0,
            'submission_closed': False,
            'active_parent_attempt_id': None,
            'summary_version': 0,
            'summary_json': None,
            'created_at': now,
            'updated_at': now,
        }
        optional_values = {
            'batch_id': parent_execution_id,
            'control_state': 'active',
            'stop_reason': None,
            'stop_requested_at': None,
            'stop_cursor': None,
            'task_list_manifest_json': manifest,
            'task_list_sha256': manifest['sha256'] if manifest is not None else None,
            'materialize_cursor_bytes': 0,
            'active_items': 0,
            'terminal_items': 0,
            'last_progress_at': now,
            'finalized_at': None,
        }
        columns = BatchRunRow.__table__.columns
        values.update({key: value for key, value in optional_values.items() if key in columns})
        self.session.add(BatchRunRow(**values))
        await self.session.flush()
        return generation_status

    async def _result(self, row: ExecutionRow, *, reused: bool, now: datetime) -> SubmissionResult:
        execution = await ExecutionRepository(self.session, shard_count=self.config.shard_count).get_execution(
            row.execution_id
        )
        allocation_rows = (
            await self.session.scalars(
                select(EnvironmentAllocationRow)
                .where(EnvironmentAllocationRow.top_level_execution_id == row.execution_id)
                .order_by(EnvironmentAllocationRow.workload, EnvironmentAllocationRow.allocation_id)
            )
        ).all()
        allocation_repository = AllocationRepository(self.session)
        allocation_records = []
        for allocation in allocation_rows:
            allocation_records.append(await allocation_repository.get_allocation(allocation.allocation_id))
        allocations = tuple(allocation_records)
        phase = await ExecutionPhaseRepository(self.session).project(execution)
        credential_kinds = await ExecutionCredentialRepository(self.session).list_kinds(
            row.execution_id,
            now=now,
        )
        return SubmissionResult(
            execution=execution,
            allocations=allocations,
            reused=reused,
            phase=phase,
            batch_id=row.execution_id if row.spec_type == 'nexus_batch' else None,
            credential_kinds=credential_kinds,
        )


def _submission_sha256(submission: Submission) -> str:
    public_payload = submission.model_dump(mode='json')
    resources = public_payload.get('resources')
    if isinstance(resources, dict):
        managers = resources.get('managers')
        if isinstance(managers, dict) and managers.get('liveness') is None:
            managers.pop('liveness', None)  # Preserve pre-feature idempotency hashes.
    if submission.run_trace_backends is None:
        public_payload.pop('run_trace_backends', None)
    if submission.data_credentials is None:
        return canonical_sha256(public_payload)
    credential_digest = canonical_sha256(submission.data_credentials.storage_payloads())
    return canonical_sha256({'submission': public_payload, 'data_credentials_sha256': credential_digest})


async def _store_submission_credentials(
    session: AsyncSession,
    *,
    execution_id: str,
    submission: Submission,
    timeouts: dict[str, int | None],
    now: datetime,
) -> tuple[str, ...]:
    if submission.data_credentials is None:
        return ()
    durations = [
        timeouts.get('provisioning_timeout_seconds'),
        timeouts.get('queue_ttl_seconds'),
        timeouts.get('execution_ttl_seconds'),
        timeouts.get('stop_timeout_seconds'),
    ]
    expires_at = (
        None
        if any(value is None for value in durations)
        else now + timedelta(seconds=sum(value for value in durations if value is not None))
    )
    return await ExecutionCredentialRepository(session).create_many(
        execution_id,
        submission.data_credentials.storage_payloads(),
        expires_at=expires_at,
        now=now,
    )


def _submission_kind(submission: Submission) -> str:
    if isinstance(submission, SingleSubmission):
        return 'single'
    if isinstance(submission, LiveSubmission):
        return 'live'
    return 'batch'


def _execution_mode(submission: SingleSubmission | LiveSubmission) -> tuple[str, str]:
    if isinstance(submission, LiveSubmission):
        return ExecutionMode.LIVE, RouteStatus.ACTIVE
    return ExecutionMode.JOB, RouteStatus.NONE


def _idempotency_scope(*, owner_principal_id: str, environment_id: str, submit_kind: str) -> str:
    digest = canonical_sha256({
        'owner_principal_id': owner_principal_id,
        'environment_id': environment_id,
        'submit_kind': submit_kind,
    })
    return f'submission:{submit_kind}:{digest}'


def _validate_environment_shape(submission: Submission, environment: EnvironmentRecord) -> None:
    if environment.allocation_type == EnvironmentAllocationType.DYNAMIC:
        if submission.pool is not None or submission.resources is None:
            raise DomainValidationError('dynamic_environment_requires_resources')
        return
    if submission.pool is None or submission.resources is not None or submission.code.mode != 'config_only':
        raise DomainValidationError('predefined_environment_requires_config_only_pool')


def _resolve_top_level_timeouts(
    submission: Submission,
    *,
    environment: EnvironmentRecord,
    grant_limits: dict[str, Any],
) -> dict[str, int | None]:
    return {
        field: _resolve_timeout(
            field,
            getattr(submission, field),
            defaults=environment.default_timeouts,
            environment_limits=environment.resource_limits,
            grant_limits=grant_limits,
        )
        for field in _TIMEOUT_FIELDS
    }


def _resolve_timeout(
    field: str,
    value: int | None,
    *,
    defaults: dict[str, Any],
    environment_limits: dict[str, Any],
    grant_limits: dict[str, Any],
) -> int | None:
    effective: Any = value
    if effective is None:
        effective = defaults.get(field)
    if effective is None:
        return None
    if (
        isinstance(effective, bool)
        or not isinstance(effective, (int, float))
        or effective <= 0
        or int(effective) != effective
    ):
        raise DomainValidationError(f'invalid_{field}')
    resolved = int(effective)
    maximum_key = f'max_{field}'
    maximums = [int(limits[maximum_key]) for limits in (environment_limits, grant_limits) if maximum_key in limits]
    if maximums and resolved > min(maximums):
        raise DomainValidationError(f'{field}_exceeds_limit')
    return resolved


def _predefined_batch_max_inflight(
    submission: BatchSubmission,
    *,
    effective_limits: dict[str, int],
) -> int:
    candidates: list[int] = []
    if 'batch_max_inflight' in effective_limits:
        candidates.append(effective_limits['batch_max_inflight'])
    if 'slots' in effective_limits:
        candidates.append(effective_limits['slots'] - 1)
    candidates = [value for value in candidates if value > 0]
    if not candidates:
        raise DomainValidationError('predefined_batch_max_inflight_not_configured')
    configured_max = min(candidates)
    requested = submission.batch.max_inflight
    if requested is not None and requested > configured_max:
        raise DomainValidationError('batch_max_inflight_exceeds_environment_limit')
    return requested or configured_max


def _generation_limits(
    environment_limits: dict[str, Any],
    *,
    timeout_seconds: int | None,
    max_artifact_size_bytes: int,
    max_data_frame_bytes: int,
    max_task_record_bytes: int,
    max_data_list_bytes: int,
) -> dict[str, int]:
    return {
        'max_items': _hard_limit(environment_limits, ('max_items',), 10_000),
        'max_task_list_bytes': _hard_limit(
            environment_limits,
            ('max_task_list_bytes',),
            _GIB,
        ),
        'max_line_bytes': _hard_limit(
            environment_limits,
            ('max_line_bytes',),
            16 * 1024**2,
        ),
        'timeout_seconds': timeout_seconds or 3600,
        'max_workspace_bytes': _hard_limit(
            environment_limits,
            ('max_workspace_bytes',),
            _DEFAULT_WORKSPACE_BYTES,
        ),
        'max_data_frame_bytes': _bounded_hard_limit(
            environment_limits,
            'max_data_frame_bytes',
            max_data_frame_bytes,
        ),
        'max_batch_task_record_bytes': _bounded_hard_limit(
            environment_limits,
            'max_batch_task_record_bytes',
            max_task_record_bytes,
        ),
        'max_data_list_bytes': min(
            _bounded_hard_limit(
                environment_limits,
                'max_data_list_bytes',
                max_data_list_bytes,
            ),
            max_artifact_size_bytes,
        ),
    }


def _bounded_hard_limit(limits: dict[str, Any], key: str, maximum: int) -> int:
    configured = _optional_hard_limit(limits, (key,))
    return maximum if configured is None else min(maximum, configured)


def _hard_limit(limits: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    value = _optional_hard_limit(limits, keys)
    return default if value is None else value


def _optional_hard_limit(limits: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        if key not in limits:
            continue
        value = limits[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 or int(value) != value:
            raise DomainValidationError(
                'invalid_environment_hard_limit',
                detail=key,
            )
        return int(value)
    return None


def _task_list_manifest(submission: BatchSubmission) -> dict[str, Any] | None:
    if not isinstance(submission.generation, LocalTaskListGeneration):
        return None
    return submission.generation.task_list.manifest.storage_payload()


def _validate_generation_admission(
    submission: BatchSubmission,
    limits: dict[str, int],
) -> None:
    generation = submission.generation
    if isinstance(generation, LocalTaskListGeneration):
        manifest = generation.task_list.manifest
        if manifest.item_count > limits['max_items']:
            raise DomainValidationError('batch_item_limit_exceeded')
        if manifest.size_bytes > limits['max_task_list_bytes']:
            raise DomainValidationError('batch_task_list_exceeds_byte_limit')
        data_list_size = 0
        if generation.data_list is not None:
            data_list_size = generation.data_list.manifest.size_bytes
            if data_list_size > limits['max_data_list_bytes']:
                raise DomainValidationError('batch_data_list_exceeds_byte_limit')
        if manifest.size_bytes + data_list_size > limits['max_workspace_bytes']:
            raise DomainValidationError('batch_workspace_limit_exceeded')
        return

    if not isinstance(generation, RemoteMaterializeGeneration):
        raise DomainValidationError('batch_generation_mode_invalid')
    source = generation.source
    if (
        source.type == 'compose'
        and source.compose is not None
        and source.compose.limit is not None
        and source.compose.limit > limits['max_items']
    ):
        raise DomainValidationError('batch_item_limit_exceeded')


async def _mark_predefined_ready(
    repository: AllocationRepository,
    allocation_id: str,
    *,
    now: datetime,
):
    claim = await repository.claim_allocation(
        allocation_id,
        operation_owner='submission',
        operation_ttl=timedelta(seconds=30),
        now=now,
    )
    transitioned = await repository.transition(
        claim,
        status='ready',
        ready_manager_count=0,
        now=now,
    )
    if not transitioned:
        raise RuntimeError('predefined_allocation_ready_transition_lost')
    return await repository.get_allocation(allocation_id)


def _submission_workload_image(submission: Submission, *, base_image: str, require_digest: bool) -> str:
    if not isinstance(submission.code, ImageCode):
        return base_image
    try:
        return resolve_workload_image(
            base_image,
            submission.code.image_version,
            require_digest=require_digest,
        )
    except ValueError as exc:
        raise DomainValidationError('workload_image_repository_unavailable') from exc


def _resolve_submission_cpu_memory_ratio(submission: Submission, *, auto_increase_cpu: bool) -> Submission:
    if submission.resources is None:
        return submission
    # Keep the original request hash stable when the server default changes on a retry.
    submission = submission.model_copy(deep=True)
    assert submission.resources is not None
    groups = [('managers', submission.resources.managers)]
    if isinstance(submission, BatchSubmission):
        assert submission.resources is not None
        groups.append(('batch_manager', submission.resources.batch_manager))
    for field, group in groups:
        try:
            normalized = normalize_resource_group(
                count=1,
                cpu_cores=group.cpu_cores,
                memory_gib=group.memory_gib,
                slots=1,
                field=field,
                auto_increase_cpu=(auto_increase_cpu if group.auto_increase_cpu is None else group.auto_increase_cpu),
            )
        except ValueError as exc:
            raise DomainValidationError('invalid_manager_resources', detail=str(exc)) from exc
        group.cpu_cores = normalized.cpu_millicores_per_manager / 1000
    return submission


def _single_resources(submission: SingleSubmission | LiveSubmission) -> AllocationResources:
    if submission.resources is None:
        raise DomainValidationError('dynamic_environment_requires_resources')
    return _allocation_resources(submission.resources.managers.normalized())


def _allocation_resources(
    normalized: NormalizedResourceGroup,
    *,
    active_batches: int = 0,
) -> AllocationResources:
    return AllocationResources(
        manager_count=normalized.manager_count,
        cpu_millicores=normalized.cpu_millicores,
        memory_bytes=normalized.memory_bytes,
        slots=normalized.slots,
        active_batches=active_batches,
    )


def _allocation_spec(
    *,
    execution_id: str,
    environment: EnvironmentRecord,
    resource_limits: dict[str, int],
    source_package_limits: SourcePackageLimitsSpec,
    allocation_id: str,
    pool_id: str,
    workload: str,
    resources: AllocationResources,
    code: dict[str, Any],
    host_env: list[dict[str, str]],
    provisioning_timeout_seconds: int | None,
    workload_image: str,
    workspace_size_bytes: int | None = None,
    dependency_build: DependencyBuildRecord | None = None,
    provider_config_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if workspace_size_bytes is None:
        workspace_size_bytes = _hard_limit(
            resource_limits,
            ('max_workspace_bytes',),
            _DEFAULT_WORKSPACE_BYTES,
        )
    spec = {
        'version': 1,
        'top_level_execution_id': execution_id,
        'environment_id': environment.environment_id,
        'config_version': environment.config_version,
        'allocation_type': environment.allocation_type,
        'provider_type': environment.provider_type,
        'provider_config_ref': environment.provider_config_ref,
        'allocation_id': allocation_id,
        'pool_id': pool_id,
        'workload': workload,
        'base_image': workload_image,
        'resources': {
            'manager_count': resources.manager_count,
            'cpu_millicores': resources.cpu_millicores,
            'memory_bytes': resources.memory_bytes,
            'slots': resources.slots,
            'active_batches': resources.active_batches,
        },
        'code': code,
        'source_package_limits': source_package_limits.model_dump(mode='json'),
    }
    if host_env:
        spec['host_env'] = [dict(item) for item in host_env]
    if provider_config_snapshot is not None:
        spec['provider_config'] = dict(provider_config_snapshot)
    if provisioning_timeout_seconds is not None:
        spec['provisioning_timeout_seconds'] = provisioning_timeout_seconds
    if dependency_build is not None:
        spec['dependency_build'] = {
            'dependency_build_id': dependency_build.dependency_build_id,
            'build_key': dependency_build.build_key,
        }
    spec['workspace_size_bytes'] = workspace_size_bytes
    return spec


def _placement(
    *,
    environment_id: str,
    pool_id: str,
    allocation_id: str,
    workload: str,
    scheduler_policy: dict | None = None,
    request_policy: dict | None = None,
) -> dict[str, Any]:
    return {
        'environment': environment_id,
        'pool': pool_id,
        'allocation_id': allocation_id,
        'labels': {'workload': workload},
        **({'scheduler_policy': scheduler_policy} if scheduler_policy else {}),
        **({'request_policy': request_policy} if request_policy else {}),
    }


def _single_execution_spec(
    submission: SingleSubmission | LiveSubmission,
    *,
    timeouts: dict[str, int | None],
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        'spec_version': 1,
        'code': submission.code.model_dump(mode='json'),
        'run_input': submission.run_input,
        'timeouts': timeouts,
    }
    if isinstance(submission, LiveSubmission):
        spec['live'] = submission.live
    return spec
