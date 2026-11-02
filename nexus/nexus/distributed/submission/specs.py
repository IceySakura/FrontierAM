from __future__ import annotations

import re

from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from nexus.dataset.uri import TAIJI_NEXUS_PREFIX, parse_dataset_uri
from nexus.distributed.artifacts.models import ArtifactRef
from nexus.distributed.batch.task_list import canonical_batch_metadata
from nexus.distributed.environment.liveness import ManagerLivenessPolicy
from nexus.distributed.policies.models import PolicySelection, RequestPolicySelection
from nexus.distributed.submission.input import normalize_live_config, normalize_run_input
from nexus.distributed.submission.resources import (
    DATABASE_BIGINT_MAX,
    DATABASE_INTEGER_MAX,
    NormalizedBatchResources,
    NormalizedResourceGroup,
    normalize_batch_resources,
    normalize_resource_group,
)
from nexus.distributed.submission.workload_image import validate_workload_image_version
from nexus.distributed.utils.identity import (
    ENVIRONMENT_ID_MAX_LENGTH,
    POOL_ID_MAX_LENGTH,
    SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH,
    validate_environment_id,
    validate_pool_id,
    validate_run_trace_backend_name,
    validate_submission_idempotency_key,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


DEFAULT_MANAGER_SLOTS = 20
HOST_ENV_MAX_ITEMS = 64
HOST_ENV_NAME_MAX_LENGTH = 128
HOST_ENV_VALUE_MAX_LENGTH = 4096
HOST_ENV_TOTAL_MAX_LENGTH = 64 * 1024
_HOST_ENV_RESERVED_NAMES = frozenset({'TMPDIR', 'TMP', 'TEMP'})


class GenerationLimits(StrictModel):
    max_items: int = Field(default=10000, ge=1, le=DATABASE_BIGINT_MAX)
    max_task_list_bytes: int = Field(default=1024 * 1024 * 1024, ge=1, le=DATABASE_BIGINT_MAX)
    timeout_seconds: int = Field(default=3600, ge=1, le=DATABASE_INTEGER_MAX)


class WedataDataCredential(StrictModel):
    subject: str = Field(min_length=1, max_length=256)
    key: SecretStr
    id: str = Field(min_length=1, max_length=256)

    def storage_payload(self) -> dict[str, str]:
        return {'subject': self.subject, 'key': self.key.get_secret_value(), 'id': self.id}


class CosDataCredential(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    secret_id: SecretStr
    secret_key: SecretStr
    token: SecretStr | None = None

    @model_validator(mode='after')
    def validate_complete(self) -> CosDataCredential:
        if not self.secret_id.get_secret_value().strip() or not self.secret_key.get_secret_value().strip():
            raise ValueError('COS requires non-empty secret_id and secret_key')
        return self

    def storage_payload(self) -> dict[str, str]:
        result = {'secret_id': self.secret_id.get_secret_value(), 'secret_key': self.secret_key.get_secret_value()}
        if self.token is not None:
            result['token'] = self.token.get_secret_value()
        return result


class DataCredentials(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    wedata: WedataDataCredential | None = None
    cos: CosDataCredential | None = None

    @model_validator(mode='after')
    def validate_non_empty(self) -> DataCredentials:
        if self.wedata is None and self.cos is None:
            raise ValueError('data_credentials requires at least one credential')
        return self

    def storage_payloads(self) -> dict[str, dict[str, str]]:
        result = {'wedata': self.wedata.storage_payload()} if self.wedata is not None else {}
        if self.cos is not None:
            result['cos'] = self.cos.storage_payload()
        return result

    def kinds(self) -> tuple[str, ...]:
        return tuple(self.storage_payloads())


class ConfigOnlyCode(StrictModel):
    mode: Literal['config_only'] = 'config_only'


class ImageCode(StrictModel):
    mode: Literal['image'] = 'image'
    image_version: str

    @field_validator('image_version')
    @classmethod
    def validate_image_version(cls, value: str) -> str:
        return validate_workload_image_version(value)


def resolve_source_package_extras(extras: list[str]) -> list[str]:
    return sorted({'distributed', 'runtime', *extras})


class SourcePackageCode(StrictModel):
    mode: Literal['source_package'] = 'source_package'
    source_package: ArtifactRef
    nexus_wheel: ArtifactRef
    uv_lock: ArtifactRef
    extras: list[str] = Field(default_factory=list)

    @field_validator('extras')
    @classmethod
    def validate_extras(cls, values: list[str]) -> list[str]:
        return _validate_extras(values)

    @model_validator(mode='after')
    def validate_artifact_kinds(self) -> SourcePackageCode:
        if self.source_package.kind != 'source_package':
            raise ValueError('code.source_package must have kind=source_package')
        if self.nexus_wheel.kind != 'nexus_wheel':
            raise ValueError('code.nexus_wheel must have kind=nexus_wheel')
        if self.uv_lock.kind != 'uv_lock':
            raise ValueError('code.uv_lock must have kind=uv_lock')
        return self


CodeSpec = Annotated[ConfigOnlyCode | ImageCode | SourcePackageCode, Field(discriminator='mode')]


class HostEnvVar(StrictModel):
    name: str = Field(min_length=1, max_length=HOST_ENV_NAME_MAX_LENGTH)
    value: str = Field(max_length=HOST_ENV_VALUE_MAX_LENGTH)

    @field_validator('name')
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value):
            raise ValueError('invalid_host_env_name')
        if value.startswith(('NEXUS_', 'POD_')) or value in _HOST_ENV_RESERVED_NAMES:
            raise ValueError('reserved_host_env_name')
        return value

    @field_validator('value')
    @classmethod
    def validate_value(cls, value: str) -> str:
        if '\x00' in value:
            raise ValueError('invalid_host_env_value')
        return value


def normalize_host_env(values: list[dict[str, Any]]) -> list[dict[str, str]]:
    parsed = [HostEnvVar.model_validate(value) for value in values]
    names = [item.name for item in parsed]
    if len(set(names)) != len(names):
        raise ValueError('duplicate_host_env_name')
    if sum(len(item.name) + len(item.value) for item in parsed) > HOST_ENV_TOTAL_MAX_LENGTH:
        raise ValueError('host_env_too_large')
    return [item.model_dump(mode='json') for item in parsed]


class _ManagerResource(StrictModel):
    cpu_cores: float = Field(gt=0, allow_inf_nan=False)
    memory_gib: float = Field(gt=0, allow_inf_nan=False)
    auto_increase_cpu: bool | None = None

    @model_serializer(mode='wrap')
    def serialize_resources(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        # Keep default request payloads stable, including on Pydantic 2.11.
        values = handler(self)
        if self.auto_increase_cpu is None:
            values.pop('auto_increase_cpu', None)
        return values

    @model_validator(mode='after')
    def normalize_cpu_memory_ratio(self) -> _ManagerResource:
        resources = normalize_resource_group(
            count=1,
            cpu_cores=self.cpu_cores,
            memory_gib=self.memory_gib,
            slots=1,
            field='manager',
            auto_increase_cpu=self.auto_increase_cpu,
        )
        if self.auto_increase_cpu is True:
            self.cpu_cores = resources.cpu_millicores_per_manager / 1000
        return self


class _ManagerLivenessResource(_ManagerResource):
    liveness: ManagerLivenessPolicy | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator('liveness', mode='before')
    @classmethod
    def reject_null_liveness(cls, value: Any) -> Any:
        if value is None:
            raise ValueError('manager_liveness_must_be_an_object')
        return value


class SingleManagerResources(_ManagerLivenessResource):
    count: Literal[1] = 1
    slots: int = Field(default=DEFAULT_MANAGER_SLOTS, ge=1, le=DATABASE_INTEGER_MAX)

    @model_validator(mode='after')
    def validate_normalized(self) -> SingleManagerResources:
        self.normalized()
        return self

    def normalized(self) -> NormalizedResourceGroup:
        return normalize_resource_group(
            count=self.count,
            cpu_cores=self.cpu_cores,
            memory_gib=self.memory_gib,
            slots=self.slots,
            field='managers',
        )


class SingleResources(StrictModel):
    managers: SingleManagerResources


class BatchManagerResources(_ManagerResource):
    @model_validator(mode='after')
    def validate_normalized(self) -> BatchManagerResources:
        normalize_resource_group(
            count=1,
            cpu_cores=self.cpu_cores,
            memory_gib=self.memory_gib,
            slots=1,
            field='batch_manager',
        )
        return self


class BatchExecutionManagerResources(_ManagerLivenessResource):
    count: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    slots: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)

    @model_validator(mode='after')
    def validate_normalized(self) -> BatchExecutionManagerResources:
        if self.count is None or self.slots is None:
            return self
        normalize_resource_group(
            count=self.count,
            cpu_cores=self.cpu_cores,
            memory_gib=self.memory_gib,
            slots=self.slots,
            field='managers',
        )
        return self


class BatchResources(StrictModel):
    batch_manager: BatchManagerResources
    managers: BatchExecutionManagerResources

    @model_validator(mode='after')
    def validate_normalized(self) -> BatchResources:
        if self.managers.count is not None and self.managers.slots is not None:
            self.normalized()
        return self

    def normalized(self) -> NormalizedBatchResources:
        if self.managers.count is None or self.managers.slots is None:
            raise ValueError('batch manager count and slots are unresolved')
        return normalize_batch_resources(
            batch_manager_cpu_cores=self.batch_manager.cpu_cores,
            batch_manager_memory_gib=self.batch_manager.memory_gib,
            manager_count=self.managers.count,
            manager_cpu_cores=self.managers.cpu_cores,
            manager_memory_gib=self.managers.memory_gib,
            manager_slots=self.managers.slots,
        )


RunTraceBackendName = Annotated[
    str,
    Field(min_length=1, max_length=64),
    AfterValidator(validate_run_trace_backend_name),
]


class RunTraceBackendSelection(StrictModel):
    sinks: list[RunTraceBackendName] = Field(min_length=1, max_length=8)

    @field_validator('sinks')
    @classmethod
    def validate_unique_sinks(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError('run trace backend sinks must not contain duplicates')
        return value


class SubmissionBase(StrictModel):
    model_config = ConfigDict(extra='forbid', strict=True, hide_input_in_errors=True)
    submission_version: Literal[1] = 1
    idempotency_key: str = Field(min_length=1, max_length=SUBMISSION_IDEMPOTENCY_KEY_MAX_LENGTH)
    environment_id: str = Field(min_length=1, max_length=ENVIRONMENT_ID_MAX_LENGTH)
    code: CodeSpec
    pool: str | None = Field(default=None, min_length=1, max_length=POOL_ID_MAX_LENGTH)
    host_env: list[HostEnvVar] = Field(
        default_factory=list,
        max_length=HOST_ENV_MAX_ITEMS,
        exclude_if=lambda value: not value,
    )
    provisioning_timeout_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    queue_ttl_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    execution_ttl_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    stop_timeout_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    scheduler_policy: PolicySelection | None = Field(default=None, exclude_if=lambda value: value is None)
    request_policy: RequestPolicySelection | None = Field(default=None, exclude_if=lambda value: value is None)
    run_trace_backends: RunTraceBackendSelection | None = None
    data_credentials: DataCredentials | None = Field(default=None, exclude=True, repr=False)

    @field_validator('idempotency_key')
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return validate_submission_idempotency_key(value)

    @field_validator('environment_id')
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return validate_environment_id(value)

    @field_validator('pool')
    @classmethod
    def validate_pool(cls, value: str | None) -> str | None:
        return validate_pool_id(value) if value is not None else None

    @field_validator('host_env')
    @classmethod
    def validate_host_env(cls, value: list[HostEnvVar]) -> list[HostEnvVar]:
        names = [item.name for item in value]
        if len(set(names)) != len(names):
            raise ValueError('duplicate_host_env_name')
        if sum(len(item.name) + len(item.value) for item in value) > HOST_ENV_TOTAL_MAX_LENGTH:
            raise ValueError('host_env_too_large')
        return value

    @model_validator(mode='after')
    def validate_host_env_allocation(self) -> SubmissionBase:
        if self.pool is not None and self.host_env:
            raise ValueError('host_env_requires_dynamic_allocation')
        return self


class SingleSubmission(SubmissionBase):
    resources: SingleResources | None = None
    run_input: dict[str, Any]

    @field_validator('run_input')
    @classmethod
    def validate_run_input(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_run_input(value)

    @model_validator(mode='after')
    def validate_allocation_shape(self) -> SingleSubmission:
        _validate_allocation_shape(code=self.code, pool=self.pool, resources=self.resources)
        return self


class LiveSubmission(SubmissionBase):
    resources: SingleResources | None = None
    run_input: dict[str, Any]
    live: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator('run_input')
    @classmethod
    def validate_run_input(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_run_input(value)

    @field_validator('live')
    @classmethod
    def validate_live(cls, value: dict[str, Any]) -> dict[str, Any]:
        return normalize_live_config(value)

    @model_validator(mode='after')
    def validate_allocation_shape(self) -> LiveSubmission:
        _validate_allocation_shape(code=self.code, pool=self.pool, resources=self.resources)
        return self


class ResolvedDatasetVersion(StrictModel):
    input_uri: str = Field(min_length=1, max_length=4096)
    resolved_uri: str = Field(min_length=1, max_length=4096)


class TaskListManifest(StrictModel):
    version: Literal[1] = 1
    record_version: Literal[1, 2, 3] = 1
    item_count: int = Field(ge=0, le=DATABASE_BIGINT_MAX)
    size_bytes: int = Field(ge=0, le=DATABASE_BIGINT_MAX)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source: dict[str, Any]
    datasets: list[ResolvedDatasetVersion] = Field(default_factory=list)
    batch_metadata: dict[str, Any] | None = None

    @field_validator('batch_metadata')
    @classmethod
    def validate_batch_metadata(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            canonical_batch_metadata(value)
        return value

    def storage_payload(self) -> dict[str, Any]:
        return self.model_dump(mode='json', exclude_none=True)


class TaskListSpec(StrictModel):
    ref: ArtifactRef
    format: Literal['jsonl'] = 'jsonl'
    manifest: TaskListManifest

    @model_validator(mode='after')
    def validate_kind(self) -> TaskListSpec:
        if self.ref.kind != 'task_list':
            raise ValueError('generation.task_list.ref must have kind=task_list')
        if self.ref.sha256 != self.manifest.sha256 or self.ref.size_bytes != self.manifest.size_bytes:
            raise ValueError('generation.task_list manifest does not match ArtifactRef')
        return self


class DataListManifest(StrictModel):
    version: Literal[1] = 1
    frame_count: int = Field(ge=1, le=DATABASE_BIGINT_MAX)
    size_bytes: int = Field(ge=1, le=DATABASE_BIGINT_MAX)
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class DataListSpec(StrictModel):
    ref: ArtifactRef
    format: Literal['nxdl'] = 'nxdl'
    manifest: DataListManifest

    @model_validator(mode='after')
    def validate_kind(self) -> DataListSpec:
        if self.ref.kind != 'data_list':
            raise ValueError('generation.data_list.ref must have kind=data_list')
        if self.ref.sha256 != self.manifest.sha256 or self.ref.size_bytes != self.manifest.size_bytes:
            raise ValueError('generation.data_list manifest does not match ArtifactRef')
        return self


class LocalTaskListGeneration(StrictModel):
    mode: Literal['local_task_list'] = 'local_task_list'
    task_list: TaskListSpec
    data_list: DataListSpec | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)

    @model_validator(mode='after')
    def validate_data_list_version(self) -> LocalTaskListGeneration:
        if (self.task_list.manifest.record_version == 3) != (self.data_list is not None):
            raise ValueError('task-list record_version=3 requires exactly one data_list')
        return self


class RemoteComposePart(StrictModel):
    kind: Literal['data', 'use', 'set', 'set_json', 'input']
    value: str | dict[str, Any]
    source: str = Field(min_length=1, max_length=4096)
    merge: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_cloud_dataset(self) -> RemoteComposePart:
        if self.kind == 'input':
            if not isinstance(self.value, dict):
                raise ValueError('Inline compose input must be an object')
            return self
        if not isinstance(self.value, str) or not 1 <= len(self.value) <= 4096:
            raise ValueError('Remote compose value must be a non-empty string of at most 4096 characters')
        if self.kind == 'data':
            validate_dataset_service_uri(self.value)
        elif self.kind == 'use' and self.value.startswith('data:'):
            validate_dataset_service_uri(self.value[len('data:') :])
        return self


class RemoteComposeSpec(StrictModel):
    parts: list[RemoteComposePart] = Field(min_length=1)
    split: str | None = None
    limit: int | None = Field(default=None, ge=1, le=DATABASE_BIGINT_MAX)
    page_size: int = Field(default=500, ge=1, le=10_000)


class RemoteGenerationSource(StrictModel):
    type: Literal['generator', 'compose']
    path: str | None = None
    compose: RemoteComposeSpec | None = None
    task_list_version: Literal[1, 2] = 2

    @model_validator(mode='after')
    def validate_source(self) -> RemoteGenerationSource:
        if self.type == 'generator':
            if not self.path or self.compose is not None:
                raise ValueError('generator source requires only path')
            _validate_safe_relative_path(self.path, field='generator path')
        elif self.compose is None or self.path is not None:
            raise ValueError('compose source requires only compose payload')
        return self


class RemoteMaterializeGeneration(StrictModel):
    mode: Literal['remote_materialize'] = 'remote_materialize'
    source: RemoteGenerationSource
    timeout_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)


GenerationSpec = Annotated[LocalTaskListGeneration | RemoteMaterializeGeneration, Field(discriminator='mode')]


class BatchPolicySpec(StrictModel):
    max_inflight: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    retry_failed: int = Field(default=0, ge=0, lt=DATABASE_INTEGER_MAX)
    fail_fast: bool = False
    child_queue_ttl_seconds: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)
    timeout_per_run: int | None = Field(default=None, ge=1, le=DATABASE_INTEGER_MAX)


class BatchSubmission(SubmissionBase):
    resources: BatchResources | None = None
    generation: GenerationSpec
    batch: BatchPolicySpec = Field(default_factory=BatchPolicySpec)

    @model_validator(mode='after')
    def validate_batch(self) -> BatchSubmission:
        _validate_allocation_shape(code=self.code, pool=self.pool, resources=self.resources)
        if (
            isinstance(self.generation, RemoteMaterializeGeneration)
            and self.generation.source.type == 'generator'
            and isinstance(self.code, ConfigOnlyCode)
        ):
            raise ValueError('remote generator requires source_package or image code mode')
        if self.resources is not None:
            managers = self.resources.managers
            max_inflight = self.batch.max_inflight
            if max_inflight is None:
                if managers.count is None or managers.slots is None:
                    raise ValueError('batch.max_inflight is required when manager count or slots is omitted')
            else:
                if managers.count is None and managers.slots is None:
                    managers.slots = DEFAULT_MANAGER_SLOTS
                if managers.count is None:
                    assert managers.slots is not None
                    managers.count = _ceil_div(max_inflight, managers.slots)
                elif managers.slots is None:
                    managers.slots = _ceil_div(max_inflight, managers.count)
            capacity = self.resources.normalized().managers.slots
            if max_inflight is None:
                self.batch.max_inflight = capacity
            elif max_inflight > capacity:
                raise ValueError('batch.max_inflight cannot exceed managers.count * managers.slots')
        return self


Submission = SingleSubmission | LiveSubmission | BatchSubmission


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def artifact_refs(submission: Submission) -> list[ArtifactRef]:
    refs: list[ArtifactRef] = []
    if isinstance(submission.code, SourcePackageCode):
        refs.extend((submission.code.source_package, submission.code.nexus_wheel, submission.code.uv_lock))
    if isinstance(submission, BatchSubmission) and isinstance(submission.generation, LocalTaskListGeneration):
        refs.append(submission.generation.task_list.ref)
        if submission.generation.data_list is not None:
            refs.append(submission.generation.data_list.ref)
    return refs


def validate_dataset_service_uri(value: str) -> str:
    raw = value.strip()
    if raw.startswith('cos://'):
        from nexus.dataset.cos import CosDatasetRef

        return CosDatasetRef.parse(raw).data_uri
    parsed = urlsplit(raw)
    if (
        not raw
        or parsed.scheme
        or raw.startswith(('/', './', '../', '~'))
        or '\\' in raw
        or any(part in {'.', '..'} for part in PurePosixPath(raw.split('?', 1)[0]).parts)
    ):
        raise ValueError('cloud dataset must be a Dataset Service URI')
    path = raw.split('?', 1)[0]
    if path.startswith(TAIJI_NEXUS_PREFIX):
        path = path[len(TAIJI_NEXUS_PREFIX) :]
    dataset_path = path.rsplit('@', 1)[0]
    if len(dataset_path.strip('/').split('/')) != 2:
        raise ValueError('cloud dataset must be a Dataset Service URI with namespace/dataset')
    try:
        parse_dataset_uri(raw)
    except ValueError as exc:
        raise ValueError('cloud dataset must be a Dataset Service URI') from exc
    return raw


def _validate_allocation_shape(*, code: CodeSpec, pool: str | None, resources: object | None) -> None:
    if pool is not None:
        if resources is not None:
            raise ValueError('predefined submissions cannot declare resources or mix pool with dynamic resources')
        if not isinstance(code, ConfigOnlyCode):
            raise ValueError('predefined submissions require code.mode=config_only')
        return
    if resources is None:
        raise ValueError('dynamic submissions require resources and cannot select a pool')


def _validate_extras(values: list[str]) -> list[str]:
    seen: set[str] = set()
    for value in values:
        if not re.fullmatch(r'[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*', value):
            raise ValueError(f'invalid wheel extra: {value}')
        if value in seen:
            raise ValueError(f'duplicate wheel extra: {value}')
        seen.add(value)
    return values


def _validate_safe_relative_path(value: str, *, field: str, allow_current: bool = False) -> None:
    if allow_current and value == '.':
        return
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '\\' in value or any(part in {'', '.', '..'} for part in path.parts):
        raise ValueError(f'{field} must be a safe relative path')
