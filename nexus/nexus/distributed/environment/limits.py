from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from nexus.distributed.submission.resources import DATABASE_BIGINT_MAX, DATABASE_INTEGER_MAX
from nexus.distributed.submission.source_package import PackageLimits


NonNegativeInteger = Annotated[StrictInt, Field(ge=0, le=DATABASE_INTEGER_MAX)]
NonNegativeBigInteger = Annotated[StrictInt, Field(ge=0, le=DATABASE_BIGINT_MAX)]
PositiveInteger = Annotated[StrictInt, Field(ge=1, le=DATABASE_INTEGER_MAX)]
PositiveBigInteger = Annotated[StrictInt, Field(ge=1, le=DATABASE_BIGINT_MAX)]


class SourcePackageLimitsSpec(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

    max_files: PositiveInteger = PackageLimits().max_files
    max_single_file_bytes: PositiveBigInteger = PackageLimits().max_single_file_bytes
    max_total_bytes: PositiveBigInteger = PackageLimits().max_total_bytes
    max_archive_bytes: PositiveBigInteger = PackageLimits().max_archive_bytes

    @classmethod
    def from_package_limits(cls, limits: PackageLimits) -> SourcePackageLimitsSpec:
        return cls.model_validate(asdict(limits))

    def to_package_limits(self) -> PackageLimits:
        return PackageLimits(**self.model_dump())


class EnvironmentResourceLimits(BaseModel):
    """Canonical deployment and grant limits, using integer base units."""

    model_config = ConfigDict(extra='forbid')

    manager_count: NonNegativeInteger | None = None
    cpu_millicores: NonNegativeBigInteger | None = None
    memory_bytes: NonNegativeBigInteger | None = None
    slots: NonNegativeInteger | None = None
    active_batches: NonNegativeInteger | None = None

    batch_max_inflight: PositiveInteger | None = None
    max_artifact_size_bytes: PositiveBigInteger | None = None
    max_source_files: PositiveInteger | None = None
    max_source_file_bytes: PositiveBigInteger | None = None
    max_source_unpacked_bytes: PositiveBigInteger | None = None
    max_workspace_bytes: PositiveBigInteger | None = None
    max_items: PositiveBigInteger | None = None
    max_task_list_bytes: PositiveBigInteger | None = None
    max_line_bytes: PositiveBigInteger | None = None
    max_data_frame_bytes: PositiveBigInteger | None = None
    max_batch_task_record_bytes: PositiveBigInteger | None = None
    max_data_list_bytes: PositiveBigInteger | None = None

    max_provisioning_timeout_seconds: PositiveInteger | None = None
    max_queue_ttl_seconds: PositiveInteger | None = None
    max_execution_ttl_seconds: PositiveInteger | None = None
    max_stop_timeout_seconds: PositiveInteger | None = None
    max_child_queue_ttl_seconds: PositiveInteger | None = None
    max_timeout_per_run: PositiveInteger | None = None
    max_generation_timeout_seconds: PositiveInteger | None = None

    @model_validator(mode='after')
    def validate_cross_limits(self) -> EnvironmentResourceLimits:
        if (
            self.slots is not None
            and self.slots > 0
            and self.batch_max_inflight is not None
            and self.batch_max_inflight >= self.slots
        ):
            raise ValueError('batch_max_inflight_must_leave_parent_slot')
        if (
            self.max_source_file_bytes is not None
            and self.max_source_unpacked_bytes is not None
            and self.max_source_file_bytes > self.max_source_unpacked_bytes
        ):
            raise ValueError('max_source_file_bytes_exceeds_unpacked_limit')
        return self


class EnvironmentDefaultQuota(BaseModel):
    """Complete per-user compute quota available through self service."""

    model_config = ConfigDict(extra='forbid')

    manager_count: PositiveInteger
    cpu_millicores: PositiveBigInteger
    memory_bytes: PositiveBigInteger
    slots: PositiveInteger
    active_batches: PositiveInteger


class EnvironmentDefaultTimeouts(BaseModel):
    """Canonical defaults applied when a submission omits a timeout."""

    model_config = ConfigDict(extra='forbid')

    provisioning_timeout_seconds: PositiveInteger | None = None
    queue_ttl_seconds: PositiveInteger | None = None
    execution_ttl_seconds: PositiveInteger | None = None
    stop_timeout_seconds: PositiveInteger | None = None
    child_queue_ttl_seconds: PositiveInteger | None = None
    timeout_per_run: PositiveInteger | None = None
    generation_timeout_seconds: PositiveInteger | None = None


_TIMEOUT_MAX_FIELDS = {
    'provisioning_timeout_seconds': 'max_provisioning_timeout_seconds',
    'queue_ttl_seconds': 'max_queue_ttl_seconds',
    'execution_ttl_seconds': 'max_execution_ttl_seconds',
    'stop_timeout_seconds': 'max_stop_timeout_seconds',
    'child_queue_ttl_seconds': 'max_child_queue_ttl_seconds',
    'timeout_per_run': 'max_timeout_per_run',
    'generation_timeout_seconds': 'max_generation_timeout_seconds',
}


def normalize_environment_resource_limits(resource_limits: Mapping[str, Any]) -> dict[str, int]:
    return EnvironmentResourceLimits.model_validate(dict(resource_limits)).model_dump(exclude_none=True)


def normalize_environment_default_timeouts(default_timeouts: Mapping[str, Any]) -> dict[str, int]:
    return EnvironmentDefaultTimeouts.model_validate(dict(default_timeouts)).model_dump(exclude_none=True)


def normalize_environment_default_quota(
    resource_limits: Mapping[str, Any],
    default_quota: Mapping[str, Any],
) -> dict[str, int]:
    limits = normalize_environment_resource_limits(resource_limits)
    quota = EnvironmentDefaultQuota.model_validate(dict(default_quota)).model_dump()
    for field, value in quota.items():
        maximum = limits.get(field)
        if maximum is None:
            raise ValueError(f'default_quota_requires_environment_{field}_limit')
        if value > maximum:
            raise ValueError(f'default_quota_{field}_exceeds_environment_limit')
    return quota


def normalize_environment_configuration(
    resource_limits: Mapping[str, Any],
    default_timeouts: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    normalized_limits = normalize_environment_resource_limits(resource_limits)
    normalized_timeouts = normalize_environment_default_timeouts(default_timeouts)
    validate_timeout_defaults(normalized_limits, normalized_timeouts)
    return normalized_limits, normalized_timeouts


def validate_timeout_defaults(
    resource_limits: Mapping[str, Any],
    default_timeouts: Mapping[str, Any],
) -> None:
    for timeout_field, maximum_field in _TIMEOUT_MAX_FIELDS.items():
        default = default_timeouts.get(timeout_field)
        maximum = resource_limits.get(maximum_field)
        if default is not None and maximum is not None and default > maximum:
            raise ValueError(f'{timeout_field}_exceeds_limit')


def effective_environment_resource_limits(
    environment_limits: Mapping[str, Any],
    grant_limits: Mapping[str, Any],
) -> dict[str, int]:
    environment = normalize_environment_resource_limits(environment_limits)
    grant = normalize_environment_resource_limits(grant_limits)
    effective = dict(environment)
    for key, value in grant.items():
        effective[key] = min(effective[key], value) if key in effective else value
    slots = effective.get('slots')
    if slots is not None and slots <= 1:
        effective.pop('batch_max_inflight', None)
    elif slots is not None and 'batch_max_inflight' in effective:
        effective['batch_max_inflight'] = min(effective['batch_max_inflight'], slots - 1)
    return normalize_environment_resource_limits(effective)


def validate_environment_hard_limits(resource_limits: Mapping[str, Any]) -> None:
    normalize_environment_resource_limits(resource_limits)


def effective_artifact_size_limit(
    resource_limits: Mapping[str, Any],
    *,
    global_max_size_bytes: int,
) -> int:
    configured = normalize_environment_resource_limits(resource_limits).get('max_artifact_size_bytes')
    if configured is None:
        return global_max_size_bytes
    return min(global_max_size_bytes, configured)


def source_package_limits(
    resource_limits: Mapping[str, Any],
    *,
    global_artifact_max_size_bytes: int,
) -> PackageLimits:
    defaults = PackageLimits()
    resource_limits = normalize_environment_resource_limits(resource_limits)
    return PackageLimits(
        max_files=_positive_int(resource_limits, 'max_source_files', defaults.max_files),
        max_single_file_bytes=_positive_int(
            resource_limits,
            'max_source_file_bytes',
            defaults.max_single_file_bytes,
        ),
        max_total_bytes=_positive_int(
            resource_limits,
            'max_source_unpacked_bytes',
            defaults.max_total_bytes,
        ),
        max_archive_bytes=effective_artifact_size_limit(
            resource_limits,
            global_max_size_bytes=global_artifact_max_size_bytes,
        ),
    )


def source_package_limits_from_allocation_spec(allocation_spec: Mapping[str, Any]) -> PackageLimits:
    raw = allocation_spec.get('source_package_limits')
    if raw is None:
        return PackageLimits()
    if not isinstance(raw, Mapping):
        raise ValueError('invalid_source_package_limits')
    try:
        return SourcePackageLimitsSpec.model_validate(dict(raw)).to_package_limits()
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid_source_package_limits') from exc


def _positive_int(resource_limits: Mapping[str, Any], key: str, default: int) -> int:
    value = resource_limits.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('invalid_environment_resource_limits')
    return value


__all__ = [
    'EnvironmentDefaultQuota',
    'EnvironmentDefaultTimeouts',
    'EnvironmentResourceLimits',
    'SourcePackageLimitsSpec',
    'effective_artifact_size_limit',
    'effective_environment_resource_limits',
    'normalize_environment_configuration',
    'normalize_environment_default_quota',
    'normalize_environment_default_timeouts',
    'normalize_environment_resource_limits',
    'source_package_limits',
    'source_package_limits_from_allocation_spec',
    'validate_environment_hard_limits',
    'validate_timeout_defaults',
]
