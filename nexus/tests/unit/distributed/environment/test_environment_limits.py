from __future__ import annotations

import pytest

from pydantic import ValidationError

from nexus.distributed.environment.limits import (
    EnvironmentDefaultQuota,
    EnvironmentDefaultTimeouts,
    EnvironmentResourceLimits,
    SourcePackageLimitsSpec,
    effective_environment_resource_limits,
    normalize_environment_configuration,
    normalize_environment_default_quota,
)
from nexus.distributed.submission.source_package import PackageLimits


def test_environment_limit_models_reject_unknown_and_non_integer_values() -> None:
    with pytest.raises(ValidationError, match='unknown_limit'):
        EnvironmentResourceLimits.model_validate({'unknown_limit': 1})
    with pytest.raises(ValidationError, match='manager_count'):
        EnvironmentResourceLimits.model_validate({'manager_count': 1.5})
    with pytest.raises(ValidationError, match='queue_ttl_seconds'):
        EnvironmentDefaultTimeouts.model_validate({'queue_ttl_seconds': True})
    with pytest.raises(ValidationError, match='max_data_frame_bytes'):
        EnvironmentResourceLimits.model_validate({'max_data_frame_bytes': True})


def test_environment_accepts_strict_data_list_pipeline_limits() -> None:
    limits = EnvironmentResourceLimits.model_validate({
        'max_data_frame_bytes': 101,
        'max_batch_task_record_bytes': 102,
        'max_data_list_bytes': 103,
    })

    assert limits.model_dump(exclude_none=True) == {
        'max_data_frame_bytes': 101,
        'max_batch_task_record_bytes': 102,
        'max_data_list_bytes': 103,
    }


def test_source_package_limits_spec_round_trips_strict_large_limits() -> None:
    limits = PackageLimits(max_total_bytes=4 * 1024**3)
    spec = SourcePackageLimitsSpec.from_package_limits(limits)

    assert SourcePackageLimitsSpec.model_validate(spec.model_dump()).to_package_limits() == limits
    with pytest.raises(ValidationError, match='max_files'):
        SourcePackageLimitsSpec.model_validate({**spec.model_dump(), 'max_files': True})


def test_environment_configuration_normalizes_nulls_and_checks_timeout_maxima() -> None:
    resource_limits, default_timeouts = normalize_environment_configuration(
        {
            'slots': 10,
            'max_queue_ttl_seconds': 60,
            'max_execution_ttl_seconds': None,
        },
        {
            'queue_ttl_seconds': 30,
            'execution_ttl_seconds': None,
        },
    )

    assert resource_limits == {'slots': 10, 'max_queue_ttl_seconds': 60}
    assert default_timeouts == {'queue_ttl_seconds': 30}

    with pytest.raises(ValueError, match='queue_ttl_seconds_exceeds_limit'):
        normalize_environment_configuration(
            {'max_queue_ttl_seconds': 29},
            {'queue_ttl_seconds': 30},
        )


def test_effective_environment_resource_limits_take_the_stricter_value() -> None:
    assert effective_environment_resource_limits(
        {
            'slots': 20,
            'max_artifact_size_bytes': 1_000,
            'max_data_frame_bytes': 900,
            'max_batch_task_record_bytes': 850,
            'max_data_list_bytes': 800,
            'max_queue_ttl_seconds': 120,
        },
        {
            'slots': 8,
            'max_artifact_size_bytes': 800,
            'max_data_frame_bytes': 700,
            'max_batch_task_record_bytes': 750,
            'max_data_list_bytes': 900,
        },
    ) == {
        'slots': 8,
        'max_artifact_size_bytes': 800,
        'max_data_frame_bytes': 700,
        'max_batch_task_record_bytes': 750,
        'max_data_list_bytes': 800,
        'max_queue_ttl_seconds': 120,
    }
    assert effective_environment_resource_limits(
        {'slots': 100, 'batch_max_inflight': 40},
        {'slots': 30},
    ) == {'slots': 30, 'batch_max_inflight': 29}


def test_environment_default_quota_requires_complete_positive_compute_limits() -> None:
    with pytest.raises(ValidationError, match='active_batches'):
        EnvironmentDefaultQuota.model_validate({
            'manager_count': 2,
            'cpu_millicores': 10_000,
            'memory_bytes': 20 * 1024**3,
            'slots': 40,
        })

    with pytest.raises(ValidationError, match='manager_count'):
        EnvironmentDefaultQuota.model_validate({
            'manager_count': 0,
            'cpu_millicores': 10_000,
            'memory_bytes': 20 * 1024**3,
            'slots': 40,
            'active_batches': 1,
        })


def test_environment_default_quota_cannot_exceed_environment_capacity() -> None:
    resource_limits = {
        'manager_count': 5,
        'cpu_millicores': 20_000,
        'memory_bytes': 40 * 1024**3,
        'slots': 100,
        'active_batches': 2,
    }
    quota = {
        'manager_count': 2,
        'cpu_millicores': 10_000,
        'memory_bytes': 20 * 1024**3,
        'slots': 40,
        'active_batches': 1,
    }

    assert normalize_environment_default_quota(resource_limits, quota) == quota

    with pytest.raises(ValueError, match='default_quota_cpu_millicores_exceeds_environment_limit'):
        normalize_environment_default_quota(resource_limits, {**quota, 'cpu_millicores': 20_001})

    with pytest.raises(ValueError, match='default_quota_requires_environment_active_batches_limit'):
        normalize_environment_default_quota(
            {key: value for key, value in resource_limits.items() if key != 'active_batches'},
            quota,
        )
