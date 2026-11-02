from __future__ import annotations

import math

from decimal import Decimal

import pytest

from pydantic import ValidationError

from nexus.distributed.environment.limits import normalize_environment_resource_limits
from nexus.distributed.repositories.allocation import AllocationResources
from nexus.distributed.repositories.quota import ComputeUsage, QuotaExceededError, _plus
from nexus.distributed.submission.resources import (
    DATABASE_BIGINT_MAX,
    DATABASE_INTEGER_MAX,
    MEMORY_BYTES_PER_GIB,
    ResourceNormalizationError,
    normalize_batch_resources,
    normalize_resource_group,
)
from nexus.distributed.submission.specs import (
    BatchExecutionManagerResources,
    BatchManagerResources,
    BatchResources,
    SingleManagerResources,
    SubmissionBase,
)


@pytest.mark.parametrize('auto_increase_cpu', [False, True])
@pytest.mark.parametrize(('cpu', 'memory'), [(1, 4), (4, 8), (0.5, 2)])
def test_cpu_memory_ratio_preserves_valid_requests(cpu, memory, auto_increase_cpu):
    resources = normalize_resource_group(
        count=3,
        cpu_cores=cpu,
        memory_gib=memory,
        slots=2,
        field='managers',
        auto_increase_cpu=auto_increase_cpu,
    )
    assert resources.cpu_millicores == cpu * 1000 * 3
    assert resources.memory_bytes == memory * MEMORY_BYTES_PER_GIB * 3


@pytest.mark.parametrize('model', [SingleManagerResources, BatchManagerResources, BatchExecutionManagerResources])
def test_cpu_memory_ratio_rejects_excess_memory_when_explicitly_disabled(model):
    with pytest.raises(ValidationError, match='enable auto_increase_cpu'):
        model(cpu_cores=2, memory_gib=16, auto_increase_cpu=False)


@pytest.mark.parametrize(('memory', 'expected_cpu'), [(16, 4), (17, 5), (4.125, 2)])
def test_auto_cpu_increase_rounds_up_and_preserves_memory(memory, expected_cpu):
    managers = SingleManagerResources(cpu_cores=1, memory_gib=memory, auto_increase_cpu=True)
    assert managers.cpu_cores == expected_cpu
    assert managers.memory_gib == memory
    assert managers.normalized().cpu_millicores == expected_cpu * 1000
    assert SingleManagerResources.model_validate(managers.model_dump()).model_dump() == managers.model_dump()


def test_batch_auto_cpu_increase_applies_before_aggregate_resources():
    resources = BatchResources.model_validate({
        'batch_manager': {'cpu_cores': 1, 'memory_gib': 8, 'auto_increase_cpu': True},
        'managers': {'cpu_cores': 2, 'memory_gib': 17, 'count': 3, 'slots': 20, 'auto_increase_cpu': True},
    })
    normalized = resources.normalized()
    assert normalized.batch_manager.cpu_millicores == 2000
    assert normalized.managers.cpu_millicores == 15000
    assert normalized.managers.memory_bytes == 3 * 17 * MEMORY_BYTES_PER_GIB
    assert normalized.managers.slots == 60


def test_default_auto_cpu_option_does_not_change_legacy_payload():
    resources = SingleManagerResources(cpu_cores=1, memory_gib=4)
    assert 'auto_increase_cpu' not in resources.model_dump()


def test_omitted_auto_cpu_option_defers_ratio_policy_to_server():
    resources = SingleManagerResources(cpu_cores=1, memory_gib=16)
    assert resources.auto_increase_cpu is None
    assert resources.cpu_cores == 1
    assert 'auto_increase_cpu' not in resources.model_dump()


def test_auto_cpu_increase_rounds_up_one_byte_above_boundary():
    resources = normalize_resource_group(
        count=1,
        cpu_cores=1,
        memory_gib=Decimal(4 * MEMORY_BYTES_PER_GIB + 1) / MEMORY_BYTES_PER_GIB,
        slots=1,
        field='managers',
        auto_increase_cpu=True,
    )
    assert resources.cpu_millicores == 2000


def test_auto_cpu_increase_checks_total_overflow_after_adjustment():
    with pytest.raises(ResourceNormalizationError, match='cpu_millicores_total_exceeds_bigint_range'):
        normalize_resource_group(
            count=DATABASE_INTEGER_MAX,
            cpu_cores=1,
            memory_gib=Decimal(DATABASE_BIGINT_MAX) / MEMORY_BYTES_PER_GIB,
            slots=1,
            field='managers',
            auto_increase_cpu=True,
        )


def test_resource_group_normalizes_minimum_provider_units():
    resources = normalize_resource_group(
        count=1,
        cpu_cores=0.001,
        memory_gib=1 / MEMORY_BYTES_PER_GIB,
        slots=1,
        field='managers',
    )

    assert resources.manager_count == 1
    assert resources.cpu_millicores == 1
    assert resources.memory_bytes == 1
    assert resources.slots == 1
    assert resources.cpu_millicores_per_manager == 1
    assert resources.memory_bytes_per_manager == 1


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        ('cpu_cores', math.inf),
        ('cpu_cores', math.nan),
        ('memory_gib', math.inf),
        ('memory_gib', math.nan),
    ],
)
def test_resource_group_rejects_non_finite_values(field_name: str, value: float):
    values = {'count': 1, 'cpu_cores': 1.0, 'memory_gib': 1.0, 'slots': 1}
    values[field_name] = value

    with pytest.raises(ResourceNormalizationError, match=f'managers.{field_name}_must_be_finite'):
        normalize_resource_group(**values, field='managers')


@pytest.mark.parametrize(
    ('field_name', 'value', 'message'),
    [
        ('cpu_cores', 0.0015, 'managers.cpu_cores_must_align_to_millicores'),
        (
            'memory_gib',
            1.5 / MEMORY_BYTES_PER_GIB,
            'managers.memory_gib_must_align_to_bytes',
        ),
    ],
)
def test_resource_group_rejects_sub_unit_values(field_name: str, value: float, message: str):
    values = {'count': 1, 'cpu_cores': 1.0, 'memory_gib': 1.0, 'slots': 1}
    values[field_name] = value

    with pytest.raises(ResourceNormalizationError, match=message):
        normalize_resource_group(**values, field='managers')


def test_resource_group_accepts_database_boundaries():
    resources = normalize_resource_group(
        count=1,
        cpu_cores=Decimal(DATABASE_BIGINT_MAX) / 1000,
        memory_gib=Decimal(DATABASE_BIGINT_MAX) / MEMORY_BYTES_PER_GIB,
        slots=DATABASE_INTEGER_MAX,
        field='managers',
    )

    assert resources.cpu_millicores == DATABASE_BIGINT_MAX
    assert resources.memory_bytes == DATABASE_BIGINT_MAX
    assert resources.slots == DATABASE_INTEGER_MAX


@pytest.mark.parametrize(
    ('values', 'message'),
    [
        (
            {'count': DATABASE_INTEGER_MAX + 1, 'cpu_cores': 1, 'memory_gib': 1, 'slots': 1},
            'managers.count_exceeds_integer_range',
        ),
        (
            {
                'count': 2,
                'cpu_cores': Decimal(DATABASE_BIGINT_MAX // 2 + 1) / 1000,
                'memory_gib': 1,
                'slots': 1,
            },
            'managers.cpu_millicores_total_exceeds_bigint_range',
        ),
        (
            {'count': 2, 'cpu_cores': 1, 'memory_gib': 1, 'slots': DATABASE_INTEGER_MAX},
            'managers.slots_total_exceeds_integer_range',
        ),
    ],
)
def test_resource_group_rejects_field_and_multiplication_overflow(
    values: dict[str, int | float | Decimal],
    message: str,
):
    with pytest.raises(ResourceNormalizationError, match=message):
        normalize_resource_group(**values, field='managers')


def test_batch_resources_reject_quota_aggregate_overflow():
    with pytest.raises(ResourceNormalizationError, match='batch.manager_count_total_exceeds_integer_range'):
        normalize_batch_resources(
            batch_manager_cpu_cores=1,
            batch_manager_memory_gib=1,
            manager_count=DATABASE_INTEGER_MAX,
            manager_cpu_cores=1,
            manager_memory_gib=1,
            manager_slots=1,
        )


@pytest.mark.parametrize(
    'payload',
    [
        {'count': 1, 'cpu_cores': math.inf, 'memory_gib': 1, 'slots': 1},
        {'count': 1, 'cpu_cores': 0.0015, 'memory_gib': 1, 'slots': 1},
        {'count': 1, 'cpu_cores': 1, 'memory_gib': 1.5 / MEMORY_BYTES_PER_GIB, 'slots': 1},
    ],
)
def test_single_resource_schema_rejects_unrepresentable_values(payload: dict[str, int | float]):
    with pytest.raises(ValidationError):
        SingleManagerResources.model_validate(payload)


def test_batch_resource_schema_rejects_total_overflow():
    with pytest.raises(ValidationError, match='managers.slots_total_exceeds_integer_range'):
        BatchExecutionManagerResources.model_validate({
            'count': 2,
            'cpu_cores': 1,
            'memory_gib': 1,
            'slots': DATABASE_INTEGER_MAX,
        })


def test_submission_timeout_schema_rejects_integer_overflow():
    with pytest.raises(ValidationError):
        SubmissionBase.model_validate({
            'idempotency_key': 'request-1',
            'environment_id': 'environment-1',
            'code': {'mode': 'config_only'},
            'provisioning_timeout_seconds': DATABASE_INTEGER_MAX + 1,
        })


@pytest.mark.parametrize(
    'resources',
    [
        {'manager_count': DATABASE_INTEGER_MAX + 1},
        {'cpu_millicores': DATABASE_BIGINT_MAX + 1},
        {'memory_bytes': DATABASE_BIGINT_MAX + 1},
        {'slots': DATABASE_INTEGER_MAX + 1},
        {'active_batches': DATABASE_INTEGER_MAX + 1},
    ],
)
def test_allocation_resources_reject_database_overflow(resources: dict[str, int]):
    with pytest.raises(ResourceNormalizationError):
        AllocationResources(**resources)


@pytest.mark.parametrize(
    'field',
    ['manager_count', 'cpu_millicores', 'memory_bytes', 'slots', 'active_batches'],
)
def test_quota_usage_rejects_database_overflow_before_write(field: str):
    maximum = DATABASE_BIGINT_MAX if field in {'cpu_millicores', 'memory_bytes'} else DATABASE_INTEGER_MAX
    left = ComputeUsage(
        manager_count=maximum if field == 'manager_count' else 0,
        cpu_millicores=maximum if field == 'cpu_millicores' else 0,
        memory_bytes=maximum if field == 'memory_bytes' else 0,
        slots=maximum if field == 'slots' else 0,
        active_batches=maximum if field == 'active_batches' else 0,
    )
    right = ComputeUsage(
        manager_count=1 if field == 'manager_count' else 0,
        cpu_millicores=1 if field == 'cpu_millicores' else 0,
        memory_bytes=1 if field == 'memory_bytes' else 0,
        slots=1 if field == 'slots' else 0,
        active_batches=1 if field == 'active_batches' else 0,
    )

    with pytest.raises(QuotaExceededError, match='compute_usage_storage_range_exceeded'):
        _plus(left, right)


@pytest.mark.parametrize(
    'limits',
    [
        {'manager_count': DATABASE_INTEGER_MAX + 1},
        {'cpu_millicores': DATABASE_BIGINT_MAX + 1},
        {'memory_bytes': DATABASE_BIGINT_MAX + 1},
        {'slots': DATABASE_INTEGER_MAX + 1},
        {'active_batches': DATABASE_INTEGER_MAX + 1},
        {'max_execution_ttl_seconds': DATABASE_INTEGER_MAX + 1},
    ],
)
def test_environment_limits_reject_unrepresentable_values(limits: dict[str, int]):
    with pytest.raises(ValidationError):
        normalize_environment_resource_limits(limits)


def test_submission_resource_schema_rejects_boolean_numbers():
    with pytest.raises(ValidationError):
        SingleManagerResources.model_validate({
            'count': 1,
            'cpu_cores': True,
            'memory_gib': 1,
            'slots': 1,
        })
