from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TypeAlias


DATABASE_INTEGER_MAX = 2**31 - 1
DATABASE_BIGINT_MAX = 2**63 - 1
CPU_MILLICORES_PER_CORE = 1000
MEMORY_BYTES_PER_GIB = 1024**3
MAX_MEMORY_GIB_PER_CPU_CORE = 4

ResourceNumber: TypeAlias = int | float | Decimal
_UNIT_ROUNDING_TOLERANCE = Decimal('1e-9')


class ResourceNormalizationError(ValueError):
    """A user resource request cannot be represented consistently."""


@dataclass(frozen=True, slots=True)
class NormalizedResourceGroup:
    manager_count: int
    cpu_millicores: int
    memory_bytes: int
    slots: int
    cpu_millicores_per_manager: int
    memory_bytes_per_manager: int
    slots_per_manager: int


@dataclass(frozen=True, slots=True)
class NormalizedBatchResources:
    batch_manager: NormalizedResourceGroup
    managers: NormalizedResourceGroup


def normalize_resource_group(
    *,
    count: int,
    cpu_cores: ResourceNumber,
    memory_gib: ResourceNumber,
    slots: int,
    field: str,
    auto_increase_cpu: bool | None = None,
) -> NormalizedResourceGroup:
    manager_count = _positive_integer(count, field=f'{field}.count')
    slots_per_manager = _positive_integer(slots, field=f'{field}.slots')
    cpu_per_manager = _positive_units(
        cpu_cores,
        scale=CPU_MILLICORES_PER_CORE,
        field=f'{field}.cpu_cores',
        unit='millicores',
    )
    memory_per_manager = _positive_units(
        memory_gib,
        scale=MEMORY_BYTES_PER_GIB,
        field=f'{field}.memory_gib',
        unit='bytes',
    )
    bytes_per_core = MAX_MEMORY_GIB_PER_CPU_CORE * MEMORY_BYTES_PER_GIB
    if (
        auto_increase_cpu is not None
        and memory_per_manager * CPU_MILLICORES_PER_CORE > cpu_per_manager * bytes_per_core
    ):
        if not auto_increase_cpu:
            raise ResourceNormalizationError(
                f'{field}.memory_gib must not exceed cpu_cores * {MAX_MEMORY_GIB_PER_CPU_CORE}; '
                'increase cpu_cores or enable auto_increase_cpu (--auto-increase-cpu)'
            )
        cpu_per_manager = (memory_per_manager + bytes_per_core - 1) // bytes_per_core * CPU_MILLICORES_PER_CORE
    return NormalizedResourceGroup(
        manager_count=manager_count,
        cpu_millicores=_checked_product(
            manager_count,
            cpu_per_manager,
            maximum=DATABASE_BIGINT_MAX,
            field=f'{field}.cpu_millicores_total',
            range_name='bigint',
        ),
        memory_bytes=_checked_product(
            manager_count,
            memory_per_manager,
            maximum=DATABASE_BIGINT_MAX,
            field=f'{field}.memory_bytes_total',
            range_name='bigint',
        ),
        slots=_checked_product(
            manager_count,
            slots_per_manager,
            maximum=DATABASE_INTEGER_MAX,
            field=f'{field}.slots_total',
            range_name='integer',
        ),
        cpu_millicores_per_manager=cpu_per_manager,
        memory_bytes_per_manager=memory_per_manager,
        slots_per_manager=slots_per_manager,
    )


def normalize_batch_resources(
    *,
    batch_manager_cpu_cores: ResourceNumber,
    batch_manager_memory_gib: ResourceNumber,
    manager_count: int,
    manager_cpu_cores: ResourceNumber,
    manager_memory_gib: ResourceNumber,
    manager_slots: int,
) -> NormalizedBatchResources:
    batch_manager = normalize_resource_group(
        count=1,
        cpu_cores=batch_manager_cpu_cores,
        memory_gib=batch_manager_memory_gib,
        slots=1,
        field='batch_manager',
    )
    managers = normalize_resource_group(
        count=manager_count,
        cpu_cores=manager_cpu_cores,
        memory_gib=manager_memory_gib,
        slots=manager_slots,
        field='managers',
    )
    _checked_sum(
        batch_manager.manager_count,
        managers.manager_count,
        maximum=DATABASE_INTEGER_MAX,
        field='batch.manager_count_total',
        range_name='integer',
    )
    _checked_sum(
        batch_manager.cpu_millicores,
        managers.cpu_millicores,
        maximum=DATABASE_BIGINT_MAX,
        field='batch.cpu_millicores_total',
        range_name='bigint',
    )
    _checked_sum(
        batch_manager.memory_bytes,
        managers.memory_bytes,
        maximum=DATABASE_BIGINT_MAX,
        field='batch.memory_bytes_total',
        range_name='bigint',
    )
    _checked_sum(
        batch_manager.slots,
        managers.slots,
        maximum=DATABASE_INTEGER_MAX,
        field='batch.slots_total',
        range_name='integer',
    )
    return NormalizedBatchResources(batch_manager=batch_manager, managers=managers)


def validate_resource_totals(
    *,
    manager_count: int,
    cpu_millicores: int,
    memory_bytes: int,
    slots: int,
    active_batches: int,
    field: str = 'resources',
) -> None:
    _non_negative_stored_integer(
        manager_count,
        maximum=DATABASE_INTEGER_MAX,
        field=f'{field}.manager_count',
        range_name='integer',
    )
    _non_negative_stored_integer(
        cpu_millicores,
        maximum=DATABASE_BIGINT_MAX,
        field=f'{field}.cpu_millicores',
        range_name='bigint',
    )
    _non_negative_stored_integer(
        memory_bytes,
        maximum=DATABASE_BIGINT_MAX,
        field=f'{field}.memory_bytes',
        range_name='bigint',
    )
    _non_negative_stored_integer(
        slots,
        maximum=DATABASE_INTEGER_MAX,
        field=f'{field}.slots',
        range_name='integer',
    )
    _non_negative_stored_integer(
        active_batches,
        maximum=DATABASE_INTEGER_MAX,
        field=f'{field}.active_batches',
        range_name='integer',
    )


def _positive_integer(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResourceNormalizationError(f'{field}_must_be_positive_integer')
    if value > DATABASE_INTEGER_MAX:
        raise ResourceNormalizationError(f'{field}_exceeds_integer_range')
    return value


def _non_negative_stored_integer(
    value: int,
    *,
    maximum: int,
    field: str,
    range_name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceNormalizationError(f'{field}_must_be_non_negative_integer')
    if value > maximum:
        raise ResourceNormalizationError(f'{field}_exceeds_{range_name}_range')


def _positive_units(
    value: ResourceNumber,
    *,
    scale: int,
    field: str,
    unit: str,
) -> int:
    if isinstance(value, bool):
        raise ResourceNormalizationError(f'{field}_must_be_finite')
    try:
        quantity = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ResourceNormalizationError(f'{field}_must_be_finite') from exc
    if not quantity.is_finite():
        raise ResourceNormalizationError(f'{field}_must_be_finite')
    if quantity <= 0:
        raise ResourceNormalizationError(f'{field}_must_be_positive')

    units = quantity * scale
    integral = units.to_integral_value()
    if abs(units - integral) > _UNIT_ROUNDING_TOLERANCE:
        raise ResourceNormalizationError(f'{field}_must_align_to_{unit}')
    normalized = int(integral)
    if normalized < 1:
        raise ResourceNormalizationError(f'{field}_must_be_at_least_one_{unit[:-1]}')
    if normalized > DATABASE_BIGINT_MAX:
        raise ResourceNormalizationError(f'{field}_exceeds_bigint_range')
    return normalized


def _checked_product(
    left: int,
    right: int,
    *,
    maximum: int,
    field: str,
    range_name: str,
) -> int:
    if left > maximum // right:
        raise ResourceNormalizationError(f'{field}_exceeds_{range_name}_range')
    return left * right


def _checked_sum(
    left: int,
    right: int,
    *,
    maximum: int,
    field: str,
    range_name: str,
) -> int:
    if left > maximum - right:
        raise ResourceNormalizationError(f'{field}_exceeds_{range_name}_range')
    return left + right


__all__ = [
    'CPU_MILLICORES_PER_CORE',
    'DATABASE_BIGINT_MAX',
    'DATABASE_INTEGER_MAX',
    'MEMORY_BYTES_PER_GIB',
    'NormalizedBatchResources',
    'NormalizedResourceGroup',
    'ResourceNormalizationError',
    'normalize_batch_resources',
    'normalize_resource_group',
    'validate_resource_totals',
]
