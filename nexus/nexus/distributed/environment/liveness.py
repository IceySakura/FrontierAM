"""Immutable liveness overrides for dynamically allocated execution Managers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


MAX_MANAGER_LIVENESS_SECONDS = 600
LIVENESS_SPEC_KEY = 'manager_liveness'
LIVENESS_STARTUP_FLAG = '--require-allocation-liveness-v1'


class LivenessPlatformConfig(Protocol):
    manager_heartbeat_interval: float
    manager_healthy_ttl: timedelta
    manager_hard_exit_timeout: timedelta


class ManagerLivenessPolicy(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    self_fence_timeout_seconds: int = Field(gt=0, le=MAX_MANAGER_LIVENESS_SECONDS)
    executor_hang_timeout_seconds: int = Field(gt=0, le=MAX_MANAGER_LIVENESS_SECONDS)
    dead_timeout_seconds: int = Field(gt=0, le=MAX_MANAGER_LIVENESS_SECONDS)

    @model_validator(mode='after')
    def validate_order(self) -> ManagerLivenessPolicy:
        if self.self_fence_timeout_seconds >= self.dead_timeout_seconds:
            raise ValueError('manager_liveness_self_fence_must_precede_dead')
        if self.self_fence_timeout_seconds > self.executor_hang_timeout_seconds:
            raise ValueError('manager_liveness_self_fence_exceeds_executor_hang')
        return self

    def validate_platform(self, config: LivenessPlatformConfig) -> None:
        healthy = config.manager_healthy_ttl.total_seconds()
        if not config.manager_heartbeat_interval < healthy < self.self_fence_timeout_seconds:
            raise ValueError('manager_liveness_invalid_healthy_interval')
        if (
            self.self_fence_timeout_seconds + config.manager_hard_exit_timeout.total_seconds()
            >= self.dead_timeout_seconds
        ):
            raise ValueError('manager_liveness_insufficient_exit_grace')

    def snapshot(self) -> dict[str, Any]:
        return {'version': 1, **self.model_dump(mode='json')}

    def config_overrides(self) -> dict[str, timedelta]:
        return {
            'manager_self_fence_timeout': timedelta(seconds=self.self_fence_timeout_seconds),
            'manager_executor_hang_timeout': timedelta(seconds=self.executor_hang_timeout_seconds),
            'manager_dead_ttl': timedelta(seconds=self.dead_timeout_seconds),
        }


def allocation_liveness(spec: Mapping[str, Any]) -> ManagerLivenessPolicy | None:
    if not isinstance(spec, Mapping):
        raise ValueError('invalid_allocation_liveness_spec')
    if LIVENESS_SPEC_KEY not in spec:
        return None
    value = spec[LIVENESS_SPEC_KEY]
    if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1:
        raise ValueError('invalid_allocation_liveness_version')
    return ManagerLivenessPolicy.model_validate({key: item for key, item in value.items() if key != 'version'})


def startup_liveness(
    spec: Mapping[str, Any],
    *,
    required: bool,
    allocation_id: str | None,
    environment_id: str,
    pool_id: str,
    workload: str,
) -> ManagerLivenessPolicy | None:
    policy = allocation_liveness(spec)
    if required != (policy is not None):
        raise ValueError('allocation_liveness_flag_mismatch')
    if policy is not None and (
        not allocation_id
        or spec.get('allocation_id') != allocation_id
        or spec.get('environment_id') != environment_id
        or spec.get('pool_id') != pool_id
        or spec.get('workload') != workload
        or workload != 'execution'
        or spec.get('allocation_type') != 'dynamic'
    ):
        raise ValueError('allocation_liveness_identity_mismatch')
    return policy
