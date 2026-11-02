"""Typed runtime topology specifications and status snapshots."""

from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


class TopologyServiceSpec(BaseModel):
    """Runtime-relevant expectations for one service."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    image: str
    one_shot: bool = False
    has_healthcheck: bool = False
    depends_on: tuple[str, ...] = ()


class ComposeTopologySpec(BaseModel):
    """Provider-neutral input for a Compose-backed topology."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    definition: str
    definition_sha256: str
    primary_service: str
    startup_timeout_sec: int = Field(gt=0)
    startup_jitter_max: float = Field(default=0.0, ge=0)
    start_retry_attempts: int = Field(default=1, ge=1)
    start_retry_backoff_sec: float = Field(default=5.0, ge=0)
    start_retry_jitter_max: float | None = Field(default=None, ge=0)
    shutdown_timeout_sec: int = Field(gt=0)
    services: dict[str, TopologyServiceSpec]
    bind_paths: tuple[str, ...] = ()
    primary_environment: dict[str, str] = Field(default_factory=dict)
    agent_transport: Literal['none', 'socket-v1'] = 'none'

    @field_validator('definition')
    @classmethod
    def _definition_is_relative(cls, value: str) -> str:
        if not value or value.startswith('/') or '..' in value.split('/'):
            raise ValueError('definition must be a safe relative path')
        return value

    @field_validator('definition_sha256')
    @classmethod
    def _definition_digest_is_valid(cls, value: str) -> str:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
            raise ValueError('definition_sha256 must be a sha256 digest')
        return value

    @field_validator('bind_paths')
    @classmethod
    def _bind_paths_are_safe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for path in value:
            if not path or path.startswith('/') or '\\' in path or '..' in path.split('/'):
                raise ValueError(f'bind path must be task-relative and safe: {path!r}')
        if len(value) != len(set(value)):
            raise ValueError('bind_paths must not contain duplicates')
        return value

    @field_validator('primary_environment')
    @classmethod
    def _primary_environment_names_are_safe(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = sorted(name for name in value if not _ENV_NAME_RE.fullmatch(name))
        if invalid:
            raise ValueError(f'primary_environment contains invalid variable names: {invalid}')
        return value

    @model_validator(mode='after')
    def _primary_service_exists(self) -> ComposeTopologySpec:
        if self.primary_service not in self.services:
            raise ValueError('primary_service must be present in services')
        return self


class ServiceStatus(BaseModel):
    """Normalized status for one topology service."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    service: str
    container_id: str | None = None
    state: str = 'unknown'
    health: str | None = None
    exit_code: int | None = None


class TopologyStatus(BaseModel):
    """One immutable topology status snapshot."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    topology_id: str
    state: Literal['new', 'staged', 'started', 'ready', 'failed', 'deleted']
    services: dict[str, ServiceStatus] = Field(default_factory=dict)


__all__ = ['ComposeTopologySpec', 'ServiceStatus', 'TopologyServiceSpec', 'TopologyStatus']
