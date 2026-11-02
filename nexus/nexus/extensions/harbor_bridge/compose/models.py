"""Typed task-facing configuration and frozen native Compose plan."""

from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_SERVICE_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def validate_service_name(value: str) -> str:
    """Return one normalized Compose service name."""
    name = value
    if name != name.strip():
        raise ValueError(f'Compose service name must not contain surrounding whitespace: {value!r}')
    if not name or not _SERVICE_NAME_RE.fullmatch(name):
        raise ValueError(f'invalid Compose service name: {value!r}')
    return name


class ComposeTaskConfig(BaseModel):
    """Strict ``[extensions.nexus.compose]`` contract."""

    model_config = ConfigDict(extra='forbid', strict=True)

    definition: str
    primary_service: str
    startup_timeout_sec: int = Field(default=180, gt=0)
    startup_jitter_max: float = Field(default=0.0, ge=0)
    start_retry_attempts: int = Field(default=1, ge=1)
    start_retry_backoff_sec: float = Field(default=5.0, ge=0)
    start_retry_jitter_max: float | None = Field(default=None, ge=0)
    shutdown_timeout_sec: int = Field(default=30, gt=0)
    one_shot_services: list[str] = Field(default_factory=list)

    @field_validator('definition')
    @classmethod
    def _definition_must_be_non_empty(cls, value: str) -> str:
        definition = value.strip()
        if not definition:
            raise ValueError('definition must be a non-empty task-relative path')
        return definition

    @field_validator('primary_service')
    @classmethod
    def _primary_service_name(cls, value: str) -> str:
        return validate_service_name(value)

    @field_validator('one_shot_services')
    @classmethod
    def _one_shot_service_names(cls, value: list[str]) -> list[str]:
        services = [validate_service_name(item) for item in value]
        if len(services) != len(set(services)):
            raise ValueError('one_shot_services must not contain duplicates')
        return services


class ComposeServiceSummary(BaseModel):
    """Runtime-relevant facts extracted from one locked Compose service."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    image: str
    depends_on: list[str] = Field(default_factory=list)
    has_healthcheck: bool = False
    one_shot: bool = False

    @field_validator('image')
    @classmethod
    def _image_is_valid(cls, value: str) -> str:
        if not value or any(character.isspace() for character in value):
            raise ValueError('image must be a non-empty reference without whitespace')
        return value


class ComposePlan(BaseModel):
    """Frozen, serializable plan consumed by the future Compose runtime."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    schema_version: Literal['nexus-compose-plan-v1'] = 'nexus-compose-plan-v1'
    definition: str
    definition_sha256: str
    controller_image: str
    primary_service: str
    primary_image: str
    workdir: str
    environment: dict[str, str] = Field(default_factory=dict)
    startup_timeout_sec: int
    startup_jitter_max: float = 0.0
    start_retry_attempts: int = 1
    start_retry_backoff_sec: float = 5.0
    start_retry_jitter_max: float | None = None
    shutdown_timeout_sec: int
    one_shot_services: list[str] = Field(default_factory=list)
    services: dict[str, ComposeServiceSummary]
    bind_paths: list[str] = Field(default_factory=list)

    @field_validator('definition')
    @classmethod
    def _definition_is_relative(cls, value: str) -> str:
        path = value.strip()
        if not path or path.startswith('/') or '..' in path.split('/'):
            raise ValueError('definition must be a safe relative path')
        return path

    @field_validator('definition_sha256')
    @classmethod
    def _definition_digest_is_valid(cls, value: str) -> str:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
            raise ValueError('definition_sha256 must be a sha256 digest')
        return value

    @field_validator('controller_image', 'primary_image')
    @classmethod
    def _images_are_valid(cls, value: str) -> str:
        if not value or any(character.isspace() for character in value):
            raise ValueError('image must be a non-empty reference without whitespace')
        return value

    @field_validator('workdir')
    @classmethod
    def _workdir_is_absolute(cls, value: str) -> str:
        if not value.startswith('/'):
            raise ValueError('workdir must be an absolute container path')
        return value

    @model_validator(mode='after')
    def _primary_service_is_present(self) -> ComposePlan:
        if self.primary_service not in self.services:
            raise ValueError('primary_service must be present in services')
        if self.primary_image != self.services[self.primary_service].image:
            raise ValueError('primary_image must match the primary service image')
        return self


__all__ = [
    'ComposePlan',
    'ComposeServiceSummary',
    'ComposeTaskConfig',
    'validate_service_name',
]
