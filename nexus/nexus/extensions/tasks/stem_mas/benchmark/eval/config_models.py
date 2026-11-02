"""Pydantic config models used by stem_mas benchmark generation."""

from __future__ import annotations

import json
import math
import numbers
import os

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nexus.extensions.tasks.stem_mas.benchmark.eval.providers import selector_text


class CacheClientOverrideConfig(BaseModel):
    model_provider: str

    model_config = ConfigDict(extra='forbid')

    @field_validator('model_provider', mode='before')
    @classmethod
    def normalize_model_provider(cls, value: Any) -> str:
        model_provider = selector_text(value)
        if not model_provider:
            raise ValueError('model_provider is required')
        return model_provider


class EmptyMetadataConfig(BaseModel):
    """Metadata is intentionally not a model_provider source."""

    model_config = ConfigDict(extra='forbid')


class OpenAIClientConfig(BaseModel):
    client_args: dict[str, Any] = Field(default_factory=dict)
    request_args: dict[str, Any] = Field(default_factory=dict)
    metadata: EmptyMetadataConfig = Field(default_factory=EmptyMetadataConfig)

    model_config = ConfigDict(extra='allow')

    @field_validator('client_args', 'request_args', 'metadata', mode='before')
    @classmethod
    def _default_optional_sections(cls, value: Any) -> Any:
        return {} if value is None else value


class OpenAIClientSpecConfig(BaseModel):
    name: str = ''
    config: OpenAIClientConfig = Field(default_factory=OpenAIClientConfig)
    metadata: EmptyMetadataConfig = Field(default_factory=EmptyMetadataConfig)

    model_config = ConfigDict(extra='allow')

    @field_validator('config', 'metadata', mode='before')
    @classmethod
    def _default_optional_sections(cls, value: Any) -> Any:
        return {} if value is None else value


class CacheMetricsConfig(BaseModel):
    client_overrides: dict[str, CacheClientOverrideConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')


class ConfigValue:
    @staticmethod
    def coerce_dict(value: Any, source: str) -> dict:
        if ConfigValue.is_missing(value):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{source} must be a JSON object') from exc
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f'{source} must be a dict or JSON object string')

    @staticmethod
    def is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, numbers.Real) and math.isnan(value):
            return True
        return type(value).__name__ in {'NAType', 'NaTType'}

    @staticmethod
    def is_blank(value: Any) -> bool:
        if ConfigValue.is_missing(value):
            return True
        return value == ''

    @staticmethod
    def env_bool(name: str) -> bool | None:
        value = os.getenv(name)
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized in {'1', 'true', 'yes', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'off'}:
            return False
        raise ValueError(f'{name} must be a boolean value')

    @staticmethod
    def env_json_dict(name: str) -> dict | None:
        value = os.getenv(name)
        if value is None:
            return None
        return ConfigValue.coerce_dict(value, name)


__all__ = [
    'CacheMetricsConfig',
    'ConfigValue',
    'OpenAIClientSpecConfig',
]
