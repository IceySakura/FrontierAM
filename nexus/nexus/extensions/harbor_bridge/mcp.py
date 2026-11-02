"""Translate Harbor task MCP declarations into Nexus agent input.

Harbor owns the server declarations, while each Nexus agent owns the final
wire format used by its runtime. This module therefore emits the shared
``MCPServerSpec`` shape instead of Claude Code-specific configuration.
"""

from __future__ import annotations

import math
import re

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nexus.core.tool import MCPServerSpec


_TRANSPORT_ALIASES = {
    'streamable-http': 'streamable_http',
    'streamable_http': 'streamable_http',
    'http': 'streamable_http',
    'sse': 'sse',
    'stdio': 'stdio',
}
_ENV_REFERENCE_RE = re.compile(r'^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$')
_MCP_NAME_RE = re.compile(r'^[A-Za-z0-9_-]+$')


class T3MCPServerConfig(BaseModel):
    """Strict T3 extension of Harbor's MCP declaration."""

    model_config = ConfigDict(extra='forbid', strict=True)

    name: str
    transport: Literal['stdio', 'sse', 'streamable-http', 'streamable_http', 'http']
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None
    prefix: str | None = None
    runtime_name: Literal['main'] = 'main'
    nexus_managed: Literal[False] = False

    @field_validator('name')
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name or name.startswith('-') or not _MCP_NAME_RE.fullmatch(name):
            raise ValueError('name must contain only letters, digits, hyphens, and underscores')
        return name

    @field_validator('command')
    @classmethod
    def _normalize_command(cls, value: str | None) -> str | None:
        if value is None:
            return None
        command = value.strip()
        if not command:
            raise ValueError('command must be non-empty when provided')
        return command

    @field_validator('cwd')
    @classmethod
    def _validate_cwd(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cwd = value.strip()
        if not cwd or not PurePosixPath(cwd).is_absolute():
            raise ValueError('cwd must be an absolute container path')
        return cwd

    @field_validator('timeout_s')
    @classmethod
    def _validate_timeout(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError('timeout_s must be a positive finite number')
        return value

    @model_validator(mode='after')
    def _validate_transport_fields(self) -> T3MCPServerConfig:
        if self.transport == 'stdio':
            if not self.command:
                raise ValueError('stdio transport requires command')
        elif not self.url:
            raise ValueError(f'{self.transport} transport requires url')
        return self


def mcp_servers_from_task(config: dict[str, Any]) -> list[Any]:
    """Read raw T3 MCP declarations without losing Bridge extension fields."""
    environment = config.get('environment', {}) or {}
    if not isinstance(environment, dict):
        raise ValueError('[environment] must be a table')
    servers = environment.get('mcp_servers', [])
    if not isinstance(servers, list):
        raise ValueError('environment.mcp_servers must be an array of tables')
    return servers


def _normalize_secret_references(
    values: Any,
    *,
    location: str,
) -> dict[str, str]:
    """Normalize optional runtime references while preserving literal values."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f'{location} must be a table')

    normalized: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if not isinstance(raw_value, str):
            raise ValueError(f'{location}.{key} must be a string')
        value = raw_value.strip()
        if key.endswith('__env'):
            if not value:
                raise ValueError(f'{location}.{key} requires a non-empty environment variable name')
            normalized[key] = value
            continue

        reference = _ENV_REFERENCE_RE.fullmatch(value)
        if reference:
            normalized[f'{key}__env'] = reference.group(1)
            continue

        normalized[key] = raw_value
    return normalized


def _parse_servers(servers: list[Any]) -> list[T3MCPServerConfig]:
    parsed: list[T3MCPServerConfig] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(servers):
        location = f'environment.mcp_servers[{index}]'
        if isinstance(entry, T3MCPServerConfig):
            data = entry.model_dump()
        else:
            if not isinstance(entry, dict):
                raise ValueError(f'{location} must be a table')
            data = dict(entry)
        if 'transport' not in data:
            raise ValueError(f'{location}.transport must be explicitly set for T3-Harbor tasks')
        if data['transport'] not in _TRANSPORT_ALIASES:
            raise ValueError(
                f'{location}.transport must be one of {sorted(_TRANSPORT_ALIASES)}, got {data["transport"]!r}'
            )
        if 'env' in data:
            data['env'] = _normalize_secret_references(data['env'], location=f'{location}.env')
        if 'headers' in data:
            data['headers'] = _normalize_secret_references(
                data['headers'],
                location=f'{location}.headers',
            )
        try:
            server = T3MCPServerConfig.model_validate(data)
        except ValueError as exc:
            raise ValueError(f'Invalid {location}: {exc}') from exc
        if server.name in seen_names:
            raise ValueError(f'{location}.name duplicates MCP server {server.name!r}')
        seen_names.add(server.name)
        parsed.append(server)
    return parsed


def translate_mcp_to_nexus(servers: list[Any]) -> list[dict[str, Any]]:
    """Translate strict T3 MCP declarations to shared Nexus dictionaries."""
    normalized: list[dict[str, Any]] = []
    for index, server in enumerate(_parse_servers(servers)):
        location = f'environment.mcp_servers[{index}]'
        normalized_transport = _TRANSPORT_ALIASES.get(server.transport)
        if normalized_transport is None:
            raise ValueError(f'{location} ({server.name!r}) has unsupported transport {server.transport!r}')
        spec_data: dict[str, Any] = {
            'name': server.name,
            'prefix': server.prefix,
            'runtime_name': 'main',
            'nexus_managed': False,
            'transport': normalized_transport,
            'args': list(server.args),
            'env': dict(server.env),
            'cwd': server.cwd,
            'headers': dict(server.headers),
            'timeout_s': server.timeout_s,
        }
        if server.url is not None:
            spec_data['url'] = server.url
        if server.command is not None:
            spec_data['command'] = server.command
        try:
            spec = MCPServerSpec.model_validate(spec_data)
        except ValueError as exc:
            raise ValueError(f'Invalid {location} ({server.name!r}): {exc}') from exc
        normalized.append(spec.model_dump(mode='json', exclude_none=True))
    return normalized


def mcp_agent_kwargs(config: dict[str, Any], _harbor_servers: list[Any] | None = None) -> dict[str, Any]:
    """Build Agent kwargs from the raw T3 declaration.

    ``_harbor_servers`` remains optional for source compatibility with the
    short-lived canonical-Harbor call shape; raw TOML stays authoritative
    because Harbor 0.20 drops T3 extension fields.
    """
    servers = translate_mcp_to_nexus(mcp_servers_from_task(config))
    return {'mcp_servers': servers} if servers else {}


__all__ = [
    'T3MCPServerConfig',
    'mcp_agent_kwargs',
    'mcp_servers_from_task',
    'translate_mcp_to_nexus',
]
