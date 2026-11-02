"""Parse the Nexus Compose extension and its locked definition."""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from pydantic import ValidationError

from nexus.extensions.harbor_bridge.compose.errors import ComposeConfigError, ComposeDefinitionError
from nexus.extensions.harbor_bridge.compose.models import ComposeTaskConfig, validate_service_name


@dataclass(frozen=True)
class LoadedComposeDefinition:
    """A verified task-local Compose lock and its parsed document."""

    relative_path: str
    path: Path
    sha256: str
    document: dict[str, Any]
    services: dict[str, dict[str, Any]]


def compose_config_from_task(config: dict[str, Any]) -> ComposeTaskConfig | None:
    """Read the strict ``[extensions.nexus.compose]`` section from raw TOML."""
    extensions = config.get('extensions')
    if extensions is None:
        return None
    if not isinstance(extensions, dict):
        raise ComposeConfigError('[extensions] must be a table')

    nexus = extensions.get('nexus')
    if nexus is None:
        return None
    if not isinstance(nexus, dict):
        raise ComposeConfigError('[extensions.nexus] must be a table')

    compose = nexus.get('compose')
    if compose is None:
        return None
    if not isinstance(compose, dict):
        raise ComposeConfigError('[extensions.nexus.compose] must be a table')
    try:
        return ComposeTaskConfig.model_validate(compose)
    except ValidationError as exc:
        raise ComposeConfigError(f'Invalid [extensions.nexus.compose]: {exc}') from exc


def load_compose_definition(task_dir: Path, definition: str) -> LoadedComposeDefinition:
    """Load one safe task-local Compose lock without resolving runtime behavior."""
    root = task_dir.resolve()
    relative = PurePosixPath(definition)
    if relative.is_absolute() or '..' in relative.parts or '\\' in definition:
        raise ComposeDefinitionError(f'Compose definition must be a safe task-relative path: {definition!r}')

    path = root.joinpath(*relative.parts)
    _reject_symlink_components(root, path)
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ComposeDefinitionError(f'Compose definition not found: {definition!r}') from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ComposeDefinitionError(f'Compose definition escapes task directory: {definition!r}') from exc
    if not resolved.is_file():
        raise ComposeDefinitionError(f'Compose definition is not a regular file: {definition!r}')

    payload = resolved.read_bytes()
    try:
        document = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ComposeDefinitionError(f'Compose definition is invalid YAML: {definition!r}: {exc}') from exc
    if not isinstance(document, dict):
        raise ComposeDefinitionError('Compose definition must contain a YAML object')

    raw_services = document.get('services')
    if not isinstance(raw_services, dict) or not raw_services:
        raise ComposeDefinitionError('Compose definition must contain a non-empty services mapping')

    services: dict[str, dict[str, Any]] = {}
    for raw_name, service in raw_services.items():
        if not isinstance(raw_name, str):
            raise ComposeDefinitionError(f'Compose service names must be strings, got {raw_name!r}')
        try:
            name = validate_service_name(raw_name)
        except ValueError as exc:
            raise ComposeDefinitionError(str(exc)) from exc
        if not isinstance(service, dict):
            raise ComposeDefinitionError(f'Compose service {name!r} must contain a mapping')
        services[name] = service

    return LoadedComposeDefinition(
        relative_path=relative.as_posix(),
        path=resolved,
        sha256=f'sha256:{hashlib.sha256(payload).hexdigest()}',
        document=document,
        services=services,
    )


def _reject_symlink_components(root: Path, path: Path) -> None:
    """Reject a lock reached through any symlink inside the task bundle."""
    current = root
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ComposeDefinitionError(f'Compose definition escapes task directory: {path}') from exc
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ComposeDefinitionError(f'Compose definition path must not contain symlinks: {current}')


__all__ = ['LoadedComposeDefinition', 'compose_config_from_task', 'load_compose_definition']
