"""Container-registry credentials shared with runtime processes."""

from __future__ import annotations

import base64
import json
import os

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, SecretStr


REGISTRY_SHARED_VALUE_PREFIX = 'registry_credentials:'
DEFAULT_REGISTRY = 'mirrors.tencent.com'


class RegistryConfigurationError(RuntimeError):
    """Registry credentials are missing or malformed."""


class RegistryCredentials(BaseModel):
    """Registry credentials kept in memory and redacted from representations."""

    model_config = ConfigDict(frozen=True)

    registry: str
    username: SecretStr
    password: SecretStr

    def to_shared_payload(self) -> dict[str, str]:
        return {
            'registry': self.registry,
            'username': self.username.get_secret_value(),
            'password': self.password.get_secret_value(),
        }

    @classmethod
    def from_shared_payload(cls, payload: Mapping[str, Any]) -> RegistryCredentials:
        registry = payload.get('registry')
        username = payload.get('username')
        password = payload.get('password')
        if not isinstance(registry, str) or not registry.strip():
            raise RegistryConfigurationError('Registry credentials are missing registry')
        if not isinstance(username, str) or not username:
            raise RegistryConfigurationError('Registry credentials are missing username')
        if not isinstance(password, str) or not password:
            raise RegistryConfigurationError('Registry credentials are missing password')
        return cls(registry=registry.strip(), username=username, password=password)


def registry_shared_value_name(registry: str) -> str:
    registry = registry.strip().rstrip('/')
    if not registry or registry.startswith(REGISTRY_SHARED_VALUE_PREFIX):
        raise ValueError(f'Invalid registry name: {registry!r}')
    return f'{REGISTRY_SHARED_VALUE_PREFIX}{registry}'


def registry_from_shared_value_name(name: str) -> str | None:
    if not name.startswith(REGISTRY_SHARED_VALUE_PREFIX):
        return None
    registry = name.removeprefix(REGISTRY_SHARED_VALUE_PREFIX).strip()
    return registry or None


def load_registry_credentials(
    registry: str = DEFAULT_REGISTRY,
    *,
    username: str | None = None,
    password: str | None = None,
    docker_config: str | Path | None = None,
) -> RegistryCredentials:
    """Load registry credentials from explicit values, environment, or Docker config."""

    resolved_username = username if username is not None else os.getenv('REGISTRY_USERNAME')
    resolved_password = password if password is not None else os.getenv('REGISTRY_PASSWORD')
    if resolved_username is not None or resolved_password is not None:
        if not resolved_username or not resolved_password:
            raise RegistryConfigurationError('REGISTRY_USERNAME and REGISTRY_PASSWORD must be set together')
        return RegistryCredentials(
            registry=registry,
            username=SecretStr(resolved_username),
            password=SecretStr(resolved_password),
        )

    config_path = _docker_config_path(docker_config)
    try:
        payload = json.loads(config_path.read_text(encoding='utf-8'))
    except FileNotFoundError as exc:
        raise RegistryConfigurationError(
            f'Registry credentials for {registry!r} are missing: set REGISTRY_USERNAME and '
            f'REGISTRY_PASSWORD, or log in through Docker at {config_path}'
        ) from exc
    except OSError as exc:
        raise RegistryConfigurationError(f'Failed to read Docker config {config_path}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise RegistryConfigurationError(f'Invalid Docker config JSON in {config_path}') from exc

    auths = payload.get('auths') if isinstance(payload, dict) else None
    entry = _find_registry_auth(auths, registry)
    encoded = entry.get('auth') if entry is not None else None
    if not isinstance(encoded, str) or not encoded:
        raise RegistryConfigurationError(f'Docker config {config_path} has no inline auth for {registry!r}')
    try:
        decoded = base64.b64decode(encoded, validate=True).decode('utf-8')
    except (ValueError, UnicodeDecodeError) as exc:
        raise RegistryConfigurationError(f'Docker config auth for {registry!r} is invalid') from exc
    resolved_username, separator, resolved_password = decoded.partition(':')
    if not separator or not resolved_username or not resolved_password:
        raise RegistryConfigurationError(f'Docker config auth for {registry!r} is invalid')
    return RegistryCredentials(
        registry=registry,
        username=SecretStr(resolved_username),
        password=SecretStr(resolved_password),
    )


def _docker_config_path(value: str | Path | None) -> Path:
    if value is not None:
        path = Path(value).expanduser()
        return path / 'config.json' if path.is_dir() else path
    docker_config_dir = os.getenv('DOCKER_CONFIG')
    if docker_config_dir:
        return Path(docker_config_dir).expanduser() / 'config.json'
    return Path('~/.docker/config.json').expanduser()


def _find_registry_auth(auths: Any, registry: str) -> Mapping[str, Any] | None:
    if not isinstance(auths, Mapping):
        return None
    candidates = (registry, f'https://{registry}', f'http://{registry}')
    for candidate in candidates:
        entry = auths.get(candidate)
        if isinstance(entry, Mapping):
            return entry
    return None


__all__ = [
    'DEFAULT_REGISTRY',
    'REGISTRY_SHARED_VALUE_PREFIX',
    'RegistryConfigurationError',
    'RegistryCredentials',
    'load_registry_credentials',
    'registry_from_shared_value_name',
    'registry_shared_value_name',
]
