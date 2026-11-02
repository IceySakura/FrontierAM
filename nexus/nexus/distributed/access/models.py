from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


class SubmitTokenKind:
    SYSTEM_DEFAULT = 'system_default'
    API = 'api'


@dataclass(frozen=True, slots=True)
class EnvironmentGrantRecord:
    principal_id: str
    environment_id: str
    limits: dict[str, Any]
    scopes: frozenset[str]
    enabled: bool
    version: int
    created_by: str
    updated_by: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class PrincipalAccessGrantRecord:
    owner_principal_id: str
    grantee_principal_id: str
    created_by: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TraceDatabaseAccountRecord:
    owner_principal_id: str
    backend_name: str
    username: str
    status: str
    credential_version: int
    last_error_code: str | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None


__all__ = [
    'EnvironmentGrantRecord',
    'PrincipalAccessGrantRecord',
    'SubmitTokenKind',
    'TraceDatabaseAccountRecord',
]
