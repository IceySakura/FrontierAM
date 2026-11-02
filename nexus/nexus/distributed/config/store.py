from __future__ import annotations

import asyncio
import copy
import logging
import re

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel, SecretStr, ValidationError
from sqlalchemy import and_, delete, or_, select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nexus.distributed.config import DistributedConfig, DistributedSecrets
from nexus.distributed.persistence.orm import DistributedConfigOverrideRow, DistributedConfigSecretOverrideRow
from nexus.distributed.repositories.audit import AuditRepository
from nexus.distributed.utils.clock import utcnow


GLOBAL_CONFIG_SCOPE = '*'
_BOOTSTRAP_ONLY_FIELDS = frozenset({'config_store_required', 'database_url', 'database_url_file'})
logger = logging.getLogger(__name__)


class ConfigVersionConflictError(RuntimeError):
    """The stored config scope changed after the caller read it."""


@dataclass(frozen=True, slots=True)
class ConfigOverrideRecord:
    role: str
    identity: str
    version: int
    fields: tuple[str, ...]
    values: dict[str, Any] = field(default_factory=dict, repr=False)


async def initialize_distributed_config(
    *,
    role: str,
    identity: str | None = None,
    environment_values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    secret_overrides: Mapping[str, Any] | None = None,
    require_stored_config: bool = False,
    minimum_schema_revision: str | None = None,
) -> DistributedConfig:
    """Build bootstrap config, wait for the schema, then load one scoped startup snapshot."""
    environment = dict(DistributedConfig.values_from_env() if environment_values is None else environment_values)
    bootstrap_values = {**environment, **(overrides or {})}
    if secret_overrides:
        bootstrap_values['secrets'] = dict(secret_overrides)
    bootstrap_config = DistributedConfig.model_validate(bootstrap_values)
    await _wait_for_database_schema(
        bootstrap_config,
        minimum_schema_revision=minimum_schema_revision,
    )

    from nexus.distributed.persistence.db import create_engine, create_session_factory

    engine = create_engine(bootstrap_config)
    try:
        return await load_distributed_config(
            create_session_factory(engine),
            database_url=bootstrap_config.read_database_url(),
            role=role,
            identity=identity,
            environment_values=environment,
            overrides=overrides,
            secret_overrides=secret_overrides,
            require_stored_config=require_stored_config,
        )
    finally:
        await engine.dispose()


async def load_distributed_config(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    database_url: str,
    role: str,
    identity: str | None = None,
    environment_values: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
    secret_overrides: Mapping[str, Any] | None = None,
    require_stored_config: bool = False,
) -> DistributedConfig:
    """Load one immutable startup config for a distributed service."""
    if not role:
        raise ValueError('distributed config role is required')

    scope_keys = [
        (GLOBAL_CONFIG_SCOPE, GLOBAL_CONFIG_SCOPE),
        (role, GLOBAL_CONFIG_SCOPE),
    ]
    if identity:
        scope_keys.append((role, identity))

    config_conditions = [
        and_(
            DistributedConfigOverrideRow.role == scope_role,
            DistributedConfigOverrideRow.identity == scope_identity,
        )
        for scope_role, scope_identity in scope_keys
    ]
    secret_conditions = [
        and_(
            DistributedConfigSecretOverrideRow.role == scope_role,
            DistributedConfigSecretOverrideRow.identity == scope_identity,
        )
        for scope_role, scope_identity in scope_keys
    ]
    async with session_factory() as session:
        config_rows = (
            await session.scalars(
                select(DistributedConfigOverrideRow).where(or_(*config_conditions)),
            )
        ).all()
        secret_rows = (
            await session.scalars(
                select(DistributedConfigSecretOverrideRow).where(or_(*secret_conditions)),
            )
        ).all()

    rows_by_scope = {(row.role, row.identity): row for row in config_rows}
    secret_rows_by_scope = {(row.role, row.identity): row for row in secret_rows}
    if require_stored_config and (
        (GLOBAL_CONFIG_SCOPE, GLOBAL_CONFIG_SCOPE) not in rows_by_scope
        or (GLOBAL_CONFIG_SCOPE, GLOBAL_CONFIG_SCOPE) not in secret_rows_by_scope
    ):
        raise ValueError('stored distributed config baseline is required at scope */*')
    merged = dict(environment_values or {})
    stored_field_scopes: dict[tuple[str, ...], tuple[str, str]] = {}
    for scope_key in scope_keys:
        row = rows_by_scope.get(scope_key)
        if row is None:
            continue
        values = row.config_json
        if not isinstance(values, dict):
            raise ValueError(f'distributed config scope {row.role}/{row.identity} must contain an object')
        _validate_db_field_boundaries(values, role=row.role, identity=row.identity)
        merged.update(values)
        for field_name in values:
            stored_field_scopes[(field_name,)] = (row.role, row.identity)

    merged_secrets: dict[str, Any] = {}
    for scope_key in scope_keys:
        row = secret_rows_by_scope.get(scope_key)
        if row is None:
            continue
        values = row.secrets_json
        if not isinstance(values, dict):
            raise ValueError(f'distributed secret scope {row.role}/{row.identity} must contain an object')
        _validate_db_secret_field_boundaries(values, role=row.role, identity=row.identity)
        merged_secrets.update(values)
        for field_name in values:
            stored_field_scopes[('secrets', field_name)] = (row.role, row.identity)
    if secret_overrides:
        _validate_secret_fields(secret_overrides, role='explicit', identity='startup')
        merged_secrets.update(secret_overrides)
        for field_name in secret_overrides:
            stored_field_scopes.pop(('secrets', field_name), None)

    merged.update(overrides or {})
    for field_name in overrides or ():
        stored_field_scopes.pop((field_name,), None)
    merged['database_url'] = database_url
    merged['config_store_required'] = require_stored_config
    if merged_secrets:
        merged['secrets'] = merged_secrets
    config = _validate_distributed_config_for_db_read(merged, stored_field_scopes=stored_field_scopes)
    if require_stored_config and config.auth_mode == 'disabled':
        raise ValueError('stored distributed config requires auth_mode=identity or auth_mode=required')
    return config


async def get_config_override(
    session: AsyncSession,
    *,
    role: str,
    identity: str,
    secret: bool = False,
    include_secret_values: bool = False,
) -> ConfigOverrideRecord | None:
    _validate_scope(role, identity)
    row_type = DistributedConfigSecretOverrideRow if secret else DistributedConfigOverrideRow
    row = await session.get(row_type, {'role': role, 'identity': identity})
    if row is None:
        return None
    return _record_from_row(row, include_secret_values=include_secret_values)


async def list_config_overrides(
    session: AsyncSession,
) -> list[ConfigOverrideRecord]:
    rows = (
        await session.scalars(
            select(DistributedConfigOverrideRow).order_by(
                DistributedConfigOverrideRow.role,
                DistributedConfigOverrideRow.identity,
            )
        )
    ).all()
    return [_record_from_row(row, include_secret_values=False) for row in rows]


async def list_secret_override_summaries(
    session: AsyncSession,
    *,
    role: str | None = None,
    identity: str | None = None,
) -> list[ConfigOverrideRecord]:
    if role is not None and not role:
        raise ValueError('distributed config role cannot be empty')
    if identity is not None and not identity:
        raise ValueError('distributed config identity cannot be empty')
    query = select(DistributedConfigSecretOverrideRow)
    if role is not None:
        query = query.where(DistributedConfigSecretOverrideRow.role == role)
    if identity is not None:
        query = query.where(DistributedConfigSecretOverrideRow.identity == identity)
    rows = (
        await session.scalars(
            query.order_by(DistributedConfigSecretOverrideRow.role, DistributedConfigSecretOverrideRow.identity)
        )
    ).all()
    return [_record_from_row(row, include_secret_values=False) for row in rows]


async def set_config_override(
    session: AsyncSession,
    *,
    role: str,
    identity: str,
    values: Mapping[str, Any],
    expected_version: int,
    actor: str,
    secret: bool = False,
) -> ConfigOverrideRecord:
    _validate_scope(role, identity)
    _validate_expected_version(expected_version)
    if not actor:
        raise ValueError('config change actor is required')
    payload = _normalize_override_values(values, secret=secret, role=role, identity=identity)
    now = utcnow()
    row_type = DistributedConfigSecretOverrideRow if secret else DistributedConfigOverrideRow
    payload_field = 'secrets_json' if secret else 'config_json'

    if expected_version == 0:
        row = row_type(
            role=role,
            identity=identity,
            **{payload_field: payload},
            version=1,
            created_at=now,
            updated_at=now,
        )
        try:
            async with session.begin_nested():
                session.add(row)
                await session.flush()
        except IntegrityError as exc:
            found = await session.scalar(
                select(row_type.version).where(row_type.role == role, row_type.identity == identity)
            )
            raise ConfigVersionConflictError(_version_conflict_detail(expected_version, found)) from exc
        version = 1
    else:
        result = cast(
            CursorResult[Any],
            await session.execute(
                update(row_type)
                .where(row_type.role == role, row_type.identity == identity, row_type.version == expected_version)
                .values(**{payload_field: payload}, version=expected_version + 1, updated_at=now)
            ),
        )
        if result.rowcount != 1:
            found = await session.scalar(
                select(row_type.version).where(row_type.role == role, row_type.identity == identity)
            )
            raise ConfigVersionConflictError(_version_conflict_detail(expected_version, found))
        version = expected_version + 1

    fields = tuple(sorted(payload))
    await AuditRepository(session).append(
        actor_principal_id=actor,
        operation='distributed_config.secret_set' if secret else 'distributed_config.set',
        resource_type='distributed_config_secret' if secret else 'distributed_config',
        resource_id=f'{role}/{identity}',
        now=now,
        metadata={'fields': list(fields), 'version': version},
    )
    return ConfigOverrideRecord(
        role=role,
        identity=identity,
        version=version,
        fields=fields,
        values={} if secret else dict(payload),
    )


async def delete_config_override(
    session: AsyncSession,
    *,
    role: str,
    identity: str,
    expected_version: int,
    actor: str,
    secret: bool = False,
) -> bool:
    _validate_scope(role, identity)
    _validate_expected_version(expected_version)
    if not actor:
        raise ValueError('config change actor is required')
    row_type = DistributedConfigSecretOverrideRow if secret else DistributedConfigOverrideRow
    payload_field = (
        DistributedConfigSecretOverrideRow.secrets_json if secret else DistributedConfigOverrideRow.config_json
    )
    existing = await session.execute(
        select(row_type.version, payload_field).where(row_type.role == role, row_type.identity == identity)
    )
    current = existing.one_or_none()
    if current is None:
        if expected_version == 0:
            return False
        raise ConfigVersionConflictError(_version_conflict_detail(expected_version, None))
    found_version, payload = current
    if found_version != expected_version:
        raise ConfigVersionConflictError(_version_conflict_detail(expected_version, found_version))
    result = cast(
        CursorResult[Any],
        await session.execute(
            delete(row_type).where(
                row_type.role == role,
                row_type.identity == identity,
                row_type.version == expected_version,
            )
        ),
    )
    if result.rowcount != 1:
        found = await session.scalar(
            select(row_type.version).where(row_type.role == role, row_type.identity == identity)
        )
        raise ConfigVersionConflictError(_version_conflict_detail(expected_version, found))
    fields = sorted(payload) if isinstance(payload, dict) else []
    await AuditRepository(session).append(
        actor_principal_id=actor,
        operation='distributed_config.secret_delete' if secret else 'distributed_config.delete',
        resource_type='distributed_config_secret' if secret else 'distributed_config',
        resource_id=f'{role}/{identity}',
        now=utcnow(),
        metadata={'fields': fields, 'version': expected_version},
    )
    return True


def schema_revision_is_compatible(
    actual: object,
    *,
    expected: str,
    allow_newer: bool,
) -> bool:
    if actual == expected:
        return True
    if not allow_newer or not isinstance(actual, str):
        return False
    actual_match = re.fullmatch(r'(\d+)(?:_.+)?', actual)
    expected_match = re.fullmatch(r'(\d+)(?:_.+)?', expected)
    if actual_match is None or expected_match is None:
        return False
    return int(actual_match.group(1)) >= int(expected_match.group(1))


async def _wait_for_database_schema(
    config: DistributedConfig,
    *,
    minimum_schema_revision: str | None = None,
) -> None:
    from nexus.distributed.persistence.db import create_engine
    from nexus.distributed.persistence.migrations import HEAD_REVISION

    expected_revision = minimum_schema_revision or HEAD_REVISION
    allow_newer = minimum_schema_revision is not None
    deadline = asyncio.get_running_loop().time() + config.db_wait_timeout
    last_error: Exception | None = None
    while True:
        engine = create_engine(config)
        try:
            async with engine.connect() as connection:
                revision = await connection.scalar(text('SELECT version_num FROM alembic_version'))
            if schema_revision_is_compatible(
                revision,
                expected=expected_revision,
                allow_newer=allow_newer,
            ):
                return
            expectation = f'revision {expected_revision} or newer' if allow_newer else f'revision {expected_revision}'
            last_error = RuntimeError(f'expected {expectation}, got {revision}')
        except Exception as exc:
            last_error = exc
        finally:
            await engine.dispose()
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError(f'distributed database schema is not ready: {last_error}') from last_error
        await asyncio.sleep(1)


def _validate_db_field_boundaries(values: Mapping[str, Any], *, role: str, identity: str) -> None:
    secret_fields = sorted((set(DistributedSecrets.model_fields) | {'secrets'}).intersection(values))
    if secret_fields:
        raise ValueError(
            f'secret field is not allowed in distributed config scope {role}/{identity}: {", ".join(secret_fields)}'
        )

    protected = sorted(_BOOTSTRAP_ONLY_FIELDS.intersection(values))
    if protected:
        raise ValueError(f'{", ".join(protected)} is bootstrap-only in distributed config scope {role}/{identity}')


def _validate_db_fields(values: Mapping[str, Any], *, role: str, identity: str) -> None:
    _validate_db_field_boundaries(values, role=role, identity=identity)
    unknown = sorted(set(values).difference(DistributedConfig.model_fields))
    if unknown:
        raise ValueError(f'unknown distributed config field in scope {role}/{identity}: {", ".join(unknown)}')


def _validate_db_secret_field_boundaries(values: Mapping[str, Any], *, role: str, identity: str) -> None:
    ordinary_fields = sorted(set(values).intersection(DistributedConfig.model_fields))
    if ordinary_fields:
        raise ValueError(
            f'ordinary field is not allowed in distributed secret scope {role}/{identity}: {", ".join(ordinary_fields)}'
        )


def _validate_distributed_config_for_db_read(
    values: Mapping[str, Any],
    *,
    stored_field_scopes: Mapping[tuple[str, ...], tuple[str, str]],
) -> DistributedConfig:
    try:
        return DistributedConfig.model_validate(values)
    except ValidationError as exc:
        errors = exc.errors(include_input=False)
        if not errors or any(error['type'] != 'extra_forbidden' for error in errors):
            raise

        ignored_fields: list[tuple[tuple[str | int, ...], tuple[str, str]]] = []
        for error in errors:
            location = tuple(error['loc'])
            scope = _find_stored_field_scope(location, stored_field_scopes)
            if scope is None:
                raise
            ignored_fields.append((location, scope))

        compatible_values = copy.deepcopy(dict(values))
        for location, _scope in ignored_fields:
            if not _remove_extra_field_at_location(compatible_values, location):
                raise

        config = DistributedConfig.model_validate(compatible_values)

    ignored_by_scope: dict[tuple[str, str], set[str]] = {}
    for location, scope in ignored_fields:
        ignored_by_scope.setdefault(scope, set()).add('.'.join(str(part) for part in location))
    for (role, identity), field_paths in ignored_by_scope.items():
        logger.warning(
            'Ignoring unknown distributed config fields in scope %s/%s for forward compatibility: %s',
            role,
            identity,
            ', '.join(sorted(field_paths)),
        )
    return config


def _find_stored_field_scope(
    location: tuple[str | int, ...],
    stored_field_scopes: Mapping[tuple[str, ...], tuple[str, str]],
) -> tuple[str, str] | None:
    matching_scopes = (
        (path, scope)
        for path, scope in stored_field_scopes.items()
        if len(location) >= len(path) and location[: len(path)] == path
    )
    match = max(matching_scopes, key=lambda item: len(item[0]), default=None)
    return None if match is None else match[1]


def _remove_extra_field_at_location(values: Any, location: tuple[str | int, ...]) -> bool:
    if not location:
        return False
    current = values
    for part in location[:-1]:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            # Pydantic inserts the discriminator value into union error paths without adding it to the input.
            if isinstance(part, str) and current.get('provider_type') == part:
                continue
            return False
        if isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
            continue
        return False

    field_name = location[-1]
    if not isinstance(current, dict) or field_name not in current:
        return False
    del current[field_name]
    return True


def _validate_secret_fields(values: Mapping[str, Any], *, role: str, identity: str) -> None:
    unknown = sorted(set(values).difference(DistributedSecrets.model_fields))
    if unknown:
        raise ValueError(f'unknown distributed secret field in scope {role}/{identity}: {", ".join(unknown)}')


def _validate_scope(role: str, identity: str) -> None:
    if not role:
        raise ValueError('distributed config role is required')
    if not identity:
        raise ValueError('distributed config identity is required')
    if len(role) > 64:
        raise ValueError('distributed config role exceeds 64 characters')
    if len(identity) > 128:
        raise ValueError('distributed config identity exceeds 128 characters')


def _validate_expected_version(expected_version: int) -> None:
    if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
        raise ValueError('expected config version must be a non-negative integer')


def _normalize_override_values(
    values: Mapping[str, Any],
    *,
    secret: bool,
    role: str,
    identity: str,
) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError('distributed config values must be an object')
    raw = dict(values)
    if secret:
        _validate_secret_fields(raw, role=role, identity=identity)
        validated = DistributedSecrets.model_validate(raw)
        return {name: _reveal_secret_value(getattr(validated, name)) for name in raw}
    _validate_db_fields(raw, role=role, identity=identity)
    # dump_for_persistence, not model_dump: the latter returns the runtime view, in which
    # extend_* entries have been folded into the ordinary field. Persisting that view would
    # leave a config no older build can load, defeating the extension fields.
    validated = DistributedConfig.model_validate(raw).dump_for_persistence(mode='json')
    return {name: validated[name] for name in raw}


def _reveal_secret_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, BaseModel):
        return {name: _reveal_secret_value(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {str(key): _reveal_secret_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_reveal_secret_value(item) for item in value]
    return value


def _record_from_row(
    row: DistributedConfigOverrideRow | DistributedConfigSecretOverrideRow,
    *,
    include_secret_values: bool,
) -> ConfigOverrideRecord:
    if isinstance(row, DistributedConfigSecretOverrideRow):
        payload = row.secrets_json
        secret = True
    else:
        payload = row.config_json
        secret = False
    if not isinstance(payload, dict):
        raise ValueError(f'distributed config scope {row.role}/{row.identity} must contain an object')
    if secret:
        _validate_secret_fields(payload, role=row.role, identity=row.identity)
    else:
        _validate_db_fields(payload, role=row.role, identity=row.identity)
    return ConfigOverrideRecord(
        role=row.role,
        identity=row.identity,
        version=row.version,
        fields=tuple(sorted(payload)),
        values=dict(payload) if not secret or include_secret_values else {},
    )


def _version_conflict_detail(expected: int, found: int | None) -> str:
    found_text = 'missing' if found is None else str(found)
    return f'distributed config version conflict: expected {expected}, found {found_text}'


__all__ = [
    'ConfigOverrideRecord',
    'ConfigVersionConflictError',
    'GLOBAL_CONFIG_SCOPE',
    'delete_config_override',
    'get_config_override',
    'initialize_distributed_config',
    'list_config_overrides',
    'list_secret_override_summaries',
    'load_distributed_config',
    'set_config_override',
    'schema_revision_is_compatible',
]
