from __future__ import annotations

import hashlib
import hmac
import secrets

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from nexus.distributed.access.models import PrincipalAccessGrantRecord, SubmitTokenKind
from nexus.distributed.access.permissions import CANONICAL_USER_SCOPES
from nexus.distributed.errors import DomainConflictError, DomainValidationError, ResourceNotFoundError
from nexus.distributed.persistence.orm import (
    EnvironmentGrantRow,
    EnvironmentRow,
    PrincipalAccessGrantRow,
    PrincipalRow,
    SubmitTokenRow,
    UserQuotaOverrideRow,
    UserRoleRow,
)
from nexus.distributed.utils.identity import (
    IDENTITY_DISPLAY_NAME_MAX_LENGTH,
    IDENTITY_ISSUER_MAX_LENGTH,
    IDENTITY_SUBJECT_MAX_LENGTH,
    IDENTITY_TENANT_MAX_LENGTH,
    PRINCIPAL_ID_MAX_LENGTH,
    new_id,
)


LEGACY_SUBMISSION_SCOPE_ALIASES = {
    'submit:batch': 'submission:batch',
    'submit:live': 'submission:live',
    'submit:single': 'submission:single',
}
# Keep preview Gateway validation compiling while persisted token scopes use the
# canonical submission:* namespace.
DEFAULT_USER_SCOPES = set(CANONICAL_USER_SCOPES) | set(LEGACY_SUBMISSION_SCOPE_ALIASES)


def _mysql_ensure_principal(values: dict[str, Any]) -> Any:
    return mysql_insert(PrincipalRow).values(values).on_duplicate_key_update(principal_id=PrincipalRow.principal_id)


@dataclass(frozen=True, slots=True)
class AccessUser:
    principal_id: str
    issuer: str
    tenant: str
    subject: str
    roles: frozenset[str]
    display_name: str | None


@dataclass(frozen=True, slots=True)
class AuthenticatedToken:
    token_id: str
    user: AccessUser
    environment_id: str
    scopes: frozenset[str]


@dataclass(frozen=True, slots=True)
class AuthenticatedAppToken:
    token_id: str
    token_kind: str
    user: AccessUser
    environment_id: str | None
    scopes: frozenset[str]


class AccessRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_user(self, principal_id: str) -> AccessUser | None:
        row = await self.session.get(PrincipalRow, principal_id)
        if row is None:
            row = await self.session.get(UserRoleRow, principal_id)
        return _user(row) if row is not None else None

    async def list_users(self) -> list[AccessUser]:
        rows = (await self.session.scalars(select(PrincipalRow).order_by(PrincipalRow.subject))).all()
        users = {row.principal_id: _user(row) for row in rows}
        legacy_rows = (await self.session.scalars(select(UserRoleRow).order_by(UserRoleRow.subject))).all()
        for row in legacy_rows:
            users.setdefault(row.principal_id, _user(row))
        return sorted(users.values(), key=lambda user: (user.subject, user.principal_id))

    async def list_system_token_app_bindings(self) -> dict[str, str]:
        rows = (
            await self.session.execute(
                select(SubmitTokenRow.owner_principal_id, SubmitTokenRow.bound_app_id).where(
                    SubmitTokenRow.token_kind == SubmitTokenKind.SYSTEM_DEFAULT,
                    SubmitTokenRow.bound_app_id.is_not(None),
                )
            )
        ).all()
        return {owner_principal_id: app_id for owner_principal_id, app_id in rows if app_id is not None}

    async def put_principal_access_grant(
        self,
        *,
        owner_principal_id: str,
        grantee_principal_id: str,
        actor: str,
        now: datetime,
    ) -> PrincipalAccessGrantRecord:
        if owner_principal_id == grantee_principal_id:
            raise DomainValidationError('principal_access_grant_self')
        if await self.get_user(owner_principal_id) is None or await self.get_user(grantee_principal_id) is None:
            raise ResourceNotFoundError('user_not_found')
        values = {
            'owner_principal_id': owner_principal_id,
            'grantee_principal_id': grantee_principal_id,
            'created_by': actor,
            'created_at': now,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(
                mysql_insert(PrincipalAccessGrantRow)
                .values(values)
                .on_duplicate_key_update(owner_principal_id=PrincipalAccessGrantRow.owner_principal_id)
            )
        elif dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(PrincipalAccessGrantRow)
                .values(values)
                .on_conflict_do_nothing(index_elements=['owner_principal_id', 'grantee_principal_id'])
            )
        elif await self.session.get(PrincipalAccessGrantRow, (owner_principal_id, grantee_principal_id)) is None:
            self.session.add(PrincipalAccessGrantRow(**values))
        await self.session.flush()
        row = await self.session.get(PrincipalAccessGrantRow, (owner_principal_id, grantee_principal_id))
        if row is None:
            raise RuntimeError('principal_access_grant_unavailable')
        return _principal_access_grant_record(row)

    async def delete_principal_access_grant(
        self,
        *,
        owner_principal_id: str,
        grantee_principal_id: str,
    ) -> bool:
        row = await self.session.get(
            PrincipalAccessGrantRow,
            (owner_principal_id, grantee_principal_id),
        )
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True

    async def list_principal_access_grants(
        self,
        *,
        owner_principal_id: str | None = None,
        grantee_principal_id: str | None = None,
    ) -> list[PrincipalAccessGrantRecord]:
        query = select(PrincipalAccessGrantRow)
        if owner_principal_id is not None:
            query = query.where(PrincipalAccessGrantRow.owner_principal_id == owner_principal_id)
        if grantee_principal_id is not None:
            query = query.where(PrincipalAccessGrantRow.grantee_principal_id == grantee_principal_id)
        rows = (
            await self.session.scalars(
                query.order_by(
                    PrincipalAccessGrantRow.owner_principal_id,
                    PrincipalAccessGrantRow.grantee_principal_id,
                )
            )
        ).all()
        return [_principal_access_grant_record(row) for row in rows]

    async def list_accessible_owner_principal_ids(self, grantee_principal_id: str) -> frozenset[str]:
        owners = await self.session.scalars(
            select(PrincipalAccessGrantRow.owner_principal_id).where(
                PrincipalAccessGrantRow.grantee_principal_id == grantee_principal_id
            )
        )
        return frozenset({grantee_principal_id, *owners})

    async def ensure_user(
        self,
        *,
        principal_id: str,
        issuer: str,
        tenant: str,
        subject: str,
        roles: set[str],
        display_name: str | None,
        actor: str,
        now: datetime,
    ) -> AccessUser:
        values = _user_values(
            principal_id=principal_id,
            issuer=issuer,
            tenant=tenant,
            subject=subject,
            roles=roles,
            display_name=display_name,
            actor=actor,
            now=now,
        )
        await self._insert_user_if_missing(values)
        row = await self.session.scalar(
            select(PrincipalRow).where(PrincipalRow.principal_id == principal_id).with_for_update()
        )
        if row is None:
            raise RuntimeError('user_role_row_unavailable')
        _require_matching_identity(row, issuer=issuer, tenant=tenant, subject=subject)
        if row.display_name is None and display_name is not None:
            row.display_name = display_name
            row.updated_at = now
            await self.session.flush()
        return _user(row)

    async def upsert_user(
        self,
        *,
        principal_id: str,
        issuer: str,
        tenant: str,
        subject: str,
        roles: set[str],
        display_name: str | None,
        actor: str,
        now: datetime,
    ) -> AccessUser:
        values = _user_values(
            principal_id=principal_id,
            issuer=issuer,
            tenant=tenant,
            subject=subject,
            roles=roles,
            display_name=display_name,
            actor=actor,
            now=now,
        )
        await self._insert_user_if_missing(values)
        row = await self.session.scalar(
            select(PrincipalRow).where(PrincipalRow.principal_id == principal_id).with_for_update()
        )
        if row is None:
            raise RuntimeError('user_role_row_unavailable')
        _require_matching_identity(row, issuer=issuer, tenant=tenant, subject=subject)
        row.roles_json = sorted(roles)
        row.display_name = display_name
        row.updated_at = now
        await self.session.flush()
        return _user(row)

    async def _insert_user_if_missing(self, values: dict[str, Any]) -> None:
        principal_id = values['principal_id']
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(_mysql_ensure_principal(values))
        elif dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(PrincipalRow).values(values).on_conflict_do_nothing(index_elements=['principal_id'])
            )
        elif await self.session.get(PrincipalRow, principal_id) is None:
            self.session.add(PrincipalRow(**values))
        await self.session.flush()

    async def ensure_system_token(
        self,
        *,
        owner_principal_id: str,
        created_by: str,
        now: datetime,
    ) -> SubmitTokenRow:
        if await self.get_user(owner_principal_id) is None:
            raise ResourceNotFoundError('user_not_found')
        token_id = _system_token_id(owner_principal_id)
        existing = await self.session.get(SubmitTokenRow, token_id)
        if existing is not None:
            return _require_system_token(existing, owner_principal_id=owner_principal_id)
        values = {
            'token_id': token_id,
            'token_kind': SubmitTokenKind.SYSTEM_DEFAULT,
            'token_prefix': None,
            'token_hash': None,
            'token_hash_key_id': None,
            'owner_principal_id': owner_principal_id,
            'environment_id': None,
            'scopes_json': [],
            'created_by': created_by,
            'created_at': now,
            'expires_at': None,
            'revoked_at': None,
            'last_used_at': None,
        }
        dialect = self.session.get_bind().dialect.name
        if dialect == 'mysql':
            await self.session.execute(
                mysql_insert(SubmitTokenRow).values(values).on_duplicate_key_update(token_id=SubmitTokenRow.token_id)
            )
        elif dialect == 'sqlite':
            await self.session.execute(
                sqlite_insert(SubmitTokenRow).values(values).on_conflict_do_nothing(index_elements=['token_id'])
            )
        elif await self.session.get(SubmitTokenRow, token_id) is None:
            self.session.add(SubmitTokenRow(**values))
        await self.session.flush()
        row = await self.session.get(SubmitTokenRow, token_id)
        if row is None:
            raise RuntimeError('system_default_token_unavailable')
        return _require_system_token(row, owner_principal_id=owner_principal_id)

    async def get_system_token(self, owner_principal_id: str) -> SubmitTokenRow | None:
        row = await self.session.get(SubmitTokenRow, _system_token_id(owner_principal_id))
        if row is None:
            return None
        try:
            return _require_system_token(row, owner_principal_id=owner_principal_id)
        except RuntimeError:
            return None

    async def create_token(
        self,
        *,
        owner_principal_id: str,
        scopes: set[str],
        created_by: str,
        key_id: str,
        key: str,
        now: datetime,
        expires_at: datetime | None = None,
        environment_id: str | None = None,
    ) -> tuple[SubmitTokenRow, str]:
        user = await self.get_user(owner_principal_id)
        if user is None:
            raise ResourceNotFoundError('user_not_found')
        if environment_id is None:
            raise DomainValidationError('environment_id_required')
        environment = await self.session.get(EnvironmentRow, environment_id)
        grant = await self.session.get(EnvironmentGrantRow, (owner_principal_id, environment_id))
        if environment is None or grant is None:
            raise DomainConflictError('environment_grant_required')
        if environment.status != 'enabled':
            raise DomainConflictError('environment_disabled')
        if not grant.enabled:
            raise DomainConflictError('environment_grant_disabled')
        canonical_scopes = _canonical_scopes(scopes)
        if not canonical_scopes <= frozenset(grant.scopes_json or []):
            raise DomainConflictError('token_scopes_exceed_grant')
        secret = secrets.token_urlsafe(32)
        token_id = new_id('tok')
        token = f'nxs_{token_id}.{secret}'
        row = SubmitTokenRow(
            token_id=token_id,
            token_kind=SubmitTokenKind.API,
            token_prefix=token[:16],
            token_hash=_token_hash(key, token),
            token_hash_key_id=key_id,
            owner_principal_id=owner_principal_id,
            environment_id=environment_id,
            scopes_json=sorted(canonical_scopes),
            created_by=created_by,
            created_at=now,
            expires_at=expires_at,
        )
        self.session.add(row)
        await self.session.flush()
        return row, token

    async def authenticate_token(
        self,
        token: str,
        *,
        keys: dict[str, str],
        now: datetime,
    ) -> AuthenticatedToken | None:
        token_id = _token_id(token)
        if token_id is None:
            return None
        row = await self.session.get(SubmitTokenRow, token_id)
        if (
            row is None
            or row.token_kind != SubmitTokenKind.API
            or not row.environment_id
            or not row.token_hash
            or not row.token_hash_key_id
            or row.revoked_at is not None
            or (row.expires_at is not None and row.expires_at <= now)
        ):
            return None
        key = keys.get(row.token_hash_key_id)
        if key is None or not hmac.compare_digest(row.token_hash, _token_hash(key, token)):
            return None
        environment = await self.session.get(EnvironmentRow, row.environment_id)
        grant = await self.session.get(
            EnvironmentGrantRow,
            (row.owner_principal_id, row.environment_id),
        )
        user = await self.get_user(row.owner_principal_id)
        if environment is None or environment.status != 'enabled' or user is None:
            return None
        if grant is None or not grant.enabled:
            return None
        row.last_used_at = now
        await self.session.flush()
        canonical_scopes = frozenset(
            LEGACY_SUBMISSION_SCOPE_ALIASES.get(scope, scope) for scope in (row.scopes_json or [])
        )
        canonical_scopes &= frozenset(grant.scopes_json or [])
        return AuthenticatedToken(
            token_id=row.token_id,
            user=user,
            environment_id=row.environment_id,
            scopes=_compatible_scopes(canonical_scopes),
        )

    async def bind_token_app_id(self, token_id: str, *, app_id: str) -> SubmitTokenRow:
        normalized_app_id = _normalize_app_id(app_id)
        row = await self.session.scalar(
            select(SubmitTokenRow).where(SubmitTokenRow.token_id == token_id).with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('submit_token_not_found')
        existing = await self.session.scalar(
            select(SubmitTokenRow).where(SubmitTokenRow.bound_app_id == normalized_app_id)
        )
        if existing is not None and existing.token_id != row.token_id:
            raise DomainConflictError('app_id_already_bound')
        row.bound_app_id = normalized_app_id
        await self.session.flush()
        return row

    async def unbind_token_app_id(self, token_id: str) -> SubmitTokenRow:
        row = await self.session.scalar(
            select(SubmitTokenRow).where(SubmitTokenRow.token_id == token_id).with_for_update()
        )
        if row is None:
            raise ResourceNotFoundError('submit_token_not_found')
        row.bound_app_id = None
        await self.session.flush()
        return row

    async def authenticate_app_id(self, app_id: str, *, now: datetime) -> AuthenticatedAppToken | None:
        try:
            normalized_app_id = _normalize_app_id(app_id)
        except DomainValidationError:
            return None
        row = await self.session.scalar(select(SubmitTokenRow).where(SubmitTokenRow.bound_app_id == normalized_app_id))
        if row is None or row.revoked_at is not None or (row.expires_at is not None and row.expires_at <= now):
            return None
        user = await self.get_user(row.owner_principal_id)
        if user is None:
            return None
        if row.token_kind == SubmitTokenKind.SYSTEM_DEFAULT:
            return AuthenticatedAppToken(
                token_id=row.token_id,
                token_kind=row.token_kind,
                user=user,
                environment_id=None,
                scopes=frozenset(CANONICAL_USER_SCOPES),
            )
        if row.token_kind != SubmitTokenKind.API or not row.environment_id:
            return None
        environment = await self.session.get(EnvironmentRow, row.environment_id)
        grant = await self.session.get(
            EnvironmentGrantRow,
            (row.owner_principal_id, row.environment_id),
        )
        if environment is None or environment.status != 'enabled' or grant is None or not grant.enabled:
            return None
        canonical_scopes = frozenset(
            LEGACY_SUBMISSION_SCOPE_ALIASES.get(scope, scope) for scope in (row.scopes_json or [])
        )
        canonical_scopes &= frozenset(grant.scopes_json or [])
        return AuthenticatedAppToken(
            token_id=row.token_id,
            token_kind=row.token_kind,
            user=user,
            environment_id=row.environment_id,
            scopes=_compatible_scopes(canonical_scopes),
        )

    async def revoke_token(self, token_id: str, *, owner_principal_id: str | None, now: datetime) -> None:
        row = await self.session.get(SubmitTokenRow, token_id)
        if row is None or (owner_principal_id is not None and row.owner_principal_id != owner_principal_id):
            raise ResourceNotFoundError('submit_token_not_found')
        row.revoked_at = now
        await self.session.flush()

    async def list_tokens(self, *, owner_principal_id: str | None = None) -> list[dict]:
        query = select(SubmitTokenRow).where(SubmitTokenRow.token_kind == SubmitTokenKind.API)
        if owner_principal_id is not None:
            query = query.where(SubmitTokenRow.owner_principal_id == owner_principal_id)
        rows = (await self.session.scalars(query.order_by(SubmitTokenRow.created_at.desc()))).all()
        return [
            {
                'token_id': row.token_id,
                'token_prefix': row.token_prefix,
                'owner_principal_id': row.owner_principal_id,
                'environment_id': row.environment_id,
                'scopes': list(row.scopes_json or []),
                'created_by': row.created_by,
                'created_at': row.created_at,
                'expires_at': row.expires_at,
                'revoked_at': row.revoked_at,
                'last_used_at': row.last_used_at,
            }
            for row in rows
        ]

    async def get_quota(self, owner_principal_id: str) -> UserQuotaOverrideRow | None:
        return await self.session.get(UserQuotaOverrideRow, owner_principal_id)

    async def set_quota(
        self,
        owner_principal_id: str,
        *,
        limits: dict,
        actor: str,
        now: datetime,
    ) -> UserQuotaOverrideRow:
        row = await self.session.get(UserQuotaOverrideRow, owner_principal_id)
        if row is None:
            row = UserQuotaOverrideRow(
                owner_principal_id=owner_principal_id,
                limits_json=limits,
                version=1,
                updated_by=actor,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
        else:
            row.limits_json = limits
            row.version += 1
            row.updated_by = actor
            row.updated_at = now
        await self.session.flush()
        return row


def _user_values(
    *,
    principal_id: str,
    issuer: str,
    tenant: str,
    subject: str,
    roles: set[str],
    display_name: str | None,
    actor: str,
    now: datetime,
) -> dict[str, Any]:
    if not roles or not roles <= {'user', 'admin'}:
        raise DomainValidationError('invalid_roles')
    if (
        not principal_id
        or len(principal_id) > PRINCIPAL_ID_MAX_LENGTH
        or not issuer
        or len(issuer) > IDENTITY_ISSUER_MAX_LENGTH
        or not tenant
        or len(tenant) > IDENTITY_TENANT_MAX_LENGTH
        or not subject
        or len(subject) > IDENTITY_SUBJECT_MAX_LENGTH
        or (display_name is not None and len(display_name) > IDENTITY_DISPLAY_NAME_MAX_LENGTH)
    ):
        raise DomainValidationError('invalid_user_identity')
    return {
        'principal_id': principal_id,
        'issuer': issuer,
        'tenant': tenant,
        'subject': subject,
        'roles_json': sorted(roles),
        'display_name': display_name,
        'created_by': actor,
        'created_at': now,
        'updated_at': now,
    }


def _require_matching_identity(
    row: PrincipalRow,
    *,
    issuer: str,
    tenant: str,
    subject: str,
) -> None:
    if (row.issuer, row.tenant, row.subject) != (issuer, tenant, subject):
        raise DomainConflictError('principal_identity_mismatch')


def _token_hash(key: str, token: str) -> str:
    return hmac.new(key.encode(), token.encode(), hashlib.sha256).hexdigest()


def _require_system_token(row: SubmitTokenRow, *, owner_principal_id: str) -> SubmitTokenRow:
    if row.token_kind != SubmitTokenKind.SYSTEM_DEFAULT or row.owner_principal_id != owner_principal_id:
        raise RuntimeError('system_default_token_unavailable')
    return row


def _system_token_id(owner_principal_id: str) -> str:
    digest = hashlib.sha256(owner_principal_id.encode()).hexdigest()
    return f'tok_default_{digest[:48]}'


def _token_id(token: str) -> str | None:
    if not token.startswith('nxs_tok_') or '.' not in token:
        return None
    token_id = token[4:].split('.', 1)[0]
    return token_id if len(token_id) <= 64 else None


def _normalize_app_id(value: str) -> str:
    normalized = str(value or '').strip().lower()
    if (
        not normalized
        or len(normalized) > 128
        or normalized[0] not in 'abcdefghijklmnopqrstuvwxyz0123456789'
        or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789._-' for character in normalized)
    ):
        raise DomainValidationError('invalid_app_id')
    return normalized


def _canonical_scopes(scopes: set[str]) -> frozenset[str]:
    canonical = frozenset(LEGACY_SUBMISSION_SCOPE_ALIASES.get(scope, scope) for scope in scopes)
    if not canonical or not canonical <= CANONICAL_USER_SCOPES:
        raise DomainValidationError('invalid_token_scopes')
    return canonical


def _compatible_scopes(scopes: frozenset[str]) -> frozenset[str]:
    values = set(scopes)
    for legacy, canonical in LEGACY_SUBMISSION_SCOPE_ALIASES.items():
        if canonical in scopes:
            values.add(legacy)
    return frozenset(values)


def _user(row: PrincipalRow | UserRoleRow) -> AccessUser:
    return AccessUser(
        principal_id=row.principal_id,
        issuer=row.issuer,
        tenant=row.tenant,
        subject=row.subject,
        roles=frozenset(row.roles_json or []),
        display_name=row.display_name,
    )


def _principal_access_grant_record(row: PrincipalAccessGrantRow) -> PrincipalAccessGrantRecord:
    return PrincipalAccessGrantRecord(
        owner_principal_id=row.owner_principal_id,
        grantee_principal_id=row.grantee_principal_id,
        created_by=row.created_by,
        created_at=row.created_at,
    )
