from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import aiohttp
import pytest

from aiohttp.test_utils import TestClient, TestServer
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from nexus.distributed.access.auth import (
    AuthContext,
    _enforce_browser_origin,
    csrf_token,
)
from nexus.distributed.access.taihu import principal_id
from nexus.distributed.artifacts.tickets import UploadTicketSigner
from nexus.distributed.config import DistributedConfig
from nexus.distributed.config.store import load_distributed_config, set_config_override
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.gateways.control.app import create_control_app
from nexus.distributed.persistence.orm import (
    ArtifactRow,
    AuditRow,
    EnvironmentGrantRow,
    EnvironmentResourceUsageRow,
    ExecutionRow,
    PrincipalRow,
    SubmitTokenRow,
    create_schema,
)
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.utils.clock import utcnow


_TAI_APPLICATION_TOKEN = 'a' * 32
_INTERNAL_TOKEN = 'i' * 32
_ADMIN_STAFF_ID = 1000
_USER_STAFF_ID = 1001
_OTHER_STAFF_ID = 1002


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode('ascii')


def _identity_headers(staff_id: int, *, login_name: str | None = None) -> dict[str, str]:
    protected = _base64url(b'{"alg":"dir","enc":"A256GCM"}')
    payload = json.dumps(
        {
            'StaffId': staff_id,
            'LoginName': login_name or f'user-{staff_id}',
            'Expiration': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
        separators=(',', ':'),
    ).encode()
    iv = os.urandom(12)
    encrypted = AESGCM(_TAI_APPLICATION_TOKEN.encode()).encrypt(iv, payload, protected.encode())
    compact = '.'.join((
        protected,
        '',
        _base64url(iv),
        _base64url(encrypted[:-16]),
        _base64url(encrypted[-16:]),
    ))
    return {'x-tai-identity': compact}


def _sgw_headers(app_id: str, *, now: datetime | None = None, sequence: str = 'test-sequence') -> dict[str, str]:
    timestamp = str(int((now or datetime.now(timezone.utc)).timestamp()))
    signature = hashlib.sha256(f'{timestamp}{_TAI_APPLICATION_TOKEN}{sequence}{timestamp}'.encode()).hexdigest().upper()
    return {
        'x-rio-paasid': app_id,
        'timestamp': timestamp,
        'x-rio-seq': sequence,
        'signature': signature,
    }


@pytest.fixture
async def auth_session_factory(tmp_path):
    engine = create_async_engine(f'sqlite+aiosqlite:///{tmp_path / "auth.db"}')
    await create_schema(engine)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


async def test_token_scopes_are_bounded_by_environment_grant(auth_session_factory):
    now = utcnow()
    async with auth_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id='principal-scoped',
            issuer='tai',
            tenant='tenant',
            subject='scoped-user',
            roles={'user'},
            display_name='Scoped User',
            actor='admin',
            now=now,
        )
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='env-scoped',
            name='Scoped Environment',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={'slots': 10},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        await environments.upsert_grant(
            principal_id='principal-scoped',
            environment_id='env-scoped',
            limits={'slots': 5},
            scopes={'submission:single', 'artifact:write'},
            enabled=True,
            actor='admin',
            now=now,
        )
        with pytest.raises(ValueError, match='token_scopes_exceed_grant'):
            await access.create_token(
                owner_principal_id='principal-scoped',
                environment_id='env-scoped',
                scopes={'execution:read'},
                created_by='principal-scoped',
                key_id='v1',
                key='token-secret',
                now=now,
            )
        _, token = await access.create_token(
            owner_principal_id='principal-scoped',
            environment_id='env-scoped',
            scopes={'submission:single', 'artifact:write'},
            created_by='principal-scoped',
            key_id='v1',
            key='token-secret',
            now=now,
        )
        await environments.upsert_grant(
            principal_id='principal-scoped',
            environment_id='env-scoped',
            limits={'slots': 5},
            scopes={'artifact:write'},
            enabled=True,
            actor='admin',
            now=utcnow(),
        )
        await session.commit()

    async with auth_session_factory() as session:
        authenticated = await AccessRepository(session).authenticate_token(
            token,
            keys={'v1': 'token-secret'},
            now=utcnow(),
        )
        assert authenticated is not None
        assert authenticated.environment_id == 'env-scoped'
        assert authenticated.scopes == frozenset({'artifact:write'})


async def test_existing_token_with_legacy_submit_scope_gets_canonical_alias(auth_session_factory):
    now = utcnow()
    async with auth_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id='principal-legacy-token',
            issuer='tai',
            tenant='tenant',
            subject='legacy-token-user',
            roles={'user'},
            display_name=None,
            actor='admin',
            now=now,
        )
        environments = EnvironmentRepository(session)
        await environments.create_environment(
            environment_id='env-legacy-token',
            name='Legacy Token Environment',
            allocation_type='predefined',
            provider_type='predefined',
            provider_config_ref=None,
            resource_limits={'slots': 1},
            default_timeouts={},
            status='enabled',
            actor='admin',
            now=now,
        )
        await environments.upsert_grant(
            principal_id='principal-legacy-token',
            environment_id='env-legacy-token',
            limits={'slots': 1},
            enabled=True,
            actor='admin',
            now=now,
        )
        row, token = await access.create_token(
            owner_principal_id='principal-legacy-token',
            environment_id='env-legacy-token',
            scopes={'submission:single'},
            created_by='principal-legacy-token',
            key_id='v1',
            key='token-secret',
            now=now,
        )
        row.scopes_json = ['submit:single']
        await session.commit()

    async with auth_session_factory() as session:
        authenticated = await AccessRepository(session).authenticate_token(
            token,
            keys={'v1': 'token-secret'},
            now=utcnow(),
        )

    assert authenticated is not None
    assert authenticated.scopes == frozenset({'submission:single', 'submit:single'})


def _identity_config(
    *,
    application_token: str = _TAI_APPLICATION_TOKEN,
    trust_x_rio_app_id: bool = False,
) -> DistributedConfig:
    return DistributedConfig(
        auth_mode='required',
        tai_trust_x_rio_app_id=trust_x_rio_app_id,
        admin_users=[f'tai/default/{_ADMIN_STAFF_ID}'],
        environment_enabled_providers=['kubernetes'],
        secrets={
            'internal_token': _INTERNAL_TOKEN,
            'tai_application_token': application_token,
        },
    )


@pytest.fixture
def identity_config() -> DistributedConfig:
    return _identity_config()


async def test_invalid_sgw_identity_logs_only_auth_header_counts(
    auth_session_factory,
    identity_config,
    caplog,
):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    headers = {
        'x-rio-paasid': 'sensitive-app-id',
        'x-rio-timestamp': 'sensitive-caller-timestamp',
        'x-rio-nonce': 'sensitive-caller-nonce',
        'x-rio-signature': 'sensitive-caller-signature',
        'x-rio-seq': 'gateway-sequence',
    }
    try:
        with caplog.at_level(logging.WARNING, logger='nexus.distributed.access.auth'):
            response = await client.get('/v1/me', headers=headers)

        assert response.status == 401
        assert "'signature': 0" in caplog.text
        assert "'timestamp': 0" in caplog.text
        assert "'x-rio-paasid': 1" in caplog.text
        assert "'x-rio-seq': 1" in caplog.text
        assert 'sensitive-app-id' not in caplog.text
        assert 'sensitive-caller' not in caplog.text
        assert 'gateway-sequence' not in caplog.text
    finally:
        await client.close()


async def test_identity_header_uses_tai_jwe_without_proxy_token(auth_session_factory, identity_config):
    config = identity_config
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    try:
        plaintext = await client.get('/v1/me', headers={'x-tai-identity': f'tai/default/{_ADMIN_STAFF_ID}'})
        assert plaintext.status == 401

        trusted = await client.get('/v1/me', headers=_identity_headers(_ADMIN_STAFF_ID, login_name='admin'))
        assert trusted.status == 200, await trusted.text()
        payload = await trusted.json()
        assert payload['subject'] == str(_ADMIN_STAFF_ID)
        assert payload['display_name'] == 'admin'
        assert payload['roles'] == ['admin', 'user']
        async with auth_session_factory() as session:
            row = await session.get(PrincipalRow, payload['principal_id'])
            assert row is not None
            assert row.display_name == 'admin'

        tai_identity_wins = await client.get(
            '/v1/me',
            headers={
                **_identity_headers(_ADMIN_STAFF_ID, login_name='admin'),
                'Authorization': 'Bearer invalid-token',
            },
        )
        assert tai_identity_wins.status == 200, await tai_identity_wins.text()
        assert (await tai_identity_wins.json())['auth_type'] == 'tai_identity'

        tai_identity_with_sgw_forwarding = await client.get(
            '/v1/me',
            headers={
                **_identity_headers(_ADMIN_STAFF_ID, login_name='admin'),
                'x-rio-paasid': 'gateway-app',
                'x-rio-timestamp': 'forwarded-timestamp',
                'x-rio-nonce': 'forwarded-nonce',
                'x-rio-signature': 'forwarded-signature',
                'x-rio-seq': 'forwarded-sequence',
            },
        )
        assert tai_identity_with_sgw_forwarding.status == 200, await tai_identity_with_sgw_forwarding.text()
        assert (await tai_identity_with_sgw_forwarding.json())['auth_type'] == 'tai_identity'
    finally:
        await client.close()


async def test_valid_tai_identity_is_registered_as_unprivileged_user_on_first_request(auth_session_factory):
    config = DistributedConfig(
        auth_mode='required',
        secrets={'tai_application_token': _TAI_APPLICATION_TOKEN},
    )
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    try:
        response = await client.get('/v1/me', headers=_identity_headers(_USER_STAFF_ID, login_name='first-visitor'))

        assert response.status == 200, await response.text()
        payload = await response.json()
        assert payload['principal_id'] == user_id
        assert payload['subject'] == str(_USER_STAFF_ID)
        assert payload['display_name'] == 'first-visitor'
        assert payload['roles'] == ['user']
        assert payload['system_token'] == {'label': '默认'}

        async with auth_session_factory() as session:
            row = await session.get(PrincipalRow, user_id)
            assert row is not None
            assert row.roles_json == ['user']
            assert row.created_by == 'system:tai-identity'
            tokens = list(
                await session.scalars(select(SubmitTokenRow).where(SubmitTokenRow.owner_principal_id == user_id))
            )
            assert len(tokens) == 1
            assert tokens[0].token_kind == 'system_default'
            assert tokens[0].token_hash is None
            assert tokens[0].environment_id is None
    finally:
        await client.close()


async def test_non_admin_cannot_read_database_config_overrides(auth_session_factory):
    async with auth_session_factory() as session:
        await set_config_override(
            session,
            role='*',
            identity='*',
            values={'scheduler_poll_interval': 0.25},
            expected_version=0,
            actor='admin-test',
        )
        await session.commit()

    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=_identity_config())))
    await client.start_server()
    try:
        response = await client.get(
            '/v1/system/config-overrides',
            headers=_identity_headers(_USER_STAFF_ID, login_name='ordinary-user'),
        )

        assert response.status == 403
        assert response.headers['Cache-Control'] == 'no-store'
        assert await response.json() == {'error': 'admin_required', 'retryable': False}
    finally:
        await client.close()


async def test_non_admin_cannot_write_database_config_overrides(auth_session_factory):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=_identity_config())))
    await client.start_server()
    try:
        response = await client.put(
            '/v1/system/config-overrides',
            headers=_identity_headers(_USER_STAFF_ID, login_name='ordinary-user'),
            json={
                'role': '*',
                'identity': '*',
                'values': {'scheduler_poll_interval': 0.25},
                'expected_version': 0,
            },
        )

        assert response.status == 403
        assert await response.json() == {'error': 'admin_required', 'retryable': False}
    finally:
        await client.close()


async def test_admin_binds_system_token_to_app_identity(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    try:
        registered = await client.get('/v1/me', headers=_identity_headers(_USER_STAFF_ID, login_name='app-owner'))
        assert registered.status == 200, await registered.text()

        async with auth_session_factory() as session:
            token = await AccessRepository(session).ensure_system_token(
                owner_principal_id=user_id,
                created_by='test',
                now=utcnow(),
            )
            token_id = token.token_id

        bound = await client.put(
            f'/v1/access/tokens/{token_id}/app-binding',
            headers=_identity_headers(_ADMIN_STAFF_ID, login_name='admin'),
            json={'app_id': 'mcp-l2m93f'},
        )
        assert bound.status == 200, await bound.text()
        assert (await bound.json()) == {
            'token_id': token_id,
            'token_kind': 'system_default',
            'owner_principal_id': user_id,
            'environment_id': None,
            'app_id': 'mcp-l2m93f',
        }

        app_request = await client.get(
            '/v1/me',
            headers=_sgw_headers('mcp-l2m93f'),
        )
        assert app_request.status == 200, await app_request.text()
        assert await app_request.json() == {
            'principal_id': user_id,
            'issuer': 'tai',
            'tenant': 'default',
            'subject': str(_USER_STAFF_ID),
            'display_name': 'app-owner',
            'roles': ['user'],
            'scopes': [
                'artifact:write',
                'execution:command',
                'execution:read',
                'execution:stop',
                'submission:batch',
                'submission:live',
                'submission:single',
            ],
            'auth_type': 'tai_app',
            'app_id': 'mcp-l2m93f',
            'system_token': {'label': '默认'},
            'csrf_token': None,
        }

        personal_with_signed_sgw = await client.get(
            '/v1/me',
            headers={
                **_identity_headers(_USER_STAFF_ID, login_name='app-owner'),
                **_sgw_headers('mcp-l2m93f'),
            },
        )
        assert personal_with_signed_sgw.status == 200, await personal_with_signed_sgw.text()
        personal_payload = await personal_with_signed_sgw.json()
        assert personal_payload['auth_type'] == 'tai_identity'
        assert 'app_id' not in personal_payload

        partial_sgw_identity = await client.get(
            '/v1/me',
            headers={
                **_identity_headers(_USER_STAFF_ID, login_name='app-owner'),
                'timestamp': '1',
            },
        )
        assert partial_sgw_identity.status == 200, await partial_sgw_identity.text()
        assert (await partial_sgw_identity.json())['auth_type'] == 'tai_identity'

        forged = await client.get('/v1/me', headers={'x-rio-paasid': 'mcp-l2m93f'})
        assert forged.status == 401

        trusted_client = TestClient(
            TestServer(
                create_control_app(
                    session_factory=auth_session_factory,
                    config=_identity_config(trust_x_rio_app_id=True),
                )
            )
        )
        await trusted_client.start_server()
        try:
            trusted = await trusted_client.get(
                '/v1/me',
                headers={'x-rio-paasid': 'mcp-l2m93f', 'x-rio-seq': 'forwarded-sequence'},
            )
            assert trusted.status == 200, await trusted.text()
            assert (await trusted.json())['app_id'] == 'mcp-l2m93f'

            partial_backend_signature = await trusted_client.get(
                '/v1/me',
                headers={
                    'x-rio-paasid': 'mcp-l2m93f',
                    'x-rio-seq': 'forwarded-sequence',
                    'timestamp': '1',
                },
            )
            assert partial_backend_signature.status == 401
        finally:
            await trusted_client.close()
    finally:
        await client.close()


async def test_internal_principal_delegation_restores_personal_identity(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    try:
        registered = await client.get('/v1/me', headers=_identity_headers(_USER_STAFF_ID, login_name='delegated-user'))
        assert registered.status == 200, await registered.text()

        delegated = await client.get(
            '/v1/me',
            headers={
                'X-Nexus-Internal-Token': _INTERNAL_TOKEN,
                'X-Nexus-Delegated-Principal-Id': user_id,
            },
        )
        assert delegated.status == 200, await delegated.text()
        payload = await delegated.json()
        assert payload['principal_id'] == user_id
        assert payload['display_name'] == 'delegated-user'
        assert payload['roles'] == ['user']
        assert payload['scopes'] == ['*']
        assert payload['auth_type'] == 'tai_identity'
        assert payload['system_token'] == {'label': '默认'}
        assert payload['csrf_token'] is None

        forged = await client.get('/v1/me', headers={'X-Nexus-Delegated-Principal-Id': user_id})
        assert forged.status == 401
    finally:
        await client.close()


async def test_admin_manages_user_system_token_app_binding(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    app_headers = {
        'X-Nexus-Internal-Token': _INTERNAL_TOKEN,
        'X-Nexus-Delegated-App-Id': 'mcp-user-binding',
    }
    try:
        registered = await client.get('/v1/me', headers=_identity_headers(_USER_STAFF_ID, login_name='app-owner'))
        assert registered.status == 200, await registered.text()

        listed = await client.get('/v1/access/users', headers=admin_headers)
        assert listed.status == 200, await listed.text()
        user = next(item for item in (await listed.json())['items'] if item['principal_id'] == user_id)
        assert user['app_id'] is None

        bound = await client.put(
            f'/v1/access/users/{user_id}/app-binding',
            headers=admin_headers,
            json={'app_id': 'MCP-User-Binding'},
        )
        assert bound.status == 200, await bound.text()
        binding = await bound.json()
        assert binding == {
            'token_id': binding['token_id'],
            'token_kind': 'system_default',
            'owner_principal_id': user_id,
            'environment_id': None,
            'app_id': 'mcp-user-binding',
        }

        listed = await client.get('/v1/access/users', headers=admin_headers)
        user = next(item for item in (await listed.json())['items'] if item['principal_id'] == user_id)
        assert user['app_id'] == 'mcp-user-binding'

        delegated = await client.get('/v1/me', headers=app_headers)
        assert delegated.status == 200, await delegated.text()
        assert (await delegated.json())['principal_id'] == user_id

        unbound = await client.delete(f'/v1/access/users/{user_id}/app-binding', headers=admin_headers)
        assert unbound.status == 200, await unbound.text()
        assert (await unbound.json())['app_id'] is None

        listed = await client.get('/v1/access/users', headers=admin_headers)
        user = next(item for item in (await listed.json())['items'] if item['principal_id'] == user_id)
        assert user['app_id'] is None

        delegated = await client.get('/v1/me', headers=app_headers)
        assert delegated.status == 401
    finally:
        await client.close()


async def test_app_identity_bound_to_api_token_cannot_cross_environment(auth_session_factory, identity_config):
    now = utcnow()
    user_id = 'principal-api-app'
    async with auth_session_factory() as session:
        access = AccessRepository(session)
        await access.upsert_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject='api-app-owner',
            roles={'user'},
            display_name='API App Owner',
            actor='admin',
            now=now,
        )
        environments = EnvironmentRepository(session)
        for environment_id in ('env-api-app-a', 'env-api-app-b'):
            await environments.create_environment(
                environment_id=environment_id,
                name=environment_id,
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={'slots': 2},
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=now,
            )
            await environments.upsert_grant(
                principal_id=user_id,
                environment_id=environment_id,
                limits={'slots': 2},
                scopes={'submission:single'},
                enabled=True,
                actor='admin',
                now=now,
            )
        token, _ = await access.create_token(
            owner_principal_id=user_id,
            environment_id='env-api-app-a',
            scopes={'submission:single'},
            created_by='admin',
            key_id='local',
            key='local-dev-submit-token-key',
            now=now,
        )
        await access.bind_token_app_id(token.token_id, app_id='mcp-env-scoped-app')
        await session.commit()

    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    try:
        response = await client.post(
            '/v1/submissions/single',
            headers=_sgw_headers('mcp-env-scoped-app'),
            json={
                'submission_version': 1,
                'idempotency_key': 'api-app-cross-environment',
                'environment_id': 'env-api-app-b',
                'code': {'mode': 'config_only'},
                'pool': 'existing',
                'run_input': {'task_name': 'must-not-run'},
            },
        )
        assert response.status == 404
        assert (await response.json())['error'] == 'resource_not_found'
    finally:
        await client.close()


async def test_live_compatibility_requires_tai_identity_and_preserves_user_ownership(
    auth_session_factory,
    identity_config,
):
    control = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await control.start_server()
    user_headers = _identity_headers(_USER_STAFF_ID, login_name='compat-user')
    other_headers = _identity_headers(_OTHER_STAFF_ID, login_name='compat-other')
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    try:
        response = await control.get('/v1/me', headers=user_headers)
        assert response.status == 200, await response.text()

        async with auth_session_factory() as session:
            environments = EnvironmentRepository(session)
            await environments.create_environment(
                environment_id='env-live-compat',
                name='Live compatibility',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={'slots': 10},
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=utcnow(),
            )
            await environments.upsert_grant(
                principal_id=user_id,
                environment_id='env-live-compat',
                limits={'slots': 10},
                scopes={'submission:live', 'execution:read', 'execution:command', 'execution:stop'},
                enabled=True,
                actor='admin',
                now=utcnow(),
            )
            await session.commit()

        compatibility = TestClient(
            TestServer(
                create_application_app(
                    session_factory=auth_session_factory,
                    config=identity_config,
                    live_compatibility_control_base_url=str(control.make_url('/')).rstrip('/'),
                    live_compatibility_application_base_url='http://application.invalid',
                    live_compatibility_default_placement={
                        'environment': 'env-live-compat',
                        'pool': 'interactive',
                    },
                )
            )
        )
        await compatibility.start_server()
        try:
            response = await compatibility.get('/compat/live/health')
            assert response.status == 200, await response.text()

            response = await compatibility.post(
                '/compat/live/v1/runs',
                json={'run_id': 'compat-run', 'task_name': 'identity'},
            )
            assert response.status == 401

            response = await compatibility.post(
                '/compat/live/v1/runs',
                headers={**user_headers, 'Authorization': 'Bearer ignored-taihu-pat'},
                json={'run_id': 'compat-run', 'task_name': 'identity'},
            )
            assert response.status == 201, await response.text()
            execution_id = (await response.json())['run_id']

            async with auth_session_factory() as session:
                execution = await session.get(ExecutionRow, execution_id)
                assert execution is not None
                assert execution.owner_principal_id == user_id
                assert execution.environment_id == 'env-live-compat'
                assert execution.pool_id == 'interactive'

            response = await compatibility.get('/compat/live/v1/runs', headers=other_headers)
            assert response.status == 200, await response.text()
            assert (await response.json())['runs'] == []
        finally:
            await compatibility.close()
    finally:
        await control.close()


async def test_live_compatibility_delegates_app_identity(auth_session_factory, identity_config):
    control = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await control.start_server()
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    app_headers = _sgw_headers('mcp-compat-app')
    try:
        registered = await control.get(
            '/v1/me',
            headers=_identity_headers(_USER_STAFF_ID, login_name='compat-app-owner'),
        )
        assert registered.status == 200, await registered.text()

        async with auth_session_factory() as session:
            environments = EnvironmentRepository(session)
            await environments.create_environment(
                environment_id='env-live-app-compat',
                name='Live app compatibility',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={'slots': 10},
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=utcnow(),
            )
            await environments.upsert_grant(
                principal_id=user_id,
                environment_id='env-live-app-compat',
                limits={'slots': 10},
                scopes={'submission:live', 'execution:read', 'execution:command', 'execution:stop'},
                enabled=True,
                actor='admin',
                now=utcnow(),
            )
            token = await AccessRepository(session).ensure_system_token(
                owner_principal_id=user_id,
                created_by='test',
                now=utcnow(),
            )
            await AccessRepository(session).bind_token_app_id(token.token_id, app_id='mcp-compat-app')
            await session.commit()

        compatibility = TestClient(
            TestServer(
                create_application_app(
                    session_factory=auth_session_factory,
                    config=identity_config,
                    live_compatibility_control_base_url=str(control.make_url('/')).rstrip('/'),
                    live_compatibility_application_base_url='http://application.invalid',
                    live_compatibility_default_placement={
                        'environment': 'env-live-app-compat',
                        'pool': 'interactive',
                    },
                )
            )
        )
        await compatibility.start_server()
        try:
            submitted = await compatibility.post(
                '/compat/live/v1/runs',
                headers=app_headers,
                json={'run_id': 'compat-app-run', 'task_name': 'app-identity'},
            )
            assert submitted.status == 201, await submitted.text()
            execution_id = (await submitted.json())['run_id']

            async with auth_session_factory() as session:
                execution = await session.get(ExecutionRow, execution_id)
                assert execution is not None
                assert execution.owner_principal_id == user_id
                assert execution.environment_id == 'env-live-app-compat'
        finally:
            await compatibility.close()
    finally:
        await control.close()


async def test_live_compatibility_websocket_preserves_tai_identity_and_csrf(auth_session_factory, identity_config):
    from aiohttp import web

    forwarded: list[dict[str, str | None]] = []

    async def runtime_socket(request: web.Request) -> web.WebSocketResponse:
        forwarded.append({
            'authorization': request.headers.get('Authorization'),
            'identity': request.headers.get('x-tai-identity'),
            'internal_token': request.headers.get('X-Nexus-Internal-Token'),
            'delegated_principal': request.headers.get('X-Nexus-Delegated-Principal-Id'),
            'protocols': request.headers.get('Sec-WebSocket-Protocol'),
        })
        websocket = web.WebSocketResponse(protocols=('runtime.v1',))
        await websocket.prepare(request)
        await websocket.send_str('ready')
        await websocket.close()
        return websocket

    upstream_app = web.Application()
    upstream_app.router.add_get(
        '/v1/executions/{execution_id}/runtimes/{runtime_name}/proxy/{tail:.*}',
        runtime_socket,
    )
    upstream = TestClient(TestServer(upstream_app))
    await upstream.start_server()
    identity_headers = _identity_headers(_USER_STAFF_ID, login_name='compat-websocket')
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    csrf = csrf_token(
        AuthContext(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=str(_USER_STAFF_ID),
            roles=frozenset({'user'}),
            scopes=frozenset({'*'}),
            auth_type='tai_identity',
        ),
        identity_config,
    )
    assert csrf is not None
    compatibility = TestClient(
        TestServer(
            create_application_app(
                session_factory=auth_session_factory,
                config=identity_config,
                live_compatibility_control_base_url='http://control.invalid',
                live_compatibility_application_base_url=str(upstream.make_url('/')).rstrip('/'),
                live_compatibility_default_placement={
                    'environment': 'env-live-compat',
                    'pool': 'interactive',
                },
            )
        )
    )
    await compatibility.start_server()
    origin = str(compatibility.make_url('/')).rstrip('/')
    protocols = (f'nexus-csrf.{csrf}', 'runtime.v1')
    try:
        websocket = await compatibility.ws_connect(
            '/compat/live/v1/runs/live-1/runtimes/main/proxy/socket',
            headers={
                **identity_headers,
                'Authorization': 'Bearer ignored-taihu-pat',
                'Origin': origin,
                'Sec-Fetch-Site': 'same-origin',
            },
            protocols=protocols,
        )
        message = await websocket.receive()
        assert message.type == aiohttp.WSMsgType.TEXT
        assert message.data == 'ready'
        await websocket.close()
    finally:
        await compatibility.close()
        await upstream.close()

    assert forwarded == [
        {
            'authorization': None,
            'identity': None,
            'internal_token': _INTERNAL_TOKEN,
            'delegated_principal': user_id,
            'protocols': ','.join(protocols),
        }
    ]


async def test_tai_identity_submission_still_requires_matching_environment_grant_scope(auth_session_factory):
    config = DistributedConfig(
        auth_mode='required',
        secrets={'tai_application_token': _TAI_APPLICATION_TOKEN},
    )
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    headers = _identity_headers(_USER_STAFF_ID, login_name='grant-scoped-user')
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    payload = {
        'submission_version': 1,
        'idempotency_key': 'tai-grant-scope',
        'environment_id': 'env-tai-grant',
        'code': {'mode': 'config_only'},
        'pool': 'existing',
        'run_input': {'task_name': 'identity'},
    }
    try:
        response = await client.get('/v1/me', headers=headers)
        assert response.status == 200, await response.text()

        async with auth_session_factory() as session:
            environments = EnvironmentRepository(session)
            await environments.create_environment(
                environment_id='env-tai-grant',
                name='Tai grant scope',
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={'slots': 4},
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=utcnow(),
            )
            await session.commit()

        response = await client.post('/v1/submissions/single', headers=headers, json=payload)
        assert response.status == 404
        assert (await response.json())['error'] == 'environment_grant_not_found'

        async with auth_session_factory() as session:
            await EnvironmentRepository(session).upsert_grant(
                principal_id=user_id,
                environment_id='env-tai-grant',
                limits={'slots': 2},
                scopes={'submission:batch'},
                enabled=True,
                actor='admin',
                now=utcnow(),
            )
            await session.commit()

        response = await client.post('/v1/submissions/single', headers=headers, json=payload)
        assert response.status == 409
        assert (await response.json())['error'] == 'environment_scope_required'

        async with auth_session_factory() as session:
            await EnvironmentRepository(session).upsert_grant(
                principal_id=user_id,
                environment_id='env-tai-grant',
                limits={'slots': 2},
                scopes={'submission:single'},
                enabled=True,
                actor='admin',
                now=utcnow(),
            )
            await session.commit()

        response = await client.post('/v1/submissions/single', headers=headers, json=payload)
        assert response.status == 202, await response.text()
    finally:
        await client.close()


async def test_user_claims_environment_default_quota_once(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID, login_name='quota-user')
    default_quota = {
        'manager_count': 2,
        'cpu_millicores': 10_000,
        'memory_bytes': 4 * 1024**3,
        'slots': 20,
        'active_batches': 1,
    }
    try:
        response = await client.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-default-quota',
                'name': 'Default quota',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': {
                    'manager_count': 20,
                    'cpu_millicores': 100_000,
                    'memory_bytes': 40 * 1024**3,
                    'slots': 200,
                    'active_batches': 10,
                },
                'default_quota': default_quota,
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()

        response = await client.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-admin-only',
                'name': 'Admin only',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': {'slots': 10},
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()

        response = await client.get('/v1/environment-catalog', headers=user_headers)
        assert response.status == 200, await response.text()
        catalog = await response.json()
        assert catalog['system_token'] == {'label': '默认'}
        assert [item['environment_id'] for item in catalog['items']] == ['env-default-quota']
        assert catalog['items'][0]['default_quota'] == default_quota
        assert catalog['items'][0]['claimable'] is True
        assert catalog['items'][0]['grant'] is None

        response = await client.put('/v1/my-environments/env-default-quota/quota', headers=user_headers)
        assert response.status == 200, await response.text()
        claimed = await response.json()
        assert claimed['system_token'] == {'label': '默认'}
        assert claimed['grant']['limits'] == default_quota
        assert claimed['grant']['enabled'] is True
        assert claimed['grant']['scopes'] == [
            'artifact:write',
            'execution:command',
            'execution:read',
            'execution:stop',
            'submission:batch',
            'submission:live',
            'submission:single',
        ]
        first_version = claimed['grant']['version']

        replacement_quota = {**default_quota, 'cpu_millicores': 5_000}
        response = await client.patch(
            '/v1/environments/env-default-quota',
            headers=admin_headers,
            json={'default_quota': replacement_quota},
        )
        assert response.status == 200, await response.text()

        response = await client.put('/v1/my-environments/env-default-quota/quota', headers=user_headers)
        assert response.status == 200, await response.text()
        repeated = await response.json()
        assert repeated['grant']['version'] == first_version
        assert repeated['grant']['limits'] == default_quota

        response = await client.get('/v1/environment-catalog', headers=user_headers)
        assert response.status == 200, await response.text()
        item = (await response.json())['items'][0]
        assert item['default_quota'] == replacement_quota
        assert item['claimable'] is False
        assert item['grant']['limits'] == default_quota
    finally:
        await client.close()


async def test_disabled_auth_can_claim_default_quota_for_devcloud(auth_session_factory):
    client = TestClient(
        TestServer(
            create_control_app(session_factory=auth_session_factory, config=DistributedConfig(auth_mode='disabled'))
        )
    )
    await client.start_server()
    quota = {
        'manager_count': 1,
        'cpu_millicores': 1_000,
        'memory_bytes': 1024**3,
        'slots': 1,
        'active_batches': 1,
    }
    try:
        response = await client.post(
            '/v1/environments',
            json={
                'environment_id': 'env-local-default-quota',
                'name': 'Local default quota',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': quota,
                'default_quota': quota,
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()

        response = await client.put('/v1/my-environments/env-local-default-quota/quota')
        assert response.status == 200, await response.text()
        assert (await response.json())['grant']['limits'] == quota

        async with auth_session_factory() as session:
            principal = await session.get(PrincipalRow, 'local-dev')
            assert principal is not None
            tokens = list(
                await session.scalars(select(SubmitTokenRow).where(SubmitTokenRow.owner_principal_id == 'local-dev'))
            )
            assert len(tokens) == 1
            assert tokens[0].token_kind == 'system_default'
    finally:
        await client.close()


async def test_user_cannot_reenable_disabled_default_quota_grant(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID, login_name='quota-user')
    user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
    quota = {
        'manager_count': 1,
        'cpu_millicores': 1_000,
        'memory_bytes': 1024**3,
        'slots': 1,
        'active_batches': 1,
    }
    try:
        response = await client.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-disabled-grant',
                'name': 'Disabled grant',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': quota,
                'default_quota': quota,
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()
        assert (await client.put('/v1/my-environments/env-disabled-grant/quota', headers=user_headers)).status == 200

        response = await client.put(
            f'/v1/environments/env-disabled-grant/grants/{user_id}',
            headers=admin_headers,
            json={'limits': quota, 'enabled': False},
        )
        assert response.status == 200, await response.text()

        response = await client.put('/v1/my-environments/env-disabled-grant/quota', headers=user_headers)
        assert response.status == 409
        assert (await response.json())['error'] == 'environment_grant_disabled'
    finally:
        await client.close()


async def test_concurrent_default_quota_claims_create_one_grant(auth_session_factory, identity_config):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID, login_name='quota-user')
    quota = {
        'manager_count': 1,
        'cpu_millicores': 1_000,
        'memory_bytes': 1024**3,
        'slots': 1,
        'active_batches': 1,
    }
    try:
        response = await client.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-concurrent-claim',
                'name': 'Concurrent claim',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': quota,
                'default_quota': quota,
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()

        responses = await asyncio.gather(
            *(client.put('/v1/my-environments/env-concurrent-claim/quota', headers=user_headers) for _ in range(4))
        )
        assert [response.status for response in responses] == [200, 200, 200, 200]
        assert {int((await response.json())['grant']['version']) for response in responses} == {1}

        user_id = principal_id('tai', 'default', str(_USER_STAFF_ID))
        async with auth_session_factory() as session:
            grants = list(
                await session.scalars(
                    select(EnvironmentGrantRow).where(
                        EnvironmentGrantRow.principal_id == user_id,
                        EnvironmentGrantRow.environment_id == 'env-concurrent-claim',
                    )
                )
            )
            usage = list(
                await session.scalars(
                    select(EnvironmentResourceUsageRow).where(
                        EnvironmentResourceUsageRow.principal_id == user_id,
                        EnvironmentResourceUsageRow.environment_id == 'env-concurrent-claim',
                    )
                )
            )
            audits = list(
                await session.scalars(
                    select(AuditRow).where(
                        AuditRow.actor_principal_id == user_id,
                        AuditRow.operation == 'environment.quota.claim',
                    )
                )
            )
        assert len(grants) == 1
        assert len(usage) == 1
        assert len(audits) == 1
    finally:
        await client.close()


async def test_concurrent_first_tai_identity_requests_register_one_user(auth_session_factory):
    config = DistributedConfig(
        auth_mode='required',
        secrets={'tai_application_token': _TAI_APPLICATION_TOKEN},
    )
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    headers = _identity_headers(_USER_STAFF_ID, login_name='concurrent-visitor')
    try:
        responses = await asyncio.gather(*(client.get('/v1/me', headers=headers) for _ in range(4)))

        assert [response.status for response in responses] == [200, 200, 200, 200]
        assert {str((await response.json())['principal_id']) for response in responses} == {
            principal_id('tai', 'default', str(_USER_STAFF_ID))
        }
        async with auth_session_factory() as session:
            users = await AccessRepository(session).list_users()
            assert [(user.subject, user.roles) for user in users] == [(str(_USER_STAFF_ID), frozenset({'user'}))]
            tokens = list(await session.scalars(select(SubmitTokenRow)))
            assert [(token.owner_principal_id, token.token_kind) for token in tokens] == [
                (principal_id('tai', 'default', str(_USER_STAFF_ID)), 'system_default')
            ]
    finally:
        await client.close()


def test_csrf_uses_tai_application_token_without_submit_token_keyring() -> None:
    config = DistributedConfig(
        auth_mode='required',
        secrets={
            'tai_application_token': _TAI_APPLICATION_TOKEN,
            'submit_token_hash_keys': {},
        },
    )
    context = AuthContext(
        principal_id='principal-a',
        issuer='tai',
        tenant='default',
        subject='1001',
        roles=frozenset({'user'}),
        scopes=frozenset({'*'}),
        auth_type='tai_identity',
    )

    assert csrf_token(context, config) is not None


async def test_nexus_api_token_routes_are_not_exposed(auth_session_factory, identity_config) -> None:
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    try:
        responses = (
            await client.get('/v1/access/tokens', headers=headers),
            await client.post('/v1/access/tokens', headers=headers, json={}),
            await client.delete('/v1/access/tokens/tok-example', headers=headers),
        )
        assert [response.status for response in responses] == [404, 404, 404]
    finally:
        await client.close()


async def test_ensure_user_does_not_downgrade_existing_admin(auth_session_factory):
    subject = str(_ADMIN_STAFF_ID)
    user_id = principal_id('tai', 'default', subject)
    async with auth_session_factory() as session:
        repository = AccessRepository(session)
        await repository.upsert_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user', 'admin'},
            display_name='configured-admin',
            actor='system:bootstrap',
            now=utcnow(),
        )

        user = await repository.ensure_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user'},
            display_name='identity-login-name',
            actor='system:tai-identity',
            now=utcnow(),
        )
        await session.commit()

    assert user.roles == frozenset({'user', 'admin'})
    assert user.display_name == 'configured-admin'


async def test_ensure_user_fills_missing_display_name_without_changing_roles(auth_session_factory):
    subject = str(_ADMIN_STAFF_ID)
    user_id = principal_id('tai', 'default', subject)
    async with auth_session_factory() as session:
        repository = AccessRepository(session)
        await repository.upsert_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user', 'admin'},
            display_name=None,
            actor='system:bootstrap',
            now=utcnow(),
        )

        user = await repository.ensure_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user'},
            display_name='first-visit-login',
            actor='system:tai-identity',
            now=utcnow(),
        )
        await session.commit()

    assert user.roles == frozenset({'user', 'admin'})
    assert user.display_name == 'first-visit-login'


async def test_admin_bootstrap_preserves_display_name_registered_by_tai_identity(auth_session_factory):
    subject = str(_ADMIN_STAFF_ID)
    user_id = principal_id('tai', 'default', subject)
    async with auth_session_factory() as session:
        await AccessRepository(session).upsert_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user'},
            display_name='first-visit-login',
            actor='system:tai-identity',
            now=utcnow(),
        )
        await session.commit()

    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=_identity_config())))
    await client.start_server()
    await client.close()

    async with auth_session_factory() as session:
        user = await AccessRepository(session).get_user(user_id)

    assert user is not None
    assert user.roles == frozenset({'user', 'admin'})
    assert user.display_name == 'first-visit-login'


async def test_db_admin_users_snapshot_drives_control_bootstrap(auth_session_factory):
    subject = str(_ADMIN_STAFF_ID)
    user_id = principal_id('tai', 'default', subject)
    async with auth_session_factory() as session:
        await AccessRepository(session).upsert_user(
            principal_id=user_id,
            issuer='tai',
            tenant='default',
            subject=subject,
            roles={'user'},
            display_name='db-config-admin',
            actor='system:tai-identity',
            now=utcnow(),
        )
        await set_config_override(
            session,
            role='*',
            identity='*',
            values={
                'admin_users': [f'tai/default/{subject}'],
                'auth_mode': 'required',
            },
            expected_version=0,
            actor='system:test-bootstrap',
        )
        await session.commit()

    config = await load_distributed_config(
        auth_session_factory,
        database_url='sqlite+aiosqlite:///bootstrap.db',
        role='control-gateway',
        secret_overrides={
            'tai_application_token': _TAI_APPLICATION_TOKEN,
        },
    )
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    await client.close()

    async with auth_session_factory() as session:
        user = await AccessRepository(session).get_user(user_id)

    assert user is not None
    assert user.roles == frozenset({'user', 'admin'})
    assert user.display_name == 'db-config-admin'


def test_identity_browser_websocket_requires_origin_and_csrf_subprotocol() -> None:
    from aiohttp import web

    config = _identity_config()
    context = AuthContext(
        principal_id='principal-a',
        issuer='tai',
        tenant='tenant',
        subject='user-a',
        roles=frozenset({'user'}),
        scopes=frozenset({'*'}),
        auth_type='tai_identity',
    )
    expected_token = csrf_token(context, config)
    assert expected_token is not None

    def websocket_request(headers):
        return SimpleNamespace(
            method='GET',
            scheme='https',
            host='console.example',
            headers={'Upgrade': 'websocket', **headers},
        )

    with pytest.raises(web.HTTPForbidden, match='Forbidden') as missing_origin:
        _enforce_browser_origin(websocket_request({}), context=context, config=config)
    assert missing_origin.value.text == 'browser_origin_required'

    origin_headers = {'Origin': 'https://console.example', 'Sec-Fetch-Site': 'same-origin'}
    with pytest.raises(web.HTTPForbidden, match='Forbidden') as missing_token:
        _enforce_browser_origin(websocket_request(origin_headers), context=context, config=config)
    assert missing_token.value.text == 'csrf_token_required'

    with pytest.raises(web.HTTPForbidden, match='Forbidden') as wrong_token:
        _enforce_browser_origin(
            websocket_request({**origin_headers, 'Sec-WebSocket-Protocol': 'nexus-csrf.wrong,runtime.v1'}),
            context=context,
            config=config,
        )
    assert wrong_token.value.text == 'invalid_csrf_token'

    _enforce_browser_origin(
        websocket_request({**origin_headers, 'Sec-WebSocket-Protocol': f'nexus-csrf.{expected_token},runtime.v1'}),
        context=context,
        config=config,
    )


async def test_identity_browser_mutations_require_principal_bound_csrf_token(
    auth_session_factory,
    identity_config,
):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    identity_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    browser_headers = {
        **identity_headers,
        'Origin': str(client.make_url('/')).rstrip('/'),
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    payload = {
        'environment_id': 'env-csrf',
        'name': 'CSRF Environment',
        'allocation_type': 'predefined',
        'provider_type': 'predefined',
        'resource_limits': {'slots': 1},
        'status': 'enabled',
    }
    try:
        response = await client.get('/v1/me', headers=identity_headers)
        assert response.status == 200
        csrf_token = (await response.json())['csrf_token']

        response = await client.post('/v1/environments', headers=browser_headers, json=payload)
        assert response.status == 403
        assert await response.json() == {
            'error': 'csrf_token_required',
            'retryable': False,
        }

        response = await client.post(
            '/v1/environments',
            headers={**browser_headers, 'X-Nexus-CSRF-Token': 'wrong'},
            json=payload,
        )
        assert response.status == 403
        assert await response.json() == {
            'error': 'invalid_csrf_token',
            'retryable': False,
        }

        response = await client.post(
            '/v1/environments',
            headers={**browser_headers, 'X-Nexus-CSRF-Token': csrf_token},
            json=payload,
        )
        assert response.status == 201, await response.text()
    finally:
        await client.close()


async def _create_user(client: TestClient, *, subject: str, admin_headers: dict[str, str]) -> str:
    user_id = principal_id('tai', 'default', subject)
    response = await client.put(
        f'/v1/access/users/{user_id}/roles',
        headers=admin_headers,
        json={'issuer': 'tai', 'tenant': 'default', 'subject': subject, 'roles': ['user']},
    )
    assert response.status == 200, await response.text()
    return user_id


async def _create_kubernetes_provider_config(
    client: TestClient,
    *,
    provider_config_ref: str,
    admin_headers: dict[str, str],
) -> None:
    response = await client.post(
        '/v1/provider-configs',
        headers=admin_headers,
        json={
            'provider_config_ref': provider_config_ref,
            'config': {
                'provider_type': 'kubernetes',
                'namespace': 'default',
                'auth_type': 'in_cluster',
            },
            'status': 'enabled',
        },
    )
    assert response.status == 201, await response.text()


async def test_admin_manages_environments_and_grants_with_owner_visibility_and_audit(
    auth_session_factory,
    identity_config,
):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    await client.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID)
    try:
        user_id = await _create_user(client, subject=str(_USER_STAFF_ID), admin_headers=admin_headers)
        other_id = await _create_user(client, subject=str(_OTHER_STAFF_ID), admin_headers=admin_headers)
        await _create_kubernetes_provider_config(
            client,
            provider_config_ref='kubernetes/kind',
            admin_headers=admin_headers,
        )

        response = await client.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-kind',
                'name': 'Kind',
                'allocation_type': 'dynamic',
                'provider_type': 'kubernetes',
                'provider_config_ref': 'kubernetes/kind',
                'resource_limits': {
                    'manager_count': 10,
                    'cpu_millicores': 40_000,
                    'memory_bytes': 80 * 1024**3,
                    'slots': 100,
                    'active_batches': 2,
                },
                'default_timeouts': {'queue_ttl_seconds': 300},
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()
        assert (await response.json())['config_version'] == 1

        response = await client.put(
            f'/v1/environments/env-kind/grants/{user_id}',
            headers=admin_headers,
            json={
                'limits': {'slots': 20, 'active_batches': 1},
                'scopes': ['submission:batch', 'artifact:write', 'execution:read', 'execution:stop'],
                'enabled': True,
            },
        )
        assert response.status == 200, await response.text()
        grant_payload = await response.json()
        assert grant_payload['scopes'] == [
            'artifact:write',
            'execution:read',
            'execution:stop',
            'submission:batch',
        ]

        response = await client.get('/v1/my-environments', headers=user_headers)
        assert response.status == 200
        payload = await response.json()
        assert [item['environment_id'] for item in payload['items']] == ['env-kind']
        assert payload['items'][0]['grant']['principal_id'] == user_id
        assert payload['items'][0]['grant']['scopes'] == grant_payload['scopes']

        response = await client.get('/v1/my-environments', headers=_identity_headers(_OTHER_STAFF_ID))
        assert response.status == 200
        assert (await response.json())['items'] == []

        response = await client.get('/v1/environments', headers=user_headers)
        assert response.status == 403
        response = await client.get('/v1/environments', headers=admin_headers)
        assert response.status == 200
        assert [item['environment_id'] for item in (await response.json())['items']] == ['env-kind']

        response = await client.get(f'/v1/environments/env-kind/grants?principal_id={other_id}', headers=admin_headers)
        assert response.status == 200
        assert (await response.json())['items'] == []

        response = await client.patch(
            '/v1/environments/env-kind',
            headers=admin_headers,
            json={'default_timeouts': {'queue_ttl_seconds': 600}},
        )
        assert response.status == 200
        assert (await response.json())['config_version'] == 2

        response = await client.post('/v1/environments/env-kind/disable', headers=admin_headers)
        assert response.status == 200
        assert (await response.json())['status'] == 'disabled'

        response = await client.get('/v1/access/audit', headers=admin_headers)
        operations = {item['operation'] for item in (await response.json())['items']}
        assert {
            'environment.create',
            'environment.update',
            'environment.disable',
            'environment.grant.upsert',
        } <= operations
    finally:
        await client.close()


async def test_typed_submit_uses_identity_grants_and_returns_idempotent_202_envelope(
    auth_session_factory,
    identity_config,
):
    application = TestClient(
        TestServer(create_application_app(session_factory=auth_session_factory, config=identity_config))
    )
    await application.start_server()
    control_config = identity_config.model_copy(
        update={
            'application_public_base_url': str(application.make_url('/')).rstrip('/'),
            'console_public_base_url': 'https://console.example/',
            'control_public_base_url': 'http://127.0.0.1',
        }
    )
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=control_config)))
    await client.start_server()
    control_config.control_public_base_url = str(client.make_url('/')).rstrip('/')
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID)
    try:
        user_id = await _create_user(client, subject=str(_USER_STAFF_ID), admin_headers=admin_headers)
        for environment_id in ('env-submit', 'env-other'):
            await _create_kubernetes_provider_config(
                client,
                provider_config_ref=f'kubernetes/{environment_id}',
                admin_headers=admin_headers,
            )
            response = await client.post(
                '/v1/environments',
                headers=admin_headers,
                json={
                    'environment_id': environment_id,
                    'name': environment_id,
                    'allocation_type': 'dynamic',
                    'provider_type': 'kubernetes',
                    'provider_config_ref': f'kubernetes/{environment_id}',
                    'resource_limits': {
                        'manager_count': 4,
                        'cpu_millicores': 8_000,
                        'memory_bytes': 16 * 1024**3,
                        'slots': 4,
                    },
                    'default_timeouts': {'queue_ttl_seconds': 300, 'stop_timeout_seconds': 30},
                    'status': 'enabled',
                },
            )
            assert response.status == 201
            response = await client.put(
                f'/v1/environments/{environment_id}/grants/{user_id}',
                headers=admin_headers,
                json={
                    'limits': {
                        'manager_count': 2,
                        'cpu_millicores': 4_000,
                        'memory_bytes': 8 * 1024**3,
                        'slots': 2,
                    },
                    'enabled': True,
                },
            )
            assert response.status == 200

        payload = {
            'submission_version': 1,
            'idempotency_key': 'http-single',
            'environment_id': 'env-submit',
            'code': {'mode': 'config_only'},
            'resources': {
                'managers': {'count': 1, 'cpu_cores': 1, 'memory_gib': 2, 'slots': 1},
            },
            'run_input': {'task_name': 'identity'},
        }

        response = await client.post('/v1/submissions/single', headers=user_headers, json=payload)
        assert response.status == 202, await response.text()
        created = await response.json()
        assert set(created) == {
            'execution_id',
            'status',
            'phase',
            'reused',
            'control_url',
            'result_url',
            'task_web_ui',
            'environment_web_ui',
            'tracedb_web_ui',
        }
        assert created['status'] == 'queued'
        assert created['phase'] == 'waiting_resources'
        assert created['reused'] is False
        assert (
            created['control_url']
            == f'{control_config.control_public_base_url}/v1/executions/{created["execution_id"]}'
        )
        assert created['result_url'].endswith(f'/v1/executions/{created["execution_id"]}/result')
        assert created['task_web_ui'] == (f'https://console.example/v1/console/executions/{created["execution_id"]}')
        assert created['environment_web_ui'] == 'https://console.example/v1/console/my-environments'
        assert created['tracedb_web_ui'] == 'https://console.example/v1/console/my-environments'
        async with aiohttp.ClientSession() as session:
            async with session.get(created['control_url'], headers=user_headers) as linked_response:
                assert linked_response.status == 200
            async with session.get(created['result_url'], headers=user_headers) as linked_response:
                assert linked_response.status == 200

        response = await client.post('/v1/submissions/single', headers=user_headers, json=payload)
        assert response.status == 202
        repeated = await response.json()
        assert repeated['execution_id'] == created['execution_id']
        assert repeated['reused'] is True

        concurrent_payload = {**payload, 'idempotency_key': 'http-single-concurrent'}
        concurrent_responses = await asyncio.gather(
            client.post('/v1/submissions/single', headers=user_headers, json=concurrent_payload),
            client.post('/v1/submissions/single', headers=user_headers, json=concurrent_payload),
        )
        assert [response.status for response in concurrent_responses] == [202, 202]
        concurrent_results = [await response.json() for response in concurrent_responses]
        assert len({result['execution_id'] for result in concurrent_results}) == 1
        assert sorted(result['reused'] for result in concurrent_results) == [False, True]

        conflicting = {**payload, 'run_input': {'task_name': 'changed'}}
        response = await client.post('/v1/submissions/single', headers=user_headers, json=conflicting)
        assert response.status == 409
        assert (await response.json())['error'] == 'submission_idempotency_conflict'

        other_environment = {**payload, 'idempotency_key': 'other-environment', 'environment_id': 'env-other'}
        response = await client.post('/v1/submissions/single', headers=user_headers, json=other_environment)
        assert response.status == 202, await response.text()

        response = await client.put(
            f'/v1/environments/env-submit/grants/{user_id}',
            headers=admin_headers,
            json={
                'limits': {
                    'manager_count': 2,
                    'cpu_millicores': 4_000,
                    'memory_bytes': 8 * 1024**3,
                    'slots': 2,
                },
                'scopes': ['submission:single'],
                'enabled': True,
            },
        )
        assert response.status == 200, await response.text()

        response = await client.get('/v1/executions', headers=user_headers)
        assert response.status == 200
        assert all(item['execution_id'] != created['execution_id'] for item in (await response.json())['items'])

        response = await client.get(f'/v1/executions/{created["execution_id"]}', headers=user_headers)
        assert response.status == 409
        assert (await response.json())['error'] == 'environment_scope_required'

        response = await client.get('/v1/me', headers=user_headers)
        assert response.status == 200
        assert (await response.json())['scopes'] == ['*']
    finally:
        await application.close()
        await client.close()


async def test_artifact_upload_requires_explicit_granted_environment(
    auth_session_factory,
    identity_config,
):
    config = identity_config
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    await client.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    user_headers = _identity_headers(_USER_STAFF_ID)
    try:
        user_id = await _create_user(client, subject=str(_USER_STAFF_ID), admin_headers=admin_headers)
        for environment_id in ('env-artifact', 'env-artifact-other'):
            response = await client.post(
                '/v1/environments',
                headers=admin_headers,
                json={
                    'environment_id': environment_id,
                    'name': environment_id,
                    'allocation_type': 'predefined',
                    'provider_type': 'predefined',
                    'resource_limits': {
                        'slots': 1,
                        **({'max_artifact_size_bytes': 10} if environment_id == 'env-artifact' else {}),
                    },
                    'status': 'enabled',
                },
            )
            assert response.status == 201
            if environment_id == 'env-artifact':
                response = await client.put(
                    f'/v1/environments/{environment_id}/grants/{user_id}',
                    headers=admin_headers,
                    json={'limits': {'slots': 1}, 'enabled': True},
                )
                assert response.status == 200

        response = await client.post(
            '/v1/artifact-uploads',
            headers=user_headers,
            json={
                'kind': 'source_package',
                'sha256': 'c' * 64,
                'size_bytes': 1,
                'idempotency_key': 'artifact-environment-required',
            },
        )
        assert response.status == 422

        response = await client.post(
            '/v1/artifact-uploads',
            headers=user_headers,
            json={
                'kind': 'source_package',
                'sha256': 'b' * 64,
                'size_bytes': 11,
                'idempotency_key': 'artifact-too-large',
                'environment_id': 'env-artifact',
            },
        )
        assert response.status == 413

        upload_payload = {
            'kind': 'source_package',
            'sha256': 'a' * 64,
            'size_bytes': 10,
            'idempotency_key': 'artifact-upload',
            'environment_id': 'env-artifact',
        }
        response = await client.post('/v1/artifact-uploads', headers=user_headers, json=upload_payload)
        assert response.status == 201, await response.text()
        created = await response.json()
        claims = UploadTicketSigner(
            keys=config.artifact_ticket_keyring(),
            current_key_id=config.artifact_upload_ticket_current_key_id,
            ttl=config.artifact_upload_ticket_ttl,
        ).verify(created['upload_ticket'], now=utcnow())
        assert claims.environment_id == 'env-artifact'
        async with auth_session_factory() as session:
            artifact = await session.get(ArtifactRow, created['artifact']['artifact_id'])
            assert artifact.environment_id == 'env-artifact'

        response = await client.post(
            '/v1/artifact-uploads',
            headers=user_headers,
            json={**upload_payload, 'idempotency_key': 'wrong-environment', 'environment_id': 'env-artifact-other'},
        )
        assert response.status == 404
        assert (await response.json())['error'] == 'environment_grant_not_found'
    finally:
        await client.close()


async def test_non_owner_execution_surfaces_return_404_and_owner_stop_is_audited(
    auth_session_factory,
    identity_config,
):
    config = identity_config
    control = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=config)))
    application = TestClient(TestServer(create_application_app(session_factory=auth_session_factory, config=config)))
    await control.start_server()
    await application.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    try:
        subject_staff_ids = {'owner': 1003, 'other': 1004}
        user_ids = {
            subject: await _create_user(control, subject=str(staff_id), admin_headers=admin_headers)
            for subject, staff_id in subject_staff_ids.items()
        }
        response = await control.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-owner-visibility',
                'name': 'Owner visibility',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': {'slots': 4},
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()

        identity_headers = {}
        for subject, user_id in user_ids.items():
            response = await control.put(
                f'/v1/environments/env-owner-visibility/grants/{user_id}',
                headers=admin_headers,
                json={'limits': {'slots': 2}, 'enabled': True},
            )
            assert response.status == 200, await response.text()
            identity_headers[subject] = _identity_headers(subject_staff_ids[subject], login_name=subject)

        response = await control.post(
            '/v1/submissions/single',
            headers=identity_headers['owner'],
            json={
                'submission_version': 1,
                'idempotency_key': 'owner-visibility',
                'environment_id': 'env-owner-visibility',
                'code': {'mode': 'config_only'},
                'pool': 'existing',
                'run_input': {'task_name': 'identity'},
            },
        )
        assert response.status == 202, await response.text()
        execution_id = (await response.json())['execution_id']

        denied_requests = [
            control.get(f'/v1/executions/{execution_id}', headers=identity_headers['other']),
            control.get(f'/v1/executions/{execution_id}/manager-logs', headers=identity_headers['other']),
            control.get(f'/v1/executions/{execution_id}/batch', headers=identity_headers['other']),
            control.post(f'/v1/executions/{execution_id}/stop', headers=identity_headers['other']),
            application.get(f'/v1/executions/{execution_id}/result', headers=identity_headers['other']),
            application.post(
                f'/v1/executions/{execution_id}/commands',
                headers=identity_headers['other'],
                json={'type': 'input.submit', 'payload': {'text': 'hidden'}},
            ),
            application.get(f'/v1/executions/{execution_id}/proxy/info', headers=identity_headers['other']),
        ]
        denied_responses = await asyncio.gather(*denied_requests)
        assert [response.status for response in denied_responses] == [404] * len(denied_responses)
        denied_texts = await asyncio.gather(*(response.text() for response in denied_responses))
        assert all(execution_id not in text for text in denied_texts)

        response = await control.get(f'/v1/executions/{execution_id}', headers=identity_headers['owner'])
        assert response.status == 200
        response = await application.get(f'/v1/executions/{execution_id}/result', headers=identity_headers['owner'])
        assert response.status == 200
        response = await control.post(f'/v1/executions/{execution_id}/stop', headers=identity_headers['owner'])
        assert response.status == 200, await response.text()

        response = await control.get('/v1/access/audit', headers=admin_headers)
        assert response.status == 200
        stop_audit = next(
            item
            for item in (await response.json())['items']
            if item['operation'] == 'execution.stop' and item['resource_id'] == execution_id
        )
        assert stop_audit['actor_principal_id'] == user_ids['owner']
        assert stop_audit['target_owner_principal_id'] == user_ids['owner']
    finally:
        await application.close()
        await control.close()


async def test_principal_grants_allow_page_reads_and_control_without_changing_actor(
    auth_session_factory,
    identity_config,
):
    control = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=identity_config)))
    application = TestClient(
        TestServer(create_application_app(session_factory=auth_session_factory, config=identity_config))
    )
    await control.start_server()
    await application.start_server()
    admin_headers = _identity_headers(_ADMIN_STAFF_ID, login_name='admin')
    staff_ids = {'owner-a': 1011, 'owner-b': 1012, 'grantee': 1013, 'hidden': 1014}
    headers = {name: _identity_headers(staff_id, login_name=name) for name, staff_id in staff_ids.items()}
    try:
        principal_ids = {
            name: await _create_user(control, subject=str(staff_id), admin_headers=admin_headers)
            for name, staff_id in staff_ids.items()
        }
        response = await control.post(
            '/v1/environments',
            headers=admin_headers,
            json={
                'environment_id': 'env-principal-sharing',
                'name': 'Principal sharing',
                'allocation_type': 'predefined',
                'provider_type': 'predefined',
                'resource_limits': {'slots': 8},
                'status': 'enabled',
            },
        )
        assert response.status == 201, await response.text()
        for principal_id in principal_ids.values():
            response = await control.put(
                f'/v1/environments/env-principal-sharing/grants/{principal_id}',
                headers=admin_headers,
                json={'limits': {'slots': 2}, 'enabled': True},
            )
            assert response.status == 200, await response.text()

        execution_ids = {}
        for name in staff_ids:
            response = await control.post(
                '/v1/submissions/single',
                headers=headers[name],
                json={
                    'submission_version': 1,
                    'idempotency_key': f'principal-sharing-{name}',
                    'environment_id': 'env-principal-sharing',
                    'code': {'mode': 'config_only'},
                    'pool': 'existing',
                    'run_input': {'task_name': 'identity'},
                },
            )
            assert response.status == 202, await response.text()
            execution_ids[name] = (await response.json())['execution_id']

        response = await control.put(
            f'/v1/access/principal-grants/{principal_ids["owner-a"]}/{principal_ids["grantee"]}',
            headers=headers['owner-a'],
        )
        assert response.status == 200, await response.text()
        response = await control.put(
            f'/v1/access/principal-grants/{principal_ids["owner-b"]}/{principal_ids["grantee"]}',
            headers=admin_headers,
        )
        assert response.status == 200, await response.text()

        response = await control.get('/v1/access/users', headers=headers['grantee'])
        assert response.status == 200, await response.text()
        directory = {item['principal_id']: item for item in (await response.json())['items']}
        assert set(principal_ids.values()) <= directory.keys()
        assert all(set(item) == {'display_name', 'issuer', 'principal_id', 'subject'} for item in directory.values())
        assert directory[principal_ids['owner-a']]['display_name'] == 'owner-a'

        denied = await control.put(
            f'/v1/access/principal-grants/{principal_ids["owner-b"]}/{principal_ids["hidden"]}',
            headers=headers['grantee'],
        )
        assert denied.status == 404

        response = await control.post(
            '/v1/submissions/single',
            headers=headers['grantee'],
            json={
                'submission_version': 1,
                'idempotency_key': 'principal-sharing-grantee-after-grant',
                'environment_id': 'env-principal-sharing',
                'code': {'mode': 'config_only'},
                'pool': 'existing',
                'run_input': {'task_name': 'identity'},
            },
        )
        assert response.status == 202, await response.text()
        grantee_execution_id = (await response.json())['execution_id']
        response = await control.get(f'/v1/executions/{grantee_execution_id}', headers=headers['grantee'])
        assert response.status == 200, await response.text()
        assert (await response.json())['owner_principal_id'] == principal_ids['grantee']

        response = await control.get('/v1/access/principal-grants', headers=headers['grantee'])
        assert response.status == 200, await response.text()
        grant_pairs = {
            (item['owner_principal_id'], item['grantee_principal_id']) for item in (await response.json())['items']
        }
        assert grant_pairs == set()

        response = await control.get('/v1/access/principal-grants', headers=headers['owner-a'])
        assert response.status == 200, await response.text()
        grant_pairs = {
            (item['owner_principal_id'], item['grantee_principal_id']) for item in (await response.json())['items']
        }
        assert grant_pairs == {
            (principal_ids['owner-a'], principal_ids['grantee']),
        }

        response = await control.get('/v1/access/audit', headers=admin_headers)
        assert response.status == 200
        grant_audits = [
            item for item in (await response.json())['items'] if item['operation'] == 'principal_access.grant'
        ]
        assert len(grant_audits) == 2
        assert all(len(item['resource_id']) <= 128 for item in grant_audits)
        assert {item['target_owner_principal_id'] for item in grant_audits} == {
            principal_ids['owner-a'],
            principal_ids['owner-b'],
        }

        response = await control.get(
            f'/v1/access/users/{principal_ids["owner-a"]}/quota',
            headers=headers['grantee'],
        )
        assert response.status == 404

        response = await control.get('/v1/executions?summary=1', headers=headers['grantee'])
        assert response.status == 200, await response.text()
        visible_execution_ids = {item['execution_id'] for item in (await response.json())['items']}
        assert visible_execution_ids == {
            execution_ids['owner-a'],
            execution_ids['owner-b'],
            execution_ids['grantee'],
            grantee_execution_id,
        }
        response = await control.get(
            f'/v1/executions?summary=1&owner={principal_ids["owner-a"]}',
            headers=headers['grantee'],
        )
        assert response.status == 200, await response.text()
        assert {item['execution_id'] for item in (await response.json())['items']} == {
            execution_ids['owner-a'],
        }

        response = await control.get(
            f'/v1/executions/{execution_ids["owner-a"]}',
            headers=headers['grantee'],
        )
        assert response.status == 200, await response.text()
        response = await application.get(
            f'/v1/executions/{execution_ids["owner-a"]}/result',
            headers=headers['grantee'],
        )
        assert response.status == 200, await response.text()
        response = await control.post(
            f'/v1/executions/{execution_ids["owner-a"]}/stop',
            headers=headers['grantee'],
        )
        assert response.status == 200, await response.text()
        response = await control.get(
            f'/v1/executions/{execution_ids["hidden"]}',
            headers=headers['grantee'],
        )
        assert response.status == 404

        response = await control.get('/v1/access/audit', headers=admin_headers)
        assert response.status == 200
        stop_audit = next(
            item
            for item in (await response.json())['items']
            if item['operation'] == 'execution.stop' and item['resource_id'] == execution_ids['owner-a']
        )
        assert stop_audit['actor_principal_id'] == principal_ids['grantee']
        assert stop_audit['target_owner_principal_id'] == principal_ids['owner-a']

        revoked = await control.delete(
            f'/v1/access/principal-grants/{principal_ids["owner-a"]}/{principal_ids["grantee"]}',
            headers=admin_headers,
        )
        assert revoked.status == 204, await revoked.text()
        response = await control.get(
            f'/v1/executions/{execution_ids["owner-a"]}',
            headers=headers['grantee'],
        )
        assert response.status == 404
    finally:
        await application.close()
        await control.close()


@pytest.mark.parametrize('path', ['/v1/system/summary', '/v1/system/status', '/v1/system/metrics', '/v1/system/metrics?catalog=1'])
async def test_platform_status_authorization_precedes_warm_cache(auth_session_factory, path):
    client = TestClient(TestServer(create_control_app(session_factory=auth_session_factory, config=_identity_config())))
    await client.start_server()
    try:
        warm = await client.get(path, headers=_identity_headers(_ADMIN_STAFF_ID, login_name='admin'))
        assert warm.status == 200
        response = await client.get(path, headers=_identity_headers(_USER_STAFF_ID, login_name='ordinary-user'))
        assert response.status == 403
        assert await response.json() == {'error': 'admin_required', 'retryable': False}
    finally:
        await client.close()
