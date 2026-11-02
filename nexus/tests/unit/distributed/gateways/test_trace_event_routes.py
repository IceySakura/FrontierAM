from __future__ import annotations

import json

from datetime import timedelta

import pytest

from aiohttp.test_utils import TestClient, TestServer
from pydantic import SecretStr
from tests.unit.distributed.test_cloud_submit_auth_integration import _identity_headers, _sgw_headers
from tests.unit.distributed.test_principal_access_grants import _create_user
from tests.unit.distributed.test_repositories import _create_execution
from tests.unit.distributed.test_repositories import session_factory as session_factory
from tests.unit.distributed.test_trace_read_service import Source, configuration, event

from nexus.distributed.access.taihu import principal_id
from nexus.distributed.gateways.application.app import create_application_app
from nexus.distributed.gateways.application.routes.trace_events import TRACE_READ_SERVICE_KEY
from nexus.distributed.persistence.orm import AssignmentRow, EnvironmentGrantRow, ExecutionRow
from nexus.distributed.repositories.access import AccessRepository
from nexus.distributed.repositories.environment import EnvironmentRepository
from nexus.distributed.trace.bindings import make_source_binding
from nexus.distributed.trace.models import TraceSourceIdentity
from nexus.distributed.utils.clock import utcnow
from nexus.runner.trace import principal_trace_scope_id


OWNER = principal_id('tai', 'default', '1001')
READER = principal_id('tai', 'default', '1002')
LIST_URL = '/v1/executions/exec/trace-streams'


@pytest.fixture
async def gateway(session_factory, tmp_path, monkeypatch):
    config = configuration()
    config.auth_mode = 'required'
    config.secrets.tai_application_token = SecretStr('a' * 32)
    config.artifact_local_root = str(tmp_path / 'artifacts')
    now = utcnow()
    async with session_factory() as session:
        access = AccessRepository(session)
        environments = EnvironmentRepository(session)
        for user in (OWNER, READER):
            await _create_user(access, user, now)
        for env in ('env-1', 'env-2'):
            await environments.create_environment(
                environment_id=env,
                name=env,
                allocation_type='predefined',
                provider_type='predefined',
                provider_config_ref=None,
                resource_limits={'slots': 2},
                default_timeouts={},
                status='enabled',
                actor='admin',
                now=now,
            )
            for user in (OWNER, READER):
                await environments.upsert_grant(
                    principal_id=user,
                    environment_id=env,
                    limits={'slots': 2},
                    enabled=True,
                    actor='admin',
                    now=now,
                    scopes={'execution:read', 'artifact:write'},
                )
        await access.put_principal_access_grant(
            owner_principal_id=OWNER, grantee_principal_id=READER, actor=OWNER, now=now
        )
        tokens = {}
        for name, scopes, env in [
            ('good', {'execution:read'}, 'env-1'),
            ('wrong-env', {'execution:read'}, 'env-2'),
            ('wrong-scope', {'artifact:write'}, 'env-1'),
        ]:
            row, token = await access.create_token(
                owner_principal_id=OWNER,
                scopes=scopes,
                environment_id=env,
                created_by=READER,
                key_id='local',
                key='local-dev-submit-token-key',
                now=now,
                expires_at=now + timedelta(hours=1),
            )
            await access.bind_token_app_id(row.token_id, app_id=f'trace-test-{name}')
            tokens[name] = (row.token_id, token)
        await session.commit()
    await _create_execution(session_factory, execution_id='exec', owner_principal_id=OWNER)
    binding = make_source_binding(
        run_id='exec',
        run_span_id='span',
        assignment_id='asn',
        attempt_id='attempt-1',
        owner_principal_id=OWNER,
        selected_sinks=('tdsql', 'artifact'),
        source_identities={'tdsql': TraceSourceIdentity(source_id='dataset-tdsql', source_revision='1')},
    )
    async with session_factory() as session:
        execution = await session.get(ExecutionRow, 'exec')
        execution.environment_id = 'env-1'
        execution.current_assignment_id = 'asn'
        execution.current_attempt_id = 'attempt-1'
        execution.status = 'running'
        session.add(
            AssignmentRow(
                assignment_id='asn',
                execution_id='exec',
                attempt_id='attempt-1',
                attempt_seq=1,
                manager_id='manager',
                manager_session_id='session',
                status='active',
                runtime_info_json={'run_trace': {'source_binding': binding}},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
    app = create_application_app(session_factory=session_factory, config=config)
    source = Source([event(1), event(3), event(5)])
    monkeypatch.setattr(app[TRACE_READ_SERVICE_KEY].resolver, 'create_source', lambda resolved: source)
    async with TestClient(TestServer(app)) as client:
        yield client, source, tokens, session_factory


async def test_preview_http_reads_omitted_row_then_switches_back_to_raw(gateway):
    client, source, _, _ = gateway
    service = client.app[TRACE_READ_SERVICE_KEY]
    service.config = service.config.model_copy(update={'max_event_bytes': 4096, 'page_target_bytes': 4096})
    large = event(2, payload=json.dumps({'text': 'x' * 6000}))
    source.records[:] = [event(1), large, event(3)]
    headers = _identity_headers(1001)
    discovery = await (await client.get(LIST_URL, headers=headers)).json()
    path = f'{LIST_URL}/{discovery["items"][0]["stream_id"]}/events'
    first = await (await client.get(path, headers=headers, params={'limit': 1})).json()
    failed = await client.get(path, headers=headers, params={'cursor': first['next_cursor']})
    assert failed.status == 413
    response = await client.get(
        path, headers=headers, params={'cursor': first['next_cursor'], 'mode': 'preview', 'limit': 1}
    )
    assert response.status == 200
    preview = await response.json()
    assert preview['mode'] == 'preview'
    assert preview['items'][0]['event_data'] is None
    assert preview['items'][0]['event_data_bytes'] == len(large['event_data'].encode())
    tail = await (await client.get(path, headers=headers, params={'cursor': preview['next_cursor']})).json()
    assert tail['mode'] == 'raw'
    assert [row['record_id'] for row in tail['items']] == ['3']
    for query in [{'mode': ''}, {'mode': 'local'}, [('mode', 'raw'), ('mode', 'preview')]]:
        assert (await client.get(path, headers=headers, params=query)).status == 400
    assert (await client.get(LIST_URL, headers=headers, params={'mode': 'preview'})).status == 400


async def test_actual_http_auth_discovery_precision_and_empty_then_follow(gateway):
    client, source, _, _ = gateway
    response = await client.get(LIST_URL)
    assert response.status == 401
    assert response.headers['Cache-Control'] == 'no-store'
    assert source.calls == []
    response = await client.get(LIST_URL, headers=_identity_headers(1001))
    assert response.status == 200
    discovery = await response.json()
    assert discovery['items'][0]['attempt_no'] == 1
    selected = discovery['items'][0]['stream_id']
    path = f'{LIST_URL}/{selected}/events'
    response = await client.get(path, params={'limit': '2'}, headers=_identity_headers(1001))
    first = await response.json()
    assert response.status == 200
    assert first['has_more'] is True
    assert first['items'][0]['time_unix_nano'] == '1789286400000000001'
    assert first['items'][0]['event_data'] == event(1)['event_data']
    second = await (
        await client.get(path, params={'cursor': first['next_cursor']}, headers=_identity_headers(1001))
    ).json()
    assert [row['record_id'] for row in second['items']] == ['5']
    empty = await (
        await client.get(path, params={'cursor': second['next_cursor']}, headers=_identity_headers(1001))
    ).json()
    assert empty['items'] == []
    source.records.append(event(8))
    followed = await (
        await client.get(path, params={'cursor': empty['next_cursor']}, headers=_identity_headers(1001))
    ).json()
    assert [row['record_id'] for row in followed['items']] == ['8']
    assert {call[1]['principal_scope_id'] for call in source.calls} == {principal_trace_scope_id(OWNER)}


async def test_shared_reader_cursor_is_owner_scoped_and_revocation_is_immediate(gateway):
    client, source, _, sessions = gateway
    response = await client.get(LIST_URL, headers=_identity_headers(1002))
    assert response.status == 200
    stream = (await response.json())['items'][0]['stream_id']
    path = f'{LIST_URL}/{stream}/events'
    page = await (await client.get(path, headers=_identity_headers(1002))).json()
    assert source.calls[-1][1]['principal_scope_id'] == principal_trace_scope_id(OWNER)
    calls = len(source.calls)
    response = await client.get(path, params={'cursor': page['next_cursor']}, headers=_identity_headers(1001))
    assert response.status == 400
    assert (await response.json())['error'] == 'invalid_trace_cursor'
    async with sessions() as session:
        await AccessRepository(session).delete_principal_access_grant(
            owner_principal_id=OWNER, grantee_principal_id=READER
        )
        await session.commit()
    response = await client.get(path, params={'cursor': page['next_cursor']}, headers=_identity_headers(1002))
    assert response.status == 404
    assert (await response.json())['error'] == 'resource_not_found'
    assert len(source.calls) == calls


@pytest.mark.parametrize(('token_name', 'status'), [('good', 200), ('wrong-env', 404), ('wrong-scope', 403)])
async def test_environment_bound_api_token_and_scope(gateway, token_name, status):
    client, source, _, _ = gateway
    response = await client.get(LIST_URL, headers=_sgw_headers(f'trace-test-{token_name}'))
    assert response.status == status
    if status != 200:
        assert source.calls == []


async def test_execution_owner_environment_grant_is_checked_for_shared_reader(gateway):
    client, source, _, sessions = gateway
    async with sessions() as session:
        grant = await session.get(EnvironmentGrantRow, (OWNER, 'env-1'))
        grant.scopes_json = ['artifact:write']
        await session.commit()
    response = await client.get(LIST_URL, headers=_identity_headers(1002))
    assert response.status == 409
    assert (await response.json())['error'] == 'environment_scope_required'
    assert source.calls == []


@pytest.mark.parametrize(
    'query',
    [
        'limit=0',
        'limit=201',
        'limit=no',
        'limit=1&limit=2',
        'follow=1',
        'wait=5',
        'backend=arbitrary;SQL',
        'cursor=' + 'x' * 4097,
    ],
)
async def test_invalid_parameters_never_query_trace_database(gateway, query):
    client, source, _, _ = gateway
    response = await client.get(LIST_URL + '?' + query, headers=_identity_headers(1001))
    assert response.status == 400
    assert source.calls == []


@pytest.mark.parametrize('spec_type', ['nexus_batch', 'nexus_live'])
async def test_type_rejection_precedes_trace_query(gateway, spec_type):
    client, source, _, sessions = gateway
    async with sessions() as session:
        row = await session.get(ExecutionRow, 'exec')
        row.spec_type = spec_type
        await session.commit()
    response = await client.get(LIST_URL, headers=_identity_headers(1001))
    assert response.status == 400
    assert (await response.json())['error'] == 'execution_type_not_supported'
    assert source.calls == []


async def test_database_error_is_sanitized_and_not_logged_as_traceback(gateway, monkeypatch, caplog):
    client, source, _, _ = gateway

    async def fail(*args, **kwargs):
        raise RuntimeError('sentinel-db-password event_data=private-payload')

    monkeypatch.setattr(source, 'list_raw_streams', fail)
    response = await client.get(LIST_URL, headers=_identity_headers(1001))
    assert response.status == 503
    value = await response.json()
    assert value == {'error': 'trace_source_unavailable', 'retryable': True}
    assert response.headers['Retry-After'] == '1'
    assert 'sentinel-db-password' not in caplog.text + json.dumps(value)
    assert 'private-payload' not in caplog.text


async def test_read_diagnostics_include_authorized_scope_and_separate_encoding_time(gateway, caplog):
    client, _, _, _ = gateway
    with caplog.at_level('INFO', logger='nexus.distributed.gateways.application.routes.trace_events'):
        response = await client.get(LIST_URL, headers=_identity_headers(1001))
    assert response.status == 200
    assert 'execution=exec' in caplog.text
    assert f'owner={OWNER}' in caplog.text
    assert 'backend=tdsql' in caplog.text
    assert 'queries=1 query_ms=' in caplog.text
    assert 'encoding_ms=' in caplog.text
    assert 'errors=0' in caplog.text
    assert response.headers['X-Request-ID'] in caplog.text
    assert 'event_data' not in caplog.text
