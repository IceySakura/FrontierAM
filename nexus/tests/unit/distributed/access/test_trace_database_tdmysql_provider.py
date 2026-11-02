from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

from nexus.distributed.access.trace_database_providers import tencent_cloud_tdmysql
from nexus.distributed.access.trace_database_providers.tencent_cloud_tdmysql import (
    TencentCloudTDMySQLTraceDatabaseAccountProvider,
)


_TABLE = 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'
_USER = 'nxs_ro_0123456789abcdef01234567'


class FakeTDMySQLClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.create_error: Exception | None = None
        # tdmysql reports flow state as a string, unlike the integer DCDB uses.
        self.flow_statuses: list[str] = ['success']
        self.tables: list[str] = [_TABLE]
        self.modify_returns_flow = False

    def CreateUsers(self, request):  # noqa: N802
        self.calls.append(('CreateUsers', request))
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(FlowId=7, RequestId='create-request')

    def ResetUsersPassword(self, request):  # noqa: N802
        self.calls.append(('ResetUsersPassword', request))
        return SimpleNamespace(FlowId=8, RequestId='reset-request')

    def DescribeDatabaseObjects(self, request):  # noqa: N802
        self.calls.append(('DescribeDatabaseObjects', request))
        page = self.tables[request.Offset : request.Offset + request.Limit]
        return SimpleNamespace(Tables=[SimpleNamespace(Table=table) for table in page])

    def ModifyUserPrivileges(self, request):  # noqa: N802
        self.calls.append(('ModifyUserPrivileges', request))
        if self.modify_returns_flow:
            return SimpleNamespace(FlowId=9, RequestId='modify-request')
        # Documented as synchronous: the response carries no FlowId.
        return SimpleNamespace(RequestId='modify-request')

    def DescribeUserPrivileges(self, request):  # noqa: N802
        self.calls.append(('DescribeUserPrivileges', request))
        return SimpleNamespace(Privileges=[], RequestId='describe-request')

    def DescribeFlow(self, request):  # noqa: N802
        self.calls.append(('DescribeFlow', request))
        return SimpleNamespace(Status=self.flow_statuses.pop(0), RequestId='flow-request')


def _provider(client: FakeTDMySQLClient) -> TencentCloudTDMySQLTraceDatabaseAccountProvider:
    return TencentCloudTDMySQLTraceDatabaseAccountProvider(
        region='ap-nanjing',
        instance_id='tdsql3-49qrr9xo6y',
        account_host='%',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        client=client,
        flow_poll_interval=0,
    )


def test_tdmysql_provider_client_uses_configured_endpoint(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def create_client(_credential, region, profile):
        captured.update(region=region, endpoint=profile.httpProfile.endpoint)
        return object()

    monkeypatch.setattr(tencent_cloud_tdmysql.tdmysql_client, 'TdmysqlClient', create_client)

    client = tencent_cloud_tdmysql._create_client(
        region='ap-nanjing',
        api_endpoint='tdmysql.internal.tencentcloudapi.com',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        request_timeout_seconds=30,
    )

    assert client is not None
    assert captured == {'region': 'ap-nanjing', 'endpoint': 'tdmysql.internal.tencentcloudapi.com'}


def test_tdmysql_provider_rejects_dcdb_endpoint() -> None:
    with pytest.raises(ValueError, match='unsupported TDSQL API endpoint'):
        TencentCloudTDMySQLTraceDatabaseAccountProvider(
            region='ap-nanjing',
            instance_id='tdsql3-49qrr9xo6y',
            api_endpoint='dcdb.tencentcloudapi.com',
            secret_id='test-secret-id',
            secret_key='test-secret-key',
            client=FakeTDMySQLClient(),
        )


async def test_tdmysql_provider_creates_account_and_grants_only_target_table() -> None:
    client = FakeTDMySQLClient()
    # Two flows now: CreateUsers, then the explicit password reset.
    client.flow_statuses = ['running', 'success', 'success']

    await _provider(client).set_readonly_account(
        username=_USER,
        password='Aa1!secure-generated-password',
        database='runner',
        table_name=_TABLE,
    )

    assert [action for action, _request in client.calls] == [
        'CreateUsers',
        'DescribeFlow',
        'DescribeFlow',
        # The password is always applied explicitly, never left to CreateUsers alone.
        'ResetUsersPassword',
        'DescribeFlow',
        'DescribeDatabaseObjects',
        'ModifyUserPrivileges',
    ]
    create_request = client.calls[0][1]
    assert create_request.InstanceId == 'tdsql3-49qrr9xo6y'
    # Users is a list of User, and the password sits outside it.
    assert [(u.UserName, u.Host) for u in create_request.Users] == [(_USER, '%')]
    assert create_request.Password == 'Aa1!secure-generated-password'

    privilege_request = next(r for action, r in client.calls if action == 'ModifyUserPrivileges')
    assert [(u.UserName, u.Host) for u in privilege_request.Users] == [(_USER, '%')]
    assert privilege_request.GlobalPrivileges == []
    assert [(p.Database, p.Privileges) for p in privilege_request.DatabasePrivileges] == [('runner', [])]
    assert [(p.Database, p.Table, p.Privileges) for p in privilege_request.TablePrivileges] == [
        ('runner', _TABLE, ['SELECT'])
    ]


async def test_tdmysql_provider_resets_existing_account_before_reconciling_privileges() -> None:
    client = FakeTDMySQLClient()
    client.create_error = TencentCloudSDKException(
        'InvalidParameterValue.UserAlreadyExists',
        'user already exists',
        'create-request',
    )

    await _provider(client).set_readonly_account(
        username=_USER,
        password='Aa1!secure-generated-password',
        database='runner',
        table_name=_TABLE,
    )

    assert [action for action, _request in client.calls] == [
        'CreateUsers',
        'ResetUsersPassword',
        'DescribeFlow',
        'DescribeDatabaseObjects',
        'ModifyUserPrivileges',
    ]
    reset_request = client.calls[1][1]
    assert reset_request.InstanceId == 'tdsql3-49qrr9xo6y'
    # Unlike CreateUsers, the password lives inside each list entry.
    assert [(u.UserName, u.Host, u.Password) for u in reset_request.Users] == [
        (_USER, '%', 'Aa1!secure-generated-password')
    ]


async def test_tdmysql_provider_sets_the_password_when_create_silently_succeeds() -> None:
    """Regression: CreateUsers accepts an existing account without applying the password.

    No already-exists error is raised, so a reset conditional on that error never ran and
    the console handed out passwords that were never in force.
    """
    client = FakeTDMySQLClient()
    client.flow_statuses = ['success', 'success']

    await _provider(client).set_readonly_account(
        username=_USER,
        password='Aa1!secure-generated-password',
        database='runner',
        table_name=_TABLE,
    )

    actions = [action for action, _request in client.calls]
    assert 'ResetUsersPassword' in actions
    assert actions.index('CreateUsers') < actions.index('ResetUsersPassword')
    # Privileges are still reconciled after the password lands.
    assert actions.index('ResetUsersPassword') < actions.index('ModifyUserPrivileges')
    reset_request = next(r for action, r in client.calls if action == 'ResetUsersPassword')
    assert [(u.UserName, u.Host, u.Password) for u in reset_request.Users] == [
        (_USER, '%', 'Aa1!secure-generated-password')
    ]


async def test_tdmysql_reset_password_survives_sdk_serialization() -> None:
    """The password must reach the wire as ``Users[].Password``.

    Attribute assertions pass even when the SDK drops the value: an undeclared name
    serializes to the silently ignored key ``Assword``.
    """
    client = FakeTDMySQLClient()

    await _provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')

    request = next(r for action, r in client.calls if action == 'ResetUsersPassword')
    payload = request._serialize()
    assert payload['InstanceId'] == 'tdsql3-49qrr9xo6y'
    assert payload['Users'] == [{'UserName': _USER, 'Host': '%', 'Password': 'Aa1!secure-generated-password'}]


async def test_tdmysql_provider_propagates_unrelated_create_errors() -> None:
    client = FakeTDMySQLClient()
    client.create_error = TencentCloudSDKException('InvalidParameterValue.InstanceNotFound', 'missing')

    with pytest.raises(TencentCloudSDKException):
        await _provider(client).set_readonly_account(
            username=_USER,
            password='Aa1!secure-generated-password',
            database='runner',
            table_name=_TABLE,
        )
    assert [action for action, _request in client.calls] == ['CreateUsers']


async def test_tdmysql_provider_disables_and_restores_table_access() -> None:
    client = FakeTDMySQLClient()
    provider = _provider(client)

    await provider.disable_account(username=_USER)
    await provider.restore_account(username=_USER, database='runner', table_name=_TABLE)

    modifications = [request for action, request in client.calls if action == 'ModifyUserPrivileges']
    assert modifications[0].TablePrivileges[0].Privileges == []
    assert modifications[1].TablePrivileges[0].Privileges == ['SELECT']


async def test_tdmysql_provider_revokes_other_tables_by_sending_empty_privileges() -> None:
    client = FakeTDMySQLClient()
    other = 'trajectory_span_events_pr_ffffffffffffffffffffffffffffffff'
    client.tables = [_TABLE, other]

    await _provider(client).reconcile_readonly_account(username=_USER, database='runner', table_name=_TABLE)

    request = next(r for action, r in client.calls if action == 'ModifyUserPrivileges')
    assert {p.Table: p.Privileges for p in request.TablePrivileges} == {_TABLE: ['SELECT'], other: []}


async def test_tdmysql_provider_pages_through_the_table_inventory() -> None:
    client = FakeTDMySQLClient()
    granted = 'trajectory_span_events_pr_' + 'a' * 32
    client.tables = [f'trajectory_span_events_pr_{index:032x}' for index in range(250)] + [granted]

    await _provider(client).reconcile_readonly_account(username=_USER, database='runner', table_name=granted)

    inventory_calls = [r for action, r in client.calls if action == 'DescribeDatabaseObjects']
    assert [r.Offset for r in inventory_calls] == [0, 100, 200]
    request = next(r for action, r in client.calls if action == 'ModifyUserPrivileges')
    # A short page ends paging, and every known table must still appear in the request.
    assert len(request.TablePrivileges) == 251
    assert {p.Table for p in request.TablePrivileges if p.Privileges} == {granted}


async def test_tdmysql_provider_refuses_partial_sync_without_inventory() -> None:
    client = FakeTDMySQLClient()
    client.DescribeDatabaseObjects = lambda request: SimpleNamespace(Tables=None)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='TDSQL table inventory is unavailable'):
        await _provider(client).reconcile_readonly_account(username=_USER, database='runner', table_name=_TABLE)


async def test_tdmysql_provider_skips_flow_polling_when_privileges_apply_synchronously() -> None:
    client = FakeTDMySQLClient()

    await _provider(client).reconcile_readonly_account(username=_USER, database='runner', table_name=_TABLE)

    assert [action for action, _request in client.calls] == ['DescribeDatabaseObjects', 'ModifyUserPrivileges']


async def test_tdmysql_provider_polls_privilege_flow_when_one_is_returned() -> None:
    client = FakeTDMySQLClient()
    client.modify_returns_flow = True
    client.flow_statuses = ['success']

    await _provider(client).reconcile_readonly_account(username=_USER, database='runner', table_name=_TABLE)

    assert [action for action, _request in client.calls] == [
        'DescribeDatabaseObjects',
        'ModifyUserPrivileges',
        'DescribeFlow',
    ]


async def test_tdmysql_provider_fails_when_account_flow_fails() -> None:
    client = FakeTDMySQLClient()
    client.flow_statuses = ['failed']

    with pytest.raises(RuntimeError, match='TDSQL account flow failed'):
        await _provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')


async def test_tdmysql_provider_waits_through_the_undocumented_initial_status() -> None:
    """Regression: a submitted flow reports 'initial' before it starts running.

    The API documents only running, success, paused and failed. Treating that list as
    exhaustive abandoned an account the server had already created, leaving the caller a
    failure for an operation that went on to succeed.
    """
    client = FakeTDMySQLClient()
    client.flow_statuses = ['initial', 'running', 'success']

    await _provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert client.flow_statuses == []


async def test_tdmysql_provider_keeps_polling_an_unrecognised_flow_status() -> None:
    """Any non-terminal value is polled, not rejected: the documented set proved partial."""
    client = FakeTDMySQLClient()
    client.flow_statuses = ['a_status_this_client_has_never_seen', 'success']

    await _provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert client.flow_statuses == []


async def test_tdmysql_provider_rejects_a_flow_response_without_a_status() -> None:
    """A broken response contract is not something polling can resolve."""
    client = FakeTDMySQLClient()
    client.flow_statuses = [0]

    with pytest.raises(RuntimeError, match='TDSQL account flow returned no status'):
        await _provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')


def _pending_provider(client: FakeTDMySQLClient) -> TencentCloudTDMySQLTraceDatabaseAccountProvider:
    return TencentCloudTDMySQLTraceDatabaseAccountProvider(
        region='ap-nanjing',
        instance_id='tdsql3-49qrr9xo6y',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        client=client,
        flow_timeout_seconds=0.01,
        flow_poll_interval=0,
    )


async def test_tdmysql_provider_times_out_while_flow_stays_pending() -> None:
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='running')  # type: ignore[method-assign]

    # The last status is named, which distinguishes a flow stuck before it ever ran from
    # one that merely ran too slowly -- the only signal left once polling gives up.
    with pytest.raises(TimeoutError, match="timed out in status 'running'"):
        await _pending_provider(client).reset_password(username=_USER, password='Aa1!secure-generated-password')


async def test_tdmysql_provider_times_out_when_no_verification_host_is_configured() -> None:
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]
    provider = _pending_provider(client)

    assert provider.verify_host is None
    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')


def _verifying_provider(
    client: FakeTDMySQLClient, accepted: list[bool]
) -> tuple[TencentCloudTDMySQLTraceDatabaseAccountProvider, list[tuple[str, str]]]:
    """A provider whose login probe is scripted rather than real, recording what it was asked."""
    provider = TencentCloudTDMySQLTraceDatabaseAccountProvider(
        region='ap-nanjing',
        instance_id='tdsql3-49qrr9xo6y',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        client=client,
        flow_timeout_seconds=0.5,
        flow_poll_interval=0,
        verify_host='21.131.14.24',
        verify_probe_interval=0,
    )
    attempts: list[tuple[str, str]] = []

    async def probe(username: str, password: str) -> bool:
        attempts.append((username, password))
        return accepted.pop(0) if accepted else False

    provider._password_accepted = probe  # type: ignore[method-assign]
    return provider, attempts


async def test_tdmysql_provider_stops_waiting_once_the_new_password_authenticates() -> None:
    """A paused flow is no longer fatal: production applies the password while it still runs."""
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]
    provider, attempts = _verifying_provider(client, [True])

    await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert attempts == [(_USER, 'Aa1!secure-generated-password')]


async def test_tdmysql_provider_keeps_polling_until_the_password_lands() -> None:
    """The password is not instant, so one rejection must not end the wait."""
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='running')  # type: ignore[method-assign]
    provider, attempts = _verifying_provider(client, [False, False, True])

    await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert len(attempts) == 3


async def test_tdmysql_provider_times_out_when_the_password_never_authenticates() -> None:
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]
    provider, attempts = _verifying_provider(client, [])

    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')
    assert attempts


async def test_tdmysql_provider_prefers_a_successful_flow_over_probing() -> None:
    """Nothing is probed on the happy path; the login check is only a fallback."""
    client = FakeTDMySQLClient()
    client.flow_statuses = ['success']
    provider, attempts = _verifying_provider(client, [True])

    await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert attempts == []


async def test_tdmysql_provider_never_probes_a_privilege_flow() -> None:
    """disable_account revokes through this path and has no password to authenticate with."""
    client = FakeTDMySQLClient()
    client.modify_returns_flow = True
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]
    provider, attempts = _verifying_provider(client, [True])

    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await provider.disable_account(username=_USER)
    assert attempts == []


async def test_tdmysql_provider_never_probes_an_account_creation_flow() -> None:
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]
    provider, attempts = _verifying_provider(client, [True])
    # CreateUsers waits first, so the failure comes from its own flow, not the reset that follows.
    provider.client.CreateUsers = lambda request: SimpleNamespace(FlowId=7, RequestId='create')  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await provider.set_readonly_account(
            username=_USER, password='Aa1!secure-generated-password', database='runner', table_name=_TABLE
        )
    assert attempts == []


async def test_tdmysql_password_probe_reports_failure_instead_of_raising(monkeypatch) -> None:
    """A refused or unreachable database must read as 'not yet', not abort the flow wait."""
    provider = _pending_provider(FakeTDMySQLClient())
    provider.verify_host = '21.131.14.24'

    async def refuse(**_kwargs):
        raise OSError('connection refused')

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', refuse)

    assert await provider._password_accepted(_USER, 'Aa1!secure-generated-password') is False


async def test_tdmysql_password_probe_closes_the_connection_it_opens(monkeypatch) -> None:
    provider = _pending_provider(FakeTDMySQLClient())
    provider.verify_host = '21.131.14.24'
    closed: list[bool] = []
    captured: dict[str, Any] = {}

    async def accept(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr('nexus.runner.trace.tdsql._driver.connect', accept)

    assert await provider._password_accepted(_USER, 'Aa1!pw') is True
    assert closed == [True]
    assert captured['host'] == '21.131.14.24'
    assert captured['user'] == _USER
    assert captured['password'] == 'Aa1!pw'
    assert captured['db'] == 'runner'


async def test_tdmysql_provider_rejects_a_negative_probe_interval() -> None:
    with pytest.raises(ValueError, match='TDSQL timeout values are invalid'):
        TencentCloudTDMySQLTraceDatabaseAccountProvider(
            region='ap-nanjing',
            instance_id='tdsql3-49qrr9xo6y',
            secret_id='test-secret-id',
            secret_key='test-secret-key',
            client=FakeTDMySQLClient(),
            verify_probe_interval=-1,
        )


async def test_tdmysql_provider_still_fails_a_paused_privilege_flow() -> None:
    client = FakeTDMySQLClient()
    client.modify_returns_flow = True
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await _pending_provider(client).disable_account(username=_USER)


async def test_tdmysql_provider_defaults_to_the_measured_production_flow_budget() -> None:
    assert _provider(FakeTDMySQLClient()).flow_timeout_seconds == 180.0


async def test_tdmysql_provider_polls_a_slow_flow_up_to_the_configured_budget() -> None:
    client = FakeTDMySQLClient()
    client.flow_statuses = ['running'] * 40 + ['success']
    provider = TencentCloudTDMySQLTraceDatabaseAccountProvider(
        region='ap-nanjing',
        instance_id='tdsql3-49qrr9xo6y',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        client=client,
        flow_timeout_seconds=180.0,
        flow_poll_interval=0,
    )

    await provider.reset_password(username=_USER, password='Aa1!secure-generated-password')

    assert client.flow_statuses == []


async def test_tdmysql_provider_rejects_a_non_positive_flow_budget() -> None:
    with pytest.raises(ValueError, match='TDSQL timeout values are invalid'):
        TencentCloudTDMySQLTraceDatabaseAccountProvider(
            region='ap-nanjing',
            instance_id='tdsql3-49qrr9xo6y',
            secret_id='test-secret-id',
            secret_key='test-secret-key',
            client=FakeTDMySQLClient(),
            flow_timeout_seconds=0,
        )


async def test_tdmysql_provider_still_fails_a_paused_account_creation_flow() -> None:
    client = FakeTDMySQLClient()
    client.DescribeFlow = lambda request: SimpleNamespace(Status='paused')  # type: ignore[method-assign]

    with pytest.raises(TimeoutError, match="timed out in status 'paused'"):
        await _pending_provider(client).set_readonly_account(
            username=_USER,
            password='Aa1!secure-generated-password',
            database='runner',
            table_name=_TABLE,
        )


@pytest.mark.parametrize('operation', ['set_readonly_account', 'restore_account', 'reconcile_readonly_account'])
async def test_tdmysql_invalid_table_fails_before_any_cloud_mutation(operation) -> None:
    client = FakeTDMySQLClient()
    kwargs: dict[str, Any] = {'username': _USER, 'database': 'runner', 'table_name': '*'}
    if operation == 'set_readonly_account':
        kwargs['password'] = 'Aa1!test-password'

    with pytest.raises(ValueError, match='invalid trace database table name'):
        await getattr(_provider(client), operation)(**kwargs)
    assert client.calls == []


async def test_tdmysql_provider_rejects_database_mismatch() -> None:
    client = FakeTDMySQLClient()

    with pytest.raises(ValueError, match='trace database account provider target mismatch'):
        await _provider(client).reconcile_readonly_account(username=_USER, database='other', table_name=_TABLE)
    assert client.calls == []


@pytest.mark.parametrize(
    ('privileges', 'expected'),
    [
        ([['SELECT']], 'database'),
        ([[], ['ALL PRIVILEGES']], 'database'),
        ([[], [], ['SELECT']], 'table'),
        ([[], [], []], 'none'),
    ],
)
async def test_tdmysql_provider_reads_access_scope(privileges, expected) -> None:
    client = FakeTDMySQLClient()
    responses = list(privileges)
    client.DescribeUserPrivileges = lambda request: SimpleNamespace(  # type: ignore[method-assign]
        Privileges=responses.pop(0)
    )

    scope = await _provider(client).read_access_scope(username=_USER, database='runner', table_name=_TABLE)

    assert scope == expected


async def test_tdmysql_provider_reports_unknown_scope_when_privileges_are_null() -> None:
    client = FakeTDMySQLClient()
    client.DescribeUserPrivileges = lambda request: SimpleNamespace(Privileges=None)  # type: ignore[method-assign]

    scope = await _provider(client).read_access_scope(username=_USER, database='runner', table_name=_TABLE)

    assert scope == 'unknown'


async def test_tdmysql_provider_read_access_scope_uses_object_type_field() -> None:
    client = FakeTDMySQLClient()

    await _provider(client).read_access_scope(username=_USER, database='runner', table_name=_TABLE)

    requests = [r for action, r in client.calls if action == 'DescribeUserPrivileges']
    # tdmysql renamed the DCDB ``Type`` field to ``ObjectType``.
    assert [(r.DbName, r.ObjectType, r.Object) for r in requests] == [
        ('*', '*', None),
        ('runner', '*', None),
        ('runner', 'table', _TABLE),
    ]
