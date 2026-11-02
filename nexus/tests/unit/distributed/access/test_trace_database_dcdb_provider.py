from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException

from nexus.distributed.access.trace_database_providers import tencent_cloud_dcdb
from nexus.distributed.access.trace_database_providers.tencent_cloud_dcdb import (
    TencentCloudDCDBTraceDatabaseAccountProvider,
)


class FakeDCDBClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.create_error: Exception | None = None
        self.flow_statuses: list[int] = [0]
        self.tables: list[str] = ['trajectory_span_events_pr_0123456789abcdef0123456789abcdef']

    def CreateAccount(self, request):  # noqa: N802
        self.calls.append(('CreateAccount', request))
        if self.create_error is not None:
            raise self.create_error
        return SimpleNamespace(RequestId='create-request')

    def ResetAccountPassword(self, request):  # noqa: N802
        self.calls.append(('ResetAccountPassword', request))
        return SimpleNamespace(RequestId='reset-request')

    def DescribeDatabaseObjects(self, request):  # noqa: N802
        self.calls.append(('DescribeDatabaseObjects', request))
        return SimpleNamespace(
            Tables=[SimpleNamespace(Table=table) for table in self.tables],
            RequestId='inventory-request',
        )

    def ModifyAccountPrivileges(self, request):  # noqa: N802
        self.calls.append(('ModifyAccountPrivileges', request))
        return SimpleNamespace(FlowId=42, RequestId='modify-request')

    def DescribeFlow(self, request):  # noqa: N802
        self.calls.append(('DescribeFlow', request))
        return SimpleNamespace(Status=self.flow_statuses.pop(0), RequestId='flow-request')


def _provider(client: FakeDCDBClient) -> TencentCloudDCDBTraceDatabaseAccountProvider:
    return TencentCloudDCDBTraceDatabaseAccountProvider(
        region='ap-nanjing',
        instance_id='tdsqlshard-9biily2z7z',
        account_host='%',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        client=client,
        flow_poll_interval=0,
    )


def test_dcdb_provider_client_uses_configured_endpoint(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def create_client(_credential, region, profile):
        captured.update(region=region, endpoint=profile.httpProfile.endpoint)
        return object()

    monkeypatch.setattr(tencent_cloud_dcdb.dcdb_client, 'DcdbClient', create_client)

    client = tencent_cloud_dcdb._create_client(
        region='ap-nanjing',
        api_endpoint='dcdb.internal.tencentcloudapi.com',
        secret_id='test-secret-id',
        secret_key='test-secret-key',
        request_timeout_seconds=30,
    )

    assert client is not None
    assert captured == {
        'region': 'ap-nanjing',
        'endpoint': 'dcdb.internal.tencentcloudapi.com',
    }


async def test_dcdb_provider_creates_primary_account_and_grants_only_database_select() -> None:
    client = FakeDCDBClient()
    client.flow_statuses = [2, 0]

    await _provider(client).set_readonly_account(
        username='nxs_ro_0123456789abcdef01234567',
        password='Aa1!secure-generated-password',
        database='runner',
        table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
    )

    assert [action for action, _request in client.calls] == [
        'CreateAccount',
        'DescribeDatabaseObjects',
        'ModifyAccountPrivileges',
        'DescribeFlow',
        'DescribeFlow',
    ]
    create_request = client.calls[0][1]
    assert create_request.InstanceId == 'tdsqlshard-9biily2z7z'
    assert create_request.UserName == 'nxs_ro_0123456789abcdef01234567'
    assert create_request.Host == '%'
    assert create_request.Password == 'Aa1!secure-generated-password'
    assert create_request.ReadOnly == 0

    privilege_request = next(request for action, request in client.calls if action == 'ModifyAccountPrivileges')
    assert privilege_request.GlobalPrivileges == []
    # The database has to be named with an empty privilege list; a bare [] would keep whatever
    # database-level grant the account already had.
    assert [(item.Database, item.Privileges) for item in privilege_request.DatabasePrivileges] == [('runner', [])]
    assert privilege_request.TablePrivileges[0].Database == 'runner'
    assert privilege_request.TablePrivileges[0].Table == 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'
    assert privilege_request.TablePrivileges[0].Privileges == ['SELECT']
    assert privilege_request.ColumnPrivileges == []
    assert privilege_request.ViewPrivileges == []


async def test_dcdb_provider_resets_existing_account_before_reconciling_privileges() -> None:
    client = FakeDCDBClient()
    client.create_error = TencentCloudSDKException(
        'InvalidParameterValue.AccountAlreadyExists',
        'account already exists',
        'create-request',
    )

    await _provider(client).set_readonly_account(
        username='nxs_ro_0123456789abcdef01234567',
        password='Aa1!secure-generated-password',
        database='runner',
        table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
    )

    assert [action for action, _request in client.calls] == [
        'CreateAccount',
        'ResetAccountPassword',
        'DescribeDatabaseObjects',
        'ModifyAccountPrivileges',
        'DescribeFlow',
    ]
    reset_request = client.calls[1][1]
    assert reset_request.InstanceId == 'tdsqlshard-9biily2z7z'
    assert reset_request.Host == '%'
    assert reset_request.Password == 'Aa1!secure-generated-password'


async def test_dcdb_provider_disables_and_restores_database_access() -> None:
    client = FakeDCDBClient()
    client.flow_statuses = [0, 0]
    provider = _provider(client)

    await provider.disable_account(username='nxs_ro_0123456789abcdef01234567')
    await provider.restore_account(
        username='nxs_ro_0123456789abcdef01234567',
        database='runner',
        table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
    )

    modifications = [request for action, request in client.calls if action == 'ModifyAccountPrivileges']
    # Disabling must revoke by naming every known table with an empty privilege list. Sending a
    # bare TablePrivileges=[] would leave the account readable while the record says disabled.
    assert [(item.Database, item.Privileges) for item in modifications[0].DatabasePrivileges] == [('runner', [])]
    assert [(item.Table, item.Privileges) for item in modifications[0].TablePrivileges] == [
        ('trajectory_span_events_pr_0123456789abcdef0123456789abcdef', [])
    ]
    assert [(item.Database, item.Privileges) for item in modifications[1].DatabasePrivileges] == [('runner', [])]
    assert modifications[1].TablePrivileges[0].Privileges == ['SELECT']


async def test_dcdb_provider_revokes_other_tables_when_narrowing_to_one_table() -> None:
    client = FakeDCDBClient()
    granted = 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef'
    stale = 'trajectory_span_events_pr_ffffffffffffffffffffffffffffffff'
    client.tables = [granted, stale]

    await _provider(client).reconcile_readonly_account(
        username='nxs_ro_0123456789abcdef01234567',
        database='runner',
        table_name=granted,
    )

    request = next(request for action, request in client.calls if action == 'ModifyAccountPrivileges')
    assert {item.Table: item.Privileges for item in request.TablePrivileges} == {granted: ['SELECT'], stale: []}


async def test_dcdb_provider_refuses_partial_sync_without_inventory() -> None:
    client = FakeDCDBClient()
    client.DescribeDatabaseObjects = lambda request: SimpleNamespace(Tables=None)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match='DCDB table inventory is unavailable'):
        await _provider(client).reconcile_readonly_account(
            username='nxs_ro_0123456789abcdef01234567',
            database='runner',
            table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        )

    assert [action for action, _request in client.calls] == []


async def test_dcdb_provider_fails_when_privilege_flow_fails() -> None:
    client = FakeDCDBClient()
    client.flow_statuses = [1]

    with pytest.raises(RuntimeError, match='DCDB privilege flow failed'):
        await _provider(client).reconcile_readonly_account(
            username='nxs_ro_0123456789abcdef01234567',
            database='runner',
            table_name='trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
        )


@pytest.mark.parametrize('operation', ['set_readonly_account', 'restore_account', 'reconcile_readonly_account'])
async def test_all_grant_paths_use_only_requested_table(operation) -> None:
    client = FakeDCDBClient()
    kwargs = {
        'username': 'nxs_ro_0123456789abcdef01234567',
        'database': 'runner',
        'table_name': 'trajectory_span_events_pr_0123456789abcdef0123456789abcdef',
    }
    if operation == 'set_readonly_account':
        kwargs['password'] = 'Aa1!test-password'
        client.create_error = TencentCloudSDKException('InvalidParameterValue.AccountAlreadyExists', 'exists')
    await getattr(_provider(client), operation)(**kwargs)
    request = next(request for action, request in client.calls if action == 'ModifyAccountPrivileges')
    assert request.GlobalPrivileges == []
    assert [(item.Database, item.Privileges) for item in request.DatabasePrivileges] == [('runner', [])]
    assert request.ColumnPrivileges == []
    assert request.ViewPrivileges == []
    assert [item.to_json_string() for item in request.TablePrivileges] == [
        '{"Database": "runner", "Table": "trajectory_span_events_pr_0123456789abcdef0123456789abcdef", "Privileges": ["SELECT"]}'
    ]


@pytest.mark.parametrize('operation', ['set_readonly_account', 'restore_account', 'reconcile_readonly_account'])
async def test_invalid_table_fails_before_any_cloud_mutation(operation) -> None:
    client = FakeDCDBClient()
    kwargs = {'username': 'nxs_ro_0123456789abcdef01234567', 'database': 'runner', 'table_name': '*'}
    if operation == 'set_readonly_account':
        kwargs['password'] = 'Aa1!test-password'
    with pytest.raises(ValueError, match='invalid trace database table name'):
        await getattr(_provider(client), operation)(**kwargs)
    assert client.calls == []
