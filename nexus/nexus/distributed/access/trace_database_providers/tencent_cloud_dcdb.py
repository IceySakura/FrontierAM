from __future__ import annotations

import asyncio

from typing import Any

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.dcdb.v20180411 import dcdb_client, models

from nexus.distributed.access.trace_database_providers.base import (
    TraceDatabaseAccountProvider,
    TraceDatabaseTarget,
    validate_account_name,
    validate_database_name,
    validate_table_name,
)


_ACCOUNT_ALREADY_EXISTS = 'InvalidParameterValue.AccountAlreadyExists'
_ALLOWED_API_ENDPOINTS = {
    'dcdb.tencentcloudapi.com',
    'dcdb.internal.tencentcloudapi.com',
}


class TencentCloudDCDBTraceDatabaseAccountProvider(TraceDatabaseTarget, TraceDatabaseAccountProvider):
    def __init__(
        self,
        *,
        region: str,
        instance_id: str,
        api_endpoint: str = 'dcdb.tencentcloudapi.com',
        database: str = 'runner',
        account_host: str = '%',
        secret_id: str,
        secret_key: str,
        client: Any | None = None,
        request_timeout_seconds: float = 30.0,
        flow_timeout_seconds: float = 90.0,
        flow_poll_interval: float = 1.0,
    ) -> None:
        if not region or not instance_id or not account_host:
            raise ValueError('DCDB region, instance ID and account host are required')
        if api_endpoint not in _ALLOWED_API_ENDPOINTS:
            raise ValueError('unsupported DCDB API endpoint')
        if request_timeout_seconds <= 0 or flow_timeout_seconds <= 0 or flow_poll_interval < 0:
            raise ValueError('DCDB timeout values are invalid')
        self.instance_id = instance_id
        self.database = validate_database_name(database)
        self.account_host = account_host
        self.request_timeout_seconds = request_timeout_seconds
        self.flow_timeout_seconds = flow_timeout_seconds
        self.flow_poll_interval = flow_poll_interval
        self.client = client or _create_client(
            region=region,
            api_endpoint=api_endpoint,
            secret_id=secret_id,
            secret_key=secret_key,
            request_timeout_seconds=request_timeout_seconds,
        )

    async def set_readonly_account(self, *, username: str, password: str, database: str, table_name: str) -> None:
        username = validate_account_name(username)
        self._require_database(database)
        table_name = validate_table_name(table_name)
        request = models.CreateAccountRequest()
        request.InstanceId = self.instance_id
        request.UserName = username
        request.Host = self.account_host
        request.Password = password
        request.ReadOnly = 0
        request.Description = 'Nexus principal trace read access'
        try:
            await self._call(self.client.CreateAccount, request)
        except TencentCloudSDKException as exc:
            if exc.code != _ACCOUNT_ALREADY_EXISTS:
                raise
            await self._reset_password(username=username, password=password)
        await self._set_table_privileges(username=username, table_name=table_name)

    async def disable_account(self, *, username: str) -> None:
        await self._set_table_privileges(username=validate_account_name(username), table_name=None)

    async def restore_account(self, *, username: str, database: str, table_name: str) -> None:
        # Disabling revokes rather than locks, so restoring is exactly a reconcile; unlike
        # the MySQL provider there is no account lock left to clear afterwards.
        await self.reconcile_readonly_account(username=username, database=database, table_name=table_name)

    async def reconcile_readonly_account(self, *, username: str, database: str, table_name: str) -> None:
        self._require_database(database)
        table_name = validate_table_name(table_name)
        await self._set_table_privileges(username=validate_account_name(username), table_name=table_name)

    async def _reset_password(self, *, username: str, password: str) -> None:
        request = models.ResetAccountPasswordRequest()
        request.InstanceId = self.instance_id
        request.UserName = username
        request.Host = self.account_host
        request.Password = password
        await self._call(self.client.ResetAccountPassword, request)

    async def _set_table_privileges(self, *, username: str, table_name: str | None) -> None:
        account = models.Account()
        account.User = username
        account.Host = self.account_host
        tables = {table_name} if table_name is not None else set()
        # An omitted or empty list means "keep the existing privileges" for every object-level
        # field of this API, not "revoke them": a revocation has to name the object and pass an
        # empty Privileges list. Sending bare empty lists therefore narrows nothing, which is
        # why the whole known table set is enumerated below and the database is named
        # explicitly. Only GlobalPrivileges is a plain string list, where an empty list does
        # clear the grant.
        inventory_request = models.DescribeDatabaseObjectsRequest()
        inventory_request.InstanceId = self.instance_id
        inventory_request.DbName = self.database
        inventory = await self._call(self.client.DescribeDatabaseObjects, inventory_request)
        if inventory.Tables is None:
            # Revoking against a partial inventory would silently leave stale grants behind
            # while reporting success, so refuse instead of narrowing halfway.
            raise RuntimeError('DCDB table inventory is unavailable; refusing partial permission sync')
        existing_tables = {validate_table_name(item.Table) for item in inventory.Tables}
        table_privileges = []
        for table in sorted(existing_tables | tables):
            privilege = models.TablePrivilege()
            privilege.Database = self.database
            privilege.Table = table
            privilege.Privileges = ['SELECT'] if table in tables else []
            table_privileges.append(privilege)

        database_privilege = models.DatabasePrivilege()
        database_privilege.Database = self.database
        database_privilege.Privileges = []

        request = models.ModifyAccountPrivilegesRequest()
        request.InstanceId = self.instance_id
        request.Accounts = [account]
        request.GlobalPrivileges = []
        request.DatabasePrivileges = [database_privilege]
        request.TablePrivileges = table_privileges
        request.ColumnPrivileges = []
        request.ViewPrivileges = []
        response = await self._call(self.client.ModifyAccountPrivileges, request)
        flow_id = getattr(response, 'FlowId', None)
        if isinstance(flow_id, bool) or not isinstance(flow_id, int) or flow_id <= 0:
            raise RuntimeError('DCDB privilege update returned an invalid flow ID')
        await self._wait_for_flow(flow_id)

    async def _wait_for_flow(self, flow_id: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.flow_timeout_seconds
        while True:
            request = models.DescribeFlowRequest()
            request.FlowId = flow_id
            response = await self._call(self.client.DescribeFlow, request)
            status = getattr(response, 'Status', None)
            if status == 0:
                return
            if status == 1:
                raise RuntimeError(f'DCDB privilege flow failed: {flow_id}')
            if status != 2:
                raise RuntimeError(f'DCDB privilege flow returned invalid status: {status}')
            if loop.time() >= deadline:
                raise TimeoutError(f'DCDB privilege flow timed out: {flow_id}')
            await asyncio.sleep(self.flow_poll_interval)

    async def _call(self, operation, request):
        return await asyncio.wait_for(
            asyncio.to_thread(operation, request),
            timeout=self.request_timeout_seconds,
        )


def _create_client(
    *,
    region: str,
    api_endpoint: str,
    secret_id: str,
    secret_key: str,
    request_timeout_seconds: float,
):
    http_profile = HttpProfile()
    http_profile.endpoint = api_endpoint
    http_profile.reqTimeout = int(request_timeout_seconds)
    return dcdb_client.DcdbClient(
        credential.Credential(secret_id, secret_key),
        region,
        ClientProfile(httpProfile=http_profile),
    )


__all__ = ['TencentCloudDCDBTraceDatabaseAccountProvider']
