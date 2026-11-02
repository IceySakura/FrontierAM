from __future__ import annotations

import asyncio
import logging

from typing import Any

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tdmysql.v20211122 import models, tdmysql_client

from nexus.distributed.access.trace_database_providers.base import (
    TraceDatabaseAccountProvider,
    TraceDatabaseTarget,
    validate_account_name,
    validate_database_name,
    validate_table_name,
)
from nexus.runner.trace.tdsql.config import build_db_config


logger = logging.getLogger(__name__)


_ALLOWED_API_ENDPOINTS = {
    'tdmysql.tencentcloudapi.com',
    'tdmysql.internal.tencentcloudapi.com',
}
# DescribeFlow reports a string here, unlike the integer status DCDB returns. The API
# documents four values -- running, success, paused, failed -- but a freshly submitted
# flow is observed to report 'initial' before it starts, so the documented set is not
# exhaustive. Only the two terminal values are therefore treated as conclusive and
# anything else is polled until the deadline: an unknown status is far more likely to be
# another undocumented pre-terminal state than a permanent condition, and failing on it
# abandons an operation the server has already accepted.
_FLOW_SUCCESS = 'success'
_FLOW_FAILED = 'failed'
# The table inventory is paginated, unlike the DCDB equivalent. Trace databases hold one
# table per principal, so the list outgrows any single page.
_INVENTORY_PAGE_SIZE = 100
_INVENTORY_PAGE_LIMIT = 1000


class TencentCloudTDMySQLTraceDatabaseAccountProvider(TraceDatabaseTarget, TraceDatabaseAccountProvider):
    """Read-only trace accounts on TDSQL 3.0 (Boundless), via the ``tdmysql`` product API.
    A DCDB sibling, not a subclass: separate endpoint, renamed actions, sync privileges, string flow."""

    def __init__(
        self,
        *,
        region: str,
        instance_id: str,
        api_endpoint: str = 'tdmysql.tencentcloudapi.com',
        database: str = 'runner',
        account_host: str = '%',
        secret_id: str,
        secret_key: str,
        client: Any | None = None,
        request_timeout_seconds: float = 30.0,
        flow_timeout_seconds: float = 180.0,
        flow_poll_interval: float = 1.0,
        verify_host: str | None = None,
        verify_port: int = 3306,
        verify_timeout_seconds: float = 15.0,
        verify_probe_interval: float = 5.0,
    ) -> None:
        if not region or not instance_id or not account_host:
            raise ValueError('TDSQL region, instance ID and account host are required')
        if api_endpoint not in _ALLOWED_API_ENDPOINTS:
            raise ValueError('unsupported TDSQL API endpoint')
        if request_timeout_seconds <= 0 or flow_timeout_seconds <= 0 or flow_poll_interval < 0:
            raise ValueError('TDSQL timeout values are invalid')
        if verify_timeout_seconds <= 0 or verify_probe_interval < 0:
            raise ValueError('TDSQL timeout values are invalid')
        self.instance_id = instance_id
        self.database = validate_database_name(database)
        self.account_host = account_host
        self.request_timeout_seconds = request_timeout_seconds
        self.flow_timeout_seconds = flow_timeout_seconds
        self.flow_poll_interval = flow_poll_interval
        self.verify_host = verify_host
        self.verify_port = verify_port
        self.verify_timeout_seconds = verify_timeout_seconds
        self.verify_probe_interval = verify_probe_interval
        self.client = client or _create_client(
            region=region,
            api_endpoint=api_endpoint,
            secret_id=secret_id,
            secret_key=secret_key,
            request_timeout_seconds=request_timeout_seconds,
        )

    async def set_readonly_account(
        self, *, username: str, password: str, database: str, table_name: str, shared_tables: tuple[str, ...] = ()
    ) -> None:
        username = validate_account_name(username)
        self._require_database(database)
        table_name = validate_table_name(table_name)
        shared_tables = tuple(validate_table_name(t) for t in shared_tables)
        request = models.CreateUsersRequest()
        request.InstanceId = self.instance_id
        request.Users = [self._user(username)]
        # One password covers every entry in Users; we only ever create one account here.
        request.Password = password
        request.Description = 'Nexus principal trace read access'
        try:
            response = await self._call(self.client.CreateUsers, request)
        except TencentCloudSDKException as exc:
            # Kept in case the API does report the collision; the reset below covers both.
            if not _is_account_exists_error(exc):
                raise
        else:
            await self._wait_for_response_flow(response, username=username, action='CreateUsers')
        # CreateUsers on an existing account reports success without applying the password,
        # so it is set explicitly here to keep this method idempotent.
        await self.reset_password(username=username, password=password)
        await self._set_table_privileges(username=username, table_name=table_name, shared_tables=shared_tables)

    async def disable_account(self, *, username: str) -> None:
        await self._set_table_privileges(username=validate_account_name(username), table_name=None)

    async def restore_account(
        self, *, username: str, database: str, table_name: str, shared_tables: tuple[str, ...] = ()
    ) -> None:
        # Disabling revokes rather than locks, so restoring is exactly a reconcile; unlike
        # the MySQL provider there is no account lock left to clear afterwards.
        await self.reconcile_readonly_account(
            username=username, database=database, table_name=table_name, shared_tables=shared_tables
        )

    async def reconcile_readonly_account(
        self, *, username: str, database: str, table_name: str, shared_tables: tuple[str, ...] = ()
    ) -> None:
        self._require_database(database)
        table_name = validate_table_name(table_name)
        await self._set_table_privileges(
            username=validate_account_name(username), table_name=table_name, shared_tables=shared_tables
        )

    async def read_access_scope(self, *, username: str, database: str, table_name: str) -> str:
        username = validate_account_name(username)
        self._require_database(database)
        table_name = validate_table_name(table_name)
        for db, object_type, obj, scope in [
            ('*', '*', None, 'database'),
            (database, '*', None, 'database'),
            (database, 'table', table_name, 'table'),
        ]:
            request = models.DescribeUserPrivilegesRequest()
            request.InstanceId = self.instance_id
            request.UserName = username
            request.Host = self.account_host
            request.DbName = db
            request.ObjectType = object_type
            request.Object = obj
            request.ColName = '*'
            response = await self._call(self.client.DescribeUserPrivileges, request)
            if response.Privileges is None:
                return 'unknown'
            if any(p.upper() in {'SELECT', 'ALL', 'ALL PRIVILEGES'} for p in response.Privileges):
                return scope
        return 'none'

    async def reset_password(self, *, username: str, password: str) -> None:
        username = validate_account_name(username)
        reset = models.ResetUserPasswordInfo()
        reset.UserName = username
        reset.Host = self.account_host
        # Unlike CreateUsers, the password rides inside each list entry here.
        reset.Password = password
        request = models.ResetUsersPasswordRequest()
        request.InstanceId = self.instance_id
        request.Users = [reset]
        response = await self._call(self.client.ResetUsersPassword, request)
        await self._wait_for_response_flow(response, username=username, action='ResetUsersPassword', password=password)

    async def _set_table_privileges(
        self, *, username: str, table_name: str | None, shared_tables: tuple[str, ...] = ()
    ) -> None:
        table_privileges = []
        tables = sorted({validate_table_name(t) for t in shared_tables} | ({table_name} if table_name else set()))
        existing_tables = await self._table_inventory()
        # Sending non-granted tables with empty privilege lists is correct under both patch and
        # replace semantics, but requires a complete inventory -- hence the strict paging below.
        missing_tables = set(tables) - existing_tables
        if missing_tables:
            logger.warning(
                'Skipping missing trace tables for account %s in database %s: %s',
                username,
                self.database,
                ', '.join(sorted(missing_tables)),
            )
        for table in sorted(existing_tables):
            privilege = models.TablePrivileges()
            privilege.Database = self.database
            privilege.Table = table
            privilege.Privileges = ['SELECT'] if table in tables else []
            table_privileges.append(privilege)

        database_privilege = models.DatabasePrivileges()
        database_privilege.Database = self.database
        database_privilege.Privileges = []
        request = models.ModifyUserPrivilegesRequest()
        request.InstanceId = self.instance_id
        request.Users = [self._user(username)]
        request.GlobalPrivileges = []
        request.DatabasePrivileges = [database_privilege]
        request.TablePrivileges = table_privileges
        response = await self._call(self.client.ModifyUserPrivileges, request)
        logger.info(
            'TDSQL privilege operation submitted',
            extra={
                'username': username,
                'database': self.database,
                'request_id': getattr(response, 'RequestId', None),
            },
        )
        # ModifyUserPrivileges is documented as synchronous (no FlowId in its response),
        # unlike the DCDB counterpart. Poll anyway if a flow ever shows up.
        await self._wait_for_response_flow(response, username=username, action='ModifyUserPrivileges', required=False)

    async def _table_inventory(self) -> set[str]:
        tables: set[str] = set()
        offset = 0
        for _ in range(_INVENTORY_PAGE_LIMIT):
            request = models.DescribeDatabaseObjectsRequest()
            request.InstanceId = self.instance_id
            request.DbName = self.database
            request.Offset = offset
            request.Limit = _INVENTORY_PAGE_SIZE
            response = await self._call(self.client.DescribeDatabaseObjects, request)
            if response.Tables is None:
                raise RuntimeError('TDSQL table inventory is unavailable; refusing partial permission sync')
            page = {validate_table_name(item.Table) for item in response.Tables}
            tables |= page
            if len(response.Tables) < _INVENTORY_PAGE_SIZE:
                return tables
            offset += _INVENTORY_PAGE_SIZE
        raise RuntimeError('TDSQL table inventory exceeded the page limit; refusing partial permission sync')

    async def _wait_for_response_flow(
        self, response: Any, *, username: str, action: str, required: bool = True, password: str | None = None
    ) -> None:
        flow_id = getattr(response, 'FlowId', None)
        if flow_id is None and not required:
            return
        if isinstance(flow_id, bool) or not isinstance(flow_id, int) or flow_id <= 0:
            raise RuntimeError(f'TDSQL {action} returned an invalid flow ID')
        logger.info(
            'TDSQL account flow submitted',
            extra={
                'username': username,
                'action': action,
                'flow_id': flow_id,
                'request_id': getattr(response, 'RequestId', None),
            },
        )
        await self._wait_for_flow(flow_id, username=username, password=password)

    async def _password_accepted(self, username: str, password: str) -> bool:
        # A rejected password and an unreachable host both answer False, so anything
        # inconclusive leaves the flow polling rather than ending it early.
        if not self.verify_host:
            return False
        from nexus.runner.trace.tdsql._driver import connect as mysql_connect

        config = build_db_config(
            host=self.verify_host, port=self.verify_port, user=username, password=password, db=self.database
        )
        try:
            connection = await asyncio.wait_for(mysql_connect(**config), timeout=self.verify_timeout_seconds)
        except Exception as exc:
            logger.debug('TDSQL password probe failed', extra={'username': username, 'error': type(exc).__name__})
            return False
        connection.close()
        return True

    async def _wait_for_flow(self, flow_id: int, *, username: str | None = None, password: str | None = None) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.flow_timeout_seconds
        # Only a password reset can settle its own outcome by logging in; privilege and
        # creation flows pass none and keep waiting on the status alone.
        credential = (username, password) if username and password else None
        next_probe = loop.time() + self.verify_probe_interval
        while True:
            request = models.DescribeFlowRequest()
            request.FlowId = flow_id
            response = await self._call(self.client.DescribeFlow, request)
            status = getattr(response, 'Status', None)
            if not isinstance(status, str):
                # A missing or non-string field breaks the response contract itself, which
                # polling cannot resolve, unlike an unrecognised status value.
                raise RuntimeError(f'TDSQL account flow returned no status: {flow_id}')
            normalized = status.lower()
            if normalized == _FLOW_SUCCESS:
                return
            if normalized == _FLOW_FAILED:
                raise RuntimeError(f'TDSQL account flow failed: {flow_id}')
            # The password applies long before the flow settles, and a paused flow never
            # settles at all, so a successful login is the earlier and more reliable answer.
            if credential is not None and loop.time() >= next_probe:
                next_probe = loop.time() + self.verify_probe_interval
                if await self._password_accepted(*credential):
                    logger.warning(
                        'TDSQL account flow unsettled but the new password authenticates',
                        extra={'flow_id': flow_id, 'status': status},
                    )
                    return
            if loop.time() >= deadline:
                # Naming the last status matters: it separates a flow stuck before it ever
                # ran from one that ran too slowly.
                raise TimeoutError(f'TDSQL account flow timed out in status {status!r}: {flow_id}')
            await asyncio.sleep(self.flow_poll_interval)

    async def _call(self, operation, request):
        return await asyncio.wait_for(
            asyncio.to_thread(operation, request),
            timeout=self.request_timeout_seconds,
        )

    def _user(self, username: str):
        user = models.User()
        user.UserName = username
        user.Host = self.account_host
        return user


def _is_account_exists_error(exc: TencentCloudSDKException) -> bool:
    # The exact tdmysql code is not published alongside the SDK; match the family instead
    # of a single literal so a create-after-create degrades to a password reset either way.
    return 'alreadyexist' in str(getattr(exc, 'code', '')).replace('_', '').lower()


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
    return tdmysql_client.TdmysqlClient(
        credential.Credential(secret_id, secret_key),
        region,
        ClientProfile(httpProfile=http_profile),
    )


__all__ = ['TencentCloudTDMySQLTraceDatabaseAccountProvider']
