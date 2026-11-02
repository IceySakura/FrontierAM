"""Async client for Unity Catalog Nexus asset metadata."""

from __future__ import annotations

import os

from typing import Any

import aiohttp

from nexus.dataset.auth import (
    DatasetAuthenticator,
    TaiPatAuthenticator,
    build_default_authenticator,
    format_dataset_auth_error,
    prefer_tai_pat,
)
from nexus.dataset.exceptions import DatasetClientError, UcAssetNotFoundError


ENV_UC_BASE_URL = 'UC_BASE_URL'
# Legacy TAuth-backed UC endpoint (direct service, no Taihu gateway).
DEFAULT_UC_BASE_URL = 'http://taiji-nexus-dataset-uc-dev.woa.com:8080'
# Taihu MCP gateway endpoint, used when authenticating with a Taihu PAT.
DEFAULT_TAIJI_UC_BASE_URL = 'https://taiji-nexus-dataset-uc-dev.mcp.woa.com'
NEXUS_CATALOG = 'taiji'
NEXUS_SCHEMA = 'nexus'
DEFAULT_LIST_BY_SCHEMA_PATH = '/api/1.0/unity-catalog/taiji/ListAssetMetasBySchema'
DEFAULT_GET_PATH = '/api/1.0/unity-catalog/taiji/GetAssetMeta'
DEFAULT_CREATE_NEXUS_PATH = '/api/1.0/unity-catalog/taiji/CreateNexusAssetMeta'
DEFAULT_UPDATE_PATH = '/api/1.0/unity-catalog/taiji/UpdateAssetMeta'
DEFAULT_NAMESPACE_PERMISSION_PATH = '/api/1.0/unity-catalog/asset-namespace-permissions'


class UcMetadataClient:
    """Client for UC dataset asset list/show operations."""

    def __init__(
        self,
        *,
        uc_base_url: str | None = None,
        authenticator: DatasetAuthenticator | None = None,
        auth_token: str | None = None,
        session: aiohttp.ClientSession | None = None,
        catalog: str = NEXUS_CATALOG,
        schema: str = NEXUS_SCHEMA,
    ) -> None:
        self._prefer_tai_pat = prefer_tai_pat(auth_token=auth_token, authenticator=authenticator)
        # Priority: explicit arg > env override > auth-mode default. Resolving the
        # env var before the auth-mode default keeps a user-set UC_BASE_URL from
        # being silently overridden in PAT mode.
        resolved_uc_base_url = uc_base_url if uc_base_url is not None else os.getenv(ENV_UC_BASE_URL)
        if resolved_uc_base_url is None:
            resolved_uc_base_url = DEFAULT_TAIJI_UC_BASE_URL if self._prefer_tai_pat else DEFAULT_UC_BASE_URL
        self.uc_base_url = resolved_uc_base_url.rstrip('/')
        self.auth_token = auth_token
        self.authenticator = None if self.auth_token is not None else authenticator
        self.session = session
        self.catalog = catalog
        self.schema = schema

    def _uses_tai_pat(self) -> bool:
        """Return ``True`` when the active authentication mode is Taihu PAT."""

        if self.auth_token is not None:
            return False
        if self.authenticator is not None:
            return isinstance(self.authenticator, TaiPatAuthenticator)
        return self._prefer_tai_pat

    def asset_name(self, namespace: str, dataset: str) -> str:
        return f'{namespace}/{dataset}'

    def full_name(self, namespace: str, dataset: str) -> str:
        return f'{self.catalog}.{self.schema}.{self.asset_name(namespace, dataset)}'

    async def get_asset(self, namespace: str, dataset: str) -> dict[str, Any]:
        return await self._post(
            DEFAULT_GET_PATH,
            {'FullName': self.full_name(namespace, dataset)},
            asset_not_found=True,
        )

    async def list_assets(
        self,
        *,
        asset_name: str | None = None,
        page_number: int | None = None,
        page_size: int | None = None,
    ) -> dict[str, Any]:
        body = _drop_none({
            'SchemaName': self.schema,
            'AssetName': asset_name,
            'PageNumber': page_number,
            'PageSize': page_size,
        })
        return await self._post(DEFAULT_LIST_BY_SCHEMA_PATH, body)

    async def register_nexus_asset(
        self,
        namespace: str,
        dataset: str,
        *,
        creator: str,
        admin_list: list[str],
        visibility: str = 'PRIVATE',
        description: str | None = None,
        status: str | None = None,
        properties: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not creator:
            raise ValueError('creator must not be empty')
        admins = [admin for admin in admin_list if admin]
        if not admins:
            raise ValueError('admin_list must not be empty')
        body: dict[str, Any] = {
            'Namespace': namespace,
            'DatasetName': dataset,
            'Creator': creator,
            'AdminList': admins,
            'Visibility': visibility or 'PRIVATE',
            'AssetDesc': description or '',
            'Status': status or 'PENDING',
        }
        if properties:
            body['Properties'] = {key: value for key, value in properties.items() if key and value is not None}
        return await self._post(DEFAULT_CREATE_NEXUS_PATH, body)

    async def update_asset(
        self,
        namespace: str,
        dataset: str,
        *,
        admin_list: list[str] | None = None,
        visibility: str | None = None,
        description: str | None = None,
        status: str | None = None,
        properties: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {'FullName': self.full_name(namespace, dataset)}
        if admin_list is not None:
            body['AdminList'] = [admin for admin in admin_list if admin]
        if visibility is not None:
            body['Visibility'] = visibility
        if description is not None:
            body['AssetDesc'] = description
        if status is not None:
            body['Status'] = status
        if properties is not None:
            body['Properties'] = {key: value for key, value in properties.items() if key and value is not None}
        return await self._post(DEFAULT_UPDATE_PATH, body)

    async def create_namespace_permission(self, namespace: str, admin_list: list[str]) -> dict[str, Any]:
        """Create the Nexus namespace permission record for a set of administrators."""

        namespace, admins = _validate_namespace_permission(namespace, admin_list)
        return await self._request_namespace_permission(
            'POST',
            json={
                'schema': NEXUS_SCHEMA,
                'namespace': namespace,
                'admin_list': admins,
            },
        )

    async def get_namespace_permission(self, namespace: str) -> dict[str, Any] | None:
        """Return the Nexus namespace permission record, or ``None`` when it is absent."""

        if not namespace:
            raise ValueError('namespace must not be empty')
        response = await self._request_namespace_permission(
            'GET',
            params={'schema': NEXUS_SCHEMA, 'namespace': namespace},
        )
        records = response.get('data')
        if isinstance(records, list) and records and isinstance(records[0], dict):
            return records[0]
        return None

    async def update_namespace_permission(self, namespace: str, admin_list: list[str]) -> dict[str, Any]:
        """Replace the administrator list for an existing Nexus namespace permission record."""

        namespace, admins = _validate_namespace_permission(namespace, admin_list)
        return await self._request_namespace_permission(
            'PATCH',
            json={
                'schema': NEXUS_SCHEMA,
                'namespace': namespace,
                'admin_list': admins,
            },
        )

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        asset_not_found: bool = False,
    ) -> dict[str, Any]:
        session = self._session()
        headers = {'Content-Type': 'application/json'}
        headers.update(await self._auth_headers(session))
        async with session.post(f'{self.uc_base_url}{path}', json=body, headers=headers) as response:
            return await _unwrap_uc_response(response, asset_not_found=asset_not_found, taihu=self._uses_tai_pat())

    async def _request_namespace_permission(
        self,
        method: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        session = self._session()
        headers = {'Content-Type': 'application/json'}
        headers.update(await self._auth_headers(session))
        async with session.request(
            method,
            f'{self.uc_base_url}{DEFAULT_NAMESPACE_PERMISSION_PATH}',
            json=json,
            params=params,
            headers=headers,
        ) as response:
            return await _unwrap_namespace_permission_response(response)

    async def _auth_headers(self, session: aiohttp.ClientSession) -> dict[str, str]:
        if self.auth_token is not None:
            return {'secure-authentication': self.auth_token}
        if self.authenticator is None:
            self.authenticator = build_default_authenticator()
        return await self.authenticator.get_headers(session)

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None:
            raise RuntimeError('UcMetadataClient requires an aiohttp session')
        return self.session


async def _unwrap_uc_response(
    response: aiohttp.ClientResponse,
    *,
    asset_not_found: bool = False,
    taihu: bool = False,
) -> dict[str, Any]:
    try:
        body = await response.json(content_type=None)
    except Exception as exc:
        text = await response.text()
        if response.status == 401:
            raise DatasetClientError(
                format_dataset_auth_error(f'UC HTTP 401: {text[:300]}', taihu=taihu),
                http_status=response.status,
            ) from exc
        if response.status == 404:
            error_cls = UcAssetNotFoundError if asset_not_found else DatasetClientError
            resource = 'asset' if asset_not_found else 'resource'
            raise error_cls(f'UC {resource} not found (HTTP 404): {text[:300]}', http_status=404) from exc
        raise DatasetClientError(
            f'UC non-JSON response (HTTP {response.status}): {text[:300]}', http_status=response.status
        ) from exc

    response_body = body.get('Response') if isinstance(body, dict) else None
    error = response_body.get('Error') if isinstance(response_body, dict) else None
    if error:
        code = str(error.get('Code') or '')
        message = str(error.get('Message') or error)
        error_message = f'UC error (code={code}): {message}'
        if response.status == 401:
            error_message = format_dataset_auth_error(error_message, taihu=taihu)
        error_cls = UcAssetNotFoundError if asset_not_found and _is_uc_asset_not_found(code) else DatasetClientError
        raise error_cls(error_message, code=code, http_status=response.status)

    if response.status == 401:
        raise DatasetClientError(
            format_dataset_auth_error(f'UC HTTP 401: {_uc_error_reason(body)}', taihu=taihu),
            http_status=response.status,
        )
    if response.status == 404:
        error_cls = UcAssetNotFoundError if asset_not_found else DatasetClientError
        resource = 'asset' if asset_not_found else 'resource'
        raise error_cls(f'UC {resource} not found (HTTP 404): {_uc_error_reason(body)}', http_status=404)
    if response.status < 200 or response.status >= 300:
        raise DatasetClientError(f'UC HTTP {response.status}: {body}', http_status=response.status)

    if response_body is None:
        raise DatasetClientError(
            f"UC response missing 'Response' (HTTP {response.status})", http_status=response.status
        )
    data = response_body.get('Data') or {}
    return data if isinstance(data, dict) else {'Data': data}


async def _unwrap_namespace_permission_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        body = await response.json(content_type=None)
    except Exception:
        body = None

    if response.status < 200 or response.status >= 300:
        text = await response.text()
        message = body.get('message') if isinstance(body, dict) else None
        raise DatasetClientError(
            f'UC namespace permission HTTP {response.status}: {message or text[:300]}',
            http_status=response.status,
        )
    return body if isinstance(body, dict) else {}


def _is_uc_asset_not_found(code: str) -> bool:
    return code == 'ResourceNotFound.MetaResourceNotFound' or code.endswith('.MetaResourceNotFound')


def _uc_error_reason(body: Any) -> str:
    if not isinstance(body, dict):
        return str(body)
    error = body.get('error')
    message = body.get('message')
    if error and message:
        return f'{error}: {message}'
    if message:
        return str(message)
    if error:
        return str(error)
    return str(body)


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _validate_namespace_permission(namespace: str, admin_list: list[str]) -> tuple[str, list[str]]:
    if not namespace:
        raise ValueError('namespace must not be empty')
    admins = [admin for admin in admin_list if admin]
    if not admins:
        raise ValueError('admin_list must not be empty')
    return namespace, admins
