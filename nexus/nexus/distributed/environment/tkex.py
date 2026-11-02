from __future__ import annotations

import hashlib
import json
import time

from typing import Any
from urllib.parse import urlencode

import aiohttp

from nexus.distributed.environment.kubernetes import (
    WorkloadConflictError,
    WorkloadNotFoundError,
    WorkloadPermanentError,
    _read_bounded_response_text,
    background_delete_options,
)
from nexus.distributed.environment.ports import WorkloadClient


class TkexWorkloadClient(WorkloadClient):
    def __init__(
        self,
        *,
        base_url: str,
        namespace: str,
        api_account: str,
        api_token: str,
        cluster_name: str,
        project_name: str,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip('/')
        self.namespace = namespace
        self.api_account = api_account
        self.api_token = api_token
        self.cluster_name = cluster_name
        self.project_name = project_name
        self._session = session
        self._owns_session = session is None

    async def get_service(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._service_path(name))

    async def create_service(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._service_path(), body=body)

    async def delete_service(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._service_path(name))

    async def list_pods(self) -> dict[str, Any]:
        return await self._request('GET', self._pod_path())

    async def get_pod(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._pod_path(name))

    def pod_delete_request(self, name: str, *, uid: str) -> tuple[str, dict[str, Any]]:
        body: dict[str, Any] = {
            **background_delete_options(),
            'gracePeriodSeconds': 0,
            'preconditions': {'uid': uid},
        }
        return self._pod_path(name), body

    async def delete_pod(self, name: str, *, uid: str) -> dict[str, Any]:
        path, body = self.pod_delete_request(name, uid=uid)
        return await self._request('DELETE', path, body=body)

    async def get_secret(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._secret_path(name))

    async def create_secret(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._secret_path(), body=body)

    async def patch_secret(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('PATCH', self._secret_path(name), body=body)

    async def delete_secret(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._secret_path(name))

    async def get_job(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._job_path(name))

    async def create_job(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._job_path(), body=body)

    async def delete_job(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._job_path(name), body=background_delete_options())

    async def list_job_pods(self, job_name: str) -> list[dict[str, Any]]:
        query = urlencode({'labelSelector': f'job-name={job_name}'})
        return _response_items(await self._request('GET', f'{self._pod_path()}?{query}'))

    async def list_resource_events(self, resource_name: str) -> list[dict[str, Any]]:
        query = urlencode({'fieldSelector': f'involvedObject.name={resource_name}'})
        return _response_items(await self._request('GET', f'{self._event_path()}?{query}'))

    async def read_pod_log(
        self,
        pod_name: str,
        *,
        container: str,
        tail_lines: int,
        limit_bytes: int,
    ) -> str:
        query = urlencode({
            'container': container,
            'tailLines': tail_lines,
            'timestamps': 'true',
            'limitBytes': limit_bytes,
        })
        return await self._request_text(f'{self._pod_path(pod_name)}/log?{query}', limit_bytes=limit_bytes)

    async def get_stateful_workload(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._stateful_workload_path(name))

    async def create_stateful_workload(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._stateful_workload_path(), body=body)

    async def patch_stateful_workload(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('PATCH', self._stateful_workload_path(name), body=body)

    async def delete_stateful_workload(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._stateful_workload_path(name), body=background_delete_options())

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _stateful_workload_path(self, name: str | None = None) -> str:
        path = f'/v2/forward/stke/apis/platform.stke/v1alpha1/namespaces/{self.namespace}/statefulsetpluses'
        return f'{path}/{name}' if name else path

    def _service_path(self, name: str | None = None) -> str:
        path = f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/services'
        return f'{path}/{name}' if name else path

    def _pod_path(self, name: str | None = None) -> str:
        path = f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/pods'
        return f'{path}/{name}' if name else path

    def _event_path(self) -> str:
        return f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/events'

    def _secret_path(self, name: str | None = None) -> str:
        path = f'/v2/forward/stke/api/v1/namespaces/{self.namespace}/secrets'
        return f'{path}/{name}' if name else path

    def _job_path(self, name: str | None = None) -> str:
        path = f'/v2/forward/stke/apis/batch/v1/namespaces/{self.namespace}/jobs'
        return f'{path}/{name}' if name else path

    async def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
        timestamp = str(int(time.time()))
        signature = self._signature(timestamp, path)
        headers = {
            'TKEX-Api-Account': self.api_account,
            'TKEX-Api-TimeStamp': timestamp,
            'TKEX-Api-Sig': signature,
            'X-TKE-ClusterName': self.cluster_name,
            'X-Request-ProjectName': self.project_name,
            'Accept': 'application/json',
        }
        if body is not None:
            headers['Content-Type'] = 'application/merge-patch+json' if method == 'PATCH' else 'application/json'
        session = await self._client()
        async with session.request(
            method,
            f'{self.base_url}{path}',
            headers=headers,
            data=json.dumps(body) if body is not None else None,
        ) as response:
            text = await response.text()
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                if response.status < 400:
                    raise WorkloadPermanentError('tkex_api_invalid_json_response') from None
                payload = {'message': text[:1024]}
            if response.status == 404:
                raise WorkloadNotFoundError(path)
            if response.status == 409:
                raise WorkloadConflictError(path)
            if 400 <= response.status < 500 and response.status != 429:
                raise WorkloadPermanentError(f'tkex_api_{response.status}: {payload}')
            if response.status >= 400:
                raise RuntimeError(f'tkex_api_{response.status}: {payload}')
            if not isinstance(payload, dict):
                raise WorkloadPermanentError('tkex_api_invalid_response_type')
            self._raise_for_body_error(path, payload)
            return self._normalize_success_payload(payload)

    async def _request_text(self, path: str, *, limit_bytes: int) -> str:
        timestamp = str(int(time.time()))
        signature = self._signature(timestamp, path)
        headers = {
            'TKEX-Api-Account': self.api_account,
            'TKEX-Api-TimeStamp': timestamp,
            'TKEX-Api-Sig': signature,
            'X-TKE-ClusterName': self.cluster_name,
            'X-Request-ProjectName': self.project_name,
            'Accept': 'text/plain',
        }
        session = await self._client()
        async with session.get(f'{self.base_url}{path}', headers=headers) as response:
            text = await _read_bounded_response_text(response, limit_bytes=limit_bytes)
            try:
                payload = json.loads(text) if text else {}
            except json.JSONDecodeError:
                payload = None
            error_payload = payload if isinstance(payload, dict) else {'message': text[:1024]}
            if response.status == 404:
                raise WorkloadNotFoundError(path)
            if 400 <= response.status < 500 and response.status != 429:
                raise WorkloadPermanentError(f'tkex_api_{response.status}: {error_payload}')
            if response.status >= 400:
                raise RuntimeError(f'tkex_api_{response.status}: {error_payload}')
            if isinstance(payload, dict) and 'code' in payload:
                self._raise_for_body_error(path, payload)
                data = payload.get('data')
                if isinstance(data, str):
                    return data
                raise WorkloadPermanentError('tkex_api_invalid_log_response')
            return text

    def _signature(self, timestamp: str, path: str) -> str:
        request_path = path.partition('?')[0]
        return hashlib.sha256(f'{self.api_token}{timestamp}{request_path}'.encode()).hexdigest()

    @staticmethod
    def _normalize_success_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if 'code' not in payload:
            return payload
        data = payload.get('data')
        if not isinstance(data, dict):
            raise WorkloadPermanentError('tkex_api_invalid_success_envelope')
        return data

    @staticmethod
    def _raise_for_body_error(path: str, payload: dict[str, Any]) -> None:
        code = payload.get('code')
        if code in {None, 0, '0'}:
            return
        if code in {404, '404'}:
            raise WorkloadNotFoundError(path)
        if code in {409, '409'}:
            raise WorkloadConflictError(path)
        reason = str(payload.get('reason') or '')
        message = str(payload.get('message') or '')
        normalized_reason = reason.replace('_', '').replace('-', '').lower()
        if normalized_reason in {'notfound', 'resourcenotfound'}:
            raise WorkloadNotFoundError(path)
        if normalized_reason in {'alreadyexists', 'conflict', 'resourceconflict'}:
            raise WorkloadConflictError(path)
        raise WorkloadPermanentError(f'tkex_api_body_error: code={code}, reason={reason}, message={message}')

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self._session


def _response_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get('items')
    if not isinstance(items, list):
        raise WorkloadPermanentError('tkex_api_invalid_list_response')
    return [item for item in items if isinstance(item, dict)]
