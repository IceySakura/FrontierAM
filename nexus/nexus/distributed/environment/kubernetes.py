from __future__ import annotations

import json
import os
import ssl

from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp

from nexus.distributed.environment.ports import WorkloadClient


class WorkloadNotFoundError(Exception):
    pass


class WorkloadConflictError(Exception):
    pass


class WorkloadPermanentError(RuntimeError):
    pass


class KubernetesWorkloadClient(WorkloadClient):
    def __init__(
        self,
        *,
        namespace: str,
        api_url: str = '',
        token: str | None = None,
        ca_certificate: str | None = None,
        token_file: str = '/var/run/secrets/kubernetes.io/serviceaccount/token',
        ca_file: str = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt',
        tls_strict: bool = True,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.namespace = namespace
        self.api_url = (api_url or _in_cluster_api_url()).rstrip('/')
        self.token = token
        self.ca_certificate = ca_certificate
        self.token_file = Path(token_file)
        self.ca_file = Path(ca_file)
        self.tls_strict = tls_strict
        self._session = session
        self._owns_session = session is None
        self._session_ca_identity: tuple[int, int, int, int] | str | None = None

    async def get_service(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._service_url(name))

    async def create_service(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._service_url(), body=body)

    async def delete_service(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._service_url(name))

    async def get_secret(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._secret_url(name))

    async def create_secret(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._secret_url(), body=body)

    async def patch_secret(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            'PATCH',
            self._secret_url(name),
            body=body,
            content_type='application/merge-patch+json',
        )

    async def delete_secret(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._secret_url(name))

    async def get_job(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._job_url(name))

    async def create_job(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._job_url(), body=body)

    async def delete_job(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._job_url(name), body=background_delete_options())

    async def list_job_pods(self, job_name: str) -> list[dict[str, Any]]:
        query = urlencode({'labelSelector': f'job-name={job_name}'})
        return _response_items(await self._request('GET', f'{self._pod_url()}?{query}'))

    async def list_resource_events(self, resource_name: str) -> list[dict[str, Any]]:
        query = urlencode({'fieldSelector': f'involvedObject.name={resource_name}'})
        return _response_items(await self._request('GET', f'{self._event_url()}?{query}'))

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
        return await self._request_text(f'{self._pod_url(pod_name)}/log?{query}', limit_bytes=limit_bytes)

    async def get_stateful_workload(self, name: str) -> dict[str, Any]:
        return await self._request('GET', self._statefulset_url(name))

    async def create_stateful_workload(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', self._statefulset_url(), body=body)

    async def patch_stateful_workload(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._request(
            'PATCH',
            self._statefulset_url(name),
            body=body,
            content_type='application/merge-patch+json',
        )

    async def delete_stateful_workload(self, name: str) -> dict[str, Any]:
        return await self._request('DELETE', self._statefulset_url(name), body=background_delete_options())

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None
            self._session_ca_identity = None

    def _statefulset_url(self, name: str | None = None) -> str:
        path = f'/apis/apps/v1/namespaces/{self.namespace}/statefulsets'
        return f'{self.api_url}{path}/{name}' if name else f'{self.api_url}{path}'

    def _service_url(self, name: str | None = None) -> str:
        path = f'/api/v1/namespaces/{self.namespace}/services'
        return f'{self.api_url}{path}/{name}' if name else f'{self.api_url}{path}'

    def _secret_url(self, name: str | None = None) -> str:
        path = f'/api/v1/namespaces/{self.namespace}/secrets'
        return f'{self.api_url}{path}/{name}' if name else f'{self.api_url}{path}'

    def _job_url(self, name: str | None = None) -> str:
        path = f'/apis/batch/v1/namespaces/{self.namespace}/jobs'
        return f'{self.api_url}{path}/{name}' if name else f'{self.api_url}{path}'

    def _pod_url(self, name: str | None = None) -> str:
        path = f'/api/v1/namespaces/{self.namespace}/pods'
        return f'{self.api_url}{path}/{name}' if name else f'{self.api_url}{path}'

    def _event_url(self) -> str:
        return f'{self.api_url}/api/v1/namespaces/{self.namespace}/events'

    async def _request(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        content_type: str = 'application/json',
    ) -> dict[str, Any]:
        session = await self._client()
        token = self.token if self.token is not None else self.token_file.read_text().strip()
        headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/json'}
        if body is not None:
            headers['Content-Type'] = content_type
        async with session.request(
            method, url, headers=headers, data=json.dumps(body) if body is not None else None
        ) as response:
            payload = await _json_response(response)
            if response.status == 404:
                raise WorkloadNotFoundError(url)
            if response.status == 409:
                raise WorkloadConflictError(url)
            if 400 <= response.status < 500 and response.status != 429:
                raise WorkloadPermanentError(f'kubernetes_api_{response.status}: {payload}')
            if response.status >= 400:
                raise RuntimeError(f'kubernetes_api_{response.status}: {payload}')
            return payload

    async def _request_text(self, url: str, *, limit_bytes: int) -> str:
        session = await self._client()
        token = self.token if self.token is not None else self.token_file.read_text().strip()
        headers = {'Authorization': f'Bearer {token}', 'Accept': 'text/plain'}
        async with session.get(url, headers=headers) as response:
            text = await _read_bounded_response_text(response, limit_bytes=limit_bytes)
            payload = _decode_response_text(text)
            if response.status == 404:
                raise WorkloadNotFoundError(url)
            if 400 <= response.status < 500 and response.status != 429:
                raise WorkloadPermanentError(f'kubernetes_api_{response.status}: {payload}')
            if response.status >= 400:
                raise RuntimeError(f'kubernetes_api_{response.status}: {payload}')
            return text

    async def _client(self) -> aiohttp.ClientSession:
        ca_identity = self._ca_file_identity()
        if self._owns_session and self._session is not None and self._session_ca_identity != ca_identity:
            await self._session.close()
            self._session = None
        if self._session is None:
            ssl_context: ssl.SSLContext | bool = True
            if self.ca_certificate is not None:
                ssl_context = ssl.create_default_context(cadata=self.ca_certificate)
                if not self.tls_strict:
                    ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
            elif self.ca_file.is_file():
                ssl_context = ssl.create_default_context(cafile=str(self.ca_file))
                if not self.tls_strict:
                    ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=ssl_context),
                timeout=aiohttp.ClientTimeout(total=30),
            )
            self._session_ca_identity = ca_identity
        return self._session

    def _ca_file_identity(self) -> tuple[int, int, int, int] | str | None:
        if self.ca_certificate is not None:
            return 'inline'
        try:
            stat = self.ca_file.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _in_cluster_api_url() -> str:
    host = os.getenv('KUBERNETES_SERVICE_HOST')
    port = os.getenv('KUBERNETES_SERVICE_PORT_HTTPS') or os.getenv('KUBERNETES_SERVICE_PORT') or '443'
    if not host:
        raise RuntimeError('kubernetes_api_url_unavailable')
    return f'https://{host}:{port}'


def background_delete_options() -> dict[str, str]:
    return {
        'apiVersion': 'v1',
        'kind': 'DeleteOptions',
        'propagationPolicy': 'Background',
    }


async def _json_response(response: aiohttp.ClientResponse) -> dict[str, Any]:
    text = await response.text()
    return _decode_response_text(text)


def _decode_response_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {'message': text[:1024]}
    return value if isinstance(value, dict) else {'value': value}


async def _read_bounded_response_text(response: aiohttp.ClientResponse, *, limit_bytes: int) -> str:
    if limit_bytes < 1:
        raise ValueError('response byte limit must be positive')
    content = bytearray()
    async for chunk in response.content.iter_chunked(min(64 * 1024, limit_bytes)):
        remaining = limit_bytes - len(content)
        if remaining <= 0:
            break
        content.extend(chunk[:remaining])
        if len(content) >= limit_bytes:
            break
    return bytes(content).decode(response.charset or 'utf-8', errors='replace')


def _response_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get('items')
    if not isinstance(items, list):
        raise WorkloadPermanentError('kubernetes_api_invalid_list_response')
    return [item for item in items if isinstance(item, dict)]
