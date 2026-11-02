"""Legacy client for the old dataset-manager service.

This is kept only for task integrations that still depend on the pre-compose
dataset-manager API. New code should use :mod:`nexus.dataset`.
"""

from __future__ import annotations

import os

from typing import Any

import aiohttp

from nexus.utils.async_http import AsyncHTTP


DEFAULT_SERVER_URL = 'http://hunyuan-dataset.woa.com'
ENV_SERVER_URL = 'NEXUS_DATASET_SERVER_URL'
ENV_DATASET_TOKEN = 'NEXUS_DATASET_TOKEN'
API_TIMEOUT_SECONDS = 10


class DatasetClientError(Exception):
    """Base exception for legacy dataset client errors."""


class DatasetClient:
    """Client for the old dataset-manager server."""

    def __init__(self, server_url: str | None = None, auth_token: str | None = None) -> None:
        self.server_url = (server_url or os.getenv(ENV_SERVER_URL, DEFAULT_SERVER_URL)).rstrip('/')
        self._auth_token = auth_token or os.getenv(ENV_DATASET_TOKEN)

        http_kwargs: dict[str, Any] = {'timeout': aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)}
        if self._auth_token:
            http_kwargs['headers'] = {'Authorization': f'Bearer {self._auth_token}'}
        self._http = AsyncHTTP(**http_kwargs)

    async def list_datasets(self) -> list[dict[str, str]]:
        url = f'{self.server_url}/api/dataset/list'
        response = await self._http.get(url)
        data = self._parse_response(response)
        return data.get('datasets', [])

    async def get_meta(self, namespace: str, dataset_name: str) -> dict[str, Any]:
        url = f'{self.server_url}/api/dataset/{namespace}/{dataset_name}/meta'
        response = await self._http.get(url)
        return self._parse_response(response)

    async def list_instances(self, namespace: str, dataset_name: str, split: str | None = None) -> dict[str, Any]:
        url = f'{self.server_url}/api/dataset/{namespace}/{dataset_name}/list'
        params = {}
        if split:
            params['split'] = split
        response = await self._http.get(url, params=params)
        return self._parse_response(response)

    async def get_instance_url(
        self,
        namespace: str,
        dataset_name: str,
        instance_id: str,
        split: str | None = None,
    ) -> dict[str, Any]:
        url = f'{self.server_url}/api/dataset/{namespace}/{dataset_name}/get/{instance_id}'
        params = {}
        if split:
            params['split'] = split
        response = await self._http.get(url, params=params)
        return self._parse_response(response)

    async def get_instance(
        self,
        namespace: str,
        dataset_name: str,
        instance_id: str,
        split: str | None = None,
    ) -> dict[str, Any]:
        url_info = await self.get_instance_url(namespace, dataset_name, instance_id, split)
        signed_url = url_info['url']

        response = await self._http.get(signed_url)
        if not response.ok:
            raise DatasetClientError(f'Failed to download instance: HTTP {response.status}')
        return response.json()

    async def sign_urls(
        self,
        namespace: str,
        dataset_name: str,
        files: list[str],
        expires: int | None = None,
    ) -> dict[str, str]:
        url = f'{self.server_url}/api/dataset/{namespace}/{dataset_name}/sign'
        body: dict[str, Any] = {'files': files}
        if expires:
            body['expires'] = expires
        response = await self._http.post(url, json=body)
        data = self._parse_response(response)
        return data.get('urls', {})

    async def close(self) -> None:
        await self._http.close()

    def _parse_response(self, response: Any) -> dict[str, Any]:
        if not response.ok:
            try:
                error = response.json()
                detail = error.get('detail', response.text)
            except Exception:
                detail = response.text
            raise DatasetClientError(f'API error (HTTP {response.status}): {detail}')

        result = response.json()
        if result.get('code', 0) != 0:
            raise DatasetClientError(f'API error: {result.get("message", "Unknown error")}')

        return result.get('data', {})

    async def __aenter__(self) -> DatasetClient:
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()


def parse_dataset_path(path: str) -> tuple[str, str]:
    parts = path.strip('/').split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid dataset path: '{path}'. Expected format: namespace/dataset_name")
    return parts[0], parts[1]


def list_instances_sync(
    dataset: str,
    split: str | None = None,
    server_url: str | None = None,
    auth_token: str | None = None,
) -> list[str]:
    """Synchronously list instance ids from the legacy dataset-manager service."""
    import asyncio
    import concurrent.futures

    namespace, dataset_name = parse_dataset_path(dataset)

    async def _fetch() -> list[str]:
        async with DatasetClient(server_url, auth_token) as client:
            result = await client.list_instances(namespace, dataset_name, split)
            return result.get('ids', [])

    # Check if we're already in an async context
    try:
        asyncio.get_running_loop()
        # We're in an async context, run in a new thread
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _fetch())
            return future.result()  # type: ignore[return-value]
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(_fetch())
