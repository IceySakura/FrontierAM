from __future__ import annotations

import asyncio
import json
import urllib.parse

from pathlib import Path
from typing import Any

import aiohttp

from nexus.cli.commands.submit import _upload


class CloudBatchApiError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, payload: object) -> None:
        super().__init__(f'{method} {url} returned {status}: {payload}')
        self.status = status
        self.payload = payload


class CloudBatchApi:
    def __init__(self, *, control_url: str, application_url: str, token: str) -> None:
        self.control_url = control_url.rstrip('/')
        self.application_url = application_url.rstrip('/')
        self.token = token
        self.consecutive_gateway_5xx = 0
        self.maximum_consecutive_gateway_5xx = 0
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> CloudBatchApi:
        await self._client()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def control_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request_json(method, f'{self.control_url}{path}', body=body)

    async def application_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request_json(method, f'{self.application_url}{path}', body=body)

    async def read_application_sse_until(
        self,
        path: str,
        event_type: str,
        *,
        timeout: float = 120,
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout
        session = await self._client()
        async with session.get(
            f'{self.application_url}{path}',
            headers=self._auth_headers,
            timeout=aiohttp.ClientTimeout(total=timeout + 5),
        ) as response:
            if response.status >= 400:
                raise CloudBatchApiError('GET', str(response.url), response.status, (await response.text())[:2048])
            current_event: str | None = None
            data_lines: list[str] = []
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    raw_line = await asyncio.wait_for(response.content.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if not raw_line:
                    break
                line = raw_line.decode('utf-8').rstrip('\r\n')
                if line.startswith('event:'):
                    current_event = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    data_lines.append(line.split(':', 1)[1].lstrip())
                elif not line:
                    if current_event == event_type and data_lines:
                        payload = json.loads('\n'.join(data_lines))
                        if not isinstance(payload, dict):
                            raise CloudBatchApiError(
                                'GET',
                                str(response.url),
                                response.status,
                                {'error': 'non_object_sse_event'},
                            )
                        return payload
                    current_event = None
                    data_lines.clear()
        raise TimeoutError(f'SSE event {event_type} not received from {path}')

    async def upload_artifact(
        self,
        path: Path,
        *,
        kind: str,
        sha256: str,
        size_bytes: int,
        idempotency_key: str,
        environment_id: str,
    ) -> dict[str, Any]:
        return await _upload(
            await self._client(),
            self.control_url,
            path,
            kind=kind,
            sha256=sha256,
            size_bytes=size_bytes,
            idempotency_key=idempotency_key,
            environment_id=environment_id,
            auth_headers=self._auth_headers,
        )

    async def list_all_batch_items(self, execution_id: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            suffix = f'&cursor={urllib.parse.quote(cursor)}' if cursor else ''
            page = await self.control_json(
                'GET',
                f'/v1/executions/{execution_id}/batch/items?limit=500{suffix}',
            )
            items.extend(page['items'])
            cursor = page.get('next_cursor')
            if not cursor:
                return items

    async def execution_facts(self, execution_ids: set[str]) -> dict[str, dict[str, Any]]:
        semaphore = asyncio.Semaphore(50)

        async def read(execution_id: str) -> dict[str, Any]:
            async with semaphore:
                return await self.control_json('GET', f'/v1/executions/{execution_id}')

        rows = await asyncio.gather(*(read(execution_id) for execution_id in sorted(execution_ids)))
        facts = {str(row['execution_id']): row for row in rows}
        missing = execution_ids - set(facts)
        if missing:
            raise AssertionError(f'missing child executions: {sorted(missing)[:20]}')
        return facts

    async def resolved_result(self, execution: dict[str, Any]) -> dict[str, Any]:
        result = execution.get('result_json') or {}
        if isinstance(result, dict) and result.get('_external') is True:
            return await self.application_json('GET', f'/v1/executions/{execution["execution_id"]}/result')
        if not isinstance(result, dict):
            raise AssertionError(f'execution result is not an object: {execution["execution_id"]}')
        return result

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.token}'}

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        attempts = 3 if method.upper() == 'GET' else 1
        session = await self._client()
        for attempt in range(attempts):
            try:
                async with session.request(method, url, json=body, headers=self._auth_headers) as response:
                    text = await response.text()
                    try:
                        payload = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        payload = {'body': text[:2048]}
                    if response.status >= 500:
                        self.consecutive_gateway_5xx += 1
                        self.maximum_consecutive_gateway_5xx = max(
                            self.maximum_consecutive_gateway_5xx,
                            self.consecutive_gateway_5xx,
                        )
                        if attempt + 1 < attempts:
                            await asyncio.sleep(2**attempt)
                            continue
                    else:
                        self.consecutive_gateway_5xx = 0
                    if response.status >= 400:
                        raise CloudBatchApiError(method, url, response.status, payload)
                    if not isinstance(payload, dict):
                        raise CloudBatchApiError(method, url, response.status, {'error': 'non_object_response'})
                    return payload
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(2**attempt)
        raise AssertionError('request retry loop exhausted')

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=60)
            )
        return self._session


__all__ = ['CloudBatchApi', 'CloudBatchApiError']
