"""DevCloud macOS RuntimeProvider."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import time

from collections.abc import Mapping
from typing import Any, cast

import aiohttp

from nexus.runtime.providers.base import InvalidPersistentRuntimeError, RuntimeProvider
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.async_http import http_client
from nexus.utils.timeout import async_timeout


logger = logging.getLogger(__name__)


_FAILED_STATUSES = {'failed', 'error', 'canceled', 'cancelled'}
_LOG_METADATA_KEYS = ('status', 'ip', 'id', 'assetId', 'os', 'name', 'cpu', 'memory', 'disk', 'createdAt', 'taskId')


def _log_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return non-sensitive DevCloud metadata for logs."""
    return {key: metadata[key] for key in _LOG_METADATA_KEYS if key in metadata}


@runtime_provider_registry.register('devcloud')
class DevCloudRuntimeProvider(RuntimeProvider):
    """Provision and release DevCloud macOS sandboxes through the sandbox API."""

    DEFAULT_LEASE_TIME = 300
    DEFAULT_SERVER_PORT = 5000
    DEFAULT_REQUEST_TIMEOUT = 1800.0
    DEFAULT_CONTROL_REQUEST_TIMEOUT = 30.0
    DEFAULT_RELEASE_TIMEOUT = 120.0
    DEFAULT_INSTANCE_READY_TIMEOUT = 300.0
    TASK_POLL_INTERVAL = 2.0

    def __init__(
        self,
        *,
        app_id: str | None = None,
        app_token: str | None = None,
        sandbox_api_base_url: str,
        proxy_url: str | None = None,
        execute_mode: str = 'sync',
        lease_time: int = DEFAULT_LEASE_TIME,
        server_port: int = DEFAULT_SERVER_PORT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        control_request_timeout: float = DEFAULT_CONTROL_REQUEST_TIMEOUT,
        release_timeout: float = DEFAULT_RELEASE_TIMEOUT,
        instance_ready_timeout: float = DEFAULT_INSTANCE_READY_TIMEOUT,
        **config: Any,
    ) -> None:
        # Older YAMLs may still carry these fields; the sandbox API no longer uses them.
        legacy_fields = {'control_api_base_url', 'project', 'pipeline_id', 'build_id', 'vm_seq_id', 'model'}
        ignored = sorted(legacy_fields.intersection(config))
        if ignored:
            logger.warning('DevCloud ignores obsolete configuration fields: %s', ', '.join(ignored))
            for key in ignored:
                config.pop(key)
        super().__init__(**config)
        self.app_id = app_id
        self.app_token = app_token
        self.sandbox_api_base_url = sandbox_api_base_url.rstrip('/')
        self.lease_time = int(lease_time)
        self.server_port = int(server_port)
        self.request_timeout = float(request_timeout)
        self.instance_ready_timeout = float(instance_ready_timeout)
        self._task_ids: dict[str, str] = {}
        self._vm_ids: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._external_instance_names: set[str] = set()
        self._acquire_locks: dict[str, asyncio.Lock] = {}
        self.proxy_url = proxy_url
        if execute_mode not in {'sync', 'async'}:
            raise ValueError('execute_mode must be sync or async')
        self.execute_mode = execute_mode
        self.control_request_timeout = float(control_request_timeout)
        self.release_timeout = float(release_timeout)
        for field in ('control_request_timeout', 'release_timeout'):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field} must be finite and positive')
        self._delete_task_ids: dict[str, str] = {}  # Instance id -> accepted deletion task, retained if unconfirmed.

    def _build_auth_headers(self, *, timestamp: str | None = None, random_digits: str | None = None) -> dict[str, str]:
        resolved_timestamp = timestamp or str(int(time.time()))
        resolved_random = random_digits or ''.join(str(random.randint(0, 9)) for _ in range(8))
        raw = f'{self.app_token}{resolved_timestamp}{resolved_random}'
        enckey = hashlib.md5(raw.encode('utf-8')).hexdigest().lower()
        return {
            'APPID': str(self.app_id),
            'TIMESTP': resolved_timestamp,
            'RANDOM': resolved_random,
            'ENCKEY': enckey,
        }

    def get_runtime_id(self, name: str) -> str | None:
        return self._vm_ids.get(name) or self._task_ids.get(name)

    async def acquire(self, name: str, **params: Any) -> DevCloudRuntime:
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()
        async with self._acquire_locks[name]:
            return await self._acquire_impl(name, **params)

    def _runtime_options(self, params: dict[str, Any]) -> tuple[type[DevCloudRuntime], dict[str, Any]]:
        # Runtime selection affects execution/bridge behavior, not sandbox ownership or API routing.
        kind = params.get('runtime_kind') or 'devcloud'
        if kind == 'macosworld_devcloud':
            from nexus.runtime.providers.macosworld_devcloud.client import MacOSWorldDevCloudRuntime

            return MacOSWorldDevCloudRuntime, {'proxy_url': self.proxy_url, 'execute_mode': self.execute_mode}
        if kind not in {'devcloud', 'osworld'}:
            raise ValueError(f'Unsupported DevCloud runtime_kind: {kind}')
        return DevCloudRuntime, {'proxy_url': self.proxy_url, 'execute_mode': self.execute_mode}

    async def _call_api(self, base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f'{base_url.rstrip("/")}{path}'
        timeout = aiohttp.ClientTimeout(total=self.control_request_timeout)
        resp = await http_client.post(
            url, json=payload, headers=self._build_auth_headers(), timeout=timeout, proxy=self.proxy_url
        )
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f'DevCloud HTTP error {resp.status_code} for {path}: {resp.text[:512]}')
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f'DevCloud returned invalid JSON for {path}: {resp.text[:512]}') from exc
        if data.get('actionCode') != 200:
            raise RuntimeError(
                f'DevCloud API error for {path}: actionCode={data.get("actionCode")} '
                f'actionMessage={data.get("actionMessage", "")}'
            )
        return data

    async def _acquire_impl(self, name: str, **params: Any) -> DevCloudRuntime:
        if name in self._active_runtimes:
            return cast(DevCloudRuntime, self._active_runtimes[name])

        merged = self.get_params(name, **params)
        runtime_type, runtime_options = self._runtime_options(merged)
        runtime_credential = merged.get('runtime_credential')
        instance_ip = ''
        server_port = int(merged.get('server_port', self.server_port))
        if runtime_credential is not None:
            if not isinstance(runtime_credential, Mapping):
                raise InvalidPersistentRuntimeError('DevCloud runtime_credential must be a mapping')
            instance_ip = str(runtime_credential.get('instance_ip') or '').strip()
            if not instance_ip:
                raise InvalidPersistentRuntimeError('DevCloud runtime_credential requires instance_ip')
            server_port = int(runtime_credential.get('server_port', server_port))
        elif merged.get('instance_ip') is not None:
            raise ValueError('DevCloud instance_ip is not supported; use runtime_credential.instance_ip')
        instance_ready_timeout = float(merged.get('instance_ready_timeout', self.instance_ready_timeout))
        if instance_ip:
            base_url = f'http://{instance_ip}:{server_port}'
            await self._wait_for_osworld_ready(base_url, instance_ready_timeout)
            runtime = runtime_type(
                base_url=base_url,
                request_timeout=self.request_timeout,
                **runtime_options,
                metadata={'ip': instance_ip, 'reused': True},
            )
            self._active_runtimes[name] = runtime
            self._external_instance_names.add(name)
            logger.debug("Reused DevCloud runtime '%s' at %s", name, base_url)
            return runtime

        if not self.app_id:
            raise ValueError('app_id is required for DevCloudRuntimeProvider when creating an instance')
        if not self.app_token:
            raise ValueError('app_token is required for DevCloudRuntimeProvider when creating an instance')

        lease_time = int(merged.get('lease_time', self.lease_time))
        if 0 < lease_time < 300:
            raise ValueError('DevCloud lease_time must be at least 300 seconds, or <= 0 to disable the lease')

        create_payload = {'leaseTime': lease_time}
        data = await self._call_api(self.sandbox_api_base_url, '/mac/sandboxCreate', create_payload)
        task_id = str((data.get('data') or {}).get('taskId') or '')
        if not task_id:
            raise RuntimeError(f'DevCloud create response missing taskId: {data}')
        self._task_ids[name] = task_id
        logger.info("DevCloud '%s': creating sandbox (task %s)", name, task_id)

        try:
            result = await self._wait_for_task(task_id)
            vm_id = result.get('id')
            if vm_id is not None and str(vm_id) != '0':
                self._vm_ids[name] = str(vm_id)
            ip = str(result.get('ip') or '')
            if not ip:
                raise RuntimeError(f'DevCloud task {task_id} succeeded without ip')
            if name not in self._vm_ids:
                raise RuntimeError(f'DevCloud task {task_id} succeeded without a valid instance id')
            self._metadata[name] = result

            base_url = f'http://{ip}:{server_port}'
            await self._wait_for_osworld_ready(base_url, instance_ready_timeout)
            runtime_metadata = {**result, 'taskId': task_id}
            runtime = runtime_type(
                base_url=base_url,
                request_timeout=self.request_timeout,
                **runtime_options,
                metadata=runtime_metadata,
            )
            self._active_runtimes[name] = runtime
            logger.debug("DevCloud '%s': READY at %s metadata=%s", name, base_url, _log_metadata(runtime_metadata))
            return runtime
        except BaseException:
            try:
                await self._delete_instance(name)
            except Exception as exc:
                logger.warning("DevCloud '%s': cleanup failed: %s", name, exc)
            raise

    async def _wait_for_task(self, task_id: str, *, raise_on_failure: bool = True) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.request_timeout
        previous_status = None
        consecutive_query_errors = 0
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f'DevCloud task {task_id} did not finish within {self.request_timeout}s')
            try:
                # Bound the in-flight query too; checking the deadline only after it returns is insufficient.
                async with async_timeout(min(self.control_request_timeout, remaining)):
                    data = await self._call_api(
                        self.sandbox_api_base_url, '/mac/sandboxTaskResult', {'taskId': task_id}
                    )
            except aiohttp.ClientSSLError:
                # Certificate/TLS configuration failures are not transient connection failures.
                raise
            except (asyncio.TimeoutError, aiohttp.ClientConnectionError) as exc:
                # Only the read-only result query is safe to retry, not create/delete submissions.
                consecutive_query_errors += 1
                backoff_cap = self.TASK_POLL_INTERVAL * 2 ** min(consecutive_query_errors - 1, 2)
                # Equal jitter spreads concurrent clients without allowing immediate retries.
                delay = random.uniform(backoff_cap / 2, backoff_cap)
                logger.warning(
                    'DevCloud task %s: result query failed (%s); retrying after up to %.2fs (consecutive errors: %s)',
                    task_id,
                    type(exc).__name__,
                    delay,
                    consecutive_query_errors,
                )
            else:
                consecutive_query_errors = 0
                delay = self.TASK_POLL_INTERVAL
                result = {key[:1].lower() + key[1:]: value for key, value in (data.get('data') or {}).items()}
                status = str(result.get('status') or '').lower()
                if status != previous_status:
                    logger.info('DevCloud task %s: status=%s', task_id, status or '<missing>')
                    previous_status = status
                if status == 'succeeded':
                    return result
                if status in _FAILED_STATUSES:
                    if raise_on_failure:
                        raise RuntimeError(
                            f'DevCloud task {task_id} failed with status={status}: {_log_metadata(result)}'
                        )
                    return result
            await asyncio.sleep(max(0, min(delay, deadline - asyncio.get_running_loop().time())))

    async def _wait_for_osworld_ready(self, base_url: str, timeout_seconds: float) -> None:
        timeout = aiohttp.ClientTimeout(total=10)
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_error = ''
        while True:
            try:
                resp = await http_client.get(f'{base_url}/health', timeout=timeout, proxy=self.proxy_url)
                if resp.status_code == 200 and resp.content:
                    return
                last_error = f'HTTP {resp.status_code}: {resp.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f'DevCloud OSWorld runtime not ready at {base_url}/health: {last_error}')
            await asyncio.sleep(self.TASK_POLL_INTERVAL)

    async def _release_impl(self, name: str) -> None:
        if name in self._external_instance_names:
            self._active_runtimes.pop(name, None)
            self._external_instance_names.discard(name)
            logger.debug("Detached reused DevCloud runtime '%s' without deleting its instance", name)
            return
        await self._delete_instance(name)
        self._active_runtimes.pop(name, None)

    async def _delete_instance(self, name: str) -> None:
        try:
            async with async_timeout(self.release_timeout):
                await self._delete_instance_impl(name)
        except asyncio.TimeoutError as exc:
            vm_id = self._vm_ids.get(name)
            if vm_id is not None and (task_id := self._delete_task_ids.get(vm_id)):
                logger.warning(
                    "DevCloud '%s': deletion accepted but confirmation timed out "
                    '(instance %s, task %s, release budget %ss); returning without confirmed deletion',
                    name,
                    vm_id,
                    task_id,
                    self.release_timeout,
                )
                self._forget_instance(name)
                return
            if vm_id is not None:
                # With a known instance and no deletion task, the timeout occurred during submission.
                # The server may have accepted it; detach locally so later cleanup cannot resubmit.
                logger.warning(
                    "DevCloud '%s': deletion submission timed out "
                    '(instance %s, release budget %ss); release state unknown; '
                    'returning without confirmed deletion and without resubmitting',
                    name,
                    vm_id,
                    self.release_timeout,
                )
                self._forget_instance(name)
                return
            message = (
                f"DevCloud '{name}': release confirmation timed out "
                f'(instance {self._vm_ids.get(name)}, release budget {self.release_timeout}s); instance state retained'
            )
            logger.warning(message)
            raise TimeoutError(message) from exc

    async def _delete_instance_impl(self, name: str) -> None:
        vm_id = self._vm_ids.get(name)
        if vm_id is None and (create_task_id := self._task_ids.get(name)):
            # An interrupted create may still provision a VM. Resolve its exact id before cleanup.
            result = await self._wait_for_task(create_task_id, raise_on_failure=False)
            if result.get('id') is not None and str(result['id']) != '0':
                vm_id = self._vm_ids[name] = str(result['id'])

        if vm_id:
            task_id = self._delete_task_ids.get(vm_id)
            if task_id is None:
                data = await self._call_api(self.sandbox_api_base_url, '/mac/sandboxDelete', {'id': vm_id})
                task_id = str((data.get('data') or {}).get('taskId') or '')
                if not task_id:
                    raise RuntimeError(f'DevCloud delete response missing taskId for instance {vm_id}')
                self._delete_task_ids[vm_id] = task_id
            logger.info("DevCloud '%s': deleting instance %s (task %s)", name, vm_id, task_id)
            await self._wait_for_task(task_id)
            logger.info("DevCloud '%s': instance %s deletion completed", name, vm_id)
            self._delete_task_ids.pop(vm_id, None)

        self._forget_instance(name)

    def _forget_instance(self, name: str) -> None:
        self._vm_ids.pop(name, None)
        self._task_ids.pop(name, None)
        self._metadata.pop(name, None)


__all__ = ['DevCloudRuntimeProvider']
