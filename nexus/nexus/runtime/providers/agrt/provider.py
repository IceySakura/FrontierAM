"""AGRT RuntimeProvider for the AGRT GPU Windows and Linux sandbox fleet."""

from __future__ import annotations

import asyncio
import inspect
import logging

from collections.abc import Mapping
from typing import Any

import aiohttp

from nexus.runtime.client.base import Runtime
from nexus.runtime.client.remote import RemoteRuntime
from nexus.runtime.providers.ags.handle import AGSInstanceHandle
from nexus.runtime.providers.ags.osworld import AGSOSWorldRuntime, AGSRawOSWorldRuntime
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, require_raw_waa_runtime
from nexus.runtime.providers.base import InvalidPersistentRuntimeError, RuntimeProvider
from nexus.runtime.providers.registry import runtime_provider_registry
from nexus.utils.async_http import http_client


logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {'STOPPING', 'STOPPED', 'ERROR', 'FAILED', 'EXPIRED'}


class _AGRTAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _normalize_timeout(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f'{int(value)}s'
    return str(value)


@runtime_provider_registry.register('agrt')
class AGRTRuntimeProvider(RuntimeProvider):
    """RuntimeProvider backed by the AGRT platform (https://agrt.nano.woa.com).

    Manages the full instance lifecycle over the AGRT REST API (create, poll,
    token, renew, stop) and reaches in-instance WAA/OSWorld servers through the AGRT
    port ingress, whose ``https://{port}-{instance_id}.{domain}`` addressing and
    ``X-Access-Token`` auth match the AGS handle contract — so the AGS WAA/OSWorld
    runtimes are reused unchanged for the data plane.

    Provider config (``runtime_provider_config``):
        api_key: AGRT API key (required).
        domain: AGRT domain shared by control plane and ingress (default 'agrt.nano.woa.com').
        template: tool/template name, e.g. 'win-hunyuan-v1' (required for new instances).
        timeout: instance lifetime, '60s'~'168h' (default: backend default 12h).
        token_ttl_seconds: instance access-token TTL (default 3600).
        interface_mode: 'nexus' (default) or 'raw'.
        os: Guest operating system: 'windows' (default) or 'linux'.
        launch_mode: 'native' (WAA on Windows, OSWorld on Linux; default) or 'nexus_server' (an existing Nexus server).
        runtime_port: Nexus server port for launch_mode='nexus_server' (default 8000).
        environment: Environment variables injected by AGRT when creating an instance.
        request_timeout: data-plane request timeout in seconds (default 1800).
        instance_ready_timeout: max wait for PENDING -> RUNNING (default 600).
        service_ready_timeout: max wait for the selected service after RUNNING (default 120).
        poll_interval: polling interval in seconds (default 10).

    Runtime params can override api_key, template, timeout, token_ttl_seconds,
    interface_mode, os, launch_mode, runtime_port, environment,
    server_port, request_timeout and resource_spec. Domain and
    readiness polling settings apply to the whole provider. The native WAA/OSWorld
    server_port defaults to 5000 and can be set in runtime params.
    worker_venv only accepts a boolean.
    Pass runtime_credential={'instance_id': '...'} to reuse a running instance
    without a template. Reused instances are never stopped on release or failure.
    Instances expire at ``created_at + timeout``; call :meth:`renew` periodically
    for long-lived usage and :meth:`refresh_token` before the token TTL elapses.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        domain: str = 'agrt.nano.woa.com',
        template: str | None = None,
        timeout: str | int | None = None,
        token_ttl_seconds: int = 3600,
        interface_mode: str = 'nexus',
        os: str = 'windows',
        launch_mode: str = 'native',
        runtime_port: int = 8000,
        environment: dict[str, str] | None = None,
        request_timeout: float = 1800.0,
        instance_ready_timeout: float = 600.0,
        service_ready_timeout: float = 120.0,
        poll_interval: float = 10.0,
        resource_spec: dict[str, int] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(**config)
        self.api_key = api_key
        self.domain = domain
        self.template = template
        self.timeout = timeout
        self.token_ttl_seconds = token_ttl_seconds
        self.interface_mode = interface_mode
        self.os = os
        self.launch_mode = launch_mode
        self.runtime_port = runtime_port
        self.environment = dict(environment or {})
        self.request_timeout = request_timeout
        self.instance_ready_timeout = instance_ready_timeout
        self.service_ready_timeout = service_ready_timeout
        self.poll_interval = poll_interval
        self.resource_spec = resource_spec or {}
        self._instance_ids: dict[str, str] = {}
        self._api_keys: dict[str, str] = {}
        self._external_instance_names: set[str] = set()
        self._acquire_locks: dict[str, asyncio.Lock] = {}

    def get_runtime_id(self, name: str) -> str | None:
        return self._instance_ids.get(name)

    async def _api(self, method: str, path: str, *, api_key: str, payload: dict[str, Any] | None = None) -> Any:
        response = await http_client.request(
            method,
            f'https://{self.domain}{path}',
            headers={'Authorization': f'Bearer {api_key}'},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30.0),
        )
        if response.status_code >= 400:
            raise _AGRTAPIError(
                f'AGRT {method} {path} failed: HTTP {response.status_code} {response.text[:500]}',
                response.status_code,
            )
        return response.json()

    async def _describe_running_instance(self, instance_id: str, api_key: str) -> dict[str, Any]:
        try:
            instance = await self._api('GET', f'/v1/instances/{instance_id}', api_key=api_key)
        except _AGRTAPIError as exc:
            if exc.status_code == 404:
                raise InvalidPersistentRuntimeError(f'AGRT instance {instance_id} was not found') from exc
            raise
        status = str(instance.get('status') or '').upper()
        if status != 'RUNNING':
            raise InvalidPersistentRuntimeError(
                f'AGRT instance {instance_id} is not running: status={status or "UNKNOWN"}'
            )
        return instance

    async def _wait_for_running(self, instance_id: str, api_key: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.instance_ready_timeout
        while True:
            instance = await self._api('GET', f'/v1/instances/{instance_id}', api_key=api_key)
            status = str(instance.get('status') or '')
            if status == 'RUNNING':
                return instance
            if status in _TERMINAL_STATUSES:
                raise RuntimeError(f'AGRT instance {instance_id} entered status {status} before becoming RUNNING')
            if loop.time() >= deadline:
                raise TimeoutError(
                    f'AGRT instance {instance_id} not RUNNING after {self.instance_ready_timeout}s (status={status})'
                )
            await asyncio.sleep(self.poll_interval)

    async def _acquire_token(self, instance_id: str, api_key: str, ttl_seconds: int) -> str:
        body = await self._api(
            'POST',
            f'/v1/instances/{instance_id}/token',
            api_key=api_key,
            payload={'ttl_seconds': int(ttl_seconds)},
        )
        token = body.get('token') if isinstance(body, dict) else None
        if not token:
            raise RuntimeError(f'AGRT token request for {instance_id} did not return a token')
        return str(token)

    async def _wait_for_service_ready(
        self, handle: AGSInstanceHandle, server_port: int, *, probe_path: str = '/probe'
    ) -> None:
        probe_url = handle.url(server_port, probe_path)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.service_ready_timeout
        last_error = ''
        while True:
            try:
                response = await http_client.get(
                    probe_url, timeout=aiohttp.ClientTimeout(total=10.0), headers=handle.headers
                )
                if response.status_code == 200 and response.content:
                    return
                last_error = f'HTTP {response.status_code}: {response.text[:200]}'
            except Exception as exc:
                last_error = f'{type(exc).__name__}: {exc}'
            if loop.time() >= deadline:
                raise TimeoutError(f'AGRT service not ready at {probe_url}: {last_error}')
            await asyncio.sleep(self.poll_interval)

    async def renew(self, name: str, timeout: str | int = '12h') -> dict[str, Any]:
        """Reset the instance expiry to now + timeout via PATCH /v1/instances/{id}."""
        instance_id = self._instance_ids.get(name)
        api_key = self._api_keys.get(name)
        if not instance_id or not api_key:
            raise KeyError(f'No active agrt runtime {name!r}')
        return await self._api(
            'PATCH',
            f'/v1/instances/{instance_id}',
            api_key=api_key,
            payload={'timeout': _normalize_timeout(timeout)},
        )

    async def refresh_token(self, name: str, ttl_seconds: int | None = None) -> str:
        """Issue a fresh instance token and swap it into the active runtime's handle."""
        runtime = self._active_runtimes.get(name)
        api_key = self._api_keys.get(name)
        if runtime is None or not api_key:
            raise KeyError(f'No active agrt runtime {name!r}')
        instance_id = self._instance_ids[name]
        token = await self._acquire_token(instance_id, api_key, int(ttl_seconds or self.token_ttl_seconds))
        if isinstance(runtime, RemoteRuntime):
            runtime.config.default_headers['X-Access-Token'] = token
        elif isinstance(runtime, (AGSOSWorldRuntime, AGSRawOSWorldRuntime)):
            runtime.handle.access_token = token
        else:
            require_raw_waa_runtime(runtime).handle.access_token = token
        return token

    async def acquire(self, name: str, **params: Any) -> Runtime:
        if name not in self._acquire_locks:
            self._acquire_locks[name] = asyncio.Lock()

        async with self._acquire_locks[name]:
            if name in self._active_runtimes:
                return self._active_runtimes[name]
            if name in self._instance_ids:
                raise RuntimeError(f'AGRT runtime {name!r} has pending cleanup; release it before acquiring again')

            merged = self.get_params(name, **params)
            runtime_credential = merged.get('runtime_credential')
            instance_id = ''
            if runtime_credential is not None:
                if not isinstance(runtime_credential, Mapping):
                    raise InvalidPersistentRuntimeError('AGRT runtime_credential must be a mapping')
                instance_id = str(runtime_credential.get('instance_id') or '').strip()
                if not instance_id:
                    raise InvalidPersistentRuntimeError('AGRT runtime_credential requires instance_id')
            elif merged.get('instance_id') is not None:
                raise ValueError('AGRT instance_id is not supported; use runtime_credential.instance_id')
            api_key = str(merged.get('api_key') or self.api_key or '').strip()
            if not api_key:
                raise ValueError(
                    f'agrt runtime {name!r} requires an api_key (provider config or runtime_configs.{name})'
                )
            template = merged.get('template') or self.template
            if not template and not instance_id:
                raise ValueError(
                    f'agrt runtime {name!r} requires a template (provider config or runtime_configs.{name})'
                )
            interface_mode = str(merged.get('interface_mode') or self.interface_mode)
            if interface_mode not in {'nexus', 'raw'}:
                raise ValueError(f'Unsupported agrt interface_mode: {interface_mode!r}')
            launch_mode = str(merged.get('launch_mode') or self.launch_mode)
            if launch_mode not in {'native', 'nexus_server'}:
                raise ValueError(f'Unsupported agrt launch_mode: {launch_mode!r}')
            if launch_mode == 'nexus_server' and interface_mode == 'raw':
                raise ValueError('AGRT interface_mode="raw" requires launch_mode="native"')
            worker_venv = merged.get('worker_venv')
            if 'worker_venv' in merged and not isinstance(worker_venv, bool):
                raise ValueError('AGRT worker_venv only accepts true or false')
            if worker_venv is not None and worker_venv is not False and launch_mode != 'nexus_server':
                raise ValueError('AGRT worker_venv requires launch_mode="nexus_server"')
            os = str(merged.get('os', self.os))
            platform = f'{os}-x86_64'
            if os not in {'windows', 'linux'}:
                raise ValueError(f'Unsupported AGRT platform: {platform!r}')
            environment = {**self.environment, **(merged.get('environment') or {})}
            if instance_id and environment:
                raise ValueError('AGRT environment only applies when creating a new instance')
            token_ttl_seconds = int(merged.get('token_ttl_seconds') or self.token_ttl_seconds)
            timeout = _normalize_timeout(merged.get('timeout', self.timeout))
            resource_spec = merged.get('resource_spec') or self.resource_spec
            request_timeout = float(merged.get('request_timeout', self.request_timeout))

            if instance_id:
                instance = await self._describe_running_instance(instance_id, api_key)
                self._external_instance_names.add(name)
            else:
                payload: dict[str, Any] = {'template': template}
                if timeout is not None:
                    payload['timeout'] = timeout
                if environment:
                    payload['env'] = environment
                instance = await self._api('POST', '/v1/instances', api_key=api_key, payload=payload)
                instance_id = str(instance.get('instance_id') or '')
                if not instance_id:
                    raise RuntimeError(f'AGRT instance creation did not return instance_id: {instance!r}')
            self._instance_ids[name] = instance_id
            self._api_keys[name] = api_key

            try:
                if name not in self._external_instance_names:
                    instance = await self._wait_for_running(instance_id, api_key)
                token = await self._acquire_token(instance_id, api_key, token_ttl_seconds)
                handle = AGSInstanceHandle(
                    instance_id=instance_id,
                    access_token=token,
                    domain=self.domain,
                    tool_id=instance.get('tool_id'),
                    tool_name=instance.get('tool_name') or template,
                    tool_type='osworld' if os == 'linux' else 'waa',
                    metadata={'instance': instance},
                )
                runtime: Runtime
                if launch_mode == 'nexus_server':
                    runtime_port = int(merged.get('runtime_port') or self.runtime_port)
                    await self._wait_for_service_ready(handle, runtime_port, probe_path='/health')
                    runtime = RemoteRuntime(
                        base_url=handle.url(runtime_port),
                        default_headers=handle.headers,
                        request_timeout=request_timeout,
                        resource_spec=resource_spec,
                    )
                else:
                    server_port = int(merged.get('server_port') or 5000)
                    await self._wait_for_service_ready(
                        handle, server_port, probe_path='/screenshot' if os == 'linux' else '/probe'
                    )
                    runtime_class = AGSRawOSWorldRuntime if os == 'linux' else AGSRawWindowsAgentArenaRuntime
                    runtime = runtime_class(
                        handle,
                        server_port=server_port,
                        request_timeout=request_timeout,
                        resource_spec=resource_spec,
                    )
                    if interface_mode == 'nexus':
                        runtime = runtime.to_nexus()
            except BaseException:
                try:
                    await self._stop_instance(name)
                except Exception:
                    logger.exception(
                        'Failed to stop AGRT instance %s; release is required to retry cleanup', instance_id
                    )
                raise

            self._active_runtimes[name] = runtime
            logger.debug('agrt runtime %s ready: instance_id=%s template=%s', name, instance_id, template)
            return runtime

    async def _release_impl(self, name: str) -> None:
        async with self._acquire_locks.setdefault(name, asyncio.Lock()):
            runtime = self._active_runtimes.pop(name, None)
            try:
                if runtime is not None:
                    if isinstance(runtime, RemoteRuntime):
                        await runtime.ws_disconnect()
                    close = getattr(runtime, 'close', None)
                    if callable(close):
                        result = close()
                        if inspect.isawaitable(result):
                            await result
            finally:
                await self._stop_instance(name)
            logger.debug('Released agrt runtime %s', name)

    async def _stop_instance(self, name: str) -> None:
        if name in self._external_instance_names:
            self._external_instance_names.discard(name)
            self._instance_ids.pop(name, None)
            self._api_keys.pop(name, None)
            logger.debug('Detached reused AGRT runtime %s without stopping its instance', name)
            return
        instance_id = self._instance_ids.get(name)
        api_key = self._api_keys.get(name)
        if instance_id and api_key:
            # Keep the ID and credentials until stop succeeds, so callers can retry.
            await self._api('POST', f'/v1/instances/{instance_id}/stop', api_key=api_key)
            self._instance_ids.pop(name, None)
            self._api_keys.pop(name, None)

    async def _release_all_impl(self) -> None:
        # Include instances whose acquire failed before a runtime was registered.
        first_error: Exception | None = None
        for name in list(self._instance_ids):
            try:
                await self._release_impl(name)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error


__all__ = ['AGRTRuntimeProvider']
