from __future__ import annotations

import logging
import uuid

from typing import Any

from label.config import LabelConfig, LabelConfigHolder
from label.schemas.sandbox import (
    Provider,
    SandboxAcquireRequest,
    SandboxCallExecutorRequest,
    SandboxData,
    SandboxReleaseRequest,
)
from label.services.errors import SandboxErrorCode


logger = logging.getLogger(__name__)


class SandboxProviderError(Exception):
    def __init__(self, code: SandboxErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SandboxService:
    """沙盒申请/释放服务。

    配置通过 `ConfigHolder` 惰性读取而非持有快照，以支持七彩石热更新。
    provider 实例按配置缓存（构造时凭据已拷入实例），因此在构造时自行注册
    重载 hook：配置变更后清空缓存，下次请求按新凭据重建，调用方无需关心。
    """

    def __init__(self, config_holder: LabelConfigHolder) -> None:
        self._holder = config_holder
        self._providers: dict[Provider, Any] = {}
        config_holder.add_reload_hook(self.invalidate)

    @property
    def _config(self) -> LabelConfig:
        return self._holder.current

    def invalidate(self) -> None:
        """丢弃已构造的 provider 实例，使其按最新配置重建。"""
        if self._providers:
            logger.info('sandbox provider cache invalidated, will rebuild with latest config')
        self._providers.clear()

    async def acquire(self, request: SandboxAcquireRequest) -> SandboxData:
        provider = self._get_provider(request.provider)
        runtime_name = f'label-{request.provider.value}-{uuid.uuid4().hex}'
        try:
            runtime = await provider.acquire(runtime_name, **self._acquire_params(request.provider))
        except Exception as exc:
            logger.exception('failed to acquire %s sandbox', request.provider.value)
            raise SandboxProviderError(
                SandboxErrorCode.ACQUIRE_FAILED, f'failed to acquire {request.provider.value} sandbox'
            ) from exc

        try:
            return self._sandbox_data(request.provider, provider, runtime_name, runtime)
        except Exception as exc:
            logger.exception('failed to read %s sandbox identity, releasing', request.provider.value)
            await self._safe_release(provider, runtime_name)
            raise SandboxProviderError(
                SandboxErrorCode.ACQUIRE_FAILED, f'failed to acquire {request.provider.value} sandbox'
            ) from exc

    async def release(self, request: SandboxReleaseRequest) -> None:
        provider = self._get_provider(request.provider)
        try:
            await self._release_by_identifiers(request.provider, provider, request)
        except Exception as exc:
            logger.exception('failed to release %s sandbox', request.provider.value)
            raise SandboxProviderError(
                SandboxErrorCode.RELEASE_FAILED, f'failed to release {request.provider.value} sandbox'
            ) from exc

    async def call_executor(self, request: SandboxCallExecutorRequest) -> Any:
        provider = self._get_provider(Provider.AGS)
        try:
            runtime = self._create_ags_waa_runtime(provider, request.instance_id)
            return await runtime.call_executor(request.executor_name, request.executor_args)
        except Exception as exc:
            logger.exception(
                'failed to call AGS WAA executor %s on instance %s', request.executor_name, request.instance_id
            )
            raise SandboxProviderError(
                SandboxErrorCode.CALL_EXECUTOR_FAILED, f'failed to call executor {request.executor_name}'
            ) from exc

    @staticmethod
    def _create_ags_waa_runtime(provider: Any, instance_id: str) -> Any:
        from nexus.runtime.providers.ags.handle import AGSInstanceHandle
        from nexus.runtime.providers.ags.waa import AGSWindowsAgentArenaRuntime

        handle = AGSInstanceHandle(instance_id=instance_id, access_token='', domain=provider.domain, tool_type='waa')
        return AGSWindowsAgentArenaRuntime(handle, request_timeout=provider.request_timeout)

    def _get_provider(self, provider: Provider) -> Any:
        if provider in self._providers:
            return self._providers[provider]
        try:
            instance = self._create_provider(provider)
        except (ImportError, ValueError) as exc:
            logger.warning('sandbox provider %s is not configured: %s', provider.value, exc)
            raise SandboxProviderError(
                SandboxErrorCode.PROVIDER_NOT_CONFIGURED, f'provider {provider.value} is not configured'
            ) from exc
        self._providers[provider] = instance
        return instance

    def _create_provider(self, provider: Provider) -> Any:
        if provider is Provider.AGS:
            from nexus.runtime.providers.ags import AGSRuntimeProvider

            config = self._config.provider.ags
            return AGSRuntimeProvider(
                secret_id=config.secret_id,
                secret_key=config.secret_key,
                region=config.region,
                domain=config.domain,
                interface_mode='raw',
                auth_mode='NONE',
            )
        if provider is Provider.DEVCLOUD:
            from nexus.runtime.providers.devcloud import DevCloudRuntimeProvider

            config = self._config.provider.devcloud
            return DevCloudRuntimeProvider(
                app_id=config.app_id,
                app_token=config.app_token,
                sandbox_api_base_url=config.sandbox_api_base_url,
                control_api_base_url=config.control_api_base_url,
                project=config.project,
                pipeline_id=config.pipeline_id,
                build_id=config.build_id,
                vm_seq_id=config.vm_seq_id,
            )
        from nexus.runtime.providers.smartrun import SmartRunRuntimeProvider

        config = self._config.provider.smartrun
        return SmartRunRuntimeProvider(
            token=config.token,
            api_base_url=config.api_base_url,
            platform='android',
            duration=config.duration,
        )

    def _acquire_params(self, provider: Provider) -> dict[str, object]:
        if provider is Provider.AGS:
            return {
                'runtime_kind': 'waa',
                'tool_type': 'waa',
                'tool_id': self._config.provider.ags.tool_id,
                'vnc_port': 8006,
            }
        if provider is Provider.SMARTRUN:
            return {'platform': 'android', 'purpose': 'GUI sandbox labeling'}
        # DevCloud takes no acquire-time params; provisioning is driven by provider config.
        return {}

    async def _release_by_identifiers(self, provider: Provider, instance: Any, request: SandboxReleaseRequest) -> None:
        """Release directly by the identifiers supplied in the request.

        Stateless: no in-memory lease is kept between acquire and release, so the
        caller must pass back the identifiers returned by acquire. Each provider is
        released through its native destroy API using only those identifiers.
        """
        if provider is Provider.AGS:
            await instance._call_api('StopSandboxInstance', {'InstanceId': request.instance_id})
            return
        if provider is Provider.DEVCLOUD:
            await instance._call_api(
                instance.control_api_base_url, '/mac/vm/delete', {'id': request.vm_id, 'force': True}
            )
            return
        await instance._call_mcp('release_device', {'device_id': request.device_id, 'json_output': True})

    @staticmethod
    async def _safe_release(instance: Any, runtime_name: str) -> None:
        try:
            await instance.release(runtime_name)
        except Exception:
            logger.exception('failed to release runtime %s during acquire rollback', runtime_name)

    @staticmethod
    def _sandbox_data(provider: Provider, instance: Any, runtime_name: str, runtime: Any) -> SandboxData:
        if provider is Provider.AGS:
            instance_id = getattr(runtime.handle, 'instance_id', None) or instance.get_runtime_id(runtime_name)
            if not instance_id:
                raise RuntimeError('AGS provider returned no instance_id')
            remote_url = str(runtime.vnc_url).rstrip('/') + '/vnc.html'
            return SandboxData(instance_id=instance_id, remote_control_url=remote_url)
        if provider is Provider.DEVCLOUD:
            metadata = runtime.metadata
            instance_ip = str(metadata.get('ip') or '')
            vm_id = instance.get_runtime_id(runtime_name)
            if not instance_ip or not vm_id:
                raise RuntimeError('DevCloud provider returned incomplete instance identity')
            return SandboxData(instance_ip=instance_ip, vm_id=vm_id, remote_control_url=f'vnc://{instance_ip}')
        device_id = instance.get_device_id(runtime_name)
        session_id = instance.get_runtime_id(runtime_name)
        if not device_id or not session_id:
            raise RuntimeError('SmartRun provider returned incomplete device identity')
        remote_url = f'https://smartrun.woa.com/live-stream?deviceId={device_id}&sessionId={session_id}'
        return SandboxData(device_id=device_id, session_id=session_id, remote_control_url=remote_url)
