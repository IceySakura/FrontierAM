"""Codex native bridge adapter protocol and factory."""

from __future__ import annotations

from typing import Any, Protocol, cast

from nexus.core.registry import EntryNotFoundError, Registry
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind, classify_native_bridge_runtime
from nexus.runtime.providers.ags.osworld import AGSOSWorldExecutorRuntime
from nexus.runtime.providers.ags.waa import WAAExecutorRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.smartrun.client import SmartRunRuntime

from .types import CodexInstallSpec, CodexRunRequest, CodexRunResult


CodexNativeBridgeRuntime = AGSOSWorldExecutorRuntime | WAAExecutorRuntime | DevCloudRuntime | SmartRunRuntime


class CodexNativeBridgeAdapter(Protocol):
    async def ensure_codex(self, spec: CodexInstallSpec) -> str: ...
    async def run_pre_start_scripts(
        self,
        scripts: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None: ...
    async def run_codex(self, request: CodexRunRequest) -> CodexRunResult: ...


def build_codex_native_bridge_adapter(runtime: CodexNativeBridgeRuntime) -> CodexNativeBridgeAdapter:
    kind = classify_native_bridge_runtime(runtime)
    if kind is NativeBridgeRuntimeKind.AGS_OSWORLD:
        from .osworld import OSWorldCodexNativeBridgeAdapter

        return OSWorldCodexNativeBridgeAdapter(cast(AGSOSWorldExecutorRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.AGS_WAA:
        from .waa import WAACodexNativeBridgeAdapter

        return WAACodexNativeBridgeAdapter(cast(WAAExecutorRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.DEVCLOUD_MACOS:
        from .devcloud_macos import DevCloudMacOSCodexNativeBridgeAdapter

        return DevCloudMacOSCodexNativeBridgeAdapter(cast(DevCloudRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.SMARTRUN_MACOS:
        from .smartrun_macos import SmartRunMacOSCodexNativeBridgeAdapter

        return SmartRunMacOSCodexNativeBridgeAdapter(cast(SmartRunRuntime, runtime))
    raise TypeError(f'Unsupported Codex native bridge runtime kind: {kind.value}')


execution_profile_registry = Registry[Any]('codex_execution_profile')


def build_execution_profile(name: str, options: dict, task_context: dict, user_prompt: Any):
    """Resolve explicitly registered policies without importing benchmark tasks."""
    try:
        factory = execution_profile_registry.get(name)
    except EntryNotFoundError as exc:
        raise ValueError(f'Unknown Codex execution profile: {name}') from exc
    return factory(options, task_context, user_prompt)


__all__ = ['CodexNativeBridgeAdapter', 'CodexNativeBridgeRuntime', 'build_codex_native_bridge_adapter']
