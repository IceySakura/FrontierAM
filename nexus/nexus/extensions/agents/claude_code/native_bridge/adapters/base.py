"""Claude Code native bridge adapter protocol and factory."""

from __future__ import annotations

from typing import Any, Protocol, cast

from nexus.core.registry import EntryNotFoundError, Registry
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind, classify_native_bridge_runtime
from nexus.runtime.providers.ags.osworld import AGSOSWorldExecutorRuntime
from nexus.runtime.providers.ags.waa import WAAExecutorRuntime
from nexus.runtime.providers.devcloud.client import DevCloudRuntime
from nexus.runtime.providers.smartrun.client import SmartRunRuntime

from .types import ClaudeInstallSpec, ClaudeRunRequest, ClaudeRunResult


ClaudeNativeBridgeRuntime = AGSOSWorldExecutorRuntime | WAAExecutorRuntime | DevCloudRuntime | SmartRunRuntime


class ClaudeNativeBridgeAdapter(Protocol):
    """Target-specific Claude Code install/run operations."""

    async def ensure_claude(self, spec: ClaudeInstallSpec) -> str:
        """Return a usable Claude Code executable path in the sandbox."""

    async def run_pre_start_scripts(
        self,
        scripts: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Run pre-start scripts inside the sandbox before Claude Code starts."""

    async def register_mcp_servers(
        self,
        claude_executable: str,
        mcp_servers: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Register stdio MCP servers in the sandbox Claude Code config."""

    async def run_claude(self, request: ClaudeRunRequest) -> ClaudeRunResult:
        """Run one Claude Code prompt and return process output."""


def build_claude_native_bridge_adapter(runtime: ClaudeNativeBridgeRuntime) -> ClaudeNativeBridgeAdapter:
    """Build the Claude Code adapter from the native runtime type."""
    kind = classify_native_bridge_runtime(runtime)
    if kind is NativeBridgeRuntimeKind.AGS_OSWORLD:
        from .osworld import OSWorldClaudeNativeBridgeAdapter

        return OSWorldClaudeNativeBridgeAdapter(cast(AGSOSWorldExecutorRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.AGS_WAA:
        from .waa import WAAClaudeNativeBridgeAdapter

        return WAAClaudeNativeBridgeAdapter(cast(WAAExecutorRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.DEVCLOUD_MACOS:
        from .devcloud_macos import DevCloudMacOSClaudeNativeBridgeAdapter

        return DevCloudMacOSClaudeNativeBridgeAdapter(cast(DevCloudRuntime, runtime))
    if kind is NativeBridgeRuntimeKind.SMARTRUN_MACOS:
        from .smartrun_macos import SmartRunMacOSClaudeNativeBridgeAdapter

        return SmartRunMacOSClaudeNativeBridgeAdapter(cast(SmartRunRuntime, runtime))
    raise TypeError(f'Unsupported Claude Code native bridge runtime kind: {kind.value}')


execution_profile_registry = Registry[Any]('claude_code_execution_profile')
multiturn_execution_profile_registry = Registry[Any]('claude_code_multiturn_execution_profile')


def build_execution_profile(
    name: str,
    options: dict[str, Any],
    task_context: dict[str, Any],
    user_prompt: Any,
    *,
    input_overrides: dict[str, Any] | None = None,
) -> Any:
    """Select explicit execution policy, never infer benchmark policy from a runtime."""
    try:
        factory = execution_profile_registry.get(name)
    except EntryNotFoundError as exc:
        raise ValueError(f'Unknown Claude Code execution profile: {name}') from exc
    if input_overrides:
        accepted = getattr(factory, 'input_overrides', ())
        options = dict(options)
        for key, value in input_overrides.items():
            if key not in accepted:
                raise ValueError(f'{key} must be configured through execution_options for execution profile {name!r}')
            options[key] = value
    return factory(options, task_context, user_prompt)


__all__ = ['ClaudeNativeBridgeAdapter', 'ClaudeNativeBridgeRuntime', 'build_claude_native_bridge_adapter']
