"""Host-side helpers for installed agents that run through a native bridge."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import Awaitable, Callable
from typing import Any, cast

from nexus.core.context import require_context
from nexus.runtime.bridge.adapters.ags_osworld import AGSOSWorldBridgeAdapter
from nexus.runtime.bridge.adapters.ags_waa import AGSWAABridgeAdapter
from nexus.runtime.bridge.adapters.base import BridgeLaunchOptions, InstalledAgentBridgeAdapter
from nexus.runtime.bridge.adapters.devcloud_macos import DevCloudMacOSBridgeAdapter
from nexus.runtime.bridge.adapters.smartrun_macos import SmartRunMacOSBridgeAdapter
from nexus.runtime.bridge.artifacts import build_bridge_artifact_cached
from nexus.runtime.bridge.host_client import BridgeHostClient, connect_bridge_llm_handler
from nexus.runtime.bridge.runtime_kind import NativeBridgeRuntimeKind, classify_native_bridge_runtime
from nexus.runtime.server.services.llm_handler import LLMRequestHandler


logger = logging.getLogger(__name__)

# Seconds to wait for the sandbox-local bridge to serve /health after launch. Only covers
# the post-launch HTTP readiness poll (the artifact build/upload happens before), so a fixed
# value is fine for every target — no need to expose it as configuration.
BRIDGE_READINESS_TIMEOUT = 300.0
BRIDGE_CLEANUP_TIMEOUT = 30.0


_BRIDGE_ADAPTERS: dict[NativeBridgeRuntimeKind, type[InstalledAgentBridgeAdapter]] = {
    NativeBridgeRuntimeKind.AGS_OSWORLD: AGSOSWorldBridgeAdapter,
    NativeBridgeRuntimeKind.AGS_WAA: AGSWAABridgeAdapter,
    NativeBridgeRuntimeKind.DEVCLOUD_MACOS: DevCloudMacOSBridgeAdapter,
    NativeBridgeRuntimeKind.SMARTRUN_MACOS: SmartRunMacOSBridgeAdapter,
}


def _build_adapter(runtime: Any) -> InstalledAgentBridgeAdapter:
    factory = getattr(runtime, 'create_native_bridge_adapter', None)
    if callable(factory):
        return factory(runtime)
    kind = classify_native_bridge_runtime(runtime)
    adapter_cls = _BRIDGE_ADAPTERS.get(kind)
    if adapter_cls is None:
        raise TypeError(f'Unsupported native bridge runtime kind: {kind.value}')
    return adapter_cls(runtime)


async def _await_cleanup(awaitable: Awaitable[Any], *, label: str) -> None:
    cleanup = asyncio.create_task(asyncio.wait_for(awaitable, timeout=BRIDGE_CLEANUP_TIMEOUT))
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise
    except Exception:
        logger.warning('Native bridge %s failed during cleanup', label, exc_info=True)


async def call_native_bridge_runtime_agent(
    runtime_agent: Callable[..., Awaitable[dict[str, Any]]],
    /,
    *,
    timeout: float = 3600.0,
    no_thinking_request_args: dict[str, Any] | None = None,
    haiku_model: str | None = None,
    preserve_thinking: bool = True,
    conversion_log_level: str = 'INFO',
    thinking_config: str = 'hunyuan',
    api_protocol: str = 'anthropic',
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a host-orchestrated installed agent through a sandbox-local bridge.

    This mirrors :func:`call_runtime_agent` at the installed-agent API boundary,
    but it intentionally does not call ``runtime.invoke``.  The supplied
    ``runtime_agent`` callable runs in the host process and drives the native sandbox
    through the provider runtime plus bridge adapter.
    """
    ctx = require_context()
    runtime = await ctx.get_runtime()

    adapter = _build_adapter(runtime)
    artifact_path = await build_bridge_artifact_cached(runtime)
    launch = None
    bridge_may_be_running = False
    bridge_client: BridgeHostClient | None = None
    handler: LLMRequestHandler | None = None
    try:
        await adapter.upload_artifact(artifact_path)
        bridge_may_be_running = True
        launch = await adapter.launch_bridge(
            BridgeLaunchOptions(
                conversion_log_level=conversion_log_level,
                thinking_config=thinking_config,
                request_timeout=timeout,
                api_protocol=api_protocol,
            )
        )
        await adapter.wait_ready(launch, timeout=BRIDGE_READINESS_TIMEOUT)

        client_factory = getattr(adapter, 'create_host_client', None)
        bridge_client = (
            client_factory(launch)
            if callable(client_factory)
            else BridgeHostClient(base_url=launch.base_url, headers=launch.headers)
        )
        handler = await connect_bridge_llm_handler(
            cast(Any, bridge_client),
            ctx.get_openai_client(),
            no_thinking_request_args=no_thinking_request_args,
            haiku_model=haiku_model,
            preserve_thinking=preserve_thinking,
        )
        result = await runtime_agent(
            runtime=runtime,
            launch=launch,
            timeout=timeout,
            **kwargs,
        )
        if isinstance(result, dict):
            result['llm_request_outcomes'] = handler.request_outcomes
            result['last_llm_error_code'] = handler.last_request_error_code
            result['last_llm_finish_reason'] = handler.last_request_finish_reason
            result['last_llm_unparsed_tool_call'] = handler.last_request_left_tool_call_unparsed
        return result
    finally:
        if handler is not None:
            handler.unregister()
        cancelled: asyncio.CancelledError | None = None
        if bridge_client is not None:
            try:
                await _await_cleanup(bridge_client.ws_disconnect(), label='Socket.IO disconnect')
            except asyncio.CancelledError as exc:
                cancelled = exc
        if bridge_may_be_running:
            try:
                await _await_cleanup(adapter.stop_bridge(), label='stop')
            except asyncio.CancelledError as exc:
                cancelled = cancelled or exc
        if cancelled is not None:
            raise cancelled


__all__ = ['BRIDGE_READINESS_TIMEOUT', 'call_native_bridge_runtime_agent']
