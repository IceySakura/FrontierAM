"""Claude Code execution through a sandbox-local native bridge."""

from __future__ import annotations

import time

from collections.abc import Callable
from typing import Any

from nexus.extensions.agents.claude_code.native_bridge.adapters import build_claude_native_bridge_adapter
from nexus.extensions.agents.claude_code.native_bridge.adapters.types import ClaudeInstallSpec, ClaudeRunRequest
from nexus.extensions.agents.installed_common import DEFAULT_NPM_REGISTRY
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult


def _result_dict(*, success: bool, output: str, stderr: str, exit_code: int, start_time: float) -> dict[str, Any]:
    return {
        'success': success,
        'output': output,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': time.time() - start_time,
    }


def _exception_detail(exc: Exception) -> str:
    return str(exc) or type(exc).__name__


async def setup_and_run_native_bridge_claude_code(
    *,
    runtime: Any,
    launch: BridgeLaunchResult,
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600.0,
    version: str = 'latest',
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    ensure_claude: bool = True,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    extra_env: dict[str, str] | None = None,
    extra_cli_args: list[str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    adapter_factory: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Set up Claude Code and run one non-interactive prompt through the native bridge."""
    start_time = time.time()

    claude_adapter = (adapter_factory or build_claude_native_bridge_adapter)(runtime)
    try:
        resolved_claude_executable = await claude_adapter.ensure_claude(
            ClaudeInstallSpec(
                ensure_claude=ensure_claude,
                npm_registry=npm_registry,
                version=version,
                binary_url=binary_url,
                binary_checksum=binary_checksum,
            )
        )
    except Exception as exc:
        return _result_dict(
            success=False,
            output='',
            stderr=f'Failed to install Claude Code: {_exception_detail(exc)}',
            exit_code=-1,
            start_time=start_time,
        )

    if pre_start_scripts:
        try:
            await claude_adapter.run_pre_start_scripts(
                pre_start_scripts,
                working_dir=working_dir,
                launch=launch,
                extra_env=extra_env,
            )
        except Exception as exc:
            return _result_dict(
                success=False,
                output='',
                stderr=f'Pre-start script failed: {_exception_detail(exc)}',
                exit_code=-1,
                start_time=start_time,
            )

    if mcp_servers:
        try:
            await claude_adapter.register_mcp_servers(
                resolved_claude_executable,
                mcp_servers,
                working_dir=working_dir,
                launch=launch,
                extra_env=extra_env,
            )
        except Exception as exc:
            return _result_dict(
                success=False,
                output='',
                stderr=f'Failed to register MCP servers: {_exception_detail(exc)}',
                exit_code=-1,
                start_time=start_time,
            )

    try:
        run = await claude_adapter.run_claude(
            ClaudeRunRequest(
                claude_executable=resolved_claude_executable,
                launch=launch,
                task=task,
                working_dir=working_dir,
                timeout=timeout,
                extra_env=extra_env,
                extra_cli_args=extra_cli_args,
            )
        )
    except Exception as exc:
        return _result_dict(
            success=False,
            output='',
            stderr=f'Failed to run Claude Code: {_exception_detail(exc)}',
            exit_code=-1,
            start_time=start_time,
        )
    return _result_dict(
        success=run.exit_code == 0,
        output=run.stdout,
        stderr=run.stderr,
        exit_code=run.exit_code,
        start_time=start_time,
    )


__all__ = ['setup_and_run_native_bridge_claude_code']
