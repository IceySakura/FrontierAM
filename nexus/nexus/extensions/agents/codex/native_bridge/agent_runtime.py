"""Runtime-side setup for Codex using a sandbox-local native bridge."""

from __future__ import annotations

import time

from typing import Any

from nexus.extensions.agents.codex.installer import DEFAULT_CODEX_VERSION
from nexus.extensions.agents.codex.native_bridge.adapters import build_codex_native_bridge_adapter
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexInstallSpec, CodexRunRequest
from nexus.extensions.agents.installed_common import DEFAULT_NPM_REGISTRY
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult


def _result_dict(*, success: bool, output: str, stderr: str, exit_code: int, start_time: float, **extra: Any):
    return {
        'success': success,
        'output': output,
        'stderr': stderr,
        'exit_code': exit_code,
        'execution_time': time.time() - start_time,
        **extra,
    }


async def setup_and_run_native_bridge_codex(
    *,
    runtime: Any,
    launch: BridgeLaunchResult,
    task: str,
    working_dir: str = '/workspace',
    timeout: float = 3600.0,
    version: str = DEFAULT_CODEX_VERSION,
    model: str | None = None,
    npm_registry: str | None = DEFAULT_NPM_REGISTRY,
    binary_url: str | None = None,
    binary_checksum: str | None = None,
    extra_cli_args: list[str] | None = None,
    developer_instructions: str | None = None,
    extra_config_toml: str | None = None,
    model_instructions_file: str | None = None,
    extra_env: dict[str, str] | None = None,
    pre_start_scripts: list[dict[str, Any]] | None = None,
    mcp_servers: list[dict[str, Any]] | None = None,
    use_azure_provider: bool = True,
    auto_compact_token_limit: int | None = None,
    emit_json_trace: bool = True,
) -> dict[str, Any]:
    """Install and run one Codex prompt against the native bridge Responses endpoint."""
    start_time = time.time()
    adapter = build_codex_native_bridge_adapter(runtime)
    try:
        executable = await adapter.ensure_codex(
            CodexInstallSpec(
                version=version,
                npm_registry=npm_registry,
                binary_url=binary_url,
                binary_checksum=binary_checksum,
            )
        )
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        return _result_dict(
            success=False,
            output='',
            stderr=f'Failed to install Codex CLI: {detail}',
            failure_kind='infrastructure',
            failure_stage='install',
            exit_code=-1,
            start_time=start_time,
        )

    if pre_start_scripts:
        try:
            await adapter.run_pre_start_scripts(
                pre_start_scripts,
                working_dir=working_dir,
                launch=launch,
                extra_env=extra_env,
            )
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            return _result_dict(
                success=False,
                output='',
                stderr=f'Pre-start script failed: {detail}',
                failure_kind='infrastructure',
                failure_stage='pre_start',
                exit_code=-1,
                start_time=start_time,
            )

    try:
        run = await adapter.run_codex(
            CodexRunRequest(
                codex_executable=executable,
                launch=launch,
                task=task,
                working_dir=working_dir,
                timeout=timeout,
                version=version,
                npm_registry=npm_registry,
                model=model,
                extra_cli_args=extra_cli_args,
                developer_instructions=developer_instructions,
                extra_config_toml=extra_config_toml,
                model_instructions_file=model_instructions_file,
                extra_env=extra_env,
                mcp_servers=mcp_servers,
                use_azure_provider=use_azure_provider,
                auto_compact_token_limit=auto_compact_token_limit,
                emit_json_trace=emit_json_trace,
            )
        )
    except Exception as exc:
        detail = str(exc) or type(exc).__name__
        return _result_dict(
            success=False,
            output='',
            stderr=f'Failed to run Codex CLI: {detail}',
            failure_kind='infrastructure',
            failure_stage='launch',
            exit_code=-1,
            start_time=start_time,
        )

    return _result_dict(
        success=run.exit_code == 0 and not run.timed_out,
        output=run.output,
        stderr=run.stderr,
        exit_code=run.exit_code,
        start_time=start_time,
        timed_out=run.timed_out,
        failure_kind='agent' if run.exit_code != 0 or run.timed_out else None,
        failure_stage='process' if run.exit_code != 0 or run.timed_out else None,
        json_events_stdout=run.stdout if emit_json_trace else '',
    )


__all__ = ['setup_and_run_native_bridge_codex']
