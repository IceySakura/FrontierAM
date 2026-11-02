"""Shared types for Codex native bridge adapters."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.runtime.bridge.adapters.base import BridgeLaunchResult


@dataclass(frozen=True)
class CodexInstallSpec:
    version: str
    npm_registry: str | None
    binary_url: str | None = None
    binary_checksum: str | None = None


@dataclass(frozen=True)
class CodexRunRequest:
    codex_executable: str
    launch: BridgeLaunchResult
    task: str
    working_dir: str
    timeout: float
    version: str
    npm_registry: str | None
    model: str | None = None
    extra_cli_args: list[str] | None = None
    developer_instructions: str | None = None
    extra_config_toml: str | None = None
    model_instructions_file: str | None = None
    model_catalog_json: str | None = None
    extra_env: dict[str, str] | None = None
    mcp_servers: list[dict[str, object]] | None = None
    use_azure_provider: bool = True
    auto_compact_token_limit: int | None = None
    emit_json_trace: bool = True
    record_shell_history: bool = False
    strict_cleanup: bool = False


@dataclass(frozen=True)
class CodexRunResult:
    exit_code: int
    stdout: str
    stderr: str
    output: str
    timed_out: bool = False
    failure_kind: str | None = None
    failure_stage: str | None = None


__all__ = ['CodexInstallSpec', 'CodexRunRequest', 'CodexRunResult']
