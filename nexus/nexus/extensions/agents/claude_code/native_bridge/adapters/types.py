"""Shared types for Claude Code native bridge adapters."""

from __future__ import annotations

from dataclasses import dataclass

from nexus.runtime.bridge.adapters.base import BridgeLaunchResult


@dataclass(frozen=True)
class ClaudeInstallSpec:
    """Claude Code install options for a native sandbox target."""

    ensure_claude: bool
    npm_registry: str | None
    version: str
    binary_url: str | None = None
    binary_checksum: str | None = None


@dataclass(frozen=True)
class ClaudeRunRequest:
    """Request to run one non-interactive Claude Code prompt."""

    claude_executable: str
    launch: BridgeLaunchResult
    task: str
    working_dir: str
    timeout: float
    extra_env: dict[str, str] | None = None
    extra_cli_args: list[str] | None = None


@dataclass(frozen=True)
class ClaudeRunResult:
    """Result from a native Claude Code process."""

    exit_code: int
    stdout: str
    stderr: str


__all__ = ['ClaudeInstallSpec', 'ClaudeRunRequest', 'ClaudeRunResult']
