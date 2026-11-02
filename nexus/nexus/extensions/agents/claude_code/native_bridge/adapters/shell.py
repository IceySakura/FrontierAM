"""Claude Code command and shell helpers for native bridge runs."""

from __future__ import annotations

from nexus.extensions.agents.installed_common import DEFAULT_PROXY_API_KEY
from nexus.utils.shell import quote_posix


def build_claude_code_env(*, local_bridge_base_url: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build environment variables that point Claude Code at the local bridge."""
    env = {
        'ANTHROPIC_BASE_URL': local_bridge_base_url.rstrip('/'),
        'ANTHROPIC_API_KEY': DEFAULT_PROXY_API_KEY,
        'IS_SANDBOX': '1',
        'NO_COLOR': '1',
        'CLAUDE_ENABLE_STREAM_WATCHDOG': '0',
    }
    if extra_env:
        env.update(extra_env)
    return env


def build_claude_code_command(
    claude_executable: str,
    *,
    task: str,
    extra_cli_args: list[str] | None = None,
) -> list[str]:
    """Build a non-interactive Claude Code command for a single task."""
    args = list(extra_cli_args or [])
    has_print = any(arg in {'--print', '-p'} or arg.startswith('--print=') for arg in args)
    cmd = [claude_executable, '--dangerously-skip-permissions']
    if not has_print:
        cmd.append('--print')
    cmd.extend(args)
    cmd.extend(['--', task])
    return cmd


def shell_join_env(env: dict[str, str]) -> str:
    """Shell-quote environment assignments for POSIX shells."""
    return ' '.join(f'{key}={quote_posix(value)}' for key, value in env.items())


def shell_join_cmd(cmd: list[str]) -> str:
    """Shell-quote a command argv list for POSIX shells."""
    return ' '.join(quote_posix(part) for part in cmd)


def build_pre_start_command(interpreter: str, code: str, *, cwd: str, env: dict[str, str]) -> str:
    """Build a POSIX shell command running a pre-start script under *cwd* with *env*."""
    if interpreter == 'bash':
        inner = f'bash -c {quote_posix(code)}'
    elif interpreter == 'python':
        inner = f'python3 -c {quote_posix(code)}'
    else:
        raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')
    quoted_cwd = quote_posix(cwd)
    return ' '.join([
        f'mkdir -p {quoted_cwd}',
        '&&',
        f'cd {quoted_cwd}',
        '&&',
        f'{shell_join_env(env)} {inner}',
    ])


__all__ = [
    'build_claude_code_command',
    'build_claude_code_env',
    'build_pre_start_command',
    'shell_join_cmd',
    'shell_join_env',
]
