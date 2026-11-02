"""Shared POSIX Claude Code native bridge adapter implementation.

Claude Code runs *detached* inside the sandbox with stdout/stderr redirected to log
files, and the host polls a sentinel until it exits. The sandbox OSWorld ``/execute``
endpoint caps any single request at ~120s, so a long Claude run cannot be driven by one
synchronous ``run_command`` — the launch + poll loop keeps every call well under that cap.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from contextlib import suppress
from typing import Any

from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.client.base import ExecutorMixin
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.utils.shell import quote_posix

from .packages import (
    DEFAULT_CLAUDE_INSTALL_DIR,
    installed_posix_claude_executable,
    npm_install_timeout,
    npm_registry_arg,
    package_spec,
)
from .shell import (
    build_claude_code_command,
    build_claude_code_env,
    build_pre_start_command,
    shell_join_cmd,
    shell_join_env,
)
from .types import ClaudeInstallSpec, ClaudeRunRequest, ClaudeRunResult


logger = logging.getLogger(__name__)

_BRIDGE_RUN_ROOT = '/tmp/nexus-claude-run'
_POLL_INTERVAL = 5.0
# Control commands (launch / poll / read / kill) are all instant, far under the sandbox
# /execute ~120s cap.
_CONTROL_COMMAND_TIMEOUT = 60.0


class _SessionPaths:
    """Sandbox paths for one detached Claude Code run."""

    def __init__(self, session_dir: str) -> None:
        self.dir = session_dir
        self.stdout_log = f'{session_dir}/stdout.log'
        self.stderr_log = f'{session_dir}/stderr.log'
        self.exit_code = f'{session_dir}/exit_code'
        self.pid = f'{session_dir}/pid'


def _command_ok(result: RunCommandResponse) -> bool:
    return result.status == CommandRunStatus.Finished and int(result.return_code or 0) == 0


class BasePosixClaudeNativeBridgeAdapter:
    """Shared install and run operations for POSIX native bridge targets."""

    def __init__(self, runtime: ExecutorMixin) -> None:
        self.runtime = runtime

    async def ensure_claude(self, spec: ClaudeInstallSpec) -> str:
        if spec.binary_url:
            return await self._install_from_binary_url(spec)
        if spec.version != 'latest':
            if not spec.ensure_claude:
                raise RuntimeError('A pinned Claude Code version requires ensure_claude=true to install it with npm.')
            return await self._install_via_npm(spec)

        existing = await self._find_existing_claude()
        if existing:
            return existing

        if not spec.ensure_claude:
            raise RuntimeError(
                'Claude Code executable not found in sandbox. Set ensure_claude=true to install it with npm.'
            )

        return await self._install_via_npm(spec)

    async def run_claude(self, request: ClaudeRunRequest) -> ClaudeRunResult:
        paths = _SessionPaths(f'{_BRIDGE_RUN_ROOT}-{uuid.uuid4().hex}')
        may_be_running = False
        try:
            may_be_running = True
            launch_result = await self.runtime.run_command(
                command=self._build_launch_command(request, paths), timeout=_CONTROL_COMMAND_TIMEOUT
            )
            if not _command_ok(launch_result) or 'LAUNCHED' not in (launch_result.stdout or ''):
                detail = (launch_result.stderr or launch_result.stdout or str(launch_result.status))[:2000]
                return ClaudeRunResult(exit_code=-1, stdout='', stderr=f'Failed to launch Claude Code: {detail}')

            exit_code, timed_out, vanished = await self._poll_until_done(paths, request.timeout)
            may_be_running = timed_out or vanished
            stdout = await self._read_file(paths.stdout_log)
            stderr = await self._read_file(paths.stderr_log)

            if timed_out:
                stderr = f'{stderr}\n[Claude Code timed out after {request.timeout:g}s]'.strip()
                return ClaudeRunResult(exit_code=-1, stdout=stdout, stderr=stderr)
            if vanished:
                stderr = f'{stderr}\n[Claude Code exited without writing an exit code]'.strip()
                return ClaudeRunResult(exit_code=-1, stdout=stdout, stderr=stderr)
            return ClaudeRunResult(exit_code=exit_code if exit_code is not None else -1, stdout=stdout, stderr=stderr)
        finally:
            await self._run_cleanup(paths, terminate=may_be_running)

    async def _poll_until_done(self, paths: _SessionPaths, timeout: float) -> tuple[int | None, bool, bool]:
        """Poll the sentinel until Claude exits. Returns (exit_code, timed_out, vanished)."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        poll_command = self._build_poll_command(paths)
        while True:
            result = await self.runtime.run_command(command=poll_command, timeout=_CONTROL_COMMAND_TIMEOUT)
            lines = (result.stdout or '').strip().splitlines()
            status = lines[-1] if lines else ''
            if status.startswith('DONE'):
                parts = status.split()
                code = parts[1] if len(parts) > 1 else ''
                return (int(code) if code.lstrip('-').isdigit() else -1), False, False
            if status == 'GONE':
                # GONE means the wrapper exited without (yet) writing the exit
                # code. Under runner load the exit-code file can land a moment
                # after the process disappears, so re-check once via the same
                # poll before concluding the code was lost.
                await asyncio.sleep(_POLL_INTERVAL)
                result = await self.runtime.run_command(command=poll_command, timeout=_CONTROL_COMMAND_TIMEOUT)
                lines = (result.stdout or '').strip().splitlines()
                status = lines[-1] if lines else ''
                if status.startswith('DONE'):
                    parts = status.split()
                    code = parts[1] if len(parts) > 1 else ''
                    return (int(code) if code.lstrip('-').isdigit() else -1), False, False
                return None, False, True
            if loop.time() >= deadline:
                return None, True, False
            await asyncio.sleep(_POLL_INTERVAL)

    async def run_pre_start_scripts(
        self,
        scripts: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """Run pre-start scripts in order inside the sandbox before Claude Code starts.

        Each script shares Claude's launch env (so it can prep auth / files against the
        same bridge) plus its own ``env`` overrides. A non-zero exit aborts when the
        script sets ``fail_on_error`` (default True).
        """
        base_env = build_claude_code_env(local_bridge_base_url=launch.local_base_url, extra_env=extra_env)
        for script in scripts:
            await self._run_pre_start_script(script, working_dir=working_dir, base_env=base_env)

    async def register_mcp_servers(
        self,
        claude_executable: str,
        mcp_servers: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        env = build_claude_code_env(local_bridge_base_url=launch.local_base_url, extra_env=extra_env)
        await self.runtime.run_command(
            command=f'mkdir -p {quote_posix(working_dir)}',
            timeout=_CONTROL_COMMAND_TIMEOUT,
        )
        for mcp_cfg in mcp_servers:
            server_name = str(mcp_cfg.get('name') or '')
            server_command = str(mcp_cfg.get('command') or '')
            server_args = [str(arg) for arg in mcp_cfg.get('args') or []]
            if not server_name or not server_command:
                logger.warning('Skipping invalid MCP server config: %s', mcp_cfg)
                continue
            command = shell_join_cmd([
                claude_executable,
                'mcp',
                'add',
                '--transport',
                'stdio',
                server_name,
                '--',
                server_command,
                *server_args,
            ])
            result = await self.runtime.run_command(
                command=f'{shell_join_env(env)} {command}',
                cwd=working_dir,
                timeout=30.0,
            )
            if not _command_ok(result):
                detail = (result.stderr or result.stdout or str(result.status))[:2000]
                logger.warning('MCP add %s exited %s: %s', server_name, result.return_code, detail)

    async def _run_pre_start_script(
        self, script: dict[str, Any], *, working_dir: str, base_env: dict[str, str]
    ) -> None:
        interpreter = script.get('interpreter', 'bash')
        code = script.get('code', '')
        timeout = float(script.get('timeout', 300.0))
        cwd = script.get('cwd') or working_dir
        fail_on_error = bool(script.get('fail_on_error', True))
        env = dict(base_env)
        if script.get('env'):
            env.update(script['env'])

        shell = build_pre_start_command(interpreter, code, cwd=cwd, env=env)
        result = await self.runtime.run_command(command=shell, timeout=timeout)
        if not _command_ok(result) and fail_on_error:
            detail = (result.stderr or result.stdout or str(result.status))[:2000]
            raise RuntimeError(
                f'pre-start ({interpreter}) failed status={result.status} rc={result.return_code}: {detail}'
            )

    async def _find_existing_claude(self) -> str | None:
        check = await self.runtime.run_command(command='command -v claude', timeout=30.0)
        detected = (check.stdout or '').strip()
        if check.return_code == 0 and detected:
            return detected.splitlines()[0]

        executable = installed_posix_claude_executable()
        check = await self.runtime.run_command(command=f'test -x {quote_posix(executable)}', timeout=30.0)
        return executable if check.return_code == 0 else None

    async def _install_via_npm(self, spec: ClaudeInstallSpec) -> str:
        install_dir = DEFAULT_CLAUDE_INSTALL_DIR
        executable = installed_posix_claude_executable()
        registry = npm_registry_arg(spec.npm_registry)
        cmd = ' '.join([
            'rm',
            '-rf',
            quote_posix(install_dir),
            '&&',
            'mkdir',
            '-p',
            quote_posix(install_dir),
            '&&',
            'npm',
            'install',
            '--prefix',
            quote_posix(install_dir),
            '--no-audit',
            '--no-fund',
            '--strict-ssl=false',
            '--registry',
            quote_posix(registry),
            quote_posix(package_spec(spec.version)),
            '&&',
            'test',
            '-x',
            quote_posix(executable),
            '&&',
            quote_posix(executable),
            '--version',
        ])
        result = await self.runtime.run_command(command=cmd, timeout=npm_install_timeout())
        if result.return_code != 0:
            raise RuntimeError(
                'Failed to install Claude Code via npm: '
                f'{(result.stderr or result.stdout or str(result.status))[:2000]}'
            )
        return executable

    async def _install_from_binary_url(self, spec: ClaudeInstallSpec) -> str:
        if not spec.binary_url:
            raise ValueError('Claude Code binary installation requires binary_url')
        install_dir = DEFAULT_CLAUDE_INSTALL_DIR
        temporary = f'{install_dir}/claude.url.tmp'
        download = (
            f'if command -v curl >/dev/null 2>&1; then '
            f'curl --fail --location --retry 3 --connect-timeout 30 '
            f'--output {quote_posix(temporary)} {quote_posix(spec.binary_url)}; '
            f'elif command -v wget >/dev/null 2>&1; then '
            f'wget --tries=4 --timeout=30 --output-document={quote_posix(temporary)} '
            f'{quote_posix(spec.binary_url)}; '
            "else echo 'curl or wget is required to download Claude Code' >&2; exit 127; fi"
        )
        checksum = ''
        if spec.binary_checksum:
            expected = quote_posix(spec.binary_checksum.lower())
            checksum = (
                f' && expected_checksum={expected}'
                ' && if command -v sha256sum >/dev/null 2>&1; then '
                f"actual_checksum=$(sha256sum {quote_posix(temporary)} | awk '{{print $1}}'); "
                'elif command -v shasum >/dev/null 2>&1; then '
                f"actual_checksum=$(shasum -a 256 {quote_posix(temporary)} | awk '{{print $1}}'); "
                "else echo 'sha256sum or shasum is required to verify Claude Code' >&2; exit 127; fi"
                ' && if [ "$actual_checksum" != "$expected_checksum" ]; then '
                "printf 'Checksum verification failed: expected %s, got %s\\n' "
                '"$expected_checksum" "$actual_checksum" >&2; '
                f'rm -f {quote_posix(temporary)}; exit 1; fi'
            )
        is_tarball = spec.binary_url.split('?')[0].endswith(('.tgz', '.tar.gz'))
        executable = f'{install_dir}/package/claude' if is_tarball else f'{install_dir}/claude'
        if is_tarball:
            unpack = f'tar xzf {quote_posix(temporary)} -C {quote_posix(install_dir)} && rm -f {quote_posix(temporary)}'
        else:
            unpack = f'chmod +x {quote_posix(temporary)} && mv {quote_posix(temporary)} {quote_posix(executable)}'
        command = (
            f'rm -rf {quote_posix(install_dir)} && '
            f'mkdir -p {quote_posix(install_dir)} && '
            f'{download}{checksum} && '
            f'{unpack} && '
            f'{quote_posix(executable)} --version'
        )
        result = await self.runtime.run_command(command=command, timeout=npm_install_timeout())
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or str(result.status))[:2000]
            raise RuntimeError(f'Failed to install Claude Code from binary URL: {detail}')
        reported = (result.stdout or '').strip().split()
        if spec.version != 'latest' and (not reported or reported[0] != spec.version):
            raise RuntimeError(f'Claude Code from binary URL reports {result.stdout!r}, expected {spec.version}')
        return executable

    def _build_launch_command(self, request: ClaudeRunRequest, paths: _SessionPaths) -> str:
        env = build_claude_code_env(
            local_bridge_base_url=request.launch.local_base_url,
            extra_env=request.extra_env,
        )
        cmd = build_claude_code_command(
            request.claude_executable,
            task=request.task,
            extra_cli_args=request.extra_cli_args,
        )
        # The detached shell: run Claude with output redirected, then record its exit code.
        inner = (
            f'{shell_join_env(env)} {shell_join_cmd(cmd)} '
            f'> {quote_posix(paths.stdout_log)} 2> {quote_posix(paths.stderr_log)}; '
            f'echo $? > {quote_posix(paths.exit_code)}'
        )
        wd = quote_posix(request.working_dir)
        return f'mkdir -p {wd}; mkdir -p {quote_posix(paths.dir)}; cd {wd} && ' + self._launch_detached_shell(
            inner, paths.pid
        )

    def _launch_detached_shell(self, command: str, pid_file: str) -> str:
        return (
            f'{{ nohup bash -c {quote_posix(command)} </dev/null >/dev/null 2>&1 & '
            f'echo $! > {quote_posix(pid_file)}; echo LAUNCHED; }}'
        )

    def _build_poll_command(self, paths: _SessionPaths) -> str:
        exit_code = quote_posix(paths.exit_code)
        pid = quote_posix(paths.pid)
        return (
            f'if [ -f {exit_code} ]; then echo "DONE $(cat {exit_code})"; '
            f'elif kill -0 "$(cat {pid} 2>/dev/null)" 2>/dev/null; then echo RUNNING; '
            f'else echo GONE; fi'
        )

    def _build_kill_command(self, paths: _SessionPaths) -> str:
        pid = quote_posix(paths.pid)
        return (
            f'pid=$(cat {pid} 2>/dev/null); '
            'descendants() { '
            'for child in $(ps -eo pid=,ppid= | while read c p; do [ "$p" = "$1" ] && echo "$c"; done); do '
            'descendants "$child"; echo "$child"; '
            'done; '
            '}; '
            'if [ -n "$pid" ]; then '
            'children=$(descendants "$pid"); '
            '[ -z "$children" ] || kill -TERM $children 2>/dev/null || true; '
            'kill -TERM "$pid" 2>/dev/null || true; '
            'sleep 1; '
            '[ -z "$children" ] || kill -KILL $children 2>/dev/null || true; '
            'kill -KILL "$pid" 2>/dev/null || true; '
            'fi; true'
        )

    async def _cleanup_session(self, paths: _SessionPaths, *, terminate: bool) -> None:
        if terminate:
            with suppress(Exception):
                await self.runtime.run_command(
                    command=self._build_kill_command(paths), timeout=_CONTROL_COMMAND_TIMEOUT
                )
        with suppress(Exception):
            await self.runtime.run_command(command=f'rm -rf {quote_posix(paths.dir)}', timeout=_CONTROL_COMMAND_TIMEOUT)

    async def _run_cleanup(self, paths: _SessionPaths, *, terminate: bool) -> None:
        cleanup = asyncio.create_task(self._cleanup_session(paths, terminate=terminate))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _read_file(self, path: str) -> str:
        result = await self.runtime.run_command(
            command=f'cat {quote_posix(path)} 2>/dev/null || true', timeout=_CONTROL_COMMAND_TIMEOUT
        )
        return result.stdout or ''


__all__ = ['BasePosixClaudeNativeBridgeAdapter']
