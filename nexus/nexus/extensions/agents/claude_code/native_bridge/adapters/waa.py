"""Windows/WAA Claude Code native bridge adapter."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from nexus.runtime.bridge.adapters.ags_waa import _encoded_powershell, _ps_literal
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, WAAExecutorRuntime, require_raw_waa_runtime

from .packages import npm_install_timeout, npm_registry_arg, package_spec
from .shell import build_claude_code_command, build_claude_code_env
from .types import ClaudeInstallSpec, ClaudeRunRequest, ClaudeRunResult


logger = logging.getLogger(__name__)

_BRIDGE_RUN_ROOT = r'C:\Windows\Temp\nexus-claude-run'
_INSTALL_DIR = r'C:\Windows\Temp\nexus-claude-code'
_POLL_INTERVAL = 5.0
_CONTROL_COMMAND_TIMEOUT = 60.0


@dataclass(frozen=True)
class _SessionPaths:
    directory: str
    stdout_log: str
    stderr_log: str
    exit_code: str
    pid: str


def _ps_array(values: list[str]) -> str:
    return '@(' + ', '.join(_ps_literal(value) for value in values) + ')'


class WAAClaudeNativeBridgeAdapter:
    def __init__(self, runtime: WAAExecutorRuntime) -> None:
        self.runtime: AGSRawWindowsAgentArenaRuntime = require_raw_waa_runtime(runtime)

    async def _execute(self, script: str, *, timeout: float = _CONTROL_COMMAND_TIMEOUT) -> dict[str, Any]:
        result = await asyncio.wait_for(self.runtime.execute(_encoded_powershell(script), shell=False), timeout=timeout)
        if not isinstance(result, dict):
            raise RuntimeError(f'WAA execute returned unexpected payload: {result!r}')
        return result

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

    async def _find_existing_claude(self) -> str | None:
        result = await self._execute(
            '$command = Get-Command claude.cmd -ErrorAction SilentlyContinue; if ($command) { $command.Source }'
        )
        detected = str(result.get('output') or '').strip()
        return detected.splitlines()[0].strip() if int(result.get('returncode', -1)) == 0 and detected else None

    async def _install_via_npm(self, spec: ClaudeInstallSpec) -> str:
        executable = rf'{_INSTALL_DIR}\node_modules\.bin\claude.cmd'
        registry = npm_registry_arg(spec.npm_registry)
        result = await self._execute(
            f'Remove-Item -LiteralPath {_ps_literal(_INSTALL_DIR)} -Recurse -Force -ErrorAction SilentlyContinue; '
            f'New-Item -ItemType Directory -Force -Path {_ps_literal(_INSTALL_DIR)} | Out-Null; '
            f'& npm.cmd install --prefix {_ps_literal(_INSTALL_DIR)} --no-audit --no-fund --strict-ssl=false '
            f'--registry {_ps_literal(registry)} {_ps_literal(package_spec(spec.version))}; '
            'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; '
            f'& {_ps_literal(executable)} --version',
            timeout=npm_install_timeout(),
        )
        if int(result.get('returncode', -1)) != 0:
            detail = result.get('error') or result.get('output') or result
            raise RuntimeError(f'Failed to install Claude Code via npm: {str(detail)[:2000]}')
        return executable

    async def _install_from_binary_url(self, spec: ClaudeInstallSpec) -> str:
        if not spec.binary_url:
            raise ValueError('Claude Code binary installation requires binary_url')
        temporary = rf'{_INSTALL_DIR}\claude.url.tmp'
        checksum = ''
        if spec.binary_checksum:
            checksum = (
                f'$expectedChecksum = {_ps_literal(spec.binary_checksum)}; '
                f'$actualChecksum = (Get-FileHash -LiteralPath {_ps_literal(temporary)} '
                '-Algorithm SHA256).Hash.ToLowerInvariant(); '
                'if ($actualChecksum -ne $expectedChecksum.ToLowerInvariant()) { '
                "throw ('Checksum verification failed: expected ' + $expectedChecksum + ', got ' "
                '+ $actualChecksum) }; '
            )
        # An npm native package (claude-code-win32-<arch>-<version>.tgz) holds the executable at
        # package\claude.exe; anything else is a standalone executable moved into place.
        is_tarball = spec.binary_url.split('?')[0].endswith(('.tgz', '.tar.gz'))
        if is_tarball:
            executable = rf'{_INSTALL_DIR}\package\claude.exe'
            unpack = (
                f'& tar.exe -xzf {_ps_literal(temporary)} -C {_ps_literal(_INSTALL_DIR)}; '
                'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; '
                f'Remove-Item -LiteralPath {_ps_literal(temporary)} -Force; '
            )
        else:
            executable = rf'{_INSTALL_DIR}\claude.exe'
            unpack = f'Move-Item -LiteralPath {_ps_literal(temporary)} -Destination {_ps_literal(executable)} -Force; '
        result = await self._execute(
            "$ProgressPreference = 'SilentlyContinue'; "
            f'Remove-Item -LiteralPath {_ps_literal(_INSTALL_DIR)} -Recurse -Force -ErrorAction SilentlyContinue; '
            f'New-Item -ItemType Directory -Force -Path {_ps_literal(_INSTALL_DIR)} | Out-Null; '
            f'Invoke-WebRequest -UseBasicParsing -Uri {_ps_literal(spec.binary_url)} '
            f'-OutFile {_ps_literal(temporary)} -ErrorAction Stop; '
            f'{checksum}'
            f'{unpack}'
            f'& {_ps_literal(executable)} --version; exit $LASTEXITCODE',
            timeout=npm_install_timeout(),
        )
        if int(result.get('returncode', -1)) != 0:
            detail = result.get('error') or result.get('output') or result
            raise RuntimeError(f'Failed to install Claude Code from binary URL: {str(detail)[:2000]}')
        reported = str(result.get('output') or '').strip().split()
        if spec.version != 'latest' and (not reported or reported[0] != spec.version):
            raise RuntimeError(f'Claude Code from binary URL reports {result.get("output")!r}, expected {spec.version}')
        return executable

    async def run_pre_start_scripts(
        self,
        scripts: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        base_env = build_claude_code_env(local_bridge_base_url=launch.local_base_url, extra_env=extra_env)
        for script in scripts:
            interpreter = script.get('interpreter', 'powershell')
            code = str(script.get('code', ''))
            timeout = float(script.get('timeout', 300.0))
            cwd = await self._expand_windows_path(str(script.get('cwd') or working_dir))
            fail_on_error = bool(script.get('fail_on_error', True))
            env = dict(base_env)
            env.update(script.get('env') or {})
            env_script = '; '.join(f'$env:{key} = {_ps_literal(str(value))}' for key, value in env.items())
            if interpreter == 'powershell':
                body = code
            elif interpreter == 'python':
                encoded = base64.b64encode(code.encode()).decode()
                body = f'& python -c "import base64;exec(base64.b64decode({_ps_literal(encoded)}))"'
            elif interpreter == 'bash':
                body = f'& bash.exe -lc {_ps_literal(code)}'
            else:
                raise ValueError(f'Unsupported pre-start interpreter: {interpreter}')
            result = await self._execute(
                f'New-Item -ItemType Directory -Force -Path {_ps_literal(cwd)} | Out-Null; '
                f'Set-Location -LiteralPath {_ps_literal(cwd)}; {env_script}; {body}; exit $LASTEXITCODE',
                timeout=timeout,
            )
            if int(result.get('returncode', -1)) != 0 and fail_on_error:
                detail = result.get('error') or result.get('output') or result
                raise RuntimeError(
                    f'pre-start ({interpreter}) failed rc={result.get("returncode")}: {str(detail)[:2000]}'
                )

    async def register_mcp_servers(
        self,
        claude_executable: str,
        mcp_servers: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        cwd = await self._expand_windows_path(working_dir)
        env = build_claude_code_env(local_bridge_base_url=launch.local_base_url, extra_env=extra_env)
        env_script = '; '.join(f'$env:{key} = {_ps_literal(str(value))}' for key, value in env.items())
        for mcp_cfg in mcp_servers:
            server_name = str(mcp_cfg.get('name') or '')
            server_command = str(mcp_cfg.get('command') or '')
            server_args = [str(arg) for arg in mcp_cfg.get('args') or []]
            if not server_name or not server_command:
                logger.warning('Skipping invalid MCP server config: %s', mcp_cfg)
                continue
            args = ['mcp', 'add', '--transport', 'stdio', server_name, '--', server_command, *server_args]
            result = await self._execute(
                f'New-Item -ItemType Directory -Force -Path {_ps_literal(cwd)} | Out-Null; '
                f'Set-Location -LiteralPath {_ps_literal(cwd)}; {env_script}; '
                f'& {_ps_literal(claude_executable)} @({_ps_array(args)[2:-1]}); exit $LASTEXITCODE',
                timeout=30.0,
            )
            if int(result.get('returncode', -1)) != 0:
                detail = result.get('error') or result.get('output') or result
                logger.warning('MCP add %s exited %s: %s', server_name, result.get('returncode'), str(detail)[:2000])

    async def run_claude(self, request: ClaudeRunRequest) -> ClaudeRunResult:
        working_dir = await self._expand_windows_path(request.working_dir)
        session_dir = f'{_BRIDGE_RUN_ROOT}-{uuid.uuid4().hex}'
        paths = _SessionPaths(
            directory=session_dir,
            stdout_log=rf'{session_dir}\stdout.log',
            stderr_log=rf'{session_dir}\stderr.log',
            exit_code=rf'{session_dir}\exit_code',
            pid=rf'{session_dir}\pid',
        )
        runner_path = rf'{session_dir}\runner.ps1'
        runner_cmd_path = rf'{session_dir}\runner.cmd'
        runner_script = self._build_runner_script(request, paths, working_dir=working_dir)
        runner_bytes = b'\xff\xfe' + runner_script.encode('utf-16-le')
        runner_encoded = base64.b64encode(runner_bytes).decode()
        runner_cmd = f'@powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{runner_path}"\r\n'
        runner_cmd_encoded = base64.b64encode(runner_cmd.encode('ascii')).decode()
        may_be_running = False
        try:
            prepared = await self._execute(
                f'New-Item -ItemType Directory -Force -Path {_ps_literal(working_dir)}, '
                f'{_ps_literal(paths.directory)} | Out-Null; '
                f'[IO.File]::WriteAllBytes({_ps_literal(runner_path)}, '
                f'[Convert]::FromBase64String({_ps_literal(runner_encoded)})); '
                f'[IO.File]::WriteAllBytes({_ps_literal(runner_cmd_path)}, '
                f'[Convert]::FromBase64String({_ps_literal(runner_cmd_encoded)}))'
            )
            if int(prepared.get('returncode', -1)) != 0:
                detail = prepared.get('error') or prepared.get('output') or prepared
                return ClaudeRunResult(exit_code=-1, stdout='', stderr=f'Failed to prepare Claude Code: {detail}')

            may_be_running = True
            await self.runtime.setup_launch(
                [
                    'cmd.exe',
                    '/d',
                    '/s',
                    '/c',
                    runner_cmd_path,
                ],
                shell=False,
            )

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

    async def _expand_windows_path(self, path: str) -> str:
        result = await self._execute(f'[Environment]::ExpandEnvironmentVariables({_ps_literal(path)})')
        expanded = str(result.get('output') or '').strip()
        if int(result.get('returncode', -1)) != 0 or not expanded:
            raise RuntimeError(f'Failed to expand WAA working directory: {result}')
        return expanded.splitlines()[-1].strip()

    def _build_runner_script(
        self,
        request: ClaudeRunRequest,
        paths: _SessionPaths,
        *,
        working_dir: str,
    ) -> str:
        env = build_claude_code_env(local_bridge_base_url=request.launch.local_base_url, extra_env=request.extra_env)
        # Windows npm command shims are batch files. A literal newline in an
        # argument is interpreted as a command boundary, so claude.cmd only
        # forwards the first line. Preserve the full prompt as one argument.
        task = request.task.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        command = build_claude_code_command(
            request.claude_executable,
            task=task,
            extra_cli_args=request.extra_cli_args,
        )
        executable, args = command[0], command[1:]
        parts = ["$ErrorActionPreference = 'Continue'"]
        parts.extend(f'$env:{key} = {_ps_literal(str(value))}' for key, value in env.items())
        parts.extend([
            f'Set-Content -LiteralPath {_ps_literal(paths.pid)} -Value $PID -Encoding ascii',
            f'Set-Location -LiteralPath {_ps_literal(working_dir)}',
            f'& {_ps_literal(executable)} @({_ps_array(args)[2:-1]}) '
            f'> {_ps_literal(paths.stdout_log)} 2> {_ps_literal(paths.stderr_log)}',
            '$code = $LASTEXITCODE',
            f'Set-Content -LiteralPath {_ps_literal(paths.exit_code)} -Value $code -Encoding ascii',
            'exit $code',
        ])
        return '; '.join(parts)

    async def _poll_until_done(self, paths: _SessionPaths, timeout: float) -> tuple[int | None, bool, bool]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            result = await self._execute(
                f'if (Test-Path -LiteralPath {_ps_literal(paths.exit_code)}) {{ '
                f'Write-Output ("DONE " + (Get-Content -LiteralPath {_ps_literal(paths.exit_code)})) }} '
                f'elseif (Test-Path -LiteralPath {_ps_literal(paths.pid)}) {{ '
                f'$processId = Get-Content -LiteralPath {_ps_literal(paths.pid)}; '
                'if (Get-Process -Id $processId -ErrorAction SilentlyContinue) { Write-Output "RUNNING" } '
                'else { Write-Output "GONE" } } else { Write-Output "STARTING" }'
            )
            lines = str(result.get('output') or '').strip().splitlines()
            status = lines[-1].strip() if lines else ''
            if status.startswith('DONE'):
                value = status.partition(' ')[2].strip()
                return (int(value) if value.lstrip('-').isdigit() else -1), False, False
            if status == 'GONE':
                return None, False, True
            if loop.time() >= deadline:
                return None, True, False
            await asyncio.sleep(_POLL_INTERVAL)

    def _build_kill_script(self, paths: _SessionPaths) -> str:
        return (
            f'if (Test-Path -LiteralPath {_ps_literal(paths.pid)}) {{ '
            f'$processId = Get-Content -LiteralPath {_ps_literal(paths.pid)}; '
            '& taskkill.exe /PID $processId /T /F 2>$null | Out-Null }'
        )

    async def _cleanup_session(self, paths: _SessionPaths, *, terminate: bool) -> None:
        if terminate:
            with suppress(Exception):
                await self._execute(self._build_kill_script(paths))
        with suppress(Exception):
            await self._execute(
                f'Remove-Item -LiteralPath {_ps_literal(paths.directory)} -Recurse -Force -ErrorAction SilentlyContinue'
            )

    async def _run_cleanup(self, paths: _SessionPaths, *, terminate: bool) -> None:
        cleanup = asyncio.create_task(self._cleanup_session(paths, terminate=terminate))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def _read_file(self, path: str) -> str:
        result = await self._execute(
            f'if (Test-Path -LiteralPath {_ps_literal(path)}) {{ Get-Content -Raw -LiteralPath {_ps_literal(path)} }}'
        )
        return str(result.get('output') or '')


__all__ = ['WAAClaudeNativeBridgeAdapter']
