"""Windows/WAA Codex native bridge adapter."""

from __future__ import annotations

import asyncio
import base64
import ntpath
import uuid

from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from nexus.runtime.bridge.adapters.ags_waa import _encoded_powershell, _ps_literal
from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.providers.ags.waa import AGSRawWindowsAgentArenaRuntime, WAAExecutorRuntime, require_raw_waa_runtime

from .packages import npm_registry_arg, package_spec
from .posix import _OUTPUT_MARKER, _extract_output
from .shell import build_codex_command, build_codex_config, build_codex_env
from .types import CodexInstallSpec, CodexRunRequest, CodexRunResult


_BRIDGE_RUN_ROOT = r'C:\Windows\Temp\nexus-codex-run'
_INSTALL_DIR = r'C:\Windows\Temp\nexus-codex-cli'
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


class WAACodexNativeBridgeAdapter:
    """Codex install and detached-run operations for WAA native bridge targets."""

    def __init__(self, runtime: WAAExecutorRuntime) -> None:
        self.runtime: AGSRawWindowsAgentArenaRuntime = require_raw_waa_runtime(runtime)

    async def _execute(self, script: str, *, timeout: float = _CONTROL_COMMAND_TIMEOUT) -> dict[str, Any]:
        result = await asyncio.wait_for(self.runtime.execute(_encoded_powershell(script), shell=False), timeout=timeout)
        if not isinstance(result, dict):
            raise RuntimeError(f'WAA execute returned unexpected payload: {result!r}')
        return result

    async def ensure_codex(self, spec: CodexInstallSpec) -> str:
        if spec.binary_url:
            executable = rf'{_INSTALL_DIR}\codex.exe'
            temporary = f'{executable}.url.tmp'
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
            result = await self._execute(
                "$ProgressPreference = 'SilentlyContinue'; "
                f'Remove-Item -LiteralPath {_ps_literal(_INSTALL_DIR)} -Recurse -Force '
                '-ErrorAction SilentlyContinue; '
                f'New-Item -ItemType Directory -Force -Path {_ps_literal(_INSTALL_DIR)} | Out-Null; '
                f'Invoke-WebRequest -UseBasicParsing -Uri {_ps_literal(spec.binary_url)} '
                f'-OutFile {_ps_literal(temporary)} -ErrorAction Stop; '
                f'{checksum}'
                f'Move-Item -LiteralPath {_ps_literal(temporary)} -Destination {_ps_literal(executable)} -Force; '
                f'& {_ps_literal(executable)} --version; exit $LASTEXITCODE',
                timeout=1800,
            )
            if int(result.get('returncode', -1)) != 0:
                detail = result.get('error') or result.get('output') or result
                raise RuntimeError(f'Failed to install Codex CLI from binary URL: {str(detail)[:2000]}')
            return executable

        npm_check = await self._execute(
            '$command = Get-Command npm.cmd -ErrorAction SilentlyContinue; if ($command) { $command.Source }'
        )
        npm_lines = str(npm_check.get('output') or '').strip().splitlines()
        if int(npm_check.get('returncode', -1)) != 0 or not npm_lines:
            detail = npm_check.get('error') or npm_check.get('output') or npm_check
            raise RuntimeError(f'npm.cmd not found in WAA sandbox: {str(detail)[:2000]}')

        npm_executable = npm_lines[-1].strip()
        executable = rf'{_INSTALL_DIR}\node_modules\.bin\codex.cmd'
        registry = npm_registry_arg(spec.npm_registry)
        result = await self._execute(
            f'Remove-Item -LiteralPath {_ps_literal(_INSTALL_DIR)} -Recurse -Force '
            '-ErrorAction SilentlyContinue; '
            f'New-Item -ItemType Directory -Force -Path {_ps_literal(_INSTALL_DIR)} | Out-Null; '
            f'& {_ps_literal(npm_executable)} install --prefix {_ps_literal(_INSTALL_DIR)} '
            '--no-audit --no-fund --strict-ssl=false '
            f'--registry {_ps_literal(registry)} {_ps_literal(package_spec(spec.version))}; '
            'if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; '
            f'if (-not (Test-Path -LiteralPath {_ps_literal(executable)})) {{ '
            f'throw {_ps_literal(f"Codex executable missing after npm install: {executable}")} }}; '
            f'& {_ps_literal(executable)} --version; exit $LASTEXITCODE',
            timeout=1800,
        )
        if int(result.get('returncode', -1)) != 0:
            detail = result.get('error') or result.get('output') or result
            raise RuntimeError(f'Failed to install Codex CLI via sandbox npm: {str(detail)[:2000]}')
        return executable

    async def run_codex(self, request: CodexRunRequest) -> CodexRunResult:
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
                return CodexRunResult(
                    exit_code=-1,
                    stdout='',
                    stderr=f'Failed to prepare Codex CLI: {detail}',
                    output='',
                )

            may_be_running = True
            await self.runtime.setup_launch(['cmd.exe', '/d', '/s', '/c', runner_cmd_path], shell=False)

            exit_code, timed_out, vanished = await self._poll_until_done(paths, request.timeout)
            may_be_running = timed_out or vanished
            stdout = await self._read_file(paths.stdout_log)
            stderr = await self._read_file(paths.stderr_log)
            stdout, output = _extract_output(stdout)
            if timed_out:
                stderr = f'{stderr}\n[Codex CLI timed out after {request.timeout:g}s]'.strip()
                return CodexRunResult(exit_code=-1, stdout=stdout, stderr=stderr, output=output, timed_out=True)
            if vanished:
                stderr = f'{stderr}\n[Codex CLI exited without writing an exit code]'.strip()
                return CodexRunResult(exit_code=-1, stdout=stdout, stderr=stderr, output=output)
            return CodexRunResult(
                exit_code=exit_code if exit_code is not None else -1,
                stdout=stdout,
                stderr=stderr,
                output=output,
            )
        finally:
            await self._run_cleanup(paths, terminate=may_be_running)

    async def run_pre_start_scripts(
        self,
        scripts: list[dict[str, Any]],
        *,
        working_dir: str,
        launch: BridgeLaunchResult,  # noqa: ARG002 - kept for the adapter protocol
        extra_env: dict[str, str] | None = None,
    ) -> None:
        base_env = dict(extra_env or {})
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

    async def _expand_windows_path(self, path: str) -> str:
        result = await self._execute(f'[Environment]::ExpandEnvironmentVariables({_ps_literal(path)})')
        expanded = str(result.get('output') or '').strip()
        if int(result.get('returncode', -1)) != 0 or not expanded:
            raise RuntimeError(f'Failed to expand WAA working directory: {result}')
        return expanded.splitlines()[-1].strip()

    def _build_runner_script(self, request: Any, paths: Any, *, working_dir: str) -> str:
        codex_home = rf'{paths.directory}\codex-home'
        config_path = rf'{codex_home}\config.toml'
        output_file = rf'{codex_home}\last_message.txt'
        config_encoded = base64.b64encode(build_codex_config(request).encode()).decode()
        env = build_codex_env(codex_home=codex_home, extra_env=request.extra_env)
        # codex.cmd treats literal newlines in an argv item as command boundaries on Windows.
        prompt_request = replace(
            request,
            task=request.task.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' '),
            developer_instructions=(request.developer_instructions or '').replace('\r\n', ' ').replace('\n', ' ')
            or None,
            working_dir=working_dir,
        )
        command = build_codex_command(prompt_request, output_file=output_file)
        executable, args = command[0], command[1:]
        parts = ["$ErrorActionPreference = 'Continue'"]
        parts.extend(f'$env:{key} = {_ps_literal(str(value))}' for key, value in env.items())
        parts.append(f'$env:PATH = {_ps_literal(ntpath.dirname(executable) + ";")} + $env:PATH')
        parts.extend([
            f'New-Item -ItemType Directory -Force -Path {_ps_literal(codex_home)} | Out-Null',
            f'[IO.File]::WriteAllBytes({_ps_literal(config_path)}, '
            f'[Convert]::FromBase64String({_ps_literal(config_encoded)}))',
            f'Set-Content -LiteralPath {_ps_literal(paths.pid)} -Value $PID -Encoding ascii',
            f'Set-Location -LiteralPath {_ps_literal(working_dir)}',
            # Close stdin explicitly. /setup/launch otherwise leaves its pipe open and Codex
            # waits for "additional input from stdin" after completing the requested task.
            f'$null | & {_ps_literal(executable)} @({_ps_array(args)[2:-1]}) '
            f'> {_ps_literal(paths.stdout_log)} 2> {_ps_literal(paths.stderr_log)}',
            '$code = $LASTEXITCODE',
            f'if ($null -eq $code) {{ $code = 1; $Error | Out-String | '
            f'Add-Content -LiteralPath {_ps_literal(paths.stderr_log)} }}',
            f'if (Test-Path -LiteralPath {_ps_literal(output_file)}) {{ '
            f'$encoded = [Convert]::ToBase64String([IO.File]::ReadAllBytes({_ps_literal(output_file)})); '
            f'Add-Content -LiteralPath {_ps_literal(paths.stdout_log)} '
            f'-Value ({_ps_literal(_OUTPUT_MARKER)} + $encoded) }}',
            # Publish the sentinel atomically. WAA polls over HTTP and can otherwise observe the
            # file between CreateFile and WriteFile, parse an empty exit code, and clean up a
            # still-running Codex process as exit_code=-1.
            f'Set-Content -LiteralPath {_ps_literal(paths.exit_code + ".tmp")} -Value $code -Encoding ascii',
            f'Move-Item -LiteralPath {_ps_literal(paths.exit_code + ".tmp")} '
            f'-Destination {_ps_literal(paths.exit_code)} -Force',
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


__all__ = ['WAACodexNativeBridgeAdapter']
