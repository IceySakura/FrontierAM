"""Compose execution transport for installed Agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from nexus.extensions.agents.installed_common.execution import ExecutionTransportError
from nexus.utils.common import terminate_process


if TYPE_CHECKING:
    from nexus.extensions.agents.installed_common.execution import (
        AgentExecutionTransport,
        ComposeExecutionTarget,
        PreparedExecution,
    )


_PROXY_URL_PLACEHOLDER = '__NEXUS_AGENT_PROXY_URL__'
_TRANSPORT_MANAGED_ENV = frozenset({
    'ANTHROPIC_BASE_URL',
    'CODEX_HOME',
    'HOME',
    'NEXUS_AGENT_PROXY_URL',
    'OPENAI_BASE_URL',
    'PATH',
})
_ENV_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _compose_process_env(target: ComposeExecutionTarget) -> dict[str, str]:
    return {**os.environ, **target.compose_env}


@contextmanager
def _rewritten_proxy_files(
    paths: list[str | Path],
    *,
    transport_host_dir: str,
    source_url: str,
    target_url: str,
) -> Iterator[None]:
    originals: dict[Path, str] = {}
    try:
        for raw_path in paths:
            path = Path(raw_path)
            try:
                path.relative_to(transport_host_dir)
            except ValueError as exc:
                raise ExecutionTransportError(f'Proxy file escapes transport directory: {path}') from exc
            if not path.is_file():
                raise ExecutionTransportError(f'Proxy file is missing: {path}')
            content = path.read_text()
            if source_url not in content:
                raise ExecutionTransportError(f'Proxy URL not found in {path}')
            originals[path] = content
            path.write_text(content.replace(source_url, target_url))
        yield
    finally:
        for path, content in originals.items():
            with contextlib.suppress(OSError):
                path.write_text(content)


async def _run_capture(
    argv: list[str],
    *,
    timeout: float = 30,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await terminate_process(process)
        raise ExecutionTransportError(f'Command timed out: {argv[0]}') from None
    except BaseException:
        await asyncio.shield(terminate_process(process))
        raise
    return process.returncode or 0, stdout.decode(errors='replace'), stderr.decode(errors='replace')


class ProxyBridgeManager:
    """Bridge one Runtime-loopback LLM proxy into a Compose service."""

    def __init__(self, target: ComposeExecutionTarget, invocation_id: str, proxy_url: str) -> None:
        self.target = target
        self.invocation_id = invocation_id
        self.proxy_url = proxy_url
        self.outer_bridge_dir = Path(target.transport_host_dir) / 'bridge' / invocation_id
        self.inner_bridge_dir = f'{target.transport_container_dir}/bridge/{invocation_id}'
        self.outer_socket_path = self.outer_bridge_dir / 'llm-proxy.sock'
        self.inner_socket_path = f'{self.inner_bridge_dir}/llm-proxy.sock'
        self.outer_ready_file = self.outer_bridge_dir / 'inner-bridge.ready'
        self.inner_ready_file = f'{self.inner_bridge_dir}/inner-bridge.ready'
        self._outer_process: asyncio.subprocess.Process | None = None
        self._inner_pid: int | None = None

    def _compose(self, *subcommand: str) -> list[str]:
        argv = ['docker', 'compose', '--project-name', self.target.project_name]
        if self.target.project_directory is not None:
            argv.extend(['--project-directory', self.target.project_directory])
        for compose_file in self.target.compose_files:
            argv.extend(['--file', compose_file])
        argv.extend(subcommand)
        return argv

    async def _checked(self, argv: list[str], *, operation: str, timeout: float = 30) -> str:
        return_code, stdout, stderr = await _run_capture(
            argv,
            timeout=timeout,
            env=_compose_process_env(self.target),
        )
        if return_code != 0:
            details = (stderr or stdout or 'no output').strip()
            raise ExecutionTransportError(f'{operation} failed with code {return_code}: {details}')
        return stdout

    @staticmethod
    def _parse_proxy_url(value: str) -> tuple[int, str]:
        parsed = urlsplit(value)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.port is None:
            raise ExecutionTransportError(f'Unsupported loopback proxy URL: {value or "<empty>"}')
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ExecutionTransportError(f'Unsupported loopback proxy URL: {value}')
        return parsed.port, parsed.path.rstrip('/')

    async def __aenter__(self) -> str:
        outer_port, base_path = self._parse_proxy_url(self.proxy_url)
        self.outer_bridge_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.outer_bridge_dir, 0o777)
        self.outer_socket_path.unlink(missing_ok=True)
        self.outer_ready_file.unlink(missing_ok=True)

        try:
            self._outer_process = await asyncio.create_subprocess_exec(
                self.target.socket_bridge_host_path,
                '--unix-listen',
                str(self.outer_socket_path),
                '--tcp-target',
                f'127.0.0.1:{outer_port}',
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            for _ in range(100):
                if self.outer_socket_path.is_socket():
                    break
                if self._outer_process.returncode is not None:
                    break
                await asyncio.sleep(0.05)
            if not self.outer_socket_path.is_socket():
                raise ExecutionTransportError('Outer proxy bridge did not start')

            await self._checked(
                self._compose(
                    'exec',
                    '-T',
                    '-d',
                    '--user',
                    'root',
                    self.target.service,
                    self.target.socket_bridge_container_path,
                    '--tcp-listen',
                    '127.0.0.1:0',
                    '--unix-target',
                    self.inner_socket_path,
                    '--ready-file',
                    self.inner_ready_file,
                ),
                operation='Start inner proxy bridge',
            )

            inner_port: int | None = None
            for _ in range(100):
                if self.outer_ready_file.is_file():
                    try:
                        raw_pid, raw_port = self.outer_ready_file.read_text().split()
                        inner_pid = int(raw_pid)
                        candidate_port = int(raw_port)
                    except (OSError, ValueError):
                        pass
                    else:
                        if inner_pid > 0 and 1 <= candidate_port <= 65535:
                            return_code, _, _ = await _run_capture(
                                self._compose(
                                    'exec',
                                    '-T',
                                    '--user',
                                    'root',
                                    self.target.service,
                                    '/nix/runtime/nix-env/bin/bash',
                                    '-c',
                                    'kill -0 "$1"',
                                    'nexus-proxy-bridge',
                                    str(inner_pid),
                                ),
                                timeout=10,
                                env=_compose_process_env(self.target),
                            )
                            if return_code == 0:
                                self._inner_pid = inner_pid
                                inner_port = candidate_port
                                break
                await asyncio.sleep(0.05)
            if self._inner_pid is None or inner_port is None:
                raise ExecutionTransportError('Inner proxy bridge did not start')
            return f'http://127.0.0.1:{inner_port}{base_path}'
        except BaseException:
            await asyncio.shield(self.close())
            raise

    async def close(self) -> None:
        if self._inner_pid is not None:
            with contextlib.suppress(Exception):
                await _run_capture(
                    self._compose(
                        'exec',
                        '-T',
                        '--user',
                        'root',
                        self.target.service,
                        '/nix/runtime/nix-env/bin/bash',
                        '-c',
                        'kill "$1" 2>/dev/null || true',
                        'nexus-proxy-bridge',
                        str(self._inner_pid),
                    ),
                    timeout=10,
                    env=_compose_process_env(self.target),
                )
            self._inner_pid = None
        if self._outer_process is not None:
            await terminate_process(self._outer_process)
            self._outer_process = None
        shutil.rmtree(self.outer_bridge_dir, ignore_errors=True)

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        await self.close()


class ComposeExecutionTransport:
    """Translate one local Agent invocation into ``docker compose exec``."""

    def __init__(self, target: ComposeExecutionTarget) -> None:
        self.target = target

    def _compose(self, *subcommand: str) -> list[str]:
        argv = ['docker', 'compose', '--project-name', self.target.project_name]
        if self.target.project_directory is not None:
            argv.extend(['--project-directory', self.target.project_directory])
        for compose_file in self.target.compose_files:
            argv.extend(['--file', compose_file])
        argv.extend(subcommand)
        return argv

    async def _checked(self, argv: list[str], *, operation: str, timeout: float = 30) -> str:
        return_code, stdout, stderr = await _run_capture(
            argv,
            timeout=timeout,
            env=_compose_process_env(self.target),
        )
        if return_code != 0:
            details = (stderr or stdout or 'no output').strip()
            raise ExecutionTransportError(f'{operation} failed with code {return_code}: {details}')
        return stdout

    def _translate_path(self, value: str) -> str:
        return value.replace(self.target.transport_host_dir, self.target.transport_container_dir)

    def _user_args(self) -> list[str]:
        return ['--user', str(self.target.user)] if self.target.user is not None else []

    async def _inner_path(self) -> str:
        output = await self._checked(
            self._compose(
                'exec',
                '-T',
                *self._user_args(),
                self.target.service,
                '/nix/runtime/nix-env/bin/python3',
                '-c',
                'import os; print(os.environ.get("PATH", ""))',
            ),
            operation='Resolve target PATH',
        )
        return (
            output.replace('\r', '').replace('\n', '') or '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        )

    async def _terminate_agent(self, pid_file: Path) -> None:
        try:
            raw_pid = pid_file.read_text().strip()
        except OSError:
            return
        if not raw_pid.isdigit():
            return
        with contextlib.suppress(Exception):
            await _run_capture(
                self._compose(
                    'exec',
                    '-T',
                    *self._user_args(),
                    self.target.service,
                    '/nix/runtime/nix-env/bin/bash',
                    '-c',
                    'kill -TERM "$1" 2>/dev/null || exit 0; '
                    'for _ in {1..20}; do kill -0 "$1" 2>/dev/null || exit 0; sleep 0.05; done; '
                    'kill -KILL "$1" 2>/dev/null || true',
                    'nexus-compose-transport',
                    raw_pid,
                ),
                timeout=10,
                env=_compose_process_env(self.target),
            )

    @asynccontextmanager
    async def prepare_legacy(
        self,
        command: list[str],
        *,
        working_dir: str,
        invocation_id: str,
        env: dict[str, str],
        proxy_files: list[str | Path] | None = None,
    ) -> AsyncIterator[PreparedExecution]:
        from nexus.extensions.agents.installed_common.execution import PreparedExecution

        proxy_url = env.get('NEXUS_AGENT_PROXY_URL') or env.get('ANTHROPIC_BASE_URL') or env.get('OPENAI_BASE_URL')
        if not proxy_url:
            raise ExecutionTransportError('Agent execution target requires a loopback proxy URL')

        outer_home = Path(self.target.transport_host_dir) / 'home' / invocation_id
        inner_home = f'{self.target.transport_container_dir}/home/{invocation_id}'
        outer_bin = Path(self.target.transport_host_dir) / 'bridge' / invocation_id / 'bin'
        inner_bin = f'{self.target.transport_container_dir}/bridge/{invocation_id}/bin'
        outer_pid_file = Path(self.target.transport_host_dir) / 'bridge' / invocation_id / 'agent.pid'
        inner_pid_file = f'{self.target.transport_container_dir}/bridge/{invocation_id}/agent.pid'
        outer_home.mkdir(parents=True, exist_ok=True)
        outer_bin.mkdir(parents=True, exist_ok=True)
        os.chmod(outer_home, 0o777)
        os.chmod(outer_bin, 0o777)
        if any(ord(character) < 32 for character in working_dir):
            raise ExecutionTransportError('Working directory contains control characters')
        git_config_path = outer_home / '.gitconfig'
        git_config_path.write_text(
            f'[safe]\n\tdirectory = {json.dumps(working_dir, ensure_ascii=False)}\n',
            encoding='utf-8',
        )
        os.chmod(git_config_path, 0o644)

        launcher = command[0]
        if launcher.startswith(f'{self.target.transport_host_dir}/'):
            inner_launcher = self._translate_path(launcher)
        elif launcher.startswith('/nix/'):
            inner_launcher = launcher
        else:
            raise ExecutionTransportError('Agent executable must be installed under the transport directory')

        for argv, operation in (
            (
                self._compose('exec', '-T', *self._user_args(), self.target.service, 'test', '-d', working_dir),
                'Validate working directory',
            ),
            (
                self._compose('exec', '-T', *self._user_args(), self.target.service, 'test', '-x', inner_launcher),
                'Validate Agent executable',
            ),
            (
                self._compose(
                    'exec',
                    '-T',
                    *self._user_args(),
                    self.target.service,
                    'test',
                    '-x',
                    self.target.socket_bridge_container_path,
                ),
                'Validate socket bridge',
            ),
        ):
            await self._checked(argv, operation=operation)

        inner_path = await self._inner_path()
        launcher_link = outer_bin / Path(inner_launcher).name
        with contextlib.suppress(FileNotFoundError):
            launcher_link.unlink()
        launcher_link.symlink_to(inner_launcher)

        async with ProxyBridgeManager(self.target, invocation_id, proxy_url) as inner_proxy_url:
            with _rewritten_proxy_files(
                proxy_files or [],
                transport_host_dir=self.target.transport_host_dir,
                source_url=proxy_url,
                target_url=inner_proxy_url,
            ):
                env_args = [
                    '--env',
                    f'NEXUS_AGENT_PROXY_URL={inner_proxy_url}',
                    '--env',
                    f'HOME={inner_home}',
                    '--env',
                    f'PATH={inner_bin}:{inner_path}',
                ]
                if 'CODEX_HOME' in env:
                    env_args.extend(['--env', f'CODEX_HOME={self._translate_path(env["CODEX_HOME"])}'])
                if 'ANTHROPIC_BASE_URL' in env:
                    env_args.extend(['--env', f'ANTHROPIC_BASE_URL={inner_proxy_url}'])
                if 'OPENAI_BASE_URL' in env:
                    env_args.extend(['--env', f'OPENAI_BASE_URL={inner_proxy_url}'])
                for name in env:
                    if not _ENV_NAME_RE.fullmatch(name):
                        raise ValueError(f'Invalid execution environment variable name: {name!r}')
                    if name not in _TRANSPORT_MANAGED_ENV:
                        env_args.extend(['--env', name])

                translated_command = [
                    self._translate_path(value).replace(_PROXY_URL_PLACEHOLDER, inner_proxy_url) for value in command
                ]
                translated_command[0] = inner_launcher
                launch_script = 'printf "%s\\n" "$$" > "$1"; shift; exec "$@"'
                prepared = PreparedExecution(
                    command=self._compose(
                        'exec',
                        '-T',
                        '--workdir',
                        working_dir,
                        *self._user_args(),
                        *env_args,
                        self.target.service,
                        '/nix/runtime/nix-env/bin/bash',
                        '-c',
                        launch_script,
                        'nexus-compose-transport',
                        inner_pid_file,
                        *translated_command,
                    ),
                    cwd='/',
                    env={**self.target.compose_env, **env},
                )
                try:
                    yield prepared
                finally:
                    cleanup_task = asyncio.create_task(self._terminate_agent(outer_pid_file))
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        await cleanup_task
                        raise
                    finally:
                        outer_pid_file.unlink(missing_ok=True)

    @asynccontextmanager
    async def prepare(
        self,
        command: list[str],
        *,
        transport: AgentExecutionTransport,
        working_dir: str,
        invocation_id: str,
        env: dict[str, str],
        proxy_files: list[str | Path] | None = None,
    ) -> AsyncIterator[PreparedExecution]:
        await transport.initialize()
        if any(ord(character) < 32 for character in working_dir):
            raise ExecutionTransportError('Working directory contains control characters')
        git_config = transport.home_dir(invocation_id=invocation_id, default='/root') / '.gitconfig'
        content = await transport.read_text(git_config) or ''
        safe_directory = f'[safe]\n\tdirectory = {json.dumps(working_dir, ensure_ascii=False)}\n'
        if safe_directory not in content:
            await transport.write_text(
                git_config, content + ('\n' if content and not content.endswith('\n') else '') + safe_directory
            )
        proxy_url = env.get('NEXUS_AGENT_PROXY_URL') or env.get('ANTHROPIC_BASE_URL') or env.get('OPENAI_BASE_URL')
        if not proxy_url:
            raise ExecutionTransportError('Agent execution target requires a loopback proxy URL')
        outer_pid = Path(self.target.transport_host_dir) / 'bridge' / invocation_id / 'agent.pid'
        inner_pid = f'{self.target.transport_container_dir}/bridge/{invocation_id}/agent.pid'
        originals: dict[str | Path, str] = {}
        async with ProxyBridgeManager(self.target, invocation_id, proxy_url) as inner_url:
            try:
                for path in proxy_files or []:
                    content = await transport.read_text(path)
                    if content is None or proxy_url not in content:
                        raise ExecutionTransportError(f'Proxy URL not found in target file: {path}')
                    originals[path] = content
                    await transport.write_text(path, content.replace(proxy_url, inner_url))
                inner_env = transport.agent_env(env)
                for key in ('NEXUS_AGENT_PROXY_URL', 'ANTHROPIC_BASE_URL', 'OPENAI_BASE_URL'):
                    if key in inner_env:
                        inner_env[key] = inner_env[key].replace(proxy_url, inner_url)
                argv = [value.replace(_PROXY_URL_PLACEHOLDER, inner_url) for value in command]
                yield transport._prepare_command(
                    [
                        '/nix/runtime/nix-env/bin/bash',
                        '-c',
                        'printf "%s\n" "$$" > "$1"; shift; exec "$@"',
                        'nexus-agent',
                        inner_pid,
                        *argv,
                    ],
                    working_dir=working_dir,
                    env=inner_env,
                )
            finally:

                async def cleanup() -> None:
                    await self._terminate_agent(outer_pid)
                    outer_pid.unlink(missing_ok=True)
                    for path, content in originals.items():
                        await transport.write_text(path, content)

                task = asyncio.create_task(cleanup())
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await task
                    raise
