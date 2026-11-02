"""Shared POSIX Codex native bridge adapter."""

from __future__ import annotations

import asyncio
import base64
import logging
import posixpath
import uuid

from contextlib import suppress
from typing import Any

from nexus.runtime.bridge.adapters.base import BridgeLaunchResult
from nexus.runtime.client.base import ExecutorMixin
from nexus.runtime.executors.models.command import CommandRunStatus, RunCommandResponse
from nexus.utils.shell import quote_posix

from .packages import (
    DEFAULT_CODEX_INSTALL_DIR,
    NATIVE_EXECUTABLE_MARKER,
    installed_posix_codex_executable,
    native_package_ref,
    npm_install_timeout,
    npm_registry_arg,
    package_spec,
    parse_native_executable,
    registry_candidates,
)
from .shell import (
    build_codex_command,
    build_codex_config,
    build_codex_env,
    build_pre_start_command,
    shell_join_cmd,
    shell_join_env,
)
from .types import CodexInstallSpec, CodexRunRequest, CodexRunResult


_BRIDGE_RUN_ROOT = '/tmp/nexus-codex-run'
_POLL_INTERVAL = 5.0
_CONTROL_COMMAND_TIMEOUT = 60.0
_OUTPUT_MARKER = 'NEXUS_CODEX_OUTPUT_BASE64='


def _extract_output(stdout: str) -> tuple[str, str]:
    kept: list[str] = []
    output = ''
    for line in stdout.splitlines(keepends=True):
        if line.rstrip('\r\n').startswith(_OUTPUT_MARKER):
            encoded = line.rstrip('\r\n')[len(_OUTPUT_MARKER) :]
            try:
                output = base64.b64decode(encoded).decode()
            except Exception:
                kept.append(line)
        else:
            kept.append(line)
    return ''.join(kept), output


logger = logging.getLogger(__name__)

# Codex 0.153.4 does not install on node 12 (measured); 20 is the current LTS floor.
_MIN_NPM_NODE_MAJOR = 20


class _SessionPaths:
    def __init__(self, session_dir: str) -> None:
        self.dir = session_dir
        self.stdout_log = f'{session_dir}/stdout.log'
        self.stderr_log = f'{session_dir}/stderr.log'
        self.exit_code = f'{session_dir}/exit_code'
        self.pid = f'{session_dir}/pid'
        self.strict_cleanup = False


def _command_ok(result: RunCommandResponse) -> bool:
    return result.status == CommandRunStatus.Finished and int(result.return_code or 0) == 0


class BasePosixCodexNativeBridgeAdapter:
    """Codex install and detached-run operations for POSIX native bridge targets."""

    def __init__(self, runtime: ExecutorMixin) -> None:
        self.runtime = runtime
        self.execution_env: dict[str, str] = {}

    async def ensure_codex(self, spec: CodexInstallSpec) -> str:
        # Backport cbf07a9b5c: an explicit binary URL is authoritative.
        if not spec.binary_url:
            existing = await self._find_existing_codex()
            if existing and (spec.version == 'latest' or await self._matches_version(existing, spec.version)):
                return existing
        await self.prepare_install(spec)
        if spec.binary_url:
            return await self._install_from_binary_url(spec)
        if await self._npm_available():
            return await self._install_via_npm(spec)
        # No node on this image. Codex publishes a self-contained binary in a
        # per-platform build, and an npm registry is just an HTTP tarball server
        # -- so fetch and unpack it ourselves. npm was only ever the downloader
        # here, which is why a node-less image is not a dead end. The 167 Gym
        # Linux application images are not guaranteed to carry node.
        return await self._install_native_package(spec)

    async def prepare_install(self, spec: CodexInstallSpec) -> None:
        """Target-specific environment preparation, without another installer."""

    async def _find_existing_codex(self) -> str | None:
        executable = installed_posix_codex_executable()
        result = await self.runtime.run_command(command=f'test -x {quote_posix(executable)}', timeout=30)
        if _command_ok(result):
            return executable
        result = await self.runtime.run_command(command='command -v codex', timeout=30)
        lines = (result.stdout or '').strip().splitlines()
        return lines[0] if _command_ok(result) and lines else None

    async def _matches_version(self, executable: str, version: str) -> bool:
        result = await self.runtime.run_command(command=f'{quote_posix(executable)} --version', timeout=30)
        return _command_ok(result) and (result.stdout or '').strip() == f'codex-cli {version}'

    def install_registries(self, primary: str) -> list[str]:
        return [primary]

    async def execute_install(self, command: str, timeout: float = 1800):
        kwargs: dict[str, Any] = {'extra_env': self.execution_env} if self.execution_env else {}
        return await self.runtime.run_command(command=command, timeout=timeout, **kwargs)

    async def _npm_available(self) -> bool:
        """True only when npm exists AND its node is new enough to run Codex.

        ``command -v npm`` alone is not enough: an image can ship npm on a node so old
        that ``npm install @openai/codex`` fails, and because that check gates the choice
        between _install_via_npm and _install_native_package, a false positive sends us
        down the npm path and the run dies with no usable diagnostic.

        MEASURED 2026-09-17 on the OSWorld-V2 AGS pool (sdt-ruk8welf): the image carries
        npm 8.5.1 on /usr/bin/node v12.22.9 (2019, EOL 2022). _npm_available() returned
        True, _install_via_npm ran, and codex 0.153.4 could not install -- surfacing on the
        host only as exit_status='tool_error' rc=-1 iters=0 with zero model calls, since the
        sandbox-side stderr is not propagated. The npm-free path works on that exact image:
        the linux-x64 build unpacks in ~30s and reports ``codex-cli 0.153.4`` (its Linux
        builds are statically linked against musl, so the system node is irrelevant).

        Node 20 is the floor: v12 is measured-broken and Codex's own published builds target
        current LTS lines. Anything below the floor -- or an unparseable version -- reports
        False so ensure_codex falls through to the self-contained binary, which is always a
        safe answer (it is what node-less images already use).
        """
        result = await self.runtime.run_command(
            command='command -v npm >/dev/null 2>&1 && node --version 2>/dev/null || true',
            timeout=30.0,
        )
        if not _command_ok(result):
            return False
        raw = (result.stdout or '').strip()
        if not raw.startswith('v'):
            # npm missing, or node absent/unreadable -> use the npm-free installer.
            return False
        try:
            major = int(raw[1:].split('.', 1)[0])
        except ValueError:
            logger.warning('codex install: unparseable node version %r; using the npm-free installer', raw)
            return False
        if major < _MIN_NPM_NODE_MAJOR:
            logger.info(
                'codex install: node %s is below the Codex floor (v%d); '
                'using the npm-free self-contained binary instead',
                raw,
                _MIN_NPM_NODE_MAJOR,
            )
            return False
        return True

    async def _install_from_binary_url(self, spec: CodexInstallSpec) -> str:
        if not spec.binary_url:
            raise ValueError('Codex binary installation requires binary_url')
        install_dir = DEFAULT_CODEX_INSTALL_DIR
        executable = f'{install_dir}/codex'
        temporary = f'{executable}.url.tmp'
        download = (
            f'if command -v curl >/dev/null 2>&1; then '
            f'curl --fail --location --retry 3 --connect-timeout 30 '
            f'--output {quote_posix(temporary)} {quote_posix(spec.binary_url)}; '
            f'elif command -v wget >/dev/null 2>&1; then '
            f'wget --tries=4 --timeout=30 --output-document={quote_posix(temporary)} '
            f'{quote_posix(spec.binary_url)}; '
            "else echo 'curl or wget is required to download Codex CLI' >&2; exit 127; fi"
        )
        checksum = ''
        if spec.binary_checksum:
            expected = quote_posix(spec.binary_checksum)
            checksum = (
                f' && expected_checksum={expected}'
                ' && if command -v sha256sum >/dev/null 2>&1; then '
                f"actual_checksum=$(sha256sum {quote_posix(temporary)} | awk '{{print $1}}'); "
                'elif command -v shasum >/dev/null 2>&1; then '
                f"actual_checksum=$(shasum -a 256 {quote_posix(temporary)} | awk '{{print $1}}'); "
                "else echo 'sha256sum or shasum is required to verify Codex CLI' >&2; exit 127; fi"
                ' && if [ "$actual_checksum" != "$expected_checksum" ]; then '
                "printf 'Checksum verification failed: expected %s, got %s\\n' "
                '"$expected_checksum" "$actual_checksum" >&2; '
                f'rm -f {quote_posix(temporary)}; exit 1; fi'
            )
        # A .tar.gz holds the npm-layout tree, not a bare executable, so it is
        # unpacked and entered through node_modules/.bin/codex; anything else is
        # a standalone binary that is chmod'd into place.
        # The query string is stripped first: a signed URL keeps its extension
        # ahead of ``?``, so matching the raw URL alone would misclassify it.
        is_tarball = spec.binary_url.split('?')[0].endswith('.tar.gz')
        npm_executable = installed_posix_codex_executable()
        if is_tarball:
            command = (
                f'rm -rf {quote_posix(install_dir)} && '
                f'mkdir -p {quote_posix(install_dir)} && '
                f'{download}{checksum} && '
                f'tar xzf {quote_posix(temporary)} -C {quote_posix(install_dir)} && '
                f'rm -f {quote_posix(temporary)} && '
                f'{quote_posix(npm_executable)} --version'
            )
        else:
            command = (
                f'rm -rf {quote_posix(install_dir)} && '
                f'mkdir -p {quote_posix(install_dir)} && '
                f'{download}{checksum} && '
                f'chmod +x {quote_posix(temporary)} && '
                f'mv {quote_posix(temporary)} {quote_posix(executable)} && '
                f'{quote_posix(executable)} --version'
            )
        result = await self.execute_install(command, timeout=1800.0)
        if result.return_code != 0:
            detail = (result.stderr or result.stdout or str(result.status))[:2000]
            raise RuntimeError(f'Failed to install Codex CLI from binary URL: {detail}')
        return npm_executable if is_tarball else executable

    async def _install_via_npm(self, spec: CodexInstallSpec) -> str:
        install_dir = DEFAULT_CODEX_INSTALL_DIR
        executable = installed_posix_codex_executable()
        registry = npm_registry_arg(spec.npm_registry)
        install = ' '.join([
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
            '--include=optional',
            '--fetch-retries=2',
            '--fetch-retry-mintimeout=1000',
            '--fetch-retry-maxtimeout=10000',
            '--fetch-timeout=60000',
            '--registry',
            '"$registry"',
            quote_posix(package_spec(spec.version)),
            '&&',
            'test',
            '-x',
            quote_posix(executable),
            '&&',
            quote_posix(executable),
            '--version',
        ])
        # Same pinned package at either mirror; bounded installation recovery is
        # independent of model-call and whole-task retry budgets.
        registries = self.install_registries(registry)
        command = (
            'for registry in ' + ' '.join(quote_posix(value) for value in registries) + '; do '
            f'if {install}; then exit 0; fi; '
            'echo "Codex package install failed; trying next configured mirror" >&2; '
            'done; exit 1'
        )
        result = await self.execute_install(command, timeout=1800)
        if result.status != CommandRunStatus.Finished or result.return_code != 0:
            detail = (result.stderr or result.stdout or str(result.status))[:2000]
            raise RuntimeError(f'Failed to install Codex CLI via npm: {detail}')
        return executable

    async def _run_install_command(self, command: str, *, timeout: float) -> RunCommandResponse:
        """Keep long npm downloads outside the AGS control request lifetime."""
        paths = _SessionPaths(f'/tmp/nexus-codex-install-{uuid.uuid4().hex}')
        paths.strict_cleanup = True
        inner = (
            f'{shell_join_env(self.execution_env)} bash -c {quote_posix(command)} '
            f'> {quote_posix(paths.stdout_log)} 2> {quote_posix(paths.stderr_log)}; '
            f'echo "$?" > {quote_posix(paths.exit_code)}'
        )
        launch = (
            f'mkdir -p {quote_posix(paths.dir)} && {{ '
            f'nohup setsid bash -c {quote_posix(inner)} </dev/null >/dev/null 2>&1 & '
            f'echo $! > {quote_posix(paths.pid)}; echo LAUNCHED; }}'
        )
        may_be_running = True
        try:
            result = await self.runtime.run_command(command=launch, timeout=_CONTROL_COMMAND_TIMEOUT)
            if result.return_code != 0 or 'LAUNCHED' not in (result.stdout or ''):
                raise RuntimeError('Failed to launch detached Codex installer')
            code, timed_out, vanished = await self._poll_until_done(paths, timeout)
            may_be_running = timed_out or vanished
            if may_be_running:
                await self._terminate_session(paths)
                may_be_running = False
            stdout = await self._read_file(paths.stdout_log)
            stderr = await self._read_file(paths.stderr_log)
            if timed_out or vanished:
                stderr += '\nCodex installation timed out' if timed_out else '\nCodex installer exited without status'
            return RunCommandResponse(
                status=(
                    CommandRunStatus.TimeLimitExceeded
                    if timed_out
                    else CommandRunStatus.Error
                    if vanished
                    else CommandRunStatus.Finished
                ),
                return_code=code,
                stdout=stdout,
                stderr=stderr,
            )
        finally:
            await self._run_cleanup(paths, terminate=may_be_running)

    async def _install_native_package(self, spec: CodexInstallSpec) -> str:
        """Install Codex from its per-platform npm build without npm.

        The platform build holds a standalone ``codex``, so a plain HTTP GET plus
        ``tar`` is a complete install on an image that has no node. Platform
        detection and the registry choice both happen *in* the sandbox: only the
        sandbox knows which registry it can reach.
        """
        install_dir = DEFAULT_CODEX_INSTALL_DIR
        tarball = f'{install_dir}/package.tgz'
        vendor_root = f'{install_dir}/package/vendor'
        registries = ' '.join(quote_posix(registry) for registry in registry_candidates(spec.npm_registry))
        # Empty means "resolve the platform dist-tag in the sandbox".
        pinned = quote_posix(native_package_ref(spec.version))
        command = '; '.join([
            'set -e',
            'arch=$(uname -m)',
            'case "$arch" in x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=arm64 ;; '
            '*) echo "unsupported architecture $arch" >&2; exit 2 ;; esac',
            # Codex links its Linux builds against musl statically and ships a
            # single linux-<arch> build, so unlike Claude Code there is no libc
            # variant to detect here.
            'case "$(uname -s)" in Linux) platform="linux-${arch}" ;; '
            'Darwin) platform="darwin-${arch}" ;; '
            '*) echo "unsupported platform $(uname -s)" >&2; exit 2 ;; esac',
            f'pinned={pinned}',
            # Codex aliases each platform build to a *version* of the same
            # package, so the platform is a version suffix. Every platform also
            # carries its own dist-tag, which is how an unpinned install
            # resolves without pulling the multi-megabyte packument.
            'if [ -n "$pinned" ]; then ref="${pinned}-${platform}"; else ref="$platform"; fi',
            'if command -v curl >/dev/null 2>&1; then '
            'fetch="curl -fsSL --retry 3 --connect-timeout 30"; save="curl -fsSL --retry 3 -o"; '
            'elif command -v wget >/dev/null 2>&1; then '
            'fetch="wget -q --tries=4 --timeout=30 -O-"; save="wget -q --tries=4 -O"; '
            "else echo 'curl or wget is required to download Codex CLI' >&2; exit 127; fi",
            f'rm -rf {quote_posix(install_dir)}',
            f'mkdir -p {quote_posix(install_dir)}',
            'downloaded=0',
            # The version manifest names the exact build, and the tarball path
            # follows from it -- so the packument is never downloaded.
            f'for registry in {registries}; do '
            'version=$($fetch "${registry%/}/@openai%2fcodex/${ref}" 2>/dev/null | tr "," "\\n" '
            "| grep -m1 '\"version\"' | sed -e 's/.*: *\"//' -e 's/\".*//') || true; "
            '[ -n "$version" ] || continue; '
            f'if $save {quote_posix(tarball)} '
            '"${registry%/}/@openai/codex/-/codex-${version}.tgz"; then '
            'echo "codex $version from $registry"; downloaded=1; break; fi; '
            'done',
            '[ "$downloaded" = 1 ] || '
            '{ echo "unable to download @openai/codex ($ref) from any registry" >&2; exit 3; }',
            f'tar -xzf {quote_posix(tarball)} -C {quote_posix(install_dir)}',
            f'rm -f {quote_posix(tarball)}',
            # The binary sits under a Rust target triple only the sandbox can
            # name, and the directory below it moved between releases (0.114.0:
            # vendor/<triple>/codex/, 0.146.1+: vendor/<triple>/bin/), so locate
            # it instead of encoding either layout here.
            f'executable=$(find {quote_posix(vendor_root)} -type f -name codex 2>/dev/null | head -n 1) || true',
            'if [ -z "$executable" ]; then for candidate in '
            f'{quote_posix(vendor_root)}/*/bin/codex {quote_posix(vendor_root)}/*/codex/codex; do '
            'if [ -f "$candidate" ]; then executable="$candidate"; break; fi; done; fi',
            f'[ -n "$executable" ] || {{ echo "codex binary not found under {vendor_root}" >&2; exit 4; }}',
            'chmod +x "$executable"',
            f'echo "{NATIVE_EXECUTABLE_MARKER}${{executable}}"',
            '"$executable" --version',
        ])
        result = await self.execute_install(command, timeout=npm_install_timeout())
        if result.return_code != 0:
            raise RuntimeError(
                'Failed to install Codex CLI from its native package: '
                f'{(result.stderr or result.stdout or str(result.status))[:2000]}'
            )
        return parse_native_executable(result.stdout or '')

    async def run_codex(self, request: CodexRunRequest) -> CodexRunResult:
        paths = _SessionPaths(f'{_BRIDGE_RUN_ROOT}-{uuid.uuid4().hex}')
        paths.strict_cleanup = request.strict_cleanup
        may_be_running = False
        try:
            may_be_running = True
            launch_result = await self.runtime.run_command(
                command=self._build_launch_command(request, paths), timeout=_CONTROL_COMMAND_TIMEOUT
            )
            if not _command_ok(launch_result) or 'LAUNCHED' not in (launch_result.stdout or ''):
                detail = (launch_result.stderr or launch_result.stdout or str(launch_result.status))[:2000]
                return CodexRunResult(
                    exit_code=-1,
                    stdout='',
                    stderr=f'Failed to launch Codex CLI: {detail}',
                    output='',
                    failure_kind='infrastructure',
                    failure_stage='launch',
                )

            exit_code, timed_out, vanished = await self._poll_until_done(paths, request.timeout)
            may_be_running = timed_out or vanished
            if may_be_running and request.strict_cleanup:
                await self._terminate_session(paths)
                may_be_running = False
            stdout = await self._read_file(paths.stdout_log)
            stderr = await self._read_file(paths.stderr_log)
            stdout, output = _extract_output(stdout)

            if timed_out:
                stderr = f'{stderr}\n[Codex CLI timed out after {request.timeout:g}s]'.strip()
                return CodexRunResult(exit_code=-1, stdout=stdout, stderr=stderr, output=output, timed_out=True)
            if vanished:
                stderr = f'{stderr}\n[Codex CLI exited without writing an exit code]'.strip()
                return CodexRunResult(
                    exit_code=-1,
                    stdout=stdout,
                    stderr=stderr,
                    output=output,
                    failure_kind='infrastructure',
                    failure_stage='process',
                )
            return CodexRunResult(
                exit_code=exit_code if exit_code is not None else -1,
                stdout=stdout,
                stderr=stderr,
                output=output,
            )
        finally:
            await self._run_cleanup(paths, terminate=may_be_running)

    async def _poll_until_done(self, paths: _SessionPaths, timeout: float) -> tuple[int | None, bool, bool]:
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
                return None, False, True
            if loop.time() >= deadline:
                return None, True, False
            await asyncio.sleep(_POLL_INTERVAL)

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
            interpreter = str(script.get('interpreter', 'bash'))
            code = str(script.get('code', ''))
            timeout = float(script.get('timeout', 300.0))
            cwd = str(script.get('cwd') or working_dir)
            fail_on_error = bool(script.get('fail_on_error', True))
            env = dict(base_env)
            env.update(script.get('env') or {})
            command = build_pre_start_command(interpreter, code, cwd=cwd, env=env)
            result = await self.runtime.run_command(command=command, timeout=timeout)
            if not _command_ok(result) and fail_on_error:
                detail = (result.stderr or result.stdout or str(result.status))[:2000]
                raise RuntimeError(
                    f'pre-start ({interpreter}) failed status={result.status} rc={result.return_code}: {detail}'
                )

    def _build_launch_command(self, request: Any, paths: Any) -> str:
        codex_home = f'{paths.dir}/codex-home'
        output_file = f'{codex_home}/last_message.txt'
        config = base64.b64encode(build_codex_config(request, codex_home).encode()).decode()
        # 模型目录:与 config.toml 同机制(base64 传入沙盒再解码落盘),路径与 shell.py
        # 里写进 config.toml 的那行保持一致(codex_home/model_catalog.json)。
        catalog = (
            base64.b64encode(request.model_catalog_json.encode()).decode()
            if getattr(request, 'model_catalog_json', None)
            else ''
        )
        env = build_codex_env(codex_home=codex_home, extra_env={**(request.extra_env or {}), **self.execution_env})
        history_setup = ''
        if request.record_shell_history:
            from .history import BASH_HISTORY_HOOK

            history_path = f'{paths.dir}/agent-bash-env'
            encoded = base64.b64encode(BASH_HISTORY_HOOK.encode()).decode()
            history_setup = f'printf %s {quote_posix(encoded)} | base64 -d > {quote_posix(history_path)}; '
            env['BASH_ENV'] = history_path
        command = build_codex_command(request, output_file=output_file)
        codex_bin_dir = posixpath.dirname(request.codex_executable)
        if 'PATH' in env:
            env['PATH'] = f'{codex_bin_dir}:{env["PATH"]}'
        inner = (
            f'export PATH={quote_posix(codex_bin_dir)}:"$PATH"; '
            f'{shell_join_env(env)} {shell_join_cmd(command)} '
            f'> {quote_posix(paths.stdout_log)} 2> {quote_posix(paths.stderr_log)}; '
            'code=$?; '
            f'if [ -f {quote_posix(output_file)} ]; then '
            f'printf "\\n{_OUTPUT_MARKER}" >> {quote_posix(paths.stdout_log)}; '
            f'base64 < {quote_posix(output_file)} | tr -d "\\r\\n" >> {quote_posix(paths.stdout_log)}; '
            f'printf "\\n" >> {quote_posix(paths.stdout_log)}; fi; '
            f'echo "$code" > {quote_posix(paths.exit_code)}'
        )
        working_dir = quote_posix(request.working_dir)
        return (
            f'mkdir -p {working_dir} {quote_posix(codex_home)}; '
            f'printf %s {quote_posix(config)} | base64 -d > {quote_posix(codex_home + "/config.toml")}; '
            + history_setup
            + (
                f'printf %s {quote_posix(catalog)} | base64 -d > {quote_posix(codex_home + "/model_catalog.json")}; '
                if catalog
                else ''
            )
            + f'cd {working_dir} && '
            + self._launch_detached_shell(inner, paths.pid, strict_cleanup=request.strict_cleanup)
        )

    def _launch_detached_shell(self, command: str, pid_file: str, *, strict_cleanup: bool) -> str:
        return (
            f'{{ nohup {"setsid " if strict_cleanup else ""}bash -c {quote_posix(command)} '
            f'</dev/null >/dev/null 2>&1 & echo $! > {quote_posix(pid_file)}; echo LAUNCHED; }}'
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
        if paths.strict_cleanup:
            return (
                f'pid=$(cat {pid} 2>/dev/null); '
                'case "$pid" in ""|*[!0-9]*) exit 1;; esac; [ "$pid" -gt 1 ] || exit 1; '
                'kill -TERM -- "-$pid" 2>/dev/null || true; sleep 1; '
                'kill -KILL -- "-$pid" 2>/dev/null || true; '
                'for i in 1 2 3 4 5 6 7 8 9 10; do '
                'if ! ps -eo pgid=,stat= | awk -v p="$pid" '
                "' $1 == p && $2 !~ /^Z/ {found=1} END {exit !found}'; then exit 0; fi; "
                'sleep 0.1; done; exit 1'
            )
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

    async def _terminate_session(self, paths: _SessionPaths) -> None:
        result = await self.runtime.run_command(
            command=self._build_kill_command(paths), timeout=_CONTROL_COMMAND_TIMEOUT
        )
        if paths.strict_cleanup and not _command_ok(result):
            raise RuntimeError('Codex process-group termination could not be confirmed')

    async def _cleanup_session(self, paths: _SessionPaths, *, terminate: bool) -> None:
        if terminate:
            if paths.strict_cleanup:
                await self._terminate_session(paths)
            else:
                with suppress(Exception):
                    await self._terminate_session(paths)
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


__all__ = ['BasePosixCodexNativeBridgeAdapter']
