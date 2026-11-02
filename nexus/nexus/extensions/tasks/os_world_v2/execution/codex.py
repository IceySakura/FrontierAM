"""Task-scoped Codex stdin transport, process probes and failure evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import posixpath
import time
import uuid

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from nexus.extensions.agents.codex.native_bridge.adapters.osworld import OSWorldCodexNativeBridgeAdapter
from nexus.extensions.agents.codex.native_bridge.adapters.posix import _SessionPaths, _extract_output
from nexus.extensions.agents.codex.native_bridge.adapters.shell import (
    build_codex_command,
    build_codex_config,
    build_codex_env,
    shell_join_cmd,
    shell_join_env,
)
from nexus.extensions.agents.codex.native_bridge.adapters.types import CodexRunRequest, CodexRunResult
from nexus.extensions.tasks.os_world_v2.execution.runtime import OSWorldTaskRuntime
from nexus.utils.shell import quote_posix


logger = logging.getLogger(__name__)
_TAIL_BYTES = 65536
_PROBE_ATTEMPTS = 3

# Parse /proc/stat after the comm field, which can contain spaces or parentheses.
_PROBE_SOURCE = r"""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def identity(role):
    raw = (root / (role + ".pid")).read_text().strip()
    if not raw.isdecimal() or int(raw) <= 1:
        return {"error": "invalid_pid"}
    pid = int(raw)
    saved = (root / (role + ".start")).read_text().strip()
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return {"pid": pid, "saved_start": saved, "alive": False}
    return {"pid": pid, "saved_start": saved, "start": stat[19],
            "ppid": int(stat[1]), "pgid": int(stat[2]), "state": stat[0],
            "alive": stat[0] != "Z", "matches": saved == stat[19]}
try:
    evidence = {}
    for role in ("wrapper", "codex"):
        try:
            evidence[role] = identity(role)
        except FileNotFoundError:
            evidence[role] = {"error": "missing_metadata"}
    exit_file = root / "exit_code"
    if exit_file.exists():
        value = exit_file.read_text().strip()
        if not value or not value.lstrip("-").isdigit():
            state = "EXIT_INVALID"
        else:
            state = "DONE"
            evidence["exit_code"] = int(value)
    else:
        parent = evidence["wrapper"]
        if parent.get("error"):
            state = "PID_METADATA_INVALID"
        elif parent.get("matches") is False:
            state = "PID_REUSED"
        elif parent.get("alive"):
            state = "RUNNING"
        else:
            state = "GONE"
    print(json.dumps({"state": state, **evidence}))
except Exception as exc:
    print(json.dumps({"state": "PROBE_FAILED", "error_type": type(exc).__name__}))
"""


class _ExecutionError(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class TaskCodexAdapter(OSWorldCodexNativeBridgeAdapter):
    """Selected only by the OSWorld profile, not by generic runtime detection."""

    def __init__(self, runtime, *, diagnostic_dir: Path | None = None):
        super().__init__(runtime, benchmark_policy=True)
        self.diagnostic_dir = diagnostic_dir
        self.diagnostics: dict[str, Any] = {}
        self.poll_interval = 5.0

    async def _command(self, command: str, timeout: float = 10):
        return await asyncio.wait_for(
            self.runtime.run_command(command=command, timeout=timeout),
            timeout=timeout + 1,
        )

    @staticmethod
    def _ok(result) -> bool:
        return str(getattr(result.status, 'value', result.status)) == 'Finished' and result.return_code == 0

    async def _stage(self, path: str, content: bytes):
        runtime = cast(OSWorldTaskRuntime, self.runtime)
        response = await asyncio.wait_for(
            runtime.setup_upload_bytes(path, content, filename=posixpath.basename(path)),
            timeout=60,
        )
        if response.status_code != 200:
            raise _ExecutionError('staging', f'Upload failed: HTTP {response.status_code}')
        result = await self._command(f'chmod 600 {quote_posix(path)} && sha256sum {quote_posix(path)}')
        if not self._ok(result) or (result.stdout or '').split()[:1] != [hashlib.sha256(content).hexdigest()]:
            raise _ExecutionError('staging', 'Uploaded file checksum verification failed')

    async def _prepare(self, request, paths):
        home = f'{paths.dir}/codex-home'
        response = await self._command(f'umask 077; mkdir -p {quote_posix(request.working_dir)} {quote_posix(home)}')
        if not self._ok(response):
            raise _ExecutionError('staging', 'Unable to create Codex session directory')
        prompt = request.task
        if request.developer_instructions:
            prompt = f'{request.developer_instructions.rstrip()}\n\n{prompt}'
        await self._stage(f'{paths.dir}/prompt.txt', prompt.encode())
        await self._stage(f'{home}/config.toml', build_codex_config(request, home).encode())
        if request.model_catalog_json:
            await self._stage(f'{home}/model_catalog.json', request.model_catalog_json.encode())
        if request.record_shell_history:
            from nexus.extensions.agents.codex.native_bridge.adapters.history import BASH_HISTORY_HOOK

            await self._stage(f'{paths.dir}/agent-bash-env', BASH_HISTORY_HOOK.encode())

    def _safe_launch(self, request, paths):
        home = f'{paths.dir}/codex-home'
        output_file = f'{home}/last_message.txt'
        command = build_codex_command(replace(request, task='-', developer_instructions=None), output_file=output_file)
        env = build_codex_env(codex_home=home, extra_env={**(request.extra_env or {}), **self.execution_env})
        if request.record_shell_history:
            env['BASH_ENV'] = f'{paths.dir}/agent-bash-env'
        # Record identities for both wrapper and CLI before waiting. No prompt,
        # config contents, or generated tool command is placed in persistent argv.
        record = (
            'import pathlib,sys; p=int(sys.argv[1]); '
            's=pathlib.Path(f"/proc/{p}/stat").read_text().rsplit(")",1)[1].split()[19]; '
            'pathlib.Path(sys.argv[2]).write_text(s)'
        )
        wrapper = quote_posix(f'{paths.dir}/wrapper')
        child = quote_posix(f'{paths.dir}/codex')
        inner = (
            f'echo "$BASHPID" > {wrapper}.pid; '
            f'python3 -c {quote_posix(record)} "$BASHPID" {wrapper}.start || exit 125; '
            f'export PATH={quote_posix(posixpath.dirname(request.codex_executable))}:"$PATH"; '
            f'{shell_join_env(env)} {shell_join_cmd(command)} < {quote_posix(paths.dir + "/prompt.txt")} '
            f'> {quote_posix(paths.stdout_log)} 2> {quote_posix(paths.stderr_log)} & child=$!; '
            f'echo "$child" > {child}.pid; '
            f'python3 -c {quote_posix(record)} "$child" {child}.start; '
            'wait "$child"; code=$?; '
            f'printf "%s\\n" "$code" > {quote_posix(paths.exit_code + ".tmp")}; '
            f'mv {quote_posix(paths.exit_code + ".tmp")} {quote_posix(paths.exit_code)}'
        )
        return (
            f'cd {quote_posix(request.working_dir)} && '
            f'{{ nohup {"setsid " if request.strict_cleanup else ""}bash -c {quote_posix(inner)} '
            f'</dev/null >/dev/null 2>&1 & echo $! > {quote_posix(paths.pid)}; echo LAUNCHED; }}'
        )

    async def _probe(self, paths, timeout=10):
        command = f'python3 -c {quote_posix(_PROBE_SOURCE)} {quote_posix(paths.dir)}'
        record: dict[str, Any] = {'time': time.time()}
        try:
            response = await self._command(command, timeout=timeout)
            record.update(
                return_code=response.return_code,
                status=str(response.status),
                stdout=(response.stdout or '')[-_TAIL_BYTES:],
                stderr=(response.stderr or '')[-4096:],
            )
            if not self._ok(response):
                return {'state': 'PROBE_FAILED'}
            try:
                value = json.loads(response.stdout or '')
                if not isinstance(value, dict) or value.get('state') not in {
                    'RUNNING',
                    'DONE',
                    'GONE',
                    'PID_METADATA_INVALID',
                    'PID_REUSED',
                    'EXIT_INVALID',
                    'PROBE_FAILED',
                }:
                    raise ValueError('invalid probe')
                if value['state'] == 'DONE' and type(value.get('exit_code')) is not int:
                    raise ValueError('invalid exit code')
                record['parsed'] = value
                return value
            except (ValueError, TypeError):
                return {'state': 'PROBE_FAILED'}
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            return {'state': 'PROBE_FAILED'}
        finally:
            history = self.diagnostics.setdefault('probes', [])
            history.append(record)
            del history[:-8]

    def _build_kill_command(self, paths):
        # setsid can fork: the launcher's $! is not necessarily the session
        # leader. Terminate the identity verified by our /proc probe.
        verified = _SessionPaths(paths.dir)
        verified.strict_cleanup = True
        verified.pid = f'{paths.dir}/wrapper.pid'
        return super()._build_kill_command(verified)

    async def _poll(self, paths, timeout):
        deadline = asyncio.get_running_loop().time() + timeout
        previous, count, failed_probes = None, 0, 0
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise _ExecutionError('timeout', 'Codex execution time budget exhausted')
            probe = await self._probe(paths, timeout=min(10, remaining))
            state = probe['state']
            if state == 'DONE':
                return probe['exit_code']
            if state == 'RUNNING':
                previous, count, failed_probes = None, 0, 0
            else:
                failed_probes += 1
                count = count + 1 if state == previous else 1
                previous = state
                if count >= _PROBE_ATTEMPTS or failed_probes >= _PROBE_ATTEMPTS:
                    confirmed_gone = state == 'GONE' and count >= _PROBE_ATTEMPTS
                    stage = 'process_missing' if confirmed_gone else 'probe'
                    raise _ExecutionError(
                        stage, f'Codex state after {failed_probes} unsuccessful read-only probes: {state}'
                    )
            await asyncio.sleep(min(self.poll_interval, max(0, deadline - asyncio.get_running_loop().time())))

    async def _capture(self, paths):
        self.diagnostics['captured_before_cleanup_at'] = time.time()
        reads = {
            'stdout_tail': paths.stdout_log,
            'stderr_tail': paths.stderr_log,
            'wrapper_pid': paths.dir + '/wrapper.pid',
            'wrapper_start': paths.dir + '/wrapper.start',
            'codex_pid': paths.dir + '/codex.pid',
            'codex_start': paths.dir + '/codex.start',
            'exit_code_file': paths.exit_code,
        }
        for key, path in reads.items():
            try:
                response = await self._command(f'tail -c {_TAIL_BYTES} -- {quote_posix(path)}', timeout=2)
                if self._ok(response):
                    self.diagnostics[key] = (response.stdout or '')[-_TAIL_BYTES:]
                else:
                    self.diagnostics.setdefault('read_errors', {})[key] = {
                        'return_code': response.return_code,
                        'status': str(response.status),
                        'stderr': (response.stderr or '')[-512:],
                    }
            except Exception as exc:
                self.diagnostics.setdefault('read_errors', {})[key] = {'error_type': type(exc).__name__}
        try:
            response = await self._command('ps -eo pid,ppid,pgid,sid,lstart,stat,comm', timeout=2)
            if self._ok(response):
                self.diagnostics['process_tree'] = (response.stdout or '')[-_TAIL_BYTES:]
        except Exception as exc:
            self.diagnostics['process_tree_error'] = type(exc).__name__
        events = []
        for line in self.diagnostics.get('stdout_tail', '').splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        self.diagnostics['last_events'] = events[-5:]
        self._save()

    def _save(self):
        if self.diagnostic_dir:
            try:
                self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
                (self.diagnostic_dir / 'codex_diagnostics.json').write_text(
                    json.dumps(self.diagnostics, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                )
            except OSError:
                logger.warning('Unable to persist Codex diagnostics', exc_info=True)

    async def run_codex(self, request: CodexRunRequest) -> CodexRunResult:
        paths = _SessionPaths(f'/tmp/nexus-codex-run-{uuid.uuid4().hex}')
        paths.strict_cleanup = request.strict_cleanup
        self.diagnostics = {
            'session_dir': paths.dir,
            'timed_out': False,
            'failure_kind': None,
            'failure_stage': None,
        }
        result = None
        launched = False
        try:
            await self._prepare(request, paths)
            launched = True  # A failed launch response may still have started a process.
            response = await self._command(self._safe_launch(request, paths), timeout=30)
            if not self._ok(response) or 'LAUNCHED' not in (response.stdout or ''):
                raise _ExecutionError('launch', 'Codex launch was not acknowledged')
            code = await self._poll(paths, request.timeout)
            self.diagnostics['exit_code'] = code
            stdout, stderr, output = [], [], []
            for path, destination in (
                (paths.stdout_log, stdout),
                (paths.stderr_log, stderr),
                (paths.dir + '/codex-home/last_message.txt', output),
            ):
                response = await self._command(f'cat -- {quote_posix(path)}', timeout=10)
                if not self._ok(response) and destination is not output:
                    raise _ExecutionError('read_output', 'Unable to read completed Codex output')
                destination.append(response.stdout or '')
            clean, embedded = _extract_output(stdout[0])
            result = CodexRunResult(
                exit_code=code,
                stdout=clean,
                stderr=stderr[0],
                output=output[0] or embedded,
                failure_kind='agent' if code != 0 else None,
                failure_stage='process' if code != 0 else None,
            )
        except asyncio.CancelledError:
            self.diagnostics.update(failure_kind='cancelled', failure_stage='execution')
            raise
        except Exception as exc:
            stage = exc.stage if isinstance(exc, _ExecutionError) else 'execution'
            result = CodexRunResult(
                exit_code=-1,
                stdout='',
                stderr=f'{type(exc).__name__}: {exc}',
                output='',
                timed_out=stage == 'timeout',
                failure_kind='infrastructure',
                failure_stage=stage,
            )
        finally:
            if result is not None:
                self.diagnostics.update(
                    timed_out=result.timed_out,
                    failure_kind=result.failure_kind,
                    failure_stage=result.failure_stage,
                    return_code=result.exit_code,
                )

            async def finalize():
                try:
                    try:
                        await self._capture(paths)
                    except Exception as exc:
                        self.diagnostics['capture_error'] = type(exc).__name__
                        self._save()
                    from nexus.extensions.tasks.os_world_v2.data.artifacts import snapshot_sandbox_artifacts

                    await snapshot_sandbox_artifacts(
                        self.runtime,
                        {
                            'stdout.log': paths.stdout_log,
                            'stderr.log': paths.stderr_log,
                            'exit_code': paths.exit_code,
                            'wrapper.pid': paths.dir + '/wrapper.pid',
                            'wrapper.start': paths.dir + '/wrapper.start',
                            'codex.pid': paths.dir + '/codex.pid',
                            'codex.start': paths.dir + '/codex.start',
                            'last_message.txt': paths.dir + '/codex-home/last_message.txt',
                        },
                        label='codex',
                    )
                    safe_to_remove = not launched or (result is not None and result.exit_code >= 0)
                    if launched and (result is None or result.exit_code < 0):
                        probe = await self._probe(paths)
                        # Never kill a recycled PID or an unverified process group.
                        parent = probe.get('wrapper', {})
                        if parent.get('matches') and parent.get('pgid') == parent.get('pid'):
                            await asyncio.wait_for(self._terminate_session(paths), timeout=15)
                            self.diagnostics['termination_confirmed'] = True
                            safe_to_remove = True
                        else:
                            self.diagnostics['termination_unconfirmed'] = True
                    if safe_to_remove:
                        response = await self._command(f'rm -rf {quote_posix(paths.dir)}', timeout=3)
                        if not self._ok(response):
                            raise RuntimeError('Unable to remove Codex session directory')
                except Exception as exc:
                    self.diagnostics['cleanup_error'] = {'type': type(exc).__name__}
                    logger.warning('Codex cleanup failed; preserving execution result', exc_info=True)
                finally:
                    if result is not None:
                        self.diagnostics.update(
                            return_code=result.exit_code,
                            timed_out=result.timed_out,
                            failure_kind=result.failure_kind,
                            failure_stage=result.failure_stage,
                        )
                    self.diagnostics['cleanup_finished_at'] = time.time()
                    self._save()

            cleanup = asyncio.create_task(finalize())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
        assert result is not None
        if result.exit_code < 0:
            result = replace(
                result,
                stdout=result.stdout or self.diagnostics.get('stdout_tail', ''),
                stderr=result.stderr + '\n' + self.diagnostics.get('stderr_tail', ''),
            )
        return result
