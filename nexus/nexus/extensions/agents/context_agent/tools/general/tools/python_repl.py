"""``python`` tool — stateful subprocess REPL in the main sandbox image.

One persistent main-image ``python -u repl_server.py`` child per rollout.
Code strings are sent as length-prefixed frames; the namespace persists across
calls (Jupyter-like). On timeout the child is hard-killed (``kill -9``) and
respawned, and a clear timeout message is returned so the rollout continues.

cwd is the rollout's log folder, so ``./workspace/<name>`` resolves and any
files the model writes land alongside the trajectory. This is how a binary
input becomes readable: extract it to a ``.txt`` here, then use the
line-based tools on that.

Concurrency: a single rollout may issue several ``python`` calls in one
assistant turn (``max_parallel_calls`` > 1). An :class:`asyncio.Lock`
single-flights them so only one runs at a time — this keeps the framed
stdin/stdout protocol intact and ensures a timeout only ever kills the
call that actually timed out (never a sibling that is mid-execution).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import threading

from pathlib import Path

from ..processes import kill_process_group
from ..subprocess_env import main_image_env, resolve_main_image_python, sandbox_command, subprocess_identity_args


logger = logging.getLogger(__name__)

_SERVER = Path(__file__).resolve().parent / 'repl_server.py'
_MAX_RESULT_CHARS = 200_000


def _write_all(stream, payload: bytes) -> None:
    """Write every byte of *payload*, looping over short writes."""
    view = memoryview(payload)
    total = 0
    while total < len(view):
        n = stream.write(view[total:])
        if n is None:  # some raw streams return None and write all
            break
        total += n
    stream.flush()


def _read_exact(stream, n: int) -> bytes:
    """Read exactly *n* bytes, looping over short reads. Raises on EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError('python REPL closed unexpectedly')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)


class PythonREPL:
    """Manages a persistent Python subprocess for one rollout."""

    def __init__(self, cwd: str, timeout: int = 120, *, server_source: str | None = None):
        self.server_source = server_source
        self.cwd = cwd
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._process_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._io_lock = threading.Lock()  # guards one blocking send/recv
        self._async_lock = asyncio.Lock()  # single-flights concurrent exec()

    # ------------------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        env = main_image_env()
        identity_args = subprocess_identity_args()
        # The worker wheel can live under /root; feed the trusted REPL source
        # as code so the sandbox user need not traverse that private directory.
        source = self.server_source or _SERVER.read_text(encoding='utf-8')
        server_args = ['-c', source] if identity_args or self.server_source else [str(_SERVER)]
        return subprocess.Popen(
            sandbox_command([resolve_main_image_python(env), '-u', *server_args]),
            start_new_session=True,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            # The REPL runs arbitrary model-authored code; keep the endpoint
            # credentials out of its os.environ. See contextagent/subprocess_env.py.
            env=env,
            **identity_args,
        )

    def _ensure(self) -> subprocess.Popen:
        with self._process_lock:
            if self._cancelled.is_set():
                raise InterruptedError('Python request cancelled before launch')
            if self._proc is None or self._proc.poll() is not None:
                if self._proc is not None:
                    kill_process_group(self._proc)
                self._proc = self._spawn()
            return self._proc

    def _kill(self) -> None:
        with self._process_lock:
            proc, self._proc = self._proc, None
            if proc is not None:
                kill_process_group(proc)

    # ------------------------------------------------------------------
    def _exec_blocking(self, code: str) -> dict:
        """Send one code frame and read one response frame (blocking)."""
        with self._io_lock:
            proc = self._ensure()
            assert proc.stdin is not None
            assert proc.stdout is not None
            payload = code.encode('utf-8')
            _write_all(proc.stdin, f'{len(payload):08x}'.encode('ascii'))
            _write_all(proc.stdin, payload)

            header = _read_exact(proc.stdout, 8)
            n = int(header.decode('ascii'), 16)
            data = _read_exact(proc.stdout, n)
            return json.loads(data.decode('utf-8', 'replace'))

    async def exec(self, code: str) -> str:
        """Execute *code*, return a formatted stdout/stderr/result block.

        Serialised across concurrent calls so a timeout never disturbs a
        sibling call's subprocess.
        """
        if not code or not code.strip():
            return '(empty code — provide Python code to execute)'
        async with self._async_lock:
            self._cancelled.clear()
            worker = asyncio.create_task(asyncio.to_thread(self._exec_blocking, code))
            try:
                resp = await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                await self._stop_request(worker)
                return (
                    f'[Error] python execution timed out after {self.timeout}s and '
                    f'the kernel was restarted (all in-memory state was lost). '
                    f'Run lighter code or break the work into smaller steps.'
                )
            except asyncio.CancelledError:
                await self._stop_request(worker)
                raise
            except Exception as exc:  # noqa: BLE001
                await self._stop_request(worker)
                return f'[Error] python REPL failed: {type(exc).__name__}: {exc}'
        return _format(resp)

    async def _stop_request(self, worker: asyncio.Task) -> None:
        self._cancelled.set()
        await asyncio.to_thread(self._kill)
        # Do not reuse the pipe/lock until the blocked reader has stopped.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(worker), timeout=6)

    def close(self) -> None:
        self._cancelled.set()
        self._kill()


def _format(resp: dict) -> str:
    parts: list[str] = []
    stdout = (resp.get('stdout') or '').rstrip()
    stderr = (resp.get('stderr') or '').rstrip()
    result = resp.get('result')
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f'[stderr]\n{stderr}')
    if result is not None:
        parts.append(f'[result] {result}')
    out = '\n'.join(parts) if parts else '(no output)'
    if len(out) > _MAX_RESULT_CHARS:
        out = out[:_MAX_RESULT_CHARS] + '\n... (output truncated)'
    return out
