"""Capture the CLI stream and reclaim its process tree on every exit path."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import psutil

from nexus.extensions.agents.installed_common import DEFAULT_CLI_STREAM_READER_LIMIT
from nexus.extensions.agents.installed_common.stream_trace import flush_stream_trace_events, record_stream_trace_event


@dataclass
class ProcessResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _children(pid: int, known: set[psutil.Process]) -> None:
    with contextlib.suppress(psutil.Error):
        known.update(psutil.Process(pid).children(recursive=True))


async def _stop(proc: asyncio.subprocess.Process, known: set[psutil.Process]) -> None:
    _children(proc.pid, known)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    for child in known:
        with contextlib.suppress(psutil.Error):
            child.terminate()
    _, alive = await asyncio.to_thread(psutil.wait_procs, list(known), timeout=3)
    for child in alive:
        with contextlib.suppress(psutil.Error):
            child.kill()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await asyncio.wait_for(proc.wait(), timeout=5)


async def run_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    transcript: Path | None = None,
) -> ProcessResult:
    """Run a Linux CLI with bounded output draining and cancellation cleanup."""
    if timeout <= 0:
        raise ValueError('Process timeout must be positive')
    with ExitStack() as files:
        if transcript is not None:
            transcript.parent.mkdir(parents=True, exist_ok=True)
        handle = files.enter_context(transcript.open('wb')) if transcript is not None else None
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=DEFAULT_CLI_STREAM_READER_LIMIT,
            start_new_session=True,
        )
        stdout: list[bytes] = []
        stderr: list[bytes] = []
        known: set[psutil.Process] = set()
        session = SimpleNamespace(session_id='', cwd=cwd)

        async def read(stream: asyncio.StreamReader, parts: list[bytes], is_stdout: bool) -> None:
            pending = b''
            while chunk := await stream.read(65536):
                parts.append(chunk)
                if not is_stdout or handle is None:
                    continue

                def persist(data: bytes = chunk) -> None:
                    handle.write(data)
                    handle.flush()

                write = asyncio.create_task(asyncio.to_thread(persist))
                try:
                    await asyncio.shield(write)
                except asyncio.CancelledError:
                    await write
                    raise
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict):
                        session.session_id = event.get('session_id') or session.session_id
                        record_stream_trace_event(session, event, agent_key='kimi_code')
                if len(pending) > DEFAULT_CLI_STREAM_READER_LIMIT:
                    pending = b''

        readers: list[asyncio.Task[None]] = []
        timed_out = False
        exit_code = -1

        async def cleanup() -> None:
            try:
                await _stop(proc, known)
                if readers:
                    await asyncio.wait(readers, timeout=5)
            finally:
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                flush_stream_trace_events(session, agent_key='kimi_code')

        try:
            if proc.stdout is None or proc.stderr is None:
                raise RuntimeError('Kimi Code output pipes were not created')
            readers = [
                asyncio.create_task(read(proc.stdout, stdout, True)),
                asyncio.create_task(read(proc.stderr, stderr, False)),
            ]
            deadline = asyncio.get_running_loop().time() + timeout
            while proc.returncode is None:
                _children(proc.pid, known)
                for reader in readers:
                    if reader.done() and not reader.cancelled() and reader.exception() is not None:
                        raise RuntimeError('Failed to capture Kimi Code output') from reader.exception()
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    timed_out = True
                    break
                await asyncio.sleep(min(0.2, remaining))
            exit_code = proc.returncode if proc.returncode is not None else -1
        finally:
            stop = asyncio.create_task(cleanup())
            try:
                await asyncio.shield(stop)
            except asyncio.CancelledError:
                await stop
                raise
        for reader in readers:
            if not reader.cancelled() and reader.exception() is not None:
                raise RuntimeError('Failed to capture Kimi Code output') from reader.exception()
        return ProcessResult(
            stdout=b''.join(stdout).decode(errors='replace'),
            stderr=b''.join(stderr).decode(errors='replace'),
            exit_code=-1 if timed_out else exit_code,
            timed_out=timed_out,
        )
