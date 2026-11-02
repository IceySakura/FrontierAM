"""Bounded IDE CLI execution, including its detached Electron descendants."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import psutil

from nexus.extensions.agents.installed_common import DEFAULT_CLI_STREAM_READER_LIMIT, find_runtime_binary
from nexus.extensions.agents.installed_common.stream_trace import flush_stream_trace_events, record_stream_trace_event


logger = logging.getLogger(__name__)
_POLL_SECONDS = 0.2
_STOP_GRACE_SECONDS = 5.0


@asynccontextmanager
async def virtual_display(env: dict[str, str]) -> AsyncIterator[dict[str, str]]:
    # Current Nexus images export DISPLAY even when GUI auto-start is disabled.
    # Always own a private X server so a stale inherited value cannot make the
    # Electron launcher silently wait for an IDE that never started.
    executable = find_runtime_binary('Xvfb')
    if not executable:
        raise RuntimeError('Xvfb is required for the CodeBuddy IDE CLI without DISPLAY')
    proc = await asyncio.create_subprocess_exec(
        executable,
        '-displayfd',
        '1',
        '-screen',
        '0',
        '1280x720x24',
        '-nolisten',
        'tcp',
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        assert proc.stdout is not None
        number = (await asyncio.wait_for(proc.stdout.readline(), timeout=15)).decode().strip()
        if not number.isdigit():
            raise RuntimeError('Xvfb did not provide a display number')
        yield {'DISPLAY': f':{number}'}
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_STOP_GRACE_SECONDS)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()


def _collect_children(pid: int, user_data_dir: str | None, known: set[psutil.Process]) -> None:
    with contextlib.suppress(psutil.Error):
        known.update(psutil.Process(pid).children(recursive=True))
    if user_data_dir is None:
        return
    # The launcher deliberately detaches the IDE. A unique user-data-dir lets us
    # recover that process after its parent has exited without touching another run.
    marker = f'--user-data-dir={user_data_dir}'
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info.get('cmdline') or []
        if process.pid == os.getpid() or marker not in args:
            continue
        known.add(process)
        with contextlib.suppress(psutil.Error):
            known.update(process.children(recursive=True))


async def _stop(pid: int, known: set[psutil.Process]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    for process in known:
        with contextlib.suppress(psutil.Error):
            process.terminate()
    _, alive = await asyncio.to_thread(psutil.wait_procs, list(known), timeout=_STOP_GRACE_SECONDS)
    for process in alive:
        with contextlib.suppress(psutil.Error):
            process.kill()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


async def _write_transcript(transcript_file: BinaryIO, chunk: bytes) -> None:
    def persist() -> None:
        transcript_file.write(chunk)
        transcript_file.flush()

    # Keep partial transcripts visible for diagnostics without blocking the loop.
    write = asyncio.create_task(asyncio.to_thread(persist))
    try:
        await asyncio.shield(write)
    except asyncio.CancelledError:
        # A cancelled reader must finish its write before the file is closed.
        await write
        raise


async def run_pre_start_command(
    argv: list[str], *, cwd: str, env: dict[str, str], timeout: float
) -> tuple[int, str, str, bool]:
    """Own the preparation process group and reap descendants before returning."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    output = asyncio.create_task(proc.communicate())
    known: set[psutil.Process] = set()
    deadline = asyncio.get_running_loop().time() + timeout
    timed_out = False

    async def cleanup() -> None:
        try:
            try:
                _collect_children(proc.pid, None, known)
            finally:
                await _stop(proc.pid, known)
            await asyncio.wait_for(asyncio.shield(output), timeout=_STOP_GRACE_SECONDS)
        finally:
            if not output.done():
                output.cancel()
            await asyncio.gather(output, return_exceptions=True)
            await asyncio.wait_for(proc.wait(), timeout=_STOP_GRACE_SECONDS)

    try:
        # communicate() also waits for inherited pipes. Poll the parent so a
        # background child cannot extend the script's lifetime after it exits.
        while proc.returncode is None:
            _collect_children(proc.pid, None, known)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timed_out = True
                break
            await asyncio.sleep(min(_POLL_SECONDS, remaining))
    finally:
        exit_code = proc.returncode
        stop = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(stop)
        except asyncio.CancelledError:
            await stop
            raise

    stdout, stderr = output.result()
    return (
        -1 if timed_out else (exit_code if exit_code is not None else -1),
        stdout.decode(errors='replace'),
        stderr.decode(errors='replace'),
        timed_out,
    )


async def run_cli(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    user_data_dir: str,
    transcript: Path,
) -> tuple[int, str, str, bool]:
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open('wb') as transcript_file:
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
        session = SimpleNamespace(session_id='', cwd=cwd)

        async def read(stream: asyncio.StreamReader, parts: list[bytes], is_stdout: bool) -> None:
            pending = b''
            while chunk := await stream.read(65536):
                parts.append(chunk)
                if is_stdout:
                    await _write_transcript(transcript_file, chunk)
                pending += chunk
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if is_stdout:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(event, dict):
                            session.session_id = event.get('session_id') or session.session_id
                            record_stream_trace_event(session, event, agent_key='cb-insiders')
                    elif line:
                        logger.debug('[cb-insiders] %s', line.decode(errors='replace'))
                if len(pending) > DEFAULT_CLI_STREAM_READER_LIMIT:
                    # Raw stdout is retained even when one malformed record is enormous.
                    pending = b''

        readers: list[asyncio.Task[None]] = []
        known: set[psutil.Process] = set()
        deadline = asyncio.get_running_loop().time() + timeout
        timed_out = False
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None
            readers.append(asyncio.create_task(read(proc.stdout, stdout, True)))
            readers.append(asyncio.create_task(read(proc.stderr, stderr, False)))
            while proc.returncode is None:
                _collect_children(proc.pid, user_data_dir, known)
                for reader in readers:
                    if reader.done() and not reader.cancelled() and reader.exception():
                        raise RuntimeError('Failed to read cb-insiders output') from reader.exception()
                if asyncio.get_running_loop().time() >= deadline:
                    timed_out = True
                    break
                await asyncio.sleep(_POLL_SECONDS)
        finally:
            exit_code = proc.returncode
            try:
                try:
                    _collect_children(proc.pid, user_data_dir, known)
                finally:
                    await _stop(proc.pid, known)
                if readers:
                    await asyncio.wait(readers, timeout=_STOP_GRACE_SECONDS)
            finally:
                for reader in readers:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
                flush_stream_trace_events(session, agent_key='cb-insiders')
        for reader in readers:
            if not reader.cancelled() and reader.exception():
                raise RuntimeError('Failed to read cb-insiders output') from reader.exception()
    return (
        (-1 if timed_out else (exit_code if exit_code is not None else -1)),
        b''.join(stdout).decode(errors='replace'),
        b''.join(stderr).decode(errors='replace'),
        timed_out,
    )
