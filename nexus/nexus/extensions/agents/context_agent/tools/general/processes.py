"""Bounded process-group lifecycle for model-authored commands."""

import asyncio
import contextlib
import os
import signal
import subprocess
import threading
import time

from typing import Any


def kill_process_group(proc: subprocess.Popen) -> None:
    # Even an exited leader may have live descendants holding output pipes.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5)


def run_command_group(command: Any, *, timeout: float, cancelled: threading.Event, **kwargs: Any):
    if cancelled.is_set():
        raise InterruptedError('Command cancelled before launch')
    proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancelled.is_set():
                raise InterruptedError('Command cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = proc.communicate(timeout=min(remaining, 0.2))
                return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    finally:
        kill_process_group(proc)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()


async def run_cancellable(function, *args):
    """Signal cancellation to blocking work and wait for its process cleanup."""
    cancelled = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(function, *args, cancelled=cancelled))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(task), timeout=6)
        raise
