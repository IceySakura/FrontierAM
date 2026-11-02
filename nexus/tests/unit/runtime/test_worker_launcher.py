import asyncio
import sys

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_worker_stream_drainers_prevent_pipe_backpressure() -> None:
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        '-c',
        "import os; os.write(1, b'o' * 1048576); os.write(2, b'e' * 1048576)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output_tasks = (
        asyncio.create_task(
            WorkerLauncher._drain_worker_stream(
                process.stdout,
                worker_id='worker-1',
                stream_name='stdout',
                output_tail=bytearray(),
            )
        ),
        asyncio.create_task(
            WorkerLauncher._drain_worker_stream(
                process.stderr,
                worker_id='worker-1',
                stream_name='stderr',
                output_tail=bytearray(),
            )
        ),
    )

    await asyncio.wait_for(asyncio.gather(*output_tasks), timeout=5)
    await asyncio.sleep(0.1)

    assert process.returncode == 0


@pytest.mark.asyncio
async def test_worker_launcher_continuously_drains_subprocess_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from nexus.runtime.server.platform import venv_python_path
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    python = venv_python_path(tmp_path)
    python.parent.mkdir()
    python.touch()

    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_data(b'o' * (256 * 1024))
    stdout.feed_eof()
    stderr.feed_data(b'e' * (256 * 1024))
    stderr.feed_eof()
    process = SimpleNamespace(pid=1234, returncode=None, stdout=stdout, stderr=stderr)

    async def create_subprocess_exec(*_args: str, **_kwargs: Any) -> Any:
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_clean_sandbox_env', lambda: {})

    launcher = WorkerLauncher()
    status = await launcher.start_worker('venv-1', tmp_path, base_url='http://base.invalid')
    worker = launcher._workers[status.worker_id]
    await asyncio.gather(*worker.output_tasks)

    assert stdout.at_eof()
    assert stderr.at_eof()


@pytest.mark.asyncio
async def test_worker_launcher_retains_bounded_startup_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from nexus.runtime.server.platform import venv_python_path
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    python = venv_python_path(tmp_path)
    python.parent.mkdir()
    python.touch()

    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_data(b'stdout diagnostic')
    stdout.feed_eof()
    stderr.feed_data(b'stderr diagnostic')
    stderr.feed_eof()
    process = SimpleNamespace(pid=1234, returncode=1, stdout=stdout, stderr=stderr)

    async def create_subprocess_exec(*_args: str, **_kwargs: Any) -> Any:
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_clean_sandbox_env', lambda: {})

    launcher = WorkerLauncher()
    status = await launcher.start_worker('venv-1', tmp_path, base_url='http://base.invalid')
    worker = launcher._workers[status.worker_id]
    await asyncio.gather(*worker.output_tasks)

    detail = await launcher.read_startup_output(status.worker_id)

    assert detail == 'worker exited rc=1 | stderr: stderr diagnostic | stdout: stdout diagnostic'


@pytest.mark.asyncio
async def test_worker_launcher_cancels_output_drainers_when_worker_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv('NEXUS_OTEL_ENABLED', raising=False)
    from nexus.runtime.server.platform import venv_python_path
    from nexus.runtime.server.worker.launcher import WorkerLauncher

    python = venv_python_path(tmp_path)
    python.parent.mkdir()
    python.touch()

    process = SimpleNamespace(
        pid=1234,
        returncode=0,
        stdout=asyncio.StreamReader(),
        stderr=asyncio.StreamReader(),
    )

    async def create_subprocess_exec(*_args: str, **_kwargs: Any) -> Any:
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', create_subprocess_exec)
    monkeypatch.setattr('nexus.utils.shell.get_clean_sandbox_env', lambda: {})

    launcher = WorkerLauncher()
    status = await launcher.start_worker('venv-1', tmp_path, base_url='http://base.invalid')
    output_tasks = launcher._workers[status.worker_id].output_tasks

    await launcher.stop_worker(status.worker_id)

    assert all(task.done() for task in output_tasks)
