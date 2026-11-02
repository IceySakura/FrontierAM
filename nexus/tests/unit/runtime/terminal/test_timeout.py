import asyncio

import psutil
import pytest

from nexus.runtime.executors.terminal.capture import BatchCapture, CommandRecord
from nexus.runtime.executors.terminal.session import TerminalSession


class _FakePty:
    async def terminate_foreground(self):
        raise AssertionError('timeout must use session-wide process cleanup')

    def killpg(self):
        pass

    def close(self):
        pass


@pytest.mark.asyncio
async def test_timeout_uses_session_process_cleanup(monkeypatch):
    session = TerminalSession()
    monkeypatch.setattr(session, '_pty', _FakePty())
    terminated = False

    running = BatchCapture(
        records=[CommandRecord(command_id=1, batch=1, command='sleep 30', start_time=1.0)],
        all_finished=False,
    )
    finished = BatchCapture(
        records=[
            CommandRecord(
                command_id=1,
                batch=1,
                command='sleep 30',
                start_time=1.0,
                end_time=2.0,
                exit_code=130,
            )
        ],
        all_finished=True,
    )

    async def fake_start_command(_command):
        return 1

    async def fake_terminate_session_processes():
        nonlocal terminated
        terminated = True

    async def fake_read_batch(_first_id):
        return finished if terminated else running

    monkeypatch.setattr(session, 'start_command', fake_start_command)
    monkeypatch.setattr(session, 'terminate_session_processes', fake_terminate_session_processes)
    monkeypatch.setattr(session._capture, 'read_batch', fake_read_batch)

    try:
        with pytest.raises(TimeoutError, match='Process termination succeeded'):
            await session.run_command('sleep 30', timeout=0)
        assert terminated is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_timeout_reports_session_cleanup_failure(monkeypatch):
    session = TerminalSession()
    monkeypatch.setattr(session, '_pty', _FakePty())
    cleanup_attempted = False

    running = BatchCapture(
        records=[CommandRecord(command_id=1, batch=1, command='sleep 30', start_time=1.0)],
        all_finished=False,
    )
    finished = BatchCapture(
        records=[
            CommandRecord(
                command_id=1,
                batch=1,
                command='sleep 30',
                start_time=1.0,
                end_time=2.0,
                exit_code=130,
            )
        ],
        all_finished=True,
    )

    async def fake_start_command(_command):
        return 1

    async def fake_terminate_session_processes():
        nonlocal cleanup_attempted
        cleanup_attempted = True
        raise RuntimeError('forced cleanup failure')

    async def fake_read_batch(_first_id):
        return finished if cleanup_attempted else running

    monkeypatch.setattr(session, 'start_command', fake_start_command)
    monkeypatch.setattr(session, 'terminate_session_processes', fake_terminate_session_processes)
    monkeypatch.setattr(session._capture, 'read_batch', fake_read_batch)

    try:
        with pytest.raises(TimeoutError, match='Process termination failed: forced cleanup failure'):
            await session.run_command('sleep 30', timeout=0)
    finally:
        await session.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_timeout_cleans_background_jobs_and_session_recovers():
    session = TerminalSession()
    await session.start()
    try:
        await session.run_command('sleep 30 &', timeout=10)
        jobs = await session.run_command('jobs -p', timeout=10)
        job_pids = [int(pid) for pid in jobs.stdout.split()]
        assert job_pids
        status = await session._capture.read_status()
        assert status is not None
        timed_out_command_id = status.command_id + 1

        with pytest.raises(TimeoutError, match='Process termination succeeded'):
            await session.run_command('sleep 30', timeout=0.2)

        timed_out = await session.get_command_info(timed_out_command_id)
        assert timed_out is not None
        assert timed_out.end_time is not None
        assert timed_out.exit_code not in (None, 0)

        for _ in range(50):
            if all(not psutil.pid_exists(pid) for pid in job_pids):
                break
            await asyncio.sleep(0.05)
        assert all(not psutil.pid_exists(pid) for pid in job_pids)

        recovered = await session.run_command('echo AFTER_TIMEOUT_RECOVERED', timeout=10)
        assert recovered.stdout == 'AFTER_TIMEOUT_RECOVERED\n'
    finally:
        await session.close()
