"""Shell recovery must preserve diagnostics and never reuse an issued command ID."""

import asyncio
import json
import uuid

from pathlib import Path
from types import SimpleNamespace

import pytest

from nexus.runtime.executors.models.terminal import (
    CapturePaneRequest,
    CreateTerminalSessionRequest,
    DestroyTerminalSessionRequest,
    QueryCommandStatusRequest,
    RunCommandInSessionRequest,
    SendKeysToSessionRequest,
    TerminateSessionProcessesRequest,
)
from nexus.runtime.executors.terminal import executors as ex, session as session_module
from nexus.runtime.executors.terminal.errors import TerminalInfraError
from nexus.runtime.executors.terminal.session import TerminalSession


@pytest.fixture
def capture_shell(monkeypatch, tmp_path):
    """A PTY double writing the real capture-file protocol; no shell processes."""
    state = SimpleNamespace(spawned=[], entered=asyncio.Event(), gate=None, fail_warmup=False)
    monkeypatch.setattr(session_module, 'TOOLS_DIR', tmp_path)
    monkeypatch.setattr(session_module, 'locate_patched_bash', lambda: '/fixture/bash')

    class CapturePty:
        def __init__(self, directory):
            self.directory = directory
            self.counter = 0
            self.closed = False
            self.killed = False
            self.proc = SimpleNamespace(wait=self.wait)

        async def wait(self):
            return 0

        def status(self):
            (self.directory / 'status.json').write_text(json.dumps({'status': 'idle', 'command_id': self.counter}))

        async def write(self, data):
            command = data.decode().strip()
            if state.fail_warmup and command == 'true':
                raise RuntimeError('warmup failed')
            self.counter += 1
            cid = self.counter
            (self.directory / f'cmd_{cid}.json').write_text(json.dumps({
                'command_id': cid, 'batch': cid, 'start_time': 1, 'end_time': 2, 'exit_code': 0,
            }))
            for prefix, content in [('cmd_content', command), ('stdout', command), ('output', command), ('stderr', '')]:
                (self.directory / f'{prefix}_{cid}').write_text(content)
            self.status()

        def killpg(self):
            self.killed = True

        def close(self):
            self.closed = True

    async def spawn(**kwargs):
        directory = Path(kwargs['env']['NEXUS_CAPTURE_DIR'])
        # An old status file must not satisfy readiness for the next process.
        assert not (directory / 'status.json').exists()
        assert not list(directory.glob('cmd_*.json'))
        pty = CapturePty(directory)
        state.spawned.append(pty)
        state.entered.set()
        if state.gate is not None:
            await state.gate.wait()
        await asyncio.sleep(0)
        pty.status()
        return pty

    monkeypatch.setattr(session_module.PtyProcess, 'spawn', spawn)
    return state


async def test_revive_clears_capture_and_invalidates_old_command_ids(capture_shell):
    session = TerminalSession()
    try:
        await session.start()
        previous = await session.run_command('OLD OUTPUT')
        session._on_exit()
        await session.revive()
        assert await session.get_command_info(previous.command_id) is None
        current = await session.run_command('NEW OUTPUT')
        assert current.command_id > previous.command_id
        assert current.stdout == 'NEW OUTPUT'
        assert (await session.get_command_info(current.command_id)).stdout == 'NEW OUTPUT'
        session._on_exit()
        await session.revive()
        assert await session.get_command_info(current.command_id) is None
        latest = await session.run_command('LATEST OUTPUT')
        assert latest.command_id > current.command_id
        assert latest.stdout == 'LATEST OUTPUT'
    finally:
        await session.close()
    assert all(pty.closed and pty.killed for pty in capture_shell.spawned)


async def test_dead_session_diagnostics_and_controls_do_not_revive(capture_shell):
    response = await ex.create_terminal_session(CreateTerminalSessionRequest())
    sid = response.session_id
    session = ex.TERMINAL_SESSIONS[sid]
    try:
        command = await session.run_command('OLD OUTPUT')
        session._on_data(b'CRASH EVIDENCE\r\n')
        session._on_exit()
        pane = await ex.capture_pane(CapturePaneRequest(session_id=sid))
        assert 'CRASH EVIDENCE' in pane.content
        info = await ex.query_command_status(QueryCommandStatusRequest(session_id=sid, command_id=command.command_id))
        assert info.stdout == 'OLD OUTPUT'
        with pytest.raises(RuntimeError, match='not alive'):
            await ex.send_keys_to_session(SendKeysToSessionRequest(session_id=sid, keys='C-c'))
        termination = await ex.terminate_session_processes(TerminateSessionProcessesRequest(session_id=sid))
        assert termination.success is False
        assert 'not alive' in termination.error
        assert len(capture_shell.spawned) == 1
        assert session.is_dead
    finally:
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))


async def test_concurrent_recovery_spawns_only_one_replacement(capture_shell):
    response = await ex.create_terminal_session(CreateTerminalSessionRequest())
    sid = response.session_id
    session = ex.TERMINAL_SESSIONS[sid]
    session._on_exit()
    capture_shell.entered.clear()
    capture_shell.gate = asyncio.Event()
    requests = [asyncio.create_task(ex._get(sid, recover=True)) for _ in range(4)]
    try:
        await asyncio.wait_for(capture_shell.entered.wait(), timeout=2)
        await asyncio.sleep(0)
        assert len(capture_shell.spawned) == 2  # original plus exactly one replacement
        capture_shell.gate.set()
        assert all(result is session for result in await asyncio.gather(*requests))
        assert not session.is_dead
        assert len(capture_shell.spawned) == 2
    finally:
        capture_shell.gate.set()
        await asyncio.gather(*requests, return_exceptions=True)
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))
    assert all(pty.closed and pty.killed for pty in capture_shell.spawned)


async def test_destroy_waits_for_recovery_and_closes_replacement(capture_shell):
    response = await ex.create_terminal_session(CreateTerminalSessionRequest())
    sid = response.session_id
    ex.TERMINAL_SESSIONS[sid]._on_exit()
    capture_shell.entered.clear()
    capture_shell.gate = asyncio.Event()
    recovery = asyncio.create_task(ex._get(sid, recover=True))
    await asyncio.wait_for(capture_shell.entered.wait(), timeout=2)
    destroy = asyncio.create_task(ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid)))
    try:
        await asyncio.sleep(0)
        assert not destroy.done()
    finally:
        capture_shell.gate.set()
        await asyncio.gather(recovery, destroy)
    assert sid not in ex.TERMINAL_SESSIONS
    assert all(pty.closed and pty.killed for pty in capture_shell.spawned)


async def test_concurrent_create_rejects_duplicate_without_orphan(capture_shell):
    sid = 'same-session'
    try:
        results = await asyncio.gather(
            *(ex.create_terminal_session(CreateTerminalSessionRequest(session_id=sid)) for _ in range(2)),
            return_exceptions=True,
        )
        assert sum(isinstance(result, ValueError) for result in results) == 1
        assert len(capture_shell.spawned) == 1
    finally:
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))
    assert capture_shell.spawned[0].closed


async def test_failed_warmup_closes_new_pty_and_reports_infrastructure_error(capture_shell):
    response = await ex.create_terminal_session(CreateTerminalSessionRequest())
    sid = response.session_id
    session = ex.TERMINAL_SESSIONS[sid]
    try:
        session._on_exit()
        capture_shell.fail_warmup = True
        with pytest.raises(TerminalInfraError, match='warmup failed'):
            await ex._get(sid, recover=True)
        assert session.is_dead
        assert not Path(session.output_dir).exists()
        assert all(pty.closed and pty.killed for pty in capture_shell.spawned)
    finally:
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))


@pytest.mark.integration
async def test_real_bash_recovery_preserves_diagnostics_and_command_identity(monkeypatch):
    """Run against an actual patched Bash, locally or inside a task sandbox."""
    spawned = []
    original_spawn = session_module.PtyProcess.spawn

    async def spawn(**kwargs):
        pty = await original_spawn(**kwargs)
        spawned.append(pty)
        return pty

    monkeypatch.setattr(session_module.PtyProcess, 'spawn', spawn)
    sid = f'recovery-{uuid.uuid4().hex}'
    try:
        await ex.create_terminal_session(CreateTerminalSessionRequest(session_id=sid))
        before = await ex.run_command_in_session(
            RunCommandInSessionRequest(session_id=sid, command="printf 'BEFORE CRASH\\n'", timeout=10)
        )
        session = ex.TERMINAL_SESSIONS[sid]
        session._pty.killpg()
        await session._pty.proc.wait()
        for _ in range(100):
            if session.is_dead:
                break
            await asyncio.sleep(0.01)
        assert session.is_dead
        assert 'BEFORE CRASH' in (await ex.capture_pane(CapturePaneRequest(session_id=sid))).content
        assert (await session.get_command_info(before.command_id)).stdout == 'BEFORE CRASH\n'
        assert len(spawned) == 1
        await asyncio.gather(*(ex._get(sid, recover=True) for _ in range(4)))
        assert len(spawned) == 2
        with pytest.raises(ValueError, match='not found'):
            await ex.query_command_status(QueryCommandStatusRequest(session_id=sid, command_id=before.command_id))
        after = await ex.run_command_in_session(
            RunCommandInSessionRequest(session_id=sid, command="printf 'AFTER RECOVERY\\n'", timeout=10)
        )
        assert after.command_id > before.command_id
        assert after.stdout == 'AFTER RECOVERY\n'
        assert (await session.get_command_info(after.command_id)).stdout == after.stdout
    finally:
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))
        await asyncio.wait_for(asyncio.gather(*(pty.proc.wait() for pty in spawned)), timeout=10)
    assert all(pty.proc.returncode is not None for pty in spawned)
