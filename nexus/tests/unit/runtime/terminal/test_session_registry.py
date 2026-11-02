"""Executor-layer session registry + duplicate-id guard.

The registry (TERMINAL_SESSIONS) and its bookkeeping live in the executor layer,
not in TerminalSession. create_terminal_session rejects a reused live id before
spawning (code-review defect #10), registers only after a successful start, and
destroy_terminal_session pops + closes. These need a real patched bash, so they
are integration tests.
"""

import pytest

from nexus.runtime.executors.models.terminal import (
    CreateTerminalSessionRequest,
    DestroyTerminalSessionRequest,
)
from nexus.runtime.executors.terminal import executors as ex


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_registers_and_destroy_unregisters():
    sid = 'reg-lifecycle'
    try:
        await ex.create_terminal_session(CreateTerminalSessionRequest(session_id=sid, height=40, width=120))
        assert sid in ex.TERMINAL_SESSIONS
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))
        assert sid not in ex.TERMINAL_SESSIONS
    finally:
        sess = ex.TERMINAL_SESSIONS.pop(sid, None)
        if sess is not None:
            await sess.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_duplicate_id_rejected_without_orphaning_first():
    sid = 'reg-dup'
    try:
        await ex.create_terminal_session(CreateTerminalSessionRequest(session_id=sid, height=40, width=120))
        first = ex.TERMINAL_SESSIONS[sid]
        first_pid = first._pty.proc.pid

        with pytest.raises(ValueError, match='already exists'):
            await ex.create_terminal_session(CreateTerminalSessionRequest(session_id=sid, height=40, width=120))

        # The original session is untouched: same object, same live shell, no leak.
        assert ex.TERMINAL_SESSIONS[sid] is first
        assert first._pty.proc.pid == first_pid
    finally:
        await ex.destroy_terminal_session(DestroyTerminalSessionRequest(session_id=sid))


@pytest.mark.asyncio
async def test_get_missing_session_raises():
    # _get raises ValueError for an unknown id (no shell needed).
    with pytest.raises(ValueError, match='not found'):
        await ex._get('no-such-session')
