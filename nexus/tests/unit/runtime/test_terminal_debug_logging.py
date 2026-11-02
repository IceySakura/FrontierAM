"""Tests that tmux terminal-session command invokes log their command content at DEBUG level."""

import logging

import pytest

from nexus.runtime.executors.models.terminal import (
    CommandInfo,
    RunCommandInSessionRequest,
    StartCommandInSessionRequest,
)
from nexus.runtime.executors.terminal import TERMINAL_SESSIONS, run_command_in_session, start_command_in_session


class _FakeTerminalSession:
    """Stand-in for a real tmux-backed TerminalSession.

    Real sessions require the sandbox-only `nexus_capture`/`nexus_bash` binaries
    (not present in the dev/CI environment), so these tests inject a fake session
    directly to exercise the executor's own logging without the tmux dependency.
    """

    is_dead = False

    async def run_command(self, command: str, timeout: float | None = None) -> CommandInfo:
        return CommandInfo(command_id=1, command=command, exit_code=0)

    async def start_command(self, command: str) -> int:
        self._last_command = command
        return 1

    async def get_command_info(self, command_id: int) -> CommandInfo | None:
        return CommandInfo(command_id=command_id, command=self._last_command, exit_code=None)


@pytest.mark.asyncio
async def test_run_command_in_terminal_session_logs_command_at_debug(caplog):
    session_id = 'test-session-run'
    TERMINAL_SESSIONS[session_id] = _FakeTerminalSession()
    try:
        with caplog.at_level(logging.DEBUG, logger='nexus.runtime.executors.terminal'):
            await run_command_in_session(
                RunCommandInSessionRequest(session_id=session_id, command='echo session-run-test')
            )

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(session_id in r.message and 'echo session-run-test' in r.message for r in debug_records)
    finally:
        del TERMINAL_SESSIONS[session_id]


@pytest.mark.asyncio
async def test_start_command_in_terminal_session_logs_command_at_debug(caplog):
    session_id = 'test-session-start'
    TERMINAL_SESSIONS[session_id] = _FakeTerminalSession()
    try:
        with caplog.at_level(logging.DEBUG, logger='nexus.runtime.executors.terminal'):
            await start_command_in_session(
                StartCommandInSessionRequest(session_id=session_id, command='echo session-start-test')
            )

        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any(session_id in r.message and 'echo session-start-test' in r.message for r in debug_records)
    finally:
        del TERMINAL_SESSIONS[session_id]
