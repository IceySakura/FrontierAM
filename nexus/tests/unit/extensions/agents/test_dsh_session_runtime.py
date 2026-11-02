from __future__ import annotations

import asyncio

from pathlib import Path
from typing import Any

import pytest

from nexus.extensions.agents.dsh import session_runtime


class _Session:
    def __init__(self, session_id: str, cwd: str) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.is_running = True
        self.stderr_lines: list[str] = []
        self.prompts: list[str] = []
        self.closed = False

    async def send_user_message(self, prompt: str, *, timeout: float, transcript_timeout: float) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {
            'success': True,
            'session_id': self.session_id,
            'output': prompt,
            'stderr': '',
            'exit_code': 0,
            'execution_time': 0.1,
            'events': [],
        }

    async def close(self) -> dict[str, Any]:
        self.closed = True
        return {'success': True, 'session_id': self.session_id, 'exit_code': 0, 'stderr': ''}


@pytest.mark.asyncio
async def test_runtime_reuses_one_session_for_multiple_turns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    starts = 0

    async def fake_start_session(**kwargs: Any) -> _Session:
        nonlocal starts
        starts += 1
        return _Session(kwargs['session_id'], str(Path(kwargs['working_dir']).resolve()))

    monkeypatch.setattr(session_runtime, '_start_session', fake_start_session)
    session_runtime._SESSIONS.clear()
    session_runtime._SESSION_LOCKS.clear()
    try:
        first = await session_runtime.run_dsh_session_turn(
            session_id='session-1', prompt='first', working_dir=str(tmp_path)
        )
        second = await session_runtime.run_dsh_session_turn(
            session_id='session-1', prompt='second', working_dir=str(tmp_path)
        )
        session = session_runtime._SESSIONS['session-1']

        assert starts == 1
        assert first['output'] == 'first'
        assert second['output'] == 'second'
        assert session.prompts == ['first', 'second']
        await session_runtime.close_dsh_session(session_id='session-1')
        assert session.closed is True
    finally:
        session_runtime._SESSIONS.clear()
        session_runtime._SESSION_LOCKS.clear()


@pytest.mark.asyncio
async def test_runtime_evicts_failed_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = _Session('session-1', str(tmp_path.resolve()))

    async def fail_turn(_prompt: str, *, timeout: float, transcript_timeout: float) -> dict[str, Any]:
        return {
            'success': False,
            'session_id': 'session-1',
            'output': '',
            'stderr': 'failed',
            'exit_code': -1,
            'execution_time': 0.1,
            'events': [],
        }

    session.send_user_message = fail_turn  # type: ignore[method-assign]
    monkeypatch.setattr(session_runtime, '_start_session', lambda **_kwargs: None)
    session_runtime._SESSIONS.clear()
    session_runtime._SESSION_LOCKS.clear()
    session_runtime._SESSIONS['session-1'] = session
    try:
        result = await session_runtime.run_dsh_session_turn(
            session_id='session-1', prompt='first', working_dir=str(tmp_path)
        )

        assert result['success'] is False
        assert session.closed is True
        assert 'session-1' not in session_runtime._SESSIONS
    finally:
        session_runtime._SESSIONS.clear()
        session_runtime._SESSION_LOCKS.clear()


@pytest.mark.asyncio
async def test_runtime_cancellation_closes_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    started = asyncio.Event()
    session = _Session('session-1', str(tmp_path.resolve()))

    async def block_turn(_prompt: str, *, timeout: float, transcript_timeout: float) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError('unreachable')

    session.send_user_message = block_turn  # type: ignore[method-assign]
    monkeypatch.setattr(session_runtime, '_start_session', lambda **_kwargs: None)
    session_runtime._SESSIONS.clear()
    session_runtime._SESSION_LOCKS.clear()
    session_runtime._SESSIONS['session-1'] = session
    task = asyncio.create_task(
        session_runtime.run_dsh_session_turn(session_id='session-1', prompt='first', working_dir=str(tmp_path))
    )
    try:
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert session.closed is True
        assert 'session-1' not in session_runtime._SESSIONS
    finally:
        task.cancel()
        session_runtime._SESSIONS.clear()
        session_runtime._SESSION_LOCKS.clear()
