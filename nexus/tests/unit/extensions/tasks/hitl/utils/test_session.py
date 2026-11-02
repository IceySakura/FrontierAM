"""Session execution reopens failed attempts and preserves completed workspace work."""

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nexus.extensions.tasks.hitl.actor.base import Session, TurnResult
from nexus.extensions.tasks.hitl.utils.session import SessionAttemptsExhaustedError, run_session
from nexus.runtime.client.base import RuntimeUnhealthyError


@pytest.fixture
def runner():
    events = []
    sessions = []

    async def start(*, name, workspace):
        session = Session(name=name, session_id=str(len(sessions)), backend='test')
        sessions.append(session)
        events.append(('start', session.session_id, workspace))
        return session

    async def send(session, message):
        events.append(('send', session.session_id, message.content))
        return TurnResult(kind='completed')

    async def close(session):
        events.append(('close', session.session_id))

    manager = SimpleNamespace(start=start, send=send, close=close)
    return SimpleNamespace(
        manager=manager, runtime=SimpleNamespace(is_unhealthy=False), events=events, sessions=sessions, send=send
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'failure',
    [
        TurnResult(kind='terminal', raw={'reason': 'Process ended'}),
        TurnResult(kind='completed', raw={'success': False, 'stderr': 'Execution failed'}),
        TurnResult(kind='completed', raw={'result_event': {'is_error': True, 'result': 'Backend error'}}),
    ],
)
async def test_failed_execution_closes_before_new_session(runner, tmp_path, failure):
    output = tmp_path / 'output.json'

    async def send(session, message):
        await runner.send(session, message)
        if session is runner.sessions[0]:
            output.write_text('{"draft": true}')
            return failure
        assert json.loads(output.read_text()) == {'draft': True}
        output.write_text('{"completed": true}')
        return TurnResult(kind='completed')

    runner.manager.send = send
    read = AsyncMock(side_effect=lambda: json.loads(output.read_text()))
    result = await run_session(
        runner.manager, runner.runtime, 'build', 'Finish the project.', read, workspace=str(tmp_path)
    )

    assert result == {'completed': True}
    assert runner.events == [
        ('start', '0', str(tmp_path)),
        ('send', '0', 'Finish the project.'),
        ('close', '0'),
        ('start', '1', str(tmp_path)),
        ('send', '1', 'Finish the project.'),
        ('close', '1'),
    ]
    assert runner.sessions[0] is not runner.sessions[1]
    read.assert_awaited_once()


@pytest.mark.asyncio
async def test_malformed_output_corrected_in_successful_session(runner):
    read = AsyncMock(side_effect=[ValueError('Missing result field'), {'result': 4}])
    result = await run_session(
        runner.manager,
        runner.runtime,
        'build',
        'Build.',
        read,
        workspace='/project',
        correction_message='Fix the output: {error}',
    )
    assert result == {'result': 4}
    assert runner.events == [
        ('start', '0', '/project'),
        ('send', '0', 'Build.'),
        ('send', '0', 'Fix the output: Missing result field'),
        ('close', '0'),
    ]


@pytest.mark.asyncio
async def test_failed_correction_starts_new_session_with_original_prompt(runner):
    async def send(session, message):
        await runner.send(session, message)
        if message.content.startswith('Correct'):
            return TurnResult(kind='terminal')
        return TurnResult(kind='completed')

    runner.manager.send = send
    read = AsyncMock(side_effect=[ValueError('Invalid output'), 'Complete'])
    result = await run_session(runner.manager, runner.runtime, 'build', 'Original prompt', read, workspace='/project')
    assert result == 'Complete'
    assert runner.events[-3:] == [('start', '1', '/project'), ('send', '1', 'Original prompt'), ('close', '1')]
    assert runner.events[3] == ('close', '0')


@pytest.mark.asyncio
async def test_exhausted_output_corrections_reopen_session(runner):
    read = AsyncMock(side_effect=[ValueError('Bad'), ValueError('Still bad'), 'Valid'])
    result = await run_session(
        runner.manager, runner.runtime, 'build', 'Build.', read, workspace='/project', output_attempts=2
    )
    assert result == 'Valid'
    assert [event[:2] for event in runner.events] == [
        ('start', '0'),
        ('send', '0'),
        ('send', '0'),
        ('close', '0'),
        ('start', '1'),
        ('send', '1'),
        ('close', '1'),
    ]


@pytest.mark.asyncio
async def test_execution_exception_opens_new_session(runner):
    async def send(session, message):
        await runner.send(session, message)
        if len(runner.sessions) == 1:
            raise ConnectionError('Connection lost')
        return TurnResult(kind='completed')

    runner.manager.send = send
    result = await run_session(
        runner.manager,
        runner.runtime,
        'build',
        'Build.',
        AsyncMock(return_value='OK'),
        workspace='/project',
    )
    assert result == 'OK'
    assert runner.events[2:4] == [('close', '0'), ('start', '1', '/project')]


@pytest.mark.asyncio
async def test_failed_session_budget_ends_without_reusing_failed_sessions(runner):
    async def send(session, message):
        await runner.send(session, message)
        return TurnResult(kind='terminal')

    runner.manager.send = send
    with pytest.raises(SessionAttemptsExhaustedError, match='2 sessions'):
        await run_session(
            runner.manager, runner.runtime, 'build', 'Build.', AsyncMock(), workspace='/project', session_attempts=2
        )
    assert [event[0] for event in runner.events] == ['start', 'send', 'close', 'start', 'send', 'close']


@pytest.mark.asyncio
async def test_cancellation_closes_current_session_and_propagates(runner):
    async def send(session, message):
        await runner.send(session, message)
        raise asyncio.CancelledError

    runner.manager.send = send
    with pytest.raises(asyncio.CancelledError):
        await run_session(
            runner.manager,
            runner.runtime,
            'build',
            'Build.',
            AsyncMock(),
            workspace='/project',
        )
    assert [event[0] for event in runner.events] == ['start', 'send', 'close']


@pytest.mark.asyncio
async def test_unhealthy_runtime_is_left_to_the_caller(runner):
    runner.runtime.is_unhealthy = True
    runner.runtime.unhealthy_reason = 'Sandbox lost'
    runner.manager.send = AsyncMock(return_value=TurnResult(kind='terminal'))
    with pytest.raises(RuntimeUnhealthyError, match='Sandbox lost'):
        await run_session(runner.manager, runner.runtime, 'build', 'Build.', AsyncMock(), workspace='/project')
    assert runner.events == [('start', '0', '/project'), ('close', '0')]


@pytest.mark.asyncio
async def test_failed_cleanup_preserves_completed_result(runner):
    runner.manager.close = AsyncMock(side_effect=RuntimeError('Cleanup failed'))
    assert (
        await run_session(
            runner.manager, runner.runtime, 'build', 'Build.', AsyncMock(return_value='Complete'), workspace='/project'
        )
        == 'Complete'
    )
